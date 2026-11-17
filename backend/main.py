import os
import sys
import shutil
import uuid
import time
import datetime
from typing import List, Dict, Optional, Any

# Add current directory to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd

from engine.profiler import profile_file
from engine.cleaner import detect_quality_issues, apply_cleaning_rules, coerce_numeric_columns
from engine.dictionary import generate_data_dictionary
from engine.etl import detect_relationships, create_master_dataset
from engine.eda import run_automated_eda
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan
from engine.ask_data import ask_your_data
from engine.exporter import export_master_excel, export_executive_pdf
from demo_data.generator import generate_demo_datasets
from agents.graph import LangGraphDataAgent, session_manager
from engine.contracts import DatasetContract

app = FastAPI(title="InsightFlow AI Platform API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.encoders import jsonable_encoder
import math
import numpy as np

def sanitize_nans(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [sanitize_nans(x) for x in obj.tolist()]
    elif isinstance(obj, dict):
        return {str(k): sanitize_nans(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_nans(v) for v in obj]
    return obj
STORE_DIR = os.getenv("STORE_DIR", os.path.join(BASE_DIR, "demo_data", "store"))
os.makedirs(STORE_DIR, exist_ok=True)

def safe_cleanup_dir(dir_path: str):
    """
    Safely purges files inside a directory without creating atomic removal race conditions
    in directory file-watchers (e.g., uvicorn StatReload/WatchFiles on Windows).
    """
    if not os.path.exists(dir_path):
        return
    try:
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
            except Exception:
                pass
        try:
            os.rmdir(dir_path)
        except OSError:
            pass
    except Exception:
        pass

import threading
from collections import OrderedDict

class BoundedProjectCache:
    """
    Thread-safe, bounded LRU project cache.
    Prevents unbounded memory growth while ensuring active projects remain accessible.
    Evicted projects remain safely persisted in STORE_DIR and can be reloaded on demand.
    """
    def __init__(self, max_size: int = 50):
        self.max_size = max(3, max_size)
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    def __getitem__(self, key: str):
        with self._lock:
            val = self._cache[key]
            self._cache.move_to_end(key)
            return val

    def __setitem__(self, key: str, value: Any):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            self._enforce_capacity()

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __bool__(self) -> bool:
        with self._lock:
            return bool(self._cache)

    def __iter__(self):
        with self._lock:
            return iter(list(self._cache.keys()))

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return default

    def values(self):
        with self._lock:
            return list(self._cache.values())

    def items(self):
        with self._lock:
            return list(self._cache.items())

    def keys(self):
        with self._lock:
            return list(self._cache.keys())

    def pop(self, key: str, *args):
        with self._lock:
            return self._cache.pop(key, *args)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def _enforce_capacity(self):
        while len(self._cache) > self.max_size:
            evict_k = None
            for k in self._cache.keys():
                if k != "proj-demo-q3-sales":
                    evict_k = k
                    break
            if evict_k:
                del self._cache[evict_k]
            else:
                self._cache.popitem(last=False)

    def cache_info(self) -> dict:
        with self._lock:
            return {
                "max_size": self.max_size,
                "current_size": len(self._cache),
                "keys": list(self._cache.keys())
            }

CACHE_MAX_SIZE = int(os.environ.get("PROJECT_CACHE_MAX_SIZE", "50"))
PROJECTS_DB = BoundedProjectCache(max_size=CACHE_MAX_SIZE)

# Upload size limits
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
UPLOAD_CHUNK_SIZE = 1024 * 1024           # 1 MB streaming chunks

class CleaningRulesRequest(BaseModel):
    approved_rules: List[dict]

class AskRequest(BaseModel):
    question: Optional[str] = None
    message: Optional[str] = None
    confirmed_field: Optional[str] = None
    selection_mode: Optional[str] = None

    def get_query(self) -> str:
        return (self.question or self.message or "").strip()

class SemanticOverrideRequest(BaseModel):
    column_name: str
    semantic_role: Optional[str] = None
    semantic_subtype: Optional[str] = None

class ActionStatusRequest(BaseModel):
    action_id: str
    status: Optional[str] = None
    owner: Optional[str] = None
    due_date: Optional[str] = None
    target: Optional[str] = None
    completion_notes: Optional[str] = None
    verification_notes: Optional[str] = None


def parse_excel_workbook(file_path: str) -> dict:
    """
    Intelligently parses single or multi-sheet Excel workbooks.
    If the workbook is an Industrial Matrix (e.g. Potline History Card),
    automatically unpivots and federates it into a Master TimeSeries and Fleet Summary.
    Otherwise, filters out metadata sheets, detects table boundaries, bypasses title banners,
    and extracts real raw data tables.
    """
    from engine.matrix_adapter import IndustrialMatrixAdapter
    from engine.parser import detect_tabular_region

    if IndustrialMatrixAdapter.is_industrial_matrix_workbook(file_path):
        return IndustrialMatrixAdapter.parse_workbook(file_path)

    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names
    
    ignore_keywords = ['readme', 'cover', 'summary', 'index', 'table of contents', 'data dictionary', 'data quality', 'exploratory data', 'eda', 'insights & story', 'insights', 'recommendations', 'action plan', 'export']
    
    valid_sheets = {}
    for s in sheet_names:
        s_low = s.lower().strip()
        is_ignored = any(k in s_low for k in ignore_keywords) and not any(k in s_low for k in ['raw', 'master', 'data', 'transaction', 'order'])
        if is_ignored:
            continue
        try:
            raw_sheet = xl.parse(s, header=None)
            if len(raw_sheet) > 0 and len(raw_sheet.columns) > 0:
                h_idx, df, meta = detect_tabular_region(raw_sheet)
                if len(df) > 0 and len(df.columns) > 1:
                    valid_sheets[s] = df
        except Exception:
            pass

    if not valid_sheets:
        for s in sheet_names:
            try:
                raw_sheet = xl.parse(s, header=None)
                if len(raw_sheet) > 0 and len(raw_sheet.columns) > 0:
                    h_idx, df, meta = detect_tabular_region(raw_sheet)
                    if len(df) > 0 and len(df.columns) > 1:
                        valid_sheets[s] = df
            except Exception:
                pass

    if not valid_sheets:
        raw_first = xl.parse(sheet_names[0], header=None)
        h_idx, df, meta = detect_tabular_region(raw_first)
        valid_sheets[sheet_names[0]] = df

    return valid_sheets

@app.on_event("startup")
def startup_event():
    # Ensure demo datasets exist
    sales_p, cust_p, prod_p = generate_demo_datasets(STORE_DIR)
    init_demo_project()

def load_persisted_projects(target_proj_id: Optional[str] = None, force_recompute: bool = False, limit: Optional[int] = 5):
    if not os.path.exists(STORE_DIR):
        return
    
    target_norm = target_proj_id.replace("proj_upload-", "").replace("proj-upload-", "") if target_proj_id else None

    # Sort folders by mtime descending so most recent projects load first
    entries = []
    for f in os.listdir(STORE_DIR):
        p = os.path.join(STORE_DIR, f)
        if os.path.isdir(p) and (f.startswith("proj-upload-") or f.startswith("proj_upload-")):
            try:
                entries.append((os.path.getmtime(p), f))
            except Exception:
                entries.append((0, f))
    entries.sort(key=lambda x: x[0], reverse=True)

    loaded_count = 0
    for _, folder_name in entries:
        if target_norm is None and limit is not None and loaded_count >= limit:
            break

        folder_path = os.path.join(STORE_DIR, folder_name)
        folder_norm = folder_name.replace("proj_upload-", "").replace("proj-upload-", "")
        
        # If target_proj_id is specified, skip folders that don't match target_proj_id
        if target_norm and folder_norm != target_norm:
            continue

        hyphen_key = f"proj-upload-{folder_norm}"
        underscore_key = f"proj_upload-{folder_norm}"

        if (hyphen_key in PROJECTS_DB or underscore_key in PROJECTS_DB or folder_name in PROJECTS_DB) and not force_recompute:
            if target_norm:
                break
            continue

        try:
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.xlsx', '.xls', '.csv', '.pdf'))]
            if not files:
                continue

            dfs_dict = {}
            primary_key_file = None
            pdf_documents_list = []

            for fp in files:
                base_name = os.path.splitext(os.path.basename(fp))[0]
                if fp.lower().endswith(('.xlsx', '.xls')):
                    parsed_sheets = parse_excel_workbook(fp)
                    for sheet_k, parsed_df in parsed_sheets.items():
                        parsed_df = coerce_numeric_columns(parsed_df)
                        dfs_dict[sheet_k if len(parsed_sheets) > 1 else base_name] = parsed_df
                elif fp.lower().endswith('.csv'):
                    from engine.parser import detect_tabular_region
                    raw_csv = pd.read_csv(fp, header=None)
                    h_idx, df, meta = detect_tabular_region(raw_csv)
                    df = coerce_numeric_columns(df)
                    dfs_dict[base_name] = df
                elif fp.lower().endswith('.pdf'):
                    from engine.pdf_engine import DocumentIntelligenceEngine
                    doc_corpus = DocumentIntelligenceEngine.extract_document(fp)
                    dfs_dict[f"{base_name}_Document"] = doc_corpus["structured_df"]
                    for t_idx, t_df in enumerate(doc_corpus.get("tables", [])):
                        dfs_dict[f"{base_name}_Table_{t_idx+1}"] = coerce_numeric_columns(t_df)
                    pdf_documents_list.append(doc_corpus)
                
            if primary_key_file is None or primary_key_file not in dfs_dict:
                primary_candidates = [k for k in dfs_dict.keys() if any(t in k.lower() for t in ['timeseries', 'master', 'order', 'transaction', 'sales', 'fact', 'event'])]
                if primary_candidates:
                    primary_key_file = primary_candidates[0]
                else:
                    primary_key_file = max(dfs_dict.keys(), key=lambda k: len(dfs_dict[k]) * len(dfs_dict[k].columns))

            primary_df = dfs_dict[primary_key_file]
            primary_fp = files[0]

            sales_profile = profile_file(primary_fp, df=primary_df)
            quality_issues = detect_quality_issues(primary_df, primary_fp)
            auto_rules = [q for q in quality_issues if q.get("type") in ["standardize_casing", "trim_whitespace", "coerce_numeric"]]
            cleaned_df, audit_trail = apply_cleaning_rules(primary_df, auto_rules)

            # Clean all loaded tables with safe normalizations so master dataset joins clean tables
            cleaned_dfs_dict = dict(dfs_dict)
            for tbl_k, tbl_df in dfs_dict.items():
                if tbl_k == primary_key_file:
                    cleaned_dfs_dict[tbl_k] = cleaned_df
                else:
                    tbl_issues = detect_quality_issues(tbl_df, tbl_k)
                    t_auto = [q for q in tbl_issues if q.get("type") in ["standardize_casing", "trim_whitespace", "coerce_numeric"]]
                    c_tbl_df, _ = apply_cleaning_rules(tbl_df, t_auto)
                    cleaned_dfs_dict[tbl_k] = c_tbl_df

            master_df, master_meta = create_master_dataset(cleaned_dfs_dict, primary_key_file)
            dict_list = generate_data_dictionary(master_df, primary_fp, table_name=primary_key_file, all_tables=cleaned_dfs_dict)
            eda_res = run_automated_eda(master_df)
            insights_res = generate_insights_and_story(master_df, eda_res, dataset_version="cleaned_v1")
            recs_res = generate_recommendations_and_action_plan(insights_res["insights"])

            if "Potline_Master_TimeSeries" in dfs_dict:
                proj_name = f"{base_name} Industrial Potline Analytics"
            elif len(files) == 1:
                proj_name = f"{primary_key_file} Analytics"
            else:
                proj_name = f"Custom Analytics ({len(files)} files)"
            proj_data = {
                "id": hyphen_key,
                "name": proj_name,
                "description": f"Persisted analytics pipeline for {len(files)} files: {', '.join(dfs_dict.keys())}",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "ready",
                "dataset_version": "cleaned_v1",
                "raw_dataset_version": "raw_v1",
                "eda_dataset_version": "cleaned_v1",
                "visualization_dataset_version": "cleaned_v1",
                "insight_dataset_version": "cleaned_v1",
                "raw_files": files,
                "dfs_dict": dfs_dict,
                "cleaned_dfs_dict": cleaned_dfs_dict,
                "primary_file": primary_key_file,
                "raw_df": primary_df,
                "cleaned_df": cleaned_df,
                "master_df": master_df,
                "master_meta": master_meta,
                "referential_integrity": master_meta.get("referential_integrity", {}),
                "profile": sales_profile,
                "quality_issues": quality_issues,
                "audit_trail": audit_trail,
                "data_dictionary": dict_list,
                "eda_results": eda_res,
                "insights": insights_res["insights"],
                "story": insights_res["story"],
                "recommendations": recs_res["recommendations"],
                "action_items": recs_res["action_items"],
                "pdf_documents": pdf_documents_list if pdf_documents_list else None
            }

            try:
                proj_data["dataset_contract"] = DatasetContract.create_from_pipeline(
                    dataset_id=hyphen_key,
                    raw_df=primary_df,
                    cleaned_df=master_df,
                    source_file_path=primary_fp,
                    dictionary=dict_list,
                    quality_issues=quality_issues,
                    audit_trail=audit_trail
                )
            except Exception as c_err:
                print(f"Could not construct DatasetContract for {hyphen_key}: {c_err}")
            
            # Alias all variations of key in PROJECTS_DB
            PROJECTS_DB[folder_name] = proj_data
            PROJECTS_DB[hyphen_key] = proj_data
            PROJECTS_DB[underscore_key] = proj_data
            print(f"Loaded persisted project: {folder_name} (keys: {hyphen_key}, {underscore_key})")
            loaded_count += 1
            if target_norm:
                break
        except Exception as e:
            print(f"Error loading persisted project {folder_name}: {e}")

def get_project_or_404(proj_id: str) -> dict:
    if proj_id == "proj-demo-q3-sales" and proj_id not in PROJECTS_DB:
        init_demo_project()
    if proj_id not in PROJECTS_DB:
        load_persisted_projects(target_proj_id=proj_id, force_recompute=False)
    if proj_id not in PROJECTS_DB:
        # Check normalized key match
        target_norm = proj_id.replace("proj_upload-", "").replace("proj-upload-", "")
        matching_key = next((k for k in PROJECTS_DB if k.replace("proj_upload-", "").replace("proj-upload-", "") == target_norm), None)
        if matching_key:
            PROJECTS_DB[proj_id] = PROJECTS_DB[matching_key]
        else:
            raise HTTPException(status_code=404, detail=f"Project '{proj_id}' not found")
    return PROJECTS_DB[proj_id]

def init_demo_project():
    proj_id = "proj-demo-q3-sales"
    sales_p = os.path.join(STORE_DIR, "Sales_2024.xlsx")
    cust_p = os.path.join(STORE_DIR, "Customers.xlsx")
    prod_p = os.path.join(STORE_DIR, "Products.xlsx")

    sales_df = pd.read_excel(sales_p)
    cust_df = pd.read_excel(cust_p)
    prod_df = pd.read_excel(prod_p)

    dfs_dict = {
        "Sales_2024": sales_df,
        "Customers": cust_df,
        "Products": prod_df
    }

    # Profile primary sales file
    sales_profile = profile_file(sales_p)
    quality_issues = detect_quality_issues(sales_df, "Sales_2024.xlsx")
    
    # Auto approve first set of safe rules for immediate demo readiness
    cleaned_df, audit_trail = apply_cleaning_rules(sales_df, quality_issues)
    
    dict_list = generate_data_dictionary(cleaned_df, "Sales_2024.xlsx")
    
    # ETL Master Dataset
    master_df, master_meta = create_master_dataset(dfs_dict, "Sales_2024")
    
    # EDA & Domain
    eda_res = run_automated_eda(master_df)
    
    # Insights & Story
    insights_res = generate_insights_and_story(master_df, eda_res)
    
    # Recommendations & Action Plan
    recs_res = generate_recommendations_and_action_plan(insights_res["insights"])

    PROJECTS_DB[proj_id] = {
        "id": proj_id,
        "name": "Q3 Enterprise Sales & Retention Analytics",
        "description": "Multi-file automated ETL and business story for Q3 Sales, Customers, and Products datasets",
        "created_at": "2024-08-21 10:00:00",
        "status": "ready",
        "raw_files": [sales_p, cust_p, prod_p],
        "dfs_dict": dfs_dict,
        "primary_file": "Sales_2024",
        "raw_df": sales_df,
        "cleaned_df": cleaned_df,
        "master_df": master_df,
        "master_meta": master_meta,
        "profile": sales_profile,
        "quality_issues": quality_issues,
        "audit_trail": audit_trail,
        "data_dictionary": dict_list,
        "eda_results": eda_res,
        "insights": insights_res["insights"],
        "story": insights_res["story"],
        "recommendations": recs_res["recommendations"],
        "action_items": recs_res["action_items"]
    }

    try:
        PROJECTS_DB[proj_id]["dataset_contract"] = DatasetContract.create_from_pipeline(
            dataset_id=proj_id,
            raw_df=sales_df,
            cleaned_df=master_df,
            source_file_path=sales_p,
            dictionary=dict_list,
            quality_issues=quality_issues,
            audit_trail=audit_trail
        )
    except Exception as c_err:
        print(f"Could not construct DatasetContract for demo project: {c_err}")

    demo_pdf_p = os.path.join(STORE_DIR, "proj-demo-q3-sales_Executive_Report.pdf")
    if os.path.exists(demo_pdf_p):
        try:
            from engine.pdf_engine import DocumentIntelligenceEngine
            PROJECTS_DB[proj_id]["pdf_documents"] = [DocumentIntelligenceEngine.extract_document(demo_pdf_p)]
        except Exception as e:
            print(f"Note: Could not preload demo PDF report: {e}")

    print(f"Demo project {proj_id} initialized.")

@app.get("/api/projects")
def list_projects():
    if not PROJECTS_DB:
        init_demo_project()

    seen_ids = set()
    unique_projects = []
    for p in PROJECTS_DB.values():
        p_id = p.get("id")
        if p_id and p_id not in seen_ids:
            seen_ids.add(p_id)
            unique_projects.append({
                "id": p["id"],
                "name": p["name"],
                "description": p["description"],
                "status": p["status"],
                "health_score": p.get("profile", {}).get("health_score", 100),
                "row_count": p.get("profile", {}).get("num_rows", len(p.get("cleaned_df", []))),
                "file_count": len(p.get("raw_files", []))
            })

    # Also list recent persisted projects without unpivoting/parsing heavy files upfront
    if os.path.exists(STORE_DIR):
        folders = [f for f in os.listdir(STORE_DIR) if f.startswith("proj-upload-") or f.startswith("proj_upload-")]
        # Sort folders by mtime descending
        entries = []
        for f in folders:
            fp_dir = os.path.join(STORE_DIR, f)
            if os.path.isdir(fp_dir):
                try:
                    entries.append((os.path.getmtime(fp_dir), f))
                except Exception:
                    entries.append((0, f))
        entries.sort(key=lambda x: x[0], reverse=True)

        for _, f in entries[:15]:
            norm_k = f.replace("proj_upload-", "").replace("proj-upload-", "")
            hyphen_k = f"proj-upload-{norm_k}"
            if hyphen_k not in seen_ids and f not in seen_ids:
                seen_ids.add(hyphen_k)
                fp_dir = os.path.join(STORE_DIR, f)
                files = [fn for fn in os.listdir(fp_dir) if fn.lower().endswith(('.xlsx', '.xls', '.csv', '.pdf'))] if os.path.isdir(fp_dir) else []
                fname = files[0] if files else "Dataset"
                bname = os.path.splitext(fname)[0]
                proj_name = f"{bname} Industrial Analytics" if any(k in bname.lower() for k in ["line", "pot", "histry", "history"]) else (f"{bname} Document Analytics" if fname.lower().endswith('.pdf') else f"{bname} Analytics")
                unique_projects.append({
                    "id": hyphen_k,
                    "name": proj_name,
                    "description": f"Persisted dataset with {len(files)} file(s)",
                    "status": "ready",
                    "health_score": 100,
                    "row_count": 0,
                    "file_count": len(files)
                })

    return jsonable_encoder(unique_projects)

@app.post("/api/projects/demo")
def trigger_demo_project():
    init_demo_project()
    return {"status": "success", "project_id": "proj-demo-q3-sales"}

@app.post("/api/projects/upload")
async def upload_project_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    timestamp = int(time.time())
    proj_id = f"proj-upload-{timestamp}-{uuid.uuid4().hex[:6]}"
    upload_dir = os.path.join(STORE_DIR, proj_id)
    os.makedirs(upload_dir, exist_ok=True)

    saved_filepaths = []
    dfs_dict = {}
    primary_key_file = None

    for uploaded_file in files:
        filename = uploaded_file.filename or "data.xlsx"
        file_path = os.path.join(upload_dir, filename)
        
        # Stream file in chunks to validate size before reading entire file into memory
        file_bytes_received = 0
        try:
            with open(file_path, "wb") as f:
                while True:
                    chunk = await uploaded_file.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    file_bytes_received += len(chunk)
                    if file_bytes_received > MAX_UPLOAD_SIZE_BYTES:
                        f.close()
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        safe_cleanup_dir(upload_dir)
                        raise HTTPException(
                            status_code=413,
                            detail=f"File '{filename}' exceeds maximum upload size limit of 50 MB."
                        )
                    f.write(chunk)
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            safe_cleanup_dir(upload_dir)
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file '{filename}': {str(e)}")
        
        saved_filepaths.append(file_path)
        base_name = os.path.splitext(filename)[0]

        try:
            if filename.lower().endswith(('.xlsx', '.xls')):
                parsed_sheets = parse_excel_workbook(file_path)
                for sheet_k, parsed_df in parsed_sheets.items():
                    parsed_df = coerce_numeric_columns(parsed_df)
                    dfs_dict[sheet_k if len(parsed_sheets) > 1 else base_name] = parsed_df
            elif filename.lower().endswith('.csv'):
                from engine.parser import detect_tabular_region
                raw_csv = pd.read_csv(file_path, header=None)
                h_idx, df, meta = detect_tabular_region(raw_csv)
                df = coerce_numeric_columns(df)
                dfs_dict[base_name] = df
            elif filename.lower().endswith('.json'):
                df = pd.read_json(file_path)
                df.columns = [str(c).strip() for c in df.columns]
                df = coerce_numeric_columns(df)
                dfs_dict[base_name] = df
            elif filename.lower().endswith('.pdf'):
                from engine.pdf_engine import DocumentIntelligenceEngine
                doc_corpus = DocumentIntelligenceEngine.extract_document(file_path)
                dfs_dict[f"{base_name}_Document"] = doc_corpus["structured_df"]
                for t_idx, t_df in enumerate(doc_corpus.get("tables", [])):
                    dfs_dict[f"{base_name}_Table_{t_idx+1}"] = coerce_numeric_columns(t_df)
                if "pdf_documents_uploaded" not in locals():
                    pdf_documents_uploaded = []
                pdf_documents_uploaded.append(doc_corpus)
            else:
                parsed_sheets = parse_excel_workbook(file_path)
                for sheet_k, parsed_df in parsed_sheets.items():
                    parsed_df = coerce_numeric_columns(parsed_df)
                    dfs_dict[sheet_k if len(parsed_sheets) > 1 else base_name] = parsed_df

            if primary_key_file is None or primary_key_file not in dfs_dict:
                primary_candidates = [k for k in dfs_dict.keys() if any(t in k.lower() for t in ['timeseries', 'master', 'order', 'transaction', 'sales', 'fact', 'event'])]
                if primary_candidates:
                    primary_key_file = primary_candidates[0]
                else:
                    primary_key_file = max(dfs_dict.keys(), key=lambda k: len(dfs_dict[k]) * len(dfs_dict[k].columns))
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    if not dfs_dict:
        safe_cleanup_dir(upload_dir)
        raise HTTPException(status_code=400, detail="Could not parse any valid dataset files")

    try:
        primary_df = dfs_dict[primary_key_file]
        primary_filepath = saved_filepaths[0]

        # Analytics Engine Pipeline
        sales_profile = profile_file(primary_filepath, df=primary_df)
        quality_issues = detect_quality_issues(primary_df, primary_filepath)
        auto_rules = [q for q in quality_issues if q.get("type") in ["standardize_casing", "trim_whitespace", "coerce_numeric"]]
        cleaned_df, audit_trail = apply_cleaning_rules(primary_df, auto_rules)

        # Clean all loaded tables with safe normalizations so master dataset joins clean tables
        cleaned_dfs_dict = dict(dfs_dict)
        for tbl_k, tbl_df in dfs_dict.items():
            if tbl_k == primary_key_file:
                cleaned_dfs_dict[tbl_k] = cleaned_df
            else:
                tbl_issues = detect_quality_issues(tbl_df, tbl_k)
                t_auto = [q for q in tbl_issues if q.get("type") in ["standardize_casing", "trim_whitespace", "coerce_numeric"]]
                c_tbl_df, _ = apply_cleaning_rules(tbl_df, t_auto)
                cleaned_dfs_dict[tbl_k] = c_tbl_df

        master_df, master_meta = create_master_dataset(cleaned_dfs_dict, primary_key_file)
        dict_list = generate_data_dictionary(master_df, primary_filepath, table_name=primary_key_file, all_tables=cleaned_dfs_dict)
        
        eda_res = run_automated_eda(master_df)
        insights_res = generate_insights_and_story(master_df, eda_res, dataset_version="cleaned_v1")
        recs_res = generate_recommendations_and_action_plan(insights_res["insights"])
    except HTTPException:
        safe_cleanup_dir(upload_dir)
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        safe_cleanup_dir(upload_dir)
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed on uploaded dataset: {str(e)}")

    if "Potline_Master_TimeSeries" in dfs_dict:
        proj_name = f"{base_name} Industrial Potline Analytics"
    elif len(files) == 1:
        proj_name = f"{primary_key_file} Analytics"
    else:
        proj_name = f"Custom Multi-File Analytics ({len(files)} files)"
    PROJECTS_DB[proj_id] = {
        "id": proj_id,
        "name": proj_name,
        "description": f"Automated analytics pipeline generated from {len(files)} uploaded files: {', '.join(dfs_dict.keys())}",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ready",
        "dataset_version": "cleaned_v1",
        "raw_dataset_version": "raw_v1",
        "eda_dataset_version": "cleaned_v1",
        "visualization_dataset_version": "cleaned_v1",
        "insight_dataset_version": "cleaned_v1",
        "raw_files": saved_filepaths,
        "dfs_dict": dfs_dict,
        "cleaned_dfs_dict": cleaned_dfs_dict,
        "primary_file": primary_key_file,
        "raw_df": primary_df,
        "cleaned_df": cleaned_df,
        "master_df": master_df,
        "master_meta": master_meta,
        "referential_integrity": master_meta.get("referential_integrity", {}),
        "profile": sales_profile,
        "quality_issues": quality_issues,
        "audit_trail": audit_trail,
        "data_dictionary": dict_list,
        "eda_results": eda_res,
        "insights": insights_res["insights"],
        "story": insights_res["story"],
        "recommendations": recs_res["recommendations"],
        "action_items": recs_res["action_items"],
        "pdf_documents": pdf_documents_uploaded if ("pdf_documents_uploaded" in locals() and pdf_documents_uploaded) else None
    }

    try:
        PROJECTS_DB[proj_id]["dataset_contract"] = DatasetContract.create_from_pipeline(
            dataset_id=proj_id,
            raw_df=primary_df,
            cleaned_df=master_df,
            source_file_path=primary_filepath,
            dictionary=dict_list,
            quality_issues=quality_issues,
            audit_trail=audit_trail
        )
    except Exception as c_err:
        print(f"Could not construct DatasetContract for {proj_id}: {c_err}")

    return {"status": "success", "project_id": proj_id, "name": proj_name}

@app.get("/api/project/{proj_id}")
def get_project_details(proj_id: str):
    p = get_project_or_404(proj_id)

    # Auto-refresh eda_results if cached payload uses legacy schema
    rec_vis = p.get("eda_results", {}).get("recommended_visualizations", [])
    heatmap_vis = [v for v in rec_vis if "HEATMAP" in str(v.get("chart_type")).upper()]
    if heatmap_vis and len(heatmap_vis[0].get("data", [])) > 0:
        sample_pt = heatmap_vis[0]["data"][0]
        if "pearson" not in sample_pt or "is_constant" not in sample_pt:
            print(f"[Backend Engine] Auto-refreshing EDA & Insights pipeline for project {proj_id}...")
            p["eda_results"] = run_automated_eda(p["master_df"])
            insights_res = generate_insights_and_story(p["master_df"], p["eda_results"])
            p["insights"] = insights_res["insights"]
            p["story"] = insights_res["story"]
            recs_res = generate_recommendations_and_action_plan(p["insights"])
            p["recommendations"] = recs_res["recommendations"]
            p["action_items"] = recs_res["action_items"]

    # Auto-refresh insights if supporting_visualization scatter plot data lacks metric keys
    insights_list = p.get("insights", [])
    scatter_insights = [ins for ins in insights_list if ins.get("supporting_visualization", {}).get("chart_type") == "SCATTER"]
    if scatter_insights:
        sv = scatter_insights[0].get("supporting_visualization", {})
        s_data = sv.get("data", [])
        m1 = sv.get("x_axis")
        if s_data and m1 and m1 not in s_data[0]:
            print(f"[Backend Engine] Auto-refreshing scatter visualization payload for project {proj_id}...")
            insights_res = generate_insights_and_story(p["master_df"], p["eda_results"])
            p["insights"] = insights_res["insights"]
            p["story"] = insights_res["story"]

    target_df = p.get("master_df", p.get("cleaned_df", p.get("raw_df")))
    visual_sample = target_df.to_dict(orient="records") if target_df is not None and len(target_df) > 0 else []

    return jsonable_encoder(sanitize_nans({
        "id": p["id"],
        "name": p["name"],
        "description": p["description"],
        "status": p["status"],
        "dataset_version": p.get("dataset_version", "cleaned_v1"),
        "raw_dataset_version": p.get("raw_dataset_version", "raw_v1"),
        "eda_dataset_version": "cleaned_v1",
        "visualization_dataset_version": "cleaned_v1",
        "insight_dataset_version": "cleaned_v1",
        "profile": p["profile"],
        "total_records": len(target_df) if target_df is not None else (p["profile"].get("num_rows", 0) if p.get("profile") else 0),
        "total_rows": len(target_df) if target_df is not None else (p["profile"].get("num_rows", 0) if p.get("profile") else 0),
        "quality_issues": p["quality_issues"],
        "audit_trail": p["audit_trail"],
        "data_dictionary": p["data_dictionary"],
        "master_meta": p["master_meta"],
        "referential_integrity": p.get("referential_integrity") or p.get("master_meta", {}).get("referential_integrity", {}),
        "eda_results": p["eda_results"],
        "insights": p["insights"],
        "story": p["story"],
        "recommendations": p["recommendations"],
        "action_items": p["action_items"],
        "visual_dataset": visual_sample,
        "dataset_contract": p.get("dataset_contract").to_dict() if hasattr(p.get("dataset_contract"), "to_dict") else p.get("dataset_contract")
    }))

@app.post("/api/project/{proj_id}/clean")
def clean_project_data(proj_id: str, req: CleaningRulesRequest):
    p = get_project_or_404(proj_id)
    raw_df = p["raw_df"]
    
    cleaned_df, audit_trail = apply_cleaning_rules(raw_df, req.approved_rules)
    p["cleaned_df"] = cleaned_df
    p["audit_trail"] = audit_trail

    # Re-run dependent downstream stages strictly using cleaned dataset
    if "cleaned_dfs_dict" not in p:
        p["cleaned_dfs_dict"] = dict(p["dfs_dict"])
    p["cleaned_dfs_dict"][p["primary_file"]] = cleaned_df
    p["master_df"], p["master_meta"] = create_master_dataset(p["cleaned_dfs_dict"], p["primary_file"])
    p["data_dictionary"] = generate_data_dictionary(p["master_df"], p["primary_file"], table_name=p["primary_file"], all_tables=p["cleaned_dfs_dict"])
    p["eda_results"] = run_automated_eda(p["master_df"])
    insights_res = generate_insights_and_story(p["master_df"], p["eda_results"], dataset_version="cleaned_v1")
    p["insights"] = insights_res["insights"]
    p["story"] = insights_res["story"]
    recs_res = generate_recommendations_and_action_plan(p["insights"])
    p["recommendations"] = recs_res["recommendations"]
    p["action_items"] = recs_res["action_items"]
    p["referential_integrity"] = p["master_meta"].get("referential_integrity", {})
    p["dataset_version"] = "cleaned_v1"

    try:
        p["dataset_contract"] = DatasetContract.create_from_pipeline(
            dataset_id=proj_id,
            raw_df=p["raw_df"],
            cleaned_df=p["master_df"],
            source_file_path=p.get("raw_files", [""])[0] if p.get("raw_files") else "",
            dictionary=p["data_dictionary"],
            quality_issues=p.get("quality_issues"),
            audit_trail=p.get("audit_trail")
        )
    except Exception as c_err:
        print(f"Could not update DatasetContract for {proj_id}: {c_err}")

    return {
        "status": "success",
        "audit_trail": audit_trail,
        "health_score": p["profile"]["health_score"],
        "rows_count": len(cleaned_df)
    }

def get_project_pdf_corpus(p: dict, proj_id: str) -> Optional[List[Dict[str, Any]]]:
    """Retrieves or dynamically extracts document corpus from project's PDF files or executive report."""
    if p.get("pdf_documents"):
        return p["pdf_documents"] if isinstance(p["pdf_documents"], list) else [p["pdf_documents"]]

    from engine.pdf_engine import DocumentIntelligenceEngine
    pdf_docs = []

    # 1. Check raw files for any uploaded PDF
    raw_files = p.get("raw_files", [])
    for rf in raw_files:
        if isinstance(rf, str) and rf.lower().endswith(".pdf") and os.path.exists(rf):
            try:
                doc = DocumentIntelligenceEngine.extract_document(rf)
                pdf_docs.append(doc)
            except Exception as e:
                print(f"Error extracting PDF {rf}: {e}")

    # 2. Check project directory in STORE_DIR for PDFs
    proj_dir = os.path.join(STORE_DIR, proj_id)
    if os.path.isdir(proj_dir):
        for fn in os.listdir(proj_dir):
            if fn.lower().endswith(".pdf"):
                fp = os.path.join(proj_dir, fn)
                if not any(d.get("file_path") == fp for d in pdf_docs):
                    try:
                        doc = DocumentIntelligenceEngine.extract_document(fp)
                        pdf_docs.append(doc)
                    except Exception as e:
                        print(f"Error extracting directory PDF {fp}: {e}")

    # 3. Check for generated Executive Report PDF
    exec_pdf = os.path.join(STORE_DIR, f"{proj_id}_Executive_Report.pdf")
    if os.path.exists(exec_pdf) and not pdf_docs:
        try:
            doc = DocumentIntelligenceEngine.extract_document(exec_pdf)
            pdf_docs.append(doc)
        except Exception as e:
            print(f"Error extracting executive report PDF: {e}")

    # 4. If demo project, also check for demo Executive Report PDF
    if not pdf_docs and proj_id == "proj-demo-q3-sales":
        demo_pdf = os.path.join(STORE_DIR, "proj-demo-q3-sales_Executive_Report.pdf")
        if os.path.exists(demo_pdf):
            try:
                doc = DocumentIntelligenceEngine.extract_document(demo_pdf)
                pdf_docs.append(doc)
            except Exception as e:
                print(f"Error extracting demo report PDF: {e}")

    if pdf_docs:
        p["pdf_documents"] = pdf_docs
        return pdf_docs

    return None

@app.post("/api/project/{proj_id}/ask")
def query_data(proj_id: str, req: AskRequest):
    p = get_project_or_404(proj_id)
    sem_layer = p.get("profile", {}).get("semantic_layer", {})
    eda_results = p.get("eda_results", {})
    insights = p.get("insights", [])
    contract = p.get("dataset_contract")
    dictionary = p.get("data_dictionary")
    pdf_docs = get_project_pdf_corpus(p, proj_id)

    res = LangGraphDataAgent.process_message(
        project_id=proj_id,
        df=p.get("master_df", p.get("cleaned_df", p.get("raw_df"))),
        sem_layer=sem_layer,
        eda_results=eda_results,
        insights=insights,
        user_message=req.get_query(),
        contract=contract,
        dictionary=dictionary,
        pdf_documents=pdf_docs,
        confirmed_field=req.confirmed_field
    )
    return jsonable_encoder(sanitize_nans(res))

@app.post("/api/project/{proj_id}/chat")
def chat_with_data(proj_id: str, req: AskRequest):
    p = get_project_or_404(proj_id)
    sem_layer = p.get("profile", {}).get("semantic_layer", {})
    eda_results = p.get("eda_results", {})
    insights = p.get("insights", [])
    contract = p.get("dataset_contract")
    dictionary = p.get("data_dictionary")
    pdf_docs = get_project_pdf_corpus(p, proj_id)

    res = LangGraphDataAgent.process_message(
        project_id=proj_id,
        df=p.get("master_df", p.get("cleaned_df", p.get("raw_df"))),
        sem_layer=sem_layer,
        eda_results=eda_results,
        insights=insights,
        user_message=req.get_query(),
        contract=contract,
        dictionary=dictionary,
        pdf_documents=pdf_docs,
        confirmed_field=req.confirmed_field
    )
    return jsonable_encoder(sanitize_nans(res))

@app.get("/api/project/{proj_id}/chat-history")
def get_chat_history(proj_id: str):
    history = session_manager.get_history(proj_id)
    return jsonable_encoder(sanitize_nans({"history": history}))

@app.delete("/api/project/{proj_id}/chat-history")
def clear_chat_history(proj_id: str):
    session_manager.clear_session(proj_id)
    return {"status": "success", "message": "Session history cleared."}

@app.post("/api/project/{proj_id}/semantic-override")
def override_semantic_role(proj_id: str, req: SemanticOverrideRequest):
    p = get_project_or_404(proj_id)
    col = req.column_name
    if col not in p["master_df"].columns:
        raise HTTPException(status_code=400, detail=f"Column '{col}' not found in master dataset")

    # Track column overrides on the project object
    if "column_overrides" not in p or not isinstance(p["column_overrides"], dict):
        p["column_overrides"] = {}

    # Handle clearing override (Automatic / No Override)
    clear_keywords = ["", "automatic", "automatic/no override", "automatic (no override)", "none", "null", "default", "no override"]
    is_clearing = req.semantic_role is None or req.semantic_role.strip().lower() in clear_keywords

    if is_clearing:
        if col in p["column_overrides"]:
            del p["column_overrides"][col]
    else:
        p["column_overrides"][col] = {
            "semantic_role": req.semantic_role,
            "semantic_subtype": req.semantic_subtype
        }

    # Re-generate data dictionary with updated overrides
    primary_name = p.get("primary_file", "dataset")
    p["data_dictionary"] = generate_data_dictionary(
        p["master_df"],
        filename=primary_name,
        column_overrides=p.get("column_overrides")
    )

    # Update profile semantic_layer if present
    if "profile" in p and "semantic_layer" in p["profile"]:
        for dict_item in p["data_dictionary"]:
            d_col = dict_item.get("column_name")
            if d_col in p["profile"]["semantic_layer"]:
                sem_role = dict_item.get("semantic_role")
                p["profile"]["semantic_layer"][d_col]["semantic_role"] = sem_role
                p["profile"]["semantic_layer"][d_col]["semantic_subtype"] = dict_item.get("semantic_subtype")
                p["profile"]["semantic_layer"][d_col]["is_measure"] = (sem_role == "Measure")
                p["profile"]["semantic_layer"][d_col]["is_dimension"] = (sem_role in ["Dimension", "Categorical"])
                p["profile"]["semantic_layer"][d_col]["is_time"] = (sem_role in ["Date/Time", "Time"])
                p["profile"]["semantic_layer"][d_col]["is_id"] = (sem_role in ["Identifier", "ID"])

    # Re-run EDA, Insights, and Action Plan based on updated semantic roles
    p["eda_results"] = run_automated_eda(p["master_df"], column_overrides=p.get("column_overrides"))
    insights_res = generate_insights_and_story(p["master_df"], p["eda_results"])
    p["insights"] = insights_res["insights"]
    p["story"] = insights_res["story"]
    recs_res = generate_recommendations_and_action_plan(p["insights"])
    p["recommendations"] = recs_res["recommendations"]
    p["action_items"] = recs_res["action_items"]

    return jsonable_encoder(sanitize_nans({
        "status": "success",
        "message": f"Updated semantic role for '{col}'",
        "semantic_layer": p["profile"].get("semantic_layer", {}) if "profile" in p else {},
        "data_dictionary": p["data_dictionary"],
        "eda_results": p["eda_results"]
    }))

@app.post("/api/project/{proj_id}/action-status")
def update_action_status(proj_id: str, req: ActionStatusRequest):
    p = get_project_or_404(proj_id)
    
    updated = False
    target_item = None
    for item in p.get("action_items", []):
        if item.get("id") == req.action_id or item.get("action_id") == req.action_id:
            if req.status is not None:
                item["status"] = req.status
                item["execution_status"] = req.status
            if req.owner is not None:
                item["owner"] = req.owner
            if req.due_date is not None:
                item["due_date"] = req.due_date
                item["timeline"] = req.due_date
            if req.target is not None:
                item["target_value"] = req.target
                item["target"] = req.target
            if req.completion_notes is not None:
                item["completion_notes"] = req.completion_notes
            if req.verification_notes is not None:
                item["verification_notes"] = req.verification_notes
            item["last_updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            target_item = item
            updated = True
            break
            
    if not updated:
        raise HTTPException(status_code=404, detail="Action item not found")

    return {
        "status": "success",
        "action_id": req.action_id,
        "new_status": target_item.get("status") if target_item else req.status,
        "item": target_item
    }

@app.get("/api/project/{proj_id}/master-preview")
def get_master_preview(proj_id: str, mode: str = "master", limit: int = 1000):
    p = get_project_or_404(proj_id)
    if mode == "raw" and "raw_df" in p and p["raw_df"] is not None:
        target_df = p["raw_df"]
    elif mode == "cleaned" and "cleaned_df" in p and p["cleaned_df"] is not None:
        target_df = p["cleaned_df"]
    else:
        target_df = p.get("master_df", p.get("cleaned_df", p.get("raw_df")))
    
    if target_df is None or len(target_df) == 0:
        return jsonable_encoder(sanitize_nans({
            "mode": mode,
            "total_rows": 0,
            "total_columns": 0,
            "columns": [],
            "master_meta": p.get("master_meta", {}),
            "rows": []
        }))

    sample_size = max(1, min(limit, 5000))
    sample_df = target_df.head(sample_size)
    rows = sample_df.to_dict(orient="records")
    
    return jsonable_encoder(sanitize_nans({
        "mode": mode,
        "total_rows": len(target_df),
        "total_columns": len(target_df.columns),
        "columns": list(target_df.columns),
        "master_meta": p.get("master_meta", {}),
        "rows": rows
    }))


@app.get("/api/project/{proj_id}/export/excel")
def download_excel(proj_id: str):
    p = get_project_or_404(proj_id)

    out_file = os.path.join(STORE_DIR, f"{proj_id}_Master_Deliverable.xlsx")
    export_master_excel(
        raw_df=p["raw_df"],
        cleaned_df=p["cleaned_df"],
        master_df=p["master_df"],
        dictionary_list=p["data_dictionary"],
        quality_issues=p["quality_issues"],
        eda_results=p["eda_results"],
        insights_list=p["insights"],
        recommendations_list=p["recommendations"],
        action_items_list=p["action_items"],
        output_path=out_file
    )
    return FileResponse(
        out_file,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"{p['name'].replace(' ', '_')}_Master_Deliverable.xlsx"
    )

@app.get("/api/project/{proj_id}/export/pdf")
def download_pdf(proj_id: str):
    p = get_project_or_404(proj_id)

    out_file = os.path.join(STORE_DIR, f"{proj_id}_Executive_Report.pdf")
    export_executive_pdf(
        eda_results=p["eda_results"],
        insights_list=p["insights"],
        recommendations_list=p["recommendations"],
        dataset_name=p["name"],
        output_path=out_file,
        dictionary_list=p.get("data_dictionary", [])
    )
    return FileResponse(
        out_file,
        media_type="application/pdf",
        filename=f"{p['name'].replace(' ', '_')}_Executive_Report.pdf"
    )


import httpx
from fastapi import Request
from fastapi.responses import Response

# Reusable AsyncClient for proxying to frontend server
HTTP_CLIENT = httpx.AsyncClient(timeout=30.0)
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://127.0.0.1:3000").rstrip("/")

@app.get("/", include_in_schema=False)
def root_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy_frontend(request: Request, path: str = ""):
    if path.startswith("api/") or path == "api" or path.startswith("docs") or path == "openapi.json":
        raise HTTPException(status_code=404, detail="Not Found")
    
    if "webpack-hmr" in path or "hmr" in path:
        return Response(status_code=204)
    
    url = f"{FRONTEND_URL}/{path}"
    if request.query_params:
        url += f"?{request.query_params}"
    
    try:
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ('host', 'content-length', 'accept-encoding')}
        from urllib.parse import urlparse
        parsed_host = urlparse(FRONTEND_URL).netloc
        if parsed_host:
            headers['Host'] = parsed_host

        resp = await HTTP_CLIENT.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body if body else None,
            follow_redirects=True
        )
        
        excluded_headers = {'content-encoding', 'transfer-encoding', 'content-length'}
        resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded_headers}
        
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp_headers
        )
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return Response(
            content="<html><head><meta http-equiv='refresh' content='2'></head><body style='font-family:sans-serif;background:#090d16;color:#94a3b8;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;'><div><h2>InsightFlow AI is starting up...</h2><p>The server is initializing. This page will reload automatically in 2 seconds.</p></div></body></html>",
            status_code=503,
            media_type="text/html"
        )
    except Exception as e:
        return Response(content=f"Frontend connection error: {e}", status_code=503)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

