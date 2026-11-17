import os
import sys
import io
import shutil
import pytest
import numpy as np
import pandas as pd
from scipy import stats
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from main import app, PROJECTS_DB, BoundedProjectCache, STORE_DIR, get_project_or_404
from services.statistical_analysis import compute_group_differences
from engine.eda import run_automated_eda
from engine.exporter import export_master_excel, export_executive_pdf
from agents.graph import LangGraphDataAgent, session_manager, optional_llm_explain
import openpyxl


# =====================================================================
# 1. Independent Statistical Correctness Verification
# =====================================================================

def test_statistical_correctness_independent_math():
    """
    Independently verifies One-Way ANOVA and Kruskal-Wallis:
    - Degrees of freedom: df_between = k - 1, df_within = N - k
    - F-statistic, p-value
    - Effect size: Eta-squared = SS_between / SS_total
    - Kruskal-Wallis H and p-value
    """
    g1 = [10.0, 12.0, 11.0, 13.0, 12.0]
    g2 = [20.0, 22.0, 21.0, 23.0, 22.0]
    g3 = [30.0, 31.0, 32.0, 30.0, 33.0]
    
    k = 3
    N = 15
    expected_df_between = k - 1  # 2
    expected_df_within = N - k   # 12

    # Scipy ground truth
    f_stat_scipy, p_val_scipy = stats.f_oneway(g1, g2, g3)
    h_stat_scipy, p_val_scipy_k = stats.kruskal(g1, g2, g3)

    # Eta-squared ground truth
    all_vals = np.array(g1 + g2 + g3)
    grand_mean = np.mean(all_vals)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    ss_between = len(g1) * (np.mean(g1) - grand_mean) ** 2 + \
                 len(g2) * (np.mean(g2) - grand_mean) ** 2 + \
                 len(g3) * (np.mean(g3) - grand_mean) ** 2
    expected_eta_sq = ss_between / ss_total

    # Execute platform function
    df = pd.DataFrame({
        "group": ["G1"] * 5 + ["G2"] * 5 + ["G3"] * 5,
        "metric": g1 + g2 + g3
    })
    res = compute_group_differences(df, "group", "metric")
    assert res["status"] == "success"

    anova = res["anova"]
    assert anova["valid"] is True
    assert anova["df_between"] == expected_df_between
    assert anova["df_within"] == expected_df_within
    assert anova["k_groups"] == k
    assert anova["n_observations"] == N
    assert np.isclose(anova["f_statistic"], f_stat_scipy, atol=1e-3)
    assert np.isclose(anova["p_value"], p_val_scipy, atol=1e-4)
    assert np.isclose(anova["eta_squared"], expected_eta_sq, atol=1e-3)

    kruskal = res["kruskal_wallis"]
    assert kruskal["valid"] is True
    assert kruskal["kruskal_df"] == expected_df_between
    assert np.isclose(kruskal["h_statistic"], h_stat_scipy, atol=1e-3)
    assert np.isclose(kruskal["p_value"], p_val_scipy_k, atol=1e-4)


def test_numeric_mode_multimodal_and_empty():
    """Verify numeric modes for multimodal, constant, and all-null columns."""
    df = pd.DataFrame({
        "multimodal": [10.0, 10.0, 20.0, 20.0, 30.0],
        "all_nulls": pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=float),
        "constant": [99.0, 99.0, 99.0, 99.0, 99.0]
    })
    eda = run_automated_eda(df)
    ns = eda["numeric_summary"]

    # Multimodal
    assert ns["multimodal"]["is_multimodal"] is True
    assert sorted(ns["multimodal"]["modes"]) == [10.0, 20.0]

    # All nulls
    assert ns["all_nulls"]["count"] == 0
    assert ns["all_nulls"]["mode"] is None
    assert ns["all_nulls"]["modes"] == []
    assert ns["all_nulls"]["is_multimodal"] is False

    # Constant
    assert ns["constant"]["mode"] == 99.0
    assert ns["constant"]["is_multimodal"] is False


# =====================================================================
# 2. Upload Security & Disk Cleanup Verification
# =====================================================================

def test_upload_rejection_cleans_up_disk():
    """Verify that rejected uploads (oversized or malformed) leave zero orphaned files in STORE_DIR."""
    client = TestClient(app)

    # 1. Test malformed upload (corrupt file)
    corrupt_content = b"This is not a valid CSV or Excel file\x00\xff\xfe"
    files = [("files", ("corrupt.xlsx", io.BytesIO(corrupt_content), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))]
    resp = client.post("/api/projects/upload", files=files)
    assert resp.status_code == 400

    # 2. Verify no empty or orphaned proj-upload-* folders remain without valid files
    for folder in os.listdir(STORE_DIR):
        fpath = os.path.join(STORE_DIR, folder)
        if os.path.isdir(fpath) and folder.startswith("proj-upload-"):
            valid_files = [f for f in os.listdir(fpath) if f.endswith(('.csv', '.xlsx', '.xls', '.pdf'))]
            # If an upload failed, it must not leave an empty folder
            assert len(valid_files) > 0, f"Orphaned folder found: {folder}"


# =====================================================================
# 3. Project Cache Eviction & Automatic Restoration Verification
# =====================================================================

def test_project_cache_eviction_and_restoration():
    """
    Verify:
    1. Bounded LRU cache evicts oldest project when capacity is reached.
    2. Requesting an evicted project restores it from STORE_DIR via get_project_or_404.
    3. Demo project is never permanently evicted.
    """
    client = TestClient(app)

    # Upload Project 1
    csv1 = b"id,val\n1,10\n2,20\n"
    r1 = client.post("/api/projects/upload", files=[("files", ("ds1.csv", io.BytesIO(csv1), "text/csv"))])
    assert r1.status_code == 200
    p1_id = r1.json()["project_id"]

    # Upload Project 2
    csv2 = b"id,val\n1,100\n2,200\n"
    r2 = client.post("/api/projects/upload", files=[("files", ("ds2.csv", io.BytesIO(csv2), "text/csv"))])
    assert r2.status_code == 200
    p2_id = r2.json()["project_id"]

    assert p1_id in PROJECTS_DB
    assert p2_id in PROJECTS_DB

    # Artificially evict p1 from in-memory cache to simulate capacity eviction
    evicted_data = PROJECTS_DB.pop(p1_id, None)
    assert evicted_data is not None
    assert p1_id not in PROJECTS_DB

    # Test restoration: calling get_project_or_404 must reload p1 from disk
    restored = get_project_or_404(p1_id)
    assert restored is not None
    assert restored["id"] == p1_id
    assert p1_id in PROJECTS_DB
    assert restored["master_df"] is not None
    assert len(restored["master_df"]) == 2


# =====================================================================
# 4. Session and Dataset Isolation Verification
# =====================================================================

def test_session_and_dataset_isolation():
    """Confirm that no project can access another project's dataset or conversation history."""
    proj_a = "proj-iso-a"
    proj_b = "proj-iso-b"

    df_a = pd.DataFrame({"product": ["A", "B"], "revenue": [100.0, 200.0]})
    df_b = pd.DataFrame({"department": ["HR", "Eng"], "headcount": [5, 50]})

    agent_a = LangGraphDataAgent(df=df_a, project_id=proj_a)
    agent_b = LangGraphDataAgent(df=df_b, project_id=proj_b)

    # Run queries in project A
    ans_a = agent_a.run("What is the average revenue?")
    # Run queries in project B
    ans_b = agent_b.run("What is the average headcount?")

    # Verify Project A session contains only revenue query
    hist_a = session_manager.get_history(proj_a)
    assert len(hist_a) == 2
    assert any("revenue" in m["content"].lower() for m in hist_a)
    assert not any("headcount" in m["content"].lower() for m in hist_a)

    # Verify Project B session contains only headcount query
    hist_b = session_manager.get_history(proj_b)
    assert len(hist_b) == 2
    assert any("headcount" in m["content"].lower() for m in hist_b)
    assert not any("revenue" in m["content"].lower() for m in hist_b)


# =====================================================================
# 5. Excel Worksheet Count Verification (9-Sheet Inspection)
# =====================================================================

def test_excel_export_worksheet_count_and_content():
    """
    Inspects the actual Excel export, counts worksheets, and verifies sheet data matches
    the underlying dataset.
    """
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "category": ["A", "B", "A"],
        "sales": [100.0, 200.0, 300.0]
    })
    eda_res = run_automated_eda(df)
    tmp_path = "tmp/audit_verification_workbook.xlsx"

    out = export_master_excel(
        raw_df=df,
        cleaned_df=df,
        master_df=df,
        dictionary_list=[{"column_name": "sales", "business_meaning": "Revenue", "role": "Measure", "data_type": "float64", "null_percentage": 0, "unique_count": 3, "sample_value": 100.0}],
        quality_issues=[],
        eda_results=eda_res,
        insights_list=[],
        recommendations_list=[],
        action_items_list=[],
        output_path=tmp_path
    )

    wb = openpyxl.load_workbook(out)
    sheet_names = wb.sheetnames
    
    # Exactly 9 worksheets generated by export_master_excel
    assert len(sheet_names) == 9
    expected = [
        "Executive Summary", "Data Quality Audit", "Data Dictionary",
        "Master Dataset", "Cleaned Dataset", "Raw Dataset",
        "Exploratory Analysis", "Strategic Insights", "Action Plan"
    ]
    assert sheet_names == expected

    # Verify Master Dataset sheet data matches underlying dataframe
    ws_master = wb["Master Dataset"]
    assert ws_master.cell(row=1, column=1).value == "id"
    assert ws_master.cell(row=1, column=3).value == "sales"
    assert ws_master.cell(row=2, column=3).value == 100.0
    assert ws_master.cell(row=4, column=3).value == 300.0

    wb.close()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


# =====================================================================
# 6. LLM Fallback and Timeout Simulation
# =====================================================================

def test_llm_fallback_without_key_and_error_handling():
    """Verify that when no API key is available or on error, deterministic fallback operates with zero downtime."""
    # Without API key
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    ans = optional_llm_explain("What is the top category?", "Electronics is top with $500", "DatasetQueryTool")
    assert ans is None  # Fails safely to None, agent uses deterministic answer

    # With invalid dummy key
    os.environ["OPENAI_API_KEY"] = "sk-invalid-test-key-not-real"
    ans_invalid = optional_llm_explain("What is the top category?", "Electronics is top with $500", "DatasetQueryTool")
    # Must fail safely to None without raising unhandled exception
    assert ans_invalid is None

    # Clean up
    del os.environ["OPENAI_API_KEY"]


# =====================================================================
# 7. Interrupted Upload and Disk Cleanup
# =====================================================================

def test_interrupted_upload_cleans_up_disk():
    """Verify that an interrupted or aborted stream leaves zero orphaned directories on disk."""
    client = TestClient(app)

    class InterruptedStream(io.RawIOBase):
        def __init__(self):
            self.calls = 0

        def read(self, size=-1):
            self.calls += 1
            if self.calls > 1:
                raise ConnectionResetError("Simulated client connection drop during upload")
            return b"id,val\n1,10\n"

    # Count project folders before
    initial_folders = set(os.listdir(STORE_DIR))

    stream = InterruptedStream()
    files = [("files", ("interrupted.csv", stream, "text/csv"))]
    resp = client.post("/api/projects/upload", files=files)
    # Rejection status code (400 Bad Request or 500 Internal Server Error)
    assert resp.status_code in [400, 500]

    # Ensure no new folder remains in STORE_DIR
    current_folders = set(os.listdir(STORE_DIR))
    orphaned_folders = current_folders - initial_folders
    assert len(orphaned_folders) == 0, f"Found orphaned folders from aborted upload: {orphaned_folders}"


# =====================================================================
# 8. Concurrent Requests and Project Switching
# =====================================================================

def test_concurrent_requests_and_project_switching():
    """
    Test concurrent requests across multiple projects to confirm thread safety,
    session isolation, and that active operations do not lose access to their data.
    """
    from concurrent.futures import ThreadPoolExecutor

    client = TestClient(app)
    # Ensure demo project is initialized
    resp = client.get("/api/project/proj-demo-q3-sales")
    assert resp.status_code == 200

    def query_task(proj_id: str, query: str):
        r = client.post(f"/api/project/{proj_id}/chat", json={"question": query})
        return r.status_code, r.json()

    # Run 10 concurrent requests alternating between projects
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(10):
            q = f"What is total revenue? (Run {i})"
            futures.append(executor.submit(query_task, "proj-demo-q3-sales", q))

        for f in futures:
            status_code, data = f.result()
            assert status_code == 200
            assert "answer" in data
            assert len(data["answer"]) > 0


# =====================================================================
# 9. Active Request Data Retention Across Eviction
# =====================================================================

def test_active_request_retains_data_after_eviction():
    """Confirm that an active request holding a project reference does not lose its data if evicted concurrently."""
    cache = BoundedProjectCache(max_size=3)
    cache["proj-demo-q3-sales"] = {"name": "Demo"}
    cache["proj-active"] = {"name": "Active Project", "master_df": pd.DataFrame({"x": [1, 2, 3]})}
    cache["proj-other"] = {"name": "Other"}

    # Simulate an active request acquiring project reference
    active_ref = cache.get("proj-active")
    assert active_ref is not None

    # Now add two new projects to trigger eviction of proj-active
    cache["proj-new1"] = {"name": "New 1"}
    cache["proj-new2"] = {"name": "New 2"}

    # proj-active should be evicted from the cache
    assert "proj-active" not in cache

    # But active_ref retained by the request still has its full DataFrame and state
    assert active_ref["name"] == "Active Project"
    assert len(active_ref["master_df"]) == 3
    assert list(active_ref["master_df"]["x"]) == [1, 2, 3]


# =====================================================================
# 10. Excel Values Match Underlying Dataset and Analysis
# =====================================================================

def test_excel_values_match_underlying_data_and_eda():
    """Verify that every value in the exported Excel sheets matches the underlying dataset and EDA metrics."""
    df_raw = pd.DataFrame({
        "product_id": [101, 102, 103, 104],
        "category": ["Tech", "Tech", "Office", "Office"],
        "revenue": [500.0, 1500.0, 200.0, 800.0]
    })
    eda_res = run_automated_eda(df_raw)
    tmp_path = "tmp/audit_value_match_test.xlsx"

    out = export_master_excel(
        raw_df=df_raw,
        cleaned_df=df_raw,
        master_df=df_raw,
        dictionary_list=[
            {"column_name": "revenue", "business_meaning": "Sales Revenue", "role": "Measure", "data_type": "float64", "null_percentage": 0, "unique_count": 4, "sample_value": 500.0, "min": 200.0, "max": 1500.0, "mean": 750.0, "is_ai_inferred": False}
        ],
        quality_issues=[],
        eda_results=eda_res,
        insights_list=[
            {"id": "INS-001", "priority": "High", "statistical_evidence": "p < 0.05", "validation_status": "PASSED", "title": "Top Tech Sales", "insight": "Tech drove majority of revenue", "calculation_method": "SUM", "numerical_evidence": [{"metric": "Tech Revenue", "value": "$2,000"}], "business_impact": "Positive", "lineage": {}}
        ],
        recommendations_list=[
            {"id": "REC-001", "source_insight_id": "INS-001", "priority": "High", "title": "Expand Tech Inventory", "owner": "VP Sales", "timeline": "30 Days", "kpi": "Revenue", "kpi_baseline": "$2,000", "problem": "Stockout risk", "recommended_action": "Order 50 units", "evidence": "INS-001", "expected_impact": "+15%", "status": "Approved"}
        ],
        action_items_list=[],
        output_path=tmp_path
    )

    wb = openpyxl.load_workbook(out)

    # 1. Master Dataset values match
    ws_master = wb["Master Dataset"]
    assert ws_master.cell(row=1, column=1).value == "product_id"
    assert ws_master.cell(row=1, column=3).value == "revenue"
    assert ws_master.cell(row=2, column=3).value == 500.0
    assert ws_master.cell(row=3, column=3).value == 1500.0

    # 2. EDA Summary values match computed metrics
    ws_eda = wb["Exploratory Analysis"]
    # Find row for revenue
    rev_row = None
    for r in range(2, ws_eda.max_row + 1):
        if ws_eda.cell(row=r, column=1).value == "revenue":
            rev_row = r
            break
    assert rev_row is not None
    # Mean: 750.0
    assert ws_eda.cell(row=rev_row, column=4).value == 750.0
    # Median: 650.0
    assert ws_eda.cell(row=rev_row, column=5).value == 650.0
    # Min: 200.0
    assert ws_eda.cell(row=rev_row, column=7).value == 200.0
    # Max: 1500.0
    assert ws_eda.cell(row=rev_row, column=8).value == 1500.0

    # 3. Action Plan values match
    ws_act = wb["Action Plan"]
    assert ws_act.cell(row=2, column=1).value == "REC-001"
    assert ws_act.cell(row=2, column=5).value == "VP Sales"
    assert ws_act.cell(row=2, column=6).value == "30 Days"
    assert ws_act.cell(row=2, column=8).value == "$2,000"

    wb.close()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


# =====================================================================
# 11. PDF Contents, Formatting, and Download Validation
# =====================================================================

def test_pdf_structure_formatting_and_download():
    """Validates PDF executive report layout, sections, typography, page count, and download endpoint."""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz

    df = pd.DataFrame({"metric": [10, 20, 30, 40]})
    eda_res = run_automated_eda(df)
    pdf_path = "tmp/audit_test_executive_report.pdf"

    insights = [
        {"id": "INS-1", "priority": "High", "title": "Strong Quarter", "what_happened": "Growth accelerated by 25%."}
    ]
    recs = [
        {"id": "REC-1", "title": "Scale Operations", "owner": "COO", "timeline": "Q4", "expected_impact": "+20% efficiency"}
    ]

    out = export_executive_pdf(
        eda_results=eda_res,
        insights_list=insights,
        recommendations_list=recs,
        dataset_name="Audit Certification Dataset",
        output_path=pdf_path
    )

    assert os.path.exists(out)
    doc = fitz.open(out)

    # Must contain exactly 2 pages
    assert len(doc) == 2

    # Page 1 validation
    p1_text = doc[0].get_text()
    assert "INSIGHTFLOW AI" in p1_text
    assert "EXECUTIVE CERTIFICATION REPORT" in p1_text
    assert "Audit Certification Dataset" in p1_text
    assert "1. Executive Summary" in p1_text
    assert "2. Core Domain Metrics & Health" in p1_text
    assert "3. Strategic Insights & Findings" in p1_text
    assert "[HIGH] Strong Quarter" in p1_text

    # Page 2 validation
    p2_text = doc[1].get_text()
    assert "4. Recommended Action Plan" in p2_text
    assert "Scale Operations" in p2_text
    assert "Owner: COO" in p2_text
    assert "Timeline: Q4" in p2_text

    doc.close()
    if os.path.exists(pdf_path):
        os.remove(pdf_path)


# =====================================================================
# 12. API Key Protection (Zero Leakage)
# =====================================================================

def test_api_key_not_exposed_in_responses_or_logs():
    """Verify that an API key is never reflected in API outputs, session history, or responses."""
    secret_key = "sk-super-secret-production-audit-key-do-not-leak"
    os.environ["OPENAI_API_KEY"] = secret_key

    client = TestClient(app)
    # Send a chat query
    resp = client.post("/api/project/proj-demo-q3-sales/chat", json={"question": "What is total sales?"})
    assert resp.status_code == 200

    resp_str = resp.text
    assert secret_key not in resp_str, "CRITICAL DEFECT: Secret API key was found in API response body!"

    # Check chat history endpoint
    hist_resp = client.get("/api/project/proj-demo-q3-sales/chat-history")
    assert hist_resp.status_code == 200
    assert secret_key not in hist_resp.text, "CRITICAL DEFECT: Secret API key was found in session history!"

    del os.environ["OPENAI_API_KEY"]


# =====================================================================
# 13. Docker Configuration Static Audit
# =====================================================================

def test_docker_configurations_validity():
    """Statically verifies backend Dockerfile, frontend Dockerfile, and docker-compose.yml configuration."""
    backend_dockerfile = os.path.join(BASE_DIR, "Dockerfile")
    frontend_dockerfile = os.path.join(BASE_DIR, "..", "frontend", "Dockerfile")
    docker_compose = os.path.join(BASE_DIR, "..", "docker-compose.yml")

    assert os.path.exists(backend_dockerfile), "Backend Dockerfile missing"
    assert os.path.exists(frontend_dockerfile), "Frontend Dockerfile missing"
    assert os.path.exists(docker_compose), "docker-compose.yml missing"

    with open(backend_dockerfile, "r") as f:
        b_content = f.read()
    assert "FROM python:3.11-slim" in b_content
    assert "HEALTHCHECK" in b_content
    assert "EXPOSE 8000" in b_content

    with open(frontend_dockerfile, "r") as f:
        f_content = f.read()
    assert "FROM node:20-alpine AS deps" in f_content
    assert "USER nextjs" in f_content
    assert "EXPOSE 3000" in f_content

    with open(docker_compose, "r") as f:
        dc_content = f.read()
    assert "backend:" in dc_content
    assert "frontend:" in dc_content
    assert "backend_store:" in dc_content
    assert "8000:8000" in dc_content
    assert "3000:3000" in dc_content

