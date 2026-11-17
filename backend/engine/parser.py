"""
InsightFlow AI — Data Ingestion, Structure Detection & Data Contract Engine
Provides smart header detection, title banner removal, empty row/column filtering,
unnamed column cleanup, duplicate column handling, and unified Dataset Contract schema generation.
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None


def detect_tabular_region(df_raw: pd.DataFrame) -> Tuple[int, pd.DataFrame, Dict[str, Any]]:
    """
    Intelligently inspects a raw parsed DataFrame (CSV or Excel sheet) to locate:
    1. Header row index (bypassing title banners, instruction notes, or empty rows).
    2. Data boundaries (excluding footer notes, empty rows, and unneeded blank columns).
    3. Duplicate and unnamed column cleaning.

    Returns (header_row_index, cleaned_df, metadata_info).
    """
    if df_raw.empty:
        return 0, df_raw, {"title_rows_removed": 0, "blank_rows_removed": 0, "unnamed_cols_removed": 0}

    # Inspect up to first 25 rows to identify the header row
    inspect_limit = min(25, len(df_raw))
    row_scores = []

    for r_idx in range(inspect_limit):
        row_vals = df_raw.iloc[r_idx].values
        non_null_count = sum(1 for x in row_vals if pd.notnull(x) and str(x).strip() != "")
        
        if non_null_count == 0:
            row_scores.append(-100) # Blank row
            continue

        # Single string spanning across row -> Title banner / report header
        if non_null_count == 1:
            row_scores.append(-50)
            continue

        # Calculate string vs numeric ratio and unique ratio for non-null cells
        str_count = 0
        num_count = 0
        val_set = set()

        for val in row_vals:
            if pd.notnull(val) and str(val).strip() != "":
                s_val = str(val).strip()
                val_set.add(s_val)
                # Check if value is numeric or float string
                try:
                    float(s_val.replace(',', ''))
                    num_count += 1
                except ValueError:
                    str_count += 1

        total_valid = str_count + num_count
        if total_valid == 0:
            score = -10
        else:
            # Headers typically have high string ratio and high uniqueness
            string_ratio = str_count / total_valid
            unique_ratio = len(val_set) / total_valid
            coverage_ratio = total_valid / max(len(row_vals), 1)

            score = (string_ratio * 40) + (unique_ratio * 40) + (coverage_ratio * 20)

            # Bonus if next row has distinct numeric/date data
            if r_idx + 1 < len(df_raw):
                next_row_vals = df_raw.iloc[r_idx + 1].values
                next_num_count = 0
                for nv in next_row_vals:
                    if pd.notnull(nv):
                        try:
                            float(str(nv).strip().replace(',', ''))
                            next_num_count += 1
                        except ValueError:
                            pass
                if next_num_count > 0:
                    score += 15

        row_scores.append(score)

    best_header_idx = int(np.argmax(row_scores)) if row_scores and max(row_scores) > 0 else 0
    title_rows_removed = best_header_idx

    # Re-slice DataFrame starting from detected header row
    header_names = df_raw.iloc[best_header_idx].values
    df_sliced = df_raw.iloc[best_header_idx + 1:].copy()
    df_sliced.columns = header_names

    # Clean column names with whitespace normalization (collapse multi-spaces, tabs, newlines)
    raw_cols = [
        re.sub(r'\s+', ' ', str(c).replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')).strip()
        if pd.notnull(c) and str(c).strip() != "" else ""
        for c in df_sliced.columns
    ]

    # Handle unnamed/empty column names & duplicate column names
    clean_cols = []
    seen_names = {}
    unnamed_cols_removed = 0

    for idx, c in enumerate(raw_cols):
        series = df_sliced.iloc[:, idx]
        is_all_empty = series.isnull().all() or (series.astype(str).str.strip() == "").all() or (series.astype(str).str.strip().isin(['nan', 'none', 'null', ''])).all()

        if is_all_empty or c == "" or c.lower().startswith("unnamed:") or c.lower().startswith("column_"):
            if is_all_empty:
                unnamed_cols_removed += 1
                clean_cols.append(f"__DROP_EMPTY_COL_{idx}__")
                continue
            else:
                c_name = f"Column_{idx + 1}"
        else:
            c_name = c

        # Handle duplicates
        if c_name in seen_names:
            seen_names[c_name] += 1
            final_name = f"{c_name}_{seen_names[c_name]}"
        else:
            seen_names[c_name] = 0
            final_name = c_name

        clean_cols.append(final_name)

    df_sliced.columns = clean_cols

    # Drop columns marked for dropping
    drop_mask = [not str(c).startswith("__DROP_EMPTY_COL_") for c in df_sliced.columns]
    df_sliced = df_sliced.loc[:, drop_mask]

    # Drop 100% completely blank rows
    initial_len = len(df_sliced)
    df_sliced = df_sliced.dropna(how='all')
    
    # Also drop rows where all cell values are empty strings
    non_empty_mask = df_sliced.apply(lambda row: any(pd.notnull(x) and str(x).strip() != "" for x in row), axis=1)
    df_sliced = df_sliced[non_empty_mask]
    blank_rows_removed = initial_len - len(df_sliced)

    # Detect trailing footer notes / metadata rows at the bottom of the table
    footer_keywords = ['information', 'report has been generated', 'search criteria', 'query builder', 'confidential', 'page ']
    end_idx = len(df_sliced)
    while end_idx > 0:
        row_vals = [x for x in df_sliced.iloc[end_idx - 1].values if pd.notnull(x) and str(x).strip() != ""]
        if not row_vals:
            end_idx -= 1
            continue
        coverage = len(row_vals) / max(len(df_sliced.columns), 1)
        row_str = " ".join(str(x).lower().strip() for x in row_vals)
        is_footer = (coverage <= 0.35 and any(k in row_str for k in footer_keywords)) or (coverage <= 0.2 and len(row_vals) <= 2)
        if is_footer:
            end_idx -= 1
        else:
            break

    footer_rows_removed = len(df_sliced) - end_idx
    if end_idx < len(df_sliced):
        df_sliced = df_sliced.iloc[:end_idx].copy()

    df_sliced.reset_index(drop=True, inplace=True)

    metadata_info = {
        "header_row_index": best_header_idx,
        "title_rows_removed": title_rows_removed,
        "blank_rows_removed": blank_rows_removed,
        "footer_rows_removed": footer_rows_removed,
        "unnamed_cols_removed": unnamed_cols_removed
    }

    return best_header_idx, df_sliced, metadata_info


def parse_and_contract_file(file_path: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Parses a raw Excel/CSV file, applies smart structure detection, cleans parsing artifacts,
    and returns a unified DatasetContract object dictionary.
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

    raw_dfs = {}
    primary_sheet = "Sheet1"

    if ext in ['.xlsx', '.xls']:
        from engine.matrix_adapter import IndustrialMatrixAdapter
        if IndustrialMatrixAdapter.is_industrial_matrix_workbook(file_path):
            matrix_dfs = IndustrialMatrixAdapter.parse_workbook(file_path)
            target_key = "Potline_Master_TimeSeries" if "Potline_Master_TimeSeries" in matrix_dfs else list(matrix_dfs.keys())[0]
            cleaned_df = matrix_dfs[target_key]
            primary_sheet = target_key
            raw_df_selected = cleaned_df
            header_idx = 0
            struct_meta = {
                "header_row_index": 0,
                "title_rows_removed": 0,
                "blank_rows_removed": 0,
                "unnamed_cols_removed": 0
            }
        else:
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names

            ignore_keywords = ['readme', 'cover', 'summary', 'index', 'table of contents', 'data dictionary', 'data quality', 'exploratory data', 'eda', 'insights & story', 'insights', 'recommendations', 'action plan', 'export']
            
            valid_sheets = {}
            for s in sheet_names:
                s_low = s.lower().strip()
                if any(k in s_low for k in ignore_keywords) and not any(k in s_low for k in ['raw', 'master', 'data', 'transaction', 'order', 'sales']):
                    continue
                try:
                    df_raw = xl.parse(s, header=None) # Read raw without assuming row 0 is header
                    if len(df_raw) > 0 and len(df_raw.columns) > 0:
                        valid_sheets[s] = df_raw
                except Exception:
                    pass

            if not valid_sheets:
                for s in sheet_names:
                    try:
                        df_raw = xl.parse(s, header=None)
                        if len(df_raw) > 0:
                            valid_sheets[s] = df_raw
                    except Exception:
                        pass

            target_sheet = sheet_name if sheet_name and sheet_name in valid_sheets else list(valid_sheets.keys())[0] if valid_sheets else sheet_names[0]
            raw_df_selected = valid_sheets.get(target_sheet, xl.parse(target_sheet, header=None))
            primary_sheet = target_sheet

            # Perform smart structure detection
            header_idx, cleaned_df, struct_meta = detect_tabular_region(raw_df_selected)

    elif ext == '.csv':
        raw_df_selected = pd.read_csv(file_path, header=None)
        primary_sheet = "Sheet1"
        header_idx, cleaned_df, struct_meta = detect_tabular_region(raw_df_selected)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Generate Column Contracts
    col_contracts = {}
    for col in cleaned_df.columns:
        series = cleaned_df[col]
        null_count = int(series.isnull().sum())
        non_null_series = series.dropna()
        unique_count = int(non_null_series.nunique())

        # Physical dtype
        dtype_str = str(series.dtype)

        col_contracts[col] = {
            "column_name": col,
            "physical_dtype": dtype_str,
            "null_count": null_count,
            "null_percentage": round((null_count / max(len(cleaned_df), 1)) * 100, 2),
            "unique_count": unique_count,
            "sample_values": [str(x) for x in non_null_series.head(3).tolist()],
            "is_constant": unique_count <= 1,
            "is_unique": unique_count == len(cleaned_df) and len(cleaned_df) > 0
        }

    # Unified Dataset Contract Schema
    dataset_contract = {
        "metadata": {
            "file_name": filename,
            "file_size_bytes": file_size,
            "source_type": ext.replace('.', '').upper(),
            "sheet_name": primary_sheet,
            "parsed_at": pd.Timestamp.now().isoformat()
        },
        "structure": {
            "header_row_index": header_idx,
            "title_rows_removed": struct_meta["title_rows_removed"],
            "blank_rows_removed": struct_meta["blank_rows_removed"],
            "unnamed_cols_removed": struct_meta["unnamed_cols_removed"],
            "record_count": len(cleaned_df),
            "column_count": len(cleaned_df.columns),
            "column_names": list(cleaned_df.columns)
        },
        "quality_summary": {
            "total_nulls": sum(c["null_count"] for c in col_contracts.values()),
            "total_cells": len(cleaned_df) * len(cleaned_df.columns) if len(cleaned_df.columns) > 0 else 1,
            "completeness_score": round((1.0 - (sum(c["null_count"] for c in col_contracts.values()) / max(len(cleaned_df) * len(cleaned_df.columns), 1))) * 100, 2)
        },
        "column_contracts": col_contracts,
        "lineage": {
            "raw_shape": raw_df_selected.shape,
            "cleaned_shape": cleaned_df.shape
        }
    }

    return {
        "dataset_contract": dataset_contract,
        "cleaned_df": cleaned_df,
        "raw_df": raw_df_selected
    }
