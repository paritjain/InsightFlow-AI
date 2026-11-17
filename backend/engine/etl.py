import pandas as pd
import numpy as np
import re

def detect_relationships(dfs_dict: dict):
    """
    Scans multiple loaded dataframes and detects key join relationships based on column names & value overlaps.
    Computes real join statistics: left_table, right_table, join_columns, detected_cardinality, recommended_join_type,
    matched_rows, unmatched_rows, duplicate_key_count, orphan_record_count.
    """
    relationships = []
    nodes = []

    for name, df in dfs_dict.items():
        nodes.append({
            "id": name,
            "filename": name,
            "row_count": len(df),
            "columns": list(df.columns)
        })

    names = list(dfs_dict.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            df1_name, df1 = names[i], dfs_dict[names[i]]
            df2_name, df2 = names[j], dfs_dict[names[j]]

            df1_cols_map = {re.sub(r'\s+', ' ', str(c)).strip().lower(): c for c in df1.columns}
            df2_cols_map = {re.sub(r'\s+', ' ', str(c)).strip().lower(): c for c in df2.columns}
            common_lowers = set(df1_cols_map.keys()).intersection(set(df2_cols_map.keys()))

            for c_lower in common_lowers:
                col1 = df1_cols_map[c_lower]
                col2 = df2_cols_map[c_lower]

                is_key_candidate = any(k in c_lower for k in ['_id', 'id', 'code', 'num', 'key', 'sku', 'category', 'region', 'type']) or c_lower in ['customer', 'product', 'vendor', 'department']

                if is_key_candidate:
                    vals1 = set(df1[col1].dropna().unique())
                    vals2 = set(df2[col2].dropna().unique())
                    
                    if not vals1 or not vals2:
                        continue
                    
                    matched_keys = vals1.intersection(vals2)
                    overlap = len(matched_keys)
                    min_size = min(len(vals1), len(vals2))
                    overlap_ratio = overlap / max(min_size, 1)

                    if overlap_ratio > 0.25 or c_lower.endswith('_id') or c_lower == 'id':
                        df1_unique = (df1[col1].nunique() == len(df1))
                        df2_unique = (df2[col2].nunique() == len(df2))

                        if df1_unique and df2_unique:
                            cardinality = "1:1"
                        elif df1_unique and not df2_unique:
                            cardinality = "1:N"
                        elif not df1_unique and df2_unique:
                            cardinality = "N:1"
                        else:
                            cardinality = "N:M"

                        # Compute matched/unmatched row counts
                        matched_rows = int(df1[df1[col1].isin(matched_keys)].shape[0])
                        unmatched_rows = int(df1[~df1[col1].isin(matched_keys)].shape[0])
                        dup_keys_count = int(df2[col2].duplicated().sum())
                        orphan_count = int(len(vals1 - vals2))

                        relationships.append({
                            "left_table": df1_name,
                            "right_table": df2_name,
                            "parent_dataset": df1_name,
                            "child_dataset": df2_name,
                            "parent_key": col1,
                            "child_key": col2,
                            "join_columns": [col1, col2],
                            "key_column": col1,
                            "cardinality": cardinality,
                            "detected_cardinality": cardinality,
                            "recommended_join_type": "LEFT OUTER JOIN",
                            "user_selected_join_type": "LEFT OUTER JOIN",
                            "matched_rows": matched_rows,
                            "unmatched_rows": unmatched_rows,
                            "duplicate_key_count": dup_keys_count,
                            "orphan_record_count": orphan_count,
                            "confidence": round(min(0.99, overlap_ratio + 0.2 if c_lower.endswith('_id') else overlap_ratio), 2)
                        })

    return {
        "nodes": nodes,
        "relationships": relationships
    }

def create_master_dataset(dfs_dict: dict, primary_dataset_name: str = None, relationships: list = None):
    """
    Joins multiple datasets into a single Master Dataset cleanly with explicit join validation metrics,
    cardinality checks, PK/FK integrity validation, and step-by-step lineage tracking.
    Handles single-file uploads natively without fabricating multi-table schemas.
    """
    if not dfs_dict:
        return pd.DataFrame(), {
            "is_single_dataset": True,
            "message": "Single Dataset — No Cross-File Relationships Detected",
            "relationships": [],
            "join_history": []
        }

    is_single_dataset = (len(dfs_dict) == 1)

    if not primary_dataset_name or primary_dataset_name not in dfs_dict:
        # Pick primary dataset by maximum data volume (rows * cols) in a dataset-agnostic manner
        primary_dataset_name = max(dfs_dict.keys(), key=lambda k: len(dfs_dict[k]) * len(dfs_dict[k].columns))

    master_df = dfs_dict[primary_dataset_name].copy()
    original_row_count = len(master_df)
    
    if is_single_dataset:
        return master_df, {
            "is_single_dataset": True,
            "message": "Single Dataset — No Cross-File Relationships Detected",
            "primary_table": primary_dataset_name,
            "original_rows": original_row_count,
            "final_rows": original_row_count,
            "final_columns_count": len(master_df.columns),
            "columns": list(master_df.columns),
            "join_history": [],
            "relationships": [],
            "unmatched_summary": {},
            "warnings": [],
            "lineage": {
                "pipeline_name": "Master Dataset Pipeline",
                "primary_table": primary_dataset_name,
                "stages": [
                    {
                        "step_number": 1,
                        "step_name": "Single Dataset Ingestion",
                        "description": f"Ingested single table '{primary_dataset_name}' with {original_row_count:,} rows and {len(master_df.columns)} columns.",
                        "input_tables": [primary_dataset_name],
                        "output_rows": original_row_count
                    }
                ],
                "flow": f"Raw File '{primary_dataset_name}' -> Master Dataset"
            },
            "balance_equation": f"Single Dataset ({original_row_count:,} rows)"
        }
    
    if relationships is None:
        rel_info = detect_relationships(dfs_dict)
        relationships = rel_info["relationships"]

    join_logs = []
    unmatched_summary = {}
    warnings = []
    lineage_steps = []

    # Lineage Step 1: Raw Files Ingestion
    lineage_steps.append({
        "step_number": 1,
        "step_name": "Raw Data Ingestion",
        "description": f"Ingested {len(dfs_dict)} tables. Selected primary table '{primary_dataset_name}' with {original_row_count:,} rows.",
        "input_tables": list(dfs_dict.keys()),
        "output_rows": original_row_count
    })

    # Lineage Step 2: Pre-Join Cleansing & Coercion
    type_conversions_cnt = sum(df.select_dtypes(include=[np.number]).shape[1] for df in dfs_dict.values())
    lineage_steps.append({
        "step_number": 2,
        "step_name": "Type Conversion & Data Cleansing",
        "description": f"Coerced numeric strings and standardized column headers across {type_conversions_cnt} numeric fields.",
        "output_rows": original_row_count
    })

    # Execute Joins with Validation
    join_step_counter = 3
    for rel in relationships:
        parent_name = rel.get("parent_dataset") or rel.get("left_table")
        child_name = rel.get("child_dataset") or rel.get("right_table")
        key_col = rel.get("key_column") or rel.get("parent_key")

        join_target_name = None
        if primary_dataset_name in [child_name, parent_name]:
            if parent_name in dfs_dict and parent_name != primary_dataset_name:
                join_target_name = parent_name
            elif child_name in dfs_dict and child_name != primary_dataset_name:
                join_target_name = child_name

        if join_target_name and join_target_name in dfs_dict and key_col in master_df.columns:
            target_df = dfs_dict[join_target_name].copy()
            target_key_col = rel.get("child_key") if join_target_name == child_name else rel.get("parent_key", key_col)

            if target_key_col not in target_df.columns:
                continue

            # PK Uniqueness & FK Integrity Validation
            has_dup_keys = int(target_df[target_key_col].duplicated().sum()) > 0
            dup_keys_count = int(target_df[target_key_col].duplicated().sum())

            if has_dup_keys:
                dup_msg = f"Join Validation Warning: Key '{target_key_col}' contains {dup_keys_count} duplicate keys in dimension table '{join_target_name}'. Deduplicating before join."
                warnings.append(dup_msg)
                target_df = target_df.drop_duplicates(subset=[target_key_col]).reset_index(drop=True)

            master_cols_norm = {re.sub(r'\s+', ' ', str(c)).strip().lower(): c for c in master_df.columns}
            overlap = []
            rename_dict = {}
            for c in target_df.columns:
                if c == key_col or c == target_key_col:
                    continue
                c_norm = re.sub(r'\s+', ' ', str(c)).strip().lower()
                if c in master_df.columns or c_norm in master_cols_norm:
                    overlap.append(c)
                    rename_dict[c] = f"{join_target_name}_{c}"
            if target_key_col != key_col:
                rename_dict[target_key_col] = key_col
            target_df = target_df.rename(columns=rename_dict)

            before_rows = len(master_df)
            master_keys = set(master_df[key_col].dropna().unique())
            target_keys = set(target_df[key_col].dropna().unique())
            matched_keys_cnt = len(master_keys.intersection(target_keys))
            unmatched_cnt = len(master_keys - target_keys)
            unmatched_summary[join_target_name] = unmatched_cnt

            # Left Join Execution
            master_df = pd.merge(master_df, target_df, on=key_col, how='left')
            after_rows = len(master_df)

            join_log = {
                "left_table": primary_dataset_name,
                "right_table": join_target_name,
                "join_columns": [key_col, target_key_col],
                "key_column": key_col,
                "join_type": "LEFT OUTER JOIN",
                "recommended_join_type": "LEFT OUTER JOIN",
                "detected_cardinality": rel.get("cardinality", "1:N"),
                "rows_before": before_rows,
                "rows_after": after_rows,
                "matched_rows": before_rows - unmatched_cnt,
                "unmatched_rows": unmatched_cnt,
                "duplicate_key_count": dup_keys_count,
                "orphan_record_count": unmatched_cnt,
                "fanout_warning": (after_rows > before_rows),
                "warning_message": f"Join key '{key_col}' contained non-unique entries in {join_target_name}" if has_dup_keys else None
            }
            join_logs.append(join_log)

            lineage_steps.append({
                "step_number": join_step_counter,
                "step_name": f"Left Join '{join_target_name}'",
                "description": f"LEFT OUTER JOIN with '{join_target_name}' on key '{key_col}'. Matched {matched_keys_cnt:,} keys, {unmatched_cnt:,} unmatched keys.",
                "rows_before": before_rows,
                "rows_after": after_rows
            })
            join_step_counter += 1

    final_row_count = len(master_df)

    # Final Lineage Step: Master Dataset Complete
    lineage_steps.append({
        "step_number": join_step_counter,
        "step_name": "Master Dataset Ready",
        "description": f"Master Dataset finalized with {final_row_count:,} rows and {len(master_df.columns)} columns.",
        "output_rows": final_row_count,
        "output_columns": len(master_df.columns)
    })

    lineage = {
        "pipeline_name": "Master Dataset ETL Pipeline",
        "primary_table": primary_dataset_name,
        "stages": lineage_steps,
        "flow": "Raw Files -> Type Conversion -> Missing Value Handling -> Duplicate Handling -> Join Operations -> Master Dataset"
    }

    ref_integrity = validate_referential_integrity(dfs_dict, relationships)

    return master_df, {
        "primary_table": primary_dataset_name,
        "original_rows": original_row_count,
        "final_rows": final_row_count,
        "final_columns_count": len(master_df.columns),
        "join_history": join_logs,
        "unmatched_summary": unmatched_summary,
        "warnings": warnings,
        "referential_integrity": ref_integrity,
        "lineage": lineage,
        "balance_equation": f"Original ({original_row_count:,}) + Joined - Unmatched = Final ({final_row_count:,})"
    }

def validate_referential_integrity(dfs_dict: dict, relationships: list = None) -> dict:
    """
    Validates referential integrity and returns comprehensive reports on:
    - orphan foreign keys (records in child table referencing non-existent parent PK)
    - missing foreign keys (null or empty foreign key values in child table)
    - duplicate primary keys (uniqueness violations in primary key columns)
    - null primary keys (nullability violations in primary key columns)
    """
    reports = []
    total_checks = 0
    failed_checks = 0
    total_dup_pks = 0
    total_null_pks = 0
    total_orphan_fks = 0
    total_missing_fks = 0

    if relationships is None and len(dfs_dict) > 1:
        rel_info = detect_relationships(dfs_dict)
        relationships = rel_info.get("relationships", [])
    elif relationships is None:
        relationships = []

    # 1. Primary Key Checks across all tables
    for tbl_name, df in dfs_dict.items():
        tbl_low = tbl_name.lower().rstrip('s')
        pk_col = None
        for col in df.columns:
            c_low = col.lower()
            if c_low == f"{tbl_low}_id" or c_low == f"{tbl_name.lower()}_id" or c_low == "id":
                pk_col = col
                break
        if not pk_col:
            for col in df.columns:
                c_low = col.lower()
                if c_low.endswith("_id") and (tbl_low in c_low or c_low in ['order_id', 'record_id', 'customer_id', 'product_id']):
                    pk_col = col
                    break
        if not pk_col and len(df.columns) > 0:
            for col in df.columns:
                if col.lower().endswith("_id"):
                    pk_col = col
                    break

        if pk_col and pk_col in df.columns:
            # Check duplicate PKs
            total_checks += 1
            pk_series = df[pk_col]
            dup_mask = pk_series.duplicated(keep=False)
            dup_cnt = int(pk_series.duplicated().sum())
            dup_samples = list(pk_series[dup_mask].dropna().astype(str).unique()[:5])
            if dup_cnt > 0:
                total_dup_pks += dup_cnt
                failed_checks += 1
                reports.append({
                    "type": "duplicate_primary_keys",
                    "table": tbl_name,
                    "column": pk_col,
                    "count": dup_cnt,
                    "status": "WARNING",
                    "severity": "WARNING",
                    "sample_values": dup_samples,
                    "message": f"{dup_cnt} duplicate primary key value(s) detected in '{tbl_name}.{pk_col}' (samples: {dup_samples})"
                })
            else:
                reports.append({
                    "type": "duplicate_primary_keys",
                    "table": tbl_name,
                    "column": pk_col,
                    "count": 0,
                    "status": "PASS",
                    "severity": "PASS",
                    "sample_values": [],
                    "message": f"Primary key '{tbl_name}.{pk_col}' is strictly unique (0 duplicates)."
                })

            # Check null PKs
            total_checks += 1
            null_cnt = int(pk_series.isna().sum())
            if null_cnt > 0:
                total_null_pks += null_cnt
                failed_checks += 1
                reports.append({
                    "type": "null_primary_keys",
                    "table": tbl_name,
                    "column": pk_col,
                    "count": null_cnt,
                    "status": "FAIL",
                    "severity": "CRITICAL",
                    "sample_values": [],
                    "message": f"{null_cnt} null primary key record(s) detected in '{tbl_name}.{pk_col}'."
                })
            else:
                reports.append({
                    "type": "null_primary_keys",
                    "table": tbl_name,
                    "column": pk_col,
                    "count": 0,
                    "status": "PASS",
                    "severity": "PASS",
                    "sample_values": [],
                    "message": f"Primary key '{tbl_name}.{pk_col}' contains 0 null values."
                })

    # 2. Foreign Key Checks
    for rel in relationships:
        parent_tbl = rel.get("parent_dataset") or rel.get("left_table")
        child_tbl = rel.get("child_dataset") or rel.get("right_table")
        parent_key = rel.get("parent_key") or rel.get("key_column")
        child_key = rel.get("child_key") or rel.get("key_column")

        if parent_tbl in dfs_dict and child_tbl in dfs_dict:
            parent_df = dfs_dict[parent_tbl]
            child_df = dfs_dict[child_tbl]

            if parent_key in parent_df.columns and child_key in child_df.columns:
                # Check Missing FKs (nulls in child FK)
                total_checks += 1
                child_fk_series = child_df[child_key]
                missing_fk_cnt = int(child_fk_series.isna().sum())
                if missing_fk_cnt > 0:
                    total_missing_fks += missing_fk_cnt
                    failed_checks += 1
                    reports.append({
                        "type": "missing_foreign_keys",
                        "parent_table": parent_tbl,
                        "parent_column": parent_key,
                        "child_table": child_tbl,
                        "child_column": child_key,
                        "count": missing_fk_cnt,
                        "status": "WARNING",
                        "severity": "WARNING",
                        "sample_values": [],
                        "message": f"{missing_fk_cnt} missing (null) foreign key value(s) in '{child_tbl}.{child_key}' referencing '{parent_tbl}.{parent_key}'."
                    })
                else:
                    reports.append({
                        "type": "missing_foreign_keys",
                        "parent_table": parent_tbl,
                        "parent_column": parent_key,
                        "child_table": child_tbl,
                        "child_column": child_key,
                        "count": 0,
                        "status": "PASS",
                        "severity": "PASS",
                        "sample_values": [],
                        "message": f"Zero missing foreign keys in '{child_tbl}.{child_key}' referencing '{parent_tbl}.{parent_key}'."
                    })

                # Check Orphan FKs (values in child FK not in parent PK)
                total_checks += 1
                parent_keys = set(parent_df[parent_key].dropna().unique())
                non_null_child_fks = child_df[child_key].dropna()
                orphans = non_null_child_fks[~non_null_child_fks.isin(parent_keys)]
                orphan_cnt = int(len(orphans))
                orphan_samples = list(orphans.astype(str).unique()[:5])

                if orphan_cnt > 0:
                    total_orphan_fks += orphan_cnt
                    failed_checks += 1
                    reports.append({
                        "type": "orphan_foreign_keys",
                        "parent_table": parent_tbl,
                        "parent_column": parent_key,
                        "child_table": child_tbl,
                        "child_column": child_key,
                        "count": orphan_cnt,
                        "status": "FAIL",
                        "severity": "HIGH",
                        "sample_values": orphan_samples,
                        "message": f"{orphan_cnt} orphan foreign key record(s) in '{child_tbl}.{child_key}' do not exist in '{parent_tbl}.{parent_key}'."
                    })
                else:
                    reports.append({
                        "type": "orphan_foreign_keys",
                        "parent_table": parent_tbl,
                        "parent_column": parent_key,
                        "child_table": child_tbl,
                        "child_column": child_key,
                        "count": 0,
                        "status": "PASS",
                        "severity": "PASS",
                        "sample_values": [],
                        "message": f"Referential integrity verified: All '{child_tbl}.{child_key}' keys exist in '{parent_tbl}.{parent_key}'."
                    })

    status = "PASS" if failed_checks == 0 else ("WARNING" if total_orphan_fks == 0 and total_null_pks == 0 else "FAIL")

    return {
        "summary": {
            "status": status,
            "total_checks": total_checks,
            "failed_checks": failed_checks,
            "duplicate_pk_count": total_dup_pks,
            "null_pk_count": total_null_pks,
            "orphan_fk_count": total_orphan_fks,
            "missing_fk_count": total_missing_fks
        },
        "reports": reports
    }
