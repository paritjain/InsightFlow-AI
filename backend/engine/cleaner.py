import pandas as pd
import numpy as np
import os
import re

def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects string columns containing formatted numbers (e.g. '$1,500.50', '₹2,000', '25%', '1,200', '€50.00', '£100')
    and converts them to numeric float64 / int64 dtypes.
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        if pd.api.types.is_string_dtype(df_clean[col]) or df_clean[col].dtype == 'object':
            series_str = (df_clean[col].astype(str)
                          .str.replace('$', '', regex=False)
                          .str.replace('₹', '', regex=False)
                          .str.replace('€', '', regex=False)
                          .str.replace('£', '', regex=False)
                          .str.replace(',', '', regex=False)
                          .str.replace('%', '', regex=False)
                          .str.strip())
            is_accounting_negative = series_str.str.startswith('(') & series_str.str.endswith(')')
            series_str_cleaned = series_str.str.replace('(', '', regex=False).str.replace(')', '', regex=False).str.strip()
            converted = pd.to_numeric(series_str_cleaned, errors='coerce')
            converted = converted.where(~is_accounting_negative, -converted)
            
            non_null_orig = df_clean[col].dropna()
            if len(non_null_orig) > 0:
                valid_ratio = converted.notna().sum() / len(non_null_orig)
                if valid_ratio >= 0.6 and converted.notna().sum() > 0:
                    df_clean[col] = converted
    return df_clean

def detect_quality_issues(df: pd.DataFrame, filename: str = "dataset"):
    """
    Scans a DataFrame and identifies specific quality issues.
    Returns a list of structured issue recommendations with full Quality Ledger metadata.
    Dataset-agnostic without domain-specific column name assumptions.
    """
    issues = []
    issue_counter = 1

    # 1. Duplicate Records
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append({
            "id": f"rule_{issue_counter}",
            "type": "duplicate_rows",
            "column": "ALL",
            "columns": list(df.columns),
            "issue": f"{dup_count} duplicate row(s) detected across dataset",
            "proposed_fix": "Remove exact duplicate rows",
            "treatment": "Remove exact duplicate rows",
            "reason": "Duplicate records inflate aggregations and distort counts.",
            "count": dup_count,
            "affected_records": dup_count,
            "impact": f"Eliminates {dup_count} redundant rows to establish unique observation baseline.",
            "reversibility": True,
            "status": "pending"
        })
        issue_counter += 1

    # 2. Formatted Numeric Columns stored as Strings
    for col in df.select_dtypes(include=['object', 'string']).columns:
        series_str = (df[col].dropna().astype(str)
                      .str.replace('$', '', regex=False)
                      .str.replace('₹', '', regex=False)
                      .str.replace('€', '', regex=False)
                      .str.replace('£', '', regex=False)
                      .str.replace(',', '', regex=False)
                      .str.replace('%', '', regex=False)
                      .str.strip())
        is_accounting_negative = series_str.str.startswith('(') & series_str.str.endswith(')')
        series_str_cleaned = series_str.str.replace('(', '', regex=False).str.replace(')', '', regex=False).str.strip()
        converted = pd.to_numeric(series_str_cleaned, errors='coerce')
        converted = converted.where(~is_accounting_negative, -converted)
        valid_cnt = converted.notna().sum()
        if len(df[col].dropna()) > 0 and (valid_cnt / len(df[col].dropna())) >= 0.7 and not col.lower().endswith('_id') and col.lower() != 'id':

            issues.append({
                "id": f"rule_{issue_counter}",
                "type": "coerce_numeric",
                "column": col,
                "columns": [col],
                "issue": f"Numeric values in '{col}' are stored as formatted text strings (e.g. '$1,500', '15%')",
                "proposed_fix": f"Parse and coerce column '{col}' to numeric float/int data type",
                "treatment": f"Coerce text column '{col}' to float/int dtype",
                "reason": "String formatted numbers cannot be aggregated, averaged, or plotted mathematically.",
                "count": int(valid_cnt),
                "affected_records": int(valid_cnt),
                "impact": f"Converts {valid_cnt} string entries to numeric values for mathematical operations.",
                "reversibility": True,
                "status": "pending"
            })
            issue_counter += 1

    # 3. String Whitespace & Casing Standardization
    for col in df.select_dtypes(include=['object', 'string']).columns:
        col_series = df[col].dropna().astype(str)
        if len(col_series) == 0:
            continue

        whitespace_mask = col_series.str.strip() != col_series
        whitespace_count = int(whitespace_mask.sum())
        if whitespace_count > 0:
            issues.append({
                "id": f"rule_{issue_counter}",
                "type": "trim_whitespace",
                "column": col,
                "columns": [col],
                "issue": f"{whitespace_count} values in '{col}' contain leading or trailing whitespace",
                "proposed_fix": f"Trim leading and trailing whitespace in column '{col}'",
                "treatment": f"Trim leading/trailing whitespace in '{col}'",
                "reason": "Unwanted whitespace causes false duplicate categories during grouping.",
                "count": whitespace_count,
                "affected_records": whitespace_count,
                "impact": f"Cleans {whitespace_count} text strings to prevent fragmented category grouping.",
                "reversibility": True,
                "status": "pending"
            })
            issue_counter += 1

        # Casing inconsistencies
        unique_vals = col_series.unique()
        lower_map = {}
        for val in unique_vals:
            lowered = str(val).strip().lower()
            if lowered in lower_map:
                lower_map[lowered].append(str(val))
            else:
                lower_map[lowered] = [str(val)]

        variants = [vals for vals in lower_map.values() if len(vals) > 1]
        if len(variants) > 0:
            val_counts = col_series.astype(str).value_counts().to_dict()
            for var_group in variants:
                # Sort by frequency so the most common casing is canonical
                sorted_by_freq = sorted(var_group, key=lambda v: val_counts.get(v, 0), reverse=True)
                canonical_val = sorted_by_freq[0]
                non_canonical = sorted_by_freq[1:]
                
                for non_can in non_canonical:
                    aff_cnt = int((col_series.astype(str) == non_can).sum())
                    if aff_cnt > 0:
                        issues.append({
                            "id": f"rule_{issue_counter}",
                            "type": "standardize_casing",
                            "column": col,
                            "columns": [col],
                            "original_value": non_can,
                            "normalized_value": canonical_val,
                            "transformation": "Standardize categorical casing",
                            "reason": "Case inconsistency / categorical normalization",
                            "issue": f"Inconsistent capitalization detected in '{col}': '{non_can}' ({aff_cnt} record(s)) ~ '{canonical_val}'. Recommended normalization: '{non_can}' -> '{canonical_val}'.",
                            "proposed_fix": f"Normalize '{non_can}' to '{canonical_val}' in column '{col}'",
                            "treatment": f"Standardize categorical casing ('{non_can}' -> '{canonical_val}')",
                            "count": aff_cnt,
                            "affected_records": aff_cnt,
                            "impact": f"Standardizes {aff_cnt} categorical record(s) from '{non_can}' to '{canonical_val}' to prevent category fragmentation.",
                            "reversibility": True,
                            "status": "pending"
                        })
                        issue_counter += 1

    # 4. Invalid / Suspicious Values (Dataset-agnostic numeric checks)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s_valid = df[col].dropna()
            if len(s_valid) > 5:
                pos_ratio = (s_valid > 0).sum() / len(s_valid)
                neg_mask = (df[col] < 0)
                neg_count = int(neg_mask.sum())
                # If column is overwhelmingly positive (>90%), negative values are likely anomalies
                if pos_ratio > 0.90 and neg_count > 0:
                    issues.append({
                        "id": f"rule_{issue_counter}",
                        "type": "fix_invalid_numeric",
                        "column": col,
                        "columns": [col],
                        "issue": f"{neg_count} invalid negative value(s) in column '{col}' (min: {df[col].min()})",
                        "proposed_fix": f"Replace negative values in '{col}' with absolute value or NaN",
                        "treatment": f"Convert negative values in '{col}' to positive absolute values",
                        "reason": "Unexpected negative numeric measurements distort calculations and baseline totals.",
                        "count": neg_count,
                        "affected_records": neg_count,
                        "impact": f"Fixes {neg_count} negative anomaly entries in '{col}'.",
                        "reversibility": True,
                        "status": "pending"
                    })
                    issue_counter += 1

    # 5. Missing Values
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            null_pct = round((null_count / len(df)) * 100, 1)
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            
            if is_numeric:
                median_val = float(df[col].median()) if not df[col].dropna().empty else 0.0
                fix_desc = f"Impute missing numeric values in '{col}' with median ({median_val})"
                treatment_text = f"Impute missing numeric values in '{col}' with median ({median_val})"
            else:
                mode_val = str(df[col].mode()[0]) if not df[col].dropna().empty else "Unknown"
                fix_desc = f"Fill missing categorical values in '{col}' with 'Unknown' (or mode: {mode_val})"
                treatment_text = f"Fill missing values in '{col}' with 'Unknown'"

            issues.append({
                "id": f"rule_{issue_counter}",
                "type": "missing_values",
                "column": col,
                "columns": [col],
                "issue": f"{null_count} missing value(s) in '{col}' ({null_pct}% of total)",
                "proposed_fix": fix_desc,
                "treatment": treatment_text,
                "reason": "Missing values cause calculations and correlations to drop incomplete rows.",
                "count": null_count,
                "affected_records": null_count,
                "impact": f"Imputes {null_count} null cells to prevent row dropping during aggregation.",
                "reversibility": True,
                "status": "pending"
            })
            issue_counter += 1

    # 6. Outliers (IQR detection for numeric columns)
    for col in df.select_dtypes(include=[np.number]).columns:
        if "id" in col.lower() or col.endswith("_ID"):
            continue
        series = pd.to_numeric(df[col], errors='coerce').astype(float).dropna()
        if len(series) < 10:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = int(outliers_mask.sum())
            if outlier_count > 0 and (outlier_count / len(df)) < 0.15:
                issues.append({
                    "id": f"rule_{issue_counter}",
                    "type": "outlier_flag",
                    "column": col,
                    "columns": [col],
                    "issue": f"{outlier_count} potential outlier value(s) in '{col}' outside IQR bounds [{round(lower_bound, 2)}, {round(upper_bound, 2)}]",
                    "proposed_fix": f"Flag outliers in column '{col}_Outlier' or cap at IQR thresholds",
                    "treatment": f"Add outlier indicator column '{col}_IsOutlier'",
                    "reason": "Outliers skew averages and distort regression and variance calculations.",
                    "count": outlier_count,
                    "affected_records": outlier_count,
                    "impact": f"Flags {outlier_count} records as extreme observations without deleting raw data.",
                    "reversibility": True,
                    "status": "pending"
                })
                issue_counter += 1

    # 7. Operational Process Gaps: Unassigned Action Owners (Prompt Section 7)
    owner_keywords = ['owner', 'assigned', 'responsible', 'lead', 'in_charge', 'assignee', 'action_by']
    for col in df.columns:
        c_low = str(col).lower().strip()
        if any(k in c_low for k in owner_keywords):
            s = df[col].astype(str).str.strip().str.lower()
            unassigned_mask = df[col].isnull() | s.isin(['', 'nan', 'none', 'null', 'unassigned', 'tbd', 'unknown', 'pending', 'n/a'])
            unassigned_count = int(unassigned_mask.sum())
            if unassigned_count > 0:
                unassigned_pct = round((unassigned_count / max(len(df), 1)) * 100, 1)
                issues.append({
                    "id": f"rule_{issue_counter}",
                    "type": "unassigned_owner",
                    "category": "OPERATIONAL_GAP",
                    "is_operational_gap": True,
                    "confirmation_status": "CONFIRMED",
                    "column": col,
                    "columns": [col],
                    "issue": f"{unassigned_count} operational record(s) in '{col}' have no designated owner ({unassigned_pct}% of total)",
                    "why_it_matters": "Actions or records without designated owners lack operational accountability and risk unaddressed non-conformances.",
                    "what_to_check": "Verify whether owners were assigned outside this recorded logging system or require immediate supervisor designation.",
                    "proposed_fix": f"Assign responsible operational owners for {unassigned_count} record(s) in '{col}'",
                    "treatment": f"Review and assign operational owner in '{col}'",
                    "reason": "Missing documentation of an owner requires operational verification to ensure plant follow-through.",
                    "count": unassigned_count,
                    "affected_records": unassigned_count,
                    "affected_percentage": unassigned_pct,
                    "impact": f"Identifies {unassigned_count} unassigned operational items for execution follow-through.",
                    "reversibility": True,
                    "status": "pending"
                })
                issue_counter += 1

    # 8. Operational Process Gaps: Open / In-Progress Workflows (Prompt Section 7)
    status_keywords = ['status', 'closure', 'resolution', 'state', 'stage']
    for col in df.columns:
        c_low = str(col).lower().strip()
        if any(k in c_low for k in status_keywords):
            s = df[col].astype(str).str.strip().str.lower()
            open_mask = s.isin(['open', 'pending', 'in progress', 'in-progress', 'active', 'reopened', 'unresolved', 'unclosed', 'ongoing'])
            open_count = int(open_mask.sum())
            if open_count > 0:
                open_pct = round((open_count / max(len(df), 1)) * 100, 1)
                issues.append({
                    "id": f"rule_{issue_counter}",
                    "type": "open_action",
                    "category": "OPERATIONAL_GAP",
                    "is_operational_gap": True,
                    "confirmation_status": "CONFIRMED",
                    "column": col,
                    "columns": [col],
                    "issue": f"{open_count} operational item(s) in '{col}' remain open / in-progress ({open_pct}% of total)",
                    "why_it_matters": "Unresolved workflow items represent ongoing operational deviations, open safety observations, or pending maintenance.",
                    "what_to_check": "Review open items against line maintenance or safety meeting logs to determine whether completion notes are pending.",
                    "proposed_fix": f"Track execution and verify closure status for {open_count} open item(s) in '{col}'",
                    "treatment": f"Monitor and follow up on {open_count} open workflow item(s)",
                    "reason": "Active operational items require continuous tracking against plant targets.",
                    "count": open_count,
                    "affected_records": open_count,
                    "affected_percentage": open_pct,
                    "impact": f"Highlights {open_count} active workflows requiring verification of completion.",
                    "reversibility": True,
                    "status": "pending"
                })
                issue_counter += 1

    # 9. Operational Inconsistencies between Related Columns (Prompt Section 7)
    status_cols = [c for c in df.columns if any(k in c.lower() for k in ['status', 'closure', 'state'])]
    date_cols = [c for c in df.columns if any(k in c.lower() for k in ['closure_date', 'resolved_date', 'completion_date', 'closed_date', 'closed_on'])]
    if status_cols and date_cols:
        s_col = status_cols[0]
        d_col = date_cols[0]
        s_series = df[s_col].astype(str).str.strip().str.lower()
        closed_mask = s_series.isin(['closed', 'resolved', 'completed', 'done'])
        missing_date_mask = closed_mask & df[d_col].isnull()
        mismatch_cnt = int(missing_date_mask.sum())
        if mismatch_cnt > 0:
            mismatch_pct = round((mismatch_cnt / max(len(df), 1)) * 100, 1)
            issues.append({
                "id": f"rule_{issue_counter}",
                "type": "field_inconsistency",
                "category": "OPERATIONAL_GAP",
                "is_operational_gap": True,
                "confirmation_status": "CONFIRMED",
                "column": f"{s_col} ~ {d_col}",
                "columns": [s_col, d_col],
                "issue": f"{mismatch_cnt} record(s) marked as '{s_col}' Closed/Resolved but lack '{d_col}' timestamp ({mismatch_pct}%)",
                "why_it_matters": "Missing closure timestamps on closed items prevent accurate cycle time and MTTR (Mean Time to Repair/Resolution) calculation.",
                "what_to_check": "Verify record audit trail to backfill true completion dates.",
                "proposed_fix": f"Reconcile {mismatch_cnt} records with missing closure timestamps in '{d_col}'",
                "treatment": f"Reconcile status with timestamp audit",
                "reason": "Inconsistencies between related fields distort turnaround performance metrics.",
                "count": mismatch_cnt,
                "affected_records": mismatch_cnt,
                "affected_percentage": mismatch_pct,
                "impact": f"Ensures valid duration and cycle-time calculation across {mismatch_cnt} records.",
                "reversibility": True,
                "status": "pending"
            })
            issue_counter += 1

    # 10. Temporal Coverage Discontinuity (Prompt Section 7)
    time_candidates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or any(k in c.lower() for k in ['date', 'timestamp'])]
    if time_candidates:
        t_col = time_candidates[0]
        try:
            parsed_ts = pd.to_datetime(df[t_col], errors='coerce').dropna().sort_values()
            if len(parsed_ts) >= 10:
                diffs = parsed_ts.diff().dropna()
                median_step = diffs.median()
                if median_step > pd.Timedelta(seconds=0):
                    large_gaps = diffs[diffs > (median_step * 3)]
                    gap_count = len(large_gaps)
                    if gap_count > 0:
                        issues.append({
                            "id": f"rule_{issue_counter}",
                            "type": "temporal_discontinuity",
                            "category": "DATA_QUALITY",
                            "is_operational_gap": False,
                            "confirmation_status": "CONFIRMED",
                            "column": t_col,
                            "columns": [t_col],
                            "issue": f"{gap_count} abnormal calendar gap(s) detected in '{t_col}' exceeding 3x expected interval ({median_step})",
                            "why_it_matters": "Discontinuous temporal coverage may conceal seasonal trends, unrecorded shifts, or unplanned downtime periods.",
                            "what_to_check": "Verify whether gaps correspond to planned plant maintenance outages or missing batch data exports.",
                            "proposed_fix": f"Audit {gap_count} time-series gap(s) against plant operating calendar",
                            "treatment": "Retain observed data without artificial interpolation and log audit warning",
                            "reason": "Unexplained time gaps can distort trend direction and period-over-period growth rates.",
                            "count": gap_count,
                            "affected_records": gap_count,
                            "affected_percentage": round((gap_count / max(len(parsed_ts), 1)) * 100, 1),
                            "impact": f"Flags calendar discontinuities to prevent misleading time-series conclusions.",
                            "reversibility": True,
                            "status": "pending"
                        })
                        issue_counter += 1
        except Exception:
            pass

    # Ensure all issues have standard fields populated
    for iss in issues:
        if "category" not in iss:
            iss["category"] = "DATA_QUALITY"
        if "is_operational_gap" not in iss:
            iss["is_operational_gap"] = (iss["category"] == "OPERATIONAL_GAP")
        if "confirmation_status" not in iss:
            iss["confirmation_status"] = "CONFIRMED"
        if "why_it_matters" not in iss:
            iss["why_it_matters"] = iss.get("reason", "Ensure data validity")
        if "what_to_check" not in iss:
            iss["what_to_check"] = iss.get("proposed_fix", "Check affected records")
        if "affected_percentage" not in iss:
            cnt = iss.get("affected_records", iss.get("count", 0))
            iss["affected_percentage"] = round((cnt / max(len(df), 1)) * 100, 2)

    return issues

def apply_cleaning_rules(df: pd.DataFrame, approved_rules: list):
    """
    Applies approved rules to a DataFrame in a reproducible sequence.
    Returns cleaned DataFrame and Audit Trail step history with explicit Quality Ledger fields.
    """
    cleaned_df = df.copy()
    audit_trail = []

    for idx, rule in enumerate(approved_rules, start=1):
        rule_type = rule.get("type")
        col = rule.get("column")
        cols = rule.get("columns", [col] if col != "ALL" else list(cleaned_df.columns))
        affected = rule.get("affected_records", rule.get("count", 0))
        issue_text = rule.get("issue", f"Quality issue in {col}")
        treatment_text = rule.get("treatment", rule.get("proposed_fix", f"Apply {rule_type}"))
        reason_text = rule.get("reason", "Ensure data validity")
        impact_text = rule.get("impact", f"Modified {affected} record(s)")
        
        before_rows = len(cleaned_df)

        if rule_type == "duplicate_rows":
            cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
            action_desc = f"Removed {before_rows - len(cleaned_df)} duplicate row(s)."

        elif rule_type == "coerce_numeric" and col in cleaned_df.columns:
            series_str = (cleaned_df[col].astype(str)
                          .str.replace('$', '', regex=False)
                          .str.replace('₹', '', regex=False)
                          .str.replace('€', '', regex=False)
                          .str.replace('£', '', regex=False)
                          .str.replace(',', '', regex=False)
                          .str.replace('%', '', regex=False)
                          .str.strip())
            cleaned_df[col] = pd.to_numeric(series_str, errors='coerce')
            action_desc = f"Coerced formatted text column '{col}' to numeric float/int data type."

        elif rule_type == "fix_invalid_numeric" and col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].apply(lambda x: abs(x) if pd.notnull(x) and x < 0 else x)
            action_desc = f"Fixed negative values in column '{col}' by taking absolute value."

        elif rule_type == "trim_whitespace" and col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
            action_desc = f"Trimmed whitespace for column '{col}' across {affected} record(s)."

        elif rule_type == "standardize_casing" and col in cleaned_df.columns:
            orig_v = rule.get("original_value")
            norm_v = rule.get("normalized_value")
            if orig_v is not None and norm_v is not None:
                cleaned_df[col] = cleaned_df[col].astype(str).replace(str(orig_v), str(norm_v))
                action_desc = f"Standardized casing for '{col}': normalized '{orig_v}' to '{norm_v}' across {affected} record(s)."
            else:
                cleaned_df[col] = cleaned_df[col].astype(str).str.title().str.strip()
                action_desc = f"Standardized casing to Title Case for column '{col}'."

        elif rule_type == "missing_values" and col in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                fill_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(fill_val)
                action_desc = f"Imputed {affected} missing numeric values in '{col}' with median ({fill_val})."
            else:
                cleaned_df[col] = cleaned_df[col].fillna("Unknown")
                action_desc = f"Filled {affected} missing string values in '{col}' with 'Unknown'."

        elif rule_type == "outlier_flag" and col in cleaned_df.columns:
            series = pd.to_numeric(cleaned_df[col], errors='coerce').astype(float).dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            upper_b = q3 + 1.5 * iqr
            lower_b = q1 - 1.5 * iqr
            cleaned_df[f"{col}_IsOutlier"] = (cleaned_df[col] < lower_b) | (cleaned_df[col] > upper_b)
            action_desc = f"Added binary indicator '{col}_IsOutlier' for {affected} record(s)."

        else:
            action_desc = f"Applied rule: {rule_type} on column {col}."

        audit_trail.append({
            "step": idx,
            "rule_id": rule.get("id"),
            "rule_type": rule_type,
            "column": col,
            "columns": cols,
            "original_value": rule.get("original_value", "N/A"),
            "normalized_value": rule.get("normalized_value", "N/A"),
            "transformation": rule.get("transformation", treatment_text),
            "reason": reason_text,
            "affected_records": affected,
            "count": affected,
            "records_affected": affected,
            "issue": issue_text,
            "treatment": treatment_text,
            "impact": impact_text,
            "reversibility": True,
            "description": action_desc,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    return cleaned_df, audit_trail


def calculate_data_health_score_breakdown(df: pd.DataFrame) -> dict:
    """
    Calculates the single source of truth canonical DatasetHealthResult.
    All modules (Overview, Quality, EDA, Insights, Recs, Actions, Exports) consume this.
    """
    from engine.governed_engines import calculate_canonical_dataset_health
    return calculate_canonical_dataset_health(df)
