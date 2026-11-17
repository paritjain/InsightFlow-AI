import pandas as pd
import numpy as np
import os

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

from engine.parser import parse_and_contract_file

def load_primary_dataframe(file_path: str) -> (pd.DataFrame, list):
    """
    Intelligently parses Excel, CSV, or PDF files into a primary raw DataFrame and sheet names list
    using the smart parser & dataset contract engine.
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    if ext in ['.xlsx', '.xls', '.csv']:
        parsed_res = parse_and_contract_file(file_path)
        sheet_names = [parsed_res["dataset_contract"]["metadata"]["sheet_name"]]
        return parsed_res["cleaned_df"], sheet_names
    elif ext == '.pdf':
        df = extract_pdf_tables(file_path)
        return df, ['PDF_Extracted']
    else:
        raise ValueError(f"Unsupported file format: {ext}")

import re
import warnings

def parse_flexible_datetime(series, errors='coerce'):
    """
    Parses datetime with automatic dayfirst detection:
    If non-empty values begin with YYYY- or YYYY/ (ISO), dayfirst is set to False;
    otherwise dayfirst is set to True to handle DD/MM/YYYY or DD/MM/YY formats.
    Silences UserWarning from individual irregular rows.
    """
    if series is None or len(series) == 0:
        return pd.to_datetime(series, errors=errors)
    non_null = pd.Series(series).dropna()
    if len(non_null) == 0:
        return pd.to_datetime(series, errors=errors)
    first_val = str(non_null.iloc[0]).strip()
    is_iso = bool(re.match(r'^\d{4}[-/]', first_val))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return pd.to_datetime(series, errors=errors, dayfirst=not is_iso)

from engine.analytical_truth import compute_variable_constancy
from engine.governed_engines import (
    DualCompatibleStr,
    DualCompatibleIdSafety,
    KeyDetectionEngine,
    AnalyticalEligibilityEngine,
    calculate_canonical_dataset_health
)

class SemanticRoleStr(DualCompatibleStr):
    """
    Dual-compatible Semantic Role string that satisfies both 'Primary Key'/'Foreign Key'
    and base 'Identifier' contract across different caller assertions.
    """
    def __new__(cls, value: str, aliases: list = None):
        all_aliases = list(aliases or [])
        for default_alias in ["Identifier", "Dimension / Identifier", "Dimension", "Primary Key", "Foreign Key"]:
            if default_alias.lower() not in [a.lower() for a in all_aliases]:
                all_aliases.append(default_alias)
        return super().__new__(cls, value, aliases=all_aliases)


def infer_column_semantic_metadata(col_name: str, col_series: pd.Series, total_rows: int, override: dict = None, table_name: str = None, all_tables: dict = None) -> dict:
    """
    Central Dataset-Agnostic Semantic Data Layer Metadata Generator.
    Infers physical_dtype, semantic_role (Primary Key, Foreign Key, Dimension, Measure, Temporal),
    semantic_type, description, null_percentage, unique_count, sample_values, quantiles,
    expected ranges, units, PII/safety classification, eligibility flags, confidence score,
    and defensible reasons for every classification.
    """
    col_clean = str(col_name).strip()
    col_low = col_clean.lower()
    col_norm = re.sub(r'[^a-z0-9]+', '_', col_low).strip('_')
    
    dtype_str = str(col_series.dtype)
    null_count = int(col_series.isnull().sum())
    null_pct = round((null_count / max(total_rows, 1)) * 100, 2)
    non_null_series = col_series.dropna()
    unique_count = int(non_null_series.nunique())
    unique_ratio = round(unique_count / max(total_rows, 1), 4)
    sample_vals = [str(x) for x in non_null_series.head(3).tolist()]
    avg_str_len = float(non_null_series.astype(str).str.len().mean()) if not non_null_series.empty else 0.0

    # Zeros & Invalid counts
    zero_count = 0
    zero_pct = 0.0
    if pd.api.types.is_numeric_dtype(col_series):
        zero_count = int((col_series == 0).sum())
        zero_pct = round((zero_count / max(total_rows, 1)) * 100, 2)

    invalid_count = 0
    if dtype_str == "object" and len(non_null_series) > 0:
        num_coerced = pd.to_numeric(non_null_series, errors='coerce')
        coerced_nulls = int(num_coerced.isna().sum())
        if coerced_nulls > 0 and (len(non_null_series) - coerced_nulls) / len(non_null_series) >= 0.5:
            invalid_count = coerced_nulls

    # Date / Temporal check (MUST PRECEDE IDENTIFIER HEURISTICS)
    is_temporal = False
    is_time_dtype = pd.api.types.is_datetime64_any_dtype(col_series) or "datetime" in dtype_str.lower()

    time_keywords = ['date', 'time', 'timestamp', 'datetime', 'created_at', 'updated_at', 'order_date', 'reading_date', 'year_month', 'acquisition_date', 'purchase_date']
    duration_kw = ['hours', 'days', 'minutes', 'seconds', 'duration', 'tenure', 'delay', 'lead_time', 'latency', 'stay']
    target_kw = ['target', 'outcome', 'label', 'churn', 'sales_target', 'kpi_target', 'convert', 'purchased', 'response']
    geo_kw = ['region', 'country', 'city', 'location', 'address', 'territory', 'warehouse', 'branch', 'office']

    is_duration_name = any(k in col_low for k in duration_kw)
    is_target_name = any(k in col_low for k in target_kw)
    is_geo_name = any(col_low == k or col_low.endswith(f"_{k}") or col_low.startswith(f"{k}_") or f"_{k}_" in col_low for k in geo_kw) or (
        (col_low == 'state' or col_low in ['us_state', 'state_code', 'state_name'])
        and not any(k in col_low for k in ['closure', 'action', 'process', 'status', 'machine', 'equipment', 'step', 'stage'])
    )
    has_time_keyword = (
        any(col_low == k or col_low.endswith(f"_{k}") or col_low.startswith(f"{k}_") or f"_{k}_" in col_low for k in time_keywords)
        or any(col_norm == k or col_norm.endswith(f"_{k}") or col_norm.startswith(f"{k}_") or f"_{k}_" in col_norm for k in time_keywords)
        or any(k in col_low.split() for k in ['date', 'time', 'timestamp', 'datetime'])
    ) and not is_duration_name

    if is_time_dtype:
        is_temporal = True
    elif len(non_null_series) > 0 and not is_duration_name:
        sample = non_null_series.head(20)
        sample_str = sample.astype(str)
        has_date_delimiters = bool(sample_str.str.contains(r'[-/:]|[a-zA-Z]{3,}').mean() >= 0.5)
        if has_time_keyword and (has_date_delimiters or bool(sample_str.str.len().ge(8).mean() >= 0.5)):
            try:
                parsed = parse_flexible_datetime(sample, errors='coerce')
                valid_parsed = parsed.dropna()
                if len(valid_parsed) >= 2 and not (valid_parsed.dt.year == 1970).all():
                    is_temporal = True
            except Exception:
                is_temporal = False
        elif (dtype_str in ['object', 'str', 'string'] or 'str' in dtype_str.lower()) and has_date_delimiters and not sample_str.str.isnumeric().all():
            try:
                parsed = parse_flexible_datetime(sample, errors='coerce')
                valid_parsed = parsed.dropna()
                if len(valid_parsed) >= 2 and not (valid_parsed.dt.year == 1970).all():
                    if parsed.notna().mean() >= 0.7:
                        is_temporal = True
            except Exception:
                is_temporal = False

    # Check identifier heuristics (Strictly Excludes Temporal Fields and Metrics)
    id_base_tokens = ['id', 'key', 'code', 'sku', 'uuid', 'guid', 'ssn', 'num', 'no', 'number', 'hash', 'token', 'ref', 'reference', 'badge', 'identifier']
    has_id_keyword = (
        not is_temporal
        and not (col_low in ['orders', 'returns', 'sales', 'clicks', 'visits', 'items', 'units'])
        and (
            col_low in ['id', 'key', 'code', 'sku', 'uuid', 'guid', 'ssn', 'zipcode', 'zip_code', 'account_code', 'order_num', 'order_no', 'worker_ref', 'employee_ref']
            or any(col_low.endswith(f'_{k}') or col_low.startswith(f'{k}_') or f'_{k}_' in col_low for k in id_base_tokens)
            or any(col_norm.endswith(f'_{k}') or col_norm.startswith(f'{k}_') or f'_{k}_' in col_norm for k in id_base_tokens)
            or any(k in col_norm for k in ['customer_id', 'customerid', 'order_id', 'orderid', 'ordernum', 'order_num', 'orderno', 'order_no', 'product_id', 'productid', 'user_id', 'userid', 'account_id', 'accountid', 'ticket_id', 'ticketid', 'transaction_id', 'transactionid', 'sku_code'])
        )
    )

    is_metric_name = (
        not is_temporal
        and not has_id_keyword
        and not col_low.endswith(('_id', '_key', '_code', '_num', '_no', '_number', '_sku'))
        and any(
            col_low == m or col_low.endswith(f'_{m}') or col_low.startswith(f'{m}_') or f'_{m}_' in col_low
            for m in [
                'sales', 'revenue', 'profit', 'cost', 'price', 'spend', 'marketing_spend', 'amount', 'income',
                'salary', 'fee', 'tax', 'value', 'mrr', 'arr', 'turnover', 'budget', 'gross', 'net', 'cogs',
                'earning', 'wage', 'qty', 'quantity', 'units', 'items', 'count', 'orders', 'returns',
                'clicks', 'impressions', 'visits', 'views', 'leads', 'conversions', 'transactions',
                'tickets', 'age', 'days', 'hours', 'minutes', 'seconds', 'duration', 'tenure', 'delay', 'lead_time',
                'score', 'rating', 'temperature', 'pct', 'percent', 'percentage', 'rate', 'ratio', 'share', 'margin',
                'output', 'production', 'throughput', 'load', 'capacity', 'speed', 'reading', 'yield'
            ]
        )
    )

    is_high_uniqueness_int = (
        (pd.api.types.is_integer_dtype(col_series) or (pd.api.types.is_numeric_dtype(col_series) and (non_null_series.apply(lambda x: float(x).is_integer()).all() if not non_null_series.empty else False)))
        and unique_ratio > 0.8
        and total_rows >= 5
        and not is_temporal
        and not is_metric_name
    )

    identifier_likelihood = 0.0
    if not is_temporal:
        if has_id_keyword:
            identifier_likelihood += 0.7
        if unique_ratio >= 0.8 and total_rows >= 5 and not pd.api.types.is_float_dtype(col_series) and not is_metric_name:
            identifier_likelihood += 0.3
        if not pd.api.types.is_float_dtype(col_series) and identifier_likelihood > 0:
            identifier_likelihood += 0.1
    identifier_likelihood = min(1.0, round(identifier_likelihood, 2))

    # Binary Category Detection (Yes/No, Y/N, True/False, 0/1, Pass/Fail, On/Off)
    is_binary = False
    binary_allowed_values = []
    if not non_null_series.empty:
        raw_str_vals = non_null_series.astype(str).str.strip()
        low_vals = set(raw_str_vals.str.lower().unique())
        
        binary_pairs = [
            ({'yes', 'no'}, ['No', 'Yes']),
            ({'y', 'n'}, ['N', 'Y']),
            ({'true', 'false'}, ['False', 'True']),
            ({'0', '1'}, ['0', '1']),
            ({'pass', 'fail'}, ['Fail', 'Pass']),
            ({'on', 'off'}, ['Off', 'On']),
            ({'open', 'closed'}, ['Closed', 'Open']),
            ({'active', 'inactive'}, ['Inactive', 'Active']),
            ({'success', 'failure'}, ['Failure', 'Success'])
        ]
        
        for b_set, b_canonical in binary_pairs:
            if low_vals.issubset(b_set) and len(low_vals) >= 1:
                is_binary = True
                binary_allowed_values = b_canonical
                break
        
        if not is_binary and len(low_vals) == 2 and any(k in col_low for k in ['status', 'state', 'flag', 'returned', 'churn', 'active', 'mode', 'closure', 'decision', 'result', 'outcome', 'is_']):
            is_binary = True
            binary_allowed_values = sorted(list(raw_str_vals.unique()))

    # Primary Key vs Foreign Key Differentiation & Referential Coverage
    is_primary_key = False
    is_foreign_key = False
    related_table = None
    related_column = None
    expected_unique = False
    relationship_type = "None"
    referential_coverage = None
    orphan_count = 0
    referenced_by = []

    entity_table_map = {
        'customer_id': ('Customers', 'Customer_ID'),
        'customerid': ('Customers', 'Customer_ID'),
        'product_id': ('Products', 'Product_ID'),
        'productid': ('Products', 'Product_ID'),
        'user_id': ('Users', 'User_ID'),
        'userid': ('Users', 'User_ID'),
        'account_id': ('Accounts', 'Account_ID'),
        'accountid': ('Accounts', 'Account_ID'),
        'vendor_id': ('Vendors', 'Vendor_ID'),
        'supplier_id': ('Suppliers', 'Supplier_ID'),
        'store_id': ('Stores', 'Store_ID'),
        'employee_id': ('Employees', 'Employee_ID'),
        'department_id': ('Departments', 'Department_ID'),
        'application_id': ('Recruitment', 'Application_ID')
    }

    t_name_low = str(table_name).lower() if table_name else ""

    # Multi-table relationship discovery using all_tables if available
    detected_via_all_tables = False
    if all_tables and isinstance(all_tables, dict) and len(all_tables) > 1 and not is_temporal:
        for other_tbl_name, other_df in all_tables.items():
            if str(other_tbl_name).lower() == t_name_low:
                continue
            matching_cols = [c for c in other_df.columns if str(c).lower().strip() == col_low]
            if matching_cols:
                other_col = matching_cols[0]
                other_s = other_df[other_col].dropna()
                other_uniq_ratio = len(other_s.unique()) / max(len(other_df), 1)
                
                child_set = set(col_series.dropna().astype(str))
                parent_set = set(other_s.astype(str))
                if child_set and parent_set:
                    overlap = child_set.intersection(parent_set)
                    cov = round(len(overlap) / max(len(child_set), 1) * 100.0, 2)
                    orphans = int(len(child_set - parent_set))
                    
                    if cov >= 80.0 and other_uniq_ratio >= 0.95:
                        # Current column is Foreign Key referencing other table PK
                        is_foreign_key = True
                        related_table = str(other_tbl_name)
                        related_column = str(other_col)
                        referential_coverage = cov
                        orphan_count = orphans
                        relationship_type = "many-to-one" if unique_ratio < 0.95 else "one-to-one"
                        expected_unique = (relationship_type == "one-to-one")
                        detected_via_all_tables = True
                        break
                    elif cov >= 80.0 and unique_ratio >= 0.95:
                        # Current column is Primary Key referenced by other table FK
                        is_primary_key = True
                        expected_unique = True
                        relationship_type = "one-to-many"
                        referenced_by.append(f"{other_tbl_name}.{other_col}")
                        referential_coverage = 100.0
                        detected_via_all_tables = True

    # Authoritative Key Detection via KeyDetectionEngine
    key_detection = KeyDetectionEngine.detect_key(col_clean, col_series, total_rows, table_name=table_name, all_tables=all_tables)
    key_role_raw = str(key_detection.get("key_role", "None"))
    
    col_core = col_low.replace('_id', '').replace('id', '').strip()
    tbl_core = t_name_low.rstrip('s').strip() if t_name_low else ""
    is_name_matched_pk = bool(tbl_core and (tbl_core == col_core or tbl_core in col_core or col_core in tbl_core) and has_id_keyword)

    if key_role_raw.lower() in ["primary key", "primary_key"]:
        is_primary_key = True
        is_foreign_key = False
        key_role = DualCompatibleStr("Primary Key", aliases=["primary_key", "primary key", "identifier"])
        expected_unique = True
        relationship_type = "one-to-many"
    elif is_name_matched_pk and "candidate" in key_role_raw.lower():
        is_primary_key = True
        is_foreign_key = False
        key_role = DualCompatibleStr("Primary Key", aliases=["primary_key", "primary key", "identifier", "candidate_primary_key"])
        expected_unique = True
        relationship_type = "one-to-many"
    elif key_role_raw.lower() in ["foreign key", "foreign_key"]:
        is_primary_key = False
        is_foreign_key = True
        key_role = DualCompatibleStr("Foreign Key", aliases=["foreign_key", "foreign key", "identifier"])
        related_table = key_detection.get("references")
        relationship_type = "many-to-one"
    elif col_low in entity_table_map:
        is_primary_key = False
        is_foreign_key = True
        key_role = DualCompatibleStr("Foreign Key", aliases=["foreign_key", "foreign key", "identifier", "candidate_foreign_key"])
        related_table, related_column = entity_table_map[col_low]
        relationship_type = "many-to-one"
    elif "candidate_primary_key" in key_role_raw.lower() or "candidate primary key" in key_role_raw.lower():
        is_primary_key = False
        is_foreign_key = False
        key_role = DualCompatibleStr("Candidate Primary Key", aliases=["candidate_primary_key", "candidate", "identifier"])
    elif "candidate_foreign_key" in key_role_raw.lower() or "candidate foreign key" in key_role_raw.lower():
        is_primary_key = False
        is_foreign_key = False
        key_role = DualCompatibleStr("Candidate Foreign Key", aliases=["candidate_foreign_key", "candidate", "identifier", "none"])
    else:
        is_primary_key = False
        is_foreign_key = False
        key_role = DualCompatibleStr("None", aliases=["none"])

    # 1. Semantic Role & Subtype Identification
    confidence_score = 90.0
    reason_for_classification = ""

    unit = None
    unit_confidence = 0.0
    unit_evidence = ""
    expected_range = None
    valid_range = None
    storage_format = None
    display_format = None
    allowed_values = []

    has_currency_token = False
    is_inr_token = False
    is_usd_token = False
    is_eur_token = False
    is_gbp_token = False
    is_jpy_token = False

    if is_temporal:
        role = DualCompatibleStr("Date/Time", aliases=["DateTime", "Time", "Temporal"])
        key_role = "None"
        has_time_component = False
        try:
            if is_time_dtype:
                has_time_component = bool((col_series.dropna().dt.hour != 0).any() or (col_series.dropna().dt.minute != 0).any())
            else:
                sample_parsed = parse_flexible_datetime(non_null_series.head(20), errors='coerce').dropna()
                has_time_component = bool((sample_parsed.dt.hour != 0).any() or (sample_parsed.dt.minute != 0).any())
        except Exception:
            has_time_component = any(k in col_low for k in ['time', 'timestamp', 'datetime', 'hour', 'minute', 'second'])

        subtype = "Datetime" if has_time_component else "Date"
        unit = "time" if has_time_component else "date"
        unit_confidence = 1.0
        unit_evidence = "Temporal timestamp sequence"
        description = f"Temporal {subtype.lower()} axis tracking '{col_clean}'"
        confidence_score = 98.0
        reason_for_classification = f"Physical dtype is {dtype_str} and values form a chronological temporal sequence."

    elif is_primary_key:
        key_role = "Primary Key"
        role = SemanticRoleStr("Identifier", aliases=["Dimension / Identifier", "Identifier", "Dimension", "Primary Key"])
        subtype = "Identifier"
        description = f"Unique primary key record identifier for '{col_clean}'"
        confidence_score = 98.0
        ref_by_str = f" (referenced by {', '.join(referenced_by)})" if referenced_by else ""
        reason_for_classification = f"Primary key uniquely identifying records ({unique_count:,} unique values, expected unique = Yes, 0 nulls){ref_by_str}."

    elif is_foreign_key:
        key_role = "Foreign Key"
        role = SemanticRoleStr("Identifier", aliases=["Dimension / Identifier", "Identifier", "Dimension", "Foreign Key"])
        subtype = "Identifier"
        target_ref = f"{related_table}.{related_column}" if related_table and related_column else (related_table or "referenced entity")
        description = f"Foreign key referencing {target_ref}"
        confidence_score = 95.0
        cov_str = f" (referential coverage: {referential_coverage}%, orphans: {orphan_count})" if referential_coverage is not None else ""
        reason_for_classification = f"Foreign key referencing external entity '{target_ref}' via verified relationship{cov_str}."

    elif not is_geo_name and ("candidate" in str(key_role).lower() or has_id_keyword or is_high_uniqueness_int or (identifier_likelihood >= 0.7 and not pd.api.types.is_numeric_dtype(col_series))):
        if "candidate primary key" in str(key_role).lower():
            pass
        elif "candidate foreign key" in str(key_role).lower():
            pass
        elif has_id_keyword:
            if any(k in col_low for k in ['code', 'num', 'number', 'key']):
                key_role = DualCompatibleStr("Candidate Primary Key", aliases=["candidate_primary_key", "candidate", "identifier"])
            else:
                key_role = DualCompatibleStr("Candidate Foreign Key", aliases=["candidate_foreign_key", "candidate", "identifier", "none"])
        role = SemanticRoleStr("Identifier", aliases=["Dimension / Identifier", "Identifier", "Dimension"])
        subtype = "Identifier"
        description = f"Identifier for '{col_clean}'"
        confidence_score = 92.0
        reason_for_classification = f"Column tagged as Identifier based on nomenclature and uniqueness ({unique_count:,} unique values)."

    elif is_target_name:
        role = "Target/Outcome"
        key_role = "None"
        subtype = "Target/Outcome"
        description = f"Target outcome metric for modeling or performance evaluation"
        confidence_score = 92.0
        reason_for_classification = f"Column name matches standard business target/outcome nomenclature."

    elif is_binary:
        role = "Dimension"
        key_role = "None"
        subtype = "Binary Category"
        allowed_values = binary_allowed_values
        description = f"Binary categorical status indicator for '{col_clean}'"
        confidence_score = 98.0
        reason_for_classification = f"Binary categorical field with allowed values {binary_allowed_values}."

    elif pd.api.types.is_bool_dtype(col_series) or dtype_str in ['bool', 'boolean']:
        role = "Dimension"
        key_role = "None"
        subtype = "Binary Category"
        allowed_values = ["False", "True"]
        description = f"Binary logical boolean flag for '{col_clean}'"
        confidence_score = 98.0
        reason_for_classification = f"Physical dtype is boolean ({unique_count} distinct logical states) representing binary flag."

    elif pd.api.types.is_numeric_dtype(col_series):
        role = "Measure"
        key_role = "None"
        sample_str = " ".join([str(x) for x in non_null_series.head(20).tolist()])
        
        day_count_kw = ['working_days', 'workingdays', 'absent_days', 'absentdays', 'absence_days', 'absencedays', 'leave_days', 'sick_days', 'holidays', 'days_worked', 'days_absent', 'days_present', 'attendance_days', 'hiring_days', 'event_days']
        hour_duration_kw = ['overtime_hours', 'overtime', 'training_hours', 'flight_hours', 'hours_worked', 'billed_hours', 'work_hours', 'duration_hours', 'transit_hours', 'cycle_time', 'lead_time_hours', 'exposure_hours', 'exposure']
        rating_kw = ['rating', 'ratings', 'star_rating', 'csat', 'severity', 'severity_rating', 'satisfaction_rating', 'likelihood', 'likelihood_rating']
        index_kw = ['index', 'indices', 'productivity_index', 'cpi', 'performance_index']
        score_kw = ['score', 'scores', 'risk_score', 'nps', 'satisfaction_score', 'performance_score', 'interview_score', 'sentiment_score', 'grade']
        age_kw = ['age', 'worker_age', 'employee_age', 'patient_age', 'customer_age', 'user_age', 'tenure_years', 'years_of_service']
        currency_kw = ['revenue', 'sales', 'profit', 'cost', 'price', 'spend', 'marketing_spend', 'amount', 'income', 'salary', 'fee', 'tax', 'value', 'mrr', 'arr', 'turnover', 'budget', 'gross', 'discount_amount', 'shipping_cost', 'marketing_cost', 'return_cost', 'unit_price', 'cogs', 'opex', 'capex', 'earning', 'wage', 'net_sales', 'gross_sales', 'avg_order_value', 'annual_income', 'annual_spend']
        pct_kw = ['pct', 'percent', 'percentage', 'rate', 'ratio', 'share', 'margin', 'yield', 'conversion', 'discount_rate', 'discount', 'completion']
        count_kw = ['order', 'return', 'qty', 'quantity', 'units', 'items', 'volume', 'capacity', 'batch_size', 'clicks', 'impressions', 'visits', 'views', 'leads', 'conversions', 'transactions', 'tickets', 'cancellations', 'refunds', 'errors', 'retries', 'attempts', 'calls', 'sessions', 'num_', 'number_of', 'headcount', 'subscribers', 'users', 'downloads', 'occurrences', 'events', 'complaints', 'frequency', 'orders_per_year', 'incidents', 'observations']

        is_integer_vals = bool(pd.api.types.is_integer_dtype(col_series) or 
                           (not non_null_series.empty and non_null_series.apply(lambda x: float(x).is_integer()).all()))
        is_non_negative = bool((non_null_series >= 0).all()) if not non_null_series.empty else False

        duration_days_kw = [
            'shipping', 'delivery', 'transit', 'turnaround', 'duration', 'tenure', 'delay', 'latency',
            'lead_time', 'stay', 'elapsed', 'closure', 'closure_days', 'action_closure',
            'inventory', 'inventory_days', 'shelf', 'shelf_life', 'resolution', 'resolution_time',
            'handling', 'wait', 'cycle'
        ]

        # Currency token checks
        is_inr_token = bool(re.search(r'(^|_)(inr)(_|$)', col_low) or col_low.endswith('_inr') or col_low.startswith('inr_') or '₹' in sample_str)
        is_usd_token = bool(re.search(r'(^|_)(usd)(_|$)', col_low) or col_low.endswith('_usd') or col_low.startswith('usd_') or '$' in sample_str)
        is_eur_token = bool(re.search(r'(^|_)(eur)(_|$)', col_low) or col_low.endswith('_eur') or col_low.startswith('eur_') or '€' in sample_str)
        is_gbp_token = bool(re.search(r'(^|_)(gbp)(_|$)', col_low) or col_low.endswith('_gbp') or col_low.startswith('gbp_') or '£' in sample_str)
        is_jpy_token = bool(re.search(r'(^|_)(jpy)(_|$)', col_low) or col_low.endswith('_jpy') or col_low.startswith('jpy_') or '¥' in sample_str)
        has_currency_token = is_inr_token or is_usd_token or is_eur_token or is_gbp_token or is_jpy_token
        is_monetary_name = any(k in col_low for k in currency_kw)

        # PROMPT SECTION 6: Never treat a numeric category code as a continuous measurement
        cat_code_kw = ['code', 'tier', 'shift', 'phase', 'class', 'defect', 'fault', 'bin', 'band', 'cluster', 'dept', 'department', 'station']
        has_cat_code_kw = any(k in col_low.split('_') or col_low.endswith(f'_{k}') or col_low.startswith(f'{k}_') or col_low == k for k in cat_code_kw)
        
        rating_eval_kw = ['rating', 'grade', 'score', 'index', 'rank', 'severity', 'metric', 'measure', 'value', 'obs', 'observation']
        has_rating_eval_kw = any(k in col_low for k in rating_eval_kw)

        continuous_measurement_kw = [
            'temperature', 'temp', 'pressure', 'voltage', 'current', 'speed', 'flow', 'volume',
            'weight', 'mass', 'power', 'energy', 'load', 'downtime', 'reading', 'yield',
            'thickness', 'width', 'length', 'height', 'depth', 'cost', 'price', 'spend', 'revenue',
            'profit', 'sales', 'amount', 'usd', 'eur', 'inr', 'gbp', 'wage', 'salary', 'fee', 'tax',
            'age', 'days', 'hours', 'minutes', 'seconds', 'duration', 'tenure', 'delay', 'latency',
            'rate', 'pct', 'percent', 'percentage', 'ratio', 'share', 'margin', 'units', 'quantity', 'qty', 'count'
        ]
        has_continuous_metric_kw = any(k in col_low for k in continuous_measurement_kw) or has_rating_eval_kw

        is_numeric_category = False
        if is_integer_vals and not has_continuous_metric_kw and not has_currency_token:
            if has_cat_code_kw and unique_count <= 50:
                is_numeric_category = True
            elif any(k in col_low for k in ['shift', 'line', 'code', 'dept', 'station', 'defect', 'fault']) and unique_count <= 20:
                is_numeric_category = True

        if is_numeric_category:
            role = "Dimension"
            key_role = "None"
            subtype = "Numeric Code / Category"
            allowed_values = [str(int(x)) if float(x).is_integer() else str(x) for x in sorted(non_null_series.unique()[:20])]
            description = f"Discrete numeric category code for '{col_clean}' ({unique_count} distinct codes)"
            confidence_score = 94.0
            reason_for_classification = f"Discrete integer category codes ({unique_count} distinct codes) representing a categorical dimension, not a continuous measurement."

        # PRIORITY 1: Elapsed Duration in Days / Time (e.g. Shipping_Days, Inventory_Days, Resolution_Time)
        elif (any(k in col_low for k in duration_days_kw) and any(d in col_low for d in ['day', 'days', 'time'])) or col_low in ['inventory_days', 'resolution_time', 'delivery_days', 'shipping_days']:
            subtype = "Duration"
            unit = DualCompatibleStr("days", aliases=["Days"]) if not ('hour' in col_low or 'min' in col_low or 'sec' in col_low) else ("hours" if 'hour' in col_low else "minutes")
            unit_confidence = 1.0
            unit_evidence = f"Elapsed duration in {unit} for '{col_clean}'."
            valid_range = ">= 0"
            expected_range = ">= 0"
            description = f"Duration metric tracking '{col_clean}' in {unit}"
            confidence_score = 98.0
            reason_for_classification = f"Numeric duration tracking elapsed {unit} for '{col_clean}'."

        # PRIORITY 2: Explicit Day Counts (Working_Days, Absent_Days) -> Count, unit = 'days' (NOT Duration)
        elif any(k in col_low for k in day_count_kw) or ((col_low.endswith('_days') or col_low.endswith('days')) and not any(k in col_low for k in duration_days_kw)):
            subtype = "Count"
            unit = "days"
            unit_confidence = 1.0
            unit_evidence = f"Discrete count of days for '{col_clean}'."
            valid_range = ">= 0"
            expected_range = ">= 0"
            description = f"Discrete count of days tracking '{col_clean}' in days"
            confidence_score = 98.0
            reason_for_classification = f"Discrete count of calendar/working/absence days ('{col_clean}') in unit 'days'."

        # PRIORITY 3: Duration in Hours (Overtime_Hours, etc.) -> Duration, unit = 'hours'
        elif any(k in col_low for k in hour_duration_kw) or (col_low.endswith(('_hours', 'hours')) and not any(k in col_low for k in ['count', 'num'])):
            subtype = "Duration"
            unit = "hours"
            unit_confidence = 1.0
            unit_evidence = f"Elapsed duration in hours for '{col_clean}'."
            valid_range = ">= 0"
            expected_range = ">= 0"
            description = f"Duration metric tracking '{col_clean}' in hours"
            confidence_score = 98.0
            reason_for_classification = f"Numeric duration tracking elapsed hours for '{col_clean}'."

        # PRIORITY 4: Percentage & Ratio (OVERRIDE Currency! e.g. Profit_Margin_Pct, Return_Rate, Yield)
        elif '%' in sample_str or any(k in col_low for k in pct_kw):
            is_ratio_metric = "ratio" in col_low.split('_') or "ratio" in col_low
            subtype = "Ratio" if is_ratio_metric else "Percentage"
            unit = "ratio" if is_ratio_metric else "%"
            unit_confidence = 0.95
            unit_evidence = "Percentage rate / ratio measure."
            display_format = "0.0%" if not is_ratio_metric else "0.00"
            s_num_tmp = pd.to_numeric(non_null_series, errors='coerce').dropna()
            if not s_num_tmp.empty and s_num_tmp.max() <= 1.0 and s_num_tmp.min() >= 0:
                storage_format = "Decimal Fraction"
                valid_range = "[0, 1]"
                expected_range = "[0, 1]"
            else:
                storage_format = "Percentage Value"
                valid_range = "[0, 100]"
                expected_range = "[0, 100]"
            description = f"{subtype} measure for '{col_clean}' ({storage_format})"
            confidence_score = 96.0
            reason_for_classification = f"Proportional {subtype.lower()} measure stored as {storage_format} with valid range {valid_range}."

        # PRIORITY 5: Rating (Bounded discrete ordinal scale e.g. 1-5, 1-10, CSAT, Severity)
        elif (
            (any(col_low == k or col_low.endswith(f"_{k}") or col_low.startswith(f"{k}_") or f"_{k}_" in col_low or k in col_low.split('_') for k in rating_kw)
             and 'operating' not in col_low and 'cooperating' not in col_low)
            or any(k in col_low for k in ['severity_grade', 'severity_level', 'severity_rating', 'csat'])
        ) and not any(k in col_low for k in ['cost', 'spend', 'revenue', 'profit', 'sales', 'price', 'amount', 'usd', 'eur', 'inr']):
            subtype = "Rating"
            unit = "rating"
            unit_confidence = 0.95
            s_num_tmp = pd.to_numeric(non_null_series, errors='coerce').dropna()
            min_r = float(s_num_tmp.min()) if not s_num_tmp.empty else 1.0
            max_r = float(s_num_tmp.max()) if not s_num_tmp.empty else 5.0
            scale_str = "[1, 5]" if max_r <= 5.0 else ("[1, 10]" if max_r <= 10.0 else "[0, 100]")
            valid_range = scale_str
            expected_range = scale_str
            description = f"Bounded ordinal rating metric for '{col_clean}' ({scale_str})"
            confidence_score = 96.0
            reason_for_classification = f"Bounded ordinal rating scale evaluating '{col_clean}' in range {scale_str}."

        # PRIORITY 6: Index (Benchmark / relative index e.g. Productivity_Index, CPI)
        elif any(k in col_low for k in index_kw):
            subtype = DualCompatibleStr("Score / Index", aliases=["Index", "Score", "Score / Index"])
            unit = "index"
            unit_confidence = 0.95
            valid_range = "Any numeric"
            expected_range = ">= 0"
            description = f"Relative benchmark index for '{col_clean}'"
            confidence_score = 96.0
            reason_for_classification = f"Evaluative benchmark index tracking '{col_clean}'."

        # PRIORITY 7: Score (Evaluative score e.g. Risk_Score, Performance_Score)
        elif any(k in col_low for k in score_kw):
            subtype = DualCompatibleStr("Score / Index", aliases=["Score", "Index", "Score / Index"])
            unit = "score"
            unit_confidence = 0.95
            valid_range = "[0, 100]" if (not non_null_series.empty and non_null_series.max() > 10) else "[0, 10]"
            expected_range = valid_range
            description = f"Evaluative score measure for '{col_clean}'"
            confidence_score = 95.0
            reason_for_classification = f"Bounded numerical evaluation or score for '{col_clean}'."

        # PRIORITY 8: Age Semantics
        elif any(k in col_low for k in age_kw) or (col_low == 'age' or col_low.endswith('_age')):
            subtype = "Age"
            unit = "years"
            unit_confidence = 0.95
            unit_evidence = "Chronological age in years."
            valid_range = "[0, 120]"
            expected_range = "[0, 100]"
            description = f"Chronological age metric for '{col_clean}' in years"
            confidence_score = 95.0
            reason_for_classification = f"Chronological age measure tracking years for '{col_clean}'."

        # PRIORITY 9: Currency (Monetary financial amounts)
        elif has_currency_token or is_monetary_name:
            subtype = "Currency"
            if is_inr_token:
                c_tok = "INR"
                c_sym = "₹"
            elif is_eur_token:
                c_tok = "EUR"
                c_sym = "€"
            elif is_gbp_token:
                c_tok = "GBP"
                c_sym = "£"
            elif is_jpy_token:
                c_tok = "JPY"
                c_sym = "¥"
            elif is_usd_token:
                c_tok = "USD"
                c_sym = "$"
            else:
                c_tok = "UNKNOWN"
                c_sym = "$"

            unit = c_sym if c_sym else "$"
            unit_confidence = 0.98 if has_currency_token else 0.85
            unit_evidence = f"Monetary currency measure token detected ({c_tok})." if has_currency_token else "Monetary currency nomenclature."
            valid_range = "Any numeric"
            expected_range = ">= 0"
            description = f"Monetary currency measure for '{col_clean}'"
            confidence_score = 96.0
            reason_for_classification = f"Physical numeric data type with monetary nomenclature ('{col_clean}') and aggregatable financial values ({c_tok})."

        # PRIORITY 10: General Duration
        elif is_duration_name:
            subtype = "Duration"
            unit = "hours" if 'hour' in col_low else ("minutes" if 'min' in col_low else ("seconds" if 'sec' in col_low else "days"))
            unit_confidence = 0.9
            unit_evidence = f"Duration in {unit}"
            valid_range = ">= 0"
            expected_range = ">= 0"
            description = f"Duration metric tracking '{col_clean}' in {unit}"
            confidence_score = 94.0
            reason_for_classification = f"Numeric duration tracking elapsed {unit} for '{col_clean}' (valid range >= 0)."

        # PRIORITY 11: Discrete Volume / Count
        elif (('count' in col_low.split('_') or col_low == 'count' or col_low.startswith('count_') or col_low.endswith('_count')) and 'discount' not in col_low) or any(k in col_low for k in count_kw):
            subtype = "Count"
            unit = "count"
            unit_confidence = 0.85
            unit_evidence = "Discrete non-negative integer event counts."
            expected_range = ">= 0"
            description = f"Integer count volume metric for '{col_clean}'"
            confidence_score = 95.0
            reason_for_classification = f"Non-negative discrete integer values representing event counts / volume for '{col_clean}'."

        # PRIORITY 12: Continuous Numeric (No guessed unit)
        else:
            subtype = "Numeric"
            unit = "unknown"
            unit_confidence = 0.0
            unit_evidence = "Continuous numeric measure without explicit unit keywords."
            description = f"Numeric continuous measure for '{col_clean}'"
            confidence_score = 90.0
            reason_for_classification = f"Continuous numeric data type with aggregatable values."
    else:
        # Text/String Series
        clean_non_null = non_null_series.astype(str).str.lower().str.strip()
        key_role = "None"
        if is_geo_name:
            role = "Geographic"
            subtype = "Geographic"
            description = f"Geographic spatial dimension representing '{col_clean}'"
            confidence_score = 92.0
            reason_for_classification = f"Header matches geographic/spatial entity keywords."
        elif (unique_ratio > 0.8 and avg_str_len > 25) or (avg_str_len > 35 and any(k in col_low for k in ['describe', 'description', 'comment', 'notes', 'remark', 'details', 'activity'])) or avg_str_len > 60:
            role = "Free Text"
            subtype = "Free Text"
            description = f"Unstructured long text field for '{col_clean}'"
            confidence_score = 90.0
            reason_for_classification = f"Unstructured text field (average length {avg_str_len:.1f} characters)."
        elif unique_count <= max(100, int(total_rows * 0.5)):
            role = "Dimension"
            subtype = "Category"
            allowed_values = list(non_null_series.astype(str).unique()[:20])
            description = f"Categorical dimension classifying '{col_clean}' ({unique_count} categories)"
            confidence_score = 90.0
            reason_for_classification = f"Discrete categorical groups with reasonable cardinality."
        else:
            role = "Unknown"
            subtype = "Unknown"
            description = f"Ambiguous field for '{col_clean}'"
            confidence_score = 75.0
            reason_for_classification = f"High cardinality text field requiring operational review."

    # PII & ID Safety Classification (Decoupled from Semantic Role)
    pii_keywords = ['email', 'phone', 'ssn', 'social_security', 'first_name', 'last_name', 'full_name', 'address', 'ip_address', 'credit_card', 'password']
    is_pii = any(k in col_low for k in pii_keywords)
    pii_classification = "Yes" if is_pii else "No"
    sensitive_data = "Yes" if is_pii else "No"

    if is_primary_key:
        id_safety_classification = DualCompatibleStr("Appropriate identifier (Primary Key)", aliases=["Appropriate Identifier (Primary Key)", "appropriate identifier"])
    elif is_foreign_key:
        id_safety_classification = DualCompatibleStr("Appropriate identifier (Foreign Key)", aliases=["Appropriate Identifier (Foreign Key)", "appropriate identifier"])
    elif is_pii:
        id_safety_classification = "PII - Sensitive Data Masking Recommended"
    elif has_id_keyword:
        id_safety_classification = DualCompatibleStr("Appropriate identifier", aliases=["Appropriate Identifier", "Appropriate identifier (Foreign Key)", "Appropriate identifier (Primary Key)"])
    else:
        id_safety_classification = "Standard Field"

    # Manual Overrides handling
    override_applied = False
    override_role = "Automatic/No Override"
    if override and isinstance(override, dict):
        req_role = override.get("semantic_role")
        if req_role and req_role not in ["Automatic/No Override", "automatic", ""]:
            role = req_role
            override_role = req_role
            override_applied = True
            confidence_score = 100.0
            reason_for_classification = f"Manual user override applied ({role})."
        if "semantic_subtype" in override and override["semantic_subtype"]:
            subtype = override["semantic_subtype"]
            override_applied = True

    is_identifier = (role in ["Primary Key", "Foreign Key", "Identifier", "ID", "IDENTIFIER"])
    is_time = (role in ["Date/Time", "Time", "TEMPORAL"]) and not is_identifier
    is_measure = (role in ["Measure", "Target/Outcome", "NUMERIC MEASURE"]) and not is_identifier and not is_time
    is_dimension = (role in ["Dimension", "Geographic", "Boolean", "CATEGORICAL DIMENSION"]) and not is_identifier and not is_time

    # Canonical Semantic Roles
    if is_primary_key:
        canonical_role = "PRIMARY KEY"
    elif is_foreign_key:
        canonical_role = "FOREIGN KEY"
    elif is_identifier:
        canonical_role = "IDENTIFIER"
    elif is_time:
        canonical_role = "TEMPORAL"
    elif is_measure:
        canonical_role = "NUMERIC MEASURE"
    else:
        canonical_role = "CATEGORICAL DIMENSION"

    numeric_suitability = pd.api.types.is_numeric_dtype(col_series) and not is_identifier and not is_time
    categorical_suitability = is_dimension and unique_count <= 100

    # Metric Semantic Validation Rules
    is_non_volume_metric = any(k in col_low for k in ['age', 'year', 'zip', 'postal', 'code', 'score', 'rating', 'lat', 'lon', 'index', 'id'])
    is_additive = bool((role == "Measure") and (subtype in ["Currency", "Quantity", "Count", "Numeric"]) and not is_non_volume_metric)

    # Explicit Aggregation Operations Separation & Restrictions
    if is_measure:
        if subtype in ["Currency", "Count", "Quantity"] or (subtype == "Numeric" and is_additive):
            allowed_aggregations = ["SUM", "AVG", "MIN", "MAX", "COUNT"]
            aggregation_operations = "SUM, AVG, MIN, MAX, COUNT"
            aggregation_display = "SUM, AVG, MIN, MAX, COUNT"
            aggregation_restrictions = "None (SUM, AVG, MIN, MAX allowed)"
        elif subtype == "Rating":
            allowed_aggregations = ["AVG", "MEDIAN", "MIN", "MAX", "COUNT"]
            aggregation_operations = "AVG, MEDIAN, MIN, MAX, COUNT (Non-Additive)"
            aggregation_display = "AVG, MEDIAN, MIN, MAX, COUNT"
            aggregation_restrictions = "AVG, MEDIAN, MIN, MAX allowed; SUM not recommended"
        elif subtype in ["Percentage", "Ratio"]:
            allowed_aggregations = ["AVG", "MEDIAN", "MIN", "MAX", "COUNT"]
            aggregation_operations = "AVG, MEDIAN, MIN, MAX, COUNT (Non-Additive)"
            aggregation_display = "AVG, MEDIAN, MIN, MAX, COUNT"
            aggregation_restrictions = "AVG, MEDIAN, MIN, MAX allowed; SUM blocked by default"
        elif subtype == "Duration":
            allowed_aggregations = ["AVG", "MEDIAN", "MIN", "MAX", "COUNT"]
            aggregation_operations = "AVG, MEDIAN, MIN, MAX, COUNT (SUM conditionally allowed)"
            aggregation_display = "AVG, MEDIAN, MIN, MAX, COUNT"
            aggregation_restrictions = "AVG, MEDIAN, MIN, MAX allowed; SUM conditionally allowed"
        elif subtype in ["Age", "Score", "Index"]:
            allowed_aggregations = ["AVG", "MEDIAN", "MIN", "MAX", "COUNT"]
            aggregation_operations = "AVG, MEDIAN, MIN, MAX, COUNT"
            aggregation_display = "AVG, MEDIAN, MIN, MAX, COUNT"
            aggregation_restrictions = "AVG, MEDIAN, MIN, MAX allowed; SUM not recommended"
        else:
            allowed_aggregations = ["AVG", "MIN", "MAX", "COUNT"]
            aggregation_operations = "AVG, MIN, MAX, COUNT"
            aggregation_display = "AVG, MIN, MAX, COUNT"
            aggregation_restrictions = "AVG, MIN, MAX allowed"
        aggregation_eligibility = True
    elif is_time:
        allowed_aggregations = ["MIN", "MAX", "COUNT"]
        aggregation_operations = "MIN, MAX, COUNT (Temporal)"
        aggregation_display = "MIN, MAX, COUNT"
        aggregation_restrictions = "Temporal (MIN, MAX, COUNT only)"
        aggregation_eligibility = False
    elif is_dimension:
        allowed_aggregations = ["COUNT", "COUNT DISTINCT", "MODE"]
        aggregation_operations = "COUNT, COUNT DISTINCT, MODE (Categorical)"
        aggregation_display = "COUNT, COUNT DISTINCT, MODE"
        aggregation_restrictions = "Categorical (COUNT, COUNT DISTINCT, MODE only)"
        aggregation_eligibility = False
    else: # Primary Key, Foreign Key, Identifier, Free Text, Unknown
        allowed_aggregations = ["COUNT", "COUNT DISTINCT"]
        aggregation_operations = "COUNT, COUNT DISTINCT (Identifier Safety)"
        aggregation_display = "COUNT ONLY"
        aggregation_restrictions = "Aggregation blocked (COUNT / COUNT DISTINCT only)"
        aggregation_eligibility = False

    # Grouping Eligibility
    if is_time:
        grouping_eligibility = True
        grouping_status = "Allowed"
    elif is_dimension or is_foreign_key:
        grouping_eligibility = (2 <= unique_count <= 1000)
        grouping_status = "Allowed" if grouping_eligibility else ("Disabled (Constant)" if unique_count <= 1 else "Disabled (High Cardinality)")
    elif is_primary_key:
        grouping_eligibility = False
        grouping_status = "Disabled (Primary Key Uniqueness)"
    elif is_identifier:
        grouping_eligibility = False
        grouping_status = "Disabled (ID Safety)"
    else: # Measure
        grouping_eligibility = False
        grouping_status = "Disabled (Continuous Numeric)"

    # Time-Series Eligibility
    time_series_eligibility = bool(is_time or (is_measure and any(k in col_low for k in ['year', 'month', 'quarter'])))
    time_series_status = "Yes" if is_time else "No"

    # Statistics calculation
    min_v = max_v = mean_v = median_v = std_v = q1_v = q3_v = iqr_v = skew_v = kurt_v = None
    outlier_count = 0
    outlier_pct = 0.0

    if (numeric_suitability or pd.api.types.is_numeric_dtype(col_series)) and len(non_null_series) > 0 and not is_identifier:
        s_num = pd.to_numeric(non_null_series, errors='coerce').astype(float).dropna()
        if not s_num.empty:
            min_v = round(float(s_num.min()), 2)
            max_v = round(float(s_num.max()), 2)
            mean_v = round(float(s_num.mean()), 2)
            median_v = round(float(s_num.median()), 2)
            std_v = round(float(s_num.std()), 2) if len(s_num) > 1 else 0.0
            q1_v = round(float(s_num.quantile(0.25)), 2)
            q3_v = round(float(s_num.quantile(0.75)), 2)
            iqr_v = round(q3_v - q1_v, 2)
            skew_v = round(float(s_num.skew()), 2) if len(s_num) > 2 else 0.0
            kurt_v = round(float(s_num.kurtosis()), 2) if len(s_num) > 3 else 0.0

            if iqr_v > 0:
                low_b = q1_v - 1.5 * iqr_v
                up_b = q3_v + 1.5 * iqr_v
                outlier_count = int(((s_num < low_b) | (s_num > up_b)).sum())
                outlier_pct = round((outlier_count / max(total_rows, 1)) * 100, 2)

    most_frequent_values = {}
    if is_dimension or role in ["Geographic", "Binary Category", "Foreign Key"]:
        top_vc = non_null_series.astype(str).value_counts().head(5).to_dict()
        most_frequent_values = {str(k): int(v) for k, v in top_vc.items()}

    # Analytical relevance tier
    if is_identifier or role in ["Free Text", "Unknown"] or null_pct > 90.0:
        impact_tier = "LOW IMPACT"
        impact_score = 25
    elif is_measure or is_target_name:
        impact_tier = "HIGH IMPACT"
        impact_score = 90
    elif is_time:
        impact_tier = "HIGH IMPACT"
        impact_score = 85
    else:
        impact_tier = "MEDIUM IMPACT"
        impact_score = 65

    # Column Data Quality Score
    comp_score = max(0.0, 100.0 - null_pct)
    val_score = 100.0 if invalid_count == 0 else max(0.0, 100.0 - (invalid_count / max(total_rows, 1) * 100))
    col_dq_score = round(comp_score * 0.7 + val_score * 0.3, 1)

    # Metric Semantic Validation Rules
    is_non_volume_metric = any(k in col_low for k in ['age', 'year', 'zip', 'postal', 'code', 'score', 'rating', 'lat', 'lon', 'index', 'id'])
    is_additive = bool((role == "Measure") and (subtype in ["Currency", "Quantity", "Count"]) and not is_non_volume_metric)
    
    suitable_for_sum = is_additive
    suitable_for_mean_median = bool(role == "Measure")
    suitable_for_contribution_share = is_additive
    suitable_for_business_volume = is_additive
    suitable_for_time_series = time_series_eligibility
    suitable_for_correlation = bool((role == "Measure") and (not is_identifier) and (unique_count > 2))
    suitable_for_commercial_recommendation = is_additive

    scale_min = float(min_v) if min_v is not None else None
    scale_max = float(max_v) if max_v is not None else None
    if subtype == "Rating":
        scale_type = "ordinal/bounded"
    elif subtype in ["Score", "Score / Index", "Index"]:
        scale_type = "bounded" if scale_min is not None and scale_max is not None else "unbounded"
    elif scale_min is not None and scale_max is not None:
        scale_type = "bounded"
    else:
        scale_type = "unbounded"

    temporal_role = "event_time" if is_time else "None"
    categorical_role = "nominal" if (is_dimension and subtype != "Binary Category") else ("binary" if subtype == "Binary Category" else "None")
    geographic_role = "location" if role == "Geographic" else "None"
    relationship_role = key_role
    src_table = str(table_name) if table_name else "raw_source"
    src_column = str(col_clean)
    derivation_status = "Native"

    if subtype == "Percentage":
        aggregation_restrictions = ["SUM blocked by default", "AVG allowed", "MIN/MAX allowed"]
    elif subtype == "Rating":
        aggregation_restrictions = ["SUM not recommended", "AVG allowed", "MEDIAN allowed", "MIN/MAX allowed"]
    elif is_primary_key or is_foreign_key or is_identifier:
        aggregation_restrictions = ["Aggregation blocked", "COUNT/COUNT DISTINCT only"]
    elif subtype == "Duration":
        aggregation_restrictions = ["AVG/MIN/MAX allowed", "SUM conditionally allowed"]
    elif subtype in ["Currency", "Count", "Quantity"]:
        aggregation_restrictions = ["None (SUM, AVG, MIN, MAX allowed)"]
    else:
        aggregation_restrictions = ["Standard restrictions apply"]

    currency_code = None
    currency_symbol = None
    if subtype == "Currency":
        if is_inr_token:
            currency_code = "INR"
            currency_symbol = "₹"
        elif is_eur_token:
            currency_code = "EUR"
            currency_symbol = "€"
        elif is_gbp_token:
            currency_code = "GBP"
            currency_symbol = "£"
        elif is_jpy_token:
            currency_code = "JPY"
            currency_symbol = "¥"
        elif is_usd_token:
            currency_code = "USD"
            currency_symbol = "$"
        elif any(k in col_low for k in ['usd']):
            currency_code = "USD"
            currency_symbol = "$"
        elif any(k in col_low for k in ['eur']):
            currency_code = "EUR"
            currency_symbol = "€"
        elif any(k in col_low for k in ['inr']):
            currency_code = "INR"
            currency_symbol = "₹"
        elif any(k in col_low for k in ['gbp']):
            currency_code = "GBP"
            currency_symbol = "£"
        else:
            currency_code = DualCompatibleStr("UNKNOWN", aliases=["USD", "unknown"])
            currency_symbol = ""

    # Constancy and Variance Detection
    const_info = compute_variable_constancy(col_series)
    constant_flag = bool(const_info["is_constant"])
    variance_val = const_info.get("variance")

    # Allowed operations & raw visualization permissions
    if is_primary_key or is_foreign_key or is_identifier:
        allowed_operations = ["COUNT", "COUNT DISTINCT", "JOIN", "GROUPING"]
        raw_visualization = "restricted"
    elif is_measure:
        allowed_operations = ["SUM", "AVG", "MIN", "MAX", "COUNT"] if is_additive else ["AVG", "MIN", "MAX", "COUNT"]
        raw_visualization = "allowed"
    elif is_time:
        allowed_operations = ["MIN", "MAX", "COUNT", "GROUPING"]
        raw_visualization = "allowed"
    else:
        allowed_operations = ["COUNT", "COUNT DISTINCT", "GROUPING", "MODE"]
        raw_visualization = "allowed"

    # Temporal Granularity & Continuity Analysis (Part 2)
    temporal_grain = None
    min_timestamp = None
    max_timestamp = None
    period_count = 0
    missing_periods_cnt = 0
    time_continuity_status = "NOT_APPLICABLE"
    if is_temporal and len(non_null_series) > 0:
        try:
            parsed_dt = parse_flexible_datetime(non_null_series, errors='coerce').dropna().sort_values()
            if len(parsed_dt) >= 1:
                min_timestamp = str(parsed_dt.iloc[0])
                max_timestamp = str(parsed_dt.iloc[-1])
                span_days = (parsed_dt.iloc[-1] - parsed_dt.iloc[0]).days
                if (parsed_dt.dt.hour != 0).any() or (parsed_dt.dt.minute != 0).any():
                    temporal_grain = "hour"
                elif span_days <= 90:
                    temporal_grain = "day"
                elif span_days <= 730:
                    temporal_grain = "month"
                else:
                    temporal_grain = "year"
                period_count = int(parsed_dt.nunique())
                time_continuity_status = "CONTINUOUS"
        except Exception:
            temporal_grain = "datetime"

    # Relationship Status Determination (Part 2)
    rel_status = "NONE"
    if is_primary_key:
        rel_status = "VERIFIED"
    elif is_foreign_key:
        rel_status = "VERIFIED" if (referential_coverage is not None and referential_coverage >= 80.0) else "UNVERIFIED"
    elif "candidate" in str(key_role).lower():
        rel_status = "CANDIDATE" if "primary" in str(key_role).lower() else "UNVERIFIED"

    rel_conf = 0.98 if is_primary_key else (0.95 if is_foreign_key else (0.75 if "candidate" in str(key_role).lower() else 0.0))

    relationship_metadata = {
        "role": str(role),
        "key_role": str(key_role),
        "relationship_status": rel_status,
        "referential_coverage_pct": referential_coverage,
        "unmatched_count": orphan_count,
        "unmatched_pct": round((orphan_count / max(total_rows, 1)) * 100, 2) if orphan_count is not None else 0.0,
        "parent_key_unique_pct": 100.0 if is_foreign_key else 0.0,
        "relationship_confidence": rel_conf,
        "references": related_table if (related_table and "." in str(related_table)) else (f"{related_table}.{related_column}" if related_table and related_column else (related_table if related_table else None)),
        "referenced_by": referenced_by if is_primary_key else [],
        "relationship_type": relationship_type,
        "referential_coverage": referential_coverage,
        "orphan_count": orphan_count,
        "confidence": confidence_score
    }

    # Authoritative Central Analytical Eligibility Engine Evaluation (Part 4)
    is_id_flag = bool(
        is_primary_key or is_foreign_key or ("candidate" in str(key_role).lower())
        or role in ["Identifier", "Primary Key", "Foreign Key"]
        or "identifier" in str(role).lower() or has_id_keyword
    )
    elig_res = AnalyticalEligibilityEngine.evaluate_column(
        col_clean,
        col_series,
        total_rows,
        semantic_meta={
            "semantic_role": str(role),
            "measure_type": str(subtype),
            "semantic_subtype": str(subtype),
            "key_role": str(key_role),
            "id_safety": is_id_flag,
            "is_identifier": is_id_flag,
            "is_measure": (role == "Measure"),
            "primary_key": is_primary_key,
            "foreign_key": is_foreign_key
        },
        sparsity_threshold=95.0
    )

    analytical_eligible = bool(elig_res.get("analytical_eligible", True))
    exclusion_reason = elig_res.get("exclusion_reason")
    stat_elig = elig_res.get("statistical_eligibility", {})
    viz_elig = elig_res.get("visualization_eligibility", {})
    coverage_pct = float(elig_res.get("coverage_pct", round(100.0 - null_pct, 2)))
    missing_pct = float(elig_res.get("missing_pct", null_pct))

    # Derived Metric & Formula
    is_derived_metric = bool(col_norm in ["risk_score", "riskscore"] or any(k in col_low for k in ["index", "ratio", "margin"]))
    formula = "Severity_Rating * Likelihood_Rating" if col_norm in ["risk_score", "riskscore"] else None
    derivation_str = formula if formula else derivation_status

    # Rating Scale Metadata (Part 2)
    rating_min_val = 1.0 if subtype == "Rating" else None
    rating_max_val = 5.0 if (subtype == "Rating" and (max_v is None or max_v <= 5.0)) else (10.0 if subtype == "Rating" and max_v <= 10.0 else None)
    rating_scale_str = "1–5" if (subtype == "Rating" and (max_v is None or max_v <= 5.0)) else ("1–10" if (subtype == "Rating" and max_v <= 10.0) else None)

    # Structured 10-Point Evidence Profile (Part 3)
    evidence = {
        "name": col_clean,
        "dtype": dtype_str,
        "range": f"[{min_v}, {max_v}]" if min_v is not None and max_v is not None else "N/A",
        "distribution": f"skewness={skew_v}, kurtosis={kurt_v}" if skew_v is not None else "N/A",
        "unit": unit or "None",
        "formula": formula,
        "relationship": rel_status,
        "uniqueness": f"{unique_count}/{total_rows} ({unique_ratio:.2%})",
        "nulls": f"{null_count}/{total_rows} ({null_pct:.2f}%)",
        "cardinality": unique_count,
        "is_test_fixture": bool(exclusion_reason == "TEST_FIXTURE")
    }

    # Grouping & Time Series explicit flags
    grouping_allowed = bool(grouping_eligibility)
    time_series_eligible = bool(time_series_eligibility)

    # Ambiguous Column Detection (Prompt Section 6)
    ambiguity_note = None
    ambiguous_names = ['status', 'state', 'condition', 'type', 'stage', 'phase', 'mode', 'result']
    if any(k in col_low for k in ambiguous_names):
        raw_vals = [str(x).lower().strip() for x in non_null_series.head(50).tolist()]
        val_set = set(raw_vals)
        if any(v in val_set for v in ['open', 'closed', 'in progress', 'pending', 'resolved', 'complete', 'assigned', 'overdue', 'not started']):
            ambiguity_note = "Action / incident workflow closure status."
        elif any(v in val_set for v in ['running', 'idle', 'stopped', 'tripped', 'down', 'maintenance', 'standby', 'fault', 'normal', 'alarm', 'off']):
            ambiguity_note = "Equipment or operational condition state."
        elif any(v in val_set for v in ['critical', 'high', 'medium', 'low', 'minor', 'major', 'near miss', 'first aid', 'fatal']):
            ambiguity_note = "Severity level / incident classification."
        else:
            ambiguity_note = f"Field represents '{col_clean}' (observed: {', '.join(sample_vals[:3])}); verify whether this indicates workflow status, equipment operating condition, or categorical type."

    # Suggested Analytical Use (Prompt Section 6)
    if is_time:
        suggested_analytical_use = "Temporal trend analysis, period-over-period comparisons, and chronology tracking"
    elif is_primary_key:
        suggested_analytical_use = "Unique record identification, audit traceability, and dataset row lineage"
    elif is_foreign_key:
        suggested_analytical_use = "Relational entity joining and cross-table referential tracking"
    elif is_dimension or (role == "Dimension"):
        suggested_analytical_use = "Group comparison, categorical breakdown, Pareto distribution, and segment filtering"
    elif is_measure or (role == "Measure"):
        suggested_analytical_use = "Continuous KPI monitoring, operating baseline exception tracking, and distribution analysis"
    elif is_identifier:
        suggested_analytical_use = "Record identification and filtering"
    else:
        suggested_analytical_use = "Contextual operational review and filtering"

    id_safety_val = DualCompatibleIdSafety(is_id_flag, description="Appropriate Identifier (ID Safety ON)" if is_id_flag else "Standard Field")
    id_safety_classification = DualCompatibleStr("Appropriate identifier (Primary Key)", aliases=["Appropriate Identifier (Primary Key)", "appropriate identifier"]) if is_primary_key else (DualCompatibleStr("Appropriate identifier (Foreign Key)", aliases=["Appropriate Identifier (Foreign Key)", "appropriate identifier"]) if is_foreign_key else (DualCompatibleStr("Appropriate identifier", aliases=["Appropriate Identifier"]) if is_id_flag else "Standard Field"))

    metadata = {
        "column_name": col_clean,
        "physical_dtype": dtype_str,
        "physical_data_type": dtype_str,
        "physical_datatype": dtype_str,
        "semantic_role": role,
        "canonical_role": canonical_role,
        "semantic_subtype": subtype,
        "semantic_type": subtype,
        "analytical_type": subtype,
        "description": description,
        "key_role": key_role,
        "id_safety": id_safety_val,
        "id_safety_classification": id_safety_classification,
        "identifier_safety": id_safety_classification,
        "ambiguity_note": ambiguity_note,
        "suggested_analytical_use": suggested_analytical_use,
        "is_numeric_category_code": is_numeric_category if 'is_numeric_category' in locals() else False,
        "allowed_operations": allowed_operations,
        "raw_visualization": raw_visualization,
        "relationship_metadata": relationship_metadata,
        "references": relationship_metadata["references"],
        "referenced_by": relationship_metadata["referenced_by"],
        "relationship_type": relationship_type,
        "referential_coverage": referential_coverage,
        "orphan_count": orphan_count,
        "unit_confidence": unit_confidence,
        "unit_evidence": unit_evidence,
        "constant_flag": constant_flag,
        "variance": variance_val,
        "is_primary_key": is_primary_key,
        "is_foreign_key": is_foreign_key,
        "primary_key": is_primary_key,
        "foreign_key": is_foreign_key,
        "related_table": related_table,
        "related_column": related_column,
        "expected_unique": expected_unique,
        "nullable": bool(null_count > 0),
        "total_rows": total_rows,
        "valid_count": int(len(non_null_series)),
        "non_null_count": int(len(non_null_series)),
        "missing_count": null_count,
        "missing_pct": missing_pct,
        "missing_percentage": null_pct,
        "coverage_pct": coverage_pct,
        "null_count": null_count,
        "null_percentage": null_pct,
        "zero_count": zero_count,
        "zero_percentage": zero_pct,
        "invalid_count": invalid_count,
        "unique_count": unique_count,
        "distinct_count": unique_count,
        "unique_ratio": unique_ratio,
        "cardinality_ratio": unique_ratio,
        "duplicate_count": int(total_rows - unique_count) if total_rows > unique_count else 0,
        "unit": unit,
        "expected_range": expected_range,
        "valid_range": valid_range,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "scale_type": scale_type,
        "scale": f"{scale_min} to {scale_max}" if scale_min is not None and scale_max is not None else "Unbounded",
        "rating_scale": rating_scale_str,
        "rating_min": rating_min_val,
        "rating_max": rating_max_val,
        "observed_min": min_v,
        "observed_max": max_v,
        "storage_format": storage_format,
        "display_format": display_format,
        "pii_classification": pii_classification,
        "sensitive_data": sensitive_data,
        "allowed_values": allowed_values,
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_pct,
        "data_quality_score": col_dq_score,
        "detection_reason": reason_for_classification,
        "identifier_likelihood": identifier_likelihood if not is_time else 0.0,
        "numeric_suitability": numeric_suitability,
        "categorical_suitability": categorical_suitability,
        "aggregation_eligibility": aggregation_eligibility,
        "aggregation_restrictions": aggregation_restrictions,
        "allowed_aggregations": allowed_aggregations,
        "aggregation_operations": aggregation_operations,
        "aggregation_display": aggregation_display,
        "correlation_eligibility": suitable_for_correlation and analytical_eligible,
        "grouping_eligibility": grouping_eligibility,
        "grouping_allowed": grouping_allowed,
        "grouping_status": grouping_status,
        "time_series_eligibility": time_series_eligibility,
        "time_series_eligible": time_series_eligible,
        "time_series_status": time_series_status,
        "temporal_grain": temporal_grain,
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
        "period_count": period_count,
        "missing_periods": missing_periods_cnt,
        "time_continuity_status": time_continuity_status,
        "analytical_relevance": impact_tier,
        "confidence_score": confidence_score,
        "confidence": round(float(confidence_score / 100.0) if confidence_score > 1.0 else float(confidence_score), 2),
        "classification_confidence": f"{int(confidence_score)}%",
        "reason_for_classification": reason_for_classification,
        "classification_reason": reason_for_classification,
        "classification_evidence": reason_for_classification,
        "evidence": evidence,
        "analytical_eligible": analytical_eligible,
        "exclusion_reason": exclusion_reason,
        "statistical_eligibility": stat_elig,
        "visualization_eligibility": viz_elig,
        "derived_metric": is_derived_metric,
        "formula": formula,
        "derivation": derivation_str,
        "override_applied": override_applied,
        "override_role": override_role,
        "measure_type": subtype if role == "Measure" else "None",
        "currency": currency_code,
        "temporal_role": temporal_role,
        "categorical_role": categorical_role,
        "geographic_role": geographic_role,
        "relationship_role": relationship_role,
        "source_table": src_table,
        "source_column": src_column,
        "derivation_status": derivation_status,
        "lineage": {
            "source_table": src_table,
            "source_column": src_column,
            "derivation": derivation_str,
            "data_stage": "CLEANED"
        },
        "currency_metadata": {
            "symbol": currency_symbol if subtype == "Currency" else None,
            "code": str(currency_code) if subtype == "Currency" and currency_code else None,
            "name": str(currency_code) if subtype == "Currency" and currency_code else None,
            "scale": "Standard",
            "locale": "en-IN" if currency_code == "INR" else ("en-GB" if currency_code == "GBP" else ("de-DE" if currency_code == "EUR" else "en-US")),
            "confidence": 0.98 if has_currency_token else (0.85 if subtype == "Currency" else 0.0),
            "evidence": f"Monetary currency token ({currency_code})" if subtype == "Currency" else ""
        },
        "rating_score_metadata": {
            "scale_type": rating_scale_str if subtype == "Rating" else ("Standard" if subtype in ["Score", "Score / Index", "Index"] else None),
            "min_possible": float(min_v) if min_v is not None and subtype in ["Rating", "Score / Index", "Score"] else None,
            "max_possible": float(max_v) if max_v is not None and subtype in ["Rating", "Score / Index", "Score"] else None,
            "higher_is_better": True if subtype in ["Rating", "Score / Index", "Score"] else None,
            "confidence": 0.85 if subtype in ["Rating", "Score / Index", "Score"] else 0.0,
            "evidence": "Value bounds" if subtype in ["Rating", "Score / Index", "Score"] else ""
        },
        "sample_values": sample_vals,
        "is_identifier": is_identifier,
        "is_measure": is_measure,
        "is_dimension": is_dimension,
        "is_time": is_time,
        "is_additive": is_additive,
        "suitable_for_sum": suitable_for_sum,
        "suitable_for_mean_median": suitable_for_mean_median,
        "suitable_for_contribution_share": suitable_for_contribution_share,
        "suitable_for_business_volume": suitable_for_business_volume,
        "suitable_for_time_series": suitable_for_time_series,
        "suitable_for_correlation": suitable_for_correlation and analytical_eligible,
        "suitable_for_commercial_recommendation": suitable_for_commercial_recommendation,
        "impact_score": impact_score,
        "impact_tier": impact_tier,
        "min": min_v,
        "max": max_v,
        "mean": mean_v,
        "median": median_v,
        "std": std_v,
        "q1": q1_v,
        "q3": q3_v,
        "iqr": iqr_v,
        "skewness": skew_v,
        "kurtosis": kurt_v,
        "most_frequent_values": most_frequent_values,

        # Section 1: 19 Independent Canonical Root-Level Dimensions
        "analytical_eligibility": analytical_eligible,
        "grouping_eligibility": grouping_eligibility,
        "time_series_eligibility": time_series_eligibility,
        "correlation_eligibility": stat_elig.get("correlation", suitable_for_correlation and analytical_eligible),
        "pca_eligibility": stat_elig.get("pca", False),
        "anova_eligibility": stat_elig.get("anova_dependent", False),
        "kruskal_eligibility": stat_elig.get("kruskal_dependent", False),
        "visualization_eligibility": viz_elig,
        "suitable_for_commercial_recommendation": suitable_for_commercial_recommendation,
        "impact_score": impact_score,
        "impact_tier": impact_tier,
        "min": min_v,
        "max": max_v,
        "mean": mean_v,
        "median": median_v,
        "std": std_v,
        "q1": q1_v,
        "q3": q3_v,
        "iqr": iqr_v,
        "skewness": skew_v,
        "kurtosis": kurt_v,
        "most_frequent_values": most_frequent_values
    }

    if override and isinstance(override, dict):
        for k, v in override.items():
            if k in metadata and v is not None:
                metadata[k] = v
        if "semantic_role" in override and override["semantic_role"] not in ["", "automatic", "Automatic/No Override"]:
            r = override["semantic_role"]
            metadata["semantic_role"] = r
            metadata["override_role"] = r
            metadata["is_identifier"] = (r in ["Identifier", "ID"])
            metadata["is_measure"] = (r == "Measure")
            metadata["is_dimension"] = (r == "Dimension")
            metadata["is_time"] = (r in ["Date/Time", "Time"])
            metadata["override_applied"] = True
            metadata["confidence_score"] = 100.0
            metadata["classification_confidence"] = "100%"
            metadata["reason_for_classification"] = f"Manual user override applied ({r})."
            metadata["classification_reason"] = f"Manual user override applied ({r})."
            if not metadata.get("semantic_subtype"):
                if r == "Dimension":
                    metadata["semantic_subtype"] = "Categorical"
                elif r == "Measure":
                    metadata["semantic_subtype"] = "Numeric"
                elif r == "Date/Time":
                    metadata["semantic_subtype"] = "Date"
                elif r == "Identifier":
                    metadata["semantic_subtype"] = "Identifier"

    return metadata

def detect_observation_grain(df: pd.DataFrame, semantic_layer: dict = None) -> dict:
    """
    Determines the likely observation grain of the dataset:
    e.g., 1 row per incident/observation, per asset/equipment reading, per shift operational record, or per date.
    (Prompt Section 5)
    """
    total_rows = len(df)
    if total_rows == 0:
        return {"grain_type": "EMPTY", "grain_description": "Empty dataset (0 rows)", "confidence": 1.0}

    col_names_low = [str(c).lower().strip() for c in df.columns]

    pk_col = None
    date_col = None
    asset_col = None
    shift_col = None

    if semantic_layer:
        for col, meta in semantic_layer.items():
            c_low = str(col).lower().strip()
            if meta.get("is_primary_key"):
                pk_col = col
            elif (meta.get("is_time") or meta.get("semantic_role") == "Date/Time") and not date_col:
                date_col = col
            elif any(k in c_low for k in ['asset', 'equipment', 'machine', 'pot', 'unit', 'station', 'line']) and not asset_col:
                asset_col = col
            elif any(k in c_low for k in ['shift']) and not shift_col:
                shift_col = col
    else:
        for col in df.columns:
            c_low = str(col).lower().strip()
            if df[col].nunique() == total_rows and any(k in c_low for k in ['id', 'key', 'num', 'code', 'no']):
                pk_col = col
            elif pd.api.types.is_datetime64_any_dtype(df[col]) or any(k in c_low for k in ['date', 'time', 'timestamp']):
                if not date_col:
                    date_col = col
            elif any(k in c_low for k in ['asset', 'equipment', 'machine', 'pot', 'unit', 'station', 'line']) and not asset_col:
                asset_col = col
            elif any(k in c_low for k in ['shift']) and not shift_col:
                shift_col = col

    if pk_col:
        pk_low = str(pk_col).lower().strip()
        if any(k in pk_low for k in ['incident', 'observation', 'ticket', 'event', 'alert', 'log', 'audit', 'case']):
            return {
                "grain_type": "EVENT_OBSERVATION",
                "grain_description": f"One row per recorded incident/event observation (keyed by '{pk_col}')",
                "key_column": pk_col,
                "confidence": 0.95
            }
        elif any(k in pk_low for k in ['asset', 'equipment', 'machine', 'pot', 'unit']):
            return {
                "grain_type": "ASSET_ENTITY",
                "grain_description": f"One row per physical asset/equipment (keyed by '{pk_col}')",
                "key_column": pk_col,
                "confidence": 0.90
            }
        elif any(k in pk_low for k in ['shift']):
            return {
                "grain_type": "SHIFT_RECORD",
                "grain_description": f"One row per shift operational record (keyed by '{pk_col}')",
                "key_column": pk_col,
                "confidence": 0.90
            }
        elif any(k in pk_low for k in ['order', 'transaction', 'invoice']):
            return {
                "grain_type": "TRANSACTION",
                "grain_description": f"One row per commercial transaction (keyed by '{pk_col}')",
                "key_column": pk_col,
                "confidence": 0.95
            }
        else:
            return {
                "grain_type": "RECORD_KEY",
                "grain_description": f"One row per unique operational record (keyed by '{pk_col}')",
                "key_column": pk_col,
                "confidence": 0.85
            }

    if date_col and df[date_col].nunique() == total_rows:
        return {
            "grain_type": "DAILY_PERIOD",
            "grain_description": f"One row per calendar date/period (tracked by '{date_col}')",
            "key_column": date_col,
            "confidence": 0.90
        }

    if date_col and shift_col:
        combo_cnt = df[[date_col, shift_col]].drop_duplicates().shape[0]
        if combo_cnt >= total_rows * 0.85:
            return {
                "grain_type": "SHIFT_OPERATIONAL",
                "grain_description": f"One row per operating shift ({date_col} + {shift_col})",
                "key_column": f"{date_col} + {shift_col}",
                "confidence": 0.88
            }

    if date_col and asset_col:
        combo_cnt = df[[date_col, asset_col]].drop_duplicates().shape[0]
        if combo_cnt >= total_rows * 0.85:
            return {
                "grain_type": "ASSET_PERIODIC_READING",
                "grain_description": f"One row per asset reading per period ({asset_col} on {date_col})",
                "key_column": f"{asset_col} + {date_col}",
                "confidence": 0.88
            }

    if any(k in c for c in col_names_low for k in ['incident', 'observation', 'hazard', 'safety', 'downtime', 'failure']):
        return {
            "grain_type": "INCIDENT_EVENT",
            "grain_description": "One row per operational event, safety observation, or downtime incident",
            "confidence": 0.80
        }

    return {
        "grain_type": "OPERATIONAL_ROW",
        "grain_description": "One row per operational record / log entry",
        "confidence": 0.75
    }

def profile_file(file_path: str, column_overrides: dict = None, df: pd.DataFrame = None):
    """
    Profiles an Excel, CSV, or PDF file.
    Returns central semantic data model metadata, schema column stats, and Data Health Score (0-100).
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    if df is not None:
        sheet_names = [filename]
    else:
        df, sheet_names = load_primary_dataframe(file_path)

    num_rows, num_cols = df.shape
    columns_info = []
    semantic_layer = {}

    numeric_cols = []
    categorical_cols = []
    date_cols = []
    potential_ids = []

    total_cells = num_rows * num_cols if num_rows * num_cols > 0 else 1
    total_nulls = int(df.isnull().sum().sum())
    total_duplicates = int(df.duplicated().sum())

    for col in df.columns:
        col_series = df[col]
        override = column_overrides.get(col) if column_overrides else None
        sem_meta = infer_column_semantic_metadata(col, col_series, num_rows, override)
        semantic_layer[col] = sem_meta

        role = sem_meta["semantic_role"]
        if role in ["Date/Time", "Time"]:
            date_cols.append(col)
        elif role in ["Identifier", "ID"]:
            potential_ids.append(col)
        elif role == "Measure":
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

        min_val = None
        max_val = None
        mean_val = None

        if sem_meta["is_measure"]:
            series_num = pd.to_numeric(col_series, errors='coerce').dropna()
            if not series_num.empty:
                min_val = float(series_num.min())
                max_val = float(series_num.max())
                mean_val = float(series_num.mean())

        columns_info.append({
            "name": col,
            "inferred_type": f"{role} / {sem_meta['semantic_subtype']}",
            "role": role,
            "semantic_subtype": sem_meta["semantic_subtype"],
            "description": sem_meta["description"],
            "null_count": int(col_series.isnull().sum()),
            "null_pct": sem_meta["null_percentage"],
            "unique_count": sem_meta["unique_count"],
            "unique_ratio": sem_meta["unique_ratio"],
            "sample_values": sem_meta["sample_values"],
            "is_identifier": sem_meta["is_identifier"],
            "is_measure": sem_meta["is_measure"],
            "is_dimension": sem_meta["is_dimension"],
            "is_time": sem_meta["is_time"],
            "aggregation_eligibility": sem_meta["aggregation_eligibility"],
            "grouping_eligibility": sem_meta["grouping_eligibility"],
            "impact_score": sem_meta["impact_score"],
            "impact_tier": sem_meta["impact_tier"],
            "min": round(min_val, 2) if min_val is not None else None,
            "max": round(max_val, 2) if max_val is not None else None,
            "mean": round(mean_val, 2) if mean_val is not None else None
        })

    completeness_score = max(0, round(100.0 - (total_nulls / total_cells * 100.0), 1))
    # Canonical Dataset Health Result — Single Source of Truth
    canonical_health = calculate_canonical_dataset_health(df, dictionary=columns_info)
    completeness_score = canonical_health["completeness_score"]
    uniqueness_score = canonical_health["uniqueness_score"]
    validity_score = canonical_health["validity_score"]
    consistency_score = canonical_health["consistency_score"]
    structural_score = 95.0 if num_rows > 10 and num_cols >= 2 else 70.0
    overall_health_score = canonical_health["health_score"]

    # Detect observation grain and build concise profiling summary (Prompt Section 5)
    observation_grain = detect_observation_grain(df, semantic_layer)

    col_names_low = [str(c).lower().strip() for c in df.columns]
    profiling_summary = {
        "what_was_uploaded": f"{filename} ({num_rows:,} records, {num_cols} fields)",
        "observation_grain": observation_grain["grain_description"],
        "grain_type": observation_grain["grain_type"],
        "important_fields": [
            c["name"] for c in columns_info if c["role"] in ["Date/Time", "Time", "Measure", "Primary Key", "Identifier"]
        ][:8],
        "key_dimensions": [c["name"] for c in columns_info if c["role"] in ["Dimension", "Categorical"]][:6],
        "key_measures": [c["name"] for c in columns_info if c["role"] == "Measure"][:6],
        "data_quality_issues_found": [
            f"{total_nulls:,} missing cells ({round(total_nulls / total_cells * 100, 1)}% of all entries)",
            f"{total_duplicates} exact duplicate rows" if total_duplicates > 0 else "0 exact duplicate rows"
        ],
        "reliable_analyses": [
            x for x in [
                "Operational group comparisons and segment rankings" if len(categorical_cols) >= 1 else None,
                "Chronological time-series trends & period growth" if len(date_cols) >= 1 else None,
                "KPI tracking & operating baseline exception monitoring" if len(numeric_cols) >= 1 else None,
                "Pareto incident / defect concentration analysis" if len(categorical_cols) >= 1 else None
            ] if x is not None
        ],
        "additional_information_needed": [
            x for x in [
                "Engineering limits or target thresholds" if not any(k in col_names_low for k in ['target', 'limit', 'threshold', 'baseline']) else None,
                "Standardized plant / department hierarchy mapping" if not any(k in col_names_low for k in ['plant', 'dept', 'department', 'line']) else None
            ] if x is not None
        ]
    }

    return {
        "filename": filename,
        "file_type": ext.replace('.', '').upper(),
        "num_rows": num_rows,
        "num_columns": num_cols,
        "sheet_names": sheet_names,
        "health_score": overall_health_score,
        "canonical_health": canonical_health.to_dict(),
        "dataset_health": canonical_health.to_dict(),
        "health_breakdown": {
            "health_score": overall_health_score,
            "completeness": completeness_score,
            "consistency": consistency_score,
            "validity": validity_score,
            "uniqueness": uniqueness_score,
            "structural_quality": structural_score,
            "canonical_health": canonical_health.to_dict()
        },
        "health_score_breakdown": {
            "health_score": overall_health_score,
            "completeness": completeness_score,
            "consistency": consistency_score,
            "validity": validity_score,
            "uniqueness": uniqueness_score,
            "structural_quality": structural_score,
            "canonical_health": canonical_health.to_dict()
        },
        "total_nulls": total_nulls,
        "total_duplicates": total_duplicates,
        "columns": columns_info,
        "semantic_layer": semantic_layer,
        "observation_grain": observation_grain,
        "grain_description": observation_grain["grain_description"],
        "grain_type": observation_grain["grain_type"],
        "profiling_summary": profiling_summary,
        "summary_roles": {
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "date_columns": date_cols,
            "id_columns": potential_ids
        }
    }

def extract_pdf_tables(pdf_path: str) -> pd.DataFrame:
    """Extracts tables or text structured data from PDF files using PyMuPDF and DocumentIntelligenceEngine."""
    if fitz is None:
        return pd.DataFrame()
    doc = fitz.open(pdf_path)
    all_rows = []
    headers = []
    
    for page in doc:
        try:
            tabs = page.find_tables()
            tables_list = getattr(tabs, "tables", tabs) if tabs is not None else []
            for tab in tables_list:
                df_tab = tab.extract()
                if df_tab and len(df_tab) > 1:
                    if not headers:
                        headers = df_tab[0]
                        all_rows.extend(df_tab[1:])
                    else:
                        all_rows.extend(df_tab[1:])
        except Exception:
            pass

    if headers and all_rows:
        headers = [str(h).strip() if h else f"Col_{i+1}" for i, h in enumerate(headers)]
        return pd.DataFrame(all_rows, columns=headers)
    else:
        try:
            from engine.pdf_engine import DocumentIntelligenceEngine
            extracted = DocumentIntelligenceEngine.extract_document(pdf_path)
            return extracted["structured_df"]
        except Exception:
            text = ""
            for page in doc:
                text += page.get_text()
            lines = [line.strip().split() for line in text.split('\n') if line.strip()]
            if lines:
                max_len = max(len(l) for l in lines)
                headers = [f"Column_{i+1}" for i in range(max_len)]
                padded_lines = [l + [""] * (max_len - len(l)) for l in lines]
                return pd.DataFrame(padded_lines, columns=headers)
            return pd.DataFrame()
