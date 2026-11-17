import pandas as pd
import numpy as np
import re
from engine.profiler import infer_column_semantic_metadata

def generate_data_dictionary(df: pd.DataFrame, filename: str = "dataset", column_overrides: dict = None, table_name: str = None, all_tables: dict = None):
    """
    Builds a central Data Dictionary table for every column in the DataFrame with exact statistics
    and semantic model metadata. Guarantees separate concepts for:
    - Physical Data Type
    - Semantic Role (Primary Key, Foreign Key, Dimension, Measure, Temporal)
    - Semantic Type (Binary Category, Percentage, Duration, Currency, Count, Quantity, etc.)
    - Business Meaning & Description
    - Unit, Expected Range, Storage/Display Format
    - Relationships (Primary Key, Foreign Key, Related Table, Related Column)
    - Privacy / Security (PII Classification, ID Safety Classification)
    - Data Quality Statistics (Total Rows, Valid, Missing, Unique, Duplicates, Min, Max, Mean, Median, Std, IQR, Skewness, Outliers)
    - Confidence Score, Data Quality Score, Detection Reason
    All values are calculated strictly from the current analytical dataset.
    """
    total_rows = len(df)
    dictionary = []

    # If table_name is not provided, infer from filename if possible
    if not table_name and filename:
        clean_fn = re.sub(r'\.(xlsx|xls|csv|json)$', '', str(filename).strip(), flags=re.IGNORECASE)
        table_name = clean_fn

    for col in df.columns:
        series = df[col]
        override = column_overrides.get(col) if column_overrides else None
        meta = infer_column_semantic_metadata(col, series, total_rows, override=override, table_name=table_name, all_tables=all_tables)

        sample_series = series.dropna()
        sample_val = str(sample_series.iloc[0]) if not sample_series.empty else "N/A"

        min_v = meta.get("min")
        max_v = meta.get("max")
        mean_v = meta.get("mean")
        median_v = meta.get("median")
        std_v = meta.get("std")
        q1_v = meta.get("q1")
        q3_v = meta.get("q3")
        iqr_v = meta.get("iqr")
        skew_v = meta.get("skewness")
        kurt_v = meta.get("kurtosis")
        outlier_count = meta.get("outlier_count", 0)
        outlier_pct = meta.get("outlier_percentage", 0.0)

        # Ensure exact calculations for all numeric columns from the analytical dataset snapshot
        is_num = (pd.api.types.is_numeric_dtype(series) or meta.get("is_measure", False)) and not meta.get("is_identifier", False)
        negative_cnt = None
        var_v = None

        if is_num and not sample_series.empty:
            series_num = pd.to_numeric(series, errors='coerce').dropna().astype(float)
            if not series_num.empty:
                min_v = round(float(series_num.min()), 2)
                max_v = round(float(series_num.max()), 2)
                mean_v = round(float(series_num.mean()), 2)
                median_v = round(float(series_num.median()), 2)
                std_v = round(float(series_num.std()), 2) if len(series_num) > 1 else 0.0
                var_v = round(float(series_num.var()), 2) if len(series_num) > 1 else 0.0
                q1_v = round(float(series_num.quantile(0.25)), 2)
                q3_v = round(float(series_num.quantile(0.75)), 2)
                iqr_v = round(q3_v - q1_v, 2)
                skew_v = round(float(series_num.skew()), 2) if len(series_num) > 2 else 0.0
                kurt_v = round(float(series_num.kurtosis()), 2) if len(series_num) > 3 else 0.0
                negative_cnt = int((series_num < 0).sum())
                if iqr_v > 0:
                    low_b = q1_v - 1.5 * iqr_v
                    up_b = q3_v + 1.5 * iqr_v
                    outlier_count = int(((series_num < low_b) | (series_num > up_b)).sum())
                    outlier_pct = round((outlier_count / max(total_rows, 1)) * 100, 2)
        else:
            min_v = max_v = mean_v = median_v = std_v = var_v = q1_v = q3_v = iqr_v = skew_v = kurt_v = None
            outlier_count = 0
            outlier_pct = 0.0

        null_cnt = int(series.isnull().sum())
        null_pct = round((null_cnt / max(total_rows, 1)) * 100, 2)
        valid_cnt = total_rows - null_cnt
        uniq_cnt = int(sample_series.nunique())
        uniq_pct = round((uniq_cnt / max(total_rows, 1)) * 100, 2)
        dup_cnt = int(total_rows - uniq_cnt) if total_rows > uniq_cnt else 0

        # Data quality score for column
        completeness = max(0.0, 100.0 - null_pct)
        validity = 100.0 if meta.get("invalid_count", 0) == 0 else max(0.0, 100.0 - (meta.get("invalid_count", 0) / max(total_rows, 1) * 100))
        dq_score = round(completeness * 0.7 + validity * 0.3, 1)

        col_unit = meta.get("unit") or "unknown"

        dictionary.append({
            "column_name": str(col),
            "table_name": str(table_name or filename),
            "physical_datatype": meta["physical_dtype"],
            "physical_data_type": meta["physical_dtype"],
            "physical_dtype": meta["physical_dtype"],
            "data_type": meta["physical_dtype"],
            "dtype": meta["physical_dtype"],
            
            # Semantic Architecture & Role Separation
            "semantic_role": meta["semantic_role"],
            "role": meta["semantic_role"],
            "key_role": meta.get("key_role", "None"),
            "semantic_type": meta.get("semantic_subtype") or meta.get("semantic_type") or "Dimension",
            "semantic_subtype": meta.get("semantic_subtype") or "General",
            "business_meaning": meta["description"],
            "description": meta["description"],
            "ambiguity_note": meta.get("ambiguity_note"),
            "suggested_analytical_use": meta.get("suggested_analytical_use"),
            "is_numeric_category_code": meta.get("is_numeric_category_code", False),
            
            # Unit, Confidence, and Range
            "unit": col_unit,
            "inferred_unit": col_unit,
            "unit_confidence": meta.get("unit_confidence", 1.0 if col_unit != "unknown" else 0.0),
            "unit_evidence": meta.get("unit_evidence", ""),
            "expected_range": meta.get("expected_range") or meta.get("valid_range") or "N/A",
            "valid_range": meta.get("valid_range") or "N/A",
            "storage_format": meta.get("storage_format") or "Standard",
            "display_format": meta.get("display_format") or "Standard",
            
            # Nullable & Counts
            "nullable": bool(null_cnt > 0),
            "row_count": total_rows,
            "total_rows": total_rows,
            "non_null_count": valid_cnt,
            "valid_values": valid_cnt,
            "valid_count": valid_cnt,
            "missing_values": null_cnt,
            "missing_count": null_cnt,
            "null_count": null_cnt,
            "missing_%": null_pct,
            "missing_percentage": null_pct,
            "null_percentage": null_pct,
            "unique_values": uniq_cnt,
            "unique_count": uniq_cnt,
            "distinct_count": uniq_cnt,
            "unique_percentage": uniq_pct,
            "unique_ratio": meta["unique_ratio"],
            "duplicate_values": dup_cnt,
            "duplicate_count": dup_cnt,
            "zero_count": meta.get("zero_count", 0),
            "zero_percentage": meta.get("zero_percentage", 0.0),
            "negative_count": negative_cnt,
            "invalid_count": meta.get("invalid_count", 0),
            "constant_flag": meta.get("constant_flag", False),
            
            # Numeric Statistics
            "min": min_v,
            "max": max_v,
            "mean": mean_v,
            "median": median_v,
            "std": std_v,
            "standard_deviation": std_v,
            "variance": var_v if var_v is not None else meta.get("variance"),
            "q1": q1_v,
            "q3": q3_v,
            "iqr": iqr_v,
            "skewness": skew_v,
            "kurtosis": kurt_v,
            "outlier_count": outlier_count,
            "outlier_%": outlier_pct,
            "outlier_percentage": outlier_pct,
            
            # Privacy & Security
            "id_safety": meta.get("id_safety"),
            "pii_classification": meta.get("pii_classification", "No"),
            "id_safety_classification": meta.get("id_safety_classification", "Standard Field"),
            "identifier_safety": meta.get("id_safety_classification", "Standard Field"),
            "sensitive_data": meta.get("sensitive_data", "No"),
            
            # Relational Model & Relationship Metadata
            "primary_key": meta.get("primary_key", False),
            "is_primary_key": meta.get("is_primary_key", False),
            "foreign_key": meta.get("foreign_key", False),
            "is_foreign_key": meta.get("is_foreign_key", False),
            "fk_violation_count": meta.get("orphan_count", 0),
            "expected_unique": meta.get("expected_unique", False),
            "references": meta.get("references") or (meta.get("related_table") if (meta.get("related_table") and "." in str(meta.get("related_table"))) else (f"{meta.get('related_table')}.{meta.get('related_column')}" if meta.get('related_table') else None)),
            "referenced_by": meta.get("referenced_by", []),
            "related_table": meta.get("related_table"),
            "related_column": meta.get("related_column"),
            "relationship_type": meta.get("relationship_type", "None"),
            "referential_coverage": meta.get("referential_coverage"),
            "orphan_count": meta.get("orphan_count", 0),
            "relationship_metadata": meta.get("relationship_metadata", {
                "role": str(meta.get("semantic_role")),
                "key_role": meta.get("key_role", "None"),
                "references": meta.get("references"),
                "referenced_by": meta.get("referenced_by", []),
                "relationship_type": meta.get("relationship_type", "None"),
                "referential_coverage": meta.get("referential_coverage"),
                "orphan_count": meta.get("orphan_count", 0)
            }),
            
            # Governance Metadata: Measure Type, Currency & Scale
            "measure_type": meta.get("measure_type") or meta.get("semantic_subtype") or "None",
            "analytical_type": meta.get("analytical_type") or meta.get("measure_type") or meta.get("semantic_subtype") or "None",
            "currency_metadata": meta.get("currency_metadata", {}),
            "currency": meta.get("currency") or (meta.get("currency_metadata", {}).get("code") if isinstance(meta.get("currency_metadata"), dict) and meta.get("currency_metadata", {}).get("code") != "None" else None),
            "scale": meta.get("scale") or (meta.get("rating_score_metadata", {}).get("scale_type", "Standard") if isinstance(meta.get("rating_score_metadata"), dict) else "Standard"),
            "scale_min": meta.get("scale_min"),
            "scale_max": meta.get("scale_max"),
            "scale_type": meta.get("scale_type", "unbounded"),
            "rating_scale": meta.get("rating_scale") or (meta.get("rating_score_metadata", {}).get("scale_type", "unknown") if isinstance(meta.get("rating_score_metadata"), dict) else "unknown"),
            "rating_score_metadata": meta.get("rating_score_metadata", {}),
            "zero_variance": meta.get("constant_flag", False),
            "zero_variance_flag": meta.get("constant_flag", False),
            "is_constant": meta.get("constant_flag", False),
            "invalid_category_count": meta.get("invalid_count", 0),
            "confidence": round(float(meta.get("confidence_score", 90.0) / 100.0) if meta.get("confidence_score", 0) > 1 else float(meta.get("confidence_score", 0.90)), 2),
            "evidence": meta.get("evidence", {}),

            # Data Stage & Stage-Specific Completeness Governance
            "data_stage": "CLEANED",
            "raw_completeness": f"{round(completeness, 2)}%",
            "cleaned_completeness": f"{round(completeness, 2)}%",
            "analytical_completeness": f"{round(completeness, 2)}%",
            "coverage_pct": meta.get("coverage_pct", round(completeness, 2)),
            "missing_pct": meta.get("missing_pct", null_pct),

            # Roles & Lineage
            "temporal_role": meta.get("temporal_role", "None"),
            "categorical_role": meta.get("categorical_role", "None"),
            "geographic_role": meta.get("geographic_role", "None"),
            "relationship_role": meta.get("relationship_role", meta.get("key_role", "None")),
            "source_table": meta.get("source_table", str(table_name or filename)),
            "source_column": meta.get("source_column", str(col)),
            "source": meta.get("source_table", str(table_name or filename)),
            "derivation_status": meta.get("derivation_status", "Native"),
            "derivation": meta.get("derivation", meta.get("derivation_status", "Native")),
            "derived_metric": meta.get("derived_metric", False),
            "formula": meta.get("formula"),
            "relationship": meta.get("relationship_type", "None"),
            "lineage": meta.get("lineage", {
                "source_table": meta.get("source_table", str(table_name or filename)),
                "source_column": str(col),
                "derivation": meta.get("derivation", "Native"),
                "data_stage": "CLEANED"
            }),

            # Permissions & Aggregations
            "allowed_operations": meta.get("allowed_operations", ["COUNT"]),
            "allowed_values": meta.get("allowed_values", []),
            "aggregation_options": meta.get("allowed_aggregations", []),
            "allowed_aggregations": meta.get("allowed_aggregations", []),
            "aggregation_restrictions": meta.get("aggregation_restrictions", []),
            "aggregation_operations": meta.get("aggregation_operations", ""),
            "aggregation_display": meta.get("aggregation_display", ""),
            
            # Scores & Detection
            "data_quality_score": dq_score,
            "confidence_score": meta.get("confidence_score", 90.0),
            "classification_confidence": meta.get("classification_confidence", "90%"),
            "detection_reason": meta.get("reason_for_classification", ""),
            "detection/inference_reason": meta.get("reason_for_classification", ""),
            "reason_for_classification": meta.get("reason_for_classification", ""),
            "classification_reason": meta.get("classification_reason", meta.get("reason_for_classification", "")),
            
            # Eligibility Flags & 19 Root Independent Fields (Section 1)
            "sample_value": sample_val,
            "sample_values": meta["sample_values"],
            "is_identifier": meta["is_identifier"],
            "is_measure": meta["is_measure"],
            "is_dimension": meta["is_dimension"],
            "is_time": meta["is_time"],
            "aggregation_eligibility": meta["aggregation_eligibility"],
            "grouping_eligibility": meta["grouping_eligibility"],
            "grouping_allowed": meta.get("grouping_allowed", meta.get("grouping_eligibility", True)),
            "grouping_status": meta.get("grouping_status", "Allowed" if meta["grouping_eligibility"] else "Disabled"),
            "time_series_eligibility": meta.get("time_series_eligibility", meta["is_time"]),
            "time_series_eligible": meta.get("time_series_eligible", meta.get("is_time", False)),
            "time_series_status": meta.get("time_series_status", "Yes" if meta["is_time"] else "No"),
            "analytical_eligibility": meta.get("analytical_eligibility", meta.get("analytical_eligible", True)),
            "analytical_eligible": meta.get("analytical_eligible", True),
            "correlation_eligibility": meta.get("correlation_eligibility", False),
            "pca_eligibility": meta.get("pca_eligibility", False),
            "anova_eligibility": meta.get("anova_eligibility", False),
            "kruskal_eligibility": meta.get("kruskal_eligibility", False),
            "exclusion_reason": meta.get("exclusion_reason"),
            "statistical_eligibility": meta.get("statistical_eligibility", {}),
            "visualization_eligibility": meta.get("visualization_eligibility", {}),
            "suitable_for_analysis": meta.get("suitable_for_analysis", True),
            "impact_score": meta.get("impact_score", 50),
            "impact_tier": meta.get("impact_tier", "MEDIUM IMPACT"),
            "override_applied": meta.get("override_applied", False),
            "override_role": meta.get("override_role", "Automatic/No Override"),
            "most_frequent_values": meta.get("most_frequent_values", {}),
            "is_ai_inferred": True,
            "filename": filename
        })

    return dictionary
