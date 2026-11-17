from typing import Any, Dict, List, Optional
from engine.contracts.base import GovernedContractBase
from engine.contracts.column_contract import ColumnSemanticContract
from engine.contracts.health_contract import DatasetHealthResult
from engine.contracts.lineage_contract import DerivationContract, RelationshipContract


class DatasetContract(GovernedContractBase):
    """
    Canonical Single Source of Truth for Dataset Metadata (Prompt Section 4).
    Every uploaded dataset generates exactly ONE DatasetContract.
    Contains raw and cleaned observation audit counts, column contracts, derivations,
    relationships, quality ledger, and the single canonical DatasetHealthResult.
    """
    dataset_id: str
    source_file_name: str
    source_file_hash: str = ""
    source_format: str = "EXCEL"  # EXCEL, CSV, PARQUET, JSON, PDF
    source_table: Optional[str] = "Master Dataset"
    ingestion_timestamp: str = ""
    
    # Audit Row & Column Counts (Section 9: Raw quality must remain visible)
    raw_row_count: int = 0
    raw_column_count: int = 0
    cleaned_row_count: int = 0
    analytical_row_count: int = 0
    raw_duplicate_count: int = 0
    cleaned_duplicate_count: int = 0
    
    schema_version: str = "2.0.0"
    
    # Governed Column Contracts
    columns: Dict[str, ColumnSemanticContract] = {}
    
    # Lineage, Derivations, and Relationships
    relationships: List[RelationshipContract] = []
    derivations: List[DerivationContract] = []
    
    # Canonical Quality, Eligibility & Health
    quality_result: Dict[str, Any] = {}
    analytical_eligibility_result: Dict[str, Any] = {}
    dataset_health_result: Optional[DatasetHealthResult] = None

    def get_column(self, col_name: str) -> Optional[ColumnSemanticContract]:
        return self.columns.get(col_name)

    def get_eligible_measures(self) -> List[str]:
        return [c for c, col in self.columns.items() if col.is_measure and col.analytical_eligible]

    def get_eligible_dimensions(self) -> List[str]:
        return [c for c, col in self.columns.items() if col.is_dimension and col.grouping_eligible]

    def get_time_columns(self) -> List[str]:
        return [c for c, col in self.columns.items() if col.is_time and col.time_series_eligible]

    def get_primary_keys(self) -> List[str]:
        return [c for c, col in self.columns.items() if col.is_primary_key]

    @classmethod
    def create_from_pipeline(
        cls,
        dataset_id: str,
        raw_df: Optional[Any],
        cleaned_df: Optional[Any],
        source_file_path: str = "",
        dictionary: Optional[List[Dict[str, Any]]] = None,
        quality_issues: Optional[List[Dict[str, Any]]] = None,
        audit_trail: Optional[List[Dict[str, Any]]] = None,
        derivations: Optional[List[Dict[str, Any]]] = None
    ) -> "DatasetContract":
        import os
        import datetime
        from engine.governed_engines import calculate_canonical_dataset_health

        r_df = raw_df if raw_df is not None else cleaned_df
        c_df = cleaned_df if cleaned_df is not None else raw_df

        raw_rows = len(r_df) if r_df is not None else 0
        raw_cols = len(r_df.columns) if r_df is not None and hasattr(r_df, "columns") else 0
        clean_rows = len(c_df) if c_df is not None else 0
        raw_dups = int(r_df.duplicated().sum()) if r_df is not None and hasattr(r_df, "duplicated") else 0
        clean_dups = int(c_df.duplicated().sum()) if c_df is not None and hasattr(c_df, "duplicated") else 0

        health_res = calculate_canonical_dataset_health(c_df, dictionary=dictionary, quality_issues=quality_issues)

        columns_dict: Dict[str, ColumnSemanticContract] = {}
        if c_df is not None and hasattr(c_df, "columns"):
            if dictionary:
                dict_map = {item.get("column_name", item.get("name")): item for item in dictionary if isinstance(item, dict)}
            else:
                from engine.profiler import infer_column_semantic_metadata
                dict_map = {c: infer_column_semantic_metadata(c, c_df[c], len(c_df)) for c in c_df.columns}
            n_total = len(c_df)
            for col in c_df.columns:
                d_item = dict_map.get(col, {})

                s = c_df[col]
                n_null = int(s.isnull().sum())
                n_valid = n_total - n_null
                null_p = round((n_null / max(n_total, 1)) * 100.0, 2)
                cov_p = round((n_valid / max(n_total, 1)) * 100.0, 2)
                n_uniq = int(s.nunique(dropna=True))

                sem_role = str(d_item.get("semantic_role", "DIMENSION")).upper()
                is_id = bool(d_item.get("is_identifier", False) or sem_role in ["IDENTIFIER", "ID"])
                is_meas = bool(d_item.get("is_measure", False) or sem_role in ["MEASURE", "METRIC", "NUMERIC"])
                is_time = bool(d_item.get("is_time", False) or sem_role in ["DATETIME", "DATE/TIME", "TIME", "DATE"])
                is_dim = not is_id and not is_meas and not is_time

                assigned_role = "IDENTIFIER" if is_id else ("MEASURE" if is_meas else ("DATETIME" if is_time else "DIMENSION"))
                analytical_eligible = bool(d_item.get("analytical_eligible", not is_id and not (n_uniq <= 1)))
                excl_reason = d_item.get("exclusion_reason")

                col_contract = ColumnSemanticContract(
                    column_name=str(col),
                    source_column=str(col),
                    physical_dtype=str(s.dtype),
                    semantic_role=assigned_role,
                    semantic_subtype=str(d_item.get("data_type", "Categorical")),
                    measure_type=d_item.get("measure_type"),
                    unit=d_item.get("unit"),
                    currency=d_item.get("currency"),
                    key_role="PRIMARY_KEY" if (is_id and n_uniq == n_total and n_total > 0) else "NONE",
                    analytical_eligible=analytical_eligible,
                    exclusion_reason=excl_reason,
                    grouping_eligible=bool(is_dim and 2 <= n_uniq <= 100),
                    time_series_eligible=bool(is_time),
                    correlation_eligible=bool(is_meas and analytical_eligible),
                    pca_eligible=bool(is_meas and analytical_eligible and cov_p >= 95.0),
                    non_null_count=n_valid,
                    missing_count=n_null,
                    missing_pct=null_p,
                    coverage_pct=cov_p,
                    unique_count=n_uniq
                )
                columns_dict[str(col)] = col_contract

        derivation_contracts = []
        if derivations:
            for d in derivations:
                if isinstance(d, dict):
                    derivation_contracts.append(DerivationContract(
                        derived_column=d.get("derived_column", ""),
                        source_columns=d.get("source_columns", []),
                        derivation_type=d.get("derivation_type", "COMPUTED_METRIC"),
                        formula=d.get("formula"),
                        confidence=float(d.get("confidence", 100.0)),
                        is_deterministic=bool(d.get("is_deterministic", False))
                    ))

        fname = os.path.basename(source_file_path) if source_file_path else f"{dataset_id}.xlsx"
        ext = os.path.splitext(fname)[1].lstrip('.').upper()
        s_format = ext if ext in ["EXCEL", "XLSX", "CSV", "PARQUET", "JSON"] else "EXCEL"

        return cls(
            dataset_id=dataset_id,
            source_file_name=fname,
            source_format=s_format,
            ingestion_timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            raw_row_count=raw_rows,
            raw_column_count=raw_cols,
            cleaned_row_count=clean_rows,
            analytical_row_count=clean_rows,
            raw_duplicate_count=raw_dups,
            cleaned_duplicate_count=clean_dups,
            columns=columns_dict,
            derivations=derivation_contracts,
            dataset_health_result=health_res,
            quality_result={"issues_detected": len(quality_issues or []), "cleaning_rules_applied": len(audit_trail or [])}
        )

