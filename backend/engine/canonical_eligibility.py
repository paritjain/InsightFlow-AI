import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from engine.contracts.enums import ExclusionReason, SemanticRole


class CanonicalEligibilityEngine:
    """
    Centralized Authoritative Analytical Eligibility Engine (Prompt Sections 11, 12, 13, 14).
    Single authority for all downstream engines (EDA, Correlation, PCA, ANOVA, Kruskal, Visualizations, Insights).
    
    Principles:
    1. Independent Eligibility Dimensions (Role does not change just because column is constant).
    2. Primary EDA Coverage Rule: >= 95% coverage (or <= 5% missing).
    3. Test Fixture Rule: Fixtures remain in raw quality ledger; blocked from all analytical engines.
    4. Zero Variance Rule: variance <= 1e-9 or nunique <= 1 blocked from SciPy/Statsmodels routines.
    5. Redundant Derived Variable Rule: Deterministic transformations excluded from primary PCA.
    """

    TEST_FIXTURE_PATTERNS = [
        r'^relationship_test_',
        r'^test_',
        r'^fixture_',
        r'^validation_',
        r'^synthetic_',
        r'^mock_',
        r'_test_fixture$',
        r'_test_col$',
        r'^tmp_test'
    ]

    @classmethod
    def is_likely_test_fixture(
        cls,
        col_name: str,
        series: pd.Series,
        total_rows: int,
        coverage_pct: float,
        non_null_count: int
    ) -> Tuple[bool, str]:
        """
        Detects whether a column is a test fixture (Section 13).
        """
        c_low = str(col_name).lower().strip()
        has_name_pattern = any(bool(re.search(pat, c_low)) for pat in cls.TEST_FIXTURE_PATTERNS)

        if has_name_pattern:
            return True, f"Matches test/fixture naming pattern ('{col_name}')"

        # Extreme isolation / sparsity with test keywords
        if total_rows >= 100 and non_null_count <= 50 and coverage_pct <= 25.0:
            if any(k in c_low for k in ['test', 'sample', 'dummy', 'mock', 'trial', 'temp', 'tmp', 'probe']):
                return True, f"Severe sparsity ({non_null_count}/{total_rows} rows) with fixture keywords"

        return False, ""

    @classmethod
    def compute_constancy(cls, series: pd.Series, tolerance: float = 1e-9) -> Dict[str, Any]:
        """
        Pre-validates series variance to prevent SciPy constant-input warnings (Section 14).
        """
        non_null = series.dropna()
        n = len(non_null)
        if n == 0:
            return {"is_constant": True, "variance": 0.0, "std": 0.0, "unique_count": 0, "min": None, "max": None}
        if n == 1:
            val = float(non_null.iloc[0]) if pd.api.types.is_numeric_dtype(non_null) else non_null.iloc[0]
            return {"is_constant": True, "variance": 0.0, "std": 0.0, "unique_count": 1, "min": val, "max": val}

        uniq_cnt = int(non_null.nunique())
        if uniq_cnt <= 1:
            return {"is_constant": True, "variance": 0.0, "std": 0.0, "unique_count": uniq_cnt, "min": non_null.iloc[0], "max": non_null.iloc[0]}

        if pd.api.types.is_numeric_dtype(non_null):
            s_num = pd.to_numeric(non_null, errors='coerce').dropna().astype(float)
            if len(s_num) <= 1 or s_num.nunique() <= 1:
                return {"is_constant": True, "variance": 0.0, "std": 0.0, "unique_count": len(s_num), "min": float(s_num.min()) if len(s_num) else None, "max": float(s_num.max()) if len(s_num) else None}
            min_v = float(s_num.min())
            max_v = float(s_num.max())
            if (max_v - min_v) <= tolerance:
                return {"is_constant": True, "variance": 0.0, "std": 0.0, "unique_count": uniq_cnt, "min": min_v, "max": max_v}

            var_v = float(s_num.var(ddof=1))
            if abs(var_v) <= tolerance:
                return {"is_constant": True, "variance": var_v, "std": 0.0, "unique_count": uniq_cnt, "min": min_v, "max": max_v}

            return {"is_constant": False, "variance": var_v, "std": float(np.sqrt(var_v)), "unique_count": uniq_cnt, "min": min_v, "max": max_v}

        return {"is_constant": False, "variance": None, "std": None, "unique_count": uniq_cnt, "min": None, "max": None}

    @classmethod
    def evaluate_column(
        cls,
        col_name: str,
        series: pd.Series,
        total_rows: Optional[int] = None,
        semantic_meta: Optional[Dict[str, Any]] = None,
        sparsity_threshold: float = 95.0
    ) -> Dict[str, Any]:
        """
        Authoritative evaluation of column eligibility across all analytical operations.
        """
        n_rows = total_rows if total_rows is not None else len(series)
        non_null_s = series.dropna()
        valid_cnt = len(non_null_s)
        missing_cnt = n_rows - valid_cnt
        missing_pct = round((missing_cnt / max(n_rows, 1)) * 100, 2)
        coverage_pct = round(100.0 - missing_pct, 2)
        unique_cnt = int(non_null_s.nunique())

        meta = semantic_meta or {}
        sem_role = str(meta.get("semantic_role", "")).upper()
        key_role = str(meta.get("key_role", "NONE")).upper()

        c_low = str(col_name).lower().strip()
        is_id_name = any(
            c_low.endswith(k) or c_low.startswith(f"{k}_") or f"_{k}_" in c_low or c_low == k
            for k in ['_id', 'id', 'uuid', 'vin', 'sku', 'guid', 'ssn', 'key', 'hash', 'token']
        )
        is_high_card = (unique_cnt == n_rows and n_rows >= 5) or (unique_cnt / max(n_rows, 1) >= 0.95 and not pd.api.types.is_numeric_dtype(series))

        if not sem_role or sem_role == "NONE":
            if is_id_name and is_high_card:
                key_role = "PRIMARY_KEY"
                sem_role = "IDENTIFIER"
            elif is_id_name:
                sem_role = "IDENTIFIER"
        elif key_role == "NONE" and is_id_name and unique_cnt == n_rows and n_rows >= 5:
            key_role = "PRIMARY_KEY"

        constancy = cls.compute_constancy(series)
        is_constant = constancy["is_constant"]
        variance_val = constancy.get("variance")


        is_num = (
            pd.api.types.is_numeric_dtype(series)
            or meta.get("is_measure", False)
            or sem_role in ["MEASURE", "NUMERIC MEASURE"]
        ) and not (
            sem_role in ["IDENTIFIER", "ID"]
            or key_role in ["PRIMARY_KEY", "PRIMARY KEY", "FOREIGN_KEY", "FOREIGN KEY"]
            or meta.get("is_identifier", False)
            or meta.get("id_safety") is True
        )

        is_fixture, fixture_reason = cls.is_likely_test_fixture(
            col_name, series, n_rows, coverage_pct, valid_cnt
        )

        analytical_eligible = True
        exclusion_reason = None

        # 1. Test Fixtures (Section 13)
        if is_fixture:
            analytical_eligible = False
            exclusion_reason = "TEST_FIXTURE"

        # 2. Identifiers / Keys (Section 6 & 7)
        elif key_role in ["PRIMARY_KEY", "PRIMARY KEY"] or meta.get("is_primary_key") or meta.get("primary_key"):
            analytical_eligible = False
            exclusion_reason = "PRIMARY_KEY"

        elif key_role in ["FOREIGN_KEY", "FOREIGN KEY"] or meta.get("is_foreign_key") or meta.get("foreign_key"):
            analytical_eligible = False
            exclusion_reason = "FOREIGN_KEY"

        elif (
            "IDENTIFIER" in sem_role
            or "ID" in sem_role
            or meta.get("is_identifier", False)
            or meta.get("id_safety") is True
            or key_role in ["CANDIDATE_PRIMARY_KEY", "CANDIDATE_FOREIGN_KEY", "CANDIDATE_FK"]
        ):
            analytical_eligible = False
            exclusion_reason = "IDENTIFIER"

        # 3. Constant / Zero-Variance (Section 14)
        elif is_constant or unique_cnt <= 1:
            analytical_eligible = False
            exclusion_reason = "CONSTANT"

        elif variance_val is not None and abs(variance_val) <= 1e-9:
            analytical_eligible = False
            exclusion_reason = "NEAR_ZERO_VARIANCE"

        # 4. Sparsity / Low Coverage (Section 12: >= 95% coverage required for primary quantitative analysis)
        elif coverage_pct < sparsity_threshold or missing_pct > (100.0 - sparsity_threshold):
            analytical_eligible = False
            exclusion_reason = "ANALYTICAL_COVERAGE_BELOW_THRESHOLD"

        # 5. Insufficient Sample
        elif valid_cnt < 5:
            analytical_eligible = False
            exclusion_reason = "INSUFFICIENT_SAMPLE"

        # 6. Dimensions and Categoricals
        elif not is_num or "DIMENSION" in sem_role or "TEXT" in sem_role or "DATE" in sem_role:
            analytical_eligible = True
            exclusion_reason = None

        # Compute specific operation eligibilities
        is_dim = ("DIMENSION" in sem_role or "CATEGORICAL" in sem_role or (not is_num and not (sem_role in ["IDENTIFIER", "ID"])))
        is_time = ("DATE" in sem_role or "TIME" in sem_role or "TEMPORAL" in sem_role)
        is_id = (
            exclusion_reason in ["PRIMARY_KEY", "FOREIGN_KEY", "IDENTIFIER", "CANDIDATE_IDENTIFIER"]
            or sem_role in ["IDENTIFIER", "ID"]
            or meta.get("is_identifier", False)
        )

        grouping_eligible = bool((is_dim or is_time) and not is_id and not is_fixture and unique_cnt >= 2)
        time_series_eligible = bool(is_time) and not is_id and not is_fixture

        corr_eligible = bool(analytical_eligible and is_num and not is_constant and not is_id and not is_fixture and coverage_pct >= sparsity_threshold)
        pca_eligible = bool(analytical_eligible and is_num and not is_constant and not is_id and not is_fixture and coverage_pct >= sparsity_threshold and valid_cnt >= 5)
        anova_eligible = bool(analytical_eligible and is_num and not is_constant and not is_id and not is_fixture)
        kruskal_eligible = bool(analytical_eligible and is_num and not is_constant and not is_id and not is_fixture)
        viz_eligible = bool(not is_fixture and not is_id)

        return {
            "column_name": col_name,
            "analytical_eligible": analytical_eligible,
            "exclusion_reason": exclusion_reason,
            "grouping_eligible": grouping_eligible,
            "time_series_eligible": time_series_eligible,
            "correlation_eligible": corr_eligible,
            "pca_eligible": pca_eligible,
            "anova_eligible": anova_eligible,
            "kruskal_eligible": kruskal_eligible,
            "visualization_eligible": viz_eligible,
            "non_null_count": valid_cnt,
            "missing_count": missing_cnt,
            "missing_pct": missing_pct,
            "coverage_pct": coverage_pct,
            "unique_count": unique_cnt,
            "is_constant": is_constant,
            "variance": variance_val,
            "constancy": constancy
        }
