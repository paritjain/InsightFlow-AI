"""
Regression and Governance Test Suite for InsightFlow AI
Master Data Dictionary + Analytical Eligibility + EDA Governance Fix
Covers PART 20: TEST A through TEST P
"""

import pytest
import pandas as pd
import numpy as np

from engine.governed_engines import (
    KeyDetectionEngine,
    AnalyticalEligibilityEngine,
    DualCompatibleStr,
    DualCompatibleIdSafety,
)
from engine.profiler import infer_column_semantic_metadata
from engine.dictionary import generate_data_dictionary
from services.statistical_analysis import (
    compute_pca,
    compute_bivariate_hypothesis_tests,
)
from engine.analytical_truth import (
    safe_correlation,
    compute_grouped_analytical_result,
    AnalyticalResult,
)
from engine.action_plan import generate_action_plan


# ==============================================================================
# TEST A — PRIMARY KEY
# ==============================================================================
def test_test_a_primary_key_detection():
    """
    100% unique, zero-null Observation_ID:
        Expected: Primary Key, Identifier, ID Safety = true, quantitative analysis blocked
    Duplicate Observation_ID:
        Expected: NOT Primary Key
    """
    n = 1200
    # Case 1: 100% unique, 0 nulls
    s_unique = pd.Series([f"OBS_{i:04d}" for i in range(n)], name="Observation_ID")
    meta_unique = infer_column_semantic_metadata("Observation_ID", s_unique, n)

    assert "Primary Key" in str(meta_unique["key_role"])
    assert "Identifier" in str(meta_unique["semantic_role"])
    assert meta_unique["id_safety"] == True
    assert bool(meta_unique["id_safety"]) is True
    # Quantitative analysis blocked
    elig = AnalyticalEligibilityEngine.evaluate_column(
        "Observation_ID", s_unique, n, semantic_meta=meta_unique
    )
    assert elig["analytical_eligible"] is False
    assert elig["exclusion_reason"] in ["PRIMARY_KEY", "IDENTIFIER"]

    # Case 2: Duplicate Observation_ID
    dup_vals = [f"OBS_{i:04d}" for i in range(n - 10)] + [f"OBS_0000" for _ in range(10)]
    s_dup = pd.Series(dup_vals, name="Observation_ID")
    meta_dup = infer_column_semantic_metadata("Observation_ID", s_dup, n)

    assert meta_dup["key_role"] != "Primary Key"
    assert "Candidate" in str(meta_dup["key_role"]) or meta_dup["key_role"] == "None"


# ==============================================================================
# TEST B — FOREIGN KEY
# ==============================================================================
def test_test_b_foreign_key_referential_evidence():
    """
    Valid child-parent relationship:
        Expected: Foreign Key, relationship verified
    Unverified relationship:
        Expected: Candidate Foreign Key, relationship unverified
    """
    n = 500
    # Case 1: Verified child-parent relationship
    parent_df = pd.DataFrame({
        "Observer_ID": [f"OBSERVER_{i:03d}" for i in range(100)],
        "Observer_Name": [f"Observer {i}" for i in range(100)],
    })
    child_df = pd.DataFrame({
        "Observation_ID": [f"REC_{i}" for i in range(n)],
        "Observer_ID": [f"OBSERVER_{(i % 100):03d}" for i in range(n)],
    })
    all_tables = {"observations": child_df, "observers": parent_df}

    det_verified = KeyDetectionEngine.detect_key(
        "Observer_ID",
        child_df["Observer_ID"],
        n,
        table_name="observations",
        all_tables=all_tables,
    )
    assert "Foreign Key" in str(det_verified["key_role"])
    assert det_verified["relationship_status"] == "VERIFIED"
    assert det_verified["referential_coverage_pct"] >= 90.0

    # Case 2: Unverified relationship (no parent table available)
    det_unverified = KeyDetectionEngine.detect_key(
        "Observer_ID", child_df["Observer_ID"], n, table_name="observations", all_tables=None
    )
    assert "Candidate Foreign Key" in str(det_unverified["key_role"])
    assert det_unverified["relationship_status"] == "UNVERIFIED"


# ==============================================================================
# TEST C — SPARSE VARIABLE
# ==============================================================================
def test_test_c_sparse_variable_exclusion():
    """
    1,200 total rows, 20 populated, 1,180 missing:
        Expected: missing_pct ≈ 98.3%, coverage_pct ≈ 1.7%,
                  analytical_eligible = false, exclusion_reason = SPARSE or TEST_FIXTURE
    """
    n = 1200
    vals = [float(i) for i in range(20)] + [np.nan] * 1180
    s_sparse = pd.Series(vals, name="Auxiliary_Metric")

    elig = AnalyticalEligibilityEngine.evaluate_column("Auxiliary_Metric", s_sparse, n)
    assert round(elig["missing_pct"], 1) == 98.3
    assert round(elig["coverage_pct"], 1) == 1.7
    assert elig["analytical_eligible"] is False
    assert elig["exclusion_reason"] in ["SPARSE", "TEST_FIXTURE"]


# ==============================================================================
# TEST D — TEST FIXTURE
# ==============================================================================
def test_test_d_test_fixture_detection():
    """
    Relationship_Test_* with sparse population:
        Expected: TEST_FIXTURE, excluded from primary EDA
    """
    n = 1200
    vals = [3.0] * 20 + [np.nan] * 1180
    s_fixture = pd.Series(vals, name="Relationship_Test_Severity_Rating")

    elig = AnalyticalEligibilityEngine.evaluate_column(
        "Relationship_Test_Severity_Rating", s_fixture, n
    )
    assert elig["analytical_eligible"] is False
    assert elig["exclusion_reason"] == "TEST_FIXTURE"


# ==============================================================================
# TEST E — CONSTANT VARIABLE
# ==============================================================================
def test_test_e_constant_variable_exclusion():
    """
    1,200 values, all equal:
        Expected: variance = 0, excluded from PCA, excluded from correlation
    """
    n = 1200
    s_const = pd.Series([1.0] * n, name="Observation_Count")
    s_norm = pd.Series(np.random.normal(10, 2, n), name="Normal_Metric")
    df = pd.DataFrame({"Observation_Count": s_const, "Normal_Metric": s_norm})

    elig = AnalyticalEligibilityEngine.evaluate_column("Observation_Count", s_const, n)
    assert elig["analytical_eligible"] is False
    assert elig["exclusion_reason"] in ["CONSTANT", "NEAR_ZERO_VARIANCE"]

    # Excluded from correlation
    corr_res = safe_correlation(df, "Observation_Count", "Normal_Metric")
    assert corr_res["correlation_status"] in ["excluded", "not_applicable", "zero_variance"]

    # Excluded from PCA
    df_pca = pd.DataFrame({
        "Observation_Count": s_const,
        "Feat_A": np.random.normal(5, 1, n),
        "Feat_B": np.random.normal(20, 3, n),
    })
    pca_res = compute_pca(df_pca)
    assert "Observation_Count" in pca_res.get("excluded_features", [])
    assert pca_res.get("exclusion_reasons", {}).get("Observation_Count") in [
        "zero_variance", "CONSTANT", "NEAR_ZERO_VARIANCE", "analytical_ineligible"
    ]


# ==============================================================================
# TEST F — VALID NUMERIC VARIABLE
# ==============================================================================
def test_test_f_valid_numeric_variable_retained():
    """
    1,200 rows with normal coverage and non-zero variance:
        Expected: retained (analytical_eligible = true)
    """
    np.random.seed(42)
    n = 1200
    s_valid = pd.Series(np.random.exponential(50.0, n), name="Process_Throughput")

    elig = AnalyticalEligibilityEngine.evaluate_column("Process_Throughput", s_valid, n)
    assert elig["analytical_eligible"] is True
    assert elig["exclusion_reason"] is None
    assert elig["coverage_pct"] == 100.0


# ==============================================================================
# TEST G — RATING SEMANTICS
# ==============================================================================
def test_test_g_rating_measure_semantics():
    """
    Severity_Rating values 1–5:
        Expected: Measure, Rating, scale 1–5, AVG/MEDIAN/MIN/MAX allowed, SUM blocked by default
    """
    np.random.seed(42)
    n = 1200
    s_rating = pd.Series(np.random.choice([1, 2, 3, 4, 5], size=n), name="Severity_Rating")
    meta = infer_column_semantic_metadata("Severity_Rating", s_rating, n)

    assert "Measure" in str(meta["semantic_role"])
    assert "Rating" in str(meta["semantic_subtype"])
    assert meta.get("scale") == "1–5" or meta.get("rating_scale") == "1–5"
    assert "SUM" not in meta["allowed_aggregations"]
    assert "AVG" in meta["allowed_aggregations"]
    assert "MEDIAN" in meta["allowed_aggregations"]
    assert "MIN" in meta["allowed_aggregations"]
    assert "MAX" in meta["allowed_aggregations"]


# ==============================================================================
# TEST H — PERCENTAGE SEMANTICS
# ==============================================================================
def test_test_h_percentage_measure_semantics():
    """
    Action_Completion_Pct:
        Expected: Percentage, unit %, AVG/MEDIAN/MIN/MAX allowed, SUM blocked
    """
    np.random.seed(42)
    n = 1200
    s_pct = pd.Series(np.random.uniform(0.0, 100.0, size=n), name="Action_Completion_Pct")
    meta = infer_column_semantic_metadata("Action_Completion_Pct", s_pct, n)

    assert "Percentage" in str(meta["semantic_subtype"])
    assert meta["unit"] == "%"
    assert "SUM" not in meta["allowed_aggregations"]
    assert "AVG" in meta["allowed_aggregations"]
    assert "MEDIAN" in meta["allowed_aggregations"]


# ==============================================================================
# TEST I — DURATION SEMANTICS
# ==============================================================================
def test_test_i_duration_measure_semantics():
    """
    Exposure_Hours: Duration, unit hours
    Action_Closure_Days: Duration, unit days
    """
    np.random.seed(42)
    n = 1200
    s_hours = pd.Series(np.random.uniform(10.0, 500.0, size=n), name="Exposure_Hours")
    meta_hours = infer_column_semantic_metadata("Exposure_Hours", s_hours, n)
    assert "Duration" in str(meta_hours["semantic_subtype"])
    assert meta_hours["unit"] == "hours"

    s_days = pd.Series(np.random.randint(1, 60, size=n), name="Action_Closure_Days")
    meta_days = infer_column_semantic_metadata("Action_Closure_Days", s_days, n)
    assert "Duration" in str(meta_days["semantic_subtype"])
    assert meta_days["unit"] == "days"


# ==============================================================================
# TEST J — CORRELATION
# ==============================================================================
def test_test_j_correlation_output():
    """
    Expected output must contain: Pearson, Spearman, valid_n, coverage_pct
    """
    np.random.seed(42)
    n = 1200
    x = np.random.normal(50, 10, n)
    y = 0.8 * x + np.random.normal(0, 5, n)
    df = pd.DataFrame({"Metric_X": x, "Metric_Y": y})

    res = safe_correlation(df, "Metric_X", "Metric_Y")
    assert "pearson" in res or "pearson_r" in res
    assert "spearman" in res or "spearman_rho" in res
    assert "valid_n" in res
    assert "coverage_pct" in res
    assert res["valid_n"] == 1200
    assert res["coverage_pct"] == 100.0


# ==============================================================================
# TEST K — PCA
# ==============================================================================
def test_test_k_pca_governance_and_effective_n():
    """
    1,200-row dataset:
        Expected: PCA effective N is explicitly reported,
                  sparse variables excluded, identifiers excluded, constants excluded.
        Never silently produce 19 observations from sparse test variables.
    """
    np.random.seed(42)
    n = 1200
    df = pd.DataFrame({
        "Observation_ID": [f"OBS_{i:04d}" for i in range(n)],
        "Feat_1": np.random.normal(10, 2, n),
        "Feat_2": np.random.normal(20, 5, n),
        "Feat_3": np.random.normal(30, 8, n),
        "Sparse_Fixture": [5.0] * 19 + [np.nan] * (n - 19),  # 19 populated rows
        "Constant_Feat": [1.0] * n,
    })

    pca_res = compute_pca(df)
    assert pca_res["is_valid"] is True
    assert pca_res["total_dataset_records"] == 1200
    assert pca_res["effective_observations"] == 1200
    assert pca_res["effective_observation_pct"] == 100.0

    # Ensure sparse fixture, identifier, and constant were excluded
    assert "Observation_ID" in pca_res["excluded_features"]
    assert "Sparse_Fixture" in pca_res["excluded_features"]
    assert "Constant_Feat" in pca_res["excluded_features"]
    assert len(pca_res["eligible_features"]) == 3


# ==============================================================================
# TEST L — VISUALIZATION DENOMINATOR
# ==============================================================================
def test_test_l_visualization_denominator():
    """
    If total = 1,200 and filtered = 132:
    Expected: valid / 1,200 (NOT 1,000)
    """
    from engine.analytical_truth import compute_grouped_analytical_result

    n = 1200
    df = pd.DataFrame({
        "Category": ["Cat_A"] * 132 + ["Cat_B"] * (n - 132),
        "Metric": [10.0] * n,
    })
    res = compute_grouped_analytical_result(
        df, "Metric", "Category", aggregation="SUM", dataset_id="ds_1200"
    )

    d = res.to_dict()
    assert d["total_n"] == 1200
    assert d["total_record_count"] == 1200
    # Lineage and denominator reflect 1,200 total records
    assert d["lineage"]["source_dataset"] == "ds_1200"


# ==============================================================================
# TEST M — DENOMINATOR PERCENTAGE SEMANTICS
# ==============================================================================
def test_test_m_percentage_semantics():
    """
    85 / 3,497:
        Expected: measure_contribution ≈ 2.4%, NOT record_share
    """
    from engine.analytical_truth import compute_grouped_analytical_result

    # 100 rows where Observer_054 has total 85 out of 3497
    observers = ["Observer_054"] * 17 + ["Other_Observer"] * 83
    severity = [5.0] * 17 + [41.1084337] * 83  # 17 * 5 = 85, rest = 3412, total = 3497
    df = pd.DataFrame({"Observer": observers, "Severity_Rating": severity})

    res = compute_grouped_analytical_result(
        df, "Severity_Rating", "Observer", aggregation="SUM", dataset_id="ds_safety"
    )
    d = res.to_dict()

    assert d["statistics"]["percentage_type"] == "measure_contribution"
    # Verify evidence label is 'Severity_Rating Contribution'
    pct_ev = next(e for e in d["evidence"] if "Contribution" in e["metric"])
    assert pct_ev["metric"] == "Severity_Rating Contribution"
    assert "Share of Valid" not in pct_ev["metric"]


# ==============================================================================
# TEST N — RECOMMENDATION DEDUPLICATION
# ==============================================================================
def test_test_n_recommendation_deduplication():
    """
    Same analytical result generates same recommendation multiple times:
        Expected: exactly one recommendation
    """
    # Create duplicate insights pointing to the same analytical result
    duplicate_insights = [
        {
            "id": f"ins_{i}",
            "insight_id": f"ins_{i}",
            "headline": "Exposure_Hours shows strong skewness (3.04)",
            "fact": "Skewness is 3.04 with mean 266.37 and median 235.85",
            "interpretation": "Median is more robust than arithmetic mean",
            "category": "Variance & Anomaly",
            "action": "Adopt Median Benchmarks and Robust IQR Fences",
            "evidence": [{"metric": "Skewness", "value": "3.04"}],
            "shared_result": {
                "result_id": "AR-100001",
                "source_columns": ["Exposure_Hours"],
                "source_table": "Safety",
            },
            "lineage": {"analytical_result_id": "AR-100001"},
        }
        for i in range(5)  # 5 duplicates
    ]

    action_plan = generate_action_plan(duplicate_insights)
    recs = action_plan["recommendations"]

    # Filter recommendations for AR-100001
    matching_recs = [r for r in recs if r.get("analytical_result_id") == "AR-100001"]
    assert len(matching_recs) == 1


# ==============================================================================
# TEST O — ACTION DEDUPLICATION
# ==============================================================================
def test_test_o_action_deduplication():
    """
    Same recommendation generates duplicate action records:
        Expected: exactly one action item
    """
    duplicate_insights = [
        {
            "id": f"ins_{i}",
            "insight_id": f"ins_{i}",
            "headline": "Hazard_Category high concentration",
            "fact": "Unsafe Act accounts for 45% of observations",
            "interpretation": "Concentration in Unsafe Act requires mitigation",
            "category": "Concentration Risk",
            "action": "Audit Segment Contribution Baseline",
            "evidence": [{"metric": "Concentration", "value": "45%"}],
            "shared_result": {
                "result_id": "AR-200002",
                "source_columns": ["Hazard_Category"],
                "source_table": "Safety",
            },
            "lineage": {"analytical_result_id": "AR-200002"},
        }
        for i in range(4)
    ]

    action_plan = generate_action_plan(duplicate_insights)
    actions = action_plan["action_items"]

    matching_actions = [a for a in actions if a.get("analytical_result_id") == "AR-200002"]
    assert len(matching_actions) == 1


# ==============================================================================
# TEST P — LINEAGE PRESERVATION
# ==============================================================================
def test_test_p_lineage_preservation():
    """
    Every analytical result must preserve:
        dataset_id, dataset_version, source_columns, filters, valid_n, excluded_n, analytical eligibility
    """
    df = pd.DataFrame({
        "Dept": ["Ops", "Safety", "HR", "Ops", "Safety"],
        "Days": [12, 15, 20, 10, 14],
    })

    res = compute_grouped_analytical_result(
        df, "Days", "Dept", aggregation="MEAN", dataset_id="ds_corp_01", dataset_version="2.1"
    )
    d = res.to_dict()

    assert d["dataset_id"] == "ds_corp_01"
    assert d["dataset_version"] == "2.1"
    assert "Dept" in d["source_columns"] and "Days" in d["source_columns"]
    assert d["filters"] is not None
    assert d["valid_n"] == 5
    assert d["excluded_n"] == 0
    assert d["coverage_pct"] == 100.0
    assert d["lineage"]["source_dataset"] == "ds_corp_01"
    assert d["lineage"]["cleaned_dataset_version"] == "2.1"
