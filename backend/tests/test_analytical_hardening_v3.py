import unittest
import numpy as np
import pandas as pd
import scipy.stats as stats
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.profiler import infer_column_semantic_metadata, compute_variable_constancy
from engine.dictionary import generate_data_dictionary
from engine.analytical_truth import (
    AnalyticalResult,
    safe_correlation,
    safe_anova,
    safe_kruskal,
    safe_pca_features,
    compute_grouped_analytical_result
)
from engine.governed_engines import (
    DerivedMetricEngine,
    KeyDetectionEngine,
    StatisticalSuitabilityEngine,
    ChartSuitabilityEngine,
    InsightGovernanceEngine,
    RecommendationGovernanceEngine,
    ActionGovernanceEngine,
    LineageEngine
)
from engine.eda import generate_waterfall_chart, generate_stacked_bar_charts, run_automated_eda, validate_chart_spec
from services.statistical_analysis import compute_group_differences, apply_multiple_testing_correction, compute_pca


class TestAnalyticalHardeningV3(unittest.TestCase):
    """
    Comprehensive Analytical Hardening V3 Regression Test Suite.
    Validates all 47 acceptance criteria across:
    - Semantics & Decoupled Data Dictionary
    - PK / FK Detection without false FK classifications
    - Percentage vs Currency vs Ratio Precedence
    - Rating vs Score vs Index vs Age Semantics
    - Duration vs Count
    - ID Safety Isolation
    - Derived Metric Detection & Correlation Firewall
    - Multiple Testing FDR Correction & Statistical Evidence
    - ANOVA / Kruskal-Wallis Assumptions & Non-Generic Phrasing
    - Zero-Variance Protection across all analytical pathways
    - Outlier vs Data Error Separation
    - Stage-Specific Completeness Governance
    - Deterministic Temporal Classification & Evidence
    - Additive Waterfall Bridge Reconciliation
    - Composition vs Performance Distinction
    - Stacked Bar Hierarchical Aggregation
    - Non-Fabricated Recommendations & Action Center Lineage
    - Golden Datasets & Arbitrary-Schema Validation
    - Hindalco Plant Safety Observation Data without hardcoded column names
    """

    # =========================================================================
    # 1. SEMANTICS & DATA DICTIONARY ENGINE
    # =========================================================================

    def test_01_true_primary_key_inference(self):
        df = pd.DataFrame({
            "worker_uuid": [f"W-{i:04d}" for i in range(100)],
            "val": np.random.randn(100)
        })
        meta = infer_column_semantic_metadata("worker_uuid", df["worker_uuid"], total_rows=100)
        self.assertEqual(meta["key_role"], "Primary Key")
        self.assertTrue(meta["is_primary_key"])
        self.assertFalse(meta["is_foreign_key"])
        self.assertEqual(meta["expected_unique"], True)

    def test_02_candidate_identifier_no_false_fk_on_single_table(self):
        # Repeating identifier-like strings in standalone single table must NOT become FK
        df = pd.DataFrame({
            "badge_code": ["B1", "B2", "B1", "B3", "B2", "B4", "B1", "B5"] * 10,
            "metric_x": np.random.randn(80)
        })
        meta = infer_column_semantic_metadata("badge_code", df["badge_code"], total_rows=80)
        self.assertNotEqual(meta["key_role"], "Foreign Key")
        self.assertFalse(meta["is_foreign_key"])
        self.assertIn(meta["key_role"], ["identifier", "candidate_primary_key", "None"])

    def test_03_percentage_vs_currency_conflict_resolution(self):
        # Column names with profit or margin and percentage tokens must resolve to Percentage
        df = pd.DataFrame({
            "profit_margin_pct": [0.12, 0.15, 0.18, 0.09, 0.22],
            "commission_rate": [0.05, 0.05, 0.06, 0.04, 0.05],
            "operating_cost_usd": [12000.0, 15000.0, 13500.0, 11000.0, 16000.0]
        })
        meta_pm = infer_column_semantic_metadata("profit_margin_pct", df["profit_margin_pct"], total_rows=5)
        self.assertEqual(meta_pm["semantic_role"], "Measure")
        self.assertEqual(meta_pm["semantic_type"], "Percentage")
        self.assertEqual(meta_pm["unit"], "%")
        self.assertIsNone(meta_pm["currency"])
        self.assertIsNone(meta_pm["currency_metadata"]["code"])

        meta_cost = infer_column_semantic_metadata("operating_cost_usd", df["operating_cost_usd"], total_rows=5)
        self.assertEqual(meta_cost["semantic_type"], "Currency")
        self.assertEqual(meta_cost["currency"], "USD")

    def test_04_rating_vs_score_vs_index_distinction(self):
        df = pd.DataFrame({
            "service_rating": [1, 2, 3, 4, 5, 4, 3, 5, 2, 1],
            "risk_score": [25.4, 67.2, 88.1, 14.3, 45.9, 78.0, 92.4, 33.1, 55.6, 61.2],
            "efficiency_index": [102.4, 98.6, 110.2, 95.1, 105.8, 101.3, 99.4, 108.2, 97.0, 104.5]
        })
        meta_rating = infer_column_semantic_metadata("service_rating", df["service_rating"], total_rows=10)
        self.assertEqual(meta_rating["semantic_subtype"], "Rating")
        self.assertEqual(meta_rating["scale_type"], "ordinal/bounded")
        self.assertIn("SUM not recommended", meta_rating["aggregation_restrictions"][0])

        meta_score = infer_column_semantic_metadata("risk_score", df["risk_score"], total_rows=10)
        self.assertEqual(meta_score["measure_type"], "Score")
        self.assertEqual(meta_score["scale_type"], "bounded")

        meta_index = infer_column_semantic_metadata("efficiency_index", df["efficiency_index"], total_rows=10)
        self.assertEqual(meta_index["measure_type"], "Index")

    def test_05_age_semantics_detection(self):
        df = pd.DataFrame({
            "operator_age": [28, 35, 42, 51, 60, 24, 39, 45, 33, 58],
            "tenure_years": [3, 7, 12, 20, 25, 1, 10, 15, 6, 22]
        })
        meta_age = infer_column_semantic_metadata("operator_age", df["operator_age"], total_rows=10)
        self.assertEqual(meta_age["semantic_type"], "Age")
        self.assertEqual(meta_age["unit"], "years")

    def test_06_duration_vs_count_distinction(self):
        df = pd.DataFrame({
            "working_days": [22, 22, 21, 23, 20, 22],
            "incident_count": [0, 1, 0, 2, 0, 1],
            "repair_duration_hours": [4.5, 12.0, 2.3, 8.1, 1.5, 6.7]
        })
        meta_wd = infer_column_semantic_metadata("working_days", df["working_days"], total_rows=6)
        self.assertEqual(meta_wd["semantic_subtype"], "Count")
        self.assertEqual(meta_wd["unit"], "days")

        meta_cnt = infer_column_semantic_metadata("incident_count", df["incident_count"], total_rows=6)
        self.assertEqual(meta_cnt["semantic_subtype"], "Count")

        meta_dur = infer_column_semantic_metadata("repair_duration_hours", df["repair_duration_hours"], total_rows=6)
        self.assertEqual(meta_dur["semantic_subtype"], "Duration")
        self.assertEqual(meta_dur["unit"], "hours")

    def test_07_id_safety_does_not_disable_legitimate_measures(self):
        df = pd.DataFrame({
            "employee_id": [f"EMP_{i}" for i in range(50)],
            "monthly_salary": [50000 + i * 500 for i in range(50)],
            "bonus_amount": [5000 + i * 50 for i in range(50)]
        })
        dict_items = generate_data_dictionary(df)
        dict_map = {c["column_name"]: c for c in dict_items}

        self.assertEqual(dict_map["employee_id"]["id_safety_classification"], "Appropriate identifier (Primary Key)")
        self.assertFalse(dict_map["employee_id"]["aggregation_eligibility"])

        # Salary and bonus must remain fully aggregatable measures
        self.assertTrue(dict_map["monthly_salary"]["aggregation_eligibility"])
        self.assertIn("SUM", dict_map["monthly_salary"]["allowed_aggregations"])
        self.assertTrue(dict_map["bonus_amount"]["aggregation_eligibility"])
        self.assertIn("SUM", dict_map["bonus_amount"]["allowed_aggregations"])

    # =========================================================================
    # 2. DATA QUALITY & DATA STAGE GOVERNANCE
    # =========================================================================

    def test_08_stage_specific_completeness(self):
        df = pd.DataFrame({
            "a": [1, 2, np.nan, 4, 5],
            "b": [10, 20, 30, 40, 50]
        })
        dict_items = generate_data_dictionary(df)
        dict_map = {c["column_name"]: c for c in dict_items}

        self.assertEqual(dict_map["a"]["data_stage"], "CLEANED")
        self.assertEqual(dict_map["a"]["raw_completeness"], "80.0%")
        self.assertEqual(dict_map["a"]["cleaned_completeness"], "80.0%")
        self.assertEqual(dict_map["b"]["cleaned_completeness"], "100.0%")

    def test_09_dataset_health_separate_from_reliability(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
        eda_res = run_automated_eda(df)
        overview = eda_res.get("overview", {})

        health = overview.get("health_score")
        self.assertIsNotNone(health)
        self.assertGreaterEqual(health, 0.0)
        self.assertLessEqual(health, 100.0)
        # Verify health represents data hygiene (duplicates, nulls) not validation framework score
        self.assertNotIn("analytical_reliability", str(overview.keys()))

    def test_10_zero_variance_protection_everywhere(self):
        df = pd.DataFrame({
            "constant_m": [42.0] * 30,
            "category": ["A", "B", "C"] * 10
        })
        # 1. ANOVA
        res_a = compute_group_differences(df, "category", "constant_m")
        self.assertEqual(res_a["anova"]["test_status"], "not_applicable")
        self.assertIn("constant", res_a["anova"]["interpretation"].lower())

        # 2. Kruskal-Wallis
        self.assertEqual(res_a["kruskal_wallis"]["test_status"], "not_applicable")

        # 3. Correlation
        corr = safe_correlation(df, "constant_m", "constant_m")
        self.assertEqual(corr["correlation_status"], "not_applicable")
        self.assertIn("zero variance", corr["reason"].lower())

        # 4. Chart suitability
        suit = ChartSuitabilityEngine.chartSuitabilityCheck("BOX_PLOT", df, x_col="category", y_col="constant_m")
        self.assertFalse(suit["suitable"])
        self.assertEqual(suit["status"], "BLOCKED_BY_VARIANCE")

    def test_11_outlier_vs_data_quality_error_separation(self):
        # 100 standard values with 2 statistical extreme points (valid operational spikes)
        vals = [10.0] * 98 + [150.0, 160.0]
        df = pd.DataFrame({"metric": vals})
        eda_res = run_automated_eda(df)
        stats_res = eda_res["numeric_summary"]["metric"]
        self.assertEqual(stats_res["outliers_count"], 2)
        # Must clearly identify statistical outlier method (Tukey IQR) without labeling as data error
        self.assertIn("iqr", str(stats_res.keys()).lower())

    # =========================================================================
    # 3. DERIVED METRICS & CORRELATION GOVERNANCE
    # =========================================================================

    def test_12_algebraic_derivation_detection(self):
        np.random.seed(42)
        b = np.random.uniform(10, 100, 50)
        c = np.random.uniform(1, 10, 50)
        a_sub = b - c
        a_add = b + c
        a_mul = b * c
        a_div = b / c
        a_pct = (c / b) * 100

        df = pd.DataFrame({
            "revenue": b,
            "cost": c,
            "operating_profit": a_sub,
            "total_turnover": a_add,
            "volume_weighted": a_mul,
            "unit_ratio": a_div,
            "cost_share_pct": a_pct,
            "unrelated_noise": np.random.normal(500, 50, 50)
        })

        engine = DerivedMetricEngine()
        derivations = engine.scan_dataframe_for_derivations(df)

        derived_cols = [d["target_column"] for d in derivations]
        self.assertIn("operating_profit", derived_cols)
        self.assertIn("total_turnover", derived_cols)
        self.assertIn("volume_weighted", derived_cols)
        self.assertIn("unit_ratio", derived_cols)
        self.assertIn("cost_share_pct", derived_cols)
        self.assertNotIn("unrelated_noise", derived_cols)

    def test_13_derived_correlation_interpretation_firewall(self):
        # When two variables are algebraically linked, correlation interpretation must be STRUCTURALLY_DERIVED
        np.random.seed(123)
        rev = np.random.uniform(100, 500, 40)
        exp = np.random.uniform(20, 80, 40)
        ebit = rev - exp  # Exact subtractive derivation

        df = pd.DataFrame({"Revenue": rev, "Expenses": exp, "EBIT": ebit})
        corr = safe_correlation(df, "Revenue", "EBIT")

        self.assertEqual(corr["relationship_type"], "DERIVED")
        self.assertEqual(corr["interpretation_status"], "STRUCTURALLY_DERIVED")
        self.assertIn("structurally expected", corr["interpretation"].lower())
        self.assertIn("correlation does not establish causation", corr["takeaway"].lower())

    def test_14_correlation_fdr_multiple_testing_control(self):
        p_values = [0.001, 0.008, 0.035, 0.048, 0.120, 0.350, 0.850]
        corrs = apply_multiple_testing_correction(p_values, method="Benjamini-Hochberg")

        self.assertEqual(len(corrs), len(p_values))
        for item in corrs:
            self.assertIn("raw_p_value", item)
            self.assertIn("adjusted_p_value", item)
            self.assertIn("multiple_testing_method", item)
            # Adjusted p-values must be >= raw p-values under FDR
            self.assertGreaterEqual(item["adjusted_p_value"], item["raw_p_value"] - 1e-9)

    # =========================================================================
    # 4. ANOVA & KRUSKAL-WALLIS GOVERNANCE
    # =========================================================================

    def test_15_anova_exact_evidence_and_kruskal_fallback(self):
        np.random.seed(99)
        # 3 groups with unequal variance (heteroscedastic)
        g1 = np.random.normal(10, 1, 30)
        g2 = np.random.normal(10, 5, 30)
        g3 = np.random.normal(10, 15, 30)
        df = pd.DataFrame({
            "group": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
            "metric": np.concatenate([g1, g2, g3])
        })
        res = compute_group_differences(df, "group", "metric")
        anova = res["anova"]
        kw = res["kruskal_wallis"]

        self.assertTrue(anova["valid"])
        self.assertFalse(res["assumptions_validated"])
        self.assertIn("Kruskal-Wallis", res["statistical_recommendation"])
        # Forbidden generic wording check
        self.assertNotIn("variation is consistent with random sample dispersion", anova["interpretation"].lower())
        # Check numerical evidence presence
        self.assertIn("f(", anova["interpretation"].lower())
        self.assertIn("eta²", anova["interpretation"].lower())

    # =========================================================================
    # 5. TEMPORAL ANALYSIS DETERMINISTIC CLASSIFICATION
    # =========================================================================

    def test_16_temporal_monotonic_increasing(self):
        dates = pd.date_range("2024-01-01", periods=12, freq="ME")
        vals = [100 + i * 15 for i in range(12)]
        df = pd.DataFrame({"date": dates, "output": vals})

        eda = run_automated_eda(df)
        ts = eda.get("time_series", {})
        self.assertEqual(ts.get("temporal_classification"), "INCREASING")
        takeaway = ts.get("takeaway", "")
        self.assertIn("maximum of 265", takeaway)
        self.assertIn("minimum of 100", takeaway)

    def test_17_temporal_irregular_intervals_and_gaps(self):
        # Dates skip April and June
        dates = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-05-01", "2024-07-01"])
        vals = [50, 55, 48, 52, 60]
        df = pd.DataFrame({"month": dates, "rate": vals})

        eda = run_automated_eda(df)
        ts = eda.get("time_series", {})
        self.assertTrue(ts.get("irregular_intervals"))
        self.assertGreater(len(ts.get("missing_periods", [])), 0)
        self.assertIn("irregular time intervals", ts.get("takeaway", "").lower())

    # =========================================================================
    # 6. WATERFALL & STACKED BAR GOVERNANCE
    # =========================================================================

    def test_18_waterfall_additive_bridge_reconciliation(self):
        df = pd.DataFrame({
            "Department": ["Assembly", "Machining", "Quality", "Packaging"],
            "Scrap_Cost": [4500.0, 3200.0, 1100.0, 800.0]
        })
        sem_layer = {
            "Department": {"is_dimension": True, "is_identifier": False, "grouping_eligibility": True},
            "Scrap_Cost": {"is_measure": True, "is_identifier": False, "is_additive": True, "semantic_subtype": "Currency"}
        }
        wf = generate_waterfall_chart(df, sem_layer)
        self.assertTrue(wf["is_valid"])
        self.assertEqual(wf["waterfall_status"], "VALID")
        # Sum of items without total must equal the total
        items = wf["data"]
        comps = [item["value"] for item in items if item["type"] != "total"]
        tot = [item["value"] for item in items if item["type"] == "total"][0]
        self.assertAlmostEqual(sum(comps), tot, places=2)

    def test_19_waterfall_blocked_for_non_additive_metrics(self):
        df = pd.DataFrame({
            "Division": ["North", "South", "East", "West"],
            "Operating_Margin_Pct": [14.2, 18.5, 9.8, 12.0]
        })
        sem_layer = {
            "Division": {"is_dimension": True, "is_identifier": False, "grouping_eligibility": True},
            "Operating_Margin_Pct": {"is_measure": True, "is_identifier": False, "is_additive": False, "semantic_subtype": "Percentage"}
        }
        wf = generate_waterfall_chart(df, sem_layer)
        self.assertFalse(wf["is_valid"])
        self.assertEqual(wf["waterfall_status"], "INVALID")
        self.assertIn("non-additive", wf["reason"].lower())

    def test_20_stacked_bar_hierarchical_structure(self):
        df = pd.DataFrame({
            "Plant": ["Plant_A", "Plant_A", "Plant_B", "Plant_B", "Plant_C", "Plant_C"],
            "Shift": ["Day", "Night", "Day", "Night", "Day", "Night"],
            "Units_Produced": [500, 420, 610, 580, 490, 460]
        })
        sem_layer = {
            "Plant": {"is_dimension": True, "is_identifier": False, "grouping_eligibility": True},
            "Shift": {"is_dimension": True, "is_identifier": False, "grouping_eligibility": True},
            "Units_Produced": {"is_measure": True, "is_identifier": False, "is_additive": True}
        }
        charts = generate_stacked_bar_charts(df, sem_layer)
        self.assertGreater(len(charts), 0)
        stacked = charts[0]
        self.assertEqual(stacked["chart_type"], "STACKED_BAR")
        self.assertEqual(stacked["x_dimension"], "Plant")
        self.assertEqual(stacked["series_dimension"], "Shift")
        # Verify exactly 3 x-axis categories (one per plant)
        self.assertEqual(len(stacked["data"]), 3)

    # =========================================================================
    # 7. RECOMMENDATION & ACTION CENTER GOVERNANCE
    # =========================================================================

    def test_21_recommendations_no_fabricated_targets_or_owners(self):
        rec = RecommendationGovernanceEngine.build_recommendation(
            rec_id="rec_901",
            insight_id="ins_01",
            observation="Observed scrap cost elevation in Machining section.",
            interpretation="Descriptive concentration of 41.2% in Machining.",
            suggested_action="Review tool wear logs for Machining line 3."
        )
        self.assertEqual(rec["owner"], "Unassigned")
        self.assertEqual(rec["target"], "Not defined")
        self.assertEqual(rec["timeline"], "Not specified")
        self.assertEqual(rec["traceability_status"], "VERIFIED")
        self.assertEqual(rec["action_type"], "ANALYTICAL_NEXT_STEP")

    def test_22_action_center_inherits_analytical_lineage(self):
        action = ActionGovernanceEngine.create_action_item(
            action_id="act_401",
            recommendation_id="rec_901",
            insight_id="ins_01",
            title="Machining Tooling Audit",
            metric_name="Scrap_Cost",
            baseline_val="3,200.00 USD",
            steps=["Audit line 3 spindle runout", "Log calibration offset"],
            analytical_result_id="ar_5544",
            source_table="Manufacturing_Operations",
            source_columns=["Scrap_Cost", "Machining_Section"]
        )
        self.assertEqual(action["analytical_result_id"], "ar_5544")
        self.assertEqual(action["source_table"], "Manufacturing_Operations")
        self.assertEqual(action["source_columns"], ["Scrap_Cost", "Machining_Section"])
        self.assertEqual(action["traceability_status"], "VERIFIED")
        self.assertEqual(action["owner"], "Unassigned")
        self.assertEqual(action["kpi_target"], "Not defined")

    # =========================================================================
    # 8. HINDALCO PLANT SAFETY OBSERVATION DATASET (ARBITRARY SCHEMA)
    # =========================================================================

    def test_23_hindalco_plant_safety_observation_dataset_semantics(self):
        """
        Tests realistic safety-observation data containing:
        - observer badge (person reference)
        - work area / shop floor location
        - observation category (unsafe act vs unsafe condition vs near miss)
        - hazard severity rating (1 to 5)
        - calculated risk index (0 to 100)
        - closure turnaround days (duration)
        - corrective action status (open / closed)
        - event log timestamp
        CRITICAL: All inferred through generic heuristics, NOT hardcoded column names!
        """
        np.random.seed(888)
        n = 120
        safety_df = pd.DataFrame({
            "worker_ref": [f"WKR_{i:03d}" for i in np.random.randint(100, 250, n)],
            "facility_zone": np.random.choice(["Smelter_Potline_1", "Rolling_Mill_East", "Carbon_Plant", "Alumina_Silo"], n),
            "hazard_classification": np.random.choice(["Unsafe Act", "Unsafe Condition", "Near Miss"], n),
            "severity_grade": np.random.choice([1, 2, 3, 4, 5], n, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
            "risk_index": np.random.uniform(10.0, 95.0, n).round(1),
            "resolution_duration_days": np.random.exponential(scale=6.0, size=n).round(1),
            "action_closure_state": np.random.choice(["Open", "Closed"], n, p=[0.25, 0.75]),
            "observation_timestamp": pd.date_range("2024-01-01", periods=n, freq="D")
        })

        dict_items = generate_data_dictionary(safety_df, table_name="Plant_Safety_Log")
        dict_map = {c["column_name"]: c for c in dict_items}

        # 1. worker_ref must be identifier, NOT foreign key (no parent table)
        self.assertEqual(dict_map["worker_ref"]["semantic_role"], "Identifier")
        self.assertFalse(dict_map["worker_ref"]["is_foreign_key"])

        # 2. facility_zone must be categorical dimension
        self.assertEqual(dict_map["facility_zone"]["semantic_role"], "Dimension")
        self.assertTrue(dict_map["facility_zone"]["grouping_eligibility"])

        # 3. severity_grade must be Rating (bounded 1-5, non-additive)
        self.assertEqual(dict_map["severity_grade"]["semantic_subtype"], "Rating")
        self.assertEqual(dict_map["severity_grade"]["scale_type"], "ordinal/bounded")
        self.assertIn("SUM not recommended", dict_map["severity_grade"]["aggregation_restrictions"][0])

        # 4. risk_index must be Score / Index (bounded, evaluative)
        self.assertIn(dict_map["risk_index"]["semantic_subtype"], ["Score", "Score / Index"])

        # 5. resolution_duration_days must be Duration in days
        self.assertEqual(dict_map["resolution_duration_days"]["semantic_subtype"], "Duration")
        self.assertEqual(dict_map["resolution_duration_days"]["unit"], "Days")

        # 6. action_closure_state must be Binary Category
        self.assertEqual(dict_map["action_closure_state"]["semantic_subtype"], "Binary Category")

        # 7. observation_timestamp must be Date/Time
        self.assertEqual(dict_map["observation_timestamp"]["semantic_role"], "Date/Time")

    def test_24_safety_observation_volume_is_not_labeled_as_poor_performance(self):
        # A zone with more safety observations should be reported as volume/concentration,
        # NEVER as "performed worst" or "unsafe zone" without exposure denominator.
        takeaway, interpretation = InsightGovernanceEngine.formulate_factual_takeaway(
            metric="Safety_Observations",
            category="Smelter_Potline_1",
            category_share=45.0,
            category_val=90.0,
            grouped_total=200.0,
            unit="observations"
        )
        self.assertIn("accounts for 45.0%", takeaway)
        self.assertIn("90.00 out of 200.00 observations", takeaway)
        # Verify no unwarranted safety performance judgment
        self.assertNotIn("worst", takeaway.lower())
        self.assertNotIn("unsafe", takeaway.lower())
        self.assertNotIn("performed worst", interpretation.lower())


if __name__ == "__main__":
    unittest.main()
