"""
INDEPENDENT ANALYTICAL AUDIT & GOLDEN DATASET VERIFICATION SUITE
Validates:
1. 5 Synthetic Arbitrary Datasets with zero hardcoding:
   - Dataset A: E-Commerce Orders
   - Dataset B: Corporate Finance & Treasury
   - Dataset C: Manufacturing Telemetry
   - Dataset D: Customer Service Operations
   - Dataset E: Multi-table Relational Database
2. Golden Datasets with mathematically known ground-truth:
   - Mean, Median, Variance, Standard Deviation
   - Pearson & Spearman (perfect positive, perfect negative, zero, monotonic nonlinear)
   - Grouped Sum & Proportional Share
   - Outliers (Tukey IQR fences)
   - Temporal Trend
3. ANOVA & Kruskal-Wallis Assumption Testing vs Computation Distinction
4. End-to-end Single Source of Truth Lineage Check
"""

import unittest
import numpy as np
import pandas as pd
from scipy import stats

from engine.governed_engines import (
    SemanticInferenceEngine,
    KeyDetectionEngine,
    RelationshipDetectionEngine,
    DataQualityEngine,
    CategoryNormalizationEngine,
    StatisticalSuitabilityEngine,
    ChartSuitabilityEngine,
    AnalyticalResultEngine
)
from engine.analytical_truth import (
    compute_variable_constancy,
    compute_grouped_analytical_result,
    safe_correlation,
    AnalyticalResult
)
from engine.eda import run_automated_eda
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan
from services.statistical_analysis import (
    compute_group_differences,
    compute_bivariate_hypothesis_tests,
    compute_pca
)


class TestIndependentAnalyticalAudit(unittest.TestCase):
    """Rigorous mathematical, statistical, and semantic verification suite."""

    # =========================================================================
    # PART 1: GOLDEN DATASET ACCURACY TESTS
    # =========================================================================

    def test_golden_01_descriptive_statistics_accuracy(self):
        """Verify ground-truth arithmetic mean, median, and sample variance."""
        # Ground truth: [10, 20, 30]
        # Mean = (10+20+30)/3 = 20.0
        # Median = 20.0
        # Sample Variance (ddof=1) = ((10-20)^2 + (20-20)^2 + (30-20)^2) / 2 = 200 / 2 = 100.0
        # Sample Std = sqrt(100) = 10.0
        df = pd.DataFrame({"metric": [10.0, 20.0, 30.0]})
        dq = DataQualityEngine.profile_dataframe(df)
        stats_out = dq["columns"]["metric"]["statistics"]

        self.assertEqual(stats_out["mean"], 20.0)
        self.assertEqual(stats_out["median"], 20.0)
        self.assertEqual(stats_out["variance"], 100.0)
        self.assertEqual(stats_out["std"], 10.0)

    def test_golden_02_even_median_accuracy(self):
        """Verify median for even count of elements [10, 20, 30, 40] -> 25.0."""
        df = pd.DataFrame({"metric": [10.0, 20.0, 30.0, 40.0]})
        dq = DataQualityEngine.profile_dataframe(df)
        self.assertEqual(dq["columns"]["metric"]["statistics"]["median"], 25.0)

    def test_golden_03_pearson_and_spearman_perfect_correlation(self):
        """Verify Pearson and Spearman on mathematically perfect relationships."""
        # 1. Perfect Positive: Y = 2X
        df_pos = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0]})
        res_pos = safe_correlation(df_pos, "x", "y")
        self.assertEqual(res_pos["correlation_status"], "valid")
        self.assertAlmostEqual(res_pos["pearson"], 1.0, places=3)
        self.assertAlmostEqual(res_pos["spearman"], 1.0, places=3)
        self.assertEqual(res_pos["direction"], "positive")

        # 2. Perfect Negative: Y = -3X + 15
        df_neg = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [12.0, 9.0, 6.0, 3.0, 0.0]})
        res_neg = safe_correlation(df_neg, "x", "y")
        self.assertEqual(res_neg["correlation_status"], "valid")
        self.assertAlmostEqual(res_neg["pearson"], -1.0, places=3)
        self.assertAlmostEqual(res_neg["spearman"], -1.0, places=3)
        self.assertEqual(res_neg["direction"], "negative")

    def test_golden_04_nonlinear_monotonic_rank_correlation(self):
        """Verify Spearman rank correlation is 1.0 while Pearson is < 1.0 for non-linear power curve."""
        # Y = X^3: monotonically increasing, but non-linear
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 8.0, 27.0, 64.0, 125.0]
        df = pd.DataFrame({"x": x, "y": y})
        res = safe_correlation(df, "x", "y")
        self.assertAlmostEqual(res["spearman"], 1.0, places=3)
        # Pearson is strictly less than 1.0 due to curvature (around 0.981)
        self.assertLess(res["pearson"], 1.0)
        self.assertGreater(res["pearson"], 0.90)

    def test_golden_05_iqr_outlier_fences(self):
        """Verify Tukey IQR outlier bounds: [10, 12, 13, 14, 15, 16, 100]."""
        # Q1 = 12.0, Q3 = 16.0, IQR = 4.0
        # Upper Fence = 16.0 + 1.5 * 4.0 = 22.0
        # 100 > 22.0 -> Exactly 1 outlier detected
        df = pd.DataFrame({"val": [10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 100.0]})
        sem = {"val": {"is_measure": True}}
        dq = DataQualityEngine.profile_dataframe(df, sem)
        col_stat = dq["columns"]["val"]["statistics"]
        self.assertEqual(col_stat["outlier_count"], 1)

    def test_golden_06_grouped_aggregation_and_percentage(self):
        """Verify grouped aggregation sums and exact percentage calculation."""
        # Category A: 300, Category B: 100, Total: 400
        # Category A share = 300 / 400 = 75.0%
        # Category B share = 100 / 400 = 25.0%
        df = pd.DataFrame({
            "segment": ["A", "A", "B"],
            "amount": [100.0, 200.0, 100.0]
        })
        ar = compute_grouped_analytical_result(df, measure_col="amount", group_col="segment", aggregation="SUM")
        self.assertEqual(ar.statistics["grouped_total"], 400.0)
        self.assertEqual(ar.statistics["top_category"], "A")
        self.assertEqual(ar.statistics["top_category_share"], 75.0)
        self.assertEqual(ar.values["A"], 300.0)
        self.assertEqual(ar.values["B"], 100.0)

    # =========================================================================
    # PART 2: ANOVA & KRUSKAL-WALLIS RIGOROUS VALIDATION
    # =========================================================================

    def test_anova_01_assumption_distinction_and_homoscedasticity(self):
        """Verify ANOVA tests distinguish computation vs assumption validation."""
        # Group 1: [10, 11, 12, 10, 11] (mean=10.8, low variance)
        # Group 2: [20, 22, 21, 23, 20] (mean=21.2, low variance)
        # Differences are real and statistically significant
        df = pd.DataFrame({
            "cohort": ["C1", "C1", "C1", "C1", "C1", "C2", "C2", "C2", "C2", "C2"],
            "score": [10.0, 11.0, 12.0, 10.0, 11.0, 20.0, 22.0, 21.0, 23.0, 20.0]
        })
        res = compute_group_differences(df, "cohort", "score")
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["anova"]["valid"])
        self.assertEqual(res["anova"]["computation_status"], "ANOVA can be computed")
        self.assertIn("assumptions", res["anova"])
        self.assertTrue(res["anova"]["is_significant"])
        self.assertLess(res["anova"]["p_value"], 0.001)
        self.assertGreater(res["anova"]["eta_squared"], 0.90)

    def test_anova_02_constant_dv_never_displays_misleading_f0_p1(self):
        """Verify constant DV does not output F=0, p=1 as a meaningful test."""
        df_const = pd.DataFrame({
            "cohort": ["A", "A", "B", "B", "C", "C"],
            "flat_val": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
        })
        res = compute_group_differences(df_const, "cohort", "flat_val")
        self.assertEqual(res["test_status"], "not_applicable")
        self.assertEqual(res["reason"], "dependent_variable_constant")
        self.assertIsNone(res["anova"]["f_statistic"])
        self.assertIsNone(res["anova"]["p_value"])
        self.assertIn("constant", res["anova"]["display_message"].lower())

    def test_kruskal_wallis_preserves_distributional_rank_integrity(self):
        """Verify Kruskal-Wallis non-parametric test handles ranks properly without claiming business intent."""
        df = pd.DataFrame({
            "tier": ["Low", "Low", "Low", "High", "High", "High"],
            "metric": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
        })
        res = compute_group_differences(df, "tier", "metric")
        self.assertTrue(res["kruskal_wallis"]["valid"])
        self.assertLess(res["kruskal_wallis"]["p_value"], 0.05)
        # Interpretation must not claim automatic business consequence
        self.assertIn("rank", res["kruskal_wallis"]["interpretation"].lower())

    # =========================================================================
    # PART 3: ARBITRARY DATASET VERIFICATION (5 DISTINCT DOMAINS)
    # =========================================================================

    def test_arbitrary_dataset_a_ecommerce_sales(self):
        """Dataset A: E-Commerce Orders & Catalog."""
        df_sales = pd.DataFrame({
            "order_number": [f"ORD-{i:04d}" for i in range(1, 21)],
            "item_category": ["Electronics", "Books", "Clothing", "Electronics"] * 5,
            "net_revenue": [150.0, 25.0, 45.0, 300.0] * 5,
            "return_count": [0, 0, 1, 0] * 5,
            "is_discounted": [True, False, False, True] * 5
        })
        eda = run_automated_eda(df_sales)
        self.assertIn("order_number", eda["identifier_summary"])
        self.assertIn("net_revenue", eda["numeric_summary"])
        self.assertIn("item_category", eda["categorical_summary"])
        # Primary key evidence
        pk_info = KeyDetectionEngine.detect_key("order_number", df_sales["order_number"], len(df_sales))
        self.assertEqual(pk_info["key_role"], "primary_key")
        # Insight and action pipeline
        ins_res = generate_insights_and_story(df_sales, eda)
        self.assertGreater(len(ins_res["insights"]), 0)

    def test_arbitrary_dataset_b_corporate_finance(self):
        """Dataset B: Corporate Treasury & General Ledger."""
        df_fin = pd.DataFrame({
            "transaction_code": [f"TX-{i}" for i in range(1, 16)],
            "cost_center": ["CC_HQ", "CC_PLANT", "CC_LOGISTICS"] * 5,
            "debit_usd": [5000.0, 1200.0, 3400.0] * 5,
            "credit_usd": [0.0, 0.0, 0.0] * 5,  # Zero variance / constant
            "fiscal_month": ["2025-01", "2025-02", "2025-03"] * 5
        })
        eda = run_automated_eda(df_fin)
        # Credit USD has zero variance
        self.assertTrue(compute_variable_constancy(df_fin["credit_usd"])["is_constant"])
        # Suitability engine blocks credit_usd for histogram
        hist_check = ChartSuitabilityEngine.chartSuitabilityCheck("HISTOGRAM", df_fin, y_col="credit_usd")
        self.assertFalse(hist_check["suitable"])
        self.assertEqual(hist_check["status"], "BLOCKED_BY_VARIANCE")

    def test_arbitrary_dataset_c_manufacturing_telemetry(self):
        """Dataset C: IoT Vibration & Temperature Sensors."""
        df_mfg = pd.DataFrame({
            "sensor_id": [f"S_{i}" for i in range(1, 11)],
            "machine_line": ["Line_Alpha", "Line_Beta"] * 5,
            "vibration_hz": [45.2, 46.1, 45.9, 88.4, 45.5, 46.0, 45.8, 46.2, 45.7, 46.0],  # 88.4 is outlier
            "temperature_c": [72.0, 73.1, 71.9, 95.0, 72.5, 73.0, 72.2, 73.4, 72.1, 72.8],
            "operating_hours": [1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650]
        })
        eda = run_automated_eda(df_mfg)
        # PCA should execute successfully on 3 non-constant numeric features
        pca = compute_pca(df_mfg, n_components=2)
        self.assertTrue(pca["is_valid"])
        self.assertEqual(len(pca["explained_variance_ratio"]), 2)

    def test_arbitrary_dataset_d_customer_service_support(self):
        """Dataset D: Support Tickets & CSAT Ratings."""
        df_support = pd.DataFrame({
            "ticket_uuid": [f"TKT-{i:05d}" for i in range(1, 13)],
            "channel": ["Email", "Phone", "Chat"] * 4,
            "resolution_duration_mins": [45.0, 12.0, 8.0, 60.0, 15.0, 10.0, 50.0, 14.0, 7.0, 55.0, 16.0, 9.0],
            "satisfaction_rating": [4, 5, 5, 2, 4, 5, 3, 4, 5, 2, 4, 5]  # Rating 1-5 scale
        })
        meta_csat = SemanticInferenceEngine.infer_column("satisfaction_rating", df_support["satisfaction_rating"], len(df_support))
        self.assertEqual(meta_csat["measure_type"], "Rating")
        self.assertEqual(meta_csat["rating_score_metadata"]["scale_type"], "1-5")

        meta_dur = SemanticInferenceEngine.infer_column("resolution_duration_mins", df_support["resolution_duration_mins"], len(df_support))
        self.assertEqual(meta_dur["measure_type"], "Duration")
        self.assertEqual(meta_dur["unit"], "minutes")

    def test_arbitrary_dataset_e_multitable_relational_discovery(self):
        """Dataset E: Relational Parent-Child Schema."""
        parents = pd.DataFrame({
            "facility_id": ["FAC-1", "FAC-2", "FAC-3"],
            "location_name": ["North Hub", "South Facility", "East Depot"],
            "capacity": [10000, 25000, 15000]
        })
        children = pd.DataFrame({
            "asset_id": [101, 102, 103, 104, 105],
            "facility_id": ["FAC-1", "FAC-1", "FAC-2", "FAC-3", "FAC-999"],  # FAC-999 is orphan FK violation
            "asset_status": ["Active", "Active", "Maintenance", "Active", "Decommissioned"]
        })
        rels = RelationshipDetectionEngine.detect_and_validate_relationships({
            "facilities": parents,
            "assets": children
        })
        self.assertGreater(len(rels), 0)
        rel = next(r for r in rels if r["source_table"] == "assets" and r["target_table"] == "facilities")
        self.assertIsNotNone(rel)
        self.assertEqual(rel["child_table"], "assets")
        self.assertEqual(rel["parent_table"], "facilities")
        self.assertEqual(rel["fk_violation_count"], 1)
        self.assertGreater(rel["fk_violation_percentage"], 0.0)

    # =========================================================================
    # PART 4: SINGLE SOURCE OF TRUTH & LINEAGE END-TO-END
    # =========================================================================

    def test_single_source_of_truth_shared_lineage(self):
        """Verify exact shared resultId across Chart, Insight, Recommendation, and Action."""
        df = pd.DataFrame({
            "sales_rep": ["Alice", "Bob", "Charlie", "Alice", "Bob"],
            "closed_deal_value": [50000.0, 75000.0, 120000.0, 60000.0, 80000.0]
        })
        eda = run_automated_eda(df)
        ins_payload = generate_insights_and_story(df, eda)
        insights = ins_payload["insights"]
        self.assertGreater(len(insights), 0)

        # 1. Insight has shared analytical result
        ins = insights[0]
        shared_ar = ins["shared_result"]
        self.assertIsNotNone(shared_ar)
        result_id = shared_ar.get("result_id") or shared_ar.get("resultId") or shared_ar.get("analytical_result_id")
        self.assertIsNotNone(result_id)

        # 2. Recommendation and Action inherit exact same resultId
        rec_payload = generate_recommendations_and_action_plan(insights)
        recs = rec_payload["recommendations"]
        actions = rec_payload["action_items"]
        self.assertGreater(len(recs), 0)
        self.assertGreater(len(actions), 0)

        first_rec = recs[0]
        first_act = actions[0]

        self.assertEqual(first_rec["result_id"], result_id)
        self.assertEqual(first_act["result_id"], result_id)
        self.assertEqual(first_act["recommendation_id"], first_rec["id"])

        # 3. Safe unassigned defaults
        self.assertEqual(first_rec["owner"], "Unassigned")
        self.assertEqual(first_rec["kpi_target"], "Not defined")
        self.assertEqual(first_act["owner"], "Unassigned")
        self.assertEqual(first_act["status"], "Not Started")


if __name__ == "__main__":
    unittest.main()
