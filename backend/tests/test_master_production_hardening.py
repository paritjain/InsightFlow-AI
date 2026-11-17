import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from engine.governed_engines import (
    KeyDetectionEngine,
    RelationshipDetectionEngine,
    AnalyticalEligibilityEngine,
    calculate_canonical_dataset_health,
    DatasetHealthResult,
    DualCompatibleStr
)
from engine.cleaner import calculate_data_health_score_breakdown
from engine.profiler import infer_column_semantic_metadata, profile_file
from engine.dictionary import generate_data_dictionary
from engine.analytical_truth import (
    compute_grouped_analytical_result,
    safe_correlation,
    safe_pca_features,
    AnalyticalResult
)
from engine.eda import run_automated_eda, generate_waterfall_chart
from engine.insights import (
    generate_insights_and_story,
    deduplicate_insights,
    format_val_str
)
from engine.action_plan import (
    generate_recommendations_and_action_plan,
    deduplicate_recommendations,
    deduplicate_actions,
    TimelineStr
)
from services.statistical_analysis import (
    compute_pca,
    compute_bivariate_hypothesis_tests,
    compute_group_differences
)


class TestMasterProductionHardening(unittest.TestCase):
    """
    Comprehensive Master Production-Hardening Regression Suite (Sections 39 A-W).
    Dataset-agnostic verification covering all 23 audit criteria.
    """

    def setUp(self):
        # Representative synthetic multi-scenario dataset
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            "order_id": [f"ORD-{i:05d}" for i in range(1, n + 1)],
            "salesperson_id": [f"SP-{((i % 7) + 1):03d}" for i in range(1, n + 1)],
            "order_date": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
            "region": ["North", "South", "East", "West"] * (n // 4),
            "revenue_inr": np.random.uniform(5000.0, 50000.0, size=n).round(2),
            "gross_profit_inr": np.random.uniform(1000.0, 15000.0, size=n).round(2),
            "return_rate_pct": np.random.uniform(0.5, 12.5, size=n).round(2),
            "inventory_days": np.random.randint(5, 60, size=n),
            "working_days": [22] * n,
            "zero_variance_col": [10.0] * n,
            "sparse_feature": [1.0 if i < 3 else np.nan for i in range(n)]
        })

    # A. PRIMARY KEY DETECTION
    def test_A_primary_key(self):
        # Unique non-null identifier -> PRIMARY_KEY with machine-readable evidence
        pk_res = KeyDetectionEngine.detect_key("order_id", self.df["order_id"], total_rows=len(self.df))
        self.assertIn(str(pk_res["key_role"]).lower(), ["primary key", "primary_key"])
        self.assertEqual(pk_res["confidence"], 0.99)
        self.assertEqual(pk_res["evidence"]["uniqueness_pct"], 100.0)
        self.assertEqual(pk_res["evidence"]["null_pct"], 0.0)

        # Duplicate identifier -> NOT_PRIMARY_KEY
        dup_series = pd.Series(["ID-1", "ID-1", "ID-2", "ID-3"])
        not_pk_res = KeyDetectionEngine.detect_key("dup_id", dup_series, total_rows=4)
        self.assertNotIn("primary_key", str(not_pk_res["key_role"]).lower().replace("not_primary_key", ""))
        self.assertEqual(not_pk_res["key_role"], "NOT_PRIMARY_KEY")

    # B. FOREIGN KEY DETECTION
    def test_B_foreign_key(self):
        # No parent relationship available -> UNVERIFIED (never promoted to verified FK)
        fk_res = KeyDetectionEngine.detect_key("salesperson_id", self.df["salesperson_id"], total_rows=len(self.df))
        self.assertEqual(fk_res["relationship_status"], "UNVERIFIED")

        # Parent table provided with valid join -> VERIFIED_FOREIGN_KEY
        parent_table = pd.DataFrame({"sp_id": [f"SP-{i:03d}" for i in range(1, 10)]})
        verified_fk_res = KeyDetectionEngine.detect_key(
            "salesperson_id",
            self.df["salesperson_id"],
            total_rows=len(self.df),
            parent_table=parent_table,
            parent_key_col="sp_id"
        )
        self.assertEqual(verified_fk_res["key_role"], "VERIFIED_FOREIGN_KEY")
        self.assertEqual(verified_fk_res["relationship_status"], "VERIFIED")

    # C. DATE / TIME SEMANTICS
    def test_C_date_semantics(self):
        meta = infer_column_semantic_metadata("order_date", self.df["order_date"], total_rows=len(self.df))
        self.assertEqual(meta["semantic_role"], "Date/Time")
        self.assertTrue(meta["time_series_eligibility"])
        self.assertTrue(meta["analytical_eligibility"])
        self.assertTrue(meta["visualization_eligibility"])

    # D. CURRENCY SEMANTICS (INR / symbol ₹)
    def test_D_currency_inr(self):
        meta = infer_column_semantic_metadata("revenue_inr", self.df["revenue_inr"], total_rows=len(self.df))
        self.assertEqual(meta["currency"], "INR")
        self.assertEqual(meta["currency_metadata"]["symbol"], "₹")
        self.assertNotEqual(meta["currency_metadata"]["symbol"], "$")
        formatted = format_val_str(12500.5, is_currency=True, currency_symbol="₹")
        self.assertTrue(formatted.startswith("₹"))

    # E. PERCENTAGE SEMANTICS
    def test_E_percentage_semantics(self):
        meta = infer_column_semantic_metadata("return_rate_pct", self.df["return_rate_pct"], total_rows=len(self.df))
        self.assertEqual(meta["measure_type"], "Percentage")
        self.assertEqual(meta["unit"], "%")
        self.assertEqual(meta["valid_range"], "[0, 100]")
        self.assertIsNone(meta.get("rating_score_metadata", {}).get("scale_type"))

    # F. DURATION VS COUNT
    def test_F_duration_vs_count(self):
        # Inventory_Days -> Duration(days)
        dur_meta = infer_column_semantic_metadata("inventory_days", self.df["inventory_days"], total_rows=len(self.df))
        self.assertEqual(dur_meta["semantic_role"], "Measure")
        self.assertEqual(dur_meta["measure_type"], "Duration")
        self.assertEqual(str(dur_meta["unit"]).lower(), "days")

        # Working_Days -> Count(days)
        cnt_meta = infer_column_semantic_metadata("working_days", self.df["working_days"], total_rows=len(self.df))
        self.assertEqual(cnt_meta["measure_type"], "Count")
        self.assertEqual(str(cnt_meta["unit"]).lower(), "days")

    # G. GROUPING ELIGIBILITY
    def test_G_grouping_eligibility(self):
        # Region (categorical dimension) -> grouping_eligibility = True
        region_meta = infer_column_semantic_metadata("region", self.df["region"], total_rows=len(self.df))
        self.assertTrue(region_meta["grouping_eligibility"])

        # Identifiers -> grouping_eligibility = False
        pk_meta = infer_column_semantic_metadata("order_id", self.df["order_id"], total_rows=len(self.df))
        self.assertFalse(pk_meta["grouping_eligibility"])

    # H. HEALTH SCORE SINGLE SOURCE OF TRUTH
    def test_H_health_score_single_source_of_truth(self):
        canonical_health = calculate_canonical_dataset_health(self.df)
        self.assertIsInstance(canonical_health, DatasetHealthResult)
        self.assertGreater(canonical_health.health_score, 0.0)
        self.assertLessEqual(canonical_health.health_score, 100.0)

        # Cleaner module consumes identical canonical health score
        cleaner_score = calculate_data_health_score_breakdown(self.df)
        self.assertEqual(cleaner_score["health_score"], canonical_health.health_score)

        # Profile consumes identical canonical health score
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="") as tmp:
            tmp_path = tmp.name
        try:
            self.df.to_csv(tmp_path, index=False)
            profile_res = profile_file(tmp_path)
            self.assertEqual(profile_res["health_score"], canonical_health.health_score)
            self.assertEqual(profile_res["health_score_breakdown"]["health_score"], canonical_health.health_score)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # I. INSIGHT DEDUPLICATION
    def test_I_insight_deduplication(self):
        eda_res = run_automated_eda(self.df)
        ins_res = generate_insights_and_story(self.df, eda_res)
        insights = ins_res["insights"]

        # Duplicate insight list
        duplicated_list = insights + [dict(ins) for ins in insights]
        deduped = deduplicate_insights(duplicated_list)
        self.assertEqual(len(deduped), len(insights))

    # J. RECOMMENDATION DEDUPLICATION
    def test_J_recommendation_deduplication(self):
        eda_res = run_automated_eda(self.df)
        ins_res = generate_insights_and_story(self.df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])
        recs = rec_res["recommendations"]

        duplicated_recs = recs + [dict(r) for r in recs]
        deduped = deduplicate_recommendations(duplicated_recs)
        self.assertEqual(len(deduped), len(recs))

    # K. ACTION IDs GLOBALLY UNIQUE
    def test_K_action_ids_unique(self):
        eda_res = run_automated_eda(self.df)
        ins_res = generate_insights_and_story(self.df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])
        actions = rec_res["action_items"]

        self.assertGreater(len(actions), 0)
        action_ids = [str(act["id"]) for act in actions]
        self.assertEqual(len(action_ids), len(set(action_ids)), "Action IDs must be globally unique!")
        # Verify ACT-00000X format
        self.assertTrue(any(a.startswith("ACT-") for a in action_ids))

    # L. ACTION CENTER DEDUPLICATION
    def test_L_action_center_deduplication(self):
        eda_res = run_automated_eda(self.df)
        ins_res = generate_insights_and_story(self.df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])
        actions = rec_res["action_items"]

        duplicated_actions = actions + [dict(a) for a in actions]
        deduped_actions = deduplicate_actions(duplicated_actions)
        self.assertEqual(len(deduped_actions), len(actions))

    # M. MEASURE CONTRIBUTION VS RECORD SHARE
    def test_M_measure_contribution(self):
        ar = compute_grouped_analytical_result(
            df=self.df,
            measure_col="revenue_inr",
            group_col="region",
            aggregation="SUM"
        )
        pct_meta = ar.percentage_metadata
        self.assertEqual(pct_meta["percentage_type"], "MEASURE_CONTRIBUTION")
        self.assertIn("revenue_inr", pct_meta["denominator"])
        self.assertGreater(pct_meta["denominator_value"], 0)

    # N. CORRELATION TRANSPARENCY
    def test_N_correlation_transparency(self):
        # 3 missing values present in pairwise correlation (97% coverage, satisfies >95% threshold)
        df_corr = self.df.copy()
        df_corr.loc[:2, "revenue_inr"] = np.nan
        corr_res = safe_correlation(df_corr, "revenue_inr", "gross_profit_inr")

        self.assertTrue(corr_res["is_valid"])
        self.assertIn("valid_n", corr_res)
        self.assertIn("total_n", corr_res)
        self.assertIn("coverage_pct", corr_res)
        self.assertEqual(corr_res["total_n"], 100)
        self.assertEqual(corr_res["valid_n"], 97)
        self.assertEqual(corr_res["coverage_pct"], 97.0)

    # O. KRUSKAL-WALLIS TRANSPARENCY
    def test_O_kruskal_wallis_transparency(self):
        diff_res = compute_group_differences(self.df, "region", "revenue_inr")
        kw = diff_res["kruskal_wallis"]
        self.assertTrue(kw["valid"])
        self.assertIn("kruskal_h", kw)
        self.assertIn("kruskal_p", kw)
        self.assertIn("epsilon_squared", kw)
        self.assertIn("valid_n", kw)
        self.assertIn("group_counts", kw)

    # P. ANOVA TRANSPARENCY
    def test_P_anova_transparency(self):
        diff_res = compute_group_differences(self.df, "region", "revenue_inr")
        anova = diff_res["anova"]
        self.assertTrue(anova["valid"])
        self.assertIn("f_statistic", anova)
        self.assertIn("p_value", anova)
        self.assertIn("eta_squared", anova)
        self.assertIn("levene_statistic", anova)
        self.assertIn("normality_status", anova)
        self.assertIn("assumptions_validated", anova)

    # Q. PCA POPULATION TRANSPARENCY
    def test_Q_pca_transparency(self):
        pca_res = compute_pca(self.df)
        self.assertTrue(pca_res["is_valid"])
        self.assertIn("source_records", pca_res)
        self.assertIn("effective_observations", pca_res)
        self.assertIn("excluded_records", pca_res)
        self.assertIn("coverage_pct", pca_res)
        self.assertEqual(pca_res["missing_value_policy"], "COMPLETE_CASE")

    # R. ZERO VARIANCE FEATURE EXCLUSION
    def test_R_zero_variance_exclusion(self):
        eval_res = AnalyticalEligibilityEngine.evaluate_column("zero_variance_col", self.df["zero_variance_col"], len(self.df))
        self.assertFalse(eval_res["correlation_eligible"])
        self.assertFalse(eval_res["pca_eligible"])
        self.assertIn(eval_res["exclusion_reason"], ["CONSTANT", "ZERO_VARIANCE"])

    # S. SPARSE VARIABLE EXCLUSION
    def test_S_sparse_variable_exclusion(self):
        eval_res = AnalyticalEligibilityEngine.evaluate_column("sparse_feature", self.df["sparse_feature"], len(self.df))
        self.assertFalse(eval_res["analytical_eligible"])
        self.assertIn(eval_res["exclusion_reason"], ["SPARSE", "HIGH_NULLS"])

    # T. OWNER SAFETY (NEVER INVENTED)
    def test_T_owner_safety(self):
        eda_res = run_automated_eda(self.df)
        ins_res = generate_insights_and_story(self.df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])
        for act in rec_res["action_items"]:
            self.assertEqual(act["owner"], "UNASSIGNED")
            self.assertIn("Suggested function:", act["suggested_owner"])

    # U. KPI VS MONITORED METRIC SAFETY
    def test_U_kpi_vs_monitored_metric_safety(self):
        eda_res = run_automated_eda(self.df)
        ins_res = generate_insights_and_story(self.df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])
        for rec in rec_res["recommendations"]:
            self.assertTrue(rec["monitored_metric"])
            self.assertEqual(rec["kpi_status"], "NOT_DEFINED")

    # V. TARGET SAFETY (NOT INVENTED)
    def test_V_target_safety(self):
        eda_res = run_automated_eda(self.df)
        ins_res = generate_insights_and_story(self.df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])
        for rec in rec_res["recommendations"]:
            self.assertEqual(rec["target_value"], "Not defined")
            self.assertIn(rec["target_status"], ["NOT_DEFINED", "Awaiting business target"])

    # W. TIMELINE SAFETY (NOT INVENTED)
    def test_W_timeline_safety(self):
        eda_res = run_automated_eda(self.df)
        ins_res = generate_insights_and_story(self.df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])
        for rec in rec_res["recommendations"]:
            self.assertEqual(rec["timeline"], "Not specified")
            self.assertEqual(rec["suggested_timeline"], "Not specified")


if __name__ == "__main__":
    unittest.main()
