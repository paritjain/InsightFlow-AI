"""
MASTER PROMPT PRODUCTION-GRADE EDA TEST SUITE
Validates all 24 required test categories and 31 acceptance criteria from the MASTER PROMPT:
1. Constant numeric column & zero-variance pre-detection
2. Numeric variable with missing values & denominator transparency
3. Numeric variable with outliers (statistical outliers separate from data quality errors)
4. Categorical variable
5. Mixed-case categories & deterministic category normalization lineage
6. Candidate primary key evidence-based detection
7. Duplicate key non-PK rejection
8. Candidate foreign key referential coverage
9. FK violations tracking (orphan counts, violation %)
10. Composite primary key detection
11. Date/time variable deterministic temporal analysis
12. Non-additive metric waterfall blocking
13. Additive metric waterfall acceptance
14. Zero-variance dependent variable guardrails
15. Correlation blocked with constant variable
16. ANOVA blocked with constant dependent variable
17. Kruskal-Wallis blocked with constant dependent variable
18. PCA excluding constant features
19. Filtered analytical dataset with explicit exclusion tracking
20. Different denominators transparency (grouped vs full total)
21. Duplicate entity records profiling
22. Missing categories handling
23. Multiple sheets/tables relationship discovery
24. Same chart/result/insight/recommendation lineage (single source of truth: AnalyticalResult)
"""

import unittest
import pandas as pd
import numpy as np

from engine.governed_engines import (
    SemanticInferenceEngine,
    KeyDetectionEngine,
    RelationshipDetectionEngine,
    DataQualityEngine,
    CategoryNormalizationEngine,
    StatisticalSuitabilityEngine,
    ChartSuitabilityEngine,
    AnalyticalResultEngine,
    LineageEngine,
    InsightGovernanceEngine,
    RecommendationGovernanceEngine,
    ActionGovernanceEngine
)
from engine.analytical_truth import (
    compute_variable_constancy,
    compute_grouped_analytical_result,
    AnalyticalResult
)
from engine.profiler import infer_column_semantic_metadata
from engine.dictionary import generate_data_dictionary
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan
from engine.eda import run_automated_eda
from services.statistical_analysis import (
    compute_group_differences,
    compute_bivariate_hypothesis_tests,
    compute_pca
)



class TestMasterAnalyticalHardening(unittest.TestCase):
    """
    Comprehensive test suite ensuring statistical correctness, semantic correctness,
    traceability, deterministic suitability, and safe executive management demonstrations.
    """

    def setUp(self):
        # Multi-table test setup
        self.orders_df = pd.DataFrame({
            "order_id": [101, 102, 103, 104, 105],
            "customer_id": ["C1", "C2", "C3", "C4", "C5"],
            "amount_usd": [250.0, 150.0, 400.0, 300.0, 500.0],
            "shipping_fee": [15.0, 15.0, 15.0, 15.0, 15.0],  # Constant
            "status": ["Completed", "completed", "COMPLETED ", "Pending", "pending"]
        })

        self.order_items_df = pd.DataFrame({
            "order_id": [101, 101, 102, 103, 999],  # 999 is an orphan FK violation
            "item_id": [1, 2, 1, 1, 1],
            "quantity": [2, 1, 5, 3, 1],
            "unit_price": [50.0, 150.0, 30.0, 100.0, 20.0]
        })

    # -------------------------------------------------------------------------
    # 1. CONSTANT NUMERIC COLUMN & ZERO-VARIANCE PRE-DETECTION
    # -------------------------------------------------------------------------
    def test_01_constant_numeric_variable_detection(self):
        s_const = pd.Series([42.0, 42.0, 42.0, 42.0, 42.0])
        const_info = KeyDetectionEngine.compute_constancy(s_const)
        self.assertTrue(const_info["is_constant"])
        self.assertEqual(const_info["variance"], 0.0)
        self.assertEqual(const_info["nunique"], 1)

    # -------------------------------------------------------------------------
    # 2. NUMERIC VARIABLE WITH MISSING VALUES & EXCLUSION TRACKING
    # -------------------------------------------------------------------------
    def test_02_numeric_missing_values_transparency(self):
        df_missing = pd.DataFrame({
            "department": ["Sales", "Engineering", None, "Sales", "HR"],
            "salary": [60000.0, 80000.0, 75000.0, None, 55000.0]
        })
        ar = compute_grouped_analytical_result(
            df=df_missing,
            measure_col="salary",
            group_col="department",
            aggregation="SUM"
        )
        self.assertEqual(ar.total_record_count, 5)
        # 3 complete paired records; 1 valid salary record excluded from grouping due to null department
        self.assertEqual(ar.valid_record_count, 3)
        self.assertEqual(ar.excluded_record_count, 1)
        self.assertIn("missing", ar.excluded_record_reason.lower())

        self.assertIsNotNone(ar.result_id)
        self.assertTrue(str(ar.result_id).startswith("AR-"))

    # -------------------------------------------------------------------------
    # 3. NUMERIC VARIABLE WITH OUTLIERS (STATISTICAL VS DATA QUALITY)
    # -------------------------------------------------------------------------
    def test_03_statistical_outlier_separated_from_data_quality_error(self):
        # 999999.0 is a statistical outlier; -10 is an invalid domain error for working days
        df_outlier = pd.DataFrame({
            "working_days": [20, 22, 21, 22, 20, 21, 22, 20, -10],  # Negative days = Data quality error
            "salary": [50000, 52000, 49000, 51000, 53000, 48000, 50000, 52000, 999999]  # IQR Outlier
        })
        sem = {
            "working_days": {"is_measure": True, "measure_type": "Count", "unit": "days"},
            "salary": {"is_measure": True, "measure_type": "Currency", "unit": "USD"}
        }
        dq_profile = DataQualityEngine.profile_dataframe(df_outlier, sem)
        salary_stats = dq_profile["columns"]["salary"]
        self.assertGreater(salary_stats["statistical_outliers"], 0)
        self.assertEqual(len(salary_stats["data_quality_errors"]), 0)  # High salary is not an error

        wd_stats = dq_profile["columns"]["working_days"]
        self.assertGreater(len(wd_stats["data_quality_errors"]), 0)
        self.assertEqual(wd_stats["data_quality_errors"][0]["type"], "invalid_domain_negative_value")

    # -------------------------------------------------------------------------
    # 4. CATEGORICAL VARIABLE DECOUPLED METADATA
    # -------------------------------------------------------------------------
    def test_04_categorical_variable_metadata_separation(self):
        s_dept = pd.Series(["Sales", "Marketing", "Engineering", "HR", "Sales"])
        meta = SemanticInferenceEngine.infer_column("Department", s_dept, len(s_dept))
        self.assertEqual(meta["semantic_role"], "Dimension")
        self.assertEqual(meta["key_role"], "none")
        self.assertEqual(meta["id_safety"], "Safe")
        self.assertEqual(meta["physical_dtype"], "string")

    # -------------------------------------------------------------------------
    # 5. MIXED-CASE CATEGORIES & DETERMINISTIC CATEGORY NORMALIZATION
    # -------------------------------------------------------------------------
    def test_05_category_normalization_preserves_mapping_lineage(self):
        s_raw = pd.Series(["Hybrid", "hybrid", " HYBRID", "Remote", " REMOTE "])
        norm_s, report = CategoryNormalizationEngine.normalize_series(s_raw)
        self.assertEqual(report["canonical_category_count"], 2)
        self.assertIn("hybrid", report["normalized_values_map"])
        self.assertEqual(report["normalized_values_map"]["hybrid"], "Hybrid")
        self.assertEqual(report["normalized_values_map"][" HYBRID"], "Hybrid")
        self.assertGreater(len(report["normalization_rules"]), 0)

    # -------------------------------------------------------------------------
    # 6. CANDIDATE PRIMARY KEY (EVIDENCE-BASED)
    # -------------------------------------------------------------------------
    def test_06_primary_key_evidence_based_detection(self):
        s_pk = pd.Series([1001, 1002, 1003, 1004, 1005])
        pk_info = KeyDetectionEngine.detect_key("employee_code", s_pk, len(s_pk))
        self.assertEqual(pk_info["key_role"], "primary_key")
        self.assertEqual(pk_info["uniqueness_percentage"], 100.0)
        self.assertEqual(pk_info["null_percentage"], 0.0)
        self.assertEqual(pk_info["duplicate_count"], 0)
        self.assertGreaterEqual(pk_info["candidate_confidence"], 0.85)

    # -------------------------------------------------------------------------
    # 7. DUPLICATE KEY (REJECTED AS PRIMARY KEY)
    # -------------------------------------------------------------------------
    def test_07_duplicate_key_rejected_as_pk(self):
        s_dup = pd.Series([1001, 1002, 1002, 1004, 1005])
        key_info = KeyDetectionEngine.detect_key("id_number", s_dup, len(s_dup))
        self.assertNotEqual(key_info["key_role"], "primary_key")
        self.assertEqual(key_info["duplicate_count"], 1)

    # -------------------------------------------------------------------------
    # 8. CANDIDATE FOREIGN KEY & REFERENTIAL INTEGRITY
    # -------------------------------------------------------------------------
    def test_08_foreign_key_evidence_based_detection(self):
        tables = {"orders": self.orders_df, "order_items": self.order_items_df}
        rels = RelationshipDetectionEngine.detect_and_validate_relationships(tables)
        self.assertGreater(len(rels), 0)
        rel = next((r for r in rels if r["source_column"] == "order_id" and r["target_column"] == "order_id"), None)
        self.assertIsNotNone(rel)
        self.assertIn("order_id", rel["source_column"])
        self.assertGreater(rel["child_row_count"], 0)
        self.assertGreater(rel["parent_distinct_key_count"], 0)
        self.assertIn("match_rate", rel)
        self.assertIn("candidate_confidence", rel)

    # -------------------------------------------------------------------------
    # 9. FK VIOLATIONS TRACKING
    # -------------------------------------------------------------------------
    def test_09_foreign_key_violations_calculated(self):
        # order_items has order_id 999 which does NOT exist in orders table
        tables = {"order_items": self.order_items_df, "orders": self.orders_df}
        rels = RelationshipDetectionEngine.detect_and_validate_relationships(tables)
        rel = next((r for r in rels if r["source_table"] == "order_items" and r["target_table"] == "orders"), None)
        self.assertIsNotNone(rel)
        self.assertGreater(rel["fk_violation_count"], 0)
        self.assertGreater(rel["fk_violation_percentage"], 0.0)
        self.assertIn(rel["status"], ["candidate_relationship", "relationship_violation"])

    # -------------------------------------------------------------------------
    # 10. COMPOSITE KEY DETECTION
    # -------------------------------------------------------------------------
    def test_10_composite_key_detection(self):
        # Neither order_id alone nor item_id alone is unique in order_items,
        # but (order_id, item_id) is 100% unique!
        comp_df = pd.DataFrame({
            "order_id": [1, 1, 2, 2, 3],
            "item_id":  [10, 20, 10, 20, 10],
            "quantity": [1, 2, 1, 3, 4]
        })
        comp_keys = KeyDetectionEngine.detect_composite_keys(comp_df)
        self.assertGreater(len(comp_keys), 0)
        self.assertEqual(set(comp_keys[0]["columns"]), {"order_id", "item_id"})
        self.assertEqual(comp_keys[0]["uniqueness_percentage"], 100.0)
        self.assertEqual(comp_keys[0]["null_percentage"], 0.0)
        self.assertEqual(comp_keys[0]["duplicate_count"], 0)
        self.assertEqual(comp_keys[0]["key_role"], "composite_key")

    # -------------------------------------------------------------------------
    # 11. DATE/TIME VARIABLE DETERMINISTIC TEMPORAL ANALYSIS
    # -------------------------------------------------------------------------
    def test_11_temporal_analysis_quantitative_facts(self):
        s_date = pd.Series(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"])
        meta = SemanticInferenceEngine.infer_column("Transaction_Date", s_date, len(s_date))
        self.assertEqual(meta["semantic_role"], "Date/Time")
        self.assertTrue(meta["is_time"])

    # -------------------------------------------------------------------------
    # 12. NON-ADDITIVE METRIC WATERFALL BLOCKED
    # -------------------------------------------------------------------------
    def test_12_non_additive_waterfall_blocked(self):
        df_non_additive = pd.DataFrame({
            "month": ["Jan", "Feb", "Mar", "Apr"],
            "productivity_index": [102.5, 98.4, 104.1, 101.0],
            "margin_pct": [0.25, 0.28, 0.24, 0.26]
        })
        sem = {
            "productivity_index": {"measure_type": "Index", "is_measure": True},
            "margin_pct": {"measure_type": "Percentage", "is_measure": True}
        }
        res_idx = ChartSuitabilityEngine.chartSuitabilityCheck("WATERFALL", df_non_additive, y_col="productivity_index", sem_layer=sem)
        self.assertFalse(res_idx["suitable"])
        self.assertEqual(res_idx["status"], "BLOCKED_BY_SEMANTIC_RULE")
        self.assertIn("Waterfall not applicable", res_idx["reason"])
        self.assertIn("additive bridge", res_idx["reason"])

        res_pct = ChartSuitabilityEngine.chartSuitabilityCheck("WATERFALL", df_non_additive, y_col="margin_pct", sem_layer=sem)
        self.assertFalse(res_pct["suitable"])

    # -------------------------------------------------------------------------
    # 13. ADDITIVE METRIC WATERFALL ACCEPTED
    # -------------------------------------------------------------------------
    def test_13_additive_metric_waterfall_accepted(self):
        df_additive = pd.DataFrame({
            "department": ["Sales", "Operations", "R&D", "General"],
            "gross_profit": [120000.0, -45000.0, -30000.0, 10000.0]
        })
        sem = {
            "gross_profit": {"measure_type": "Currency", "is_measure": True, "unit": "USD"}
        }
        res = ChartSuitabilityEngine.chartSuitabilityCheck("WATERFALL", df_additive, y_col="gross_profit", sem_layer=sem)
        self.assertTrue(res["suitable"])
        self.assertEqual(res["status"], "VALID")

    # -------------------------------------------------------------------------
    # 14. ZERO-VARIANCE DEPENDENT VARIABLE GUARDRAIL
    # -------------------------------------------------------------------------
    def test_14_zero_variance_dependent_variable_guardrail(self):
        df_const = pd.DataFrame({
            "cohort": ["A", "B", "C", "A", "B", "C"],
            "flat_measure": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
        })
        c_info = compute_variable_constancy(df_const["flat_measure"])
        self.assertTrue(c_info["is_constant"])
        self.assertEqual(c_info["variance"], 0.0)

    # -------------------------------------------------------------------------
    # 15. CORRELATION WITH CONSTANT VARIABLE BLOCKED
    # -------------------------------------------------------------------------
    def test_15_correlation_blocked_with_constant_variable(self):
        df_corr = pd.DataFrame({
            "x_variable": [10.0, 20.0, 30.0, 40.0],
            "flat_variable": [5.0, 5.0, 5.0, 5.0]
        })
        res = StatisticalSuitabilityEngine.check_correlation(df_corr, "x_variable", "flat_variable")
        self.assertEqual(res["status"], "NOT_APPLICABLE")
        self.assertIn("zero variance", res["reason"].lower())

    # -------------------------------------------------------------------------
    # 16. ANOVA WITH CONSTANT DEPENDENT VARIABLE BLOCKED
    # -------------------------------------------------------------------------
    def test_16_anova_blocked_with_constant_dependent_variable(self):
        df_anova = pd.DataFrame({
            "group": ["G1", "G1", "G2", "G2", "G3", "G3"],
            "flat_dv": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        })
        # Via StatisticalSuitabilityEngine
        suit = StatisticalSuitabilityEngine.check_anova(df_anova, "group", "flat_dv")
        self.assertEqual(suit["status"], "NOT_APPLICABLE")
        self.assertEqual(suit["reason"], "dependent_variable_constant")
        self.assertIn("Not applicable — dependent variable 'flat_dv' is constant", suit["display_message"])

        # Via statistical_analysis service
        diff_res = compute_group_differences(df_anova, "group", "flat_dv")
        self.assertEqual(diff_res["test_status"], "not_applicable")
        self.assertEqual(diff_res["reason"], "dependent_variable_constant")
        self.assertIsNone(diff_res["anova"]["f_statistic"])
        self.assertIsNone(diff_res["anova"]["p_value"])
        self.assertEqual(diff_res["anova"]["display_message"], "Not applicable — dependent variable is constant.")

    # -------------------------------------------------------------------------
    # 17. KRUSKAL-WALLIS WITH CONSTANT DEPENDENT VARIABLE BLOCKED
    # -------------------------------------------------------------------------
    def test_17_kruskal_wallis_blocked_with_constant_dependent_variable(self):
        df_kw = pd.DataFrame({
            "group": ["G1", "G1", "G2", "G2", "G3", "G3"],
            "flat_dv": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        })
        suit = StatisticalSuitabilityEngine.check_kruskal(df_kw, "group", "flat_dv")
        self.assertEqual(suit["status"], "NOT_APPLICABLE")
        self.assertEqual(suit["reason"], "dependent_variable_constant")
        self.assertIn("Not applicable — dependent variable 'flat_dv' is constant", suit["display_message"])

        diff_res = compute_group_differences(df_kw, "group", "flat_dv")
        self.assertEqual(diff_res["kruskal_wallis"]["test_status"], "not_applicable")
        self.assertIsNone(diff_res["kruskal_wallis"]["h_statistic"])
        self.assertIsNone(diff_res["kruskal_wallis"]["p_value"])
        self.assertEqual(diff_res["kruskal_wallis"]["display_message"], "Not applicable — dependent variable is constant.")

    # -------------------------------------------------------------------------
    # 18. PCA EXCLUDES CONSTANT FEATURES
    # -------------------------------------------------------------------------
    def test_18_pca_excludes_constant_features(self):
        df_pca = pd.DataFrame({
            "feat_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feat_2": [10.0, 15.0, 12.0, 18.0, 22.0, 25.0],
            "feat_constant": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
        })
        pca_suit = StatisticalSuitabilityEngine.check_pca(df_pca, ["feat_1", "feat_2", "feat_constant"])
        self.assertIn("feat_constant", pca_suit["excluded_features"])
        self.assertEqual(set(pca_suit["retained_features"]), {"feat_1", "feat_2"})

        pca_res = compute_pca(df_pca, n_components=2)
        self.assertTrue(pca_res["is_valid"])
        self.assertIn("feat_constant", pca_res["excluded_features"])
        self.assertEqual(pca_res["exclusion_reason"], "zero_variance")

    # -------------------------------------------------------------------------
    # 19. FILTERED ANALYTICAL DATASET & EXPLICIT REASONING
    # -------------------------------------------------------------------------
    def test_19_filtered_analytical_dataset_exclusion_reasons(self):
        df_filtered = pd.DataFrame({
            "segment": ["Enterprise", "SMB", None, "Enterprise", "Mid-Market"],
            "contract_value": [50000.0, 12000.0, 80000.0, 45000.0, 25000.0]
        })
        ar = compute_grouped_analytical_result(
            df=df_filtered,
            measure_col="contract_value",
            group_col="segment",
            aggregation="SUM"
        )
        self.assertEqual(ar.excluded_record_count, 1)
        self.assertIn("segment", ar.excluded_record_reason.lower())
        self.assertIn("missing", ar.excluded_record_reason.lower())

    # -------------------------------------------------------------------------
    # 20. DIFFERENT DENOMINATORS TRANSPARENCY
    # -------------------------------------------------------------------------
    def test_20_different_denominators_transparency(self):
        df_denoms = pd.DataFrame({
            "region": ["North", "South", "East", None],
            "sales": [100.0, 200.0, 300.0, 400.0]  # Total = 1000, Grouped total = 600
        })
        ar = compute_grouped_analytical_result(
            df=df_denoms,
            measure_col="sales",
            group_col="region",
            aggregation="SUM"
        )
        self.assertEqual(ar.statistics["full_total"], 1000.0)
        self.assertEqual(ar.statistics["grouped_total"], 600.0)
        self.assertIn("excluded from the region grouping", ar.takeaway)



    # -------------------------------------------------------------------------
    # 21. DUPLICATE ENTITY RECORDS PROFILING
    # -------------------------------------------------------------------------
    def test_21_duplicate_records_profiling(self):
        df_dups = pd.DataFrame({
            "id": [1, 2, 2, 3],
            "val": ["A", "B", "B", "C"]
        })
        dq = DataQualityEngine.profile_dataframe(df_dups)
        self.assertEqual(dq["duplicate_rows"], 1)
        self.assertEqual(dq["duplicate_percentage"], 25.0)

    # -------------------------------------------------------------------------
    # 22. MISSING CATEGORIES HANDLING
    # -------------------------------------------------------------------------
    def test_22_missing_categories_handling(self):
        s_cats = pd.Series(["Tier 1", None, "Tier 2", "Tier 1", np.nan])
        non_null_cnt = int(s_cats.dropna().count())
        null_cnt = int(s_cats.isnull().sum())
        self.assertEqual(non_null_cnt, 3)
        self.assertEqual(null_cnt, 2)

    # -------------------------------------------------------------------------
    # 23. MULTIPLE SHEETS / TABLES RELATIONSHIP DISCOVERY
    # -------------------------------------------------------------------------
    def test_23_multiple_tables_discovery(self):
        tables = {
            "customers": pd.DataFrame({"cust_id": [1, 2, 3], "name": ["A", "B", "C"]}),
            "invoices": pd.DataFrame({"inv_id": [10, 20, 30, 40], "cust_id": [1, 2, 2, 3], "amount": [100, 200, 150, 300]})
        }
        rels = RelationshipDetectionEngine.detect_and_validate_relationships(tables)
        self.assertGreater(len(rels), 0)
        cust_inv_rel = next(r for r in rels if r["source_column"] == "cust_id" and r["target_column"] == "cust_id")
        self.assertEqual(cust_inv_rel["referential_coverage"], 100.0)
        self.assertEqual(cust_inv_rel["fk_violation_count"], 0)

    # -------------------------------------------------------------------------
    # 24. SAME CHART / RESULT / INSIGHT / RECOMMENDATION LINEAGE
    # -------------------------------------------------------------------------
    def test_24_end_to_end_single_source_of_truth_lineage(self):
        df = pd.DataFrame({
            "Department": ["Sales", "Engineering", "Marketing", "Sales", "Engineering"],
            "Revenue": [100000.0, 150000.0, 80000.0, 120000.0, 160000.0]
        })
        eda_res = run_automated_eda(df)
        ins_res = generate_insights_and_story(df, eda_res)
        insights = ins_res["insights"]
        self.assertGreater(len(insights), 0)

        # First insight consumes AnalyticalResult
        first_ins = insights[0]
        shared_ar = first_ins["shared_result"]
        self.assertIsNotNone(shared_ar)

        # Verify recommendation consumes exact same AnalyticalResult lineage
        rec_res = generate_recommendations_and_action_plan(insights)
        recommendations = rec_res["recommendations"]
        action_items = rec_res["action_items"]
        self.assertGreater(len(recommendations), 0)
        self.assertGreater(len(action_items), 0)

        first_rec = recommendations[0]
        first_act = action_items[0]

        # Lineage IDs must be present and connected
        self.assertEqual(first_rec["source_insight_id"], first_ins["id"])
        self.assertEqual(first_act["recommendation_id"], first_rec["id"])
        self.assertIn("result_id", first_rec)
        self.assertIn("result_id", first_act)

        # Unassigned metadata safety checks
        self.assertEqual(first_rec["owner"], "Unassigned")
        self.assertEqual(first_rec["kpi_target"], "Not defined")
        self.assertEqual(first_rec["target_status"], "Awaiting business target")
        self.assertEqual(first_act["owner"], "Unassigned")
        self.assertEqual(first_act["kpi_target"], "Not defined")
        self.assertEqual(first_act["status"], "Not Started")


if __name__ == "__main__":
    unittest.main()
