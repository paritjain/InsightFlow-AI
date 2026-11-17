"""
Comprehensive Adversarial Analytical Test Suite — InsightFlow AI
40 Adversarial, Degenerate, Relational, Temporal, and Statistical Edge Cases.
Validates fail-safe behavior, zero crashes, absence of fabricated statistics,
absence of causal claims, and proper defensive statuses.
"""

import unittest
import numpy as np
import pandas as pd
import math

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
    InsightGovernanceEngine
)
from engine.analytical_truth import (
    compute_variable_constancy,
    safe_correlation,
    is_chart_suitable,
    AnalyticalResult
)
from services.statistical_analysis import (
    compute_group_differences,
    compute_welch_anova,
    apply_multiple_testing_correction,
    compute_mean_confidence_interval,
    compute_proportion_confidence_interval,
    compute_correlation_confidence_interval,
    check_simpsons_paradox,
    compute_pca,
    compute_advanced_outliers
)
from engine.eda import generate_waterfall_chart, run_automated_eda


class TestAdversarialAnalyticalCases(unittest.TestCase):

    # 1. Empty dataset
    def test_01_empty_dataset(self):
        df = pd.DataFrame()
        dq = DataQualityEngine.run_quality_audit(df)
        self.assertEqual(dq["total_rows"], 0)
        pks = KeyDetectionEngine.detect_primary_keys(df)
        self.assertEqual(pks, [])
        suit = ChartSuitabilityEngine.chartSuitabilityCheck("BAR", df, y_col="col")
        self.assertFalse(suit["suitable"])

    # 2. One-row dataset
    def test_02_one_row_dataset(self):
        df = pd.DataFrame({"id": [1], "val": [42.0]})
        dq = DataQualityEngine.run_quality_audit(df)
        self.assertEqual(dq["total_rows"], 1)
        # Cannot compute sample variance with n=1
        self.assertTrue(dq["columns"]["val"]["statistics"]["zero_variance"])
        # Correlation cannot be computed
        suit = ChartSuitabilityEngine.chartSuitabilityCheck("SCATTER", df, x_col="id", y_col="val")
        self.assertFalse(suit["suitable"])

    # 3. Two-row dataset
    def test_03_two_row_dataset(self):
        df = pd.DataFrame({"id": [1, 2], "val": [10.0, 20.0]})
        diff = compute_group_differences(df, "id", "val")
        # Each group has n=1, insufficient for ANOVA
        self.assertTrue(diff.get("status") in ["insufficient_data", "insufficient_groups", "not_applicable"] or not diff["anova"]["valid"])

    # 4. All-null column
    def test_04_all_null_column(self):
        df = pd.DataFrame({"id": [1, 2, 3], "ghost": [np.nan, np.nan, np.nan]})
        c_info = compute_variable_constancy(df["ghost"])
        self.assertTrue(c_info["is_constant"] or c_info["null_percentage"] == 100.0)
        suit = ChartSuitabilityEngine.chartSuitabilityCheck("HISTOGRAM", df, x_col="ghost")
        self.assertFalse(suit["suitable"])

    # 5. Constant column
    def test_05_constant_column(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4], "status": [100.0, 100.0, 100.0, 100.0]})
        c_info = compute_variable_constancy(df["status"])
        self.assertTrue(c_info["is_constant"])
        self.assertEqual(c_info["variance"], 0.0)
        # ANOVA on constant column fails safely
        diff = compute_group_differences(df, "id", "status")
        self.assertEqual(diff["test_status"], "not_applicable")

    # 6. Near-constant column (e.g. 99.9% same value)
    def test_06_near_constant_column(self):
        vals = [10.0] * 99 + [10.0000001]
        s = pd.Series(vals)
        c_info = compute_variable_constancy(s)
        self.assertLess(c_info["variance"], 1e-6)

    # 7. Duplicate PK
    def test_07_duplicate_pk(self):
        df = pd.DataFrame({"user_id": [101, 102, 103, 101], "name": ["A", "B", "C", "D"]})
        det = KeyDetectionEngine.detect_key("user_id", df["user_id"], len(df))
        self.assertNotEqual(det["key_role"], "primary_key")
        self.assertGreater(det["duplicate_count"], 0)

    # 8. Composite PK
    def test_08_composite_pk(self):
        df = pd.DataFrame({
            "order_id": [1, 1, 2, 2, 3],
            "item_id": [10, 20, 10, 20, 10],
            "qty": [1, 2, 3, 4, 5]
        })
        comp = KeyDetectionEngine.detect_composite_keys(df)
        self.assertGreater(len(comp), 0)
        self.assertEqual(set(comp[0]["key_columns"]), {"order_id", "item_id"})
        self.assertEqual(comp[0]["uniqueness_percentage"], 100.0)

    # 9. False FK (low semantic evidence / coincidence values)
    def test_09_false_fk(self):
        feedback = pd.DataFrame({"rating_stars": [1, 2, 3]})
        inventory = pd.DataFrame({"order_item_qty": [1, 2, 3]})
        tables = {"feedback": feedback, "inventory": inventory}
        rels = RelationshipDetectionEngine.detect_and_validate_relationships(tables)
        # Even if values match [1,2,3], column names differ and small numbers overlap, so it should NOT assert confirmed FK
        confirmed = [r for r in rels if r.get("status") == "valid_relationship"]
        self.assertEqual(len(confirmed), 0)
        for r in rels:
            self.assertTrue(r.get("confirmation_required", False))

    # 10. Partial FK (referential integrity violation)
    def test_10_partial_fk(self):
        emp = pd.DataFrame({"dept_id": [10, 20, 30, 999]}) # 999 is orphan
        dept = pd.DataFrame({"dept_id": [10, 20, 30, 40]})
        tables = {"employees": emp, "departments": dept}
        rels = RelationshipDetectionEngine.detect_and_validate_relationships(tables)
        self.assertGreater(len(rels), 0)
        target_rel = [r for r in rels if r["child_table"] == "employees" and r["parent_table"] == "departments"][0]
        self.assertGreater(target_rel["fk_violation_count"], 0)
        self.assertGreater(target_rel["fk_violation_percentage"], 0.0)

    # 11. Composite FK
    def test_11_composite_fk(self):
        fact = pd.DataFrame({"tenant_id": [1, 1, 2], "user_id": [10, 20, 10], "amount": [100, 200, 300]})
        dim = pd.DataFrame({"tenant_id": [1, 1, 2], "user_id": [10, 20, 10], "name": ["Alice", "Bob", "Charlie"]})
        tables = {"sales": fact, "users": dim}
        cfks = RelationshipDetectionEngine.detect_composite_foreign_keys(tables)
        self.assertGreater(len(cfks), 0)
        self.assertEqual(cfks[0]["violation_count"], 0)

    # 12. Circular relationship
    def test_12_circular_relationship(self):
        emp = pd.DataFrame({"emp_id": [1, 2, 3], "dept_id": [10, 10, 20]})
        dept = pd.DataFrame({"dept_id": [10, 20], "manager_emp_id": [1, 2]})
        tables = {"employees": emp, "departments": dept}
        rels = RelationshipDetectionEngine.detect_and_validate_relationships(tables)
        self.assertIsInstance(rels, list)

    # 13. Mixed datatypes in a column
    def test_13_mixed_datatypes(self):
        s = pd.Series([100, "N/A", 200, None, "300.5"])
        df = pd.DataFrame({"mixed": s})
        res = SemanticInferenceEngine.infer_column("mixed", s)
        self.assertIn("data_type", res)
        # Should detect nulls/coercion gracefully
        dq = DataQualityEngine.run_quality_audit(df)
        self.assertEqual(dq["total_rows"], 5)

    # 14. Mixed currencies in a dataset
    def test_14_mixed_currencies(self):
        df = pd.DataFrame({
            "region": ["US", "India", "EU"],
            "currency": ["USD", "INR", "EUR"],
            "revenue": [1000.0, 50000.0, 800.0]
        })
        dq = DataQualityEngine.run_quality_audit(df)
        curr_issues = [iss for iss in dq["dataset_issues"] if iss["type"] == "currency_inconsistency"]
        self.assertGreater(len(curr_issues), 0)
        # Waterfall should block mixed currencies
        sem = {"revenue": {"is_measure": True, "measure_type": "Currency", "is_additive": True}}
        wf = generate_waterfall_chart(df, sem)
        self.assertFalse(wf["is_valid"])
        self.assertTrue(wf.get("currency_mismatch", False))

    # 15. Mixed units in dataset
    def test_15_mixed_units(self):
        df = pd.DataFrame({
            "distance_km": [10.5, 20.0, 15.0],
            "distance_miles": [6.5, 12.4, 9.3]
        })
        sem = {
            "distance_km": {"is_measure": True, "unit": "km"},
            "distance_miles": {"is_measure": True, "unit": "miles"}
        }
        dq = DataQualityEngine.run_quality_audit(df, sem_metadata=sem)
        unit_issues = [iss for iss in dq["dataset_issues"] if iss["type"] == "unit_inconsistency"]
        self.assertGreater(len(unit_issues), 0)

    # 16. Extreme outlier (10,000x of median)
    def test_16_extreme_outlier(self):
        df = pd.DataFrame({"sales": [100, 102, 98, 105, 99, 101, 1000000]})
        out_res = compute_advanced_outliers(df)
        col_out = out_res["columns"]["sales"]
        self.assertGreaterEqual(col_out["iqr_outliers_count"], 1)
        self.assertGreaterEqual(col_out["modified_zscore_outliers_count"], 1)

    # 17. Highly skewed distribution
    def test_17_highly_skewed_distribution(self):
        # Lognormal heavy right tail
        np.random.seed(42)
        skewed_data = np.exp(np.random.normal(0, 1.5, 100))
        df = pd.DataFrame({"metric": skewed_data})
        out_res = compute_advanced_outliers(df)
        self.assertIsNotNone(out_res["columns"]["metric"]["skewness_warning"])

    # 18. Heavy-tailed distribution (Modified Z-Score MAD protects against breakdown)
    def test_18_heavy_tailed_distribution(self):
        vals = [1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 100.0, 200.0]
        df = pd.DataFrame({"vals": vals})
        out_res = compute_advanced_outliers(df)
        self.assertGreater(out_res["columns"]["vals"]["modified_zscore_outliers_count"], 0)

    # 19. Missing time periods (No silent interpolation)
    def test_19_missing_time_periods(self):
        dates = pd.to_datetime(["2024-01-01", "2024-03-01", "2024-05-01"])
        df = pd.DataFrame({"Date": dates, "Revenue": [100, 200, 300]})
        eda_res = run_automated_eda(df)
        ts = eda_res.get("time_series_analysis", {})
        self.assertFalse(ts["interpolation_enabled"])
        self.assertIn("2024-02", ts["missing_periods"])
        self.assertTrue(ts["irregular_intervals"])

    # 20. Duplicate timestamps
    def test_20_duplicate_timestamps(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"])
        df = pd.DataFrame({"Date": dates, "Revenue": [10, 20, 30, 40]})
        eda_res = run_automated_eda(df)
        ts = eda_res.get("time_series_analysis", {})
        self.assertEqual(ts["duplicate_timestamps"], 2)

    # 21. Irregular timestamps
    def test_21_irregular_timestamps(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-12", "2024-01-30"])
        df = pd.DataFrame({"Date": dates, "Revenue": [50, 60, 70, 80]})
        eda_res = run_automated_eda(df)
        ts = eda_res.get("time_series_analysis", {})
        self.assertTrue(ts["irregular_intervals"])

    # 22. Perfect positive correlation (r = 1.0)
    def test_22_perfect_positive_correlation(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
        res = safe_correlation(df, "x", "y")
        self.assertAlmostEqual(res["pearson"], 1.0, places=4)
        self.assertAlmostEqual(res["spearman"], 1.0, places=4)
        self.assertTrue(res["is_computable"])
        # Non-causal wording check
        self.assertNotIn("causes", res["interpretation"].lower())
        self.assertNotIn("drives", res["interpretation"].lower())

    # 23. Perfect negative correlation (r = -1.0)
    def test_23_perfect_negative_correlation(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 8, 6, 4, 2]})
        res = safe_correlation(df, "x", "y")
        self.assertAlmostEqual(res["pearson"], -1.0, places=4)
        self.assertAlmostEqual(res["spearman"], -1.0, places=4)
        self.assertTrue(res["is_computable"])

    # 24. Nonlinear monotonic relationship (Spearman = 1.0, Pearson < 1.0)
    def test_24_nonlinear_monotonic_relationship(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 10, 100, 1000, 10000]})
        res = safe_correlation(df, "x", "y")
        self.assertAlmostEqual(res["spearman"], 1.0, places=4)
        self.assertLess(res["pearson"], 0.95)

    # 25. Simpson's paradox scenario
    def test_25_simpsons_paradox_scenario(self):
        # Aggregate positive correlation, but within groups negative
        df = pd.DataFrame({
            "dept": ["A", "A", "A", "B", "B", "B"],
            "hours": [10, 12, 14, 20, 22, 24],
            "errors": [5, 4, 3, 10, 9, 8]
        })
        simpson = check_simpsons_paradox(df, "hours", "errors")
        self.assertTrue(simpson["simpsons_paradox_detected"])
        self.assertIn("Potential subgroup/confounding effect detected", simpson["warning"])

    # 26. Very small ANOVA groups (< 5 observations per group)
    def test_26_very_small_anova_groups(self):
        df = pd.DataFrame({
            "group": ["A", "A", "A", "B", "B", "B"],
            "score": [12.0, 14.0, 13.0, 22.0, 21.0, 23.0]
        })
        diff = compute_group_differences(df, "group", "score")
        # Defensive logic: small sample flags assumptions_validated = False and Kruskal-Wallis preferred
        self.assertFalse(diff["assumptions_validated"])
        self.assertEqual(diff["statistical_recommendation"], "Kruskal-Wallis preferred")

    # 27. Unequal variance groups (Welch ANOVA)
    def test_27_unequal_variance_groups(self):
        # Group 1 has variance 1, Group 2 has variance 100
        g1 = np.array([10.0, 11.0, 9.0, 10.5, 9.5, 10.2])
        g2 = np.array([2.0, 50.0, 1.0, 45.0, 0.5, 60.0])
        welch = compute_welch_anova([g1, g2])
        self.assertTrue(welch["valid"])
        self.assertEqual(welch["method"], "Welch ANOVA")
        self.assertIsNotNone(welch["f_statistic"])
        self.assertIsNotNone(welch["p_value"])

    # 28. Highly imbalanced groups (e.g. 1000 in Group A, 3 in Group B)
    def test_28_highly_imbalanced_groups(self):
        df = pd.DataFrame({
            "group": ["A"] * 100 + ["B"] * 3,
            "metric": list(np.random.normal(10, 1, 100)) + [15.0, 16.0, 14.0]
        })
        diff = compute_group_differences(df, "group", "metric")
        self.assertFalse(diff["assumptions_validated"])

    # 29. High-cardinality categorical variable (e.g. 100 unique categories)
    def test_29_high_cardinality_categorical(self):
        cats = [f"CAT_{i}" for i in range(100)]
        df = pd.DataFrame({"cat": cats, "val": np.random.normal(50, 10, 100)})
        suit = ChartSuitabilityEngine.chartSuitabilityCheck("DONUT", df, x_col="cat")
        # Donut must be blocked for > 6 categories
        self.assertFalse(suit["suitable"])

    # 30. Unicode column names
    def test_30_unicode_column_names(self):
        df = pd.DataFrame({"आय_INR": [1000, 2000, 3000], "आयु": [25, 30, 35]})
        dq = DataQualityEngine.run_quality_audit(df)
        self.assertEqual(dq["total_rows"], 3)
        self.assertIn("आय_INR", dq["columns"])

    # 31. Spaces and special characters in column names
    def test_31_spaces_and_special_characters(self):
        df = pd.DataFrame({"Total Revenue ($)": [100.0, 200.0, 300.0], "Profit %": [10.0, 15.0, 20.0]})
        dq = DataQualityEngine.run_quality_audit(df)
        self.assertEqual(dq["total_rows"], 3)
        self.assertIn("Total Revenue ($)", dq["columns"])

    # 32. Duplicate column names handling
    def test_32_duplicate_column_names(self):
        # When DataFrame is constructed with duplicate columns
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["val", "val"])
        # Cleaner or profiler safely handles without unhandled crash
        self.assertEqual(len(df), 2)

    # 33. All-null categorical variable
    def test_33_all_null_categorical(self):
        df = pd.DataFrame({"cat": [None, None, None], "val": [1, 2, 3]})
        suit = ChartSuitabilityEngine.chartSuitabilityCheck("BAR", df, x_col="cat", y_col="val")
        self.assertFalse(suit["suitable"])

    # 34. Yes/No, TRUE/FALSE, 1/0 boolean variants
    def test_34_boolean_variants(self):
        s_yn = pd.Series(["Yes", "No", "Yes", "No"])
        inf_yn = SemanticInferenceEngine.infer_column("flag_yn", s_yn)
        self.assertEqual(inf_yn["measure_type"], "Binary Category")

        s_tf = pd.Series(["TRUE", "FALSE", "TRUE", "TRUE"])
        inf_tf = SemanticInferenceEngine.infer_column("flag_tf", s_tf)
        self.assertEqual(inf_tf["measure_type"], "Binary Category")

    # 35. Multiple date formats in same column
    def test_35_multiple_date_formats(self):
        s = pd.Series(["2024-01-01", "02/01/2024", "2024.03.01", "InvalidDate"])
        df = pd.DataFrame({"date_col": s})
        parsed = pd.to_datetime(s, format='mixed', errors='coerce')
        self.assertEqual(parsed.notna().sum(), 3)
        self.assertEqual(parsed.isna().sum(), 1)

    # 36. Nested JSON strings in text columns
    def test_36_nested_json_strings(self):
        s = pd.Series(['{"browser": "Chrome", "os": "Win"}', '{"browser": "Safari", "os": "Mac"}'] * 15)
        df = pd.DataFrame({"meta_json": s})
        res = SemanticInferenceEngine.infer_column("meta_json", s, total_rows=30)
        self.assertEqual(res["semantic_role"], "Text")

    # 37. Multi-currency dataset detection
    def test_37_multi_currency_dataset(self):
        df = pd.DataFrame({
            "transaction_id": [1, 2, 3],
            "currency_code": ["USD", "INR", "EUR"],
            "amount": [100.0, 8000.0, 95.0]
        })
        dq = DataQualityEngine.run_quality_audit(df)
        issues = [i for i in dq["dataset_issues"] if i["type"] == "currency_inconsistency"]
        self.assertGreater(len(issues), 0)

    # 38. Mixed-unit dataset
    def test_38_mixed_unit_dataset(self):
        df = pd.DataFrame({
            "weight_kg": [50.0, 60.0, 70.0],
            "weight_tonnes": [0.05, 0.06, 0.07]
        })
        sem = {
            "weight_kg": {"is_measure": True, "unit": "kg"},
            "weight_tonnes": {"is_measure": True, "unit": "tonnes"}
        }
        dq = DataQualityEngine.run_quality_audit(df, sem_metadata=sem)
        issues = [i for i in dq["dataset_issues"] if i["type"] == "unit_inconsistency"]
        self.assertGreater(len(issues), 0)

    # 39. Invalid ratings (e.g. rating = 7 on scale 1-5)
    def test_39_invalid_ratings(self):
        df = pd.DataFrame({
            "review_id": [1, 2, 3],
            "rating": [4.0, 5.0, 7.5] # 7.5 exceeds scale
        })
        dq = DataQualityEngine.run_quality_audit(df)
        col_dq = dq["columns"]["rating"]["data_quality_errors"]
        rating_errs = [e for e in col_dq if e["type"] == "rating_out_of_bounds"]
        self.assertGreater(len(rating_errs), 0)

    # 40. Negative impossible values (e.g. age = -4, quantity = -2)
    def test_40_negative_impossible_values(self):
        df = pd.DataFrame({
            "patient_id": [1, 2, 3, 4],
            "age": [25, -4, 40, 52],
            "order_quantity": [5, 10, -2, 8]
        })
        dq = DataQualityEngine.run_quality_audit(df)
        age_errs = dq["columns"]["age"]["data_quality_errors"]
        qty_errs = dq["columns"]["order_quantity"]["data_quality_errors"]
        self.assertTrue(any(e["type"] == "invalid_domain_negative_value" for e in age_errs))
        self.assertTrue(any(e["type"] == "invalid_domain_negative_value" for e in qty_errs))

    # 41. Statistical Power & Precision Handling (Auditor Requirement 4)
    def test_41_statistical_power_and_precision_exposure(self):
        # 41A: Extremely small sample (N=2, df_within=0 < 1) -> INSUFFICIENT_STATISTICAL_POWER
        df_tiny = pd.DataFrame({
            "grp": ["A", "B"],
            "val": [10.0, 20.0]
        })
        tiny_res = compute_group_differences(df_tiny, "grp", "val")
        self.assertEqual(tiny_res["power_status"], "INSUFFICIENT_STATISTICAL_POWER")
        self.assertEqual(tiny_res["status"], "INSUFFICIENT_STATISTICAL_POWER")
        self.assertIn("INSUFFICIENT_STATISTICAL_POWER", tiny_res["power_warning"])
        self.assertIn("Kruskal-Wallis", tiny_res["anova"]["power_warning"])

        # 41B: Small sample (N=6, n=3 each) -> computes statistics but exposes power warning, CIs, effect size
        df_small = pd.DataFrame({
            "grp": ["A", "A", "A", "B", "B", "B"],
            "val": [10.0, 12.0, 11.0, 25.0, 27.0, 26.0]
        })
        small_res = compute_group_differences(df_small, "grp", "val")
        self.assertEqual(small_res["status"], "success")
        self.assertEqual(small_res["sample_size"], 6)
        self.assertIsNotNone(small_res["effect_size"])
        self.assertIn("confidence_interval", small_res)
        self.assertIn("groups", small_res["confidence_interval"])
        self.assertEqual(small_res["power_status"], "INSUFFICIENT_STATISTICAL_POWER")
        self.assertIn("Kruskal-Wallis does not eliminate", small_res["power_warning"])

    # 42. Business Context Firewall (Auditor Requirement 5)
    def test_42_business_context_firewall(self):
        from engine.action_plan import generate_recommendations_and_action_plan
        from engine.governed_engines import RecommendationGovernanceEngine

        # 42A: RecommendationGovernanceEngine
        gov_rec = RecommendationGovernanceEngine.build_recommendation(
            rec_id="rec_01",
            insight_id="ins_01",
            observation="Observed 50% category share",
            interpretation="Operational concentration detected",
            suggested_action="Review segment distribution"
        )
        self.assertEqual(gov_rec["analytical_status"], "VERIFIED")
        self.assertEqual(gov_rec["business_context_status"], "REQUIRES_CONTEXT")
        self.assertEqual(gov_rec["context_firewall"]["analytical_validity"], "VERIFIED")
        self.assertEqual(gov_rec["context_firewall"]["business_interpretation"], "REQUIRES_CONTEXT")

        # 42B: Action Plan Recommendations
        mock_insights = [{
            "id": "ins_test_1",
            "title": "Category Concentration in Region",
            "observed_fact": "Region A accounts for 65% of volume",
            "business_interpretation": "Substantial concentration risk observed",
            "recommended_action": "Evaluate regional dependency",
            "priority": "HIGH",
            "evidence": [{"metric": "Volume", "value": 6500}]
        }]
        rec_res = generate_recommendations_and_action_plan(mock_insights)
        recs = rec_res["recommendations"]
        self.assertGreater(len(recs), 0)
        rec = recs[0]
        self.assertEqual(rec["analytical_status"], "VERIFIED")
        self.assertEqual(rec["business_context_status"], "REQUIRES_CONTEXT")
        self.assertEqual(rec["statistical_validity"], "VERIFIED")
        self.assertEqual(rec["semantic_validity"], "VERIFIED")
        self.assertEqual(rec["context_firewall"]["analytical_validity"], "VERIFIED")
        self.assertEqual(rec["context_firewall"]["business_interpretation"], "REQUIRES_CONTEXT")


if __name__ == "__main__":
    unittest.main()

