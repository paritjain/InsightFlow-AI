"""
Automated Test Suite for the 30 Required Scenarios.
Section 33 of MASTER PROMPT:
1. Constant numeric variable
2. Single-category categorical variable
3. Mixed-case categorical values
4. Numeric count variable
5. Duration variable
6. Currency variable
7. Score/rating variable
8. Percentage variable
9. Candidate primary key
10. Candidate foreign key
11. FK violation
12. Duplicate records
13. Missing values
14. Statistical outliers
15. Invalid values
16. Non-additive metric
17. Additive metric
18. Time-series data
19. Missing time periods
20. Repeated entities across time
21. Strong correlation
22. Zero-variance correlation attempt
23. ANOVA with valid variance
24. ANOVA with zero variance
25. PCA with constant variables
26. Chart using filtered data
27. Chart using grouped data
28. Recommendation without business context
29. User-defined KPI target
30. User-defined owner/timeline
"""

import unittest
import numpy as np
import pandas as pd
import datetime

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
    is_chart_suitable,
    safe_anova,
    safe_kruskal,
    safe_correlation,
    safe_pca_features,
    compute_grouped_analytical_result
)


class TestGovernedAnalyticalEngine30(unittest.TestCase):

    # =========================================================================
    # 1. Constant numeric variable
    # =========================================================================
    def test_01_constant_numeric_variable(self):
        s = pd.Series([22.0] * 100)
        c_info = KeyDetectionEngine.compute_constancy(s)
        self.assertTrue(c_info["is_constant"])
        self.assertEqual(c_info["variance"], 0.0)
        self.assertEqual(c_info["nunique"], 1)

        # Chart suitability must suppress Histograms and Box Plots
        df = pd.DataFrame({"constant_num": s})
        hist_check = ChartSuitabilityEngine.chartSuitabilityCheck("HISTOGRAM", df, x_col="constant_num")
        self.assertFalse(hist_check["suitable"])
        self.assertEqual(hist_check["status"], "BLOCKED_BY_VARIANCE")

    # =========================================================================
    # 2. Single-category categorical variable
    # =========================================================================
    def test_02_single_category_categorical_variable(self):
        df = pd.DataFrame({
            "mode": ["Office"] * 50,
            "metric": list(range(50))
        })
        # Violin plot requires >= 2 distinct categories
        v_check = ChartSuitabilityEngine.chartSuitabilityCheck("VIOLIN", df, x_col="mode", y_col="metric")
        self.assertFalse(v_check["suitable"])
        self.assertIn("at least 2 distinct categories", v_check["reason"])

    # =========================================================================
    # 3. Mixed-case categorical values
    # =========================================================================
    def test_03_mixed_case_categorical_values(self):
        raw_series = pd.Series(["Hybrid", "hybrid", "HYBRID", "Remote", "remote", "Office"])
        norm_s, report = CategoryNormalizationEngine.normalize_series(raw_series)
        self.assertEqual(report["original_category_count"], 6)
        self.assertEqual(report["canonical_category_count"], 3)
        self.assertIn("Hybrid", report["canonical_category_count"] and list(norm_s.unique()))
        self.assertIn("Remote", list(norm_s.unique()))
        self.assertIn("Office", list(norm_s.unique()))
        self.assertEqual(norm_s.iloc[0], norm_s.iloc[1])
        self.assertEqual(norm_s.iloc[1], norm_s.iloc[2])

    # =========================================================================
    # 4. Numeric count variable
    # =========================================================================
    def test_04_numeric_count_variable(self):
        s = pd.Series([22, 22, 21, 20, 22])
        meta = SemanticInferenceEngine.infer_column("Working_Days", s)
        self.assertEqual(meta["semantic_role"], "Measure")
        self.assertEqual(meta["measure_type"], "Count")
        self.assertEqual(meta["unit"], "days")
        self.assertNotEqual(meta["measure_type"], "Duration")

    # =========================================================================
    # 5. Duration variable
    # =========================================================================
    def test_05_duration_variable(self):
        s = pd.Series([5.5, 7.2, 0.0, 3.4, 8.1])
        meta = SemanticInferenceEngine.infer_column("Overtime_Hours", s)
        self.assertEqual(meta["semantic_role"], "Measure")
        self.assertEqual(meta["measure_type"], "Duration")
        self.assertEqual(meta["unit"], "hours")

    # =========================================================================
    # 6. Currency variable
    # =========================================================================
    def test_06_currency_variable(self):
        s = pd.Series(["$1,200", "$1,500", "$2,000", "$800"])
        meta = SemanticInferenceEngine.infer_column("Revenue", s)
        self.assertEqual(meta["measure_type"], "Currency")
        self.assertEqual(meta["currency_metadata"]["code"], "USD")
        self.assertEqual(meta["currency_metadata"]["symbol"], "$")

    # =========================================================================
    # 7. Score/rating variable
    # =========================================================================
    def test_07_score_rating_variable(self):
        s = pd.Series([78.5, 82.0, 91.2, 65.0, 88.4])
        meta = SemanticInferenceEngine.infer_column("Productivity_Index", s)
        self.assertEqual(meta["semantic_role"], "Measure")
        self.assertEqual(meta["measure_type"], "Score / Index")
        self.assertEqual(meta["unit"], "index")
        self.assertIn(meta["rating_score_metadata"]["scale_type"], ["0-100", "unknown"])

    # =========================================================================
    # 8. Percentage variable
    # =========================================================================
    def test_08_percentage_variable(self):
        s = pd.Series(["15%", "25%", "10%", "5%"])
        meta = SemanticInferenceEngine.infer_column("Discount_Pct", s)
        self.assertEqual(meta["measure_type"], "Percentage")
        self.assertEqual(meta["unit"], "percent")

    # =========================================================================
    # 9. Candidate primary key
    # =========================================================================
    def test_09_candidate_primary_key(self):
        s = pd.Series([f"EMP_{i:04d}" for i in range(100)])
        key_res = KeyDetectionEngine.detect_key("Employee_ID", s, total_rows=100)
        self.assertEqual(key_res["key_role"], "primary_key")
        self.assertEqual(key_res["uniqueness_ratio"], 1.0)
        self.assertEqual(key_res["null_count"], 0)

    # =========================================================================
    # 10. Candidate foreign key
    # =========================================================================
    def test_10_candidate_foreign_key(self):
        parent_df = pd.DataFrame({"Emp_ID": [f"E{i}" for i in range(50)], "Name": ["Emp"] * 50})
        child_df = pd.DataFrame({"Record_ID": range(100), "Emp_ID": [f"E{i % 50}" for i in range(100)]})
        tables = {"Employees": parent_df, "Attendance": child_df}
        
        rels = RelationshipDetectionEngine.detect_and_validate_relationships(tables)
        self.assertTrue(len(rels) > 0)
        rel = next(r for r in rels if r["source_table"] == "Attendance" and r["source_column"] == "Emp_ID")
        self.assertEqual(rel["target_table"], "Employees")
        self.assertEqual(rel["relationship_type"], "many-to-one")
        self.assertEqual(rel["referential_coverage"], 100.0)
        self.assertEqual(rel["orphan_count"], 0)

    # =========================================================================
    # 11. FK violation
    # =========================================================================
    def test_11_fk_violation(self):
        parent_df = pd.DataFrame({"Dept_ID": ["D1", "D2", "D3"]})
        child_df = pd.DataFrame({"Emp_ID": [1, 2, 3, 4], "Dept_ID": ["D1", "D2", "D99", "D100"]})
        tables = {"Departments": parent_df, "Employees": child_df}
        
        rels = RelationshipDetectionEngine.detect_and_validate_relationships(tables)
        rel = next(r for r in rels if r["source_table"] == "Employees" and r["source_column"] == "Dept_ID")
        self.assertEqual(rel["orphan_count"], 2)
        self.assertEqual(rel["fk_violation_count"], 2)
        self.assertEqual(rel["referential_coverage"], 50.0)

    # =========================================================================
    # 12. Duplicate records
    # =========================================================================
    def test_12_duplicate_records(self):
        df = pd.DataFrame({
            "A": [1, 2, 2, 3],
            "B": ["x", "y", "y", "z"]
        })
        dq = DataQualityEngine.profile_dataframe(df)
        self.assertEqual(dq["duplicate_rows"], 1)
        self.assertEqual(dq["duplicate_percentage"], 25.0)

    # =========================================================================
    # 13. Missing values
    # =========================================================================
    def test_13_missing_values(self):
        df = pd.DataFrame({
            "metric": [10.0, None, 20.0, None, 30.0]
        })
        dq = DataQualityEngine.profile_dataframe(df)
        self.assertEqual(dq["columns"]["metric"]["null_count"], 2)
        self.assertEqual(dq["columns"]["metric"]["null_percentage"], 40.0)

    # =========================================================================
    # 14. Statistical outliers
    # =========================================================================
    def test_14_statistical_outliers(self):
        # A normal distribution with one extreme point
        vals = [10.0, 11.0, 10.5, 9.8, 10.2, 10.1, 9.9, 100.0]
        df = pd.DataFrame({"measure": vals})
        sem = {"measure": {"is_measure": True, "is_identifier": False}}
        dq = DataQualityEngine.profile_dataframe(df, sem_metadata=sem)
        
        # Outlier is detected via IQR
        self.assertEqual(dq["columns"]["measure"]["statistical_outliers"], 1)
        # But data_quality_errors should NOT treat a statistical outlier as an error automatically
        self.assertEqual(len(dq["columns"]["measure"]["data_quality_errors"]), 0)

    # =========================================================================
    # 15. Invalid values
    # =========================================================================
    def test_15_invalid_values(self):
        # Negative working days is an invalid domain value
        df = pd.DataFrame({"Working_Days": [22, -5, 22, -10]})
        sem = {"Working_Days": {"is_measure": True, "measure_type": "Count", "unit": "days"}}
        dq = DataQualityEngine.profile_dataframe(df, sem_metadata=sem)
        errs = dq["columns"]["Working_Days"]["data_quality_errors"]
        self.assertTrue(len(errs) > 0)
        self.assertEqual(errs[0]["type"], "invalid_domain_negative_value")
        self.assertEqual(errs[0]["count"], 2)

    # =========================================================================
    # 16. Non-additive metric
    # =========================================================================
    def test_16_non_additive_metric(self):
        df = pd.DataFrame({
            "Profit_Margin_Pct": [0.15, 0.22, 0.18, 0.25],
            "Productivity_Score": [85, 90, 78, 88]
        })
        sem = {
            "Profit_Margin_Pct": {"measure_type": "Percentage"},
            "Productivity_Score": {"measure_type": "Score"}
        }
        # Waterfall must be suppressed for non-additive percentages or scores
        wf_pct = ChartSuitabilityEngine.chartSuitabilityCheck("WATERFALL", df, y_col="Profit_Margin_Pct", sem_layer=sem)
        self.assertFalse(wf_pct["suitable"])
        self.assertIn("Waterfall unavailable", wf_pct["reason"])

        wf_score = ChartSuitabilityEngine.chartSuitabilityCheck("WATERFALL", df, y_col="Productivity_Score", sem_layer=sem)
        self.assertFalse(wf_score["suitable"])
        self.assertIn("Waterfall unavailable", wf_score["reason"])

    # =========================================================================
    # 17. Additive metric
    # =========================================================================
    def test_17_additive_metric(self):
        df = pd.DataFrame({
            "Revenue": [1000.0, 1500.0, 2000.0, 1200.0]
        })
        sem = {"Revenue": {"measure_type": "Currency"}}
        wf = ChartSuitabilityEngine.chartSuitabilityCheck("WATERFALL", df, y_col="Revenue", sem_layer=sem)
        self.assertTrue(wf["suitable"])
        self.assertEqual(wf["status"], "VALID")

    # =========================================================================
    # 18. Time-series data
    # =========================================================================
    def test_18_time_series_data(self):
        dates = pd.date_range("2025-01-01", periods=12, freq="MS")
        df = pd.DataFrame({
            "Date": dates,
            "Sales": [100 + i * 10 for i in range(12)]
        })
        line_check = ChartSuitabilityEngine.chartSuitabilityCheck("LINE", df, x_col="Date", y_col="Sales")
        self.assertTrue(line_check["suitable"])

    # =========================================================================
    # 19. Missing time periods
    # =========================================================================
    def test_19_missing_time_periods(self):
        # Dates skip February and April
        dates = pd.to_datetime(["2025-01-01", "2025-03-01", "2025-05-01"])
        df = pd.DataFrame({"Month": dates, "Working_Days": [22, 22, 22]})
        # Constant metric statement across recorded months
        const_info = KeyDetectionEngine.compute_constancy(df["Working_Days"])
        takeaway, interp = InsightGovernanceEngine.formulate_factual_takeaway(
            metric="Working_Days",
            is_constant=True,
            const_val=22.0,
            unit="days"
        )
        self.assertIn("constant at 22", takeaway)
        self.assertNotIn("consistent progression", takeaway.lower())

    # =========================================================================
    # 20. Repeated entities across time
    # =========================================================================
    def test_20_repeated_entities_across_time(self):
        # 10 employees tracked across 12 months = 120 records
        emp_ids = [f"E{i}" for i in range(10)] * 12
        df = pd.DataFrame({
            "Employee_ID": emp_ids,
            "Working_Days": [22] * 120
        })
        # Key role of Employee_ID in longitudinal table is NOT primary key because it repeats
        key_res = KeyDetectionEngine.detect_key("Employee_ID", df["Employee_ID"], total_rows=120)
        self.assertNotEqual(key_res["key_role"], "primary_key")
        self.assertEqual(key_res["duplicate_count"], 110)

    # =========================================================================
    # 21. Strong correlation
    # =========================================================================
    def test_21_strong_correlation(self):
        x = np.linspace(10, 100, 20)
        y = x * 2.5 + np.random.normal(0, 1, 20)
        df = pd.DataFrame({"x": x, "y": y})
        corr_res = safe_correlation(df, "x", "y")
        self.assertEqual(corr_res["correlation_status"], "valid")
        self.assertTrue(corr_res["pearson"] > 0.9)
        self.assertEqual(corr_res["direction"], "positive")

    # =========================================================================
    # 22. Zero-variance correlation attempt
    # =========================================================================
    def test_22_zero_variance_correlation_attempt(self):
        df = pd.DataFrame({
            "metric_a": [10.0, 20.0, 30.0, 40.0],
            "constant_b": [22.0, 22.0, 22.0, 22.0]
        })
        suit = StatisticalSuitabilityEngine.check_correlation(df, "metric_a", "constant_b")
        self.assertEqual(suit["status"], "NOT_APPLICABLE")
        self.assertIn("zero variance", suit["reason"])

        res = safe_correlation(df, "metric_a", "constant_b")
        self.assertEqual(res["correlation_status"], "not_applicable")

    # =========================================================================
    # 23. ANOVA with valid variance
    # =========================================================================
    def test_23_anova_with_valid_variance(self):
        df = pd.DataFrame({
            "Department": ["Sales"] * 10 + ["Support"] * 10 + ["Engineering"] * 10,
            "Salary": [50000 + i * 500 for i in range(10)] + [40000 + i * 400 for i in range(10)] + [80000 + i * 800 for i in range(10)]
        })
        suit = StatisticalSuitabilityEngine.check_anova(df, "Department", "Salary")
        self.assertEqual(suit["status"], "VALID")
        
        anova_res = safe_anova(df, "Department", "Salary")
        self.assertEqual(anova_res["test_status"], "completed")
        self.assertTrue(anova_res["is_significant"])

    # =========================================================================
    # 24. ANOVA with zero variance
    # =========================================================================
    def test_24_anova_with_zero_variance(self):
        df = pd.DataFrame({
            "Department": ["Sales"] * 10 + ["Support"] * 10,
            "Working_Days": [22.0] * 20
        })
        suit = StatisticalSuitabilityEngine.check_anova(df, "Department", "Working_Days")
        self.assertEqual(suit["status"], "NOT_APPLICABLE")
        self.assertEqual(suit["reason"], "dependent_variable_constant")

        anova_res = safe_anova(df, "Department", "Working_Days")
        self.assertEqual(anova_res["test_status"], "not_applicable")
        self.assertIsNone(anova_res["f_stat"])
        self.assertIsNone(anova_res["p_val"])

    # =========================================================================
    # 25. PCA with constant variables
    # =========================================================================
    def test_25_pca_with_constant_variables(self):
        df = pd.DataFrame({
            "feat_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "feat_const": [22.0, 22.0, 22.0, 22.0, 22.0]
        })
        retained, excluded, reason = safe_pca_features(df, ["feat_1", "feat_2", "feat_const"])
        self.assertIn("feat_const", excluded)
        self.assertEqual(reason, "zero_variance")
        self.assertEqual(retained, ["feat_1", "feat_2"])

    # =========================================================================
    # 26. Chart using filtered data
    # =========================================================================
    def test_26_chart_using_filtered_data(self):
        raw_df = pd.DataFrame({
            "Region": ["North", "North", "South", "South"],
            "Sales": [100.0, 200.0, 500.0, 600.0]
        })
        # Filter: Region == 'North'
        filtered_df = raw_df[raw_df["Region"] == "North"]
        
        ar = AnalyticalResultEngine.create_result(
            dataset_id="ds_1",
            metric="Sales",
            source_columns=["Sales", "Region"],
            cleaned_columns=["Sales", "Region"],
            total_record_count=len(raw_df),
            valid_record_count=len(filtered_df),
            excluded_record_count=len(raw_df) - len(filtered_df),
            excluded_record_reason="Filtered to Region == 'North'",
            aggregation="SUM",
            values={"North": 300.0},
            chart_data=[{"category": "North", "value": 300.0}],
            takeaway="Total Sales for North is 300.00 across 2 records."
        )
        self.assertEqual(ar["valid_record_count"], 2)
        self.assertEqual(ar["excluded_record_count"], 2)
        self.assertEqual(ar["chart_data"][0]["value"], 300.0)
        self.assertIn("300.00", ar["takeaway"])

    # =========================================================================
    # 27. Chart using grouped data
    # =========================================================================
    def test_27_chart_using_grouped_data(self):
        # 10 records: 4 A, 4 B, 2 None
        # Metric is 10 each -> Full total = 100. Grouped valid = 80. Difference = 20.
        df = pd.DataFrame({
            "Group": ["A"] * 4 + ["B"] * 4 + [None, None],
            "Metric": [10.0] * 10
        })
        ar = compute_grouped_analytical_result(
            df=df,
            measure_col="Metric",
            group_col="Group",
            aggregation="SUM",
            unit="units"
        )
        self.assertEqual(ar.statistics["full_total"], 100.0)
        self.assertEqual(ar.statistics["grouped_total"], 80.0)
        self.assertEqual(ar.statistics["excluded_difference"], 20.0)
        self.assertEqual(ar.excluded_record_count, 2)
        # Check that top category share uses grouped_total as denominator: 40 / 80 = 50.0%
        self.assertEqual(ar.statistics["top_category_share"], 50.0)

    # =========================================================================
    # 28. Recommendation without business context
    # =========================================================================
    def test_28_recommendation_without_business_context(self):
        # Plain category share (Hybrid = 50.6%) must NOT invent business goals
        finding_class = InsightGovernanceEngine.classify_finding("CATEGORY_DISTRIBUTION")
        self.assertEqual(finding_class, "DESCRIPTIVE")

        # When building recommendation, owner, timeline, target are unassigned
        rec = RecommendationGovernanceEngine.build_recommendation(
            rec_id="rec_1",
            insight_id="ins_1",
            observation="Hybrid accounts for 50.6% of records.",
            interpretation="Distribution reflects workforce allocation.",
            suggested_action="Review segment distribution during quarterly planning."
        )
        self.assertEqual(rec["owner"], "Unassigned")
        self.assertEqual(rec["timeline"], "Not specified")
        self.assertEqual(rec["target"], "Not defined")
        self.assertNotIn("optimize contribution share mix", rec["recommended_action"].lower())

    # =========================================================================
    # 29. User-defined KPI target
    # =========================================================================
    def test_29_user_defined_kpi_target(self):
        action = ActionGovernanceEngine.create_action_item(
            action_id="act_1",
            recommendation_id="rec_1",
            insight_id="ins_1",
            title="Overtime Review",
            metric_name="Overtime_Hours",
            baseline_val=7.38,
            steps=["Audit overtime logging"],
            user_target="Reduce to < 5.0 hours",
            status="In Progress"
        )
        self.assertEqual(action["observed_baseline"], "7.38")
        self.assertEqual(action["kpi_target"], "Reduce to < 5.0 hours")
        self.assertEqual(action["target_status"], "Target defined by user")
        self.assertEqual(action["status"], "In Progress")

    # =========================================================================
    # 30. User-defined owner/timeline
    # =========================================================================
    def test_30_user_defined_owner_timeline(self):
        action = ActionGovernanceEngine.create_action_item(
            action_id="act_2",
            recommendation_id="rec_2",
            insight_id="ins_2",
            title="Data Quality Remediation",
            metric_name="Missing_Values",
            baseline_val="12 records",
            steps=["Validate upstream pipeline"],
            user_owner="Jane Doe (Data Engineer)",
            user_timeline="Q4 2026",
            status="Not Started"
        )
        self.assertEqual(action["owner"], "Jane Doe (Data Engineer)")
        self.assertEqual(action["timeline"], "Q4 2026")
        self.assertEqual(action["status"], "Not Started")


if __name__ == "__main__":
    unittest.main()
