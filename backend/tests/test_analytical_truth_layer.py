import unittest
import pandas as pd
import numpy as np
import os
import sys

# Ensure backend root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.analytical_truth import (
    AnalyticalResult,
    compute_variable_constancy,
    generate_constant_variable_statement,
    is_chart_suitable,
    safe_anova,
    safe_kruskal,
    safe_correlation,
    safe_pca_features,
    compute_grouped_analytical_result
)
from engine.profiler import infer_column_semantic_metadata
from engine.dictionary import generate_data_dictionary
from engine.eda import run_automated_eda, validate_chart_spec
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan
from services.statistical_analysis import compute_pca, compute_bivariate_hypothesis_tests


class TestAnalyticalTruthLayer(unittest.TestCase):
    """
    Comprehensive regression tests enforcing Section 37 and Section 35/36 Employee HR criteria:
    Guarantees mathematical correctness, deterministic single source of truth,
    and prevention of hallucinations across the entire InsightFlow AI pipeline.
    """

    def setUp(self):
        # Sample synthetic dataset with constant and non-constant variables
        np.random.seed(42)
        n = 100
        self.test_df = pd.DataFrame({
            "Emp_ID": [f"E{i:03d}" for i in range(n)],
            "Working_Days": [22.0] * n,  # Constant variable: nunique=1, var=0
            "Absent_Days": np.random.poisson(2, n),
            "Overtime_Hours": np.random.exponential(5, n),
            "Department": ["Sales"] * 40 + ["Engineering"] * 35 + ["Marketing"] * 20 + [None] * 5,
            "Month": pd.date_range("2025-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
        })

    # =========================================================================
    # TEST 1: Constant numeric variable detection
    # =========================================================================
    def test_1_constant_numeric_variable_detection(self):
        constancy = compute_variable_constancy(self.test_df["Working_Days"])
        self.assertTrue(constancy["is_constant"])
        self.assertEqual(constancy["nunique"], 1)
        self.assertEqual(constancy["constant_value"], 22.0)

    # =========================================================================
    # TEST 2: Zero variance check
    # =========================================================================
    def test_2_zero_variance_check(self):
        constancy = compute_variable_constancy(self.test_df["Working_Days"])
        self.assertEqual(constancy["variance"], 0.0)
        self.assertEqual(constancy["std"], 0.0)
        
        # Non-constant variable check
        non_const = compute_variable_constancy(self.test_df["Absent_Days"])
        self.assertFalse(non_const["is_constant"])
        self.assertGreater(non_const["variance"], 0.0)

    # =========================================================================
    # TEST 3 & 4: Missing category in grouping & grouped vs full denominator
    # =========================================================================
    def test_3_and_4_missing_category_and_grouped_denominator(self):
        # 5 records have missing Department. Working_Days is 22 each.
        # Full total = 100 * 22 = 2,200. Grouped valid total = 95 * 22 = 2,090.
        ar = compute_grouped_analytical_result(
            df=self.test_df,
            measure_col="Working_Days",
            group_col="Department",
            aggregation="SUM",
            unit="days"
        )
        self.assertEqual(ar.statistics["full_total"], 2200.0)
        self.assertEqual(ar.statistics["grouped_total"], 2090.0)
        self.assertEqual(ar.excluded_record_count, 5)
        self.assertEqual(ar.statistics["excluded_difference"], 110.0)
        
        # Verify takeaway accurately explains the missing records difference
        self.assertIn("2,090", ar.takeaway)
        self.assertIn("2,200", ar.takeaway)
        self.assertIn("110", ar.takeaway)
        self.assertIn("5 records excluded", ar.takeaway)

    # =========================================================================
    # TEST 5: PK / FK detection
    # =========================================================================
    def test_5_pk_fk_detection(self):
        parent_df = pd.DataFrame({"Employee_ID": [f"EMP_{i}" for i in range(100)], "Name": ["A"] * 100})
        child_df = pd.DataFrame({
            "Record_ID": range(200),
            "Employee_ID": [f"EMP_{i % 100}" for i in range(200)],
            "Days": [22] * 200
        })
        all_tables = {"Employees": parent_df, "Attendance": child_df}

        child_meta = infer_column_semantic_metadata(
            "Employee_ID", child_df["Employee_ID"], len(child_df),
            table_name="Attendance", all_tables=all_tables
        )
        self.assertEqual(child_meta["key_role"], "Foreign Key")
        self.assertEqual(child_meta["references"], "Employees.Employee_ID")
        self.assertEqual(child_meta["referential_coverage"], 100.0)
        self.assertEqual(child_meta["orphan_count"], 0)
        self.assertEqual(child_meta["relationship_type"], "many-to-one")

    # =========================================================================
    # TEST 6: Categorical normalization (NO -> No)
    # =========================================================================
    def test_6_categorical_normalization(self):
        s_binary = pd.Series(["YES", "NO", "yes", "no", "NO", "YES"])
        meta = infer_column_semantic_metadata("Active_Status", s_binary, len(s_binary))
        self.assertEqual(meta["allowed_values"], ["No", "Yes"])

    # =========================================================================
    # TEST 7: Chart data = Takeaway data
    # =========================================================================
    def test_7_chart_data_matches_takeaway_data(self):
        ar = compute_grouped_analytical_result(
            df=self.test_df,
            measure_col="Working_Days",
            group_col="Department",
            aggregation="SUM",
            unit="days"
        )
        top_cat = ar.statistics["top_category"]
        top_val = ar.values[top_cat]
        # Ensure chart_data top value matches takeaway
        chart_top = [c for c in ar.chart_data if c["category"] == top_cat][0]
        self.assertEqual(chart_top["value"], top_val)
        self.assertIn(f"{top_val:,.0f}", ar.takeaway.replace(",", ""))

    # =========================================================================
    # TEST 8: Constant variable chart suppression
    # =========================================================================
    def test_8_constant_variable_chart_suppression(self):
        # Histograms, Box plots, and Violins MUST be rejected for constant variables
        hist_spec = validate_chart_spec("HISTOGRAM", self.test_df, {"variable": "Working_Days"})
        self.assertFalse(hist_spec["is_valid"])
        self.assertIn("constant", hist_spec["fallback_message"].lower())

        box_spec = validate_chart_spec("BOX_PLOT", self.test_df, {"category": "Department", "metric": "Working_Days"})
        self.assertFalse(box_spec["is_valid"])
        self.assertIn("zero variance", box_spec["fallback_message"].lower())

        violin_spec = validate_chart_spec("VIOLIN", self.test_df, {"category": "Department", "metric": "Working_Days"})
        self.assertFalse(violin_spec["is_valid"])
        self.assertIn("zero variance", violin_spec["fallback_message"].lower())

    # =========================================================================
    # TEST 9: Constant dependent variable -> ANOVA not applicable
    # =========================================================================
    def test_9_constant_dependent_variable_anova_not_applicable(self):
        res = safe_anova(self.test_df, "Department", "Working_Days")
        self.assertEqual(res["test_status"], "not_applicable")
        self.assertEqual(res["reason"], "dependent_variable_constant")
        self.assertIsNone(res["f_stat"])
        self.assertIsNone(res["p_val"])

    # =========================================================================
    # TEST 10: Zero variance -> Correlation not applicable
    # =========================================================================
    def test_10_zero_variance_correlation_not_applicable(self):
        res = safe_correlation(self.test_df, "Working_Days", "Absent_Days")
        self.assertEqual(res["correlation_status"], "not_applicable")
        self.assertIn("zero variance", res["reason"].lower())
        self.assertIsNone(res["pearson"])
        self.assertIsNone(res["spearman"])

    # =========================================================================
    # TEST 11: Zero variance -> PCA exclusion
    # =========================================================================
    def test_11_zero_variance_pca_exclusion(self):
        retained, excluded, reason = safe_pca_features(self.test_df, ["Working_Days", "Absent_Days", "Overtime_Hours"])
        self.assertIn("Working_Days", excluded)
        self.assertNotIn("Working_Days", retained)
        self.assertEqual(reason, "zero_variance")

    # =========================================================================
    # TEST 12: Recommendation cannot generate invented owner
    # =========================================================================
    def test_12_recommendation_cannot_generate_invented_owner(self):
        eda_res = run_automated_eda(self.test_df)
        ins_res = generate_insights_and_story(self.test_df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])
        for rec in rec_res["recommendations"]:
            self.assertEqual(rec["owner"], "Unassigned")
            self.assertNotIn(rec["owner"], ["Head of Commercial Operations", "Data Governance Lead", "Analytics Team"])
        for act in rec_res["action_items"]:
            self.assertEqual(act["owner"], "Unassigned")

    # =========================================================================
    # TEST 13: Recommendation cannot generate invented timeline
    # =========================================================================
    def test_13_recommendation_cannot_generate_invented_timeline(self):
        eda_res = run_automated_eda(self.test_df)
        ins_res = generate_insights_and_story(self.test_df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])
        for rec in rec_res["recommendations"]:
            self.assertIn(rec["timeline"], ["Not specified", "Not assigned"])
            self.assertNotIn(rec["timeline"], ["14–30 days", "30 days", "45–60 days", "60–90 days"])

    # =========================================================================
    # TEST 14: Recommendation cannot generate invented KPI target
    # =========================================================================
    def test_14_recommendation_cannot_generate_invented_kpi_target(self):
        eda_res = run_automated_eda(self.test_df)
        ins_res = generate_insights_and_story(self.test_df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])
        for rec in rec_res["recommendations"]:
            self.assertEqual(rec["kpi_target"], "Not defined")
            self.assertEqual(rec["target_status"], "Awaiting business target")
            self.assertIn("Baseline established from current data", rec["target_recommendation"])

    # =========================================================================
    # TEST 15: Recommendation cannot invent business objective
    # =========================================================================
    def test_15_recommendation_cannot_invent_business_objective(self):
        eda_res = run_automated_eda(self.test_df)
        ins_res = generate_insights_and_story(self.test_df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])
        for rec in rec_res["recommendations"]:
            self.assertNotIn("optimize contribution share mix", rec["title"].lower())
            self.assertNotIn("optimize contribution share mix", str(rec).lower())

    # =========================================================================
    # EMPLOYEE HR DATASET REGRESSION TEST (Section 35, 36 & 40)
    # =========================================================================
    def test_employee_hr_regression_dataset(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate_paths = [
            os.path.join(base_dir, "demo_data", "store", "proj-upload-1790832709-1b9f2f", "employee_hr_eda_test.xlsx"),
            os.path.join(base_dir, "backend", "demo_data", "store", "proj-upload-1790832709-1b9f2f", "employee_hr_eda_test.xlsx"),
            "backend/demo_data/store/proj-upload-1790832709-1b9f2f/employee_hr_eda_test.xlsx",
            "demo_data/store/proj-upload-1790832709-1b9f2f/employee_hr_eda_test.xlsx"
        ]
        test_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if not test_path:
            self.skipTest("Dataset employee_hr_eda_test.xlsx not found.")

        excel = pd.ExcelFile(test_path)
        tables = {sheet: excel.parse(sheet) for sheet in excel.sheet_names}
        
        self.assertIn("Employees", tables)
        self.assertIn("Attendance", tables)
        
        att_df = tables["Attendance"]
        emp_df = tables["Employees"]

        # 1. Multi-table PK/FK Detection
        dict_res = generate_data_dictionary(att_df, table_name="Attendance", all_tables=tables)
        dict_items = dict_res["dictionary"] if isinstance(dict_res, dict) and "dictionary" in dict_res else dict_res
        dict_map = {c["column_name"]: c for c in dict_items}
        
        att_emp_id = dict_map["Employee_ID"]
        self.assertEqual(att_emp_id["key_role"], "Foreign Key")
        self.assertEqual(att_emp_id["references"], "Employees.Employee_ID")
        self.assertEqual(att_emp_id["referential_coverage"], 100.0)
        self.assertEqual(att_emp_id["orphan_count"], 0)

        # 2. Semantic Subtypes and Explicit Units
        self.assertEqual(dict_map["Working_Days"]["semantic_subtype"], "Count")
        self.assertEqual(dict_map["Working_Days"]["unit"], "days")
        self.assertTrue(dict_map["Working_Days"]["constant_flag"])
        self.assertEqual(dict_map["Working_Days"]["variance"], 0.0)

        self.assertEqual(dict_map["Absent_Days"]["semantic_subtype"], "Count")
        self.assertEqual(dict_map["Absent_Days"]["unit"], "days")
        self.assertFalse(dict_map["Absent_Days"]["constant_flag"])

        self.assertEqual(dict_map["Overtime_Hours"]["semantic_subtype"], "Duration")
        self.assertEqual(dict_map["Overtime_Hours"]["unit"], "hours")

        self.assertEqual(dict_map["Productivity_Index"]["semantic_subtype"], "Score / Index")

        # 3. Ground Truth Grouped Totals with Missing Categories
        # Total Working_Days = 110,880. Work_Mode has 12 missing values -> difference = 264.
        # Hybrid share = 55,946 / 110,616 = 50.6%
        ar_workmode = compute_grouped_analytical_result(
            df=att_df,
            measure_col="Working_Days",
            group_col="Work_Mode",
            aggregation="SUM",
            unit="days"
        )
        self.assertEqual(ar_workmode.statistics["full_total"], 110880.0)
        self.assertEqual(ar_workmode.statistics["grouped_total"], 110616.0)
        self.assertEqual(ar_workmode.excluded_record_count, 12)
        self.assertEqual(ar_workmode.statistics["excluded_difference"], 264.0)
        self.assertEqual(ar_workmode.statistics["top_category"], "Hybrid")
        self.assertEqual(ar_workmode.statistics["top_category_share"], 50.6)
        self.assertEqual(ar_workmode.values["Hybrid"], 55946.0)

        # 4. Check that Working_Days generates factual constancy statement without variation/performance language
        self.assertIn("constant at 22 days", ar_workmode.takeaway)
        self.assertIn("110,616", ar_workmode.takeaway)
        self.assertIn("110,880", ar_workmode.takeaway)
        self.assertIn("264", ar_workmode.takeaway)
        self.assertIn("12 records excluded", ar_workmode.takeaway)
        self.assertNotIn("performance drives", ar_workmode.takeaway.lower())

        # 5. Full EDA on Attendance: verify constant variable exclusions
        eda_res = run_automated_eda(att_df)
        
        # Working_Days must NOT be in histograms, scatter plots, or box plots
        self.assertNotIn("Working_Days", eda_res.get("histograms", {}))
        for sp in eda_res.get("scatter_plots", []):
            self.assertNotEqual(sp.get("metric_x"), "Working_Days")
            self.assertNotEqual(sp.get("metric_y"), "Working_Days")
        for bp_name in eda_res.get("box_plots", {}).keys():
            self.assertFalse(bp_name.startswith("Working_Days_by_"))

        # Working_Days must NOT be in strong associations
        for assoc in eda_res.get("relationships", {}).get("strong_associations", []):
            self.assertNotEqual(assoc.get("metric_1"), "Working_Days")
            self.assertNotEqual(assoc.get("metric_2"), "Working_Days")

        # 6. Advanced Statistics: ANOVA & PCA exclusions
        tests = compute_bivariate_hypothesis_tests(att_df, eda_res.get("semantic_layer"))
        wd_tests = [t for t in tests if t.get("measure") == "Working_Days"]
        for t in wd_tests:
            self.assertEqual(t.get("test_status"), "not_applicable")
            self.assertEqual(t.get("reason"), "dependent_variable_constant")

        pca_res = compute_pca(att_df, sem_layer=eda_res.get("semantic_layer"))
        if pca_res.get("is_valid") or pca_res.get("excluded_features"):
            self.assertIn("Working_Days", pca_res.get("excluded_features", []))


if __name__ == "__main__":
    unittest.main()
