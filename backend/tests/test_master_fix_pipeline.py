import unittest
import pandas as pd
import numpy as np
import os
import sys

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from engine.profiler import infer_column_semantic_metadata
from engine.dictionary import generate_data_dictionary
from engine.cleaner import detect_quality_issues, apply_cleaning_rules
from engine.etl import detect_relationships, create_master_dataset, validate_referential_integrity
from engine.eda import run_automated_eda
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan
from services.statistical_analysis import compute_pca, compute_group_differences


class TestMasterFixPipeline(unittest.TestCase):
    """
    Comprehensive regression test suite verifying items A through W
    of the InsightFlow AI Master Fix Prompt.
    """

    def setUp(self):
        # 1. Orders table
        self.orders_df = pd.DataFrame({
            "Order_ID": ["ORD-101", "ORD-102", "ORD-103", "ORD-104", "ORD-105", "ORD-106", "ORD-106"],
            "Customer_ID": ["CUST-01", "CUST-02", "CUST-01", "CUST-03", "CUST-02", "CUST-01", "CUST-01"],
            "Product_ID": ["PRD-01", "PRD-02", "PRD-01", "PRD-02", "PRD-01", "PRD-02", "PRD-02"],
            "Quantity": [2, 5, 1, 3, 10, 2, 2],
            "Unit_Price": [50.0, 100.0, 50.0, 100.0, 50.0, 100.0, 100.0],
            "Discount": [0.05, 0.10, 0.00, 0.15, 0.20, 0.10, 0.10],
            "Sales": [95.0, 450.0, 50.0, 255.0, 400.0, 180.0, 180.0],
            "Cost": [70.0, 320.0, 35.0, 190.0, 280.0, 130.0, 130.0],
            "Profit": [25.0, 130.0, 15.0, 65.0, 120.0, 50.0, 50.0],
            "Shipping_Days": [3.0, 5.0, 2.0, 4.0, 7.0, 3.0, 3.0],
            "Returned": ["No", "Yes", "No", "No", "NO", "No", "No"]  # Intentional casing inconsistency
        })

        # 2. Customers table
        self.customers_df = pd.DataFrame({
            "Customer_ID": ["CUST-01", "CUST-02", "CUST-03"],
            "Customer_Name": ["Alice", "Bob", "Charlie"],
            "Email": ["alice@corp.com", "bob@corp.com", "charlie@corp.com"],
            "Region": ["North", "South", "East"]
        })

        # 3. Products table
        self.products_df = pd.DataFrame({
            "Product_ID": ["PRD-01", "PRD-02"],
            "Product_Name": ["Widget A", "Widget B"],
            "Category": ["Electronics", "Appliances"]
        })

        self.tables = {
            "Orders": self.orders_df,
            "Customers": self.customers_df,
            "Products": self.products_df
        }

    # A. Primary key detection
    def test_A_primary_key_detection(self):
        meta = infer_column_semantic_metadata("Order_ID", self.orders_df["Order_ID"], len(self.orders_df), table_name="Orders")
        self.assertEqual(meta["semantic_role"], "Primary Key")
        self.assertTrue(meta["is_primary_key"])
        self.assertTrue(meta["is_identifier"])
        self.assertEqual(meta["physical_data_type"], "str")

    # B. Foreign key detection
    def test_B_foreign_key_detection(self):
        meta_cust = infer_column_semantic_metadata("Customer_ID", self.orders_df["Customer_ID"], len(self.orders_df), table_name="Orders")
        self.assertEqual(meta_cust["semantic_role"], "Foreign Key")
        self.assertTrue(meta_cust["is_foreign_key"])
        self.assertEqual(meta_cust["related_table"], "Customers")
        self.assertEqual(meta_cust["related_column"], "Customer_ID")

        meta_prod = infer_column_semantic_metadata("Product_ID", self.orders_df["Product_ID"], len(self.orders_df), table_name="Orders")
        self.assertEqual(meta_prod["semantic_role"], "Foreign Key")
        self.assertTrue(meta_prod["is_foreign_key"])
        self.assertEqual(meta_prod["related_table"], "Products")
        self.assertEqual(meta_prod["related_column"], "Product_ID")

    # C. Referential integrity
    def test_C_referential_integrity(self):
        ref_report = validate_referential_integrity(self.tables)
        summary = ref_report["summary"]
        self.assertIn("status", summary)
        self.assertIn("duplicate_pk_count", summary)
        self.assertIn("orphan_fk_count", summary)
        self.assertIn("null_pk_count", summary)
        self.assertIn("missing_fk_count", summary)
        # Orders has 1 duplicate Order_ID ('ORD-106')
        self.assertEqual(summary["duplicate_pk_count"], 1)
        # All foreign keys exist in parent tables
        self.assertEqual(summary["orphan_fk_count"], 0)

    # D. Binary category detection
    def test_D_binary_category_detection(self):
        meta_ret = infer_column_semantic_metadata("Returned", self.orders_df["Returned"], len(self.orders_df))
        self.assertEqual(meta_ret["semantic_type"], "Binary Category")
        self.assertEqual(meta_ret["semantic_role"], "Dimension")
        self.assertIn("No", meta_ret["allowed_values"])
        self.assertIn("Yes", meta_ret["allowed_values"])

    # E. Percentage semantic detection
    def test_E_percentage_semantic_detection(self):
        meta_disc = infer_column_semantic_metadata("Discount", self.orders_df["Discount"], len(self.orders_df))
        self.assertEqual(meta_disc["semantic_type"], "Percentage")
        self.assertEqual(meta_disc["unit"], "%")
        self.assertEqual(meta_disc["valid_range"], "[0, 1]")
        self.assertEqual(meta_disc["display_format"], "0.0%")

    # F. Duration semantic detection
    def test_F_duration_semantic_detection(self):
        meta_dur = infer_column_semantic_metadata("Shipping_Days", self.orders_df["Shipping_Days"], len(self.orders_df))
        self.assertEqual(meta_dur["semantic_type"], "Duration")
        self.assertEqual(meta_dur["unit"], "Days")
        self.assertEqual(meta_dur["valid_range"], ">= 0")

    # G. PII vs identifier separation
    def test_G_pii_vs_identifier_separation(self):
        meta_cust_id = infer_column_semantic_metadata("Customer_ID", self.orders_df["Customer_ID"], len(self.orders_df))
        self.assertEqual(meta_cust_id["pii_classification"], "No")
        self.assertEqual(meta_cust_id["id_safety_classification"], "Appropriate identifier (Foreign Key)")

        meta_email = infer_column_semantic_metadata("Email", self.customers_df["Email"], len(self.customers_df))
        self.assertEqual(meta_email["pii_classification"], "Yes")

    # H. Categorical normalization
    def test_H_categorical_normalization(self):
        issues = detect_quality_issues(self.orders_df, "Orders")
        casing_issues = [i for i in issues if i["type"] == "standardize_casing" and i["column"] == "Returned"]
        self.assertGreater(len(casing_issues), 0)
        issue = casing_issues[0]
        self.assertEqual(issue["original_value"], "NO")
        self.assertEqual(issue["normalized_value"], "No")
        self.assertEqual(issue["affected_records"], 1)

    # I. Cleaning log generation
    def test_I_cleaning_log_generation(self):
        issues = detect_quality_issues(self.orders_df, "Orders")
        cleaned_df, audit_trail = apply_cleaning_rules(self.orders_df, issues)
        self.assertGreater(len(audit_trail), 0)
        ret_log = [l for l in audit_trail if l["column"] == "Returned"][0]
        self.assertEqual(ret_log["original_value"], "NO")
        self.assertEqual(ret_log["normalized_value"], "No")
        self.assertEqual(ret_log["reason"], "Case inconsistency / categorical normalization")
        self.assertEqual(ret_log["transformation"], "Standardize categorical casing")
        self.assertIn("timestamp", ret_log)

    # J. Cleaned dataset generation
    def test_J_cleaned_dataset_generation(self):
        issues = detect_quality_issues(self.orders_df, "Orders")
        auto_rules = [q for q in issues if q.get("type") in ["standardize_casing", "trim_whitespace", "coerce_numeric"]]
        cleaned_df, _ = apply_cleaning_rules(self.orders_df, auto_rules)
        # Raw data preserved
        self.assertIn("NO", self.orders_df["Returned"].values)
        # Cleaned dataset has NO normalized to No
        self.assertNotIn("NO", cleaned_df["Returned"].values)
        self.assertEqual(list(cleaned_df["Returned"].unique()), ["No", "Yes"])

    # K. Dominant category calculation & L. Category percentage calculation
    def test_K_and_L_dominant_category_and_percentage(self):
        eda_res = run_automated_eda(self.orders_df)
        cat_summary = eda_res.get("categorical_summary", {})
        ret_summary = cat_summary.get("Returned", {})
        self.assertIsNotNone(ret_summary.get("dominant_category"))
        self.assertNotEqual(ret_summary.get("dominant_category"), "undefined")
        self.assertIsNotNone(ret_summary.get("top_category_pct"))
        self.assertNotEqual(str(ret_summary.get("top_category_pct")), "undefined%")
        self.assertGreater(ret_summary["top_category_pct"], 0.0)

    # M. Chart/insight result consistency
    def test_M_chart_insight_result_consistency(self):
        eda_res = run_automated_eda(self.orders_df)
        insights_res = generate_insights_and_story(self.orders_df, eda_res, dataset_version="cleaned_v1")
        insights = insights_res["insights"]
        for ins in insights:
            sv = ins.get("supporting_visualization", {})
            shared = ins.get("shared_result", {})
            self.assertEqual(shared.get("chart"), sv.get("title"))
            self.assertEqual(shared.get("dataset_version"), "cleaned_v1")
            self.assertGreater(len(shared.get("result", [])), 0)

    # N. Pearson correlation & O. Spearman correlation
    def test_N_and_O_pearson_and_spearman_correlation(self):
        eda_res = run_automated_eda(self.orders_df)
        bivar = eda_res.get("bivariate_analysis", {})
        corr_matrix = bivar.get("correlation_matrix", {})
        if "Sales" in corr_matrix and "Cost" in corr_matrix["Sales"]:
            p_val = corr_matrix["Sales"]["Cost"]
            self.assertGreater(p_val, 0.9)

        insights_res = generate_insights_and_story(self.orders_df, eda_res)
        corr_ins = [i for i in insights_res["insights"] if i.get("category") == "Correlation"]
        if corr_ins:
            text = corr_ins[0].get("observed_fact", "")
            self.assertIn("Pearson", text)
            self.assertIn("Spearman", text)
            interp = corr_ins[0].get("business_interpretation", "")
            self.assertIn("does not establish causation", interp)

    # P. PCA & Q. ANOVA
    def test_P_and_Q_pca_and_anova(self):
        # 1. PCA Test
        pca_res = compute_pca(self.orders_df, n_components=2)
        self.assertTrue(pca_res.get("is_valid"))
        self.assertIn("explained_variance_ratio", pca_res)
        self.assertEqual(len(pca_res["explained_variance_ratio"]), 2)

        # 2. ANOVA Test
        anova_res = compute_group_differences(self.orders_df, "Returned", "Quantity")
        anova = anova_res.get("anova", {})
        self.assertTrue(anova.get("valid"))
        self.assertIn("f_statistic", anova)
        self.assertIn("p_value", anova)
        self.assertIn("df_between", anova)
        self.assertIn("df_within", anova)

    # R. Outlier detection & S. Missing-value statistics
    def test_R_and_S_outlier_and_missing_statistics(self):
        data_dict = generate_data_dictionary(self.orders_df, "Orders", table_name="Orders")
        col_names = [d["column_name"] for d in data_dict]
        self.assertIn("Quantity", col_names)
        q_dict = [d for d in data_dict if d["column_name"] == "Quantity"][0]
        self.assertIn("outlier_count", q_dict)
        self.assertIn("missing_count", q_dict)
        self.assertIn("missing_percentage", q_dict)
        self.assertEqual(q_dict["missing_count"], 0)
        self.assertEqual(q_dict["total_rows"], len(self.orders_df))

    # T. Recommendation lineage
    def test_T_recommendation_lineage(self):
        eda_res = run_automated_eda(self.orders_df)
        insights_res = generate_insights_and_story(self.orders_df, eda_res)
        recs_res = generate_recommendations_and_action_plan(insights_res["insights"])
        for act in recs_res["action_items"]:
            self.assertIn("lineage", act)
            lin = act["lineage"]
            self.assertIn("action_id", lin)
            self.assertIn("recommendation_id", lin)
            self.assertIn("insight_id", lin)
            self.assertIn("dataset_version", lin)

    # U. Undefined/undefined% regression
    def test_U_undefined_bug_eliminated(self):
        eda_res = run_automated_eda(self.orders_df)
        insights_res = generate_insights_and_story(self.orders_df, eda_res)
        for ins in insights_res["insights"]:
            self.assertNotIn("undefined", ins["title"].lower())
            self.assertNotIn("undefined%", ins["title"].lower())
            self.assertNotIn("undefined", ins["insight"].lower())
            self.assertNotIn("undefined%", ins["insight"].lower())
            self.assertNotIn("undefined", ins.get("takeaway", "").lower())
            self.assertNotIn("undefined%", ins.get("takeaway", "").lower())

    # V. Owner, target, timeline must not be invented
    def test_V_owner_target_timeline_not_invented(self):
        eda_res = run_automated_eda(self.orders_df)
        insights_res = generate_insights_and_story(self.orders_df, eda_res)
        recs_res = generate_recommendations_and_action_plan(insights_res["insights"])
        for act in recs_res["action_items"]:
            self.assertEqual(act["owner"], "Unassigned")
            self.assertIn("Suggested function:", act["suggested_owner"])
            self.assertEqual(act["timeline"], "Not assigned")
            self.assertEqual(act["kpi_target"], "Not defined")
            self.assertEqual(act["target_status"], "Awaiting business target")

    # W. Priority explanation
    def test_W_priority_explanation(self):
        eda_res = run_automated_eda(self.orders_df)
        insights_res = generate_insights_and_story(self.orders_df, eda_res)
        recs_res = generate_recommendations_and_action_plan(insights_res["insights"])
        for act in recs_res["action_items"]:
            self.assertIn("priority", act)
            self.assertIn(act["priority"], ["HIGH", "MEDIUM", "LOW"])
            self.assertIn("priority_reason", act)
            self.assertGreater(len(act["priority_reason"]), 5)


if __name__ == "__main__":
    unittest.main()
