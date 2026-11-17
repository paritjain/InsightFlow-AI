import unittest
import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from engine.profiler import profile_file, load_primary_dataframe
from engine.cleaner import detect_quality_issues, apply_cleaning_rules, coerce_numeric_columns
from engine.dictionary import generate_data_dictionary
from engine.etl import detect_relationships, create_master_dataset
from engine.eda import run_automated_eda, recommend_visualizations
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan
from engine.ask_data import ask_your_data
from engine.exporter import export_master_excel, export_executive_pdf
from test_datasets.generate_test_datasets import generate_all_test_datasets

class TestInsightFlowFullQA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join(BASE_DIR, "test_datasets")
        generate_all_test_datasets(cls.test_dir)
        
        cls.sales_path = os.path.join(cls.test_dir, "test_sales.xlsx")
        cls.cust_path = os.path.join(cls.test_dir, "test_customers.xlsx")
        cls.prod_path = os.path.join(cls.test_dir, "test_products.xlsx")
        cls.mkt_path = os.path.join(cls.test_dir, "test_marketing.csv")
        cls.hr_path = os.path.join(cls.test_dir, "test_hr.xlsx")
        cls.messy_path = os.path.join(cls.test_dir, "test_messy_sales.xlsx")

    def test_01_file_ingestion_and_profiling(self):
        """Test file detection, sheet loading, and mathematical health score accuracy."""
        profile_sales = profile_file(self.sales_path)
        self.assertIn("health_score", profile_sales)
        self.assertGreaterEqual(profile_sales["health_score"], 50.0)
        self.assertLessEqual(profile_sales["health_score"], 100.0)
        self.assertGreater(profile_sales["num_rows"], 1000)

        profile_messy = profile_file(self.messy_path)
        self.assertIn("total_nulls", profile_messy)
        self.assertGreater(profile_messy["total_duplicates"], 0)

    def test_02_data_quality_and_cleaning_engine(self):
        """Test cleaning engine on messy sales dataset."""
        raw_df, _ = load_primary_dataframe(self.messy_path)
        issues = detect_quality_issues(raw_df, "test_messy_sales.xlsx")
        
        self.assertIsInstance(issues, list)
        self.assertGreater(len(issues), 0)

        issue_types = [i["type"] for i in issues]
        self.assertTrue(any(t in issue_types for t in ["duplicate_rows", "coerce_numeric", "trim_whitespace", "standardize_casing"]))

        cleaned_df, audit_trail = apply_cleaning_rules(raw_df, issues)
        self.assertIsInstance(audit_trail, list)
        self.assertGreater(len(audit_trail), 0)
        self.assertLessEqual(len(cleaned_df), len(raw_df))  # Duplicates removed

    def test_03_data_dictionary_generation(self):
        """Test data dictionary observed vs AI inferred classification."""
        df_sales, _ = load_primary_dataframe(self.sales_path)
        dictionary = generate_data_dictionary(df_sales, "test_sales.xlsx")
        
        self.assertEqual(len(dictionary), len(df_sales.columns))
        for item in dictionary:
            self.assertIn("column_name", item)
            self.assertIn("business_meaning", item)
            self.assertIn("role", item)
            self.assertIn("is_ai_inferred", item)

    def test_04_multi_file_etl_and_duplicate_warnings(self):
        """Test multi-file ETL join detection and duplicate key fanout warnings."""
        df_sales, _ = load_primary_dataframe(self.sales_path)
        df_cust, _ = load_primary_dataframe(self.cust_path)
        df_prod, _ = load_primary_dataframe(self.prod_path)

        dfs = {
            "Sales": df_sales,
            "Customers": df_cust,
            "Products": df_prod
        }

        rel_info = detect_relationships(dfs)
        self.assertIn("relationships", rel_info)
        self.assertGreaterEqual(len(rel_info["relationships"]), 1)

        master_df, meta = create_master_dataset(dfs, "Sales")
        self.assertIn("original_rows", meta)
        self.assertIn("final_rows", meta)
        self.assertIn("balance_equation", meta)
        self.assertEqual(len(master_df), len(df_sales))

    def test_05_eda_and_visualization_intelligence(self):
        """Test automated EDA and visualization recommendation engine."""
        df_sales, _ = load_primary_dataframe(self.sales_path)
        eda_res = run_automated_eda(df_sales)
        
        self.assertIn("domain", eda_res)
        self.assertIn("numeric_summary", eda_res)
        self.assertIn("recommended_visualizations", eda_res)
        
        recs = eda_res["recommended_visualizations"]
        self.assertGreaterEqual(len(recs), 3)
        chart_types = [r["chart_type"] for r in recs]
        self.assertTrue(any(ct in chart_types for ct in ["LINE", "BAR", "SCATTER", "HISTOGRAM", "CORRELATION_HEATMAP", "BOX_PLOT"]))

    def test_06_insight_generation_and_numerical_validation(self):
        """Test insight generation and numerical claim recalculation guard."""
        df_sales, _ = load_primary_dataframe(self.sales_path)
        eda_res = run_automated_eda(df_sales)
        insights_res = generate_insights_and_story(df_sales, eda_res)
        
        self.assertIn("insights", insights_res)
        self.assertIn("story", insights_res)
        self.assertGreater(len(insights_res["insights"]), 0)

        for ins in insights_res["insights"]:
            self.assertIn("title", ins)
            self.assertIn("evidence", ins)
            self.assertIn("business_impact", ins)

    def test_07_action_plan_and_recommendations(self):
        """Test action plan requirements (Action, Owner, Priority, Timeline, KPI, Impact, Evidence)."""
        df_sales, _ = load_primary_dataframe(self.sales_path)
        eda_res = run_automated_eda(df_sales)
        insights_res = generate_insights_and_story(df_sales, eda_res)
        recs_res = generate_recommendations_and_action_plan(insights_res["insights"])
        
        self.assertIn("recommendations", recs_res)
        self.assertIn("action_items", recs_res)

        for rec in recs_res["recommendations"]:
            self.assertIn("action", rec)
            self.assertIn("owner", rec)
            self.assertIn("priority", rec)
            self.assertIn("timeline", rec)
            self.assertIn("kpi", rec)
            self.assertIn("expected_impact", rec)

    def test_08_ask_your_data(self):
        """Test natural language queries over sales dataset."""
        df_sales, _ = load_primary_dataframe(self.sales_path)
        
        q1 = ask_your_data(df_sales, "Why did revenue decline?")
        self.assertIn("answer", q1)
        self.assertIn("calculation_formula", q1)

        q2 = ask_your_data(df_sales, "Which region is performing best?")
        self.assertIn("answer", q2)
        self.assertIn("calculation_formula", q2)

    def test_09_export_excel_and_pdf(self):
        """Test deliverable Excel and PDF report exports."""
        df_sales, _ = load_primary_dataframe(self.sales_path)
        eda_res = run_automated_eda(df_sales)
        insights_res = generate_insights_and_story(df_sales, eda_res)
        recs_res = generate_recommendations_and_action_plan(insights_res["insights"])
        dict_list = generate_data_dictionary(df_sales)
        issues = detect_quality_issues(df_sales)

        excel_out = os.path.join(self.test_dir, "Full_QA_Master_Deliverable.xlsx")
        pdf_out = os.path.join(self.test_dir, "Full_QA_Executive_Report.pdf")

        excel_path = export_master_excel(
            raw_df=df_sales,
            cleaned_df=df_sales,
            master_df=df_sales,
            dictionary_list=dict_list,
            quality_issues=issues,
            eda_results=eda_res,
            insights_list=insights_res["insights"],
            recommendations_list=recs_res["recommendations"],
            action_items_list=recs_res["action_items"],
            output_path=excel_out
        )
        self.assertTrue(os.path.exists(excel_path))

        pdf_path = export_executive_pdf(
            eda_results=eda_res,
            insights_list=insights_res["insights"],
            recommendations_list=recs_res["recommendations"],
            dataset_name="Sales_QA_Test",
            output_path=pdf_out
        )
        self.assertTrue(os.path.exists(pdf_path))

if __name__ == "__main__":
    unittest.main()
