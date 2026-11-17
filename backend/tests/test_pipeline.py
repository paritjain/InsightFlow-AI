import unittest
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.profiler import profile_file
from engine.cleaner import detect_quality_issues, apply_cleaning_rules
from engine.dictionary import generate_data_dictionary
from engine.etl import detect_relationships, create_master_dataset
from engine.eda import run_automated_eda
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan
from engine.ask_data import ask_your_data
from engine.exporter import export_master_excel
from demo_data.generator import generate_demo_datasets

class TestInsightFlowPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.store_dir = "backend/demo_data/store"
        cls.sales_p, cls.cust_p, cls.prod_p = generate_demo_datasets(cls.store_dir)
        cls.sales_df = pd.read_excel(cls.sales_p)
        cls.cust_df = pd.read_excel(cls.cust_p)
        cls.prod_df = pd.read_excel(cls.prod_p)

    def test_01_profiler(self):
        profile = profile_file(self.sales_p)
        self.assertIn("health_score", profile)
        self.assertGreater(profile["health_score"], 50)
        self.assertEqual(profile["num_rows"], len(self.sales_df))

    def test_02_cleaner(self):
        issues = detect_quality_issues(self.sales_df, "Sales_2024.xlsx")
        self.assertIsInstance(issues, list)
        cleaned_df, audit_trail = apply_cleaning_rules(self.sales_df, issues)
        self.assertIsInstance(audit_trail, list)
        self.assertEqual(len(cleaned_df), len(self.sales_df))

    def test_03_dictionary(self):
        dictionary = generate_data_dictionary(self.sales_df, "Sales_2024.xlsx")
        self.assertEqual(len(dictionary), len(self.sales_df.columns))
        roles = [c["role"] for c in dictionary]
        self.assertIn("Measure", roles)

    def test_04_etl(self):
        dfs_dict = {
            "Sales": self.sales_df,
            "Customers": self.cust_df,
            "Products": self.prod_df
        }
        rel_info = detect_relationships(dfs_dict)
        self.assertGreaterEqual(len(rel_info["relationships"]), 1)
        master_df, master_meta = create_master_dataset(dfs_dict, "Sales")
        self.assertEqual(len(master_df), len(self.sales_df))

    def test_05_eda(self):
        master_df, _ = create_master_dataset({
            "Sales": self.sales_df,
            "Customers": self.cust_df,
            "Products": self.prod_df
        }, "Sales")
        eda_res = run_automated_eda(master_df)
        self.assertIn("domain", eda_res)
        self.assertIn("numeric_summary", eda_res)

    def test_06_insights(self):
        master_df, _ = create_master_dataset({
            "Sales": self.sales_df,
            "Customers": self.cust_df,
            "Products": self.prod_df
        }, "Sales")
        eda_res = run_automated_eda(master_df)
        insights_res = generate_insights_and_story(master_df, eda_res)
        self.assertIn("insights", insights_res)
        self.assertIn("story", insights_res)

    def test_07_ask_data(self):
        res = ask_your_data(self.sales_df, "Which customers are most valuable?")
        self.assertIn("answer", res)
        self.assertIn("calculation_formula", res)
        self.assertGreater(len(res["evidence"]), 0)

    def test_08_exporter(self):
        out_excel = os.path.join(self.store_dir, "Test_Deliverable.xlsx")
        dict_list = generate_data_dictionary(self.sales_df)
        issues = detect_quality_issues(self.sales_df)
        eda_res = run_automated_eda(self.sales_df)
        insights_res = generate_insights_and_story(self.sales_df, eda_res)
        recs_res = generate_recommendations_and_action_plan(insights_res["insights"])

        out_path = export_master_excel(
            raw_df=self.sales_df,
            cleaned_df=self.sales_df,
            master_df=self.sales_df,
            dictionary_list=dict_list,
            quality_issues=issues,
            eda_results=eda_res,
            insights_list=insights_res["insights"],
            recommendations_list=recs_res["recommendations"],
            action_items_list=recs_res["action_items"],
            output_path=out_excel
        )
        self.assertTrue(os.path.exists(out_path))

if __name__ == "__main__":
    unittest.main()
