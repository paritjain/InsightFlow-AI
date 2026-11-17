import unittest
import pandas as pd
import numpy as np

from engine.eda import run_automated_eda
from engine.insights import generate_insights_and_story
from engine.validator import validate_all_insights


class TestPart4InsightsEngine(unittest.TestCase):

    def setUp(self):
        # Sample Dataset 1: Retail/Sales Dataset
        self.sales_df = pd.DataFrame({
            "Transaction_ID": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
            "Order_Date": pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d"),
            "Region": ["North", "South", "East", "West", "North", "South", "East", "West", "North", "South"],
            "Category": ["Tech", "Tech", "Furniture", "Furniture", "Tech", "Office", "Office", "Tech", "Tech", "Furniture"],
            "Sales_Amount": [250.0, 450.0, 120.0, 300.0, 550.0, 80.0, 210.0, 600.0, 400.0, 310.0],
            "Discount": [0.0, 0.1, 0.0, 0.05, 0.0, 0.2, 0.0, 0.15, 0.0, 0.0]
        })

    def test_1_insight_schema_and_categories(self):
        eda_res = run_automated_eda(self.sales_df)
        res = generate_insights_and_story(self.sales_df, eda_res)
        insights = res["insights"]
        
        self.assertGreater(len(insights), 0)
        valid_categories = {"Performance", "Concentration Risk", "Trend & Seasonality", "Variance & Anomaly", "Correlation"}
        
        for ins in insights:
            self.assertIn("id", ins)
            self.assertIn("title", ins)
            self.assertIn("category", ins)
            self.assertIn(ins["category"], valid_categories)
            self.assertIn("business_impact", ins)
            self.assertIn(ins["business_impact"], {"HIGH", "MEDIUM", "LOW"})
            self.assertIn("evidence", ins)
            self.assertGreater(len(ins["evidence"]), 0)
            self.assertIn("business_relevance", ins)
            self.assertIn("narrative_summary", ins)

    def test_2_non_causal_correlation_language(self):
        eda_res = run_automated_eda(self.sales_df)
        res = generate_insights_and_story(self.sales_df, eda_res)
        corr_insights = [i for i in res["insights"] if i["category"] == "Correlation"]
        
        causal_words = ["causes", "caused by", "drives", "driven by", "leads to", "creates"]
        for ins in corr_insights:
            title_low = ins["title"].lower()
            text_low = ins["insight"].lower()
            for cw in causal_words:
                self.assertNotIn(cw, title_low, f"Causal word '{cw}' found in correlation insight title")
                self.assertNotIn(cw, text_low, f"Causal word '{cw}' found in correlation insight text")

    def test_3_pre_display_validation_gate(self):
        eda_res = run_automated_eda(self.sales_df)
        res = generate_insights_and_story(self.sales_df, eda_res)
        passed, rejected = validate_all_insights(res["insights"], self.sales_df, eda_res)
        
        self.assertEqual(len(rejected), 0, f"Rejected insights found: {rejected}")
        self.assertEqual(len(passed), len(res["insights"]))

    def test_4_executive_story_structure(self):
        eda_res = run_automated_eda(self.sales_df)
        res = generate_insights_and_story(self.sales_df, eda_res)
        story = res.get("story", {})
        
        self.assertIn("title", story)
        self.assertIn("executive_summary", story)
        self.assertIn("key_findings", story)
        self.assertIn("drivers", story)
        self.assertIn("risks", story)
        self.assertIn("opportunities", story)


if __name__ == "__main__":
    unittest.main()
