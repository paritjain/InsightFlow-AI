import unittest
import pandas as pd

from engine.eda import run_automated_eda
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan


class TestPart5RecommendationsEngine(unittest.TestCase):

    def setUp(self):
        self.sales_df = pd.DataFrame({
            "Transaction_ID": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
            "Order_Date": pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d"),
            "Region": ["North", "South", "East", "West", "North", "South", "East", "West", "North", "South"],
            "Category": ["Tech", "Tech", "Furniture", "Furniture", "Tech", "Office", "Office", "Tech", "Tech", "Furniture"],
            "Sales_Amount": [250.0, 450.0, 120.0, 300.0, 550.0, 80.0, 210.0, 600.0, 400.0, 310.0],
            "Discount": [0.0, 0.1, 0.0, 0.05, 0.0, 0.2, 0.0, 0.15, 0.0, 0.0]
        })

    def test_1_recommendation_schema_and_fields(self):
        eda_res = run_automated_eda(self.sales_df)
        ins_res = generate_insights_and_story(self.sales_df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])
        recs = rec_res["recommendations"]

        self.assertGreater(len(recs), 0)
        valid_action_types = {"Risk Mitigation", "Process Optimization", "Strategic Growth", "Data Governance"}

        for r in recs:
            self.assertIn("id", r)
            self.assertIn("title", r)
            self.assertIn("action_type", r)
            self.assertIn(r["action_type"], valid_action_types)
            self.assertIn("priority", r)
            self.assertIn(r["priority"], {"CRITICAL", "HIGH", "MEDIUM", "LOW"})
            self.assertIn("impact_score", r)
            self.assertIsInstance(r["impact_score"], (int, float))
            self.assertIn("feasibility_score", r)
            self.assertIsInstance(r["feasibility_score"], (int, float))
            self.assertIn("effort_level", r)
            self.assertIn(r["effort_level"], {"LOW", "MEDIUM", "HIGH"})
            self.assertIn("target_metric", r)
            self.assertTrue("linked_insight_id" in r or "source_insight_id" in r)
            self.assertIn("rationale", r)
            self.assertIn("implementation_steps", r)
            self.assertIsInstance(r["implementation_steps"], list)
            self.assertGreater(len(r["implementation_steps"]), 0)
            self.assertIn("expected_outcome", r)

    def test_2_action_items_lineage(self):
        eda_res = run_automated_eda(self.sales_df)
        ins_res = generate_insights_and_story(self.sales_df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])
        action_items = rec_res["action_items"]

        self.assertGreater(len(action_items), 0)
        for act in action_items:
            self.assertIn("id", act)
            self.assertIn("recommendation_id", act)
            self.assertIn("source_insight_id", act)
            self.assertIn("task_name", act)
            self.assertIn("owner", act)
            self.assertIn("priority", act)
            self.assertIn("status", act)
            self.assertIn("lineage", act)


if __name__ == "__main__":
    unittest.main()
