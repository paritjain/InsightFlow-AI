import unittest
import pandas as pd

from engine.eda import run_automated_eda
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan


class TestPart6ActionCenterEngine(unittest.TestCase):

    def setUp(self):
        self.sales_df = pd.DataFrame({
            "Transaction_ID": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
            "Order_Date": pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d"),
            "Region": ["North", "South", "East", "West", "North", "South", "East", "West", "North", "South"],
            "Category": ["Tech", "Tech", "Furniture", "Furniture", "Tech", "Office", "Office", "Tech", "Tech", "Furniture"],
            "Sales_Amount": [250.0, 450.0, 120.0, 300.0, 550.0, 80.0, 210.0, 600.0, 400.0, 310.0],
            "Discount": [0.0, 0.1, 0.0, 0.05, 0.0, 0.2, 0.0, 0.15, 0.0, 0.0]
        })

    def test_1_action_center_kpi_and_lineage_traceability(self):
        eda_res = run_automated_eda(self.sales_df)
        ins_res = generate_insights_and_story(self.sales_df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])
        action_items = rec_res["action_items"]

        self.assertGreater(len(action_items), 0)

        for item in action_items:
            self.assertIn("id", item)
            self.assertIn("action_id", item)
            self.assertIn("recommendation_id", item)
            self.assertIn("source_insight_id", item)
            self.assertIn("task_name", item)
            self.assertIn("action_type", item)
            self.assertIn("owner", item)
            self.assertIn("priority", item)
            self.assertIn("status", item)
            self.assertIn(item["status"], {"In Progress", "Not Started", "Completed", "Blocked"})

            # KPI schema traceability
            self.assertIn("kpi", item)
            self.assertIn("kpi_baseline", item)
            self.assertIn("kpi_target", item)
            self.assertIn("kpi_schema", item)
            k_schema = item["kpi_schema"]
            self.assertIn("kpi_name", k_schema)
            self.assertIn("baseline_value", k_schema)
            self.assertIn("objective", k_schema)

            # Machine-readable lineage check
            self.assertIn("lineage", item)
            lineage = item["lineage"]
            self.assertIn("action_id", lineage)
            self.assertIn("recommendation_id", lineage)
            self.assertIn("insight_id", lineage)
            self.assertIn("metric_name", lineage)
            self.assertIn("baseline_value", lineage)
            self.assertEqual(lineage["traceability_status"], "VERIFIED_LINEAGE")


if __name__ == "__main__":
    unittest.main()
