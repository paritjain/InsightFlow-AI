import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

from engine.profiler import infer_column_semantic_metadata
from engine.parser import parse_and_contract_file
from engine.cleaner import calculate_data_health_score_breakdown
from engine.dictionary import generate_data_dictionary
from engine.eda import run_automated_eda
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan
from engine.validator import validate_all_insights


class TestPart7CrossModuleConsistency(unittest.TestCase):

    def setUp(self):
        # Dataset 1: E-Commerce Sales
        self.sales_df = pd.DataFrame({
            "Order_ID": [f"ORD-{i:04d}" for i in range(1, 21)],
            "Order_Date": pd.date_range("2024-01-01", periods=20, freq="D").strftime("%Y-%m-%d"),
            "Customer_Segment": ["Enterprise", "SMB", "Consumer", "Enterprise", "SMB"] * 4,
            "Sales": [150.0, 300.0, 450.0, 120.0, 800.0, 950.0, 200.0, 110.0, 640.0, 720.0,
                      310.0, 490.0, 880.0, 130.0, 250.0, 410.0, 990.0, 150.0, 320.0, 580.0],
            "Profit": [30.0, 60.0, 90.0, 20.0, 160.0, 190.0, 40.0, 22.0, 128.0, 144.0,
                       62.0, 98.0, 176.0, 26.0, 50.0, 82.0, 198.0, 30.0, 64.0, 116.0]
        })

        # Dataset 2: Logistics / Operations
        self.logistics_df = pd.DataFrame({
            "Shipment_ID": [f"SHP-{i:03d}" for i in range(101, 121)],
            "Warehouse_Zone": ["North", "South", "East", "West"] * 5,
            "Delivery_Time_Days": [2.5, 4.0, 1.5, 5.0, 3.0, 2.0, 4.5, 3.5, 1.0, 6.0,
                                   2.8, 3.8, 1.8, 4.8, 3.2, 2.2, 4.2, 3.2, 1.2, 5.5],
            "Shipping_Cost": [15.0, 25.0, 12.0, 35.0, 20.0, 18.0, 30.0, 22.0, 10.0, 40.0,
                              16.0, 24.0, 13.0, 33.0, 21.0, 19.0, 28.0, 21.0, 11.0, 38.0]
        })

    def test_1_profiler_to_contract_consistency(self):
        meta = infer_column_semantic_metadata("Order_ID", self.sales_df["Order_ID"], len(self.sales_df))
        self.assertIn(meta["semantic_role"], ["Primary Key", "Identifier"])
        self.assertTrue(meta["is_identifier"])

    def test_2_strict_identifier_protection_across_pipeline(self):
        eda_res = run_automated_eda(self.sales_df)
        charts = eda_res.get("recommended_visualizations", [])

        for chart in charts:
            x_col = chart.get("x_axis", "")
            y_col = chart.get("y_axis", "")
            metrics = chart.get("metrics", [])

            self.assertNotEqual(x_col, "Order_ID", "Order_ID used as x-axis in chart")
            self.assertNotEqual(y_col, "Order_ID", "Order_ID used as y-axis in chart")
            self.assertNotIn("Order_ID", metrics, "Order_ID included in metric aggregation")

    def test_3_end_to_end_lineage_and_traceability(self):
        eda_res = run_automated_eda(self.sales_df)
        ins_res = generate_insights_and_story(self.sales_df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])

        insights = ins_res["insights"]
        recommendations = rec_res["recommendations"]
        action_items = rec_res["action_items"]

        insight_ids = {i["id"] for i in insights}
        rec_ids = {r["id"] for r in recommendations}

        # Validate Recommendation -> Insight linkage
        for r in recommendations:
            src_ins_id = r.get("linked_insight_id", r.get("source_insight_id"))
            self.assertIn(src_ins_id, insight_ids, f"Recommendation {r['id']} links to non-existent insight {src_ins_id}")

        # Validate Action Item -> Recommendation & Insight linkage
        for a in action_items:
            rec_id = a.get("recommendation_id")
            src_ins_id = a.get("source_insight_id")
            self.assertIn(rec_id, rec_ids, f"Action item {a['id']} links to non-existent recommendation {rec_id}")
            self.assertIn(src_ins_id, insight_ids, f"Action item {a['id']} links to non-existent insight {src_ins_id}")

            lineage = a.get("lineage", {})
            self.assertEqual(lineage.get("traceability_status"), "VERIFIED_LINEAGE")
            self.assertEqual(lineage.get("recommendation_id"), rec_id)
            self.assertEqual(lineage.get("insight_id"), src_ins_id)

    def test_4_cross_module_metric_accuracy(self):
        # Verify total Sales sum in insights matches actual DataFrame sum
        actual_sales_sum = float(self.sales_df["Sales"].sum())

        eda_res = run_automated_eda(self.sales_df)
        ins_res = generate_insights_and_story(self.sales_df, eda_res)

        perf_insights = [i for i in ins_res["insights"] if i.get("category") == "Performance"]
        self.assertGreater(len(perf_insights), 0)

        perf_ins = perf_insights[0]
        ev_items = perf_ins.get("evidence", [])
        total_ev = next((e for e in ev_items if "Total Cumulative" in e.get("metric", "")), None)

        if total_ev:
            val_str = str(total_ev.get("value", "")).replace(",", "").replace("$", "")
            extracted_val = float(val_str)
            self.assertAlmostEqual(actual_sales_sum, extracted_val, places=1)


if __name__ == "__main__":
    unittest.main()
