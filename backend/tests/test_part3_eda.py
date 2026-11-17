import unittest
import pandas as pd
import numpy as np

from engine.profiler import infer_column_semantic_metadata
from engine.eda import run_automated_eda, recommend_visualizations, validate_chart_spec


class TestPart3AutomatedEDAEngine(unittest.TestCase):

    def setUp(self):
        # Sample Dataset 1: Retail/Sales Dataset
        self.sales_df = pd.DataFrame({
            "Transaction_ID": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
            "Order_Date": pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d"),
            "Region": ["North", "South", "East", "West", "North", "South", "East", "West", "North", "South"],
            "Category": ["Tech", "Tech", "Furniture", "Furniture", "Tech", "Office", "Office", "Tech", "Tech", "Furniture"],
            "Sales_Amount": [250.0, 450.0, 120.0, 300.0, 550.0, 80.0, 210.0, 600.0, 400.0, 310.0],
            "Discount": [0.0, 0.1, 0.0, 0.05, 0.0, 0.2, 0.0, 0.15, 0.0, 0.0] # Zero-inflated
        })

    def test_1_dataset_agnostic_chart_recommendations(self):
        eda_res = run_automated_eda(self.sales_df)
        self.assertIn("recommended_visualizations", eda_res)
        charts = eda_res["recommended_visualizations"]
        self.assertGreater(len(charts), 0)

        for chart in charts:
            self.assertIn("id", chart)
            self.assertIn("chart_type", chart)
            self.assertTrue(chart["is_valid"])
            self.assertGreater(len(chart["data"]), 0)
            self.assertIn("takeaway", chart)
            self.assertIn("suitability_reason", chart)

    def test_2_identifier_column_exclusion(self):
        sem_layer = {c: infer_column_semantic_metadata(c, self.sales_df[c], len(self.sales_df)) for c in self.sales_df.columns}
        charts = recommend_visualizations(self.sales_df, sem_layer)
        
        # Verify Transaction_ID is never used as measure or aggregation
        for chart in charts:
            y_ax = str(chart.get("y_axis", ""))
            x_ax = str(chart.get("x_axis", ""))
            self.assertNotEqual(y_ax, "Transaction_ID")
            self.assertNotEqual(x_ax, "Transaction_ID")

    def test_3_zero_inflated_and_numeric_summary(self):
        eda_res = run_automated_eda(self.sales_df)
        num_summary = eda_res["numeric_summary"]
        self.assertIn("Discount", num_summary)
        self.assertTrue(num_summary["Discount"]["is_zero_inflated"])
        self.assertGreaterEqual(num_summary["Discount"]["zero_percentage"], 40.0)

    def test_4_chart_types_presence(self):
        eda_res = run_automated_eda(self.sales_df)
        chart_types = [c["chart_type"] for c in eda_res["recommended_visualizations"]]
        
        # Expect at least Line, Bar, and Histogram
        self.assertTrue(any(ct in chart_types for ct in ["LINE", "BAR", "HORIZONTAL_BAR", "HISTOGRAM"]))

    def test_5_fatal_hotspot_with_sparse_null_impact(self):
        # Regression test: dataset with nulls in impact/severity column containing 'Fatal' records
        df = pd.DataFrame({
            "Plant": ["Plant A", "Plant B", "Plant A", "Plant C", "Plant B"],
            "Severity": [None, "Fatal Injury", None, "Near Miss", "Fatal Risk"],
            "Observation Date": pd.date_range("2024-01-01", periods=5, freq="D").strftime("%Y-%m-%d"),
            "Observation Count": [1, 1, 1, 1, 1]
        })
        eda_res = run_automated_eda(df)
        self.assertIn("operational_eda", eda_res)
        pareto = eda_res["operational_eda"]["where_is_it_happening"]["pareto_findings"]
        self.assertTrue(any("CRITICAL SAFETY HOTSPOT" in p for p in pareto))


if __name__ == "__main__":
    unittest.main()
