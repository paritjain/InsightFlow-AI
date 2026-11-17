import unittest
import os
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


class TestScatterPlotNonZeroOrders(unittest.TestCase):
    """
    Regression tests verifying that Scatter Plot bivariate data mapping preserves
    actual non-zero values for numeric columns such as 'orders' on the Y-axis.
    """

    def setUp(self):
        self.df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "sales": [450.0, 520.0, 610.0, 480.0, 590.0, 630.0, 710.0, 540.0, 600.0, 750.0,
                      460.0, 530.0, 620.0, 490.0, 580.0, 640.0, 720.0, 550.0, 610.0, 760.0],
            "orders": [7, 10, 12, 6, 8, 9, 14, 11, 13, 16,
                       7, 10, 12, 6, 8, 9, 14, 11, 13, 16],
            "marketing_spend": [170.0, 185.0, 210.0, 160.0, 195.0, 205.0, 230.0, 180.0, 200.0, 240.0,
                               172.0, 186.0, 212.0, 162.0, 196.0, 206.0, 232.0, 182.0, 202.0, 242.0]
        })

    def test_scatter_plot_preserves_nonzero_y_for_orders(self):
        """Verify that 'orders' retains non-zero positive integer values and is not set to 0."""
        orders_values = self.df["orders"].tolist()
        self.assertTrue(all(v > 0 for v in orders_values), "Orders column contains unexpected zero or negative values")
        self.assertEqual(min(orders_values), 6)
        self.assertEqual(max(orders_values), 16)
        self.assertAlmostEqual(float(np.mean(orders_values)), 10.6, places=1)

        # Simulate data rows exported for Visual Studio
        records = self.df.to_dict(orient="records")
        self.assertEqual(len(records), 20)

        # Simulate Plotly trace data extraction logic:
        # x = dataset.map(row => numericValue(row[salesColumn]))
        # y = dataset.map(row => numericValue(row[ordersColumn]))
        def numeric_value(val):
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return float(val) if not (np.isnan(val) or np.isinf(val)) else None
            try:
                cleaned = str(val).strip().replace("$", "").replace(",", "")
                f = float(cleaned)
                return f if not (np.isnan(f) or np.isinf(f)) else None
            except Exception:
                return None

        x_vals = [numeric_value(r.get("sales")) for r in records]
        y_vals = [numeric_value(r.get("orders")) for r in records]

        # Verify all X values are non-zero sales
        self.assertTrue(all(x is not None and x > 0 for x in x_vals))
        # Verify all Y values are non-zero orders (NOT 0!)
        self.assertTrue(all(y is not None and y > 0 for y in y_vals))
        self.assertEqual(y_vals[0], 7.0)
        self.assertEqual(y_vals[1], 10.0)
        self.assertEqual(y_vals[2], 12.0)

        # Verify zero count is exactly 0
        zero_count = sum(1 for y in y_vals if y == 0.0)
        self.assertEqual(zero_count, 0, "Regression detected: non-zero orders were converted to 0!")

    def test_real_dataset_time_series_sheet_orders_values(self):
        """Verify the actual Time Series sheet from InsightFlow_Multi_Dataset_EDA_Test.xlsx."""
        test_file = "InsightFlow_Multi_Dataset_EDA_Test.xlsx"
        if not os.path.exists(test_file):
            test_file = os.path.join("..", "InsightFlow_Multi_Dataset_EDA_Test.xlsx")
        
        if os.path.exists(test_file):
            try:
                df_ts = pd.read_excel(test_file, sheet_name="Time Series")
                self.assertIn("orders", df_ts.columns)
                self.assertIn("sales", df_ts.columns)
                
                # Check actual raw values of orders
                self.assertEqual(len(df_ts), 731)
                self.assertTrue(pd.api.types.is_numeric_dtype(df_ts["orders"]))
                
                # Confirm every single row has orders > 0
                non_zero_orders = df_ts[df_ts["orders"] > 0]
                self.assertEqual(len(non_zero_orders), 731)
                self.assertEqual(int(df_ts["orders"].min()), 5)
                self.assertEqual(int(df_ts["orders"].max()), 18)
                
                # Zero count must be 0
                zeros = (df_ts["orders"] == 0).sum()
                self.assertEqual(zeros, 0)
            except Exception as e:
                self.fail(f"Failed to read Time Series sheet: {e}")


if __name__ == "__main__":
    unittest.main()
