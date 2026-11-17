import unittest
import os
import sys
import pandas as pd
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(BASE_DIR, 'backend'))

from engine.eda import run_automated_eda, validate_chart_spec
from engine.insights import generate_insights_and_story
from engine.cleaner import coerce_numeric_columns

class TestEDAHeatmapFixes(unittest.TestCase):

    def setUp(self):
        # Sample dataframe with small non-zero correlation
        self.df_sample = pd.DataFrame({
            'Revenue': [1000.50, 2500.00, 3100.25, 4000.00, 5200.00, 6100.00, 7500.00, 8200.00, 9000.00, 10500.00, 1000.50],
            'Quantity': [10.0, 25.0, 30.0, 42.0, 50.0, 61.0, 74.0, 83.0, 91.0, 102.0, 10.0],
            'SmallVar': [1.001, 1.002, 1.001, 1.003, 1.002, 1.001, 1.004, 1.002, 1.003, 1.001, 1.001],
            'ConstantVar': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        })
        self.df_formatted = pd.DataFrame({
            'Revenue': ['$1,000.50', '$2,500.00', '$3,100.25', '$4,000.00', '$5,200.00', '$6,100.00', '$7,500.00', '$8,200.00', '$9,000.00', '$10,500.00', '$1,000.50'],
            'Quantity': ['10', '25', '30', '42', '50', '61', '74', '83', '91', '102', '10']
        })

    def test_coerce_numeric_in_eda(self):
        clean_formatted = coerce_numeric_columns(self.df_formatted)
        eda_res = run_automated_eda(clean_formatted)
        measures = list(eda_res.get("numeric_summary", {}).keys())
        self.assertIn("Revenue", measures)
        self.assertIn("Quantity", measures)

    def test_heatmap_precision_and_constant_handling(self):
        clean_df = coerce_numeric_columns(self.df_sample)
        spec = validate_chart_spec("CORRELATION_HEATMAP", clean_df, {"variables": ["Revenue", "Quantity", "SmallVar", "ConstantVar"]})
        self.assertTrue(spec["is_valid"])
        data = spec["data"]
        
        # Check that constant variable returns is_constant flag
        const_pts = [d for d in data if d["x"] == "ConstantVar" and d["y"] != "ConstantVar"]
        for pt in const_pts:
            self.assertTrue(pt.get("is_constant"))
            self.assertEqual(pt.get("value"), 0.0)

        # Check that non-zero correlation retains decimal precision
        rev_qty_pts = [d for d in data if d["x"] == "Revenue" and d["y"] == "Quantity"]
        self.assertEqual(len(rev_qty_pts), 1)
        val = rev_qty_pts[0]["value"]
        self.assertGreater(val, 0.9)  # Revenue and Quantity are highly correlated (~0.999)

    def test_real_datasets_heatmap_precision(self):
        mfg_path = os.path.join(BASE_DIR, "tmp_test_datasets", "mfg.csv")
        if os.path.exists(mfg_path):
            df_mfg = pd.read_csv(mfg_path)
            eda_mfg = run_automated_eda(df_mfg)
            for v in eda_mfg.get("recommended_visualizations", []):
                if v.get("chart_type") == "CORRELATION_HEATMAP":
                    pts = v.get("data", [])
                    out_def = [p for p in pts if p["x"] == "Output_Units" and p["y"] == "Defect_Rate"]
                    if out_def:
                        self.assertEqual(out_def[0]["value"], 0.017) # Must retain 3 decimals, not truncated to 0.0 or 0.02

if __name__ == '__main__':
    unittest.main()
