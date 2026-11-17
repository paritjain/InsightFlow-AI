import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import openpyxl

from engine.eda import run_automated_eda
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan
from engine.exporter import export_master_excel, export_executive_pdf


class TestPart9ExportsEngine(unittest.TestCase):

    def setUp(self):
        self.sales_df = pd.DataFrame({
            "Transaction_ID": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
            "Order_Date": pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d"),
            "Region": ["North", "South", "East", "West", "North", "South", "East", "West", "North", "South"],
            "Category": ["Tech", "Tech", "Furniture", "Furniture", "Tech", "Office", "Office", "Tech", "Tech", "Furniture"],
            "Sales_Amount": [250.0, 450.0, 120.0, 300.0, 550.0, 80.0, 210.0, 600.0, 400.0, 310.0],
            "Discount": [0.0, 0.1, 0.0, 0.05, 0.0, 0.2, 0.0, 0.15, 0.0, 0.0]
        })
        self.excel_path = "tmp/test_master_deliverable.xlsx"
        self.pdf_path = "tmp/test_executive_report.pdf"

    def tearDown(self):
        if os.path.exists(self.excel_path):
            try:
                os.remove(self.excel_path)
            except Exception:
                pass
        if os.path.exists(self.pdf_path):
            try:
                os.remove(self.pdf_path)
            except Exception:
                pass

    def test_1_export_master_excel_generation(self):
        eda_res = run_automated_eda(self.sales_df)
        ins_res = generate_insights_and_story(self.sales_df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])

        out_file = export_master_excel(
            raw_df=self.sales_df,
            cleaned_df=self.sales_df,
            master_df=self.sales_df,
            dictionary_list=[{"column_name": "Sales_Amount", "business_meaning": "Revenue", "role": "Measure", "data_type": "float64", "null_percentage": 0, "unique_count": 10, "sample_value": 250.0}],
            quality_issues=[],
            eda_results=eda_res,
            insights_list=ins_res["insights"],
            recommendations_list=rec_res["recommendations"],
            action_items_list=rec_res["action_items"],
            output_path=self.excel_path
        )

        self.assertTrue(os.path.exists(out_file))
        wb = openpyxl.load_workbook(out_file)
        sheet_names = wb.sheetnames

        expected_sheets = [
            "Executive Summary", "Data Quality Audit", "Data Dictionary",
            "Master Dataset", "Cleaned Dataset", "Raw Dataset",
            "Exploratory Analysis", "Strategic Insights", "Action Plan"
        ]

        for s in expected_sheets:
            self.assertIn(s, sheet_names, f"Missing sheet {s} in exported master Excel workbook")

    def test_2_export_executive_pdf_generation(self):
        eda_res = run_automated_eda(self.sales_df)
        ins_res = generate_insights_and_story(self.sales_df, eda_res)
        rec_res = generate_recommendations_and_action_plan(ins_res["insights"])

        out_file = export_executive_pdf(
            eda_results=eda_res,
            insights_list=ins_res["insights"],
            recommendations_list=rec_res["recommendations"],
            dataset_name="Sales_Master_Dataset",
            output_path=self.pdf_path
        )

        self.assertTrue(os.path.exists(out_file))
        self.assertGreater(os.path.getsize(out_file), 1000, "PDF export file payload is empty or too small")


if __name__ == "__main__":
    unittest.main()
