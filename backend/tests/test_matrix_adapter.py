import unittest
import os
import sys
import pandas as pd
import numpy as np

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.matrix_adapter import IndustrialMatrixAdapter


class TestIndustrialMatrixAdapter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.excel_path = "Histry Card Line-1.xlsx"
        if not os.path.exists(cls.excel_path):
            cls.excel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Histry Card Line-1.xlsx"))
        cls.dfs = None
        if os.path.exists(cls.excel_path):
            cls.dfs = IndustrialMatrixAdapter.parse_workbook(cls.excel_path)

    def test_01_detection_heuristic(self):
        if os.path.exists(self.excel_path):
            is_matrix = IndustrialMatrixAdapter.is_industrial_matrix_workbook(self.excel_path)
            self.assertTrue(is_matrix, "Should identify Histry Card Line-1.xlsx as an industrial matrix workbook")

        # Negative test on non-existent or regular file
        self.assertFalse(IndustrialMatrixAdapter.is_industrial_matrix_workbook("non_existent.xlsx"))

    def test_02_parse_workbook_master_timeseries(self):
        if self.dfs is None:
            self.skipTest(f"{self.excel_path} not found")

        self.assertIn("Potline_Master_TimeSeries", self.dfs)
        df_master = self.dfs["Potline_Master_TimeSeries"]

        self.assertGreater(len(df_master), 5000)
        self.assertIn("Pot_No", df_master.columns)
        self.assertIn("Date", df_master.columns)
        self.assertIn("Working_Voltage_V", df_master.columns)
        self.assertIn("Bath_Temperature_C", df_master.columns)
        self.assertIn("Current_Efficiency_Pct", df_master.columns)
        self.assertIn("DC_Energy_kWh_t", df_master.columns)
        self.assertIn("Iron_Fe_Pct", df_master.columns)
        self.assertIn("Pot_Age_Days", df_master.columns)
        self.assertIn("Cathode_Vendor", df_master.columns)

        # Statistical sanity checks on parsed metrics
        self.assertTrue((df_master["Bath_Temperature_C"] > 800).all() or df_master["Bath_Temperature_C"].isna().any())
        self.assertTrue((df_master["Working_Voltage_V"] > 2.0).all() or df_master["Working_Voltage_V"].isna().any())

    def test_03_parse_workbook_pot_fleet_summary(self):
        if self.dfs is None:
            self.skipTest(f"{self.excel_path} not found")

        self.assertIn("Pot_Fleet_Summary", self.dfs)
        df_summary = self.dfs["Pot_Fleet_Summary"]

        self.assertGreaterEqual(len(df_summary), 190)
        self.assertIn("Pot_No", df_summary.columns)
        self.assertIn("Pot_Age_Days", df_summary.columns)
        self.assertIn("Cathode_Vendor", df_summary.columns)

    def test_04_parse_workbook_cathode_lining_history(self):
        if self.dfs is None:
            self.skipTest(f"{self.excel_path} not found")

        self.assertIn("Cathode_Lining_History", self.dfs)
        df_lining = self.dfs["Cathode_Lining_History"]

        self.assertGreater(len(df_lining), 1000)
        self.assertIn("Pot_No", df_lining.columns)
        self.assertIn("Evaluation_Month", df_lining.columns)
        self.assertIn("Cathode_Lining_Drop_V", df_lining.columns)


if __name__ == "__main__":
    unittest.main()
