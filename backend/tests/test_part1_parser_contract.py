import unittest
import os
import tempfile
import pandas as pd
import numpy as np

from engine.parser import detect_tabular_region, parse_and_contract_file


class TestPart1ParserAndDatasetContract(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_1_clean_excel_dataset(self):
        df_clean = pd.DataFrame({
            "Region": ["North", "South", "East", "West"],
            "Sales": [1000, 1500, 1200, 1800],
            "Units": [10, 15, 12, 18]
        })
        file_path = os.path.join(self.temp_dir, "clean_data.xlsx")
        df_clean.to_excel(file_path, index=False)

        res = parse_and_contract_file(file_path)
        contract = res["dataset_contract"]
        cleaned_df = res["cleaned_df"]

        self.assertEqual(contract["structure"]["record_count"], 4)
        self.assertEqual(contract["structure"]["column_count"], 3)
        self.assertEqual(contract["structure"]["header_row_index"], 0)
        self.assertEqual(list(cleaned_df.columns), ["Region", "Sales", "Units"])

    def test_2_excel_with_title_rows_above_header(self):
        # Create Excel with 2 title/instruction rows above real header
        raw_rows = [
            ["ANNUAL REGIONAL SALES REPORT 2024", None, None],
            ["Confidential - Internal Business Intelligence", None, None],
            ["Region", "Revenue", "Profit"],
            ["North", 50000, 12000],
            ["South", 65000, 15000],
            ["East", 48000, 11000]
        ]
        df_raw = pd.DataFrame(raw_rows)
        file_path = os.path.join(self.temp_dir, "title_rows.xlsx")
        df_raw.to_excel(file_path, index=False, header=False)

        res = parse_and_contract_file(file_path)
        contract = res["dataset_contract"]
        cleaned_df = res["cleaned_df"]

        self.assertEqual(contract["structure"]["header_row_index"], 2)
        self.assertEqual(contract["structure"]["title_rows_removed"], 2)
        self.assertEqual(contract["structure"]["record_count"], 3)
        self.assertEqual(list(cleaned_df.columns), ["Region", "Revenue", "Profit"])

    def test_3_dataset_containing_blank_columns(self):
        # DataFrame with an entirely blank middle column
        df = pd.DataFrame({
            "Customer_ID": [101, 102, 103],
            "Empty_Col": [None, None, None],
            "Amount": [250.5, 400.0, 150.0]
        })
        file_path = os.path.join(self.temp_dir, "blank_cols.xlsx")
        df.to_excel(file_path, index=False)

        res = parse_and_contract_file(file_path)
        contract = res["dataset_contract"]
        cleaned_df = res["cleaned_df"]

        self.assertEqual(contract["structure"]["column_count"], 2)
        self.assertNotIn("Empty_Col", cleaned_df.columns)
        self.assertIn("Customer_ID", cleaned_df.columns)
        self.assertIn("Amount", cleaned_df.columns)

    def test_4_dataset_containing_unnamed_columns(self):
        # Header has empty string column
        df_raw = pd.DataFrame([
            ["ID", "", "Category", "Value"],
            [1, "Extra Data", "Electronics", 99.9],
            [2, "More Data", "Apparel", 49.5]
        ])
        file_path = os.path.join(self.temp_dir, "unnamed_cols.csv")
        df_raw.to_csv(file_path, index=False, header=False)

        res = parse_and_contract_file(file_path)
        contract = res["dataset_contract"]
        cleaned_df = res["cleaned_df"]

        self.assertEqual(contract["structure"]["record_count"], 2)
        self.assertIn("Column_2", cleaned_df.columns) # Renamed meaningfully from empty string
        self.assertEqual(cleaned_df["Column_2"].tolist(), ["Extra Data", "More Data"])

    def test_5_mixed_data_types(self):
        df_mixed = pd.DataFrame({
            "Transaction_ID": ["TX101", "TX102", "TX103"],
            "Score": [95.5, "Unknown", 88.0],
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"]
        })
        file_path = os.path.join(self.temp_dir, "mixed_data.csv")
        df_mixed.to_csv(file_path, index=False)

        res = parse_and_contract_file(file_path)
        contract = res["dataset_contract"]
        cleaned_df = res["cleaned_df"]

        self.assertEqual(contract["structure"]["record_count"], 3)
        self.assertIn("Score", contract["column_contracts"])

    def test_6_dataset_containing_missing_values(self):
        df_nulls = pd.DataFrame({
            "Item": ["A", "B", None, "D"],
            "Quantity": [10, None, 30, 40]
        })
        file_path = os.path.join(self.temp_dir, "nulls.csv")
        df_nulls.to_csv(file_path, index=False)

        res = parse_and_contract_file(file_path)
        contract = res["dataset_contract"]
        quality = contract["quality_summary"]

        self.assertEqual(contract["structure"]["record_count"], 4)
        self.assertEqual(quality["total_nulls"], 2)
        self.assertEqual(quality["completeness_score"], 75.0)

    def test_7_csv_dataset(self):
        df_csv = pd.DataFrame({
            "Store": ["NYC", "LA", "CHI"],
            "Sales": [500, 700, 600]
        })
        file_path = os.path.join(self.temp_dir, "store_sales.csv")
        df_csv.to_csv(file_path, index=False)

        res = parse_and_contract_file(file_path)
        contract = res["dataset_contract"]

        self.assertEqual(contract["metadata"]["source_type"], "CSV")
        self.assertEqual(contract["structure"]["record_count"], 3)


if __name__ == "__main__":
    unittest.main()
