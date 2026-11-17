import os
import unittest
import pandas as pd
from main import parse_excel_workbook
from engine.cleaner import coerce_numeric_columns
from engine.dictionary import generate_data_dictionary
from engine.eda import run_automated_eda
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan

class TestDetailObservationReportPipeline(unittest.TestCase):
    def setUp(self):
        # Look for the file in workspace root or store
        self.file_path = "Detail Observation Report-Smelter-Jul'26.xlsx"
        if not os.path.exists(self.file_path):
            self.file_path = os.path.join("..", "Detail Observation Report-Smelter-Jul'26.xlsx")
        self.assertTrue(os.path.exists(self.file_path), f"File {self.file_path} not found")

    def test_smart_header_detection_and_table_boundary(self):
        sheets = parse_excel_workbook(self.file_path)
        self.assertGreater(len(sheets), 0)
        
        sheet_key = list(sheets.keys())[0]
        df = sheets[sheet_key]

        # Verify header starts at row 4 ('Entity', 'Observation Id', ...)
        expected_cols = [
            'Entity', 'Observation Id', 'Observer', "Observer's Job Title",
            'Observation Date', 'Location', 'Describe activity', 'Comment',
            'Check List', 'Behaviour', 'Potential Impact', 'Department head', 'Closure Status'
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Expected column '{col}' not in parsed columns: {df.columns.tolist()}")

        # Verify no Unnamed or title banner columns
        for col in df.columns:
            self.assertFalse(str(col).lower().startswith("unnamed:"), f"Found unnamed column: {col}")
            self.assertNotIn("Detail Observation Report-", str(col))

        # Verify footer rows trimmed: exactly 3,484 operational observation records
        self.assertEqual(len(df), 3484)

    def test_semantic_profiling_and_data_dictionary(self):
        sheets = parse_excel_workbook(self.file_path)
        df = list(sheets.values())[0]

        d_dict = generate_data_dictionary(df, self.file_path)
        dict_map = {c['column_name']: c for c in d_dict}

        # Observation Id must be Identifier
        self.assertEqual(dict_map['Observation Id']['semantic_role'], 'Identifier')

        # Observation Date must be Date/Time
        self.assertEqual(dict_map['Observation Date']['semantic_role'], 'Date/Time')
        self.assertEqual(dict_map['Observation Date']['semantic_subtype'], 'Date')

        # Location must be Geographic or Dimension
        self.assertIn(dict_map['Location']['semantic_role'], ['Geographic', 'Dimension'])

        # Potential Impact and Closure Status must be Dimensions
        self.assertEqual(dict_map['Potential Impact']['semantic_role'], 'Dimension')
        self.assertEqual(dict_map['Closure Status']['semantic_role'], 'Dimension')

    def test_automated_eda_event_visualizations(self):
        sheets = parse_excel_workbook(self.file_path)
        df = list(sheets.values())[0]

        eda = run_automated_eda(df)
        
        # Verify time series is computed
        self.assertEqual(eda['time_series_analysis']['status'], 'sufficient_data')
        self.assertEqual(len(eda['time_series_analysis']['monthly_trend']), 31)

        # Verify recommended visualizations
        rec_vis = eda['recommended_visualizations']
        self.assertGreaterEqual(len(rec_vis), 5)

        chart_types = [v['chart_type'] for v in rec_vis]
        self.assertIn('LINE', chart_types)
        self.assertTrue(any(t in chart_types for t in ['HORIZONTAL_BAR', 'BAR']))
        self.assertIn('DONUT', chart_types)
        self.assertIn('STACKED_BAR', chart_types)

        # Verify all recommended visualizations have valid data and takeaways
        for v in rec_vis:
            self.assertTrue(v['is_valid'])
            self.assertGreater(len(v['data']), 0)
            self.assertIsNotNone(v.get('takeaway'))
            self.assertIsNotNone(v.get('suitability_reason'))

    def test_insights_and_action_plan(self):
        sheets = parse_excel_workbook(self.file_path)
        df = list(sheets.values())[0]

        eda = run_automated_eda(df)
        ins_res = generate_insights_and_story(df, eda)
        insights = ins_res['insights']

        self.assertGreater(len(insights), 0)
        for ins in insights:
            self.assertIsNotNone(ins.get('observed_fact'))
            self.assertIsNotNone(ins.get('business_interpretation'))
            self.assertIsNotNone(ins.get('supporting_visualization'))

        recs = generate_recommendations_and_action_plan(insights)
        self.assertGreater(len(recs['recommendations']), 0)
        self.assertGreater(len(recs['action_items']), 0)

if __name__ == '__main__':
    unittest.main()
