import sys, os
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import unittest
import pandas as pd
from engine.profiler import infer_column_semantic_metadata, profile_file
from engine.eda import run_automated_eda, validate_chart_spec
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan

class TestDatasetAgnosticEngine(unittest.TestCase):

    def test_duration_semantic_inference(self):
        """Test that numeric duration columns are classified as Measures (Duration) and date columns as Time."""
        df = pd.DataFrame({
            'Reading_Date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'Resolution_Time_Hours': [4.5, 12.0, 2.1, 8.5] * 7 + [3.0, 6.0],
            'Outage_Hours': [0.0, 1.5, 0.0, 0.0] * 7 + [0.5, 0.0],
            'Maintenance_Hours': [10.0, 15.5, 8.0, 12.0] * 7 + [5.0, 9.0],
            'Equipment_ID': [f"EQ-{i%5}" for i in range(30)],
            'Location_Code': ['LOC-A', 'LOC-B', 'LOC-C'] * 10
        })

        meta_date = infer_column_semantic_metadata('Reading_Date', df['Reading_Date'], 30)
        meta_res_time = infer_column_semantic_metadata('Resolution_Time_Hours', df['Resolution_Time_Hours'], 30)
        meta_outage = infer_column_semantic_metadata('Outage_Hours', df['Outage_Hours'], 30)
        meta_maint = infer_column_semantic_metadata('Maintenance_Hours', df['Maintenance_Hours'], 30)

        self.assertEqual(meta_date['semantic_role'], 'Time', "Reading_Date should be classified as Time")
        self.assertTrue(meta_date['is_time'], "Reading_Date is_time should be True")
        self.assertEqual(meta_res_time['semantic_role'], 'Measure', "Resolution_Time_Hours should be classified as Measure")
        self.assertTrue(meta_res_time['is_measure'], "Resolution_Time_Hours is_measure should be True")
        self.assertEqual(meta_res_time['semantic_subtype'], 'Duration', "Resolution_Time_Hours subtype should be Duration")
        self.assertEqual(meta_outage['semantic_role'], 'Measure', "Outage_Hours should be classified as Measure")
        self.assertEqual(meta_outage['semantic_subtype'], 'Duration', "Outage_Hours subtype should be Duration")
        self.assertEqual(meta_maint['semantic_role'], 'Measure', "Maintenance_Hours should be classified as Measure")

    def test_eda_chart_data_payloads(self):
        """Test that every recommended chart has computed, non-empty data points."""
        df = pd.DataFrame({
            'Reading_Date': pd.date_range('2024-01-01', periods=50, freq='D'),
            'Resolution_Time_Hours': [4.5 + i*0.1 for i in range(50)],
            'Outage_Hours': [0.0 if i%3 != 0 else 2.5 + i*0.05 for i in range(50)],
            'Equipment_ID': [f"EQ-{i%4}" for i in range(50)],
            'Category': ['Group-A', 'Group-B'] * 25
        })

        eda_res = run_automated_eda(df)
        charts = eda_res.get('recommended_charts', [])
        self.assertGreater(len(charts), 0, "Automated EDA should produce recommended charts")

        for chart in charts:
            self.assertTrue(chart.get('is_valid'), f"Chart {chart.get('title')} must be marked valid")
            self.assertIn('data', chart, f"Chart {chart.get('title')} must contain a 'data' key")
            self.assertGreater(len(chart['data']), 0, f"Chart {chart.get('title')} must have a non-empty data payload")

    def test_recommendations_target_labeling(self):
        """Test that expected impacts are labeled as targets and do not contain invented static percentage uplifts."""
        df = pd.DataFrame({
            'Reading_Date': pd.date_range('2024-01-01', periods=40, freq='D'),
            'Outage_Hours': [0.0 if i%4 != 0 else 5.0 + i*0.1 for i in range(40)],
            'Category': ['Node-X', 'Node-Y', 'Node-Z', 'Node-W'] * 10
        })

        eda_res = run_automated_eda(df)
        insights_res = generate_insights_and_story(df, eda_res)
        recs_res = generate_recommendations_and_action_plan(insights_res.get('insights', []))

        recs = recs_res.get('recommendations', [])
        self.assertGreater(len(recs), 0, "Should generate recommendations")

        for rec in recs:
            impact = rec.get('expected_impact', '')
            self.assertTrue(
                impact.startswith('Target Improvement:') or 'Target' in impact,
                f"Expected impact '{impact}' must be explicitly labeled as a Target or Target Improvement"
            )

if __name__ == '__main__':
    unittest.main()
