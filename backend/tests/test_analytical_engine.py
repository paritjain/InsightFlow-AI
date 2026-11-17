import os
import sys
import unittest
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from engine.profiler import infer_column_semantic_metadata, profile_file
from engine.cleaner import detect_quality_issues, apply_cleaning_rules
from engine.dictionary import generate_data_dictionary
from engine.eda import run_automated_eda, validate_chart_spec
from engine.insights import generate_insights_and_story
from engine.validator import validate_insight, validate_all_insights, build_insight_lineage
from engine.action_plan import generate_recommendations_and_action_plan
from engine.exporter import export_master_excel

class TestAnalyticalEngine(unittest.TestCase):

    def setUp(self):
        self.sample_mixed_df = pd.DataFrame({
            'revenue': ['$1,000.50', '$2,500.00', '$1,000.50', '-$500.00', '15%'],
            'region': [' North ', ' north', 'South', 'South', 'North'],
            'transaction_date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
        })
        self.sample_small_df = pd.DataFrame({
            'item': ['A', 'B'],
            'score': [10, 20]
        })

    def test_profiler_and_data_dictionary(self):
        dict_list = generate_data_dictionary(self.sample_mixed_df, "mixed_data.csv")
        self.assertEqual(len(dict_list), 3)
        col_names = [d["column_name"] for d in dict_list]
        self.assertIn("revenue", col_names)
        self.assertIn("region", col_names)

    def test_quality_ledger_and_cleaning(self):
        issues = detect_quality_issues(self.sample_mixed_df)
        self.assertTrue(len(issues) > 0)
        for issue in issues:
            self.assertIn("issue", issue)
            self.assertIn("treatment", issue)
            self.assertIn("reason", issue)
            self.assertIn("impact", issue)
            self.assertIn("reversibility", issue)
            self.assertTrue(issue["reversibility"])

        cleaned_df, audit_trail = apply_cleaning_rules(self.sample_mixed_df, issues)
        self.assertEqual(len(audit_trail), len(issues))
        for entry in audit_trail:
            self.assertIn("step", entry)
            self.assertIn("rule_id", entry)
            self.assertIn("impact", entry)

    def test_eda_insufficient_data(self):
        eda_res = run_automated_eda(self.sample_small_df)
        self.assertEqual(eda_res["overview"]["num_rows"], 2)
        self.assertEqual(eda_res["relationships"]["status"], "insufficient_data")
        self.assertEqual(eda_res["time_series_analysis"]["status"], "insufficient_data")

    def test_validator_and_causal_language_rejection(self):
        df_10 = pd.DataFrame({'revenue': range(10), 'region': ['North']*10})
        bad_insight = {
            "id": "insight_test",
            "title": "Revenue Causes Regional Growth",
            "insight": "Higher revenue causes expansion into new regions.",
            "priority": "High",
            "calculation_method": "Pearson correlation",
            "statistical_evidence": "Verified calculation",
            "numerical_evidence": [{"metric": "Pearson r", "value": "0.85"}]
        }
        is_valid, errors, validated = validate_insight(bad_insight, df_10)
        self.assertFalse(is_valid)
        self.assertTrue(any("causal" in err.lower() for err in errors))
        self.assertEqual(validated["validation_status"], "FAILED")

    def test_valid_insight_lineage_generation(self):
        eda_res = run_automated_eda(self.sample_mixed_df)
        insights_res = generate_insights_and_story(self.sample_mixed_df, eda_res)
        insights = insights_res["insights"]
        self.assertTrue(len(insights) > 0)
        for ins in insights:
            self.assertEqual(ins["validation_status"], "PASSED")
            self.assertIn("lineage", ins)
            lineage = ins["lineage"]
            self.assertEqual(lineage["insight_id"], ins["id"])
            self.assertEqual(lineage["validation_status"], "PASSED")

    def test_recommendations_9_required_fields(self):
        eda_res = run_automated_eda(self.sample_mixed_df)
        insights_res = generate_insights_and_story(self.sample_mixed_df, eda_res)
        recs_res = generate_recommendations_and_action_plan(insights_res["insights"])
        recs = recs_res["recommendations"]
        self.assertTrue(len(recs) > 0)
        for r in recs:
            for field in ["title", "problem", "recommended_action", "evidence", "priority", "owner", "kpi", "expected_impact", "timeline"]:
                self.assertIn(field, r)
                self.assertIsNotNone(r[field])

    def test_export_deliverable(self):
        issues = detect_quality_issues(self.sample_mixed_df)
        cleaned_df, audit_trail = apply_cleaning_rules(self.sample_mixed_df, issues)
        dict_list = generate_data_dictionary(cleaned_df, "test.csv")
        eda_res = run_automated_eda(cleaned_df)
        insights_res = generate_insights_and_story(cleaned_df, eda_res)
        recs_res = generate_recommendations_and_action_plan(insights_res["insights"])

        excel_path = os.path.abspath("test_deliverable_unit.xlsx")
        out_excel = export_master_excel(
            raw_df=self.sample_mixed_df,
            cleaned_df=cleaned_df,
            master_df=cleaned_df,
            dictionary_list=dict_list,
            quality_issues=issues,
            eda_results=eda_res,
            insights_list=insights_res["insights"],
            recommendations_list=recs_res["recommendations"],
            action_items_list=recs_res["action_items"],
            output_path=excel_path
        )
        self.assertTrue(os.path.exists(out_excel))
        if os.path.exists(out_excel):
            os.remove(out_excel)

if __name__ == '__main__':
    unittest.main()
