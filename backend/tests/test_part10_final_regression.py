import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from engine.parser import parse_and_contract_file
from engine.cleaner import calculate_data_health_score_breakdown
from engine.dictionary import generate_data_dictionary
from engine.eda import run_automated_eda
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan
from engine.exporter import export_master_excel, export_executive_pdf
from engine.validator import validate_all_insights


class TestPart10FinalRegression(unittest.TestCase):

    def setUp(self):
        self.dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tmp_test_datasets"))
        self.datasets = ["ecom.csv", "saas.csv", "mkt.csv", "mfg.csv", "hr.csv"]

    def test_end_to_end_multi_dataset_pipeline(self):
        for ds_name in self.datasets:
            ds_path = os.path.join(self.dataset_dir, ds_name)
            if not os.path.exists(ds_path):
                print(f"Skipping missing dataset: {ds_path}")
                continue

            with self.subTest(dataset=ds_name):
                print(f"\n--- Testing Dataset: {ds_name} ---")
                # 1. Parsing & Contract
                parsed = parse_and_contract_file(ds_path)
                cleaned_df = parsed["cleaned_df"]
                contract = parsed["dataset_contract"]
                self.assertIsNotNone(cleaned_df)
                self.assertGreater(len(cleaned_df), 0)

                # 2. Data Quality & Health Score
                health = calculate_data_health_score_breakdown(cleaned_df)
                self.assertIn("overall_score", health)
                self.assertGreaterEqual(health["overall_score"], 0)

                # 3. Data Dictionary
                dict_list = generate_data_dictionary(cleaned_df)
                self.assertEqual(len(dict_list), len(cleaned_df.columns))

                # 4. Automated EDA
                eda_res = run_automated_eda(cleaned_df)
                self.assertIn("recommended_visualizations", eda_res)

                # 5. Insights & Validation Gate
                ins_res = generate_insights_and_story(cleaned_df, eda_res)
                insights = ins_res["insights"]
                self.assertGreater(len(insights), 0)

                valid_insights, rejected_insights = validate_all_insights(insights, cleaned_df, eda_res)
                self.assertGreater(len(valid_insights), 0, f"Validation gate failed for {ds_name}: no valid insights")

                # 6. Recommendations & Action Center Lineage
                rec_res = generate_recommendations_and_action_plan(insights)
                recs = rec_res["recommendations"]
                actions = rec_res["action_items"]

                self.assertGreater(len(recs), 0)
                self.assertGreater(len(actions), 0)

                for act in actions:
                    self.assertIn("lineage", act)
                    self.assertEqual(act["lineage"]["traceability_status"], "VERIFIED_LINEAGE")

                # 7. Deliverables Export
                excel_out = f"tmp/test_{ds_name}.xlsx"
                pdf_out = f"tmp/test_{ds_name}.pdf"

                export_master_excel(
                    raw_df=cleaned_df,
                    cleaned_df=cleaned_df,
                    master_df=cleaned_df,
                    dictionary_list=dict_list,
                    quality_issues=[],
                    eda_results=eda_res,
                    insights_list=insights,
                    recommendations_list=recs,
                    action_items_list=actions,
                    output_path=excel_out
                )
                self.assertTrue(os.path.exists(excel_out))

                export_executive_pdf(
                    eda_results=eda_res,
                    insights_list=insights,
                    recommendations_list=recs,
                    dataset_name=ds_name,
                    output_path=pdf_out
                )
                self.assertTrue(os.path.exists(pdf_out))

                # Clean up test output files
                if os.path.exists(excel_out):
                    os.remove(excel_out)
                if os.path.exists(pdf_out):
                    os.remove(pdf_out)

                print(f"PASSED Dataset: {ds_name}")


if __name__ == "__main__":
    unittest.main()
