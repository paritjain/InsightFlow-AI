import unittest
import pandas as pd
import numpy as np

from engine.profiler import infer_column_semantic_metadata
from engine.dictionary import generate_data_dictionary
from engine.cleaner import calculate_data_health_score_breakdown, detect_quality_issues


class TestPart2DataQualityAndSemanticDictionary(unittest.TestCase):

    def test_1_semantic_role_inference(self):
        # 1. Identifier
        s_id = pd.Series([101, 102, 103, 104, 105], name="Customer_ID")
        meta_id = infer_column_semantic_metadata("Customer_ID", s_id, 5)
        self.assertIn(meta_id["semantic_role"], ["Foreign Key", "Identifier"])
        self.assertTrue(meta_id["is_identifier"])
        self.assertFalse(meta_id["aggregation_eligibility"])

        # 2. Measure (Currency)
        s_sales = pd.Series([150.50, 200.00, 350.25, 99.00, 500.00], name="Sales_Amount")
        meta_sales = infer_column_semantic_metadata("Sales_Amount", s_sales, 5)
        self.assertEqual(meta_sales["semantic_role"], "Measure")
        self.assertTrue(meta_sales["aggregation_eligibility"])
        self.assertTrue(meta_sales["is_additive"])

        # 3. Dimension (Category)
        s_cat = pd.Series(["Electronics", "Apparel", "Electronics", "Home", "Apparel"], name="Category")
        meta_cat = infer_column_semantic_metadata("Category", s_cat, 5)
        self.assertEqual(meta_cat["semantic_role"], "Dimension")
        self.assertTrue(meta_cat["grouping_eligibility"])

        # 4. Date/Time
        s_date = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], name="Order_Date")
        meta_date = infer_column_semantic_metadata("Order_Date", s_date, 5)
        self.assertEqual(meta_date["semantic_role"], "Date/Time")

        # 5. Boolean / Binary Category
        s_bool = pd.Series(["Yes", "No", "Yes", "No", "Yes"], name="Is_Active")
        meta_bool = infer_column_semantic_metadata("Is_Active", s_bool, 5)
        self.assertIn(meta_bool["semantic_role"], ["Dimension", "Boolean"])
        self.assertEqual(meta_bool["semantic_type"], "Binary Category")

        # 6. Geographic
        s_geo = pd.Series(["California", "Texas", "New York", "Florida", "Illinois"], name="State")
        meta_geo = infer_column_semantic_metadata("State", s_geo, 5)
        self.assertEqual(meta_geo["semantic_role"], "Geographic")

        # 7. Target/Outcome
        s_target = pd.Series([1, 0, 1, 1, 0], name="Churn_Target")
        meta_target = infer_column_semantic_metadata("Churn_Target", s_target, 5)
        self.assertEqual(meta_target["semantic_role"], "Target/Outcome")

    def test_2_manual_semantic_overrides(self):
        s_num = pd.Series([1, 2, 3, 4, 5], name="Custom_Code")
        override = {"semantic_role": "Dimension"}
        meta_overridden = infer_column_semantic_metadata("Custom_Code", s_num, 5, override=override)
        
        self.assertEqual(meta_overridden["semantic_role"], "Dimension")
        self.assertTrue(meta_overridden["override_applied"])
        self.assertTrue(meta_overridden["grouping_eligibility"])

    def test_3_data_health_score_breakdown(self):
        df_imperfect = pd.DataFrame({
            "ID": [1, 2, 3, 4, 4], # 1 duplicate row
            "Revenue": [100.0, None, 300.0, 400.0, 400.0], # 1 null
            "Status": ["Active", "Active", "Active", "Active", "Active"] # Constant column
        })

        score_res = calculate_data_health_score_breakdown(df_imperfect)

        self.assertIn("overall_score", score_res)
        self.assertIn("completeness_score", score_res)
        self.assertIn("validity_score", score_res)
        self.assertIn("uniqueness_score", score_res)
        self.assertIn("schema_integrity_score", score_res)
        self.assertIn("breakdown_reason", score_res)
        self.assertLess(score_res["overall_score"], 100.0)


if __name__ == "__main__":
    unittest.main()
