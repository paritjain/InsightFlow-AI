import unittest
import pandas as pd
import numpy as np

from engine.semantic_resolver import UniversalSchemaAndSemanticResolver
from engine.contracts.chat_contracts import ResolutionStatus


class TestUniversalSchemaAndSemanticResolver(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 50
        self.df = pd.DataFrame({
            "Observation_ID": [f"OBS_{100+i}" for i in range(n)],
            "Plant_Zone": np.random.choice(["Workshop", "Assembly Line", "Warehouse", "Chemical Bay"], n),
            "Severity_Rating": np.random.randint(1, 6, n),
            "Likelihood_Rating": np.random.randint(1, 6, n),
            "Risk_Score": np.random.uniform(1, 25, n),
            "Risk_Index_Pct": np.random.uniform(4, 100, n),
            "Exposure_Hours": np.random.uniform(0.5, 40, n),
            "Incident_Date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "Is_Open": np.random.choice([True, False], n)
        })

        self.sem_dict = {
            "Observation_ID": {"semantic_role": "Identifier", "is_identifier": True, "physical_dtype": "string"},
            "Plant_Zone": {"semantic_role": "Dimension", "is_dimension": True, "semantic_subtype": "Geographic"},
            "Severity_Rating": {"semantic_role": "Measure", "is_measure": True, "semantic_subtype": "Rating"},
            "Likelihood_Rating": {"semantic_role": "Measure", "is_measure": True, "semantic_subtype": "Rating"},
            "Risk_Score": {"semantic_role": "Measure", "is_measure": True, "semantic_subtype": "Rating"},
            "Risk_Index_Pct": {"semantic_role": "Measure", "is_measure": True, "semantic_subtype": "Percentage"},
            "Exposure_Hours": {"semantic_role": "Measure", "is_measure": True, "semantic_subtype": "Duration"},
            "Incident_Date": {"semantic_role": "Date/Time", "is_time": True, "physical_dtype": "datetime64[ns]"},
            "Is_Open": {"semantic_role": "Dimension", "is_dimension": True, "physical_dtype": "bool"}
        }

        self.resolver = UniversalSchemaAndSemanticResolver(
            df=self.df,
            sem_layer_dict=self.sem_dict
        )

    def test_01_exact_and_normalized_matching(self):
        # Exact match
        res_exact = self.resolver.resolve_term("Exposure_Hours")
        self.assertEqual(res_exact.status, ResolutionStatus.EXACT_MATCH)
        self.assertEqual(res_exact.resolved_column, "Exposure_Hours")
        self.assertEqual(res_exact.confidence, 1.0)

        # Normalized match with spaces
        res_norm = self.resolver.resolve_term("exposure hours")
        self.assertEqual(res_norm.status, ResolutionStatus.EXACT_MATCH)
        self.assertEqual(res_norm.resolved_column, "Exposure_Hours")
        self.assertGreaterEqual(res_norm.confidence, 0.95)

    def test_02_value_entity_resolution(self):
        # Recognizes 'Workshop' as a cell value in Plant_Zone
        val_match = self.resolver.resolve_value_entity("Workshop")
        self.assertIsNotNone(val_match)
        col, val = val_match
        self.assertEqual(col, "Plant_Zone")
        self.assertEqual(val, "Workshop")

    def test_03_role_filtering_and_identifier_safety(self):
        # Primary key should be resolved as Identifier, NOT Measure
        res_id = self.resolver.resolve_term("Observation_ID", preferred_role="Measure")
        self.assertEqual(res_id.status, ResolutionStatus.NOT_FOUND)

        res_id_ok = self.resolver.resolve_term("Observation_ID", preferred_role="ID")
        self.assertEqual(res_id_ok.status, ResolutionStatus.EXACT_MATCH)
        self.assertTrue(res_id_ok.is_identifier)

    def test_04_disambiguation_and_ambiguity_detection(self):
        # 'Rating' matches both Severity_Rating and Likelihood_Rating
        res_rating = self.resolver.resolve_term("rating")
        self.assertEqual(res_rating.status, ResolutionStatus.AMBIGUOUS)
        self.assertIn("Severity_Rating", res_rating.candidate_alternatives)
        self.assertIn("Likelihood_Rating", res_rating.candidate_alternatives)

    def test_05_synonym_resolution(self):
        # Duration synonym
        res_dur = self.resolver.resolve_term("duration")
        self.assertEqual(res_dur.status, ResolutionStatus.SEMANTIC_MATCH)
        self.assertEqual(res_dur.resolved_column, "Exposure_Hours")

        # Temporal synonym
        res_time = self.resolver.resolve_term("date")
        self.assertEqual(res_time.status, ResolutionStatus.SEMANTIC_MATCH)
        self.assertEqual(res_time.resolved_column, "Incident_Date")

    def test_06_backward_compatibility_resolve_column(self):
        match = self.resolver.resolve_column("severity", role_filter="Measure")
        self.assertIsNotNone(match)
        col, conf = match
        self.assertEqual(col, "Severity_Rating")
        self.assertGreater(conf, 0.7)

    def test_07_academic_and_ecommerce_cross_domain_resolution(self):
        # E-Commerce Schema
        ecom_df = pd.DataFrame({
            "Order_ID": ["ORD_1", "ORD_2"],
            "Customer_UUID": ["UUID_1", "UUID_2"],
            "Unit_Price": [19.99, 49.50],
            "Net_Revenue": [199.90, 495.00],
            "Discount_Pct": [0.10, 0.20],
            "Product_Category": ["Apparel", "Footwear"]
        })
        ecom_resolver = UniversalSchemaAndSemanticResolver(df=ecom_df)

        # Revenue vs Discount_Pct disambiguation
        rev_match = ecom_resolver.resolve_term("sales", preferred_role="Measure")
        self.assertEqual(rev_match.resolved_column, "Net_Revenue")

        disc_match = ecom_resolver.resolve_term("discount", preferred_role="Measure")
        self.assertEqual(disc_match.resolved_column, "Discount_Pct")

        # Identifier resolution
        cust_match = ecom_resolver.resolve_term("customer", preferred_role="Dimension")
        self.assertEqual(cust_match.resolved_column, "Customer_UUID")

    def test_08_direct_binding_to_dataset_contract(self):
        from engine.contracts import DatasetContract
        contract = DatasetContract.create_from_pipeline(
            dataset_id="contract-test-1",
            raw_df=self.df,
            cleaned_df=self.df,
            source_file_path="safety.xlsx",
            dictionary=[{"column_name": c, "role": "Measure"} for c in self.df.columns]
        )
        contract_resolver = UniversalSchemaAndSemanticResolver(contract=contract, df=self.df)
        self.assertGreater(len(contract_resolver.columns), 0)
        res = contract_resolver.resolve_term("severity_rating")
        self.assertEqual(res.status, ResolutionStatus.EXACT_MATCH)
        self.assertEqual(res.resolved_column, "Severity_Rating")


    def test_09_extract_filters_from_query(self):
        # 1. Categorical cell value extraction
        q1 = "What is the average risk score for Workshop incidents?"
        filters1 = self.resolver.extract_filters_from_query(q1)
        self.assertEqual(len(filters1), 1)
        self.assertEqual(filters1[0].column, "Plant_Zone")
        self.assertEqual(filters1[0].value, "Workshop")

        # 2. Relational numeric comparison extraction
        q2 = "Filter records where Risk_Score > 15 and Exposure_Hours <= 20"
        filters2 = self.resolver.extract_filters_from_query(q2)
        self.assertEqual(len(filters2), 2)
        cols = {f.column: (f.operator, f.value) for f in filters2}
        self.assertIn("Risk_Score", cols)
        self.assertIn("Exposure_Hours", cols)
        self.assertEqual(cols["Risk_Score"], (">", 15))
        self.assertEqual(cols["Exposure_Hours"], ("<=", 20))

    def test_10_extract_entities_from_query(self):
        q = "Calculate the average risk score and exposure hours grouped by plant zone"
        entities = self.resolver.extract_entities_from_query(q)
        meas_cols = [m.resolved_column for m in entities["measures"]]
        dim_cols = [d.resolved_column for d in entities["dimensions"]]

        self.assertIn("Risk_Score", meas_cols)
        self.assertIn("Exposure_Hours", meas_cols)
        self.assertIn("Plant_Zone", dim_cols)


if __name__ == "__main__":
    unittest.main()

