import unittest
import pandas as pd
import numpy as np

from agents.graph import LangGraphDataAgent, session_manager
from agents.tools import DatasetQueryTool
from engine.contracts import DatasetContract
from engine.contracts.chat_contracts import ResultValidationStatus, QueryIntent


class TestPlatformIntegrationMultiTurn(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 80
        self.df = pd.DataFrame({
            "Observation_ID": [f"OBS_{100+i}" for i in range(n)],
            "Plant_Zone": np.random.choice(["Workshop", "Assembly Line", "Warehouse", "Chemical Bay"], n),
            "Severity_Rating": np.random.randint(1, 6, n),
            "Likelihood_Rating": np.random.randint(1, 6, n),
            "Risk_Score": np.random.uniform(1, 25, n),
            "Exposure_Hours": np.random.uniform(0.5, 40, n),
            "Incident_Date": pd.date_range("2026-01-01", periods=n, freq="D")
        })

        self.dictionary = [
            {"column_name": "Observation_ID", "role": "Identifier", "is_identifier": True},
            {"column_name": "Plant_Zone", "role": "Dimension", "is_dimension": True, "semantic_subtype": "Geographic"},
            {"column_name": "Severity_Rating", "role": "Measure", "is_measure": True, "semantic_subtype": "Rating"},
            {"column_name": "Likelihood_Rating", "role": "Measure", "is_measure": True, "semantic_subtype": "Rating"},
            {"column_name": "Risk_Score", "role": "Measure", "is_measure": True, "semantic_subtype": "Rating"},
            {"column_name": "Exposure_Hours", "role": "Measure", "is_measure": True, "semantic_subtype": "Duration"},
            {"column_name": "Incident_Date", "role": "Date/Time", "is_time": True}
        ]

        self.contract = DatasetContract.create_from_pipeline(
            dataset_id="test-proj-1",
            raw_df=self.df,
            cleaned_df=self.df,
            source_file_path="safety.xlsx",
            dictionary=self.dictionary
        )

        self.proj_id = "test-proj-1"
        session_manager.clear_session(self.proj_id)

    def test_01_multiturn_context_carryover_pronoun(self):
        # Turn 1: Ask about Risk_Score
        res1 = LangGraphDataAgent.process_message(
            project_id=self.proj_id,
            df=self.df,
            sem_layer={},
            eda_results={},
            insights=[],
            user_message="What is the average risk score across the dataset?",
            contract=self.contract,
            dictionary=self.dictionary
        )
        self.assertIn("Risk_Score", res1["answer"])

        # Turn 2: Follow-up using pronoun "in it"
        res2 = LangGraphDataAgent.process_message(
            project_id=self.proj_id,
            df=self.df,
            sem_layer={},
            eda_results={},
            insights=[],
            user_message="Are there outliers in it?",
            contract=self.contract,
            dictionary=self.dictionary
        )
        # Should understand "it" refers to Risk_Score
        self.assertIn("Risk_Score", res2["answer"])
        self.assertIn("outlier", res2["answer"].lower())

    def test_02_multiturn_context_carryover_filter_shorthand(self):
        # Turn 1: Group comparison of Severity_Rating
        res1 = LangGraphDataAgent.process_message(
            project_id=self.proj_id,
            df=self.df,
            sem_layer={},
            eda_results={},
            insights=[],
            user_message="Show average severity rating by plant zone",
            contract=self.contract,
            dictionary=self.dictionary
        )
        self.assertIn("Severity_Rating", res1["answer"])

        # Turn 2: Follow-up shorthand "What about for workshop?"
        res2 = LangGraphDataAgent.process_message(
            project_id=self.proj_id,
            df=self.df,
            sem_layer={},
            eda_results={},
            insights=[],
            user_message="What about for Workshop?",
            contract=self.contract,
            dictionary=self.dictionary
        )
        # Should evaluate Severity_Rating for Workshop
        self.assertIn("Severity_Rating", res2["answer"])
        self.assertIn("Workshop", res2["answer"])

    def test_03_direct_dataset_contract_binding_through_agent(self):
        res = LangGraphDataAgent.process_message(
            project_id=self.proj_id,
            df=self.df,
            sem_layer={},
            eda_results={},
            insights=[],
            user_message="Top 3 plant zones by exposure hours",
            contract=self.contract,
            dictionary=self.dictionary
        )
        self.assertEqual(res["tool_used"], "DatasetQueryTool")
        self.assertIn("Exposure_Hours", res["answer"])
        self.assertIn("Plant_Zone", res["answer"])

    def test_04_session_isolation_across_projects(self):
        proj_a = "proj-alpha"
        proj_b = "proj-beta"
        session_manager.clear_session(proj_a)
        session_manager.clear_session(proj_b)

        # Set context in Project A
        LangGraphDataAgent.process_message(
            project_id=proj_a,
            df=self.df,
            sem_layer={},
            eda_results={},
            insights=[],
            user_message="What is the average risk score?",
            contract=self.contract
        )
        self.assertEqual(session_manager.get_context(proj_a).get("last_metric"), "Risk_Score")

        # Project B context should be empty
        self.assertIsNone(session_manager.get_context(proj_b).get("last_metric"))

    def test_05_dataset_query_tool_structured_response(self):
        res = DatasetQueryTool.run(
            df=self.df,
            sem_layer={},
            query="What is the correlation between Severity_Rating and Risk_Score?",
            contract=self.contract,
            dictionary=self.dictionary,
            project_id=self.proj_id
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["tool"], "DatasetQueryTool")
        self.assertEqual(res["status"], ResultValidationStatus.PASS)
        self.assertIsNotNone(res["numerical_value"])
        self.assertIn("response_contract", res)


if __name__ == "__main__":
    unittest.main()
