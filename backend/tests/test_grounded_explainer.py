import unittest
import pandas as pd
import numpy as np

from engine.duckdb_engine import DuckDBAnalyticalEngine
from engine.query_planner import UniversalQueryPlanner
from engine.result_validator import UniversalResultValidator
from engine.grounded_explainer import GroundedAnalyticalExplainer
from engine.contracts.chat_contracts import (
    ResultValidationStatus,
    QueryIntent,
    ColumnFilter,
    FilterOperator,
    ChatResponseContract
)


class TestGroundedExplainerAndValidator(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 80
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

        self.planner = UniversalQueryPlanner(df=self.df, sem_layer_dict=self.sem_dict)
        self.engine = DuckDBAnalyticalEngine(df=self.df)
        self.validator = UniversalResultValidator(col_metadata=self.sem_dict)
        self.explainer = GroundedAnalyticalExplainer(col_metadata=self.sem_dict)

    def test_01_result_validator_filter_satisfaction(self):
        plan = self.planner.plan("Show incidents in Workshop where Exposure_Hours <= 20")
        raw_res = self.engine.execute_plan(plan)
        val_res = self.validator.validate(plan, raw_res, self.df)

        self.assertEqual(val_res.status, ResultValidationStatus.PASS)
        # All rows strictly satisfied
        self.assertGreater(val_res.rows_returned, 0)

    def test_02_result_validator_sample_size_and_power(self):
        # Create a tiny 2-row dataset
        tiny_df = self.df.head(2).copy()
        tiny_engine = DuckDBAnalyticalEngine(df=tiny_df)
        plan = self.planner.plan("What is the correlation between Severity_Rating and Risk_Score?")

        raw_res = tiny_engine.execute_plan(plan)
        val_res = self.validator.validate(plan, raw_res, tiny_df)

        self.assertEqual(val_res.status, ResultValidationStatus.WARNING)
        self.assertTrue(any("Sample size" in w for w in val_res.warnings))

    def test_03_result_validator_constant_measure(self):
        const_df = self.df.copy()
        const_df["Risk_Score"] = 10.0
        const_engine = DuckDBAnalyticalEngine(df=const_df)

        plan = self.planner.plan("What is the average risk score across the dataset?")
        raw_res = const_engine.execute_plan(plan)
        val_res = self.validator.validate(plan, raw_res, const_df)

        self.assertTrue(any("constant" in w for w in val_res.warnings))

    def test_04_grounded_explainer_global_aggregation(self):
        plan = self.planner.plan("What is the average risk score across the dataset?")
        raw_res = self.engine.execute_plan(plan)
        val_res = self.validator.validate(plan, raw_res, self.df)
        response = self.explainer.explain(plan, val_res)

        self.assertIsInstance(response, ChatResponseContract)
        self.assertEqual(response.status, ResultValidationStatus.PASS)
        self.assertIsNotNone(response.numerical_value)
        self.assertIn("average", response.answer.lower())
        self.assertIn("Risk_Score", response.answer)
        # Directly grounds numerical value
        self.assertIn(f"{response.numerical_value:.2f}", response.answer)

    def test_05_grounded_explainer_group_comparison_and_table(self):
        plan = self.planner.plan("Show average severity rating by plant zone")
        raw_res = self.engine.execute_plan(plan)
        val_res = self.validator.validate(plan, raw_res, self.df)
        response = self.explainer.explain(plan, val_res)

        self.assertEqual(response.status, ResultValidationStatus.PASS)
        self.assertIn("Plant_Zone", response.answer)
        self.assertIn("|", response.answer)  # Markdown table present
        self.assertGreater(len(response.evidence), 0)

    def test_06_grounded_explainer_correlation(self):
        plan = self.planner.plan("What is the correlation between Severity_Rating and Risk_Score?")
        raw_res = self.engine.execute_plan(plan)
        val_res = self.validator.validate(plan, raw_res, self.df)
        response = self.explainer.explain(plan, val_res)

        self.assertEqual(response.status, ResultValidationStatus.PASS)
        self.assertIn("Pearson correlation", response.answer)
        self.assertIn("Severity_Rating", response.answer)
        self.assertIn("Risk_Score", response.answer)
        self.assertIn("relationship", response.answer.lower())

    def test_07_grounded_explainer_outlier_summary(self):
        # Inject extreme outlier
        outlier_df = self.df.copy()
        outlier_df.loc[0, "Exposure_Hours"] = 999.0
        engine = DuckDBAnalyticalEngine(df=outlier_df)

        plan = self.planner.plan("Are there outliers in Exposure_Hours?")
        raw_res = engine.execute_plan(plan)
        val_res = self.validator.validate(plan, raw_res, outlier_df)
        response = self.explainer.explain(plan, val_res)

        self.assertEqual(response.status, ResultValidationStatus.PASS)
        self.assertIn("outlier", response.answer.lower())
        self.assertIn("Exposure_Hours", response.answer)
        self.assertGreater(len(response.evidence), 0)

    def test_08_grounded_explainer_zero_hallucination(self):
        # Querying an average must NOT claim to be Pearson correlation
        plan = self.planner.plan("What is the average risk score across the dataset?")
        raw_res = self.engine.execute_plan(plan)
        val_res = self.validator.validate(plan, raw_res, self.df)
        response = self.explainer.explain(plan, val_res)

        self.assertNotIn("pearson", response.answer.lower())
        self.assertNotIn("correlation", response.answer.lower())

    def test_09_grounded_explainer_ambiguity_clarification(self):
        plan = self.planner.plan("What is the average rating?")
        raw_res = self.engine.execute_plan(plan)
        val_res = self.validator.validate(plan, raw_res, self.df)
        response = self.explainer.explain(plan, val_res)

        self.assertEqual(response.status, ResultValidationStatus.AMBIGUOUS)
        self.assertTrue(response.clarification_required)
        self.assertIn("Severity_Rating", response.answer)
        self.assertIn("Likelihood_Rating", response.answer)

    def test_10_chat_response_contract_pydantic_and_dict_access(self):
        plan = self.planner.plan("What is the average risk score across the dataset?")
        raw_res = self.engine.execute_plan(plan)
        val_res = self.validator.validate(plan, raw_res, self.df)
        response = self.explainer.explain(plan, val_res)

        # Attribute access
        self.assertEqual(response.intent, QueryIntent.GLOBAL_AGGREGATION)
        # Dictionary bracket access
        self.assertEqual(response["intent"], "global_aggregation")
        self.assertIn("answer", response)
        self.assertEqual(response["status"], ResultValidationStatus.PASS)


if __name__ == "__main__":
    unittest.main()
