import unittest
import pandas as pd
import numpy as np

from engine.duckdb_engine import DuckDBAnalyticalEngine
from engine.query_planner import UniversalQueryPlanner
from engine.contracts.chat_contracts import ResultValidationStatus, QueryIntent, ColumnFilter, FilterOperator


class TestDuckDBAnalyticalEngine(unittest.TestCase):

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

    def test_01_global_aggregation_execution(self):
        plan = self.planner.plan("What is the average risk score across the dataset?")
        res = self.engine.execute_plan(plan)

        self.assertEqual(res.status, ResultValidationStatus.PASS)
        self.assertEqual(res.engine_used, "DuckDB")
        self.assertGreater(res.execution_time_ms, 0.0)
        self.assertLess(res.execution_time_ms, 150.0)
        self.assertIsNotNone(res.scalar_value)
        self.assertAlmostEqual(res.scalar_value, self.df["Risk_Score"].mean(), places=4)
        self.assertEqual(res.coverage_pct, 100.0)

    def test_02_group_comparison_execution(self):
        plan = self.planner.plan("Show average severity rating by plant zone")
        res = self.engine.execute_plan(plan)

        self.assertEqual(res.status, ResultValidationStatus.PASS)
        self.assertEqual(res.rows_returned, 4)  # 4 distinct plant zones
        self.assertIn("Plant_Zone", res.tabular_columns)
        self.assertIn("metric_val", res.tabular_columns)

        # Verify values match pandas groupby
        expected = self.df.groupby("Plant_Zone")["Severity_Rating"].mean().sort_values(ascending=False)
        top_group = res.data_sample[0]["Plant_Zone"]
        self.assertEqual(top_group, expected.index[0])
        self.assertAlmostEqual(res.data_sample[0]["metric_val"], expected.iloc[0], places=4)

    def test_03_ranking_top_n_execution(self):
        plan = self.planner.plan("Top 3 plant zones by exposure hours")
        res = self.engine.execute_plan(plan)

        self.assertEqual(res.status, ResultValidationStatus.PASS)
        self.assertEqual(res.rows_returned, 3)
        self.assertIn("rank_val", res.tabular_columns)

    def test_04_filtered_records_execution(self):
        plan = self.planner.plan("Show incidents in Workshop where Exposure_Hours <= 20")
        res = self.engine.execute_plan(plan)

        self.assertEqual(res.status, ResultValidationStatus.PASS)
        self.assertGreater(res.rows_returned, 0)
        for record in res.data_sample:
            self.assertEqual(record["Plant_Zone"], "Workshop")
            self.assertLessEqual(record["Exposure_Hours"], 20.0)

    def test_05_correlation_execution(self):
        plan = self.planner.plan("What is the correlation between Severity_Rating and Risk_Score?")
        res = self.engine.execute_plan(plan)

        self.assertEqual(res.status, ResultValidationStatus.PASS)
        self.assertIsNotNone(res.scalar_value)

        expected_corr = self.df["Severity_Rating"].corr(self.df["Risk_Score"])
        self.assertAlmostEqual(res.scalar_value, expected_corr, places=4)

    def test_06_time_series_trend_execution(self):
        plan = self.planner.plan("What is the monthly trend of risk score over time?")
        res = self.engine.execute_plan(plan)

        self.assertEqual(res.status, ResultValidationStatus.PASS)
        self.assertGreater(res.rows_returned, 0)
        self.assertIn("period", res.tabular_columns)
        self.assertIn("avg_val", res.tabular_columns)

    def test_07_distribution_analysis_execution(self):
        plan = self.planner.plan("Show the distribution of Risk_Score")
        res = self.engine.execute_plan(plan)

        self.assertEqual(res.status, ResultValidationStatus.PASS)
        data = res.data_sample[0]
        self.assertIn("mean", data)
        self.assertIn("median", data)
        self.assertIn("q25", data)
        self.assertIn("q75", data)
        self.assertIn("stddev", data)
        self.assertAlmostEqual(data["median"], self.df["Risk_Score"].median(), places=4)

    def test_08_outlier_analysis_execution(self):
        # Inject an extreme outlier
        outlier_df = self.df.copy()
        outlier_df.loc[0, "Exposure_Hours"] = 999.0
        engine = DuckDBAnalyticalEngine(df=outlier_df)

        plan = self.planner.plan("Are there outliers in Exposure_Hours?")
        res = engine.execute_plan(plan)

        self.assertEqual(res.status, ResultValidationStatus.PASS)
        self.assertGreaterEqual(res.rows_returned, 1)
        # Check that top outlier is the 999.0 record
        top_outlier = res.data_sample[0]
        self.assertEqual(top_outlier["Exposure_Hours"], 999.0)

    def test_09_security_and_read_only_guard(self):
        plan = self.planner.plan("What is the average risk score across the dataset?")
        # Tamper plan with malicious injection
        plan.metric = "Risk_Score; DROP TABLE dataset; --"
        res = self.engine.execute_plan(plan)

        # Either compile safely with quotes or block
        # With proper double quoting: "Risk_Score; DROP TABLE dataset; --" is treated as column name
        # If DuckDB reports column doesn't exist, it returns status ERROR, not crashing
        self.assertIn(res.status, [ResultValidationStatus.ERROR, ResultValidationStatus.BLOCKED])

    def test_10_special_characters_and_spaces_in_columns(self):
        custom_df = pd.DataFrame({
            "Plant Zone": ["Alpha", "Beta", "Alpha"],
            "Risk-Score (%)": [12.5, 45.0, 30.0],
            "Order #": [101, 102, 103]
        })
        custom_engine = DuckDBAnalyticalEngine(df=custom_df)
        custom_planner = UniversalQueryPlanner(df=custom_df)

        plan = custom_planner.plan("Show average Risk-Score (%) by Plant Zone")
        res = custom_engine.execute_plan(plan)

        self.assertEqual(res.status, ResultValidationStatus.PASS)
        self.assertEqual(res.rows_returned, 2)
        self.assertIn("Plant Zone", res.tabular_columns)


if __name__ == "__main__":
    unittest.main()
