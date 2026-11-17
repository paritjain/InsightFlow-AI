import unittest
import pandas as pd
import numpy as np

from engine.query_planner import UniversalQueryPlanner, parse_query_plan
from engine.contracts.chat_contracts import QueryIntent, FilterOperator


class TestUniversalQueryPlanner(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 60
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

        self.planner = UniversalQueryPlanner(
            df=self.df,
            sem_layer_dict=self.sem_dict
        )

    def test_01_global_aggregation_planning(self):
        query = "What is the average risk score across the dataset?"
        plan = self.planner.plan(query)

        self.assertEqual(plan.intent, QueryIntent.GLOBAL_AGGREGATION)
        self.assertEqual(plan.metric, "Risk_Score")
        self.assertIn("MEAN", plan.aggregations)
        self.assertIsNone(plan.dimension)

    def test_02_group_comparison_planning(self):
        query = "Show average severity rating by plant zone"
        plan = self.planner.plan(query)

        self.assertEqual(plan.intent, QueryIntent.GROUP_COMPARISON)
        self.assertEqual(plan.metric, "Severity_Rating")
        self.assertEqual(plan.dimension, "Plant_Zone")

    def test_03_ranking_top_n_planning(self):
        query = "Top 5 plant zones by exposure hours"
        plan = self.planner.plan(query)

        self.assertEqual(plan.intent, QueryIntent.RANKING_TOP_N)
        self.assertEqual(plan.metric, "Exposure_Hours")
        self.assertEqual(plan.dimension, "Plant_Zone")
        self.assertEqual(plan.limit, 5)
        self.assertEqual(plan.sort, "descending")

    def test_04_time_series_trend_planning(self):
        query = "What is the trend of risk score over time?"
        plan = self.planner.plan(query)

        self.assertEqual(plan.intent, QueryIntent.TIME_SERIES_TREND)
        self.assertEqual(plan.metric, "Risk_Score")
        self.assertEqual(plan.time_column, "Incident_Date")

    def test_05_correlation_analysis_planning(self):
        query = "What is the correlation between Severity_Rating and Exposure_Hours?"
        plan = self.planner.plan(query)

        self.assertEqual(plan.intent, QueryIntent.CORRELATION_ANALYSIS)
        self.assertIn("Severity_Rating", plan.variables)
        self.assertIn("Exposure_Hours", plan.variables)
        self.assertEqual(plan.methods, ["pearson"])

    def test_06_filtered_records_planning(self):
        query = "Show incidents in Workshop where Exposure_Hours <= 20"
        plan = self.planner.plan(query)

        self.assertEqual(plan.intent, QueryIntent.FILTERED_RECORDS)
        self.assertEqual(len(plan.filters), 2)
        cols = {f.column: f.value for f in plan.filters}
        self.assertEqual(cols.get("Plant_Zone"), "Workshop")
        self.assertEqual(cols.get("Exposure_Hours"), 20)

    def test_07_row_lookup_planning(self):
        query = "Lookup incident OBS_105"
        plan = self.planner.plan(query)

        self.assertEqual(plan.intent, QueryIntent.ROW_LOOKUP)

    def test_08_outliers_and_distribution_planning(self):
        # Outlier query
        q_out = "Are there outliers in Exposure_Hours?"
        plan_out = self.planner.plan(q_out)
        self.assertEqual(plan_out.intent, QueryIntent.OUTLIER_ANALYSIS)
        self.assertEqual(plan_out.metric, "Exposure_Hours")

        # Distribution query
        q_dist = "Show the distribution of Risk_Score"
        plan_dist = self.planner.plan(q_dist)
        self.assertEqual(plan_dist.intent, QueryIntent.DISTRIBUTION_ANALYSIS)
        self.assertEqual(plan_dist.metric, "Risk_Score")

    def test_09_data_quality_and_dataset_summary(self):
        # Data Quality
        q_dq = "Which columns have missing values in this dataset?"
        plan_dq = self.planner.plan(q_dq)
        self.assertEqual(plan_dq.intent, QueryIntent.DATA_QUALITY)

        # Dataset Summary
        q_sum = "Give me an overview and summarize the dataset"
        plan_sum = self.planner.plan(q_sum)
        self.assertEqual(plan_sum.intent, QueryIntent.DATASET_SUMMARY)

    def test_10_ambiguity_clarification_prompting(self):
        # 'Rating' matches both Severity_Rating and Likelihood_Rating
        query = "What is the average rating?"
        plan = self.planner.plan(query)

        self.assertEqual(plan.intent, QueryIntent.AMBIGUOUS)
        self.assertTrue(plan.clarification_needed)
        self.assertIsNotNone(plan.clarification_prompt)
        self.assertIn("Severity_Rating", plan.clarification_prompt)
        self.assertIn("Likelihood_Rating", plan.clarification_prompt)

    def test_11_backward_compatibility_dict_access(self):
        query = "What is the total baseline volume and mean value for Exposure_Hours across the dataset?"
        plan = self.planner.plan(query)

        # Dictionary bracket access compatibility
        self.assertEqual(plan["intent"], "global_aggregation")
        self.assertIn("Exposure_Hours (SUM)", plan["Metrics"])
        self.assertIn("Exposure_Hours (MEAN)", plan["Metrics"])
        self.assertIsNone(plan["Dimension"])

        # Legacy function entry point
        legacy_plan = parse_query_plan(self.df, query)
        self.assertEqual(legacy_plan["intent"], "global_aggregation")

    def test_12_grouping_words_as_for_in_by(self):
        # Test 'as', 'for', 'in', 'per', 'grouped as', 'group for'
        for phrase in [
            "severity rating as plant zone",
            "severity rating for plant zone",
            "severity rating in plant zone",
            "severity rating per plant zone",
            "severity rating grouped as plant zone",
            "severity rating group for plant zone",
            "severity rating by plant zone"
        ]:
            plan = self.planner.plan(phrase)
            self.assertEqual(plan.intent, QueryIntent.GROUP_COMPARISON, f"Failed for phrase: {phrase}")
            self.assertEqual(plan.dimension, "Plant_Zone", f"Failed dimension for phrase: {phrase}")
            self.assertEqual(plan.metric, "Severity_Rating", f"Failed metric for phrase: {phrase}")

    def test_13_direct_dimension_mention_grouping(self):
        # Direct mention of dimension with categorical/count query
        plan1 = self.planner.plan("incidents plant zone")
        self.assertEqual(plan1.intent, QueryIntent.GROUP_COMPARISON)
        self.assertEqual(plan1.dimension, "Plant_Zone")

        plan2 = self.planner.plan("workshop incidents plant zone")
        self.assertEqual(plan2.intent, QueryIntent.GROUP_COMPARISON)
        self.assertEqual(plan2.dimension, "Plant_Zone")
        self.assertTrue(any(f.column == "Plant_Zone" and f.value == "Workshop" for f in plan2.filters))


if __name__ == "__main__":
    unittest.main()

