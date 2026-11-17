import unittest
import pandas as pd
import numpy as np

from services.statistical_analysis import compute_pca, compute_advanced_outliers, compute_bivariate_hypothesis_tests
from agents.tools import DatasetQueryTool, EDATool, ChartGenerationTool, InsightTool
from agents.graph import LangGraphDataAgent, session_manager
from engine.eda import run_automated_eda


class TestAgentAndAdvancedStats(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            "revenue": np.random.uniform(100, 1000, n),
            "cost": np.random.uniform(50, 500, n),
            "margin": np.random.uniform(10, 500, n),
            "units": np.random.randint(1, 50, n),
            "region": np.random.choice(["North", "South", "East", "West"], n),
            "category": np.random.choice(["Electronics", "Furniture", "Clothing"], n)
        })
        self.sem_layer = {
            "revenue": {"is_measure": True, "is_dimension": False, "is_identifier": False},
            "cost": {"is_measure": True, "is_dimension": False, "is_identifier": False},
            "margin": {"is_measure": True, "is_dimension": False, "is_identifier": False},
            "units": {"is_measure": True, "is_dimension": False, "is_identifier": False},
            "region": {"is_measure": False, "is_dimension": True, "is_identifier": False},
            "category": {"is_measure": False, "is_dimension": True, "is_identifier": False}
        }
        self.eda_results = run_automated_eda(self.df)

    def test_01_pca_computation(self):
        res = compute_pca(self.df)
        self.assertTrue(res["is_valid"])
        self.assertGreaterEqual(res["components_count"], 2)
        self.assertIn("explained_variance_ratio", res)
        self.assertIn("cumulative_variance_ratio", res)
        self.assertIn("loadings", res)
        self.assertIn("projections", res)
        self.assertIn("interpretation", res)
        self.assertGreater(len(res["projections"]), 0)

    def test_02_advanced_outliers(self):
        res = compute_advanced_outliers(self.df)
        self.assertIn("columns", res)
        self.assertIn("revenue", res["columns"])
        self.assertIn("iqr_outliers_count", res["columns"]["revenue"])
        self.assertIn("zscore_outliers_count", res["columns"]["revenue"])

    def test_03_hypothesis_testing(self):
        tests = compute_bivariate_hypothesis_tests(self.df, self.sem_layer)
        self.assertIsInstance(tests, list)
        if tests:
            t = tests[0]
            self.assertIn("anova_f_stat", t)
            self.assertIn("anova_p_value", t)
            self.assertIn("kruskal_h_stat", t)
            self.assertIn("interpretation", t)

    def test_04_dataset_query_tool(self):
        res = DatasetQueryTool.run(self.df, self.sem_layer, "What is the average revenue?")
        self.assertTrue(res["success"])
        self.assertIn("answer", res)
        self.assertIn("calculation_formula", res)

    def test_05_eda_tool(self):
        # Missing values
        missing_res = EDATool.get_missing_values(self.df, self.eda_results)
        self.assertEqual(missing_res["clean_columns_count"], len(self.df.columns))

        # Correlations
        corr_res = EDATool.get_correlations(self.eda_results)
        self.assertEqual(corr_res["tool"], "EDATool")

        # PCA
        pca_res = EDATool.get_pca(self.eda_results)
        self.assertTrue(pca_res["pca_analysis"]["is_valid"])

    def test_06_chart_generation_tool(self):
        res = ChartGenerationTool.run(self.df, self.sem_layer, "Generate a bar chart for revenue by region", chart_type="BAR")
        self.assertTrue(res["success"])
        self.assertIn("chart_spec", res)
        self.assertEqual(res["chart_spec"]["chart_type"], "BAR")

    def test_07_agent_workflow_and_session_isolation(self):
        proj_a = "proj-test-a"
        proj_b = "proj-test-b"

        # Ask in Project A
        res_a = LangGraphDataAgent.process_message(
            project_id=proj_a,
            df=self.df,
            sem_layer=self.sem_layer,
            eda_results=self.eda_results,
            insights=[],
            user_message="What is the average revenue?"
        )
        self.assertIn("answer", res_a)
        hist_a = session_manager.get_history(proj_a)
        self.assertEqual(len(hist_a), 2)  # 1 user + 1 assistant

        # Check Project B is isolated
        hist_b = session_manager.get_history(proj_b)
        self.assertEqual(len(hist_b), 0)

        # Clear session
        session_manager.clear_session(proj_a)
        self.assertEqual(len(session_manager.get_history(proj_a)), 0)


if __name__ == "__main__":
    unittest.main()
