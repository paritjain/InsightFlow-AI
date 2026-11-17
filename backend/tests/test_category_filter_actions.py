import os
import unittest
import pandas as pd

from engine.semantic_resolver import UniversalSchemaAndSemanticResolver
from engine.query_planner import UniversalQueryPlanner
from engine.duckdb_engine import DuckDBAnalyticalEngine
from engine.grounded_explainer import GroundedAnalyticalExplainer
from engine.contracts.chat_contracts import QueryIntent, FilterOperator, ResultValidationStatus


class TestCategoryFilterActions(unittest.TestCase):
    def setUp(self):
        # Create dataset simulating the user's near-miss data with Category having numeric and string values
        self.df = pd.DataFrame({
            "Sl.No": [1, 2, 3, 4, 5, 6, 7, 8],
            "Category": ["0-Near Miss", "3", "3", "3", "4", "2", "3", "nan"],
            "Department": ["Rodding", "PRO", "Rodding", "PRO", "Smelter", "CH-1", "PRO", "Security"],
            "Location": ["Zone 1", "Zone 2", "Zone 1", "Zone 3", "Zone 4", "Zone 1", "Zone 2", "Zone 5"],
            "Corrective_Actions": ["Fix guard", "Inspect pipe", "Clean area", "Replace valve", "Audit check", "Train team", "Review SOP", "Report"]
        })
        self.resolver = UniversalSchemaAndSemanticResolver(df=self.df)
        self.planner = UniversalQueryPlanner(resolver=self.resolver, dataset_id="test_cat_proj")
        self.engine = DuckDBAnalyticalEngine(df=self.df, table_name="dataset")
        self.explainer = GroundedAnalyticalExplainer(col_metadata=self.resolver.col_metadata)

    def test_category_3_count_extraction(self):
        plan = self.planner.plan("category 3 count")
        self.assertEqual(plan.intent, QueryIntent.GLOBAL_AGGREGATION)
        self.assertEqual(len(plan.filters), 1)
        self.assertEqual(plan.filters[0].column, "Category")
        self.assertEqual(str(plan.filters[0].value), "3")

        res = self.engine.execute_plan(plan)
        self.assertEqual(res.status, ResultValidationStatus.PASS)
        self.assertEqual(res.scalar_value, 4)

        ans = self.explainer.explain(plan, res)
        self.assertIn("Category = '3'", ans.answer)
        self.assertIn("4", ans.answer)
        self.assertIn("SELECT COUNT(*)", ans.calculation_formula)

    def test_count_of_category_3(self):
        plan = self.planner.plan("count of category 3")
        self.assertEqual(plan.intent, QueryIntent.GLOBAL_AGGREGATION)
        self.assertEqual(len(plan.filters), 1)
        self.assertEqual(plan.filters[0].column, "Category")

        res = self.engine.execute_plan(plan)
        self.assertEqual(res.scalar_value, 4)

    def test_category_3_filtered_records(self):
        for query in ["category 3", "show category 3", "filter category 3", "category is 3", "category = 3", "category: 3", "category '3'"]:
            plan = self.planner.plan(query)
            self.assertEqual(plan.intent, QueryIntent.FILTERED_RECORDS, f"Failed for query: {query}")
            self.assertEqual(len(plan.filters), 1, f"Failed for query: {query}")
            self.assertEqual(plan.filters[0].column, "Category")

            res = self.engine.execute_plan(plan)
            self.assertEqual(res.status, ResultValidationStatus.PASS)
            self.assertEqual(res.rows_returned, 4)

            ans = self.explainer.explain(plan, res)
            self.assertIn("Category = '3'", ans.answer)

    def test_category_3_by_department(self):
        plan = self.planner.plan("category 3 by department")
        self.assertEqual(plan.intent, QueryIntent.GROUP_COMPARISON)
        self.assertEqual(plan.dimension, "Department")
        self.assertEqual(len(plan.filters), 1)
        self.assertEqual(plan.filters[0].column, "Category")

        res = self.engine.execute_plan(plan)
        self.assertEqual(res.status, ResultValidationStatus.PASS)
        # PRO has 3 cases in category 3, Rodding has 1
        self.assertEqual(res.data_sample[0]["Department"], "PRO")
        self.assertEqual(res.data_sample[0]["count"], 3)

    def test_top_3_categories_not_filtered(self):
        plan = self.planner.plan("top 3 categories")
        self.assertEqual(plan.intent, QueryIntent.RANKING_TOP_N)
        self.assertEqual(len(plan.filters), 0)

    def test_duckdb_conversion_protection(self):
        # Mixed column where DuckDB would crash if comparing VARCHAR = integer 3
        filters = [self.planner.resolver.extract_filters_from_query("category = 3")[0]]
        where_clause, params = self.engine._build_where_clause(filters)
        self.assertIn('"Category" = ?', where_clause)
        self.assertEqual(params[0], "3")  # Parameter coerced to string matching column dtype


if __name__ == "__main__":
    unittest.main()
