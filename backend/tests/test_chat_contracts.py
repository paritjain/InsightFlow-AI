import unittest
from engine.contracts.chat_contracts import (
    QueryIntent,
    FilterOperator,
    ColumnFilter,
    AggregationFunction,
    ResolutionStatus,
    ResolvedEntity,
    ResultValidationStatus,
    QueryPlan,
    ExecutionResult,
    QueryTrace,
    ChatResponseContract
)


class TestChatContracts(unittest.TestCase):

    def test_01_query_intent_enum_and_legacy_equality(self):
        # Case-insensitive equality
        self.assertEqual(QueryIntent.GLOBAL_AGGREGATION, "global_aggregation")
        self.assertEqual(QueryIntent.GLOBAL_AGGREGATION, "GLOBAL_AGGREGATION")
        self.assertEqual(QueryIntent.GROUP_COMPARISON, "group_comparison")
        self.assertEqual(QueryIntent.TIME_SERIES_TREND, "time_series_trend")
        self.assertEqual(QueryIntent.CORRELATION_ANALYSIS, "correlation_analysis")
        
        # Legacy aliases
        self.assertEqual(QueryIntent.GLOBAL_AGGREGATION, "aggregate")
        self.assertEqual(QueryIntent.GROUP_COMPARISON, "ranking")
        self.assertEqual(QueryIntent.TIME_SERIES_TREND, "growth")
        self.assertEqual(QueryIntent.CORRELATION_ANALYSIS, "relationship_single")
        self.assertEqual(QueryIntent.CORRELATION_ANALYSIS, "correlation_matrix")

    def test_02_column_filter_creation_and_dict_access(self):
        f = ColumnFilter(
            column="Revenue",
            operator=FilterOperator.GREATER_THAN,
            value=1000.0
        )
        self.assertEqual(f.column, "Revenue")
        self.assertEqual(f["column"], "Revenue")
        self.assertEqual(f.operator, FilterOperator.GREATER_THAN)
        self.assertEqual(f["operator"], FilterOperator.GREATER_THAN)
        self.assertEqual(f.value, 1000.0)
        self.assertFalse(f.is_negated)

        # Mutation via dict interface
        f["is_negated"] = True
        self.assertTrue(f.is_negated)

    def test_03_query_plan_contract(self):
        plan = QueryPlan(
            dataset_id="proj-sales-q3",
            intent=QueryIntent.GLOBAL_AGGREGATION,
            target_columns=["Revenue"],
            metric="Revenue",
            aggregations=["SUM", "MEAN"],
            filters=[
                ColumnFilter(column="Region", operator=FilterOperator.EQUALS, value="North")
            ],
            sort="DESC",
            limit=10,
            query_text="What is total and average revenue in North region?"
        )

        self.assertEqual(plan.dataset_id, "proj-sales-q3")
        self.assertEqual(plan.intent, QueryIntent.GLOBAL_AGGREGATION)
        self.assertEqual(plan["intent"], "global_aggregation")
        
        # Legacy uppercase property aliases
        self.assertEqual(plan.Intent, QueryIntent.GLOBAL_AGGREGATION)
        self.assertEqual(plan.Metric, "Revenue")
        self.assertEqual(plan.Aggregations, ["SUM", "MEAN"])
        self.assertEqual(plan.Metrics, ["Revenue (SUM)", "Revenue (MEAN)"])

        # Dict serialization
        d = plan.to_dict()
        self.assertIn("plan_id", d)
        self.assertIn("dataset_id", d)
        self.assertIn("intent", d)
        self.assertEqual(len(d["filters"]), 1)

    def test_04_execution_result_and_trace(self):
        exec_res = ExecutionResult(
            plan_id="plan-1234",
            engine_used="DuckDB",
            status=ResultValidationStatus.PASS,
            execution_time_ms=1.45,
            rows_returned=4,
            records_evaluated=1200,
            total_dataset_rows=1200,
            data_sample=[{"Region": "North", "Revenue": 45000}],
            tabular_columns=["Region", "Revenue"]
        )

        self.assertEqual(exec_res.engine_used, "DuckDB")
        self.assertEqual(exec_res.status, ResultValidationStatus.PASS)
        self.assertEqual(exec_res["rows_returned"], 4)

        trace = QueryTrace(
            project_id="proj-sales-q3",
            user_query="What is total revenue by region?"
        )
        trace.record_stage("intent_classification", "PASS", duration_ms=0.5, diagnostics={"intent": "GROUP_COMPARISON"})
        trace.record_stage("duckdb_execution", "PASS", duration_ms=1.2)
        
        self.assertIn("intent_classification", trace.stages)
        self.assertIn("duckdb_execution", trace.stages)
        self.assertEqual(trace.stages["intent_classification"]["status"], "PASS")

    def test_05_chat_response_contract(self):
        resp = ChatResponseContract(
            project_id="proj-demo-1",
            dataset_id="proj-demo-1",
            question="Show average cost",
            query="Show average cost",
            intent=QueryIntent.GLOBAL_AGGREGATION,
            answer="The average cost is $42.50 across 1,000 valid records.",
            numerical_value=42.50,
            calculation_formula="AVG(Cost)",
            evidence=[{"metric": "Average Cost", "value": "$42.50"}]
        )

        self.assertEqual(resp.question, "Show average cost")
        self.assertEqual(resp["answer"], "The average cost is $42.50 across 1,000 valid records.")
        self.assertEqual(resp["numerical_value"], 42.50)
        self.assertEqual(resp.status, ResultValidationStatus.PASS)


if __name__ == "__main__":
    unittest.main()
