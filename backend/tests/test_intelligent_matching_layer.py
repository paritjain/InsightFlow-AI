import unittest
import pandas as pd
import numpy as np

from engine.matcher import IntelligentFieldMatcher, DynamicAliasDictionary
from engine.semantic_resolver import UniversalSchemaAndSemanticResolver
from engine.query_planner import UniversalQueryPlanner
from engine.analytical_engine import UniversalAnalyticalChatEngine
from engine.contracts.chat_contracts import QueryIntent, ResultValidationStatus
from agents.graph import LangGraphDataAgent, session_manager


class TestIntelligentMatchingLayer(unittest.TestCase):
    """
    Comprehensive verification suite for Intelligent Semantic Matching,
    Dynamic Alias Mapping, Fuzzy Resolution, and Confidence Decisions.
    """

    def setUp(self):
        np.random.seed(42)
        n = 100

        # Multi-column industrial & enterprise dataset matching user examples
        self.df = pd.DataFrame({
            "Incident_ID": [f"INC-{1000+i}" for i in range(n)],
            "Department": np.random.choice(["Smelter", "Refinery", "Maintenance", "Logistics", "Safety"], n),
            "Deviation Category": np.random.choice(["Process Drift", "Equipment Malfunction", "Human Factor", "Environmental"], n),
            "Category": np.random.choice(["Cat A", "Cat B", "Cat C"], n),
            "Deviation Type": np.random.choice(["Minor", "Major", "Critical"], n),
            "Near Miss/Audit/Whistleblowing": np.random.choice(["Near Miss", "Internal Audit", "Whistleblower Alert", "Routine"], n),
            "Observation Date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "Performance Score": np.random.uniform(50.0, 100.0, n),
            "Deviation Count": np.random.randint(1, 25, n),
            "Risk Score": np.random.uniform(1.0, 10.0, n)
        })

        self.dictionary = [
            {"column_name": "Incident_ID", "semantic_role": "Identifier", "is_identifier": True, "business_meaning": "Unique tracking key for safety and operational incidents"},
            {"column_name": "Department", "semantic_role": "Dimension", "is_dimension": True, "business_meaning": "Plant department, division, or organizational operating unit"},
            {"column_name": "Deviation Category", "semantic_role": "Dimension", "is_dimension": True, "business_meaning": "Classification category of process and operational deviations"},
            {"column_name": "Category", "semantic_role": "Dimension", "is_dimension": True, "business_meaning": "General category classification bucket"},
            {"column_name": "Deviation Type", "semantic_role": "Dimension", "is_dimension": True, "business_meaning": "Severity grading of deviation instances"},
            {"column_name": "Near Miss/Audit/Whistleblowing", "semantic_role": "Dimension", "is_dimension": True, "business_meaning": "Safety near miss observations, whistleblowing reports, and compliance audit findings"},
            {"column_name": "Observation Date", "semantic_role": "Date/Time", "is_time": True, "business_meaning": "Date when observation or deviation was recorded"},
            {"column_name": "Performance Score", "semantic_role": "Measure", "is_measure": True, "semantic_subtype": "Rating", "business_meaning": "Operational performance score metric"},
            {"column_name": "Deviation Count", "semantic_role": "Measure", "is_measure": True, "semantic_subtype": "Count", "business_meaning": "Total count of observed deviations"},
            {"column_name": "Risk Score", "semantic_role": "Measure", "is_measure": True, "semantic_subtype": "Rating", "business_meaning": "Computed hazard risk score"}
        ]

        self.project_id = "test_matching_proj_01"
        session_manager.clear_session(self.project_id)

        self.engine = UniversalAnalyticalChatEngine(
            df=self.df,
            dictionary=self.dictionary,
            project_id=self.project_id
        )

    def test_01_exact_and_case_insensitive_matching(self):
        # Exact column name
        res1 = self.engine.process_query("What is the average Performance Score?")
        self.assertEqual(res1.status, ResultValidationStatus.PASS)
        self.assertEqual(res1.query_plan["metric"], "Performance Score")
        self.assertEqual(res1.query_plan["aggregation"], "MEAN")
        self.assertIsNotNone(res1.numerical_value)

        # Case-insensitive
        res2 = self.engine.process_query("calculate sum of deviation count")
        self.assertEqual(res2.status, ResultValidationStatus.PASS)
        self.assertEqual(res2.query_plan["metric"], "Deviation Count")
        self.assertEqual(res2.query_plan["aggregation"], "SUM")

    def test_02_abbreviations_and_synonyms(self):
        # Example 1: 'department wise' -> Department
        res_dept_wise = self.engine.process_query("department wise deviation count")
        self.assertEqual(res_dept_wise.status, ResultValidationStatus.PASS)
        self.assertEqual(res_dept_wise.query_plan["dimension"], "Department")
        self.assertEqual(res_dept_wise.query_plan["metric"], "Deviation Count")

        # Example 2: 'dept analysis' -> Department
        res_dept_analysis = self.engine.process_query("dept analysis")
        self.assertEqual(res_dept_analysis.status, ResultValidationStatus.PASS)
        self.assertEqual(res_dept_analysis.query_plan["dimension"], "Department")

        # Example 3: 'near miss' -> Near Miss/Audit/Whistleblowing
        res_near_miss = self.engine.process_query("breakdown by near miss")
        self.assertEqual(res_near_miss.status, ResultValidationStatus.PASS)
        self.assertEqual(res_near_miss.query_plan["dimension"], "Near Miss/Audit/Whistleblowing")

    def test_03_typographical_error_correction(self):
        # Typo: 'deprtment' -> Department
        res_typo_dept = self.engine.process_query("show deprtment breakdown")
        self.assertEqual(res_typo_dept.status, ResultValidationStatus.PASS)
        self.assertEqual(res_typo_dept.query_plan["dimension"], "Department")

        # Typo: 'deviaton count' -> Deviation Count
        res_typo_dev = self.engine.process_query("total deviaton count")
        self.assertEqual(res_typo_dev.status, ResultValidationStatus.PASS)
        self.assertEqual(res_typo_dev.query_plan["metric"], "Deviation Count")

    def test_04_did_you_mean_suggestion_and_confirmation(self):
        # Ambiguous query: 'deviation' matches Deviation Category, Deviation Type, Deviation Count
        matcher = self.engine.resolver.matcher
        matches = matcher.find_matches("deviation", top_k=4)
        self.assertGreaterEqual(len(matches), 2)
        dev_cols = [m.column for m in matches]
        self.assertTrue(any("Deviation" in c for c in dev_cols))

        # Query requiring confirmation: 'deviation category wise' has candidates Deviation Category (100%), Category (95%)
        # When confirmed_field is supplied by user action, it executes cleanly
        res_confirmed = self.engine.process_query(
            "Show deviation category wise analysis",
            confirmed_field="Deviation Category"
        )
        self.assertEqual(res_confirmed.status, ResultValidationStatus.PASS)
        self.assertEqual(res_confirmed.query_plan["dimension"], "Deviation Category")
        self.assertEqual(res_confirmed.confirmed_field, "Deviation Category")
        self.assertEqual(res_confirmed.suggestion_mode, "exact")

    def test_05_multi_turn_context_carryover_and_slicing(self):
        # Turn 1: Show department-wise deviation
        res1 = self.engine.process_query("Show department-wise deviation count")
        self.assertEqual(res1.status, ResultValidationStatus.PASS)
        self.assertEqual(res1.query_plan["dimension"], "Department")
        self.assertEqual(res1.query_plan["metric"], "Deviation Count")

        # Context update
        session_manager.update_context(self.project_id, {
            "last_dimension": res1.query_plan["dimension"],
            "last_metric": res1.query_plan["metric"],
            "last_intent": res1.query_plan["intent"]
        })

        # Turn 2: Follow-up: 'Now show it month-wise.'
        ctx = session_manager.get_context(self.project_id)
        res2 = self.engine.process_query("Now show it month-wise.", context=ctx)
        self.assertEqual(res2.status, ResultValidationStatus.PASS)
        # Dimension is updated to temporal/month, but metric Deviation Count is retained!
        self.assertEqual(res2.query_plan["time_grain"], "month")
        self.assertEqual(res2.query_plan["metric"], "Deviation Count")

    def test_06_low_confidence_clarification_safety(self):
        # Gibberish / non-existent field
        matcher = self.engine.resolver.matcher
        dec = matcher.evaluate_decision("completely_unknown_metric_987654")
        self.assertEqual(dec["decision"], "LOW_CONFIDENCE")
        self.assertEqual(dec["suggestion_mode"], "clarification_required")
        self.assertIsNone(dec["resolved_field"])
        self.assertIn("available columns", dec["clarification_prompt"].lower())

    def test_07_api_level_confirmed_field_integration(self):
        # Test full LangGraphDataAgent dispatcher with confirmed_field
        res = LangGraphDataAgent.process_message(
            project_id=self.project_id,
            df=self.df,
            sem_layer={},
            eda_results={},
            insights=[],
            user_message="breakdown by deviation category",
            dictionary=self.dictionary,
            confirmed_field="Deviation Category"
        )
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["query_plan"]["dimension"], "Deviation Category")
        self.assertIsNotNone(res.get("chart_spec"))


if __name__ == "__main__":
    unittest.main()
