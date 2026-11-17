import unittest
import pandas as pd
import numpy as np

from engine.query_planner import UniversalQueryPlanner
from engine.semantic_resolver import UniversalSchemaAndSemanticResolver, filter_prefixed_candidates
from engine.contracts.chat_contracts import QueryIntent


class TestAmbiguityDeduplicationHardening(unittest.TestCase):
    """
    Validates fixes for duplicate/whitespace column ambiguities,
    table-prefixed column disambiguation, and edge preposition clause handling.
    """

    def test_01_filter_prefixed_candidates(self):
        # 1. Base column + table prefixed column -> choose base column
        cands = ["Location/ Area", "24-25_Location/ Area"]
        filtered = filter_prefixed_candidates(cands, "location has")
        self.assertEqual(filtered, ["Location/ Area"])

        # 2. When query explicitly includes the prefix -> keep both / allow prefix
        filtered_explicit = filter_prefixed_candidates(cands, "24-25 location")
        self.assertIn("24-25_Location/ Area", filtered_explicit)

        # 3. Two genuinely different columns -> preserved
        genuine = ["Severity_Rating", "Likelihood_Rating"]
        filtered_genuine = filter_prefixed_candidates(genuine, "average rating")
        self.assertEqual(filtered_genuine, ["Severity_Rating", "Likelihood_Rating"])

    def test_02_whitespace_variant_column_deduplication(self):
        # Dataset containing columns with varying internal spaces
        df = pd.DataFrame({
            "Description of Incident": ["A", "B", "C"],
            "Description of  Incident": ["A", "B", "C"],
            "Status": ["Open", "Closed", "Open"]
        })
        sem = {
            "Description of Incident": {"semantic_role": "Dimension", "is_dimension": True},
            "Description of  Incident": {"semantic_role": "Dimension", "is_dimension": True},
            "Status": {"semantic_role": "Dimension", "is_dimension": True}
        }
        planner = UniversalQueryPlanner(df=df, sem_layer_dict=sem)

        # Query 1: Exact column name
        plan1 = planner.plan("Description of Incident")
        self.assertNotEqual(plan1.intent, QueryIntent.AMBIGUOUS)
        self.assertFalse(plan1.clarification_needed)

        # Query 2: Grouping query with preposition
        plan2 = planner.plan("Description of Incident by closure status")
        self.assertNotEqual(plan2.intent, QueryIntent.AMBIGUOUS)
        self.assertFalse(plan2.clarification_needed)
        self.assertEqual(plan2.dimension, "Status")

        # Query 3: Partial substring query should not ask to choose between identical names
        plan3 = planner.plan("Description of")
        if plan3.clarification_needed:
            # If clarification is asked, options must never be identical
            alts = plan3.clarification_prompt.split(":")[-1]
            self.assertNotIn("'Description of Incident', 'Description of Incident'", alts)

    def test_03_word_boundary_versus_subword(self):
        # 'event' must match 'Event ID' and NOT collision with 'Preventive'
        df = pd.DataFrame({
            "Event ID": [101, 102],
            "Corrective Actions and Preventive Actions Planned": ["Fix wire", "Repair pipe"]
        })
        resolver = UniversalSchemaAndSemanticResolver(df=df)
        res = resolver.resolve_term("event")
        self.assertEqual(res.resolved_column, "Event ID")

    def test_04_genuine_ambiguity_clarification_prompt(self):
        # Genuine ambiguity between two distinct ratings
        df = pd.DataFrame({
            "Severity_Rating": [1, 2, 3],
            "Likelihood_Rating": [4, 5, 6]
        })
        sem = {
            "Severity_Rating": {"semantic_role": "Measure", "is_measure": True},
            "Likelihood_Rating": {"semantic_role": "Measure", "is_measure": True}
        }
        planner = UniversalQueryPlanner(df=df, sem_layer_dict=sem)
        plan = planner.plan("What is the average rating?")

        self.assertEqual(plan.intent, QueryIntent.AMBIGUOUS)
        self.assertTrue(plan.clarification_needed)
        self.assertIn("Severity_Rating", plan.clarification_prompt)
        self.assertIn("Likelihood_Rating", plan.clarification_prompt)


if __name__ == "__main__":
    unittest.main()
