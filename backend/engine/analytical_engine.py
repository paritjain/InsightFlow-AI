"""
Universal Governed Analytical Chat Engine.

Coordinates schema-semantic resolution, typed query planning, DuckDB in-memory execution,
result validation, and zero-hallucination explanation with multi-turn conversational memory.
"""

import time
import uuid
import re
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from engine.contracts.chat_contracts import (
    QueryPlan,
    QueryIntent,
    ExecutionResult,
    ResultValidationStatus,
    ChatResponseContract,
    ColumnFilter
)
from engine.contracts.dataset_contract import DatasetContract
from engine.semantic_resolver import UniversalSchemaAndSemanticResolver
from engine.query_planner import UniversalQueryPlanner
from engine.duckdb_engine import DuckDBAnalyticalEngine
from engine.result_validator import UniversalResultValidator
from engine.grounded_explainer import GroundedAnalyticalExplainer


class UniversalAnalyticalChatEngine:
    """
    End-to-End Governed Analytical Engine for Natural Language Chat.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        contract: Optional[DatasetContract] = None,
        dictionary: Optional[List[Dict[str, Any]]] = None,
        sem_layer_dict: Optional[Dict[str, Any]] = None,
        project_id: str = "project_active",
        dataset_id: str = "dataset_active"
    ):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.df = df
        self.contract = contract
        self.dictionary = dictionary or []
        self.sem_layer_dict = sem_layer_dict or {}

        # 1. Initialize Semantic Resolver
        self.resolver = UniversalSchemaAndSemanticResolver(
            contract=self.contract,
            dictionary=self.dictionary,
            df=self.df,
            sem_layer_dict=self.sem_layer_dict
        )

        # 2. Initialize Query Planner
        self.planner = UniversalQueryPlanner(
            resolver=self.resolver,
            dataset_id=self.dataset_id
        )

        # 3. Initialize Execution Engine
        self.execution_engine = DuckDBAnalyticalEngine(
            df=self.df,
            table_name="dataset"
        )

        # 4. Initialize Validator & Explainer
        self.validator = UniversalResultValidator(
            col_metadata=self.resolver.col_metadata
        )
        self.explainer = GroundedAnalyticalExplainer(
            col_metadata=self.resolver.col_metadata
        )

    def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        df: Optional[pd.DataFrame] = None,
        confirmed_field: Optional[str] = None
    ) -> ChatResponseContract:
        """
        Executes a single conversational query end-to-end with validation,
        grounding, and multi-turn context carryover.
        """
        if df is not None:
            self.df = df
            self.execution_engine.register_dataframe(df)

        # Multi-turn Context Enrichment
        enriched_query = self._apply_multi_turn_context(query, context)

        # Check for confirmed field in request or conversational context
        effective_confirmed_field = confirmed_field
        if not effective_confirmed_field and context:
            effective_confirmed_field = context.get("last_confirmed_field")
            # Also check if query matches any previously confirmed term
            confirmed_map = context.get("confirmed_fields", {})
            for term, col in confirmed_map.items():
                if term in query.lower():
                    effective_confirmed_field = col
                    break

        # Phase 4: Plan Query
        plan = self.planner.plan(enriched_query, confirmed_field=effective_confirmed_field)

        # Fallback to context metric/dimension if plan missed it due to conversational shorthand
        if context:
            if not plan.metric and context.get("last_metric"):
                plan.metric = context["last_metric"]
                if plan.intent in [QueryIntent.FILTERED_RECORDS, QueryIntent.DATASET_SUMMARY]:
                    plan.intent = context.get("last_intent", QueryIntent.GLOBAL_AGGREGATION)
            if not plan.dimension and context.get("last_dimension") and plan.intent == QueryIntent.GROUP_COMPARISON:
                plan.dimension = context["last_dimension"]

        # If confirmation or clarification is required, do not execute raw DuckDB query blindly
        if plan.clarification_needed:
            ambig_result = ExecutionResult(
                plan_id=plan.plan_id,
                status=ResultValidationStatus.AMBIGUOUS,
                rows_returned=0,
                records_evaluated=len(self.df) if self.df is not None else 0,
                total_dataset_rows=len(self.df) if self.df is not None else 0
            )
            return self.explainer.explain(
                plan=plan,
                result=ambig_result,
                project_id=self.project_id,
                dataset_id=self.dataset_id
            )

        # Phase 5: Execute via DuckDB
        raw_result = self.execution_engine.execute_plan(plan, df=self.df)

        # Phase 6A: Validate Execution Result
        validated_result = self.validator.validate(plan, raw_result, df=self.df)

        # Phase 6B: Grounded Zero-Hallucination Natural Language Explanation
        response = self.explainer.explain(
            plan=plan,
            result=validated_result,
            project_id=self.project_id,
            dataset_id=self.dataset_id
        )

        return response

    def _apply_multi_turn_context(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Resolves conversational pronouns and context carryover across turns."""
        if not context or not query:
            return query

        q_low = query.lower().strip()
        last_metric = context.get("last_metric")

        # Handle pronoun resolution: "in it", "for it", "its distribution", "its trend"
        if last_metric and any(p in q_low for p in ["in it", "for it", "of it", "its ", "about it"]):
            query = re.sub(r'\b(in|for|of|about)\s+it\b', rf'\1 {last_metric}', query, flags=re.IGNORECASE)
            query = re.sub(r'\bits\b', f"{last_metric}'s", query, flags=re.IGNORECASE)
        elif last_metric and any(w in q_low for w in ["now", "what about", "how about", "also show"]):
            # Slicing shorthand e.g. "Now show month-wise" -> "Now show month-wise of Deviation"
            if not any(m_word in q_low for m_word in ["sum", "count", "average", "mean", "rate", "score", "metric"]):
                query = f"{query} of {last_metric}"

        return query
