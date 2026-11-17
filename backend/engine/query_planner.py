"""
Universal Query Intelligence & Structured Query Planner.

Replaces brittle ad-hoc regex heuristics with typed, schema-grounded,
universal analytical intent classification and structured QueryPlan compilation.
"""

import re
import uuid
from typing import Dict, List, Optional, Tuple, Any, Set
import pandas as pd

from engine.contracts.chat_contracts import (
    QueryIntent,
    FilterOperator,
    ColumnFilter,
    AggregationFunction,
    ResolutionStatus,
    ResolvedEntity,
    QueryPlan
)
from engine.contracts.dataset_contract import DatasetContract
from engine.semantic_resolver import UniversalSchemaAndSemanticResolver, filter_prefixed_candidates


class UniversalQueryPlanner:
    """
    Universal Analytical Query Planner.
    
    Transforms natural language questions into deterministic, typed QueryPlan contracts
    grounded in schema semantics, mathematical roles, and categorical cell values.
    """

    def __init__(
        self,
        resolver: Optional[UniversalSchemaAndSemanticResolver] = None,
        contract: Optional[DatasetContract] = None,
        df: Optional[pd.DataFrame] = None,
        dictionary: Optional[List[Dict[str, Any]]] = None,
        sem_layer_dict: Optional[Dict[str, Any]] = None,
        dataset_id: str = "dataset_active"
    ):
        if resolver is not None:
            self.resolver = resolver
        else:
            self.resolver = UniversalSchemaAndSemanticResolver(
                contract=contract,
                dictionary=dictionary,
                df=df,
                sem_layer_dict=sem_layer_dict
            )
        self.dataset_id = dataset_id

    def plan(self, query: str, confirmed_field: Optional[str] = None) -> QueryPlan:
        """
        Parses and compiles a user query into a validated, typed QueryPlan contract.
        Supports confirmed field overrides from the Did-You-Mean suggestion panel.
        """
        if not query or not query.strip():
            return QueryPlan(
                dataset_id=self.dataset_id,
                intent=QueryIntent.AMBIGUOUS,
                query_text="",
                is_valid=False,
                validation_message="Query is empty.",
                clarification_needed=True,
                clarification_prompt="Please ask a question about your dataset."
            )

        q_raw = query.strip()
        q_low = q_raw.lower()

        # Step 1: Extract Filters and Entities from Semantic Resolver
        filters = self.resolver.extract_filters_from_query(q_raw)
        covered_spans = getattr(self.resolver, "last_covered_spans", None)
        entities = self.resolver.extract_entities_from_query(q_raw, covered_spans=covered_spans)

        # Step 2: Parse Constraint and Method Flags
        only_numeric = bool(
            re.search(r'\breturn\s+only\s+(?:the\s+)?(?:numerical\s+value|number|value)\b', q_low)
            or any(k in q_low for k in ['return only the numerical value', 'only numerical value', 'return only the number', 'only numerical', 'only value', 'only number', 'return only numerical'])
        )
        no_spearman = bool(
            'spearman' in q_low and any(neg in q_low for neg in [
                'no spearman', 'without spearman', 'not spearman', 'do not include spearman', 
                'exclude spearman', 'excluding spearman', 'do not calculate spearman', 
                'do not mention spearman', "don't calculate spearman", "don't mention spearman"
            ])
        )
        only_pearson = bool(
            re.search(r'\breturn\s+only\s+(?:the\s+)?pearson\b', q_low)
            or ('return only' in q_low and 'pearson' in q_low and 'spearman' not in q_low)
        )

        has_pearson = 'pearson' in q_low and not any(neg in q_low for neg in ['no pearson', 'without pearson', 'not pearson'])
        has_spearman = 'spearman' in q_low and not no_spearman

        methods = ["pearson"]
        if has_spearman and not has_pearson:
            methods = ["spearman"]
        elif has_pearson and has_spearman:
            methods = ["pearson", "spearman"]

        # Step 3: Parse Numeric Limits and Ordering
        limit: Optional[int] = None
        sort_order: Optional[str] = None

        limit_match = re.search(r'\b(?:top|bottom|first|last|limit|highest|lowest)\s+([0-9]+)\b', q_low)
        if limit_match:
            limit = int(limit_match.group(1))

        if any(w in q_low for w in ['top', 'highest', 'best', 'most', 'peak', 'maximum', 'descending', 'desc']):
            sort_order = "descending"
        elif any(w in q_low for w in ['bottom', 'lowest', 'worst', 'least', 'minimum', 'ascending', 'asc']):
            sort_order = "ascending"

        # Step 4: Detect Aggregation Functions
        aggregations: List[str] = []
        primary_agg: Optional[str] = None

        agg_keywords = {
            "SUM": ['sum', 'total', 'volume', 'aggregate sum'],
            "MEAN": ['average', 'avg', 'mean', 'baseline mean'],
            "COUNT": ['count', 'number of', 'how many'],
            "COUNT_DISTINCT": ['distinct count', 'unique count', 'distinct', 'unique'],
            "MIN": ['minimum', 'min', 'lowest'],
            "MAX": ['maximum', 'max', 'highest', 'peak'],
            "MEDIAN": ['median'],
            "STDDEV": ['standard deviation', 'std', 'stddev'],
            "VARIANCE": ['variance', 'var']
        }

        # Check for combined aggregate requests e.g. "total baseline volume and mean value"
        for agg_name, kws in agg_keywords.items():
            for kw in kws:
                if re.search(rf'\b{re.escape(kw)}\b', q_low):
                    if agg_name not in aggregations:
                        aggregations.append(agg_name)
                    break

        if aggregations:
            primary_agg = aggregations[0]

        # Step 5: Resolve Target Dimension
        dimension: Optional[str] = None
        dim_entity = None

        # Look specifically after grouping prepositions / phrases or "-wise" patterns:
        wise_pattern = r'\b([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+){0,2})[-_\s]+wise\b'
        wise_match = re.search(wise_pattern, q_low)
        if wise_match:
            wise_term = wise_match.group(1).strip()
            dim_res = self.resolver.resolve_term(wise_term, preferred_role="Dimension")
            if dim_res.status in [ResolutionStatus.EXACT_MATCH, ResolutionStatus.SEMANTIC_MATCH, ResolutionStatus.FUZZY_MATCH] and dim_res.resolved_column:
                dimension = dim_res.resolved_column
                dim_entity = dim_res

        # "by [dim]", "as [dim]", "for [dim]", "in [dim]", "per [dim]", "across [dim]",
        # "grouped by [dim]", "grouped as [dim]", "grouped for [dim]", "group by [dim]", "group as [dim]",
        # "split by [dim]", "split as [dim]", "breakdown by [dim]", "breakdown as [dim]", "break down by [dim]",
        # "categorized by [dim]", "categorized as [dim]", "classified by [dim]", "classified as [dim]",
        # "based on [dim]", "according to [dim]", "on [dim]"
        prep_pattern = (
            r'\b(?:'
            r'grouped\s+by|group\s+by|grouped\s+as|group\s+as|grouped\s+for|group\s+for|'
            r'split\s+by|split\s+as|breakdown\s+by|breakdown\s+as|break\s+down\s+by|break\s+down\s+as|broken\s+down\s+by|broken\s+down\s+as|'
            r'categorized\s+by|categorized\s+as|categorize\s+by|categorize\s+as|'
            r'classified\s+by|classified\s+as|classify\s+by|classify\s+as|'
            r'across\s+each|for\s+each|in\s+each|per\s+each|'
            r'by|as|for|in|per|across|on|based\s+on|according\s+to'
            r')\s+([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+){0,2})'
        )
        prep_match = re.search(prep_pattern, q_low)
        if prep_match:
            prep_term = prep_match.group(1).strip()
            # Clean trailing prepositions and stopwords
            prep_term = re.sub(r'\s+(?:in|for|with|and|at|across|where|each|all)$', '', prep_term).strip()
            dim_res = self.resolver.resolve_term(prep_term, preferred_role="Dimension")
            if dim_res.status in [ResolutionStatus.EXACT_MATCH, ResolutionStatus.SEMANTIC_MATCH, ResolutionStatus.FUZZY_MATCH] and dim_res.resolved_column:
                dimension = dim_res.resolved_column
                dim_entity = dim_res

        # Fallback to first resolved dimension from query extraction
        filtered_cols = {f.column for f in filters}
        if not dimension and entities["dimensions"]:
            unfiltered_dims = [d for d in entities["dimensions"] if d.resolved_column not in filtered_cols]
            if unfiltered_dims:
                dim_entity = unfiltered_dims[0]
                dimension = dim_entity.resolved_column
            else:
                # If all resolved dimensions are in filtered_cols, check whether any dimension was explicitly
                # mentioned outside of the filter predicate (e.g. "workshop incidents plant zone" vs "category 3")
                for cand_dim in entities["dimensions"]:
                    cand_col = cand_dim.resolved_column
                    rel_filters = [f for f in filters if f.column == cand_col]
                    remainder = q_low
                    for f in rel_filters:
                        if f.raw_predicate_text:
                            remainder = remainder.replace(f.raw_predicate_text.lower(), " ")
                    # Check if column name or known aliases appear in the remainder
                    cand_aliases = {cand_col.lower(), cand_col.lower().replace("_", " ")}
                    if hasattr(self.resolver, "column_map"):
                        cand_aliases.update({a for a, c in self.resolver.column_map.items() if c == cand_col and len(a) >= 3})
                    if any(re.search(rf'\b{re.escape(a)}\b', remainder) for a in cand_aliases):
                        dim_entity = cand_dim
                        dimension = cand_col
                        break

        # Step 6: Resolve Temporal Column & Time Grain
        time_column: Optional[str] = None
        time_grain: Optional[str] = None

        if entities["temporal"]:
            time_column = entities["temporal"][0].resolved_column
        elif self.resolver.time_cols:
            if any(k in q_low for k in ['over time', 'trend', 'monthly', 'daily', 'yearly', 'date', 'time', 'period']):
                time_column = self.resolver.time_cols[0]

        if any(k in q_low for k in ['monthly', 'month', 'by month']):
            time_grain = "month"
        elif any(k in q_low for k in ['daily', 'day', 'by day']):
            time_grain = "day"
        elif any(k in q_low for k in ['yearly', 'year', 'annual', 'by year']):
            time_grain = "year"

        # Step 7: Resolve Target Measures
        resolved_measures: List[str] = [m.resolved_column for m in entities["measures"] if m.resolved_column]

        # If no measures found from query, check if any column name directly appears in query
        if not resolved_measures:
            for m_col in self.resolver.measures:
                if m_col.lower() in q_low:
                    resolved_measures.append(m_col)

        primary_metric = resolved_measures[0] if resolved_measures else None
        secondary_metric = resolved_measures[1] if len(resolved_measures) > 1 else None

        # Bind confirmed field if provided
        if confirmed_field:
            if confirmed_field in self.resolver.measures:
                primary_metric = confirmed_field
                if confirmed_field not in resolved_measures:
                    resolved_measures = [confirmed_field] + resolved_measures
            else:
                dimension = confirmed_field

        # Check for Unresolved Ambiguities (only if none of the candidate alternatives were resolved)
        all_resolved_cols = set(resolved_measures)
        if dimension:
            all_resolved_cols.add(dimension)
        if time_column:
            all_resolved_cols.add(time_column)
        for cat in ["dimensions", "measures", "temporal", "identifiers"]:
            for e in entities.get(cat, []):
                if e.resolved_column:
                    all_resolved_cols.add(e.resolved_column)
        for f in filters:
            if f.column:
                all_resolved_cols.add(f.column)

        unresolved_ambiguities = []
        for ambig in entities["ambiguous"]:
            cand_list = filter_prefixed_candidates(
                list(dict.fromkeys([c.strip() for c in ambig.candidate_alternatives])),
                q_low
            )
            if len(cand_list) <= 1:
                if cand_list:
                    all_resolved_cols.add(cand_list[0])
                continue
            candidates = set(cand_list)

            # 1. If any candidate was already resolved in the query, this is resolved
            if candidates.intersection(all_resolved_cols):
                continue

            # 2. If all candidates normalize to the exact same column name (e.g. whitespace variant), auto-resolve
            cand_norm_map = {re.sub(r'[^a-z0-9]+', '', c.lower()): c for c in cand_list}
            if len(cand_norm_map) <= 1:
                continue

            # 3. If the requested term is a substring or word of an already resolved column, skip
            ambig_norm = re.sub(r'[^a-z0-9]+', '', ambig.requested_term.lower())
            if any(ambig_norm in re.sub(r'[^a-z0-9]+', '', res_col.lower()) for res_col in all_resolved_cols):
                continue

            # 4. If one candidate is an exact match for the term without table prefix, resolve it
            unprefixed_match = [c for c in cand_list if re.sub(r'^[a-zA-Z0-9_\-]+_', '', c).lower() == ambig.requested_term.lower().strip()]
            if len(unprefixed_match) == 1:
                all_resolved_cols.add(unprefixed_match[0])
                continue

            unresolved_ambiguities.append(ambig)

        if unresolved_ambiguities and not confirmed_field:
            ambig = unresolved_ambiguities[0]
            unique_alts = list(dict.fromkeys([c.strip() for c in ambig.candidate_alternatives]))
            if len(unique_alts) > 1:
                top_field = unique_alts[0]
                alt_str = ", ".join([f"'{c}'" for c in unique_alts[:3]])

                # Build rich match suggestions using matcher
                match_suggs = []
                if hasattr(self.resolver, "matcher") and self.resolver.matcher:
                    suggs = self.resolver.matcher.find_matches(ambig.requested_term, top_k=4)
                    match_suggs = [s.to_dict() for s in suggs]

                prompt_lines = [
                    f"I found some potentially matching fields for '{ambig.requested_term}':",
                    ""
                ]
                for idx, alt in enumerate(unique_alts[:3], 1):
                    conf_pct = next((s["confidence_pct"] for s in match_suggs if s["column"] == alt), 85)
                    prompt_lines.append(f"{idx}. {alt} – {conf_pct}% match")
                prompt_lines.append("")
                prompt_lines.append(f"Would you like me to use **{top_field}**?")
                clarif_prompt = "\n".join(prompt_lines)

                return QueryPlan(
                    dataset_id=self.dataset_id,
                    intent=QueryIntent.AMBIGUOUS,
                    query_text=q_raw,
                    is_valid=False,
                    validation_message=f"Term '{ambig.requested_term}' is ambiguous.",
                    clarification_needed=True,
                    clarification_prompt=clarif_prompt,
                    resolved_entities=unresolved_ambiguities,
                    match_suggestions=match_suggs,
                    suggestion_mode="confirmation_required",
                    resolved_field=top_field,
                    interactive_options=[
                        f"Yes, use {top_field}",
                        "Show other matches",
                        "Search again"
                    ]
                )

        # Step 8: Classify Query Intent
        intent = self._classify_intent(
            q_low=q_low,
            entities=entities,
            filters=filters,
            dimension=dimension,
            time_column=time_column,
            measures=resolved_measures,
            limit=limit,
            aggregations=aggregations
        )

        # Intelligent Fallbacks & Defaults
        if intent == QueryIntent.GROUP_COMPARISON and not dimension:
            # Fallback Scenario A: group comparison missing dimension -> downgrade to global aggregation
            if primary_metric:
                intent = QueryIntent.GLOBAL_AGGREGATION
            else:
                intent = QueryIntent.DATASET_SUMMARY

        if intent == QueryIntent.TIME_SERIES_TREND and not time_column:
            # Fallback Scenario B: time series trend missing date column -> fall back to group comparison
            if self.resolver.dimensions:
                dimension = self.resolver.dimensions[0]
                intent = QueryIntent.GROUP_COMPARISON
            elif primary_metric:
                intent = QueryIntent.GLOBAL_AGGREGATION
            else:
                intent = QueryIntent.DATASET_SUMMARY

        if intent == QueryIntent.CORRELATION_ANALYSIS:
            # Fallback Scenario C: correlation with single or no metric -> default pairwise across measures
            if not resolved_measures and len(self.resolver.measures) >= 2:
                resolved_measures = self.resolver.measures[:2]
                primary_metric = resolved_measures[0]
                secondary_metric = resolved_measures[1]

        # Default aggregation if none specified but aggregation intent active
        if intent in [QueryIntent.GLOBAL_AGGREGATION, QueryIntent.GROUP_COMPARISON, QueryIntent.RANKING_TOP_N]:
            if not aggregations:
                if primary_metric:
                    meta = self.resolver.col_metadata.get(primary_metric, {})
                    subtype = str(meta.get("semantic_subtype", meta.get("measure_type", "")))
                    if subtype in ["Rating", "Duration", "Percentage"]:
                        aggregations = ["MEAN"]
                        primary_agg = "MEAN"
                    else:
                        aggregations = ["SUM"]
                        primary_agg = "SUM"
                else:
                    aggregations = ["COUNT"]
                    primary_agg = "COUNT"

        # Build list of all resolved entities for telemetry
        all_resolved_entities: List[ResolvedEntity] = (
            entities["measures"] + entities["dimensions"] + entities["temporal"] + entities["identifiers"]
        )

        # Determine resolved field and suggestions for UI
        target_resolved_field = dimension or primary_metric
        plan_suggestions = []
        if hasattr(self.resolver, "matcher") and self.resolver.matcher and target_resolved_field:
            suggs = self.resolver.matcher.find_matches(target_resolved_field, top_k=3)
            plan_suggestions = [s.to_dict() for s in suggs]

        return QueryPlan(
            dataset_id=self.dataset_id,
            intent=intent,
            sub_intent=intent.value.lower(),
            target_columns=resolved_measures,
            metric=primary_metric,
            metric2=secondary_metric,
            dimension=dimension,
            time_column=time_column,
            time_grain=time_grain,
            aggregations=aggregations,
            aggregation=primary_agg,
            filters=filters,
            sort=sort_order,
            limit=limit,
            methods=methods,
            variables=resolved_measures if intent == QueryIntent.CORRELATION_ANALYSIS else [],
            only_numeric=only_numeric,
            no_spearman=no_spearman,
            only_pearson_result=only_pearson,
            query_text=q_raw,
            resolved_entities=all_resolved_entities,
            match_suggestions=plan_suggestions,
            suggestion_mode="exact" if confirmed_field else "auto_resolved",
            resolved_field=target_resolved_field,
            confirmed_field=confirmed_field,
            interactive_options=[]
        )

    def _classify_intent(
        self,
        q_low: str,
        entities: Dict[str, Any],
        filters: List[ColumnFilter],
        dimension: Optional[str],
        time_column: Optional[str],
        measures: List[str],
        limit: Optional[int],
        aggregations: List[str]
    ) -> QueryIntent:
        """Determines analytical intent using multi-layer structural heuristics."""
        
        # 1. Dataset Summary & Overview
        if any(p in q_low for p in [
            'summarize dataset', 'dataset summary', 'summary of the dataset', 'overview of dataset',
            'how many rows and columns', 'shape of the dataset', 'dataset overview', 'describe dataset',
            'what does this dataset contain', 'what columns are in', 'give me an overview', 'summarize data'
        ]):
            return QueryIntent.DATASET_SUMMARY

        # 2. Data Quality & Hygiene
        if any(p in q_low for p in [
            'missing values', 'null values', 'missing cells', 'nulls in', 'data quality', 
            'cleanliness', 'duplicates', 'duplicate rows', 'any nulls', 'columns with missing'
        ]):
            return QueryIntent.DATA_QUALITY

        # 3. Derivation & Lineage
        if any(p in q_low for p in [
            'how is', 'how was', 'formula for', 'derivation of', 'lineage of', 
            'where does', 'calculated from', 'how did you calculate'
        ]) and any(p in q_low for p in ['calculated', 'derived', 'computed', 'formula', 'lineage']):
            return QueryIntent.DERIVATION_LINEAGE

        # 4. Explain Insight
        if any(q_low.startswith(p) for p in ['why is', 'why did', 'explain why', 'what caused', 'root cause of']):
            return QueryIntent.EXPLAIN_INSIGHT

        # 5. Row Lookup (Specific ID / Entity)
        if any(q_low.startswith(p) for p in ['show row', 'find record', 'lookup', 'look up', 'find transaction', 'find order', 'find incident']):
            return QueryIntent.ROW_LOOKUP
        if entities["identifiers"] and any(p in q_low for p in ['find', 'lookup', 'show', 'where id', 'id =', 'id is']):
            return QueryIntent.ROW_LOOKUP

        # 6. Correlation Analysis
        if any(p in q_low for p in [
            'correlation', 'correlate', 'correlated', 'relationship', 'scatter plot',
            'associated with', 'association between', 'correlation matrix'
        ]):
            return QueryIntent.CORRELATION_ANALYSIS
        if len(measures) >= 2 and (' vs ' in q_low or ' versus ' in q_low or ' relationship ' in q_low):
            return QueryIntent.CORRELATION_ANALYSIS

        # 7. Time Series Trend
        if (time_column or any(p in q_low for p in ['over time', 'trend', 'monthly', 'daily', 'yearly', 'timeline'])) and (
            'trend' in q_low or 'over time' in q_low or 'monthly' in q_low or 'daily' in q_low or 'growth' in q_low
        ):
            return QueryIntent.TIME_SERIES_TREND

        # 8. Hypothesis Testing
        if any(p in q_low for p in [
            'hypothesis test', 'significant difference', 'statistically significant',
            'anova', 'kruskal', 't-test', 'chi-square', 'p-value'
        ]):
            return QueryIntent.HYPOTHESIS_TEST

        # 9. Outlier Analysis
        if any(p in q_low for p in ['outlier', 'outliers', 'anomaly', 'anomalies', 'extreme values', 'unusual values', 'iqr']):
            return QueryIntent.OUTLIER_ANALYSIS

        # 10. Distribution Analysis
        if any(p in q_low for p in [
            'distribution', 'histogram', 'skewness', 'skewed', 'kurtosis', 
            'spread', 'quartiles', 'percentiles', 'variance of', 'std dev'
        ]):
            return QueryIntent.DISTRIBUTION_ANALYSIS

        # 11. Distinct Count
        if any(p in q_low for p in [
            'distinct count', 'unique count', 'how many unique', 'how many distinct', 
            'number of unique', 'number of different', 'how many different'
        ]):
            return QueryIntent.DISTINCT_COUNT

        # 12. Ranking Top N
        if (limit is not None or any(p in q_low for p in ['top', 'bottom', 'highest', 'lowest', 'best', 'worst', 'rank', 'ranking'])) and (
            dimension or any(p in q_low for p in ['by', 'per', 'as', 'for', 'in'])
        ):
            return QueryIntent.RANKING_TOP_N

        # 13. Filtered Records (Show rows matching filter without aggregate)
        is_record_request = any(p in q_low for p in [
            'show', 'list', 'filter', 'display', 'get all', 'find all',
            'raw records', 'raw data', 'details of'
        ])
        filtered_dim_cols = {f.column for f in filters}
        if filters and not aggregations and is_record_request and (not dimension or dimension in filtered_dim_cols):
            return QueryIntent.FILTERED_RECORDS

        # 14. Group Comparison
        # Triggers when a target dimension is identified and the query represents grouping,
        # count breakdown, comparison, or filtered breakdown across categories.
        has_group_marker = any(re.search(rf'\b{p}\b', q_low) for p in [
            'by', 'as', 'for', 'in', 'per', 'across', 'grouped', 'group', 'split',
            'breakdown', 'categorized', 'classified', 'distribution', 'break down',
            'counts', 'count', 'how many'
        ])
        if dimension and (measures or 'COUNT' in aggregations or has_group_marker or filters or not is_record_request):
            return QueryIntent.GROUP_COMPARISON

        # Filtered measures check: if a measure is only referenced in a filter condition and no aggregation requested
        unfiltered_measures = [m for m in measures if not any(f.column == m for f in filters)]

        # 15. Global Aggregation
        if unfiltered_measures or aggregations:
            return QueryIntent.GLOBAL_AGGREGATION

        # Default fallback
        if filters:
            return QueryIntent.FILTERED_RECORDS
        return QueryIntent.DATASET_SUMMARY


def parse_query_plan(df: Optional[pd.DataFrame], query: str, sem_engine: Any = None) -> QueryPlan:
    """
    Backward-compatible entry point for parse_query_plan matching the legacy signature.
    Returns the authoritative QueryPlan contract.
    """
    planner = UniversalQueryPlanner(df=df, sem_layer_dict=getattr(sem_engine, "sem_layer", None) if sem_engine else None)
    return planner.plan(query)
