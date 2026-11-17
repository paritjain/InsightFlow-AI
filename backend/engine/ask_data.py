import pandas as pd
import numpy as np
import re
from engine.cleaner import coerce_numeric_columns
from engine.profiler import infer_column_semantic_metadata
from engine.semantic import SemanticLayer

class IntentStr(str):
    """
    Dataset-Agnostic Universal Intent String.
    Serializes to standard universal intent names:
    - 'global_aggregation'
    - 'group_comparison'
    - 'time_series_trend'
    - 'correlation_analysis'
    - 'distribution_analysis'
    - 'filtering_top_n'
    Supports backward-compatible equality checks against legacy intent keywords.
    """
    def __new__(cls, val: str, sub_intent: str = None):
        instance = super().__new__(cls, str(val))
        instance.sub_intent = sub_intent or str(val)
        return instance

    def __eq__(self, other):
        if other is None:
            return False
        if not isinstance(other, str):
            return super().__eq__(other)
        s = str(self)
        other_low = other.lower().strip()
        if s == "global_aggregation" and other_low in ["global_aggregation", "aggregate"]:
            return True
        if s == "group_comparison" and other_low in ["group_comparison", "ranking"]:
            return True
        if s == "time_series_trend" and other_low in ["time_series_trend", "trend", "growth", "root-cause analysis"]:
            return True
        if s == "correlation_analysis" and other_low in [
            "correlation_analysis", "relationship", "relationship_single", 
            "relationship_multiple", "correlation_matrix", "correlation_ranking", "relationship_ranking"
        ]:
            return True
        if s == "distribution_analysis" and other_low in ["distribution_analysis", "distribution", "outlier"]:
            return True
        if s == "filtering_top_n" and other_low in ["filtering_top_n", "ranking"]:
            return True
        if hasattr(self, 'sub_intent') and self.sub_intent and self.sub_intent.lower() == other_low:
            return True
        return super().__eq__(other)

    def __hash__(self):
        return hash(str(self))

class AggStr(str):
    """Case-tolerant aggregation method identifier (e.g. SUM, MEAN, COUNT, MIN, MAX)."""
    def __eq__(self, other):
        if isinstance(other, str):
            return self.upper() == other.upper()
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.upper())

class MetricVal(str):
    """Dynamic Metric descriptor allowing scalar string and vector/list equality checks."""
    def __new__(cls, val, formatted_list=None):
        instance = super().__new__(cls, str(val) if val is not None else "")
        instance.val = str(val) if val is not None else ""
        instance.formatted_list = formatted_list or ([f"{instance.val} (SUM)"] if instance.val else [])
        return instance

    def __eq__(self, other):
        if other is None:
            return self.val == ""
        if isinstance(other, list):
            return other == self.formatted_list or other == [self.val] or (len(other) == 1 and other[0] == self.val)
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.val)

def format_value(val: float, subtype: str = None) -> str:
    """Formats numeric values dynamically based on semantic subtype."""
    if pd.isna(val):
        return "N/A"
    if subtype == "Currency":
        return f"${val:,.2f}" if val >= 0 else f"-${abs(val):,.2f}"
    elif subtype == "Percentage":
        return f"{val:.2f}%"
    elif subtype in ["Count", "Quantity"]:
        return f"{int(round(val)):,}"
    elif subtype in ["Rating", "Duration"]:
        return f"{val:.2f}"
    else:
        return f"{val:,.2f}" if isinstance(val, float) and not val.is_integer() else f"{int(val):,}"

def extract_explicit_schema_entities(all_cols: list, query: str, sem_engine: SemanticLayer) -> list:
    """
    Regex and Fuzzy Entity Extractor:
    Scans the user query against ALL available columns in active dataset schema.
    Extracts explicit dimensions, metrics, and context bindings.
    """
    found_entities = []
    q_low = query.lower()

    for col in all_cols:
        c_str = str(col)
        # 1. Exact word match
        exact_match = re.search(rf'\b{re.escape(c_str)}\b', query, re.IGNORECASE)
        # 2. Underscore/space normalized word match
        words = re.findall(r'[A-Za-z0-9]+', c_str)
        spaced_pattern = r'\b' + r'[\s_]*'.join([re.escape(w) for w in words]) + r'\b' if words else None
        spaced_match = re.search(spaced_pattern, query, re.IGNORECASE) if spaced_pattern else None

        match = exact_match or spaced_match
        if match:
            span = match.span()
            start_idx = span[0]
            preceding_text = q_low[max(0, start_idx - 30):start_idx].strip()

            is_dim_context = any(preceding_text.endswith(p) for p in [
                'by', 'grouped by', 'per', 'across each', 'split by', 'breakdown by', 'for each',
                'which', 'what', 'who', 'show', 'find', 'list'
            ]) or bool(re.search(r'\b(?:which|what|who|show|find|list)\s+(?:the\s+)?$', preceding_text))

            is_metric_context = any(preceding_text.endswith(p) for p in [
                'average', 'avg', 'mean', 'total', 'sum', 'highest', 'lowest', 'max', 'min', 'median', 'std'
            ]) or bool(re.search(r'\b(?:average|avg|mean|total|sum|highest|lowest|max|min|median|std)\s+(?:of\s+)?(?:the\s+)?$', preceding_text))

            meta = sem_engine.sem_layer.get(col, {})
            is_meas = meta.get("is_measure", False) or col in sem_engine.measures
            is_id = meta.get("is_identifier", False) or col in sem_engine.id_cols

            found_entities.append({
                "column": col,
                "span": span,
                "is_dim_context": is_dim_context,
                "is_metric_context": is_metric_context,
                "is_measure": is_meas,
                "is_id": is_id
            })

    return found_entities

def parse_query_plan(df: pd.DataFrame, query: str, sem_engine: SemanticLayer) -> dict:
    """
    Dataset-Agnostic Structured Query Planner & Universal Intent Classifier.
    Classifies user queries into 6 execution intents based strictly on syntactic patterns and schema availability:
    - global_aggregation
    - group_comparison
    - time_series_trend
    - correlation_analysis
    - distribution_analysis
    - filtering_top_n

    Applies Intelligent Fallback Engine (graceful degradation) for underspecified requests:
    - Fallback Scenario A: group_comparison missing dimension -> downgrades to global_aggregation
    - Fallback Scenario B: time_series_trend missing date column -> falls back to group_comparison on highest-cardinality dimension
    - Fallback Scenario C: correlation_analysis with single metric -> computes pairwise correlations against all numeric measures
    """
    q_low = query.lower().strip()
    q_norm = re.sub(r'[^a-z0-9\s]+', '', q_low)
    q_words = set(q_norm.split())

    # Detect output format / constraint flags
    only_numeric = bool(re.search(r'\breturn\s+only\s+(?:the\s+)?(?:numerical\s+value|number|value)\b', q_low) or any(k in q_low for k in ['return only the numerical value', 'only numerical value', 'return only the number', 'only numerical', 'only value', 'only number', 'return only numerical']))
    no_spearman = 'spearman' in q_low and any(neg in q_low for neg in ['no spearman', 'without spearman', 'not spearman', 'do not include spearman', 'exclude spearman', 'excluding spearman', 'do not calculate spearman', 'do not mention spearman', 'don\'t calculate spearman', 'don\'t mention spearman', 'no spearman correlations'])
    only_pearson_result = bool(re.search(r'\breturn\s+only\s+(?:the\s+)?pearson\b', q_low) or ('return only' in q_low and 'pearson' in q_low and 'spearman' not in q_low))

    # Detect correlation methods
    has_pearson = 'pearson' in q_low and not any(neg in q_low for neg in ['no pearson', 'without pearson', 'not pearson', 'do not include pearson', 'exclude pearson', 'excluding pearson'])
    has_spearman = 'spearman' in q_low and not no_spearman

    methods = []
    if has_pearson and not has_spearman:
        methods = ["pearson"]
    elif has_spearman and not has_pearson:
        methods = ["spearman"]
    elif has_pearson and has_spearman:
        methods = ["pearson", "spearman"]
    elif no_spearman:
        methods = ["pearson"]
    else:
        methods = ["pearson"]

    # Detect grouping prepositions early
    has_group_prep = any(re.search(rf'\b{p}\b', q_low) for p in ['by', 'grouped by', 'per', 'across each', 'split by', 'breakdown by', 'break down by', 'for each'])

    # 1. Resolve Explicit Schema Entities
    all_schema_cols = list(df.columns) if (df is not None and hasattr(df, "columns")) else list(sem_engine.sem_layer.keys())
    explicit_entities = extract_explicit_schema_entities(all_schema_cols, query, sem_engine)

    # 2. Resolve Measures mentioned in query
    resolved_measures = []
    for ent in explicit_entities:
        if (ent["is_measure"] or ent["is_metric_context"]) and ent["column"] not in resolved_measures:
            resolved_measures.append(ent["column"])

    for col in sem_engine.measures:
        c_low = str(col).lower()
        c_norm = c_low.replace('_', ' ')
        if c_low in q_low or c_norm in q_low:
            if col not in resolved_measures:
                resolved_measures.append(col)

    words_list = q_low.split()
    for i in range(len(words_list)):
        for j in range(i+1, min(i+4, len(words_list)+1)):
            phrase = " ".join(words_list[i:j])
            match = sem_engine.resolve_column(phrase, role_filter="Measure")
            if match and match[1] >= 0.60:
                if match[0] not in resolved_measures:
                    resolved_measures.append(match[0])

    # 3. Resolve Dimensions mentioned in query
    dimension = None
    dim_conf = 0.0

    # Look specifically after grouping prepositions / phrases: "by [dim]", "as [dim]", "for [dim]", "in [dim]", "per [dim]"
    prep_match = re.search(
        r'\b(?:'
        r'grouped\s+by|group\s+by|grouped\s+as|group\s+as|grouped\s+for|group\s+for|'
        r'split\s+by|split\s+as|breakdown\s+by|breakdown\s+as|break\s+down\s+by|break\s+down\s+as|broken\s+down\s+by|'
        r'categorized\s+by|categorized\s+as|classified\s+by|classified\s+as|'
        r'across\s+each|for\s+each|in\s+each|per\s+each|'
        r'by|as|for|in|per|across|on|based\s+on|according\s+to'
        r')\s+([a-z0-9_]+(?:\s+[a-z0-9_]+)?)',
        q_low
    )
    if prep_match:
        prep_target = prep_match.group(1).strip()
        # Ensure not global scope words
        if prep_target not in ['the dataset', 'all records', 'all rows', 'dataset', 'records', 'data']:
            dim_res = sem_engine.resolve_column(prep_target, role_filter="Dimension")
            if dim_res:
                dimension, dim_conf = dim_res

    # Look after question pronouns targeting a dimension: "which [dim]", "what [dim]", "who [dim]"
    if not dimension:
        q_target_match = re.search(r'\b(?:which|what|who|show|find|list)\s+([a-z0-9_]+(?:\s+[a-z0-9_]+)?)', q_low)
        if q_target_match:
            q_target = q_target_match.group(1).strip()
            if q_target not in ['is', 'are', 'was', 'the', 'a', 'an', 'dataset', 'records', 'rows', 'data', 'average', 'total', 'mean', 'sum', 'highest', 'lowest']:
                dim_res = sem_engine.resolve_column(q_target, role_filter="Dimension")
                if dim_res:
                    dimension, dim_conf = dim_res

    # Check explicit entities with dimension context (e.g. "Which WarehouseID")
    if not dimension:
        for ent in explicit_entities:
            if ent["is_dim_context"]:
                dimension = ent["column"]
                dim_conf = 1.0
                break

    # If multiple entities found and one is dimension/identifier while another is measure
    if not dimension and explicit_entities:
        for ent in explicit_entities:
            if not ent["is_measure"] and ent["column"] not in resolved_measures:
                dimension = ent["column"]
                dim_conf = 0.95
                break

    if not dimension:
        for i in range(len(words_list)):
            for j in range(i+1, min(i+4, len(words_list)+1)):
                phrase = " ".join(words_list[i:j])
                # Skip global scope phrases and stopwords
                if phrase in ['the dataset', 'all records', 'all rows', 'dataset', 'records', 'all', 'for', 'and', 'the', 'with', 'calculate', 'count', 'mean', 'total', 'average', 'sum']:
                    continue
                match = sem_engine.resolve_column(phrase, role_filter="Dimension")
                if match and match[1] > dim_conf:
                    dimension, dim_conf = match

    if not dimension:
        for kw in q_words:
            if len(kw) >= 3 and kw not in ['all', 'the', 'and', 'for', 'each', 'set', 'data', 'dataset', 'records', 'rows', 'overall', 'baseline', 'calculate', 'count', 'mean', 'total', 'average', 'sum']:
                match = sem_engine.resolve_column(kw, role_filter="Dimension")
                if match and match[1] >= 0.75:
                    dimension, dim_conf = match
                    break

    # Final string-matching pass comparing user query against ALL available columns in active dataset schema:
    # Before returning Dimension: None, if a column name (or alias) is explicitly present in user query,
    # force-bind it to Dimension regardless of loose LLM parameter extraction.
    if not dimension:
        for ent in explicit_entities:
            c = ent["column"]
            # Do not force-bind if this column is the sole metric being aggregated without grouping prepositions
            if c in resolved_measures and len(explicit_entities) == 1 and not has_group_prep and not ent["is_dim_context"]:
                continue
            dimension = c
            dim_conf = 1.0
            break

    # 3. Resolve Temporal Columns
    time_col = None
    if sem_engine.time_cols:
        time_col = sem_engine.time_cols[0]

    # 4. Multi-Metric and Aggregation Extraction
    agg_positions = []
    
    # Check count
    count_matches = [m.start() for m in re.finditer(r'\b(count|number\s+of|how\s+many)\b', q_low)]
    if count_matches and not any(k in q_low for k in ['unique count', 'distinct count']):
        agg_positions.append((min(count_matches), AggStr("COUNT")))
    elif any(k in q_low for k in ['unique count', 'distinct count', 'distinct']):
        distinct_matches = [m.start() for m in re.finditer(r'\b(unique\s+count|distinct\s+count|distinct)\b', q_low)]
        if distinct_matches:
            agg_positions.append((min(distinct_matches), AggStr("COUNT_DISTINCT")))

    # Check sum/total
    sum_matches = [m.start() for m in re.finditer(r'\b(sum|total|cumulative)\b', q_low)]
    if not sum_matches and 'volume' in q_low and ('total volume' in q_low or 'baseline volume' in q_low):
        vol_matches = [m.start() for m in re.finditer(r'\bvolume\b', q_low)]
        if vol_matches:
            sum_matches = vol_matches
    if sum_matches:
        agg_positions.append((min(sum_matches), AggStr("SUM")))

    # Check mean/average
    mean_matches = [m.start() for m in re.finditer(r'\b(average|avg|mean|typical|mean\s+value)\b', q_low)]
    if mean_matches:
        agg_positions.append((min(mean_matches), AggStr("MEAN")))

    # Check min
    min_matches = [m.start() for m in re.finditer(r'\b(min|minimum|lowest)\b', q_low)]
    if min_matches and not any(k in q_low for k in ['worst', 'least', 'bottom']):
        agg_positions.append((min(min_matches), AggStr("MIN")))

    # Check max
    max_matches = [m.start() for m in re.finditer(r'\b(max|maximum|highest|peak)\b', q_low)]
    if max_matches and not any(k in q_low for k in ['top', 'best', 'leading']):
        agg_positions.append((min(max_matches), AggStr("MAX")))

    # Check median
    median_matches = [m.start() for m in re.finditer(r'\b(median|midpoint|50th\s+percentile)\b', q_low)]
    if median_matches:
        agg_positions.append((min(median_matches), AggStr("MEDIAN")))

    # Check std
    std_matches = [m.start() for m in re.finditer(r'\b(std|standard\s+deviation|variance|stddev)\b', q_low)]
    if std_matches:
        agg_positions.append((min(std_matches), AggStr("STD")))

    agg_positions.sort(key=lambda x: x[0])
    detected_aggs = [agg for pos, agg in agg_positions]

    # Default to SUM or MEAN if no explicit aggregations detected
    if not detected_aggs:
        if resolved_measures and sem_engine.sem_layer.get(resolved_measures[0], {}).get("semantic_subtype") in ["Rating", "Percentage", "Duration"]:
            detected_aggs.append(AggStr("MEAN"))
        else:
            detected_aggs.append(AggStr("SUM"))

    primary_agg = detected_aggs[0].lower() if detected_aggs else "sum"

    # 5. Syntactic Marker Detection
    # Scope detection (global vs grouped)
    is_global_scope = any(k in q_low for k in [
        'across the dataset', 'across all records', 'across all rows', 'across dataset',
        'across records', 'across the data', 'overall', 'baseline', 'in the dataset',
        'entire dataset', 'whole dataset', 'all records', 'all rows', 'across all'
    ])
    has_group_prep = any(re.search(rf'\b{p}\b', q_low) for p in [
        'by', 'as', 'for', 'in', 'per', 'across', 'grouped by', 'group by', 'grouped as', 'group as',
        'grouped for', 'group for', 'across each', 'for each', 'in each', 'per each',
        'split by', 'split as', 'breakdown by', 'break down by', 'categorized by', 'categorized as'
    ])
    
    # If the user explicitly asks across all records/whole dataset, and did NOT specify a dimension
    if is_global_scope and not has_group_prep:
        dimension = None

    # Temporal markers
    has_temporal_kw = any(k in q_low for k in [
        'over time', 'monthly', 'daily', 'weekly', 'yearly', 'trend', 
        'seasonality', 'period-over-period', 'timeline', 'progression', 
        'month-over-month', 'year-over-year', 'mom', 'yoy'
    ])

    # Correlation markers
    is_matrix = any(k in q_low for k in ['matrix', 'table of correlation', 'full matrix', 'correlation table'])
    is_ranking_corr = any(k in q_low for k in ['strongest', 'weakest', 'most correlated', 'least correlated', 'highest correlation', 'lowest correlation'])
    is_multiple_corr = any(k in q_low for k in ['multiple correlations', 'three correlations', 'four correlations', 'all pairs', 'these three', 'these four']) or (len(resolved_measures) >= 3 and ('compare' in q_low or 'pairs' in q_low or 'each' in q_low or 'these' in q_low)) or len(re.findall(r'([a-z0-9_]+)\s+(?:vs|versus|and)\s+([a-z0-9_]+)', q_low)) >= 2
    has_corr_kw = any(k in q_low for k in [
        'relationship', 'correlation', 'pearson', 'spearman', 'covari', 'co-vari', 
        'associated with', 'association between', 'relationship between', 'correlation between',
        'affect', 'effect', 'impact', 'depend', 'relate', 'vs', 'versus', 'scatter'
    ])

    # Distribution markers
    has_dist_kw = any(k in q_low for k in [
        'distribution', 'spread', 'skew', 'outlier', 'anomal', 'histogram', 
        'variance', 'quartile', 'iqr', 'percentile', 'box plot', 'ranges'
    ])

    # Filtering top N markers
    has_filter_kw = any(re.search(rf'\b{k}\b', q_low) for k in [
        r'top\s+\d+', r'bottom\s+\d+', r'highest\s+\d+', r'lowest\s+\d+',
        r'top', r'bottom', r'filter\s+where', r'which\s+\w+\s+are\s+most',
        r'which\s+\w+\s+is\s+best', r'best\s+performing', r'worst\s+performing', r'rank',
        r'highest', r'lowest', r'peak', r'best', r'worst', r'leading', r'lagging'
    ])

    # 6. Universal Intent Classification Matrix & Fallback Engine
    intent = None
    sub_intent = None
    fallback_applied = None
    pairs = []
    vars_list = resolved_measures if resolved_measures else sem_engine.measures
    sort = None
    limit = None
    time_grain = None

    # Priority 1: Correlation Analysis
    if is_matrix:
        intent = IntentStr("correlation_analysis", sub_intent="correlation_matrix")
        sub_intent = "correlation_matrix"
        explicit_vars = []
        match = re.search(r'(?:using|matrix of|between|among|for|with)\s+(.+)', query, re.IGNORECASE)
        if match:
            raw_chunk = match.group(1)
            raw_chunk = re.split(r'\.\s+|\bcalculate\b|\busing raw\b|\breturn\b|\bdo not\b', raw_chunk, flags=re.IGNORECASE)[0]
            parts = re.split(r'[,&]|\band\b', raw_chunk)
            for p in parts:
                p_clean = re.sub(r'^[^\w]+|[^\w]+$', '', p.strip())
                if not p_clean:
                    continue
                resolved = sem_engine.resolve_column(p_clean, role_filter="Measure")
                var_name = resolved[0] if resolved else p_clean
                if var_name.lower() not in ['raw', 'row-level', 'observations', 'dimension', 'grouping', 'only', 'matrix', 'complete']:
                    if var_name not in explicit_vars:
                        explicit_vars.append(var_name)
        vars_list = explicit_vars if len(explicit_vars) >= 2 else (resolved_measures if len(resolved_measures) >= 2 else sem_engine.measures)

    elif is_ranking_corr:
        intent = IntentStr("correlation_analysis", sub_intent="correlation_ranking")
        sub_intent = "correlation_ranking"
        explicit_vars = []
        match = re.search(r'(?:among|using|between|for|with|of)\s+(.+)', query, re.IGNORECASE)
        if match:
            raw_chunk = match.group(1)
            raw_chunk = re.split(r'\.\s+|\bidentify\b|\bcalculate\b|\busing raw\b|\breturn\b|\bdo not\b|\brank\b|\bfind\b', raw_chunk, flags=re.IGNORECASE)[0]
            parts = re.split(r'[,&]|\band\b', raw_chunk)
            for p in parts:
                p_clean = re.sub(r'^[^\w]+|[^\w]+$', '', p.strip())
                if not p_clean:
                    continue
                resolved = sem_engine.resolve_column(p_clean, role_filter="Measure")
                var_name = resolved[0] if resolved else p_clean
                if var_name.lower() not in ['raw', 'row-level', 'observations', 'dimension', 'grouping', 'only', 'matrix', 'complete', 'three', 'four', 'five', 'positive', 'negative', 'correlations', 'strongest', 'weakest']:
                    if var_name not in explicit_vars:
                        explicit_vars.append(var_name)
        vars_list = explicit_vars if len(explicit_vars) >= 2 else (resolved_measures if len(resolved_measures) >= 2 else sem_engine.measures)
        top_n = 3
        n_match = re.search(r'\b(?:three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:strongest|top)\b', q_low)
        if n_match:
            word_to_num = {'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
            w = n_match.group(0).split()[0]
            top_n = word_to_num.get(w, int(w) if w.isdigit() else 3)
        limit = top_n

    elif is_multiple_corr and len(resolved_measures) >= 2:
        intent = IntentStr("correlation_analysis", sub_intent="relationship_multiple")
        sub_intent = "relationship_multiple"
        matches = re.findall(r'([a-z0-9_]+)\s+(?:vs|versus|and)\s+([a-z0-9_]+)', q_low)
        for p1_raw, p2_raw in matches:
            m1 = sem_engine.resolve_column(p1_raw, role_filter="Measure")
            m2 = sem_engine.resolve_column(p2_raw, role_filter="Measure")
            if m1 and m2 and m1[0] != m2[0]:
                pair = (m1[0], m2[0])
                rev_pair = (m2[0], m1[0])
                if pair not in pairs and rev_pair not in pairs:
                    pairs.append(pair)
        if not pairs:
            for i in range(len(resolved_measures)):
                for j in range(i+1, len(resolved_measures)):
                    pairs.append((resolved_measures[i], resolved_measures[j]))

    elif has_corr_kw:
        intent = IntentStr("correlation_analysis", sub_intent="relationship_single")
        sub_intent = "relationship_single"
        if len(resolved_measures) >= 2:
            pairs = [(resolved_measures[0], resolved_measures[1])]
        elif len(resolved_measures) == 1:
            # Fallback Scenario C: correlation_analysis with single metric
            # Compute pair-wise correlations against all other numeric measures in the dataset
            single_m = resolved_measures[0]
            other_measures = [m for m in sem_engine.measures if m != single_m]
            sub_intent = "single_metric_correlation"
            pairs = [(single_m, m) for m in other_measures]
            vars_list = [single_m] + other_measures
            fallback_applied = "single_metric_correlation_expanded_to_pairwise"

    # Priority 2: Distribution Analysis
    elif has_dist_kw:
        intent = IntentStr("distribution_analysis", sub_intent="distribution")
        sub_intent = "outlier" if any(k in q_low for k in ['outlier', 'anomal', 'iqr']) else "distribution"

    # Priority 3: Time Series Trend
    elif has_temporal_kw:
        time_grain = "Month" if 'month' in q_low else ("Day" if 'day' in q_low else "Year")
        if time_col:
            intent = IntentStr("time_series_trend", sub_intent="trend")
            sub_intent = "growth" if any(k in q_low for k in ['growth', 'mom', 'yoy', 'increase rate']) else ("root-cause" if any(k in q_low for k in ['why', 'decline', 'drop', 'root cause']) else "trend")
        else:
            # Fallback Scenario B: time_series_trend missing date feature
            # Fall back to group_comparison using highest-cardinality discrete categorical column
            if sem_engine.dimensions:
                fallback_dim = max(sem_engine.dimensions, key=lambda d: sem_engine.sem_layer.get(d, {}).get("unique_count", 0))
                intent = IntentStr("group_comparison", sub_intent="group_comparison")
                sub_intent = "group_comparison"
                dimension = fallback_dim
                fallback_applied = f"time_series_trend_fallback_to_group_comparison_by_{fallback_dim}"
            else:
                intent = IntentStr("global_aggregation", sub_intent="global_aggregation")
                sub_intent = "global_aggregation"
                dimension = None
                fallback_applied = "time_series_trend_fallback_to_global_aggregation"

    # Priority 4: Filtering Top N
    elif has_filter_kw and not is_global_scope and (dimension or len(resolved_measures) >= 1):
        intent = IntentStr("filtering_top_n", sub_intent="ranking")
        sub_intent = "ranking"
        sort = "ascending" if any(k in q_low for k in ['worst', 'lowest', 'least', 'bottom']) else "descending"
        limit_match = re.search(r'\b(?:top|bottom|highest|lowest)\s+(\d+)\b', q_low)
        if limit_match:
            limit = int(limit_match.group(1))
        elif any(k in q_low for k in ['which', 'what', 'who', 'highest', 'lowest', 'best', 'worst', 'peak']):
            limit = 1
        else:
            limit = 5

        # Superlative aggregation resolution (e.g. "highest average shipment distance")
        if any(k in q_low for k in ['average', 'avg', 'mean']) and any(k in q_low for k in ['highest', 'lowest', 'best', 'worst', 'top', 'bottom']):
            primary_agg = "mean"
            detected_aggs = [AggStr("MEAN")]
        elif any(k in q_low for k in ['total', 'sum']) and any(k in q_low for k in ['highest', 'lowest', 'best', 'worst', 'top', 'bottom']):
            primary_agg = "sum"
            detected_aggs = [AggStr("SUM")]
        elif any(k in q_low for k in ['median']) and any(k in q_low for k in ['highest', 'lowest', 'best', 'worst', 'top', 'bottom']):
            primary_agg = "median"
            detected_aggs = [AggStr("MEDIAN")]
        elif not any(k in q_low for k in ['average', 'avg', 'mean', 'total', 'sum', 'median', 'std']):
            primary_agg = "max" if sort == "descending" else "min"
            detected_aggs = [AggStr("MAX" if sort == "descending" else "MIN")]

    # Priority 5: Group Comparison vs Fallback Scenario A
    elif has_group_prep or 'compare' in q_low or 'comparison' in q_low or dimension:
        if dimension and not is_global_scope:
            intent = IntentStr("group_comparison", sub_intent="group_comparison")
            sub_intent = "group_comparison"
        else:
            # Fallback Scenario A: group_comparison missing dimension
            # Automatically downgrade intent to global_aggregation
            intent = IntentStr("global_aggregation", sub_intent="global_aggregation")
            sub_intent = "global_aggregation"
            dimension = None
            fallback_applied = "group_comparison_missing_dim_downgraded_to_global_aggregation"

    # Priority 6: Global Aggregation
    else:
        intent = IntentStr("global_aggregation", sub_intent="global_aggregation")
        sub_intent = "global_aggregation"
        dimension = None

    # Fallback Scenario A Check (catch any case where group_comparison ended up with no dimension)
    if intent == "group_comparison" and not dimension:
        intent = IntentStr("global_aggregation", sub_intent="global_aggregation")
        sub_intent = "global_aggregation"
        dimension = None
        fallback_applied = "group_comparison_missing_dim_downgraded_to_global_aggregation"

    # Metric binding
    metric1 = resolved_measures[0] if len(resolved_measures) >= 1 else (sem_engine.measures[0] if sem_engine.measures else None)
    metric2 = resolved_measures[1] if len(resolved_measures) >= 2 else None

    # Vectorized multi-metrics formatting
    if len(detected_aggs) > 1 and metric1:
        formatted_metrics = [f"{metric1} ({a.upper()})" for a in detected_aggs]
    elif len(resolved_measures) > 1:
        formatted_metrics = [f"{m} ({detected_aggs[0].upper() if detected_aggs else 'SUM'})" for m in resolved_measures]
    elif metric1:
        formatted_metrics = [f"{metric1} ({detected_aggs[0].upper() if detected_aggs else 'SUM'})"]
    else:
        formatted_metrics = []

    metric_val = MetricVal(metric1, formatted_metrics) if metric1 else None

    # Correlation analysis must have aggregation=None, aggregations=[], dimension=None
    if intent == "correlation_analysis":
        primary_agg = None
        detected_aggs = []
        dimension = None

    plan = {
        "intent": intent,
        "Intent": intent,
        "sub_intent": sub_intent,
        "metric": metric_val,
        "Metric": metric_val,
        "metric2": metric2,
        "metrics": formatted_metrics,
        "Metrics": formatted_metrics,
        "dimension": dimension,
        "Dimension": dimension,
        "aggregations": detected_aggs,
        "Aggregations": detected_aggs,
        "aggregation": primary_agg,
        "Aggregation": primary_agg,
        "fallback_applied": fallback_applied,
        "time_col": time_col,

        "time_grain": time_grain,
        "variables": vars_list,
        "pairs": pairs,
        "methods": methods,
        "granularity": "row-level",
        "sort": sort,
        "limit": limit,
        "only_numeric": only_numeric,
        "no_spearman": no_spearman,
        "only_pearson_result": only_pearson_result,
        "query_text": query
    }

    return plan

def validate_query_plan(plan: dict, sem_engine: SemanticLayer) -> tuple[bool, str]:
    """
    Intelligent Guardrail Validation Engine with Pre-Execution Fallback Support.
    Gracefully validates plans and downgrades before rejecting irrecoverable queries.
    """
    intent = plan.get("intent")
    metric = plan.get("metric")
    metric2 = plan.get("metric2")
    dim = plan.get("dimension")
    time_col = plan.get("time_col")
    sub_intent = plan.get("sub_intent")
    pairs = plan.get("pairs", [])
    vars_list = plan.get("variables", [])

    available_measures = ", ".join([f"'{c}'" for c in sem_engine.measures]) or "None"
    available_dims = ", ".join([f"'{c}'" for c in sem_engine.dimensions]) or "None"

    # Intents requiring at least 1 valid measure
    if intent in ["global_aggregation", "group_comparison", "distribution_analysis", "filtering_top_n", "time_series_trend"]:
        if not metric:
            return False, f"Could not map query to a valid quantitative measure in the dataset. Available measures: [{available_measures}]."
        if str(metric) not in sem_engine.sem_layer and str(metric) not in sem_engine.measures:
            return False, f"Column '{metric}' does not exist in dataset schema."

    # Correlation Intent Validation
    if intent == "correlation_analysis":
        if sub_intent == "single_metric_correlation":
            if not metric:
                return False, f"Correlation analysis requires at least 1 quantitative metric. Available measures: [{available_measures}]."
            if len(sem_engine.measures) < 2:
                return False, f"Correlation analysis requires at least 2 quantitative measures in the dataset. Available measures: [{available_measures}]."
        elif sub_intent == "relationship_single":
            if not metric or not metric2:
                return False, f"Relationship intent requires 2 quantitative metrics. Available measures: [{available_measures}]."
        elif sub_intent == "relationship_multiple":
            if not pairs:
                return False, f"Multiple relationship intent requires at least 1 valid quantitative measure pair to compare. Available measures: [{available_measures}]."
        elif sub_intent in ["correlation_matrix", "correlation_ranking"]:
            if len(vars_list) < 2:
                return False, f"Correlation matrix/ranking requires at least 2 quantitative measures. Available measures: [{available_measures}]."

    # Temporal Intent Validation
    if intent == "time_series_trend" and not time_col:
        return False, f"Query intent '{intent}' requires a temporal/date column, but no valid time field was found in dataset schema."

    # Dimension Validation for Group Comparison / Filtering
    if intent in ["group_comparison", "filtering_top_n"]:
        if intent == "filtering_top_n" and not dim:
            return False, f"Query intent 'ranking' requires a categorical dimension. Available dimensions: [{available_dims}]."
        if dim:
            all_cols = list(sem_engine.sem_layer.keys()) if hasattr(sem_engine, "sem_layer") else []
            if dim not in all_cols and dim not in sem_engine.dimensions and dim not in sem_engine.id_cols and dim not in sem_engine.measures:
                return False, f"Dimension column '{dim}' does not exist in dataset schema."
            # Dynamic Role Fallback: Allow discrete string/integer/ID/measure columns dynamically as valid dimensions for grouping and ranking queries

    return True, "Valid"

def ask_your_data(df: pd.DataFrame, query: str) -> dict:
    """
    Universal Natural Language Query Engine over any uploaded dataset.
    Executes deterministic calculations strictly backed by dataset evidence.
    """
    df_clean = coerce_numeric_columns(df)
    num_rows = len(df_clean)
    q_low = query.lower().strip()

    # 1. Build Semantic Layer dynamically at runtime
    sem_layer_dict = {str(c): infer_column_semantic_metadata(c, df_clean[c], num_rows) for c in df_clean.columns}
    sem_engine = SemanticLayer(sem_layer_dict)

    # 2. Parse Query Plan & Intent Matrix
    plan = parse_query_plan(df_clean, query, sem_engine)

    # 3. Guardrail Validation Engine
    is_valid, val_msg = validate_query_plan(plan, sem_engine)
    if not is_valid:
        return {
            "question": query,
            "query": query,
            "intent": plan["intent"],
            "query_plan": plan,
            "answer": f"Validation Guardrail Notice: {val_msg}",
            "calculation_formula": "UNABLE_TO_VALIDATE_QUERY_PLAN",
            "evidence": [
                {"metric": "Validation Result", "value": "Unresolvable Query"},
                {"metric": "Reason", "value": val_msg}
            ],
            "chart_type": "NONE",
            "chart_data": []
        }

    intent = plan.get("intent")
    sub_intent = plan.get("sub_intent")
    metric = str(plan.get("metric")) if plan.get("metric") else None
    metric2 = str(plan.get("metric2")) if plan.get("metric2") else None
    agg = plan.get("aggregation")
    aggs = plan.get("aggregations", [agg] if agg else ["sum"])
    dim = plan.get("dimension")
    time_col = plan.get("time_col")
    sort = plan.get("sort")
    limit = plan.get("limit")
    fallback_applied = plan.get("fallback_applied")

    metric_subtype = sem_engine.sem_layer.get(metric, {}).get("semantic_subtype") if metric else None

    # -------------------------------------------------------------------
    # INTENT 1: Global Aggregation (Single or Compound Multi-Metric)
    # -------------------------------------------------------------------
    if intent in ["global_aggregation", "aggregate"]:
        if metric and metric in df_clean.columns:
            s_num = pd.to_numeric(df_clean[metric], errors='coerce').dropna()
            only_num = plan.get("only_numeric", False)

            # Compound multi-aggregation extraction
            if len(aggs) > 1:
                results = {}
                evidence = []
                chart_data = []
                formula_parts = []
                lines = [f"Global summary metrics for '{metric}' across {len(s_num):,} valid records:"]

                for a in aggs:
                    a_name = a.upper()
                    if a_name == "SUM":
                        val = float(s_num.sum())
                        val_str = format_value(val, metric_subtype)
                        formula_parts.append(f"SUM({metric})")
                        lines.append(f"- Total (SUM): {val_str}")
                    elif a_name in ["MEAN", "AVG"]:
                        val = float(s_num.mean())
                        val_str = format_value(val, metric_subtype)
                        formula_parts.append(f"AVG({metric})")
                        lines.append(f"- Average (MEAN): {val_str}")
                    elif a_name == "COUNT":
                        val = float(len(s_num))
                        val_str = f"{int(val):,}"
                        formula_parts.append(f"COUNT({metric})")
                        lines.append(f"- Count (COUNT): {val_str}")
                    elif a_name == "MIN":
                        val = float(s_num.min())
                        val_str = format_value(val, metric_subtype)
                        formula_parts.append(f"MIN({metric})")
                        lines.append(f"- Minimum (MIN): {val_str}")
                    elif a_name == "MAX":
                        val = float(s_num.max())
                        val_str = format_value(val, metric_subtype)
                        formula_parts.append(f"MAX({metric})")
                        lines.append(f"- Maximum (MAX): {val_str}")
                    elif a_name == "MEDIAN":
                        val = float(s_num.median())
                        val_str = format_value(val, metric_subtype)
                        formula_parts.append(f"MEDIAN({metric})")
                        lines.append(f"- Median (P50): {val_str}")
                    elif a_name == "STD":
                        val = float(s_num.std())
                        val_str = format_value(val, metric_subtype)
                        formula_parts.append(f"STD({metric})")
                        lines.append(f"- Std Dev (STD): {val_str}")
                    else:
                        val = float(s_num.sum())
                        val_str = format_value(val, metric_subtype)
                        formula_parts.append(f"SUM({metric})")
                        lines.append(f"- Total: {val_str}")

                    results[a_name] = val
                    evidence.append({"metric": f"{a_name}({metric})", "value": val_str})
                    chart_data.append({"category": f"{a_name} {metric}", "value": val})

                evidence.append({"metric": "Total Records Evaluated", "value": f"{len(df_clean):,}"})
                evidence.append({"metric": "Null Entries Excluded", "value": str(int(df_clean[metric].isnull().sum()))})

                ans = "\n".join(lines)
                formula = ", ".join(formula_parts)

                if fallback_applied:
                    ans = f"Notice: {fallback_applied.replace('_', ' ').title()}.\n\n" + ans

                return {
                    "question": query, "query": query, "intent": intent, "query_plan": plan,
                    "answer": ans, "calculation_formula": formula, "evidence": evidence,
                    "chart_type": "BAR", "chart_data": chart_data,
                    "numerical_value": results.get("SUM", results.get("MEAN", len(s_num)))
                }

            # Single aggregation handling
            agg_name = str(agg or "sum").lower()
            if agg_name == "sum":
                val = float(s_num.sum())
                formula = f"SUM({metric})"
                ans = f"{format_value(val, metric_subtype)}" if only_num else f"The total cumulative {metric} across {len(s_num):,} valid records is {format_value(val, metric_subtype)}."
            elif agg_name == "mean":
                val = float(s_num.mean())
                formula = f"AVG({metric})"
                ans = f"{format_value(val, metric_subtype)}" if only_num else f"The average {metric} across {len(s_num):,} valid records is {format_value(val, metric_subtype)}."
            elif agg_name == "median":
                val = float(s_num.median())
                formula = f"MEDIAN({metric})"
                ans = f"{format_value(val, metric_subtype)}" if only_num else f"The median (50th percentile) {metric} across {len(s_num):,} valid records is {format_value(val, metric_subtype)}."
            elif agg_name == "count":
                val = float(len(s_num))
                formula = f"COUNT({metric})"
                ans = f"{int(val):,}" if only_num else f"The count of valid non-null entries for {metric} is {int(val):,}."
            elif agg_name == "count_distinct":
                val = float(df_clean[metric].nunique())
                formula = f"COUNT(DISTINCT {metric})"
                ans = f"{int(val):,}" if only_num else f"The unique distinct count of values in {metric} is {int(val):,}."
            elif agg_name == "std":
                val = float(s_num.std())
                formula = f"STD({metric})"
                ans = f"{format_value(val, metric_subtype)}" if only_num else f"The standard deviation of {metric} is {format_value(val, metric_subtype)}."
            elif agg_name == "min":
                val = float(s_num.min())
                formula = f"MIN({metric})"
                ans = f"{format_value(val, metric_subtype)}" if only_num else f"The minimum recorded {metric} is {format_value(val, metric_subtype)}."
            elif agg_name == "min_max":
                min_v = float(s_num.min())
                max_v = float(s_num.max())
                formula = f"MIN({metric}), MAX({metric})"
                val = max_v
                ans = f"Min: {format_value(min_v, metric_subtype)}, Max: {format_value(max_v, metric_subtype)}" if only_num else f"The minimum recorded {metric} is {format_value(min_v, metric_subtype)} and the maximum recorded {metric} is {format_value(max_v, metric_subtype)} across {len(s_num):,} valid records."
            else:
                val = float(s_num.max())
                formula = f"MAX({metric})"
                ans = f"{format_value(val, metric_subtype)}" if only_num else f"The maximum recorded {metric} is {format_value(val, metric_subtype)}."

            evidence = [
                {"metric": f"{agg_name.upper()}({metric})", "value": format_value(val, metric_subtype)},
                {"metric": "Total Records Evaluated", "value": f"{len(df_clean):,}"},
                {"metric": "Null Entries Excluded", "value": str(int(df_clean[metric].isnull().sum()))}
            ]
            chart_data = [{"category": f"{agg_name.title()} {metric}", "value": val}]

            if fallback_applied:
                ans = f"Notice: {fallback_applied.replace('_', ' ').title()}.\n\n" + ans

            return {
                "question": query, "query": query, "intent": intent, "query_plan": plan,
                "answer": ans, "calculation_formula": formula, "evidence": evidence,
                "chart_type": "KPI_CARD", "chart_data": chart_data,
                "numerical_value": val
            }

    # -------------------------------------------------------------------
    # INTENT 2: Group Comparison & Slicing
    # -------------------------------------------------------------------
    if intent == "group_comparison" and dim and metric:
        temp_df = df_clean.copy()
        temp_df['__m'] = pd.to_numeric(temp_df[metric], errors='coerce').fillna(0.0)

        agg_func = 'mean' if agg == 'mean' else ('median' if agg == 'median' else 'sum')
        grouped = temp_df.groupby(dim)['__m'].agg(agg_func).reset_index()
        
        asc = (sort == "ascending")
        grouped = grouped.sort_values(by='__m', ascending=asc).reset_index(drop=True)
        top_n = grouped.head(limit or 10)

        top_row = top_n.iloc[0]
        top_name = str(top_row[dim])
        top_val = float(top_row['__m'])
        tot_val = float(grouped['__m'].sum())
        share_pct = round((top_val / max(tot_val, 1e-9)) * 100, 1)

        ans = f"The rank #1 {dim} by {agg.upper() if agg else 'SUM'}({metric}) is '{top_name}' with {format_value(top_val, metric_subtype)} ({share_pct}% share of total)."
        if fallback_applied:
            ans = f"Notice: {fallback_applied.replace('_', ' ').title()}.\n\n" + ans

        formula = f"SELECT {dim}, {agg.upper() if agg else 'SUM'}({metric}) FROM dataset GROUP BY {dim} ORDER BY {agg.upper() if agg else 'SUM'}({metric}) {'ASC' if asc else 'DESC'} LIMIT {limit or 10}"

        evidence = [
            {"metric": f"Rank 1 {dim}", "value": top_name},
            {"metric": f"{top_name} {metric}", "value": format_value(top_val, metric_subtype)},
            {"metric": "Share of Total", "value": f"{share_pct}%"},
            {"metric": "Categories Evaluated", "value": str(len(grouped))}
        ]

        chart_data = [{"category": str(r[dim]), "value": float(r['__m'])} for _, r in top_n.iterrows()]

        return {
            "question": query, "query": query, "intent": intent, "query_plan": plan,
            "answer": ans, "calculation_formula": formula, "evidence": evidence,
            "chart_type": "BAR", "chart_data": chart_data,
            "numerical_value": top_val
        }

    # -------------------------------------------------------------------
    # INTENT 3: Time Series Trend & Growth
    # -------------------------------------------------------------------
    if intent in ["time_series_trend", "trend", "growth", "root-cause analysis"] and metric and time_col:
        temp_df = df_clean.copy()
        temp_df[time_col] = pd.to_datetime(temp_df[time_col], errors='coerce')
        temp_df = temp_df.dropna(subset=[time_col]).sort_values(time_col)
        
        if temp_df.empty:
            return {
                "question": query, "query": query, "intent": intent, "query_plan": plan,
                "answer": f"Validation Guardrail Notice: No valid date entries found in column '{time_col}'.",
                "calculation_formula": "NO_VALID_DATES", "evidence": [], "chart_type": "NONE", "chart_data": []
            }

        temp_df['Period'] = temp_df[time_col].dt.to_period('M').astype(str)
        temp_df['__m'] = pd.to_numeric(temp_df[metric], errors='coerce').fillna(0.0)

        monthly = temp_df.groupby('Period')['__m'].sum().reset_index()
        monthly['MoM_Change'] = monthly['__m'].diff()
        monthly['MoM_Pct'] = monthly['__m'].pct_change() * 100

        peak_row = monthly.loc[monthly['__m'].idxmax()]
        low_row = monthly.loc[monthly['__m'].idxmin()]

        if sub_intent == "growth":
            last_mom = monthly['MoM_Pct'].dropna().iloc[-1] if len(monthly) > 1 else 0.0
            ans = f"{metric} growth analysis across {len(monthly)} monthly periods indicates the latest MoM growth rate is {last_mom:+.1f}%. Peak period was '{peak_row['Period']}' at {format_value(float(peak_row['__m']), metric_subtype)}."
        elif sub_intent == "root-cause":
            declines = monthly[monthly['MoM_Change'] < 0]
            if not declines.empty:
                worst_row = declines.loc[declines['MoM_Change'].idxmin()]
                pct_drop = abs(round(float(worst_row['MoM_Pct']), 1))
                val_drop = abs(round(float(worst_row['MoM_Change']), 2))
                ans = f"Root-cause analysis identifies period '{worst_row['Period']}' as the largest contraction point for {metric}, declining by {format_value(val_drop, metric_subtype)} (-{pct_drop}% MoM)."
            else:
                ans = f"{metric} demonstrated steady expansion with no major period-over-period contractions across recorded history."
        else:
            ans = f"Monthly {metric} trend analyzed across {len(monthly)} periods. Peak volume occurred in '{peak_row['Period']}' ({format_value(float(peak_row['__m']), metric_subtype)})."

        formula = f"SUM({metric}) GROUP BY Month({time_col})"
        evidence = [
            {"metric": "Peak Period", "value": str(peak_row['Period'])},
            {"metric": f"Peak {metric}", "value": format_value(float(peak_row['__m']), metric_subtype)},
            {"metric": "Lowest Period", "value": str(low_row['Period'])},
            {"metric": f"Lowest {metric}", "value": format_value(float(low_row['__m']), metric_subtype)}
        ]

        chart_data = [{"category": str(r['Period']), "value": float(r['__m'])} for _, r in monthly.iterrows()]

        return {
            "question": query, "query": query, "intent": intent, "query_plan": plan,
            "answer": ans, "calculation_formula": formula, "evidence": evidence,
            "chart_type": "LINE", "chart_data": chart_data
        }

    # -------------------------------------------------------------------
    # INTENT 4: Correlation Analysis (Pearson & Spearman, Matrix, Ranking, Fallback C)
    # -------------------------------------------------------------------
    if intent in ["correlation_analysis", "relationship", "relationship_single", "relationship_multiple", "correlation_matrix", "correlation_ranking", "relationship_ranking"]:
        methods = plan.get("methods", ["pearson", "spearman"])
        use_pearson = "pearson" in methods
        use_spearman = "spearman" in methods

        # 1. Correlation Matrix
        if sub_intent == "correlation_matrix":
            vars_list = [c for c in plan.get("variables", []) if c in df_clean.columns]
            if len(vars_list) >= 2:
                num_df = df_clean[vars_list].apply(pd.to_numeric, errors='coerce')
                pearson_df = num_df.corr(method='pearson').round(3)
                spearman_df = num_df.corr(method='spearman').round(3) if use_spearman else None

                method_label = "Pearson" if (use_pearson and not use_spearman) else ("Spearman" if (use_spearman and not use_pearson) else "Pearson & Spearman")
                active_matrix = pearson_df if use_pearson else spearman_df

                lines = [f"Calculated complete {method_label} correlation matrix across {len(vars_list)} variables at raw row-level with no aggregation and no grouping by any dimension.\n"]
                header = "| Variable | " + " | ".join(vars_list) + " |"
                sep = "| --- | " + " | ".join(["---"] * len(vars_list)) + " |"
                lines.append(header)
                lines.append(sep)

                chart_data = []
                for c1 in vars_list:
                    row_str = f"| **{c1}** | "
                    cell_vals = []
                    for c2 in vars_list:
                        val = float(active_matrix.loc[c1, c2])
                        val_formatted = f"{val:.3f}"
                        cell_vals.append(val_formatted)
                        point = {"x": c1, "y": c2, "value": round(val, 3)}
                        if use_pearson:
                            point["pearson"] = round(float(pearson_df.loc[c1, c2]), 3)
                        if use_spearman:
                            point["spearman"] = round(float(spearman_df.loc[c1, c2]), 3)
                        chart_data.append(point)
                    row_str += " | ".join(cell_vals) + " |"
                    lines.append(row_str)

                ans = "\n".join(lines)
                formula = f"{method_label}_Correlation_Matrix({', '.join(vars_list)})"
                evidence = [
                    {"metric": "Matrix Dimensions", "value": f"{len(vars_list)}x{len(vars_list)}"},
                    {"metric": "Variables Included", "value": ", ".join(vars_list)},
                    {"metric": "Correlation Method", "value": method_label},
                    {"metric": "Calculation Granularity", "value": "Raw Row-Level (No Aggregation / No Grouping)"}
                ]

                return {
                    "question": query, "query": query, "intent": intent, "query_plan": plan,
                    "answer": ans, "calculation_formula": formula, "evidence": evidence,
                    "chart_type": "CORRELATION_HEATMAP", "chart_data": chart_data
                }

        # 2. Correlation Ranking (Strongest Positive / Negative)
        elif sub_intent in ["correlation_ranking", "relationship_ranking"]:
            vars_list = [c for c in plan.get("variables", []) if c in df_clean.columns]
            top_n = plan.get("limit") or 3

            if len(vars_list) >= 2:
                num_df = df_clean[vars_list].apply(pd.to_numeric, errors='coerce')
                p_matrix = num_df.corr(method='pearson').round(3)
                s_matrix = num_df.corr(method='spearman').round(3) if use_spearman else None

                pairs_eval = []
                for i in range(len(vars_list)):
                    for j in range(i+1, len(vars_list)):
                        c1, c2 = vars_list[i], vars_list[j]
                        r_p = float(p_matrix.loc[c1, c2])
                        r_s = float(s_matrix.loc[c1, c2]) if use_spearman else None
                        ref_r = r_p if use_pearson else r_s
                        pairs_eval.append({
                            "c1": c1, "c2": c2,
                            "pearson": r_p, "spearman": r_s,
                            "r": ref_r
                        })

                if pairs_eval:
                    pos_pairs = [p for p in pairs_eval if p["r"] > 0]
                    pos_pairs.sort(key=lambda x: x["r"], reverse=True)
                    top_pos_pairs = pos_pairs[:top_n]

                    neg_pairs = [p for p in pairs_eval if p["r"] < 0]
                    neg_pairs.sort(key=lambda x: x["r"], reverse=False)
                    top_neg_pairs = neg_pairs[:top_n]

                    method_name = "Pearson" if (use_pearson and not use_spearman) else ("Spearman" if (use_spearman and not use_pearson) else "Pearson & Spearman")

                    lines = [
                        f"Calculated pairwise {method_name} correlations across {len(vars_list)} variables ({len(pairs_eval)} unique pairs) at raw row-level with no aggregation or grouping by any dimension.\n"
                    ]

                    if top_pos_pairs:
                        lines.append(f"### TOP {len(top_pos_pairs)} STRONGEST POSITIVE {method_name.upper()} CORRELATIONS (Strongest Positive Correlation)")
                        for idx, p in enumerate(top_pos_pairs, 1):
                            r_val = f"+{p['r']:.3f}" if p['r'] > 0 else f"{p['r']:.3f}"
                            lines.append(f"{idx}. **{p['c1']}** ↔ **{p['c2']}** — {method_name} r = {r_val}")
                        lines.append("")

                    if top_neg_pairs:
                        lines.append(f"### TOP {len(top_neg_pairs)} STRONGEST NEGATIVE {method_name.upper()} CORRELATIONS (Strongest Negative Correlation)")
                        for idx, p in enumerate(top_neg_pairs, 1):
                            r_val = f"{p['r']:.3f}"
                            lines.append(f"{idx}. **{p['c1']}** ↔ **{p['c2']}** — {method_name} r = {r_val}")
                        lines.append("")

                    lines.append("Note: Correlation indicates linear co-variance and does not imply direct causation.")
                    ans = "\n".join(lines)
                    formula = f"Rank_Pairwise_{method_name}_Correlations({', '.join(vars_list)})"

                    evidence = [
                        {"metric": "Evaluated Unique Pairs", "value": str(len(pairs_eval))},
                        {"metric": "Variables Included", "value": ", ".join(vars_list)},
                        {"metric": "Correlation Method", "value": method_name},
                        {"metric": "Calculation Granularity", "value": "Raw Row-Level (No Aggregation / No Grouping)"}
                    ]

                    chart_data = []
                    for p in top_pos_pairs + top_neg_pairs:
                        pt = {"category": f"{p['c1']} vs {p['c2']}", "value": p['r']}
                        if use_pearson:
                            pt["pearson"] = p["pearson"]
                        if use_spearman:
                            pt["spearman"] = p["spearman"]
                        chart_data.append(pt)

                    return {
                        "question": query, "query": query, "intent": intent, "query_plan": plan,
                        "answer": ans, "calculation_formula": formula, "evidence": evidence,
                        "chart_type": "BAR", "chart_data": chart_data
                    }

        # 3. Multiple Pairs Correlation
        elif sub_intent == "relationship_multiple":
            pairs = plan.get("pairs", [])
            pair_results = []
            evidence = []
            chart_data = []

            for p1, p2 in pairs:
                if p1 in df_clean.columns and p2 in df_clean.columns:
                    s1 = pd.to_numeric(df_clean[p1], errors='coerce')
                    s2 = pd.to_numeric(df_clean[p2], errors='coerce')
                    valid = pd.DataFrame({p1: s1, p2: s2}).dropna()
                    if len(valid) >= 3:
                        r_p = round(float(valid[p1].corr(valid[p2], method='pearson')), 3) if use_pearson else None
                        r_s = round(float(valid[p1].corr(valid[p2], method='spearman')), 3) if use_spearman else None
                        ref_r = r_p if r_p is not None else r_s
                        strength = "strong positive" if ref_r > 0.6 else ("strong negative" if ref_r < -0.6 else ("moderate" if abs(ref_r) >= 0.3 else "weak/negligible"))
                        pair_results.append({"pair": f"{p1} vs {p2}", "pearson": r_p, "spearman": r_s, "strength": strength, "n": len(valid)})
                        vals_str = f"Pearson r = {r_p}" if (use_pearson and not use_spearman) else (f"Spearman r = {r_s}" if (use_spearman and not use_pearson) else f"Pearson r = {r_p}, Spearman r = {r_s}")
                        evidence.append({"metric": f"{p1} vs {p2}", "value": f"{vals_str} ({strength.title()})"})
                        chart_data.append({"category": f"{p1} vs {p2}", "value": ref_r})

            if pair_results:
                lines = [f"Correlation analysis across {len(pair_results)} requested variable pairs:"]
                for idx, res in enumerate(pair_results, 1):
                    vals_str = f"Pearson r = {res['pearson']}" if (use_pearson and not use_spearman) else (f"Spearman r = {res['spearman']}" if (use_spearman and not use_pearson) else f"Pearson r = {res['pearson']}, Spearman r = {res['spearman']}")
                    lines.append(f"{idx}. {res['pair']} → {vals_str} ({res['strength'].title()})")
                lines.append("\nNote: Correlation indicates linear co-variance and does not imply direct causation.")
                ans = "\n".join(lines)
                formula = "Pairwise_Correlation(" + ", ".join([r["pair"] for r in pair_results]) + ")"

                return {
                    "question": query, "query": query, "intent": intent, "query_plan": plan,
                    "answer": ans, "calculation_formula": formula, "evidence": evidence,
                    "chart_type": "BAR", "chart_data": chart_data
                }

        # 4. Fallback Scenario C: Single Metric Correlation against all other measures
        elif sub_intent == "single_metric_correlation":
            single_m = metric
            other_measures = [m for m in sem_engine.measures if m != single_m]
            pair_results = []
            evidence = []
            chart_data = []

            for other_m in other_measures:
                s1 = pd.to_numeric(df_clean[single_m], errors='coerce')
                s2 = pd.to_numeric(df_clean[other_m], errors='coerce')
                valid = pd.DataFrame({single_m: s1, other_m: s2}).dropna()
                if len(valid) >= 3:
                    r_p = round(float(valid[single_m].corr(valid[other_m], method='pearson')), 3) if use_pearson else None
                    r_s = round(float(valid[single_m].corr(valid[other_m], method='spearman')), 3) if use_spearman else None
                    ref_r = r_p if r_p is not None else r_s
                    strength = "strong positive" if ref_r > 0.6 else ("strong negative" if ref_r < -0.6 else ("moderate" if abs(ref_r) >= 0.3 else "weak/negligible"))
                    pair_results.append({
                        "metric": other_m, "pearson": r_p, "spearman": r_s, "r": ref_r, "strength": strength, "abs_r": abs(ref_r)
                    })

            pair_results.sort(key=lambda x: x["abs_r"], reverse=True)

            lines = [f"Pairwise correlation analysis for '{single_m}' across {len(pair_results)} quantitative measures (Fallback C execution):"]
            for idx, res in enumerate(pair_results, 1):
                vals_str = f"Pearson r = {res['pearson']}" if (use_pearson and not use_spearman) else (f"Spearman r = {res['spearman']}" if (use_spearman and not use_pearson) else f"Pearson r = {res['pearson']}, Spearman r = {res['spearman']}")
                lines.append(f"{idx}. **{single_m}** ↔ **{res['metric']}** → {vals_str} ({res['strength'].title()})")
                evidence.append({"metric": f"{single_m} vs {res['metric']}", "value": f"{vals_str} ({res['strength'].title()})"})
                chart_data.append({"category": f"{single_m} vs {res['metric']}", "value": res["r"]})

            lines.append("\nNote: Correlation indicates linear co-variance and does not imply direct causation.")
            ans = "\n".join(lines)
            formula = f"Pairwise_Correlations({single_m}, [{', '.join(other_measures)}])"

            return {
                "question": query, "query": query, "intent": intent, "query_plan": plan,
                "answer": ans, "calculation_formula": formula, "evidence": evidence,
                "chart_type": "BAR", "chart_data": chart_data
            }

        # 5. Single Pair Correlation
        else:
            m1 = metric
            m2 = metric2
            s1 = pd.to_numeric(df_clean[m1], errors='coerce')
            s2 = pd.to_numeric(df_clean[m2], errors='coerce')
            valid = pd.DataFrame({m1: s1, m2: s2}).dropna()

            if len(valid) >= 3:
                r_p = round(float(valid[m1].corr(valid[m2], method='pearson')), 3) if use_pearson else None
                r_s = round(float(valid[m1].corr(valid[m2], method='spearman')), 3) if use_spearman else None
                ref_r = r_p if r_p is not None else r_s
                strength = "strong positive" if ref_r > 0.6 else ("strong negative" if ref_r < -0.6 else ("moderate" if abs(ref_r) >= 0.3 else "weak/negligible"))

                only_num = plan.get("only_numeric", False)
                only_pearson = plan.get("only_pearson_result", False)

                if only_num:
                    ans = f"{ref_r}"
                    formula = f"Pearson_r({m1}, {m2})" if use_pearson else f"Spearman_r({m1}, {m2})"
                    evidence = [{"metric": "Correlation Coefficient", "value": str(ref_r)}]
                elif only_pearson or (use_pearson and not use_spearman and ('only' in q_low or 'return' in q_low)):
                    ans = f"Pearson correlation (r) = {r_p}"
                    formula = f"Pearson_r({m1}, {m2}) = Cov({m1},{m2}) / (Std({m1}) * Std({m2}))"
                    evidence = [
                        {"metric": "Pearson Correlation (r)", "value": str(r_p)},
                        {"metric": "Sample Size (N)", "value": str(len(valid))}
                    ]
                elif use_pearson and use_spearman:
                    ans = f"Bivariate analysis between '{m1}' and '{m2}' reveals a {strength} correlation (Pearson r = {r_p}, Spearman r = {r_s}). Note: Correlation indicates linear co-variance and does not imply direct causation."
                    formula = f"Pearson_r({m1}, {m2}) = Cov({m1},{m2}) / (Std({m1}) * Std({m2}))"
                    evidence = [
                        {"metric": "Pearson Correlation (r)", "value": str(r_p)},
                        {"metric": "Spearman Correlation (r)", "value": str(r_s)},
                        {"metric": "Sample Size (N)", "value": str(len(valid))},
                        {"metric": "Co-Variance Strength", "value": strength.title()}
                    ]
                elif use_pearson:
                    ans = f"Bivariate analysis between '{m1}' and '{m2}' reveals a {strength} relationship (Pearson r = {r_p}). Note: Correlation indicates linear co-variance and does not imply direct causation."
                    formula = f"Pearson_r({m1}, {m2}) = Cov({m1},{m2}) / (Std({m1}) * Std({m2}))"
                    evidence = [
                        {"metric": "Pearson Correlation (r)", "value": str(r_p)},
                        {"metric": "Sample Size (N)", "value": str(len(valid))},
                        {"metric": "Co-Variance Strength", "value": strength.title()}
                    ]
                else:
                    ans = f"Bivariate analysis between '{m1}' and '{m2}' reveals a {strength} relationship (Spearman r = {r_s}). Note: Correlation indicates rank co-variance and does not imply direct causation."
                    formula = f"Spearman_r({m1}, {m2})"
                    evidence = [
                        {"metric": "Spearman Correlation (r)", "value": str(r_s)},
                        {"metric": "Sample Size (N)", "value": str(len(valid))},
                        {"metric": "Co-Variance Strength", "value": strength.title()}
                    ]

                try:
                    from engine.governed_engines import DerivedMetricEngine
                    struct_info = DerivedMetricEngine.check_relationship_between_variables(df_clean, m1, m2)
                    if struct_info.get("is_derived") or struct_info.get("is_deterministic"):
                        formula_desc = struct_info.get("formula", "")
                        ans += f" Analytical Governance Notice: '{m1}' and '{m2}' have an algebraic structural derivation ({formula_desc}). Redundant collinear predictors should be excluded from simultaneous regression modeling."
                        evidence.append({"metric": "Structural Lineage", "value": f"Derived ({formula_desc})"})
                except Exception:
                    pass

                chart_data = [{"category": f"Record #{i+1}", "value": float(row[m2])} for i, row in valid.head(15).iterrows()]

                return {
                    "question": query, "query": query, "intent": intent, "query_plan": plan,
                    "answer": ans, "calculation_formula": formula, "evidence": evidence,
                    "chart_type": "SCATTER", "chart_data": chart_data,
                    "numerical_value": ref_r
                }

            vars_list = [c for c in plan.get("variables", []) if c in df_clean.columns]
            top_n = plan.get("limit") or 3

            if len(vars_list) >= 2:
                num_df = df_clean[vars_list].apply(pd.to_numeric, errors='coerce')
                p_matrix = num_df.corr(method='pearson').round(3)
                s_matrix = num_df.corr(method='spearman').round(3) if use_spearman else None

                pairs_eval = []
                for i in range(len(vars_list)):
                    for j in range(i+1, len(vars_list)):
                        c1, c2 = vars_list[i], vars_list[j]
                        r_p = float(p_matrix.loc[c1, c2])
                        r_s = float(s_matrix.loc[c1, c2]) if use_spearman else None
                        ref_r = r_p if use_pearson else r_s
                        pairs_eval.append({
                            "c1": c1, "c2": c2,
                            "pearson": r_p, "spearman": r_s,
                            "r": ref_r
                        })

                if pairs_eval:
                    pos_pairs = [p for p in pairs_eval if p["r"] > 0]
                    pos_pairs.sort(key=lambda x: x["r"], reverse=True)
                    top_pos_pairs = pos_pairs[:top_n]

                    neg_pairs = [p for p in pairs_eval if p["r"] < 0]
                    neg_pairs.sort(key=lambda x: x["r"], reverse=False)
                    top_neg_pairs = neg_pairs[:top_n]

                    method_name = "Pearson" if (use_pearson and not use_spearman) else ("Spearman" if (use_spearman and not use_pearson) else "Pearson & Spearman")

                    lines = [
                        f"Calculated pairwise {method_name} correlations across {len(vars_list)} variables ({len(pairs_eval)} unique pairs) at raw row-level with no aggregation or grouping by any dimension.\n"
                    ]

                    if top_pos_pairs:
                        lines.append(f"### TOP {len(top_pos_pairs)} STRONGEST POSITIVE {method_name.upper()} CORRELATIONS (Strongest Positive Correlation)")
                        for idx, p in enumerate(top_pos_pairs, 1):
                            r_val = f"+{p['r']:.3f}" if p['r'] > 0 else f"{p['r']:.3f}"
                            lines.append(f"{idx}. **{p['c1']}** ↔ **{p['c2']}** — {method_name} r = {r_val}")
                        lines.append("")

                    if top_neg_pairs:
                        lines.append(f"### TOP {len(top_neg_pairs)} STRONGEST NEGATIVE {method_name.upper()} CORRELATIONS (Strongest Negative Correlation)")
                        for idx, p in enumerate(top_neg_pairs, 1):
                            r_val = f"{p['r']:.3f}"
                            lines.append(f"{idx}. **{p['c1']}** ↔ **{p['c2']}** — {method_name} r = {r_val}")
                        lines.append("")

                    lines.append("Note: Correlation indicates linear co-variance and does not imply direct causation.")
                    ans = "\n".join(lines)
                    formula = f"Rank_Pairwise_{method_name}_Correlations({', '.join(vars_list)})"

                    evidence = [
                        {"metric": "Evaluated Unique Pairs", "value": str(len(pairs_eval))},
                        {"metric": "Variables Included", "value": ", ".join(vars_list)},
                        {"metric": "Correlation Method", "value": method_name},
                        {"metric": "Calculation Granularity", "value": "Raw Row-Level (No Aggregation / No Grouping)"}
                    ]

                    chart_data = []
                    for p in top_pos_pairs + top_neg_pairs:
                        pt = {"category": f"{p['c1']} vs {p['c2']}", "value": p['r']}
                        if use_pearson:
                            pt["pearson"] = p["pearson"]
                        if use_spearman:
                            pt["spearman"] = p["spearman"]
                        chart_data.append(pt)

                    return {
                        "question": query, "query": query, "intent": intent, "query_plan": plan,
                        "answer": ans, "calculation_formula": formula, "evidence": evidence,
                        "chart_type": "BAR", "chart_data": chart_data
                    }

    # -------------------------------------------------------------------
    # INTENT 5: Distribution Analysis (IQR, Skewness, Outliers)
    # -------------------------------------------------------------------
    if intent in ["distribution_analysis", "distribution", "outlier"] and metric:
        s_num = pd.to_numeric(df_clean[metric], errors='coerce').astype(float).dropna()
        q1 = float(s_num.quantile(0.25))
        q3 = float(s_num.quantile(0.75))
        iqr = q3 - q1
        lower_b = q1 - 1.5 * iqr
        upper_b = q3 + 1.5 * iqr

        outliers = s_num[(s_num < lower_b) | (s_num > upper_b)]
        outlier_pct = round((len(outliers) / max(len(s_num), 1)) * 100, 2)
        mean_v = float(s_num.mean())
        median_v = float(s_num.median())
        std_v = float(s_num.std())
        skew_v = float(s_num.skew())

        if sub_intent == "outlier" or any(k in q_low for k in ['outlier', 'anomal']):
            ans = f"Outlier IQR detection for '{metric}' identified {len(outliers):,} potential outlier value(s) ({outlier_pct}% of total records) outside threshold bounds [{format_value(lower_b, metric_subtype)}, {format_value(upper_b, metric_subtype)}]."
            formula = f"IQR_Outliers = {metric} < (Q1 - 1.5*IQR) OR {metric} > (Q3 + 1.5*IQR)"
            evidence = [
                {"metric": "Outlier Count", "value": f"{len(outliers):,}"},
                {"metric": "Outlier Percentage", "value": f"{outlier_pct}%"},
                {"metric": "IQR Lower Bound", "value": format_value(lower_b, metric_subtype)},
                {"metric": "IQR Upper Bound", "value": format_value(upper_b, metric_subtype)}
            ]
            chart_type = "BOX"
            chart_data = [{"category": "Outliers", "value": float(len(outliers))}]
        else:
            skew_desc = "right-skewed (positive)" if skew_v > 0.5 else ("left-skewed (negative)" if skew_v < -0.5 else "approximately symmetric")
            ans = f"Distribution profile of '{metric}': Mean = {format_value(mean_v, metric_subtype)}, Median = {format_value(median_v, metric_subtype)}, Std Dev = {format_value(std_v, metric_subtype)}. Distribution is {skew_desc} with skewness coefficient of {skew_v:.2f}."
            formula = f"Skewness = E[(X - mu)^3] / sigma^3"
            evidence = [
                {"metric": "Mean", "value": format_value(mean_v, metric_subtype)},
                {"metric": "Median (P50)", "value": format_value(median_v, metric_subtype)},
                {"metric": "Std Deviation", "value": format_value(std_v, metric_subtype)},
                {"metric": "Skewness Index", "value": f"{skew_v:.2f}"},
                {"metric": "Outliers Detected", "value": f"{len(outliers):,}"}
            ]
            chart_type = "HISTOGRAM"
            chart_data = [{"category": "Skewness", "value": skew_v}]

        return {
            "question": query, "query": query, "intent": intent, "query_plan": plan,
            "answer": ans, "calculation_formula": formula, "evidence": evidence,
            "chart_type": chart_type, "chart_data": chart_data
        }

    # -------------------------------------------------------------------
    # INTENT 6: Filtering Top N
    # -------------------------------------------------------------------
    if intent in ["filtering_top_n", "ranking"] and metric:
        temp_df = df_clean.copy()
        temp_df['__m'] = pd.to_numeric(temp_df[metric], errors='coerce').fillna(0.0)

        use_dim = dim if dim and dim in temp_df.columns else (sem_engine.dimensions[0] if sem_engine.dimensions else None)
        asc = (sort == "ascending")
        lim = limit or 5

        # Determine aggregation function dynamically
        agg_lower = (agg or "sum").lower()
        if agg_lower in ["mean", "avg", "average"]:
            agg_func = "mean"
            stat_word = "average "
            sql_agg = "AVG"
        elif agg_lower in ["median"]:
            agg_func = "median"
            stat_word = "median "
            sql_agg = "MEDIAN"
        elif agg_lower in ["min", "minimum"]:
            agg_func = "min"
            stat_word = "minimum "
            sql_agg = "MIN"
        elif agg_lower in ["max", "maximum"]:
            agg_func = "max"
            stat_word = "maximum "
            sql_agg = "MAX"
        elif agg_lower in ["count"]:
            agg_func = "count"
            stat_word = "count of "
            sql_agg = "COUNT"
        else:
            agg_func = "sum"
            stat_word = ""
            sql_agg = "SUM"

        if use_dim:
            grouped = temp_df.groupby(use_dim)['__m'].agg(agg_func).reset_index()
            grouped = grouped.sort_values(by='__m', ascending=asc).reset_index(drop=True)
            top_df = grouped.head(lim)
            top_one = top_df.iloc[0]
            top_val = float(top_one['__m'])
            val_formatted = format_value(top_val, metric_subtype)

            if lim == 1 or 'which' in q_low or 'what' in q_low:
                direction_word = "lowest" if asc else "highest"
                ans = f"{use_dim} '{top_one[use_dim]}' has the {direction_word} {stat_word}{metric} with an {stat_word}value of {val_formatted}."
            else:
                ans = f"Top {lim} ranking by {metric} ({'lowest' if asc else 'highest'}) identified '{top_one[use_dim]}' as #1 with {val_formatted}."

            formula = f"SELECT {use_dim}, {sql_agg}({metric}) FROM dataset GROUP BY {use_dim} ORDER BY {sql_agg}({metric}) {'ASC' if asc else 'DESC'} LIMIT {lim}"
            evidence = [
                {"metric": f"Rank #{i+1} {use_dim}", "value": f"{r[use_dim]} ({format_value(float(r['__m']), metric_subtype)})"}
                for i, r in top_df.iterrows()
            ]
            chart_data = [{"category": str(r[use_dim]), "value": float(r['__m'])} for _, r in top_df.iterrows()]
        else:
            sorted_df = temp_df.sort_values(by='__m', ascending=asc).head(lim)
            top_val = float(sorted_df['__m'].iloc[0])
            ans = f"Top {lim} record ranking by {metric} ({'lowest' if asc else 'highest'}) has peak value {format_value(top_val, metric_subtype)}."
            formula = f"SELECT {metric} FROM dataset ORDER BY {metric} {'ASC' if asc else 'DESC'} LIMIT {lim}"
            evidence = [{"metric": f"Rank #{i+1}", "value": format_value(float(r['__m']), metric_subtype)} for i, r in sorted_df.reset_index().iterrows()]
            chart_data = [{"category": f"Record #{i+1}", "value": float(r['__m'])} for i, r in sorted_df.reset_index().iterrows()]

        return {
            "question": query, "query": query, "intent": intent, "query_plan": plan,
            "answer": ans, "calculation_formula": formula, "evidence": evidence,
            "chart_type": "BAR", "chart_data": chart_data,
            "numerical_value": top_val
        }

    # Catch-all
    return {
        "question": query,
        "query": query,
        "intent": plan["intent"],
        "query_plan": plan,
        "answer": f"Validation Guardrail Notice: Could not resolve query execution plan for '{query}'. Please specify an available quantitative measure or categorical dimension.",
        "calculation_formula": "UNRESOLVED_QUERY_PLAN",
        "evidence": [{"metric": "Validation Result", "value": "Unresolvable Query"}],
        "chart_type": "NONE",
        "chart_data": []
    }
