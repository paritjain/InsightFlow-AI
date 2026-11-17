import uuid
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import Field

from engine.contracts.base import GovernedContractBase


class QueryIntent(str, Enum):
    """
    Typed, universal analytical intent taxonomy.
    Subclasses str and overrides __eq__ to support case-insensitive and
    legacy lowercase string comparisons (e.g. plan['intent'] == 'global_aggregation').
    """
    # 1. Data Retrieval & Row-Level Lookup
    ROW_LOOKUP = "ROW_LOOKUP"                       # Find specific record by identifier or entity
    FILTERED_RECORDS = "FILTERED_RECORDS"           # Retrieve subset of records satisfying WHERE predicates
    DATASET_SUMMARY = "DATASET_SUMMARY"             # High-level overview, dimensions, row counts, schema
    
    # 2. Aggregations & Metrics
    GLOBAL_AGGREGATION = "GLOBAL_AGGREGATION"       # SUM, MEAN, MEDIAN, MIN, MAX over population
    DISTINCT_COUNT = "DISTINCT_COUNT"               # COUNT(DISTINCT column)
    GROUP_COMPARISON = "GROUP_COMPARISON"           # Full breakdown across categories of a dimension
    RANKING_TOP_N = "RANKING_TOP_N"                 # TOP / BOTTOM N with ordering
    
    # 3. Statistical, Temporal & Distribution
    DISTRIBUTION_ANALYSIS = "DISTRIBUTION_ANALYSIS" # Skewness, quartiles, dispersion, shape
    OUTLIER_ANALYSIS = "OUTLIER_ANALYSIS"           # IQR fences, anomaly detection
    CORRELATION_ANALYSIS = "CORRELATION_ANALYSIS"   # Bivariate / Matrix correlation
    TIME_SERIES_TREND = "TIME_SERIES_TREND"         # Trend, seasonality, growth over time
    HYPOTHESIS_TEST = "HYPOTHESIS_TEST"             # ANOVA, Kruskal, Chi-squared
    
    # 4. Governance & Meta
    DATA_QUALITY = "DATA_QUALITY"                   # Missing values, invalid cells, completeness
    DERIVATION_LINEAGE = "DERIVATION_LINEAGE"       # Explanation of algebraic transformations
    EXPLAIN_INSIGHT = "EXPLAIN_INSIGHT"             # Deep dive into existing platform insight
    
    # 5. Clarification & Edge Cases
    AMBIGUOUS = "AMBIGUOUS"                         # Multiple conflicting schema interpretations
    UNSUPPORTED = "UNSUPPORTED"                     # Out of scope for dataset

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            s_val = self.value.lower()
            o_val = other.lower().strip()
            if s_val == o_val:
                return True
            # Support legacy keyword aliases
            aliases = {
                "global_aggregation": ["global_aggregation", "aggregate"],
                "group_comparison": ["group_comparison", "ranking"],
                "time_series_trend": ["time_series_trend", "trend", "growth", "root-cause analysis"],
                "correlation_analysis": [
                    "correlation_analysis", "relationship", "relationship_single", 
                    "relationship_multiple", "correlation_matrix", "correlation_ranking", "relationship_ranking"
                ],
                "ranking_top_n": ["ranking_top_n", "ranking", "filtering_top_n"]
            }
            if s_val in aliases and o_val in aliases[s_val]:
                return True
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.value)


class FilterOperator(str, Enum):
    """Supported filter operators for deterministic SQL & DataFrame execution."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    IN = "IN"
    NOT_IN = "NOT IN"
    LIKE = "LIKE"
    ILIKE = "ILIKE"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    BETWEEN = "BETWEEN"


class ColumnFilter(GovernedContractBase):
    """Typed conditional filter predicate applied in a query plan."""
    column: str
    operator: FilterOperator = FilterOperator.EQUALS
    value: Any = None
    secondary_value: Optional[Any] = None  # For BETWEEN operator
    is_negated: bool = False
    raw_predicate_text: Optional[str] = None


class AggregationFunction(str, Enum):
    """Supported aggregation mathematical functions."""
    SUM = "SUM"
    AVG = "AVG"
    MEAN = "MEAN"
    COUNT = "COUNT"
    COUNT_DISTINCT = "COUNT_DISTINCT"
    MIN = "MIN"
    MAX = "MAX"
    MEDIAN = "MEDIAN"
    STDDEV = "STDDEV"
    VARIANCE = "VARIANCE"


class ResolutionStatus(str, Enum):
    """Entity and column resolution states."""
    EXACT_MATCH = "EXACT_MATCH"
    SEMANTIC_MATCH = "SEMANTIC_MATCH"
    FUZZY_MATCH = "FUZZY_MATCH"
    DERIVED_MATCH = "DERIVED_MATCH"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_FOUND = "NOT_FOUND"


class ResolvedEntity(GovernedContractBase):
    """Schema-grounded entity resolution output."""
    requested_term: str
    resolved_column: Optional[str] = None
    canonical_identifier: Optional[str] = None
    semantic_role: Optional[str] = None
    measure_type: Optional[str] = None
    data_type: Optional[str] = None
    status: ResolutionStatus = ResolutionStatus.NOT_FOUND
    confidence: float = 0.0
    candidate_alternatives: List[str] = Field(default_factory=list)
    resolution_method: str = "EXACT"
    is_measure: bool = False
    is_dimension: bool = False
    is_identifier: bool = False
    is_temporal: bool = False


class ResultValidationStatus(str, Enum):
    """Standardized validation status for query plans and execution results."""
    PASS = "PASS"
    WARNING = "WARNING"
    AMBIGUOUS = "AMBIGUOUS"
    NO_MATCH = "NO_MATCH"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


class SuggestionMode(str, Enum):
    """Modes for intelligent query resolution and suggestion panel."""
    EXACT = "exact"
    AUTO_RESOLVED = "auto_resolved"
    CONFIRMATION_REQUIRED = "confirmation_required"
    CLARIFICATION_REQUIRED = "clarification_required"


class MatchSuggestionContract(GovernedContractBase):
    """Schema for individual field match suggestion."""
    column: str
    confidence: float
    confidence_pct: int
    match_type: str = "FUZZY"
    explanation: str = ""
    semantic_role: Optional[str] = None
    data_type: Optional[str] = None
    is_primary: bool = False


class QueryPlan(GovernedContractBase):
    """
    Authoritative Pydantic Contract for Analytical Query Plans.
    Represents the structured interpretation of the user query prior to execution.
    Maintains backward compatibility with legacy dictionary access.
    """
    plan_id: str = Field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:8]}")
    dataset_id: str
    dataset_version: str = "cleaned_v1"
    intent: QueryIntent
    sub_intent: Optional[str] = None
    
    # Target Entities
    target_columns: List[str] = Field(default_factory=list)
    metric: Optional[Any] = None
    metric2: Optional[Any] = None
    dimension: Optional[str] = None
    secondary_dimension: Optional[str] = None
    time_column: Optional[str] = None
    time_grain: Optional[str] = None
    
    # Aggregations & Filters
    aggregations: List[Any] = Field(default_factory=list)
    aggregation: Optional[str] = None
    filters: List[ColumnFilter] = Field(default_factory=list)
    
    # Ordering & Limits
    sort: Optional[str] = None       # "ascending" / "descending" or "ASC" / "DESC"
    limit: Optional[int] = None
    
    # Statistical & Matrix Parameters
    variables: List[str] = Field(default_factory=list)
    pairs: List[Any] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=lambda: ["pearson"])
    statistical_method: Optional[str] = None
    
    # Format & Flags
    only_numeric: bool = False
    no_spearman: bool = False
    only_pearson_result: bool = False
    granularity: str = "row-level"
    fallback_applied: Optional[str] = None
    
    # Validation & Compilation
    is_valid: bool = True
    validation_message: str = "Valid"
    compiled_sql: Optional[str] = None
    sql_parameters: Dict[str, Any] = Field(default_factory=dict)
    clarification_needed: bool = False
    clarification_prompt: Optional[str] = None
    
    # Traceability Context
    query_text: str = ""
    resolved_entities: List[ResolvedEntity] = Field(default_factory=list)

    # Intelligent Matching & Suggestion Metadata
    match_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    suggestion_mode: Optional[str] = None
    resolved_field: Optional[str] = None
    confirmed_field: Optional[str] = None
    interactive_options: List[str] = Field(default_factory=list)

    # Legacy Aliases
    @property
    def Intent(self):
        return self.intent

    @property
    def Metric(self):
        return self.metric

    @property
    def Dimension(self):
        return self.dimension

    @property
    def Aggregations(self):
        return self.aggregations

    @property
    def Aggregation(self):
        return self.aggregation

    @property
    def Metrics(self):
        if self.aggregations and self.metric:
            return [f"{self.metric} ({str(a).upper()})" for a in self.aggregations]
        return [f"{self.metric} (SUM)"] if self.metric else []


class ExecutionResult(GovernedContractBase):
    """
    Contract for raw deterministic query execution output (DuckDB or Pandas).
    """
    execution_id: str = Field(default_factory=lambda: f"exec-{uuid.uuid4().hex[:8]}")
    plan_id: str
    engine_used: str = "DuckDB"
    status: ResultValidationStatus = ResultValidationStatus.PASS
    execution_time_ms: float = 0.0
    rows_returned: int = 0
    records_evaluated: int = 0
    total_dataset_rows: int = 0
    coverage_pct: float = 100.0
    
    data_sample: List[Dict[str, Any]] = Field(default_factory=list)
    scalar_value: Optional[Any] = None
    tabular_columns: List[str] = Field(default_factory=list)
    compiled_query: Optional[str] = None
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


class QueryTrace(GovernedContractBase):
    """
    Structured execution telemetry trace tracking every query pipeline stage:
    request -> intent -> resolution -> plan -> validation -> execution -> response.
    """
    trace_id: str = Field(default_factory=lambda: f"trace-{uuid.uuid4().hex[:12]}")
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    project_id: str
    user_query: str
    stages: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    total_duration_ms: float = 0.0
    final_status: ResultValidationStatus = ResultValidationStatus.PASS

    def record_stage(self, stage_name: str, status: str, duration_ms: float = 0.0, diagnostics: Dict[str, Any] = None):
        self.stages[stage_name] = {
            "status": status,
            "duration_ms": duration_ms,
            "diagnostics": diagnostics or {}
        }


class ChatResponseContract(GovernedContractBase):
    """
    Authoritative Container for Final Chatbot Responses.
    Served via /api/project/{id}/chat and /api/project/{id}/ask.
    Provides direct compatibility with UI expectations while strictly enforcing grounded truth.
    """
    response_id: str = Field(default_factory=lambda: f"resp-{uuid.uuid4().hex[:8]}")
    project_id: str
    dataset_id: str
    dataset_version: str = "cleaned_v1"
    question: str
    query: str
    intent: Union[QueryIntent, str]
    tool_used: str = "DeterministicAnalyticalEngine"
    status: ResultValidationStatus = ResultValidationStatus.PASS
    
    # Ground-Truth Analytical Content
    answer: str
    numerical_value: Optional[Any] = None
    calculation_formula: str = ""
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    chart_spec: Optional[Dict[str, Any]] = None
    
    # Metadata & Lineage
    query_plan: Optional[Dict[str, Any]] = None
    execution_trace_id: str = ""
    warnings: List[str] = Field(default_factory=list)
    clarification_required: bool = False
    records_evaluated: int = 0
    coverage_pct: float = 100.0

    # Intelligent Match Suggestions & Confirmed Fields
    match_suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    suggestion_mode: Optional[str] = None
    resolved_field: Optional[str] = None
    confirmed_field: Optional[str] = None
    interactive_options: List[str] = Field(default_factory=list)
