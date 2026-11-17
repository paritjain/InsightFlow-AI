from typing import Any, Dict, List, Optional, Union
from engine.contracts.base import GovernedContractBase
from engine.contracts.enums import (
    SemanticRole,
    MeasureType,
    KeyRole,
    DerivationType,
    ExclusionReason
)


class ColumnSemanticContract(GovernedContractBase):
    """
    Canonical Single Source of Truth for Column Semantics & Analytical Eligibility (Prompt Section 5).
    Every column in the dataset has exactly one ColumnSemanticContract.
    All properties (role, measure type, eligibility, keys, statistics, lineage) are decoupled and independent.
    """
    # IDENTITY
    column_name: str
    source_table: str = "Master Dataset"
    source_column: str = ""
    physical_dtype: str = "object"

    # SEMANTICS
    semantic_role: str = "DIMENSION"  # IDENTIFIER, DIMENSION, DATETIME, MEASURE, TEXT, BOOLEAN
    semantic_subtype: str = "Categorical"
    measure_type: Optional[str] = None  # COUNT, CURRENCY, PERCENTAGE, DURATION, RATING, SCORE_INDEX, ORDINAL, CONTINUOUS, DISCRETE
    unit: Optional[str] = None
    currency: Optional[str] = None
    rating_scale: Optional[str] = None
    ordinal: bool = False
    boolean_semantics: bool = False
    description: str = ""

    # KEYS & RELATIONSHIPS
    key_role: str = "NONE"  # PRIMARY_KEY, FOREIGN_KEY, CANDIDATE_FK, NONE
    id_safety: bool = False
    uniqueness_pct: float = 0.0
    null_pct: float = 0.0
    candidate_parent: Optional[str] = None
    relationship_status: str = "UNVERIFIED"

    # ANALYTICAL ELIGIBILITY
    analytical_eligible: bool = True
    exclusion_reason: Optional[str] = None  # CONSTANT, TEST_FIXTURE, IDENTIFIER, LOW_COVERAGE, etc.
    grouping_eligible: bool = False
    time_series_eligible: bool = False
    correlation_eligible: bool = False
    pca_eligible: bool = False
    anova_eligible: bool = False
    kruskal_eligible: bool = False
    visualization_eligible: bool = True

    # STATISTICS
    non_null_count: int = 0
    missing_count: int = 0
    missing_pct: float = 0.0
    coverage_pct: float = 100.0
    unique_count: int = 0
    cardinality_pct: float = 0.0
    variance: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    sample_values: List[str] = []

    # LINEAGE
    derivation_type: str = "OBSERVED"  # OBSERVED, DERIVED, DETERMINISTIC_TRANSFORMATION, AGGREGATED, UNKNOWN
    derived_from: List[str] = []
    derivation_formula: Optional[str] = None
    transformation_type: Optional[str] = None

    # EVIDENCE
    confidence: float = 1.0
    evidence: List[str] = []
    semantic_evidence: Dict[str, Any] = {}

    def __init__(self, **data):
        if not data.get("source_column") and data.get("column_name"):
            data["source_column"] = data["column_name"]
        super().__init__(**data)

    # Legacy Compatibility Properties
    @property
    def is_identifier(self) -> bool:
        r = str(self.semantic_role).upper()
        k = str(self.key_role).upper()
        return r in ["IDENTIFIER", "ID"] or k in ["PRIMARY_KEY", "PRIMARY KEY", "FOREIGN_KEY", "FOREIGN KEY"] or self.id_safety

    @property
    def is_measure(self) -> bool:
        return str(self.semantic_role).upper() in ["MEASURE", "NUMERIC MEASURE"] and not self.is_identifier

    @property
    def is_dimension(self) -> bool:
        r = str(self.semantic_role).upper()
        return r in ["DIMENSION", "CATEGORICAL", "CATEGORICAL DIMENSION", "BOOLEAN", "TEXT"] and not self.is_identifier and not self.is_measure

    @property
    def is_time(self) -> bool:
        return str(self.semantic_role).upper() in ["DATETIME", "DATE/TIME", "TIME", "TEMPORAL"]

    @property
    def is_primary_key(self) -> bool:
        return "PRIMARY" in str(self.key_role).upper()

    @property
    def is_foreign_key(self) -> bool:
        return "FOREIGN" in str(self.key_role).upper()

    @property
    def null_percentage(self) -> float:
        return self.null_pct

    @property
    def missing_percentage(self) -> float:
        return self.missing_pct
