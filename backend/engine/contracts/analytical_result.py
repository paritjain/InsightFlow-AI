from typing import Any, Dict, List, Optional
from engine.contracts.base import GovernedContractBase
from engine.contracts.enums import AnalysisStatus, RelationshipType


class AnalyticalResult(GovernedContractBase):
    """
    Canonical Single Source of Analytical Truth (Prompt Section 26).
    Every chart, narrative takeaway, statistical test, insight, recommendation,
    action, and export MUST consume and reference an AnalyticalResult ID.
    No downstream module may recalculate the result independently.
    """
    _counter: int = 0

    analytical_result_id: str = ""
    dataset_id: str = "main"
    analysis_type: str = "GENERAL"  # CORRELATION, PCA, ANOVA, KRUSKAL, DESCRIPTIVE, TEMPORAL, OUTLIER, etc.
    status: str = "PASS"  # PASS, WARNING, NOT_APPLICABLE, INSUFFICIENT_DATA, BLOCKED, ERROR
    source_columns: List[str] = []
    source_table: str = "Master Dataset"
    method: str = "Exact Analytical Result"
    parameters: Dict[str, Any] = {}
    
    # Numerical evidence & metrics
    sample: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    effect_size: Dict[str, Any] = {}
    assumptions: Dict[str, Any] = {}
    coverage: Dict[str, Any] = {}
    
    # Relationship and derivation governance
    relationship_type: str = "INDEPENDENT"  # INDEPENDENT, DETERMINISTIC_TRANSFORMATION, STRUCTURAL_RELATIONSHIP
    derivation_status: str = "OBSERVED"
    evidence: Dict[str, Any] = {}
    interpretation_status: str = "VERIFIED"  # VERIFIED, CAUTION, NOT_APPLICABLE
    
    # Narrative and visual representation
    takeaway: str = ""
    interpretation: str = ""
    recommendation_eligibility: bool = True
    chart_spec: Optional[Dict[str, Any]] = None
    chart_type: str = "TABLE"
    chart_data: List[Dict[str, Any]] = []
    
    # Audit counts & grain
    total_record_count: int = 0
    valid_record_count: int = 0
    excluded_record_count: int = 0
    excluded_record_reason: Optional[str] = None
    grouping: Optional[str] = None
    aggregation: str = "NONE"
    unit: Optional[str] = None
    currency: Optional[str] = None
    confidence: float = 0.95
    warnings: List[str] = []
    lineage: Dict[str, Any] = {}
    values: Dict[str, Any] = {}
    statistics: Dict[str, Any] = {}

    def __init__(self, **data):
        if not data.get("analytical_result_id") and not data.get("result_id"):
            AnalyticalResult._counter += 1
            data["analytical_result_id"] = f"AR-{AnalyticalResult._counter:06d}"
        elif data.get("result_id") and not data.get("analytical_result_id"):
            data["analytical_result_id"] = data["result_id"]
        super().__init__(**data)
        
        # Keep result_id and analytical_result_id synchronized
        if not self.result_id:
            self.result_id = self.analytical_result_id
            
        # Ensure lineage has analytical_result_id
        if not self.lineage:
            self.lineage = {
                "analytical_result_id": self.analytical_result_id,
                "result_id": self.analytical_result_id,
                "dataset_id": self.dataset_id,
                "analysis_type": self.analysis_type,
                "method": self.method,
                "source_columns": self.source_columns,
                "relationship_type": self.relationship_type,
                "total_record_count": self.total_record_count,
                "valid_record_count": self.valid_record_count,
                "status": self.status
            }

    # Backward compatibility properties
    @property
    def result_id(self) -> str:
        return self.analytical_result_id or self.get("result_id", "")

    @result_id.setter
    def result_id(self, val: str):
        self.analytical_result_id = val

    @property
    def resultId(self) -> str:
        return self.result_id

    @property
    def datasetId(self) -> str:
        return self.dataset_id

    @property
    def sourceColumns(self) -> List[str]:
        return self.source_columns

    @property
    def rowCount(self) -> int:
        return self.total_record_count

    @property
    def sample_size(self) -> int:
        return self.valid_record_count

    @property
    def metric(self) -> str:
        return self.source_columns[0] if self.source_columns else ""

    @property
    def measure(self) -> str:
        return self.metric

    @property
    def numericalEvidence(self) -> List[Dict[str, Any]]:
        if isinstance(self.evidence, list):
            return self.evidence
        return [self.evidence] if self.evidence else []

    @property
    def numerical_evidence(self) -> List[Dict[str, Any]]:
        return self.numericalEvidence
