from typing import Any, Dict, List, Optional
from engine.contracts.base import GovernedContractBase
from engine.contracts.enums import DerivationType, RelationshipType


class DerivationContract(GovernedContractBase):
    """
    Canonical Derivation and Variable Lineage Contract (Prompt Sections 15 & 16).
    Tracks mathematical lineage:
    A = B * C, A = B / C, A = B + C, A = B - C, A = B * constant, A = B / constant,
    and percentage transformations (A = B / 25 * 100).
    """
    derivation_id: str
    target_column: str
    derivation_type: DerivationType = DerivationType.OBSERVED
    formula: Optional[str] = None
    source_columns: List[str] = []
    transformation_type: Optional[str] = None
    relationship_type: RelationshipType = RelationshipType.INDEPENDENT
    multiplier: Optional[float] = None
    constant_term: Optional[float] = None
    is_deterministic: bool = False
    confidence: float = 1.0
    validation_metrics: Dict[str, Any] = {}
    description: Optional[str] = None


class RelationshipContract(GovernedContractBase):
    """
    Canonical Multi-Table Referential Relationship Contract (Prompt Section 7 & 39).
    Defines relationships with empirical join coverage evidence.
    """
    relationship_id: str
    left_table: str
    right_table: str
    left_key: str
    right_key: str
    cardinality: str = "1:1"  # 1:1, 1:N, N:1, N:M
    relationship_status: str = "UNVERIFIED"  # VERIFIED, CANDIDATE_FK, UNVERIFIED
    referential_coverage_pct: float = 0.0
    orphan_count: int = 0
    duplicate_key_count: int = 0
    matched_rows: int = 0
    unmatched_rows: int = 0
    confidence: float = 1.0
    recommended_join_type: str = "LEFT OUTER JOIN"
