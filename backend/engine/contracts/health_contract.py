from typing import Any, Dict, List, Optional
from engine.contracts.base import GovernedContractBase


class DatasetHealthResult(GovernedContractBase):
    """
    Canonical Single Source of Truth for Dataset Health (Prompt Section 10).
    Every module (Profile, Cleaner, EDA, Insights, Recs, Actions, Exports) must consume this.
    No module is permitted to calculate a conflicting ad-hoc health score.
    """
    health_score: float = 100.0  # Overall canonical dataset health (0-100)
    completeness_score: float = 100.0
    duplicate_quality_score: float = 100.0
    uniqueness_score: float = 100.0
    semantic_quality_score: float = 100.0
    analytical_coverage_score: float = 100.0
    validity_score: float = 100.0
    consistency_score: float = 100.0
    structural_quality_score: float = 100.0

    # Underlying raw vs cleaned counts
    raw_row_count: int = 0
    cleaned_row_count: int = 0
    raw_duplicate_count: int = 0
    cleaned_duplicate_count: int = 0
    total_null_cells: int = 0
    total_cells: int = 0
    completeness_pct: float = 100.0

    health_status: str = "EXCELLENT"  # EXCELLENT, GOOD, MODERATE, DEGRADED, CRITICAL
    status: str = "PASS"
    breakdown: Dict[str, float] = {}
    warnings: List[str] = []

    def __init__(self, **data):
        super().__init__(**data)
        if not self.breakdown:
            self.breakdown = {
                "health_score": self.health_score,
                "completeness": self.completeness_score,
                "duplicate_quality": self.duplicate_quality_score,
                "uniqueness": self.uniqueness_score,
                "semantic_quality": self.semantic_quality_score,
                "analytical_coverage": self.analytical_coverage_score,
                "validity": self.validity_score,
                "consistency": self.consistency_score,
                "structural_quality": self.structural_quality_score
            }
