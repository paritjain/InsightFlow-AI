from typing import Any, Dict, List, Optional
from engine.contracts.base import GovernedContractBase
from engine.contracts.enums import (
    InsightType,
    RecommendationType,
    ActionExecutionStatus,
    RelationshipType
)


class InsightContract(GovernedContractBase):
    """
    Canonical Insight Contract (Prompt Section 28 & 29).
    Insights must NEVER be generated directly from raw data.
    Every insight traces directly to a verified AnalyticalResult ID.
    Deterministic relationships are tagged and prevented from high priority promotion.
    """
    insight_id: str
    analytical_result_id: str
    dataset_id: str = "main"
    insight_type: str = "EXPLORATORY"  # DESCRIPTIVE_BASELINE, DISTRIBUTION, COMPOSITION, ASSOCIATION, etc.
    category: str = "General"
    title: str = ""
    metric: str = ""
    dimensions: List[str] = []
    
    # Separation of Fact, Interpretation & Action
    observed_fact: str = ""
    fact: str = ""
    business_interpretation: str = ""
    interpretation: str = ""
    recommended_action: str = ""
    recommendation: str = ""
    conclusion: str = ""
    
    # Evidence & Rigor
    numerical_evidence: List[Dict[str, Any]] = []
    denominator_metadata: Dict[str, Any] = {}
    percentage_metadata: Dict[str, Any] = {}
    confidence: float = 0.95
    statistical_strength: str = "MODERATE"
    business_relevance: str = "MEDIUM"
    actionability: str = "ACTIONABLE"
    priority: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    priority_reason: str = ""
    
    # Governance & Non-causality
    relationship_type: str = "INDEPENDENT"
    causality_status: str = "NON_CAUSAL_ASSOCIATION"  # Prompt Section 19: Do not claim causality
    traceability_status: str = "FULLY_TRACEABLE"
    supporting_visualization: Optional[Dict[str, Any]] = None

    def __init__(self, **data):
        super().__init__(**data)
        if not self.fact and self.observed_fact:
            self.fact = self.observed_fact
        elif not self.observed_fact and self.fact:
            self.observed_fact = self.fact
            
        if not self.interpretation and self.business_interpretation:
            self.interpretation = self.business_interpretation
        elif not self.business_interpretation and self.interpretation:
            self.business_interpretation = self.interpretation
            
        if not self.recommendation and self.recommended_action:
            self.recommendation = self.recommended_action
        elif not self.recommended_action and self.recommendation:
            self.recommended_action = self.recommendation
            
        if not self.conclusion:
            self.conclusion = f"{self.observed_fact} {self.business_interpretation}".strip()


class RecommendationContract(GovernedContractBase):
    """
    Canonical Recommendation Contract (Prompt Section 31).
    Recommendations are generated strictly from verified insights.
    Deterministic relationships generate ANALYTICAL_GOVERNANCE, not bivariate regression.
    """
    recommendation_id: str
    insight_id: str
    analytical_result_id: str
    dataset_id: str = "main"
    recommendation_type: str = "ANALYTICAL_FOLLOW_UP"  # MONITORING_ACTION, DATA_QUALITY_ACTION, etc.
    action_type: str = "Operational"
    action_step: str = ""
    suggested_action: str = ""
    rationale: str = ""
    priority: str = "MEDIUM"
    
    # Metric Governance (Section 32: Monitored Metric, not KPI unless explicitly configured)
    monitored_metric: str = ""
    target_metric: str = ""
    
    # Lineage & Non-Hallucination Defaults (Section 31 & 33)
    baseline: str = "Observed baseline"
    target: str = "Not defined"
    owner: str = "UNASSIGNED"
    timeline: str = "Not specified"
    traceability_status: str = "FULLY_TRACEABLE"

    def __init__(self, **data):
        super().__init__(**data)
        if not self.suggested_action and self.action_step:
            self.suggested_action = self.action_step
        elif not self.action_step and self.suggested_action:
            self.action_step = self.suggested_action
        if not self.monitored_metric and self.target_metric:
            self.monitored_metric = self.target_metric


class ActionContract(GovernedContractBase):
    """
    Canonical Action Item Contract (Prompt Section 33 & 34).
    Full lineage: Dataset -> AnalyticalResult -> Insight -> Recommendation -> Action.
    Globally unique ACT-000001 IDs.
    Defaults enforce zero hallucination: owner = UNASSIGNED, target = Not defined, timeline = Not specified.
    """
    action_id: str
    recommendation_id: str
    insight_id: str
    analytical_result_id: str
    dataset_id: str = "main"
    
    action_type: str = "ANALYTICAL_FOLLOW_UP"
    priority: str = "MEDIUM"
    action_description: str = ""
    action_step: str = ""
    business_problem: str = ""
    
    # Zero-Hallucination Defaults (Prompt Section 33)
    owner: str = "UNASSIGNED"
    timeline: str = "Not specified"
    baseline: str = "Not defined"
    target: str = "Not defined"
    
    # Metric governance (Section 32)
    monitored_metric: str = ""
    source_columns: List[str] = []
    source_table: str = "Master Dataset"
    
    traceability_status: str = "FULLY_TRACEABLE"
    execution_status: str = "pending"  # pending, in_progress, completed, dismissed
    status: str = "pending"

    def __init__(self, **data):
        super().__init__(**data)
        if not self.action_description and self.action_step:
            self.action_description = self.action_step
        elif not self.action_step and self.action_description:
            self.action_step = self.action_description
        if not self.status:
            self.status = self.execution_status
        elif not self.execution_status:
            self.execution_status = self.status
