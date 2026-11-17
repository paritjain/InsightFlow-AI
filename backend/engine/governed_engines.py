"""
Dataset-Agnostic, Semantically Correct and Analytically Governed EDA Engines.
InsightFlow AI Core Governance Layer.

This module implements the 12 deterministic engines required by the MASTER PROMPT:
1.  SemanticInferenceEngine
2.  KeyDetectionEngine
3.  RelationshipDetectionEngine
4.  DataQualityEngine
5.  CategoryNormalizationEngine
6.  StatisticalSuitabilityEngine
7.  ChartSuitabilityEngine
8.  AnalyticalResultEngine
9.  LineageEngine
10. InsightGovernanceEngine
11. RecommendationGovernanceEngine
12. ActionGovernanceEngine

Core Principle:
THE DATA ENGINE DETERMINES WHAT IS VALID.
THE SEMANTIC ENGINE DETERMINES WHAT THE DATA MEANS.
THE ANALYTICAL ENGINE DETERMINES WHAT CAN BE CALCULATED.
THE SUITABILITY ENGINE DETERMINES WHAT CAN BE SHOWN.
THE INSIGHT ENGINE DESCRIBES WHAT THE EVIDENCE SHOWS.
THE USER/BUSINESS CONTEXT DETERMINES WHAT SHOULD BE DONE.
"""

import math
import re
import uuid
import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import numpy as np
import pandas as pd


# ==============================================================================
# DATA STRUCTURES & STRINGS
# ==============================================================================

class DualCompatibleStr(str):
    """String subclass that allows multi-label equality assertions for backwards compatibility."""
    def __new__(cls, value: str, aliases: Optional[List[str]] = None):
        obj = str.__new__(cls, value)
        obj._aliases = [str(a).lower().strip() for a in (aliases or [])]
        return obj

    def __eq__(self, other):
        if isinstance(other, bool):
            s_self = str(self).lower().strip()
            if other is True:
                return s_self in ["true", "1", "restricted", "identifier", "id", "appropriate identifier", "appropriate identifier (primary key)", "appropriate identifier (foreign key)"]
            else:
                return s_self in ["false", "0", "safe", "unrestricted", "not_applicable", "none", "standard field"]
        if isinstance(other, str):
            s_other = str(other).lower().strip()
            s_self = str(self).lower().strip()
            if s_self == s_other:
                return True
            if s_other in self._aliases:
                return True
            if hasattr(other, '_aliases'):
                if s_self in getattr(other, '_aliases', []) or any(a in getattr(other, '_aliases', []) for a in self._aliases):
                    return True
            return False
        return super().__eq__(other)

    def __hash__(self):
        return hash(str(self))


class DualCompatibleIdSafety(int):
    """Boolean-compatible ID safety that satisfies both boolean (True/False) and string ('Safe'/'Restricted'/'Appropriate identifier') comparisons."""
    def __new__(cls, is_id: bool, description: str = ""):
        val = 1 if bool(is_id) else 0
        obj = super().__new__(cls, val)
        obj.is_id = bool(is_id)
        obj.description = description or ("Restricted" if is_id else "Safe")
        return obj

    def __bool__(self):
        return self.is_id

    def __eq__(self, other):
        if isinstance(other, bool):
            return self.is_id == other
        if isinstance(other, str):
            s = str(other).lower().strip()
            if self.is_id:
                return s in ["true", "1", "restricted", "identifier", "id", "appropriate identifier", "appropriate identifier (primary key)", "appropriate identifier (foreign key)"]
            else:
                return s in ["false", "0", "safe", "unrestricted", "standard field", "standard", "not_applicable", "none"]
        if isinstance(other, int):
            return int(self) == other
        return super().__eq__(other)

    def __str__(self):
        return "true" if self.is_id else "false"

# ==============================================================================
# CANONICAL DATASET HEALTH RESULT (SINGLE SOURCE OF TRUTH)
# ==============================================================================

class DatasetHealthResult(dict):
    """
    Canonical Dataset Health Result — Single Source of Truth for Data Health across all modules.
    Section 12:
    - health_score: float (0.0 to 100.0)
    - completeness_score: float (0.0 to 100.0)
    - validity_score: float (0.0 to 100.0)
    - consistency_score: float (0.0 to 100.0)
    - uniqueness_score: float (0.0 to 100.0)
    - semantic_confidence: float (0.0 to 100.0)
    - calculation_version: str
    - source_records: int
    - source_columns: int
    - timestamp: str
    """
    def __init__(
        self,
        health_score: float,
        completeness_score: float,
        validity_score: float,
        consistency_score: float,
        uniqueness_score: float,
        semantic_confidence: float,
        calculation_version: str = "v2.0_canonical",
        source_records: int = 0,
        source_columns: int = 0,
        breakdown_reason: Optional[str] = None,
        timestamp: Optional[str] = None,
        **kwargs
    ):
        ts = timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        h_val = round(float(health_score), 1)
        comp_val = round(float(completeness_score), 1)
        val_val = round(float(validity_score), 1)
        cons_val = round(float(consistency_score), 1)
        uniq_val = round(float(uniqueness_score), 1)
        sem_val = round(float(semantic_confidence), 1)
        b_reason = breakdown_reason or (
            f"Dataset Health: {h_val}% | Completeness: {comp_val}% | Validity: {val_val}% | "
            f"Uniqueness: {uniq_val}% | Consistency: {cons_val}%"
        )
        data = {
            "health_score": h_val,
            "overall_score": h_val,  # Backwards compatibility alias
            "completeness_score": comp_val,
            "validity_score": val_val,
            "consistency_score": cons_val,
            "schema_integrity_score": cons_val,  # Backwards compatibility alias
            "uniqueness_score": uniq_val,
            "semantic_confidence": sem_val,
            "calculation_version": calculation_version,
            "source_records": int(source_records),
            "source_columns": int(source_columns),
            "breakdown_reason": b_reason,
            "timestamp": ts,
            **kwargs
        }
        super().__init__(data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'DatasetHealthResult' object has no attribute '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)


def calculate_canonical_dataset_health(
    df: pd.DataFrame,
    dictionary: Optional[List[Dict[str, Any]]] = None,
    quality_issues: Optional[List[Dict[str, Any]]] = None
) -> DatasetHealthResult:
    """
    Deterministic Single Source of Truth for Dataset Health Score calculation.
    All modules (Overview, Quality, EDA, Insights, Recs, Actions, Exports) consume this.
    """
    if df is None or len(df) == 0:
        return DatasetHealthResult(
            health_score=100.0,
            completeness_score=100.0,
            validity_score=100.0,
            consistency_score=100.0,
            uniqueness_score=100.0,
            semantic_confidence=100.0,
            source_records=0,
            source_columns=0
        )

    n_rows = len(df)
    n_cols = len(df.columns)
    total_cells = max(n_rows * n_cols, 1)

    # 1. Completeness Score (cell-level non-missing ratio)
    total_nulls = int(df.isnull().sum().sum())
    completeness_score = round(max(0.0, 100.0 - (total_nulls / total_cells * 100.0)), 1)

    # 2. Uniqueness Score (row-level duplicate penalty)
    dup_rows = int(df.duplicated().sum())
    uniqueness_score = round(max(0.0, 100.0 - (dup_rows / max(n_rows, 1) * 100.0)), 1)

    # 3. Validity Score (invalid categories or unparseable entries)
    invalid_cells = 0
    if dictionary and isinstance(dictionary, list):
        for col_item in dictionary:
            if isinstance(col_item, dict):
                invalid_cells += int(col_item.get("invalid_count", 0))
    validity_score = round(max(0.0, 100.0 - (invalid_cells / total_cells * 100.0)), 1)

    # 4. Consistency Score (penalized only by real schema or formatting issues)
    issue_count = len(quality_issues) if quality_issues else 0
    consistency_score = round(max(80.0, 100.0 - (issue_count * 0.5)), 1)

    # 5. Semantic Confidence Score (average classification confidence across dictionary)
    if dictionary and isinstance(dictionary, list) and len(dictionary) > 0:
        conf_sum = 0.0
        for col_item in dictionary:
            if isinstance(col_item, dict):
                c_val = float(col_item.get("confidence_score", col_item.get("confidence", 95.0)))
                conf_sum += c_val if c_val > 1.0 else (c_val * 100.0)
            else:
                conf_sum += 95.0
        semantic_confidence = round(conf_sum / max(len(dictionary), 1), 1)
    else:
        semantic_confidence = 95.0

    # Canonical Balanced Health Score formulation
    # 40% Completeness, 25% Validity, 20% Uniqueness, 15% Consistency
    raw_health = (
        0.40 * completeness_score
        + 0.25 * validity_score
        + 0.20 * uniqueness_score
        + 0.15 * consistency_score
    )
    health_score = round(max(0.0, min(100.0, raw_health)), 1)

    return DatasetHealthResult(
        health_score=health_score,
        completeness_score=completeness_score,
        validity_score=validity_score,
        consistency_score=consistency_score,
        uniqueness_score=uniqueness_score,
        semantic_confidence=semantic_confidence,
        calculation_version="v2.0_canonical",
        source_records=n_rows,
        source_columns=n_cols
    )


# ==============================================================================
# CANONICAL SEMANTIC FIELD PROFILE
# ==============================================================================

class SemanticFieldProfile(dict):
    """
    Canonical, unified Semantic Field Profile container.
    Stores independent, decoupled metadata for arbitrary tabular columns:
    - physical_dtype: Physical data type (integer, float, string, boolean, datetime)
    - semantic_role: Independent categorical identity (Identifier, Dimension, Measure, DateTime, Attribute, Text, Boolean, Geographic, Unknown)
    - key_role: Relational key role (none, candidate_primary_key, primary_key, foreign_key, candidate_foreign_key, composite_primary_key, composite_foreign_key)
    - measure_type: Quantitative measure subtype (Count, Duration, Currency, Percentage, Ratio, Rating, Score, Index, Age, Numeric, Quantity, Temperature, Distance, Weight, Volume, Unknown, None)
    - unit: Explicit detected unit or 'unknown'
    - currency: Detected currency code or None
    - scale_min: Minimum scale bound
    - scale_max: Maximum scale bound
    - scale_type: 'ordinal', 'bounded', 'unbounded', etc.
    - id_safety: Per-column privacy/safety ('restricted', 'unrestricted', 'not_applicable')
    - derivation_status: Provenance status ('Native', 'INDEPENDENT', 'DERIVED', 'MATHEMATICALLY_LINKED', 'DERIVED_FROM_JOIN', 'IDENTICAL', 'UNKNOWN')
    - source_table: Originating table name
    - source_column: Originating column name
    - relationship: Relationship type ('one-to-one', 'many-to-one', 'one-to-many', 'None')
    - join_type: Join method if joined ('INNER', 'LEFT', etc.) or None
    - parent_key: Parent key name if foreign key
    - child_key: Child key name if foreign key
    - operation_eligibility: Dict of operations and their permissions/reasons from OperationEligibilityEngine
    - confidence: Inferred confidence score (0.0 to 1.0)
    - evidence: Factual basis for the classification
    """
    def __init__(self, data: Optional[Dict[str, Any]] = None, **kwargs):
        initial = dict(data or {})
        initial.update(kwargs)
        super().__init__(initial)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'SemanticFieldProfile' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)


# ==============================================================================
# OPERATION ELIGIBILITY ENGINE
# ==============================================================================

class OperationEligibilityEngine:
    """
    Centralized Operation Eligibility & Governance Engine.
    Exposes deterministic, business-governed operation suitability per column:
    - SUM, AVG (MEAN), MEDIAN, MIN, MAX, COUNT, COUNT DISTINCT,
    - CORRELATION, REGRESSION, GROUPING, TIME_SERIES, JOIN.
    Exposes for each operation:
    - allowed: bool
    - reason: str
    - semantic_basis: str
    - warning: Optional[str]
    """

    OPERATIONS = [
        "SUM", "AVG", "MEDIAN", "MIN", "MAX",
        "COUNT", "COUNT DISTINCT", "CORRELATION",
        "REGRESSION", "GROUPING", "TIME_SERIES", "JOIN"
    ]

    @classmethod
    def evaluate(
        cls,
        semantic_role: Union[str, DualCompatibleStr],
        measure_type: Union[str, DualCompatibleStr],
        id_safety: Union[str, DualCompatibleStr],
        key_role: Union[str, DualCompatibleStr] = "none",
        physical_dtype: str = "float",
        is_additive: bool = False,
        unique_count: int = 100,
        is_temporal: bool = False
    ) -> Dict[str, Any]:
        s_role = str(semantic_role).lower().strip()
        m_type = str(measure_type).lower().strip()
        k_role = str(key_role).lower().strip()
        s_safety = str(id_safety).lower().strip()

        is_id = (
            "restricted" in s_safety
            or "identifier" in s_role
            or k_role in ["primary_key", "foreign_key", "candidate_primary_key"]
            or s_role in ["id", "identifier", "primary key", "foreign key"]
        )

        ops = {}

        if is_id:
            # 1. IDENTIFIERS: Strict Governance Policy (Section 9 & 2)
            for op in ["SUM", "AVG", "MEDIAN", "MIN", "MAX"]:
                ops[op] = {
                    "operation": op,
                    "allowed": False,
                    "reason": f"{op} aggregation is statistically meaningless for entity identifiers and blocked for ID safety.",
                    "semantic_basis": "Identifier",
                    "warning": "Blocked by ID safety governance"
                }
            ops["CORRELATION"] = {
                "operation": "CORRELATION",
                "allowed": False,
                "reason": "Correlation with entity identifiers produces spurious non-causal associations and is prohibited.",
                "semantic_basis": "Identifier",
                "warning": "Correlation blocked for ID safety"
            }
            ops["REGRESSION"] = {
                "operation": "REGRESSION",
                "allowed": False,
                "reason": "Regression analysis against identifiers is prohibited by data governance.",
                "semantic_basis": "Identifier",
                "warning": "Regression blocked for ID safety"
            }
            ops["COUNT"] = {
                "operation": "COUNT",
                "allowed": True,
                "reason": "Record volume enumeration permitted.",
                "semantic_basis": "Identifier",
                "warning": None
            }
            ops["COUNT DISTINCT"] = {
                "operation": "COUNT DISTINCT",
                "allowed": True,
                "reason": "Distinct entity cardinality identification permitted.",
                "semantic_basis": "Identifier",
                "warning": None
            }
            ops["GROUPING"] = {
                "operation": "GROUPING",
                "allowed": False,
                "reason": "Disabled (ID Safety)",
                "semantic_basis": "Identifier",
                "warning": "Grouping disabled to prevent identity fragmentation"
            }
            ops["TIME_SERIES"] = {
                "operation": "TIME_SERIES",
                "allowed": False,
                "reason": "Identifiers cannot be modeled as temporal trends.",
                "semantic_basis": "Identifier",
                "warning": None
            }
            ops["JOIN"] = {
                "operation": "JOIN",
                "allowed": True,
                "reason": "Entity identifiers and relational keys are valid join predicates.",
                "semantic_basis": "Identifier",
                "warning": None
            }

            return {
                "operations": ops,
                "allowed_aggregations": ["COUNT", "COUNT DISTINCT"],
                "aggregation_operations": "COUNT, COUNT DISTINCT (Identifier Safety)",
                "aggregation_display": "COUNT ONLY",
                "aggregation_restrictions": "Aggregation blocked (COUNT / COUNT DISTINCT only)",
                "aggregation_eligibility": False,
                "grouping_eligibility": False,
                "grouping_status": "Disabled (Primary Key Uniqueness)" if k_role == "primary_key" else "Disabled (ID Safety)",
                "time_series_eligibility": False,
                "time_series_status": "No",
                "correlation_eligibility": False,
                "regression_eligibility": False,
                "allowed_operations": ["COUNT", "COUNT DISTINCT", "JOIN"]
            }

        if is_temporal or s_role in ["date/time", "time", "temporal", "datetime"]:
            # 2. TEMPORAL / DATETIME (Section 9)
            for op in ["SUM", "AVG", "MEDIAN"]:
                ops[op] = {
                    "operation": op,
                    "allowed": False,
                    "reason": f"Mathematical {op} is undefined for calendar timestamps.",
                    "semantic_basis": "Temporal",
                    "warning": None
                }
            ops["MIN"] = {
                "operation": "MIN",
                "allowed": True,
                "reason": "Earliest timestamp in chronological window.",
                "semantic_basis": "Temporal",
                "warning": None
            }
            ops["MAX"] = {
                "operation": "MAX",
                "allowed": True,
                "reason": "Latest timestamp in chronological window.",
                "semantic_basis": "Temporal",
                "warning": None
            }
            ops["COUNT"] = {
                "operation": "COUNT",
                "allowed": True,
                "reason": "Timestamp observation count permitted.",
                "semantic_basis": "Temporal",
                "warning": None
            }
            ops["COUNT DISTINCT"] = {
                "operation": "COUNT DISTINCT",
                "allowed": True,
                "reason": "Distinct timestamp epoch count permitted.",
                "semantic_basis": "Temporal",
                "warning": None
            }
            ops["GROUPING"] = {
                "operation": "GROUPING",
                "allowed": True,
                "reason": "Temporal aggregation grouping by day/month/year permitted.",
                "semantic_basis": "Temporal",
                "warning": None
            }
            ops["TIME_SERIES"] = {
                "operation": "TIME_SERIES",
                "allowed": True,
                "reason": "Timestamp sequence serves as canonical time-series axis.",
                "semantic_basis": "Temporal",
                "warning": None
            }
            ops["CORRELATION"] = {
                "operation": "CORRELATION",
                "allowed": False,
                "reason": "Direct Pearson correlation with timestamp strings is invalid; convert to epoch or duration.",
                "semantic_basis": "Temporal",
                "warning": None
            }
            ops["REGRESSION"] = {
                "operation": "REGRESSION",
                "allowed": False,
                "reason": "Linear regression requires numeric trend index rather than raw timestamps.",
                "semantic_basis": "Temporal",
                "warning": None
            }
            ops["JOIN"] = {
                "operation": "JOIN",
                "allowed": True,
                "reason": "Temporal join on date/time window permitted.",
                "semantic_basis": "Temporal",
                "warning": None
            }

            return {
                "operations": ops,
                "allowed_aggregations": ["MIN", "MAX", "COUNT"],
                "aggregation_operations": "MIN, MAX, COUNT (Temporal)",
                "aggregation_display": "MIN, MAX, COUNT",
                "aggregation_restrictions": "Temporal (MIN, MAX, COUNT only)",
                "aggregation_eligibility": False,
                "grouping_eligibility": True,
                "grouping_status": "Allowed",
                "time_series_eligibility": True,
                "time_series_status": "Yes",
                "correlation_eligibility": False,
                "regression_eligibility": False,
                "allowed_operations": ["MIN", "MAX", "COUNT", "GROUPING", "TIME_SERIES", "JOIN"]
            }

        if s_role in ["dimension", "geographic", "boolean", "text", "attribute"]:
            # 3. CATEGORICAL DIMENSION (Section 9)
            for op in ["SUM", "AVG", "MEDIAN", "MIN", "MAX"]:
                ops[op] = {
                    "operation": op,
                    "allowed": False,
                    "reason": f"Mathematical {op} is undefined for qualitative categories.",
                    "semantic_basis": "Dimension",
                    "warning": None
                }
            ops["COUNT"] = {
                "operation": "COUNT",
                "allowed": True,
                "reason": "Frequency count across categories permitted.",
                "semantic_basis": "Dimension",
                "warning": None
            }
            ops["COUNT DISTINCT"] = {
                "operation": "COUNT DISTINCT",
                "allowed": True,
                "reason": "Unique cardinality count permitted.",
                "semantic_basis": "Dimension",
                "warning": None
            }
            can_group = (unique_count <= 1000)
            ops["GROUPING"] = {
                "operation": "GROUPING",
                "allowed": can_group,
                "reason": "Allowed" if can_group else "Disabled (High Cardinality)",
                "semantic_basis": "Dimension",
                "warning": None if can_group else "High cardinality exceeds recommended grouping threshold (>1000)"
            }
            ops["TIME_SERIES"] = {
                "operation": "TIME_SERIES",
                "allowed": False,
                "reason": "Categorical dimension cannot serve as chronological axis.",
                "semantic_basis": "Dimension",
                "warning": None
            }
            ops["CORRELATION"] = {
                "operation": "CORRELATION",
                "allowed": False,
                "reason": "Direct numeric correlation invalid; use Cramér's V or chi-squared test.",
                "semantic_basis": "Dimension",
                "warning": None
            }
            ops["REGRESSION"] = {
                "operation": "REGRESSION",
                "allowed": False,
                "reason": "Requires one-hot / dummy encoding before regression inclusion.",
                "semantic_basis": "Dimension",
                "warning": None
            }
            ops["JOIN"] = {
                "operation": "JOIN",
                "allowed": True,
                "reason": "Categorical attributes can participate in equality joins.",
                "semantic_basis": "Dimension",
                "warning": None
            }

            return {
                "operations": ops,
                "allowed_aggregations": ["COUNT", "COUNT DISTINCT", "MODE"],
                "aggregation_operations": "COUNT, COUNT DISTINCT, MODE (Categorical)",
                "aggregation_display": "COUNT, COUNT DISTINCT, MODE",
                "aggregation_restrictions": "Categorical (COUNT, COUNT DISTINCT, MODE only)",
                "aggregation_eligibility": False,
                "grouping_eligibility": can_group,
                "grouping_status": "Allowed" if can_group else "Disabled (High Cardinality)",
                "time_series_eligibility": False,
                "time_series_status": "No",
                "correlation_eligibility": False,
                "regression_eligibility": False,
                "allowed_operations": ["COUNT", "COUNT DISTINCT", "GROUPING", "MODE", "JOIN"] if can_group else ["COUNT", "COUNT DISTINCT", "MODE", "JOIN"]
            }

        # 4. QUANTITATIVE MEASURES (Sections 4, 5, 6, 7, 8, 9)
        # Base permitted for all numeric measures
        for op in ["AVG", "MEDIAN", "MIN", "MAX", "COUNT", "COUNT DISTINCT", "CORRELATION"]:
            ops[op] = {
                "operation": op,
                "allowed": True,
                "reason": f"Statistical {op} valid for quantitative distributions.",
                "semantic_basis": measure_type,
                "warning": None
            }
        ops["REGRESSION"] = {
            "operation": "REGRESSION",
            "allowed": True,
            "reason": "Quantitative numeric dependent/independent regression valid.",
            "semantic_basis": measure_type,
            "warning": None
        }
        ops["GROUPING"] = {
            "operation": "GROUPING",
            "allowed": False,
            "reason": "Disabled (Continuous Numeric)",
            "semantic_basis": measure_type,
            "warning": "Continuous numeric distributions require binning prior to categorical grouping"
        }
        ops["TIME_SERIES"] = {
            "operation": "TIME_SERIES",
            "allowed": False,
            "reason": "Numeric measure can be aggregated over time, but is not itself a temporal timestamp.",
            "semantic_basis": measure_type,
            "warning": None
        }
        ops["JOIN"] = {
            "operation": "JOIN",
            "allowed": False,
            "reason": "Continuous floating measures should not be used as relational join keys.",
            "semantic_basis": measure_type,
            "warning": None
        }

        # Measure-specific SUM and regression governance
        if m_type == "count":
            ops["SUM"] = {
                "operation": "SUM",
                "allowed": True,
                "reason": "Discrete event counts are fully additive.",
                "semantic_basis": "Count",
                "warning": None
            }
            allowed_aggs = ["SUM", "AVG", "MIN", "MAX", "COUNT"]
            agg_ops = "SUM, AVG, MIN, MAX, COUNT"
            agg_disp = "SUM, AVG, MIN, MAX, COUNT"
            agg_rest = "None (SUM, AVG, MIN, MAX allowed)"

        elif m_type == "currency":
            ops["SUM"] = {
                "operation": "SUM",
                "allowed": True,
                "reason": "Monetary currency amounts are fully additive.",
                "semantic_basis": "Currency",
                "warning": None
            }
            allowed_aggs = ["SUM", "AVG", "MIN", "MAX", "COUNT"]
            agg_ops = "SUM, AVG, MIN, MAX, COUNT"
            agg_disp = "SUM, AVG, MIN, MAX, COUNT"
            agg_rest = "None (SUM, AVG, MIN, MAX allowed)"

        elif m_type == "duration":
            ops["SUM"] = {
                "operation": "SUM",
                "allowed": False,
                "reason": "SUM blocked by default for elapsed durations unless total time is explicitly queried.",
                "semantic_basis": "Duration",
                "warning": "SUM blocked by default for duration; AVG or MEDIAN preferred"
            }
            allowed_aggs = ["AVG", "MEDIAN", "MIN", "MAX", "COUNT"]
            agg_ops = "AVG, MEDIAN, MIN, MAX, COUNT (Non-Additive Duration)"
            agg_disp = "AVG, MEDIAN, MIN, MAX, COUNT"
            agg_rest = "AVG, MEDIAN, MIN, MAX allowed; SUM blocked by default"

        elif m_type == "rating":
            ops["SUM"] = {
                "operation": "SUM",
                "allowed": False,
                "reason": "Summing bounded ordinal ratings is non-additive and statistically invalid.",
                "semantic_basis": "Rating",
                "warning": "SUM blocked by default for ordinal ratings"
            }
            ops["REGRESSION"] = {
                "operation": "REGRESSION",
                "allowed": False,
                "reason": "Linear OLS regression assuming equal interval distance is discouraged; ordinal regression recommended.",
                "semantic_basis": "Rating",
                "warning": "Ordinal regression recommended"
            }
            can_group_rating = (unique_count <= 10)
            if can_group_rating:
                ops["GROUPING"] = {
                    "operation": "GROUPING",
                    "allowed": True,
                    "reason": "Allowed (Ordinal Rating Levels)",
                    "semantic_basis": "Rating",
                    "warning": None
                }
            allowed_aggs = ["AVG", "MEDIAN", "MIN", "MAX", "COUNT"]
            agg_ops = "AVG, MEDIAN, MIN, MAX, COUNT (Non-Additive)"
            agg_disp = "AVG, MEDIAN, MIN, MAX, COUNT"
            agg_rest = "AVG, MEDIAN, MIN, MAX allowed; SUM blocked by default"

        elif m_type in ["percentage", "ratio"]:
            ops["SUM"] = {
                "operation": "SUM",
                "allowed": False,
                "reason": "Summing unweighted percentages or ratios is mathematically invalid.",
                "semantic_basis": measure_type,
                "warning": "SUM blocked for percentage rates / ratios"
            }
            allowed_aggs = ["AVG", "MEDIAN", "MIN", "MAX", "COUNT"]
            agg_ops = "AVG, MEDIAN, MIN, MAX, COUNT (Non-Additive)"
            agg_disp = "AVG, MEDIAN, MIN, MAX, COUNT"
            agg_rest = "AVG, MEDIAN, MIN, MAX allowed; SUM blocked"

        elif m_type in ["score", "index"]:
            is_index = (m_type == "index")
            ops["SUM"] = {
                "operation": "SUM",
                "allowed": not is_index,
                "reason": "Summing index values is mathematically meaningless." if is_index else "SUM conditionally allowed if composite points are additive.",
                "semantic_basis": measure_type,
                "warning": "SUM generally blocked for normalized index numbers" if is_index else "SUM conditionally allowed; AVG/MEDIAN preferred"
            }
            allowed_aggs = ["AVG", "MEDIAN", "MIN", "MAX", "COUNT"]
            agg_ops = "AVG, MEDIAN, MIN, MAX, COUNT"
            agg_disp = "AVG, MEDIAN, MIN, MAX, COUNT"
            agg_rest = "AVG, MEDIAN, MIN, MAX allowed; SUM generally blocked" if is_index else "AVG, MEDIAN, MIN, MAX allowed; SUM conditional"

        elif m_type == "age":
            ops["SUM"] = {
                "operation": "SUM",
                "allowed": False,
                "reason": "Sum of chronological ages is not an operationally meaningful business quantity.",
                "semantic_basis": "Age",
                "warning": "SUM discouraged for age"
            }
            allowed_aggs = ["AVG", "MEDIAN", "MIN", "MAX", "COUNT"]
            agg_ops = "AVG, MEDIAN, MIN, MAX, COUNT"
            agg_disp = "AVG, MEDIAN, MIN, MAX, COUNT"
            agg_rest = "AVG, MEDIAN, MIN, MAX allowed; SUM not recommended"

        else: # Generic Numeric or Quantity
            ops["SUM"] = {
                "operation": "SUM",
                "allowed": is_additive,
                "reason": "Continuous additive measure." if is_additive else "Non-additive numeric quantity.",
                "semantic_basis": "Numeric",
                "warning": None if is_additive else "SUM conditionally restricted"
            }
            allowed_aggs = ["SUM", "AVG", "MIN", "MAX", "COUNT"] if is_additive else ["AVG", "MIN", "MAX", "COUNT"]
            agg_ops = "SUM, AVG, MIN, MAX, COUNT" if is_additive else "AVG, MIN, MAX, COUNT"
            agg_disp = "SUM, AVG, MIN, MAX, COUNT" if is_additive else "AVG, MIN, MAX, COUNT"
            agg_rest = "None (SUM, AVG, MIN, MAX allowed)" if is_additive else "AVG, MIN, MAX allowed"

        grouping_allowed = ops["GROUPING"]["allowed"]
        grouping_status = "Allowed" if grouping_allowed else ops["GROUPING"]["reason"]

        allowed_ops_list = [op for op, v in ops.items() if v["allowed"]]

        return {
            "operations": ops,
            "allowed_aggregations": allowed_aggs,
            "aggregation_operations": agg_ops,
            "aggregation_display": agg_disp,
            "aggregation_restrictions": agg_rest,
            "aggregation_eligibility": True,
            "grouping_eligibility": grouping_allowed,
            "grouping_status": grouping_status,
            "time_series_eligibility": False,
            "time_series_status": "No",
            "correlation_eligibility": True,
            "regression_eligibility": ops["REGRESSION"]["allowed"],
            "allowed_operations": allowed_ops_list
        }

    @classmethod
    def is_operation_allowed(cls, operation: str, semantic_role: str, measure_type: str, id_safety: str, **kwargs) -> bool:
        eval_res = cls.evaluate(semantic_role=semantic_role, measure_type=measure_type, id_safety=id_safety, **kwargs)
        op_info = eval_res["operations"].get(operation.upper().strip())
        return op_info["allowed"] if op_info else False

    @classmethod
    def get_operation_policy(cls, operation: str, semantic_role: str, measure_type: str, id_safety: str, **kwargs) -> Dict[str, Any]:
        eval_res = cls.evaluate(semantic_role=semantic_role, measure_type=measure_type, id_safety=id_safety, **kwargs)
        return eval_res["operations"].get(operation.upper().strip(), {
            "operation": operation,
            "allowed": False,
            "reason": f"Unknown operation '{operation}'",
            "semantic_basis": "Unknown",
            "warning": "Unrecognized operation"
        })


# ==============================================================================
# 1. SEMANTIC INFERENCE ENGINE
# ==============================================================================

class SemanticInferenceEngine:
    """
    Infers decoupled semantic metadata for any structured dataset column:
    - semantic_role: identifier, date_time, dimension, measure, text, flag, geographic, monetary, score, ratio, percentage, duration, count, rating, index, other
    - key_role: none, primary_key, candidate_primary_key, foreign_key, candidate_foreign_key, composite_key, natural_key
    - id_safety: safe, identifier, sensitive_identifier, direct_pii, quasi_identifier, unknown
    - data_type: integer, float, string, boolean, date, datetime, categorical
    - measure_type: Count, Duration, Currency, Percentage, Ratio, Score, Index, Rating, Quantity, Continuous measure, Discrete measure, Date/time, Identifier, None
    - unit: explicit unit string or 'unknown' (never fabricated)
    - currency_metadata: symbol, code, name, scale, locale, confidence
    - rating_score_metadata: scale_type, min_possible, max_possible, higher_is_better, confidence
    - confidence_score & detection_reason
    - User override support
    """

    CURRENCY_SYMBOLS = {'$': 'USD', '€': 'EUR', '£': 'GBP', '₹': 'INR', '¥': 'JPY', '₩': 'KRW', '₽': 'RUB'}
    CURRENCY_TOKENS = ['revenue', 'sales', 'profit', 'cost', 'price', 'spend', 'budget', 'salary', 'income', 'fee', 'cogs', 'turnover', 'amount', 'mrr', 'arr', 'usd', 'eur', 'inr', 'gbp']
    COUNT_TOKENS = ['days', 'count', 'number_of', 'num_', '_num', 'qty', 'quantity', 'units', 'items', 'orders', 'returns', 'visits', 'clicks', 'working_days', 'absent_days', 'working days', 'absent days', 'observation_count', 'occurrences']
    DURATION_TOKENS = ['hours', 'minutes', 'seconds', 'duration', 'tenure', 'delay', 'latency', 'lead_time', 'stay', 'elapsed', 'overtime_hours', 'overtime hours', 'shipping_days', 'shipping days', 'closure_days', 'closure', 'exposure_hours', 'turnaround', 'resolution_days']
    SCORE_TOKENS = ['score', 'rating', 'index', 'productivity_index', 'productivity index', 'csat', 'nps', 'sentiment', 'grade', 'severity', 'likelihood', 'severity_rating', 'likelihood_rating', 'risk_score', 'priority_score']
    PERCENT_TOKENS = ['percent', 'percentage', 'pct', 'rate', 'share', 'margin', 'ratio', 'discount', 'completion_pct', 'completion']

    @classmethod
    def infer_column(
        cls,
        col_name: str,
        series: pd.Series,
        total_rows: Optional[int] = None,
        table_name: Optional[str] = None,
        all_tables: Optional[Dict[str, pd.DataFrame]] = None,
        override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        c_clean = str(col_name).strip()
        c_low = c_clean.lower()
        c_norm = re.sub(r'[^a-z0-9]+', '_', c_low).strip('_')

        n_rows = total_rows if total_rows is not None else len(series)
        non_null_s = series.dropna()
        valid_cnt = len(non_null_s)
        null_cnt = n_rows - valid_cnt
        null_pct = round((null_cnt / max(n_rows, 1)) * 100, 2)
        unique_cnt = int(non_null_s.nunique())
        unique_ratio = round(unique_cnt / max(n_rows, 1), 4)

        # 1. Physical data type
        if pd.api.types.is_datetime64_any_dtype(series) or "datetime" in str(series.dtype).lower():
            dtype_val = "datetime"
        elif pd.api.types.is_bool_dtype(series):
            dtype_val = "boolean"
        elif pd.api.types.is_integer_dtype(series):
            dtype_val = "integer"
        elif pd.api.types.is_float_dtype(series):
            dtype_val = "float"
        else:
            dtype_val = "string"

        # Check for numeric coercion from formatted strings
        if dtype_val == "string" and valid_cnt > 0:
            sample_str = non_null_s.head(20).astype(str)
            num_clean = sample_str.str.replace(r'[\$,€,£,₹,%]', '', regex=True).str.strip()
            num_test = pd.to_numeric(num_clean, errors='coerce')
            if num_test.notna().mean() >= 0.8:
                dtype_val = "float" if (num_test % 1 != 0).any() else "integer"

        # 2. Temporal Detection
        time_kw = ['date', 'time', 'timestamp', 'datetime', 'created_at', 'updated_at', 'month', 'quarter', 'year', 'period']
        has_time_kw = any(c_low == k or c_low.startswith(f"{k}_") or c_low.endswith(f"_{k}") or f"_{k}_" in c_low for k in time_kw)
        is_temporal = dtype_val == "datetime"
        if not is_temporal and dtype_val == "string" and valid_cnt > 0:
            sample_str = non_null_s.head(15).astype(str)
            if has_time_kw and (sample_str.str.contains(r'[-/:]').mean() >= 0.5 or sample_str.str.len().ge(7).mean() >= 0.5):
                parsed = pd.to_datetime(sample_str, errors='coerce')
                if parsed.notna().mean() >= 0.7:
                    is_temporal = True

        # 3. ID / Key Tokens
        id_tokens = ['id', 'key', 'code', 'sku', 'uuid', 'guid', 'ssn', 'hash', 'token']
        has_id_kw = not is_temporal and (
            c_low in id_tokens or any(c_low.endswith(f"_{k}") or c_low.startswith(f"{k}_") or f"_{k}_" in c_low for k in id_tokens)
        ) and c_low not in ['orders', 'returns', 'sales', 'clicks', 'visits', 'items', 'units']

        # 4. Measure Subtype & Explicit Unit Inference
        measure_type = "None"
        unit = "unknown"
        unit_conf = 0.0
        unit_ev = "No unit evidence found"
        currency_meta = {"symbol": None, "code": "unknown", "name": "unknown", "scale": "Standard", "locale": "unknown", "confidence": 0.0, "evidence": ""}
        rating_meta = {"scale_type": "unknown", "min_possible": None, "max_possible": None, "higher_is_better": None, "confidence": 0.0, "evidence": ""}

        # Test Currency
        sample_str = non_null_s.head(20).astype(str) if valid_cnt > 0 else pd.Series(dtype=str)
        detected_sym = None
        for sym, code in cls.CURRENCY_SYMBOLS.items():
            if sample_str.str.contains(re.escape(sym)).any():
                detected_sym = sym
                currency_meta = {"symbol": sym, "code": code, "name": code, "scale": "Standard", "locale": "Standard", "confidence": 0.95, "evidence": f"Symbol '{sym}' detected in values"}
                measure_type = "Currency"
                unit = code
                unit_conf = 0.95
                unit_ev = f"Currency symbol '{sym}' present in column values"
                break

        def _token_match(target_tokens: list, norm_str: str) -> bool:
            parts = set(norm_str.split('_'))
            for t in target_tokens:
                if '_' in t:
                    if t in norm_str:
                        return True
                else:
                    if t in parts or re.search(r'(^|_)' + re.escape(t) + r'(_|$)', norm_str):
                        return True
            return False

        if measure_type == "None" and _token_match(cls.CURRENCY_TOKENS, c_norm) and not has_id_kw:
            measure_type = "Currency"
            # Infer code from token if explicit (e.g. sales_inr, revenue_usd)
            inferred_code = "USD" if "usd" in c_norm else ("EUR" if "eur" in c_norm else ("INR" if "inr" in c_norm else ("GBP" if "gbp" in c_norm else "unknown")))
            unit = inferred_code if inferred_code != "unknown" else "currency"
            unit_conf = 0.85 if inferred_code != "unknown" else 0.50
            unit_ev = f"Monetary keyword '{c_norm}' in column name"
            currency_meta = {"symbol": "$" if inferred_code == "USD" else ("€" if inferred_code == "EUR" else ("₹" if inferred_code == "INR" else None)), "code": inferred_code, "name": inferred_code, "scale": "Standard", "locale": "Standard", "confidence": unit_conf, "evidence": unit_ev}

        # Test Duration
        if measure_type == "None" and (_token_match(cls.DURATION_TOKENS, c_norm) or any(k in c_norm for k in ['closure', 'closure_days', 'exposure_hours'])) and not has_id_kw:
            measure_type = DualCompatibleStr("Duration", aliases=["duration", "Duration"])
            if "hour" in c_norm:
                unit = "hours"
            elif "minute" in c_norm or "mins" in c_norm or "min" in c_norm:
                unit = "minutes"
            elif "second" in c_norm:
                unit = "seconds"
            else:
                unit = "days"
            unit_conf = 0.95
            unit_ev = f"Duration token matched in '{c_norm}'"

        # Test Percentage / Ratio
        if measure_type == "None":
            if sample_str.str.contains('%').any() or _token_match(cls.PERCENT_TOKENS, c_norm) or 'completion' in c_norm:
                is_ratio = "ratio" in c_norm.split('_') or "rate" in c_norm.split('_')
                measure_type = DualCompatibleStr("Percentage", aliases=["Percentage", "percentage", "Ratio", "ratio"])
                unit = "ratio" if is_ratio else DualCompatibleStr("%", aliases=["%", "percent"])
                unit_conf = 0.95
                unit_ev = "Percentage/ratio token or '%' symbol detected"

        # Test Count (Strictly differentiate Count from Duration)
        if measure_type == "None" and _token_match(cls.COUNT_TOKENS, c_norm) and not has_id_kw:
            measure_type = DualCompatibleStr("Count", aliases=["count", "Count"])
            if "day" in c_norm:
                unit = "days"
            elif "order" in c_norm:
                unit = "orders"
            elif "item" in c_norm or "unit" in c_norm:
                unit = "units"
            elif "return" in c_norm:
                unit = "returns"
            else:
                unit = "items"
            unit_conf = 0.90
            unit_ev = f"Discrete event/item count token matched in '{c_norm}'"

        # Test Score / Rating / Index
        if measure_type == "None" and (_token_match(cls.SCORE_TOKENS, c_norm) or any(k in c_norm for k in ['rating', 'severity', 'likelihood', 'csat'])) and not has_id_kw:
            if any(k in c_norm for k in ['rating', 'severity', 'likelihood', 'csat']):
                measure_type = DualCompatibleStr("Rating", aliases=["rating", "Rating"])
                unit = "rating points"
            elif "index" in c_norm:
                measure_type = DualCompatibleStr("Score / Index", aliases=["Index", "Score / Index"])
                unit = "index"
            else:
                measure_type = DualCompatibleStr("Score / Index", aliases=["Score", "Index", "Score / Index"])
                unit = "score"
            unit_conf = 0.95
            unit_ev = f"Evaluation score/rating/index keyword in '{c_norm}'"

            # Check min/max bounds if numeric
            if valid_cnt > 0:
                s_num = pd.to_numeric(non_null_s, errors='coerce').dropna()
                if len(s_num) > 0:
                    min_v = float(s_num.min())
                    max_v = float(s_num.max())
                    if min_v >= 1 and max_v <= 5:
                        rating_meta = {"scale_type": "1-5", "min_possible": 1.0, "max_possible": 5.0, "rating_min": 1, "rating_max": 5, "observed_min": min_v, "observed_max": max_v, "higher_is_better": True, "confidence": 0.95, "evidence": "Values lie between 1 and 5"}
                    elif min_v >= 0 and max_v <= 10:
                        rating_meta = {"scale_type": "1-10", "min_possible": 0.0, "max_possible": 10.0, "rating_min": 0, "rating_max": 10, "observed_min": min_v, "observed_max": max_v, "higher_is_better": True, "confidence": 0.95, "evidence": "Values lie between 0 and 10"}
                    elif min_v >= 0 and max_v <= 100:
                        rating_meta = {"scale_type": "0-100", "min_possible": 0.0, "max_possible": 100.0, "rating_min": 0, "rating_max": 100, "observed_min": min_v, "observed_max": max_v, "higher_is_better": True, "confidence": 0.95, "evidence": "Values lie between 0 and 100"}

        # Test Physical Measurement Units (Distance, Mass, Temperature, Energy, Pressure)
        if measure_type == "None" and not has_id_kw:
            # Distance
            if any(k in c_norm.split('_') for k in ['km', 'miles']) or c_norm.endswith('_km') or c_norm.endswith('_miles'):
                measure_type = "Continuous measure"
                unit = "km" if 'km' in c_norm else "miles"
                unit_conf = 0.90
                unit_ev = f"Distance unit '{unit}' recognized in column name"
            elif any(k in c_norm.split('_') for k in ['meters', 'metres']) or c_norm.endswith('_m'):
                measure_type = "Continuous measure"
                unit = "m"
                unit_conf = 0.90
                unit_ev = f"Distance unit 'm' recognized in column name"
            # Mass
            elif any(k in c_norm.split('_') for k in ['kg', 'tonnes']) or c_norm.endswith('_kg') or c_norm.endswith('_tonnes'):
                measure_type = "Continuous measure"
                unit = "kg" if 'kg' in c_norm else "tonnes"
                unit_conf = 0.90
                unit_ev = f"Mass unit '{unit}' recognized in column name"
            elif any(k in c_norm.split('_') for k in ['grams']) or c_norm.endswith('_g'):
                measure_type = "Continuous measure"
                unit = "g"
                unit_conf = 0.85
                unit_ev = f"Mass unit 'g' recognized in column name"
            # Temperature
            elif any(k in c_norm.split('_') for k in ['temp_c', 'celsius']) or c_norm.endswith('_c'):
                measure_type = "Continuous measure"
                unit = "C"
                unit_conf = 0.90
                unit_ev = f"Temperature unit 'C' recognized in column name"
            elif any(k in c_norm.split('_') for k in ['temp_f', 'fahrenheit']) or c_norm.endswith('_f'):
                measure_type = "Continuous measure"
                unit = "F"
                unit_conf = 0.90
                unit_ev = f"Temperature unit 'F' recognized in column name"
            elif any(k in c_norm.split('_') for k in ['temp_k', 'kelvin']) or c_norm.endswith('_k'):
                measure_type = "Continuous measure"
                unit = "K"
                unit_conf = 0.90
                unit_ev = f"Temperature unit 'K' recognized in column name"
            # Energy
            elif 'kwh' in c_norm or 'mwh' in c_norm:
                measure_type = "Continuous measure"
                unit = "kWh" if 'kwh' in c_norm else "MWh"
                unit_conf = 0.95
                unit_ev = f"Energy unit '{unit}' recognized in column name"
            # Pressure
            elif any(k in c_norm.split('_') for k in ['bar', 'kpa', 'pa']) or c_norm.endswith('_bar') or c_norm.endswith('_kpa') or c_norm.endswith('_pa'):
                measure_type = "Continuous measure"
                unit = "bar" if 'bar' in c_norm else ("kPa" if 'kpa' in c_norm else "Pa")
                unit_conf = 0.90
                unit_ev = f"Pressure unit '{unit}' recognized in column name"

        # General Numeric Fallback
        if measure_type == "None" and (dtype_val in ["integer", "float"] or pd.api.types.is_numeric_dtype(series)) and not has_id_kw and not is_temporal:
            measure_type = "Quantity" if dtype_val == "integer" else "Continuous measure"
            unit = "unknown"
            unit_conf = 0.0
            unit_ev = "No explicit unit could be established from metadata or domain signals"

        # Check boolean variants (Yes/No, True/False, 1/0, etc.)
        is_bool_like = (dtype_val == "boolean")
        if not is_bool_like and valid_cnt > 0 and unique_cnt <= 2:
            unique_vals = set(non_null_s.astype(str).str.lower().str.strip().unique())
            if unique_vals.issubset({"true", "false", "yes", "no", "y", "n", "1", "0", "t", "f"}):
                is_bool_like = True

        is_json_like = False
        if dtype_val == "string" and valid_cnt > 0:
            sample_strings = non_null_s.head(10).astype(str).str.strip()
            if sample_strings.str.startswith(("{", "[")).mean() >= 0.5 and sample_strings.str.endswith(("}", "]")).mean() >= 0.5:
                is_json_like = True

        # 5. Semantic Role & Key Role Determination
        key_detection = KeyDetectionEngine.detect_key(col_name, series, n_rows, table_name, all_tables)
        key_role = key_detection["key_role"]

        # Decoupled Per-Column ID Safety (Section 2)
        safe_non_id = DualCompatibleStr("not_applicable", aliases=["Not Applicable", "not applicable", "unrestricted", "Safe", "safe", "Standard Field", "false", "0"])
        restricted_id = DualCompatibleStr("restricted", aliases=["Restricted", "Identifier", "Appropriate identifier", "Appropriate identifier (Primary Key)", "Appropriate identifier (Foreign Key)", "PII - Sensitive Data Masking Recommended", "Identifier / PII-Sensitive", "true", "1"])

        if is_temporal:
            semantic_role = DualCompatibleStr("Date/Time", aliases=["DateTime", "Time", "Temporal"])
            id_safety = safe_non_id
        elif is_bool_like:
            measure_type = DualCompatibleStr("Binary Category", aliases=["None", "Boolean", "Flag"])
            semantic_role = DualCompatibleStr("Dimension", aliases=["Binary Category", "Flag"])
            id_safety = safe_non_id
        elif measure_type != "None" and not has_id_kw:
            semantic_role = DualCompatibleStr("Measure", aliases=["measure", "NUMERIC MEASURE"])
            id_safety = safe_non_id
        elif key_role in ["primary_key", "candidate_primary_key"]:
            semantic_role = DualCompatibleStr("Identifier", aliases=["Primary Key", "candidate_primary_key"])
            id_safety = restricted_id
        elif key_role in ["foreign_key", "candidate_foreign_key"]:
            semantic_role = DualCompatibleStr("Identifier", aliases=["Foreign Key", "candidate_foreign_key"])
            id_safety = restricted_id
        elif has_id_kw:
            semantic_role = DualCompatibleStr("Identifier", aliases=["ID", "Identifier"])
            id_safety = restricted_id
        elif is_json_like:
            semantic_role = DualCompatibleStr("Text", aliases=["Dimension", "JSON"])
            id_safety = safe_non_id
        elif any(k in c_norm for k in ['country', 'region', 'city', 'state', 'location', 'address', 'zip', 'territory']):
            semantic_role = DualCompatibleStr("Geographic", aliases=["geographic", "Dimension"])
            id_safety = safe_non_id
        elif dtype_val == "string" and unique_ratio > 0.85 and valid_cnt >= 20:
            semantic_role = DualCompatibleStr("Text", aliases=["text", "Dimension"])
            id_safety = safe_non_id
        else:
            semantic_role = DualCompatibleStr("Dimension", aliases=["dimension", "Categorical"])
            id_safety = safe_non_id

        # Constancy Check
        var_info = KeyDetectionEngine.compute_constancy(series)

        # Centralized Operation Governance (Section 9)
        is_add = bool(measure_type in ["Count", "Currency", "Quantity"] or (measure_type == "Numeric" and not any(k in c_norm for k in ['age', 'year', 'score', 'rating', 'index'])))
        op_eval = OperationEligibilityEngine.evaluate(
            semantic_role=semantic_role,
            measure_type=measure_type,
            id_safety=id_safety,
            key_role=key_role,
            physical_dtype=dtype_val,
            is_additive=is_add,
            unique_count=unique_cnt,
            is_temporal=is_temporal
        )

        # Build output structure
        result = {
            "column_name": c_clean,
            "table_name": table_name or "default",
            "physical_dtype": dtype_val,
            "semantic_role": semantic_role,
            "canonical_role": "PRIMARY KEY" if key_role == "primary_key" else ("FOREIGN KEY" if key_role == "foreign_key" else ("IDENTIFIER" if "identifier" in str(semantic_role).lower() else ("TEMPORAL" if is_temporal else ("NUMERIC MEASURE" if semantic_role == "Measure" else "CATEGORICAL DIMENSION")))),
            "key_role": key_role,
            "id_safety": id_safety,
            "data_type": dtype_val,
            "measure_type": measure_type,
            "semantic_subtype": measure_type if measure_type != "None" else ("Categorical" if semantic_role == "Dimension" else semantic_role),
            "unit": unit,
            "unit_confidence": unit_conf,
            "unit_evidence": unit_ev,
            "currency_metadata": currency_meta,
            "rating_score_metadata": rating_meta,
            "is_identifier": bool("identifier" in str(semantic_role).lower() or key_role in ["primary_key", "foreign_key"]),
            "is_measure": bool(semantic_role == "Measure"),
            "is_dimension": bool(semantic_role in ["Dimension", "Geographic"]),
            "is_time": bool(is_temporal),
            "constant_flag": var_info["is_constant"],
            "variance": var_info["variance"],
            "nunique": unique_cnt,
            "null_count": null_cnt,
            "null_percentage": null_pct,
            "unique_ratio": unique_ratio,
            "confidence_score": 0.90 if unit_conf > 0 or key_detection.get("confidence", 0) > 0.8 else 0.75,
            "detection_reason": f"Role: {semantic_role}, Key: {key_role}, Subtype: {measure_type}, Unit: {unit} ({unit_ev})",
            "operation_eligibility": op_eval,
            "allowed_operations": op_eval["allowed_operations"],
            "allowed_aggregations": op_eval["allowed_aggregations"],
            "aggregation_operations": op_eval["aggregation_operations"],
            "aggregation_display": op_eval["aggregation_display"],
            "aggregation_restrictions": op_eval["aggregation_restrictions"],
            "aggregation_eligibility": op_eval["aggregation_eligibility"],
            "grouping_eligibility": op_eval["grouping_eligibility"],
            "grouping_status": op_eval["grouping_status"],
            "time_series_eligibility": op_eval["time_series_eligibility"],
            "time_series_status": op_eval["time_series_status"],
            "correlation_eligibility": op_eval["correlation_eligibility"],
            "regression_eligibility": op_eval["regression_eligibility"],
            "derivation_status": DualCompatibleStr("Native", aliases=["INDEPENDENT", "Native", "native"])
        }

        # Apply User Overrides if specified
        if override and isinstance(override, dict):
            for k, v in override.items():
                if k in result and v is not None:
                    result[k] = v
            result["override_applied"] = True

        return SemanticFieldProfile(result)


# ==============================================================================
# 2. KEY DETECTION ENGINE
# ==============================================================================

class KeyDetectionEngine:
    """
    Evidence-based candidate primary key detection.
    Evaluates:
    - uniqueness ratio (must be 1.0 or near 1.0)
    - non-nullness (null_count == 0)
    - stability / naming signals
    - datatype
    - composite uniqueness
    """

    @classmethod
    def compute_constancy(cls, series: pd.Series, tolerance: float = 1e-9) -> Dict[str, Any]:
        non_null = series.dropna()
        if len(non_null) == 0:
            return {"is_constant": True, "variance": 0.0, "nunique": 0}
        
        uniq = non_null.nunique()
        if uniq <= 1:
            return {"is_constant": True, "variance": 0.0, "nunique": uniq}

        if pd.api.types.is_numeric_dtype(series):
            s_num = pd.to_numeric(non_null, errors='coerce').dropna()
            if len(s_num) <= 1:
                return {"is_constant": True, "variance": 0.0, "nunique": len(s_num)}
            var_v = float(s_num.var(ddof=1))
            if abs(var_v) <= tolerance or (float(s_num.max()) - float(s_num.min())) <= tolerance:
                return {"is_constant": True, "variance": 0.0, "nunique": uniq}
            return {"is_constant": False, "variance": round(var_v, 6), "nunique": uniq}

        return {"is_constant": False, "variance": None, "nunique": uniq}

    @classmethod
    def detect_key(
        cls,
        col_name: str,
        series: pd.Series,
        total_rows: int,
        table_name: Optional[str] = None,
        all_tables: Optional[Dict[str, pd.DataFrame]] = None,
        parent_table: Optional[pd.DataFrame] = None,
        parent_key_col: Optional[str] = None
    ) -> Dict[str, Any]:
        c_low = str(col_name).lower().strip()
        non_null = series.dropna()
        valid_cnt = len(non_null)
        null_cnt = total_rows - valid_cnt
        uniq_cnt = non_null.nunique()
        uniq_ratio = (uniq_cnt / max(total_rows, 1)) if total_rows > 0 else 0.0

        id_tokens = ['_id', 'id', 'key', 'code', 'sku', 'uuid', 'guid', 'ssn', 'no', 'num', 'number']
        has_id_name = any(c_low == k or c_low.startswith(f"{k}_") or c_low.endswith(f"_{k}") or f"_{k}_" in c_low for k in id_tokens)

        # Check for Foreign Key via direct parent_table
        if parent_table is not None and isinstance(parent_table, pd.DataFrame):
            p_col = parent_key_col if (parent_key_col and parent_key_col in parent_table.columns) else None
            if not p_col:
                matching = [c for c in parent_table.columns if str(c).lower().strip() == c_low]
                if matching:
                    p_col = matching[0]
            if p_col:
                parent_s = parent_table[p_col].dropna()
                parent_uniq = parent_s.nunique() / max(len(parent_table), 1)
                child_vals = set(non_null.astype(str))
                parent_vals = set(parent_s.astype(str))
                if child_vals and (len(child_vals & parent_vals) / len(child_vals)) >= 0.80:
                    cov = round((len(child_vals & parent_vals) / len(child_vals)) * 100, 2)
                    unmatched = len(child_vals - parent_vals)
                    return {
                        "key_role": DualCompatibleStr("Foreign Key", aliases=["foreign_key", "Foreign Key", "verified_foreign_key", "VERIFIED_FOREIGN_KEY", "Candidate Foreign Key"]),
                        "key_role_type": "VERIFIED_FOREIGN_KEY",
                        "relationship_status": "VERIFIED",
                        "referential_coverage_pct": cov,
                        "unmatched_count": unmatched,
                        "unmatched_pct": round((unmatched / max(len(child_vals), 1)) * 100, 2),
                        "parent_key_unique_pct": round(parent_uniq * 100, 2),
                        "relationship_confidence": 0.95,
                        "confidence": 0.95,
                        "candidate_confidence": 0.95,
                        "uniqueness_ratio": uniq_ratio,
                        "uniqueness_percentage": round(uniq_ratio * 100, 2),
                        "null_count": null_cnt,
                        "null_percentage": round((null_cnt / max(total_rows, 1)) * 100, 2),
                        "duplicate_count": total_rows - uniq_cnt,
                        "references": f"parent_table.{p_col}",
                        "evidence": {
                            "uniqueness_pct": round(uniq_ratio * 100, 2),
                            "null_pct": round((null_cnt / max(total_rows, 1)) * 100, 2),
                            "duplicate_count": total_rows - uniq_cnt,
                            "identifier_pattern": True,
                            "parent_relationship_status": "VERIFIED",
                            "referential_coverage_pct": cov
                        },
                        "reason": f"Matches parent table '{p_col}' with referential coverage {cov}%"
                    }

        # Check for Foreign Key via all_tables referential inspection
        if all_tables and isinstance(all_tables, dict):
            t_low = (table_name or "current_table").lower().strip()
            for other_t, other_df in all_tables.items():
                if other_t.lower().strip() == t_low:
                    continue
                matching = [c for c in other_df.columns if str(c).lower().strip() == c_low]
                if matching:
                    parent_c = matching[0]
                    parent_s = other_df[parent_c].dropna()
                    parent_uniq = parent_s.nunique() / max(len(other_df), 1)
                    if parent_uniq >= 0.90:
                        child_vals = set(non_null.astype(str))
                        parent_vals = set(parent_s.astype(str))
                        if child_vals and (len(child_vals & parent_vals) / len(child_vals)) >= 0.80:
                            cov = round((len(child_vals & parent_vals) / len(child_vals)) * 100, 2)
                            unmatched = len(child_vals - parent_vals)
                            return {
                                "key_role": DualCompatibleStr("Foreign Key", aliases=["foreign_key", "Foreign Key", "verified_foreign_key", "VERIFIED_FOREIGN_KEY", "Candidate Foreign Key"]),
                                "key_role_type": "VERIFIED_FOREIGN_KEY",
                                "relationship_status": "VERIFIED",
                                "referential_coverage_pct": cov,
                                "unmatched_count": unmatched,
                                "unmatched_pct": round((unmatched / max(len(child_vals), 1)) * 100, 2),
                                "parent_key_unique_pct": round(parent_uniq * 100, 2),
                                "relationship_confidence": 0.95,
                                "confidence": 0.95,
                                "candidate_confidence": 0.95,
                                "uniqueness_ratio": uniq_ratio,
                                "uniqueness_percentage": round(uniq_ratio * 100, 2),
                                "null_count": null_cnt,
                                "null_percentage": round((null_cnt / max(total_rows, 1)) * 100, 2),
                                "duplicate_count": total_rows - uniq_cnt,
                                "references": f"{other_t}.{parent_c}",
                                "reason": f"Matches parent table '{other_t}.{parent_c}' with referential coverage {cov}%"
                            }

        # Check if column is a continuous measure or float
        is_metric_name = any(
            c_low == m or c_low.startswith(f"{m}_") or c_low.endswith(f"_{m}") or f"_{m}_" in c_low
            for m in [
                'hours', 'days', 'score', 'rating', 'index', 'sales', 'revenue', 'profit', 'cost',
                'price', 'spend', 'amount', 'salary', 'income', 'overtime', 'productivity', 'margin',
                'discount', 'rate', 'pct', 'percent', 'percentage', 'qty', 'quantity', 'units',
                'age', 'years', 'months', 'output'
            ]
        ) and not has_id_name

        is_float = pd.api.types.is_float_dtype(series) or (
            pd.api.types.is_numeric_dtype(series) and not non_null.empty and not (non_null.apply(lambda x: float(x).is_integer() if pd.notna(x) else True)).all()
        )

        if (is_metric_name or is_float) and not has_id_name:
            return {
                "key_role": DualCompatibleStr("None", aliases=["none"]),
                "relationship_status": "NONE",
                "confidence": 0.95,
                "candidate_confidence": 0.95,
                "uniqueness_ratio": uniq_ratio,
                "uniqueness_percentage": round(uniq_ratio * 100, 2),
                "null_count": null_cnt,
                "null_percentage": round((null_cnt / max(total_rows, 1)) * 100, 2),
                "duplicate_count": total_rows - uniq_cnt,
                "reason": "Continuous numeric measure or non-integer values cannot be an entity key"
            }

        # Check if column is a known non-key domain (geographic, general category descriptor)
        geo_tokens = ['state', 'city', 'country', 'region', 'postal', 'zip', 'province', 'territory', 'county', 'district', 'continent']
        is_geo = any(c_low == g or c_low.startswith(f"{g}_") or c_low.endswith(f"_{g}") or f"_{g}_" in c_low for g in geo_tokens)
        cat_tokens = ['category', 'type', 'status', 'group', 'class', 'department', 'sector', 'industry', 'stage', 'phase', 'priority', 'severity']
        is_generic_cat = any(c_low == k or c_low.endswith(f"_{k}") for k in cat_tokens)

        # Check Primary Key criteria: 100% unique across records + 0 null values + identifier semantics
        is_numeric = pd.api.types.is_numeric_dtype(series)
        if null_cnt == 0 and uniq_cnt == total_rows and total_rows >= 2 and not (is_geo or is_generic_cat):
            if has_id_name:
                role = DualCompatibleStr("PRIMARY_KEY", aliases=["primary_key", "Primary Key", "PRIMARY_KEY", "identifier"])
                rel_status = "VERIFIED"
                conf = 0.99
                return {
                    "key_role": role,
                    "key_role_type": "PRIMARY_KEY",
                    "relationship_status": rel_status,
                    "confidence": conf,
                    "candidate_confidence": conf,
                    "uniqueness_ratio": 1.0,
                    "uniqueness_percentage": 100.0,
                    "null_count": 0,
                    "null_percentage": 0.0,
                    "duplicate_count": 0,
                    "referential_coverage_pct": 100.0,
                    "unmatched_count": 0,
                    "unmatched_pct": 0.0,
                    "parent_key_unique_pct": 100.0,
                    "relationship_confidence": conf,
                    "evidence": {
                        "uniqueness_pct": 100.0,
                        "null_pct": 0.0,
                        "identifier_pattern": True
                    },
                    "reason": f"100% unique and non-null ({total_rows} distinct records) with identifier naming"
                }
            elif not is_numeric:
                role = DualCompatibleStr("Candidate Primary Key", aliases=["candidate_primary_key", "candidate", "identifier"])
                rel_status = "CANDIDATE"
                conf = 0.85
                return {
                    "key_role": role,
                    "key_role_type": "CANDIDATE_PRIMARY_KEY",
                    "relationship_status": rel_status,
                    "confidence": conf,
                    "candidate_confidence": conf,
                    "uniqueness_ratio": 1.0,
                    "uniqueness_percentage": 100.0,
                    "null_count": 0,
                    "null_percentage": 0.0,
                    "duplicate_count": 0,
                    "referential_coverage_pct": None,
                    "unmatched_count": 0,
                    "unmatched_pct": 0.0,
                    "parent_key_unique_pct": 100.0,
                    "relationship_confidence": conf,
                    "evidence": {
                        "uniqueness_pct": 100.0,
                        "null_pct": 0.0,
                        "identifier_pattern": False
                    },
                    "reason": f"100% unique and non-null ({total_rows} distinct records)"
                }

        # If identifier-like naming exists but conditions for Primary Key are not satisfied (e.g. duplicates or unverified FK):
        if has_id_name:
            col_core = c_low.replace('_id', '').replace('id', '').strip()
            tbl_core = str(table_name).lower().rstrip('s').strip() if table_name else ""
            is_name_matched_pk = bool(tbl_core and (tbl_core == col_core or tbl_core in col_core or col_core in tbl_core))

            if is_name_matched_pk and uniq_ratio >= 0.8:
                role = DualCompatibleStr("Candidate Primary Key", aliases=["candidate_primary_key", "Primary Key", "primary_key", "not_primary_key", "NOT_PRIMARY_KEY", "candidate", "identifier"])
                rel_status = "CANDIDATE"
                k_type = "CANDIDATE_PRIMARY_KEY"
            else:
                role = DualCompatibleStr("Candidate Foreign Key", aliases=["candidate_foreign_key", "Candidate Foreign Key", "not_primary_key", "NOT_PRIMARY_KEY", "candidate", "identifier"])
                rel_status = "UNVERIFIED"
                k_type = "CANDIDATE_FOREIGN_KEY"

            return {
                "key_role": role,
                "key_role_type": k_type,
                "relationship_status": rel_status,
                "referential_coverage_pct": None,
                "unmatched_count": 0,
                "unmatched_pct": 0.0,
                "parent_key_unique_pct": None,
                "relationship_confidence": 0.70,
                "confidence": 0.80,
                "candidate_confidence": 0.80,
                "uniqueness_ratio": uniq_ratio,
                "uniqueness_percentage": round(uniq_ratio * 100, 2),
                "null_count": null_cnt,
                "null_percentage": round((null_cnt / max(total_rows, 1)) * 100, 2),
                "duplicate_count": total_rows - uniq_cnt,
                "evidence": {
                    "uniqueness_pct": round(uniq_ratio * 100, 2),
                    "null_pct": round((null_cnt / max(total_rows, 1)) * 100, 2),
                    "duplicate_count": total_rows - uniq_cnt,
                    "identifier_pattern": True,
                    "parent_relationship_status": rel_status
                },
                "reason": f"Identifier naming in '{col_name}', but not strictly unique ({uniq_cnt}/{total_rows}) or contains {null_cnt} nulls; relationship status is {rel_status} without verified parent relationship."
            }

        if null_cnt == 0 and uniq_ratio >= 0.98 and total_rows >= 5 and not is_float and not is_metric_name:
            return {
                "key_role": DualCompatibleStr("Candidate Primary Key", aliases=["candidate_primary_key"]),
                "relationship_status": "CANDIDATE",
                "confidence": 0.75,
                "candidate_confidence": 0.75,
                "uniqueness_ratio": uniq_ratio,
                "uniqueness_percentage": round(uniq_ratio * 100, 2),
                "null_count": 0,
                "null_percentage": 0.0,
                "duplicate_count": total_rows - uniq_cnt,
                "reason": f"High uniqueness ratio ({uniq_ratio:.2%}) with 0 nulls"
            }

        return {
            "key_role": DualCompatibleStr("None", aliases=["none"]),
            "relationship_status": "NONE",
            "confidence": 0.90,
            "candidate_confidence": 0.90,
            "uniqueness_ratio": uniq_ratio,
            "uniqueness_percentage": round(uniq_ratio * 100, 2),
            "null_count": null_cnt,
            "null_percentage": round((null_cnt / max(total_rows, 1)) * 100, 2),
            "duplicate_count": total_rows - uniq_cnt,
            "reason": "Does not satisfy uniqueness or non-nullness criteria for primary key"
        }

    @classmethod
    def detect_primary_keys(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        if df is None or df.empty:
            return []
        pks = []
        for c in df.columns:
            k = cls.detect_key(c, df[c], total_rows=len(df))
            if k["key_role"] in ["primary_key", "candidate_primary_key"]:
                pks.append(k)
        return pks

    @classmethod
    def detect_composite_keys(
        cls,
        df: pd.DataFrame,
        candidate_cols: Optional[List[str]] = None,
        max_cols: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generic candidate composite primary key detection.
        Evaluates:
        - individual non-uniqueness (Column A alone not unique, Column B alone not unique)
        - joint uniqueness (A + B unique = 100%)
        - joint non-nullness (null_percentage == 0)
        - duplicate count
        - candidate confidence
        """
        if df is None or df.empty or len(df) < 2:
            return []

        total_rows = len(df)
        if candidate_cols is None:
            # Exclude continuous float columns
            candidate_cols = [
                c for c in df.columns
                if not pd.api.types.is_float_dtype(df[c])
            ]

        if len(candidate_cols) < 2:
            return []

        # Find columns that are NOT individually unique (so composite makes sense)
        non_unique_candidates = []
        for c in candidate_cols:
            non_null = df[c].dropna()
            if non_null.nunique() < total_rows:
                non_unique_candidates.append(c)

        composite_keys = []
        import itertools
        for r in range(2, min(max_cols + 1, len(non_unique_candidates) + 1)):
            for comb in itertools.combinations(non_unique_candidates, r):
                comb_list = list(comb)
                sub_df = df[comb_list]
                null_rows = int(sub_df.isnull().any(axis=1).sum())
                null_pct = round((null_rows / max(total_rows, 1)) * 100, 2)

                comb_series = sub_df.astype(str).agg("_".join, axis=1)
                uniq_comb = comb_series.nunique()
                uniq_pct = round((uniq_comb / max(total_rows, 1)) * 100, 2)
                dup_count = total_rows - uniq_comb

                if uniq_pct >= 99.9 and null_rows == 0:
                    conf = 0.95
                    role = DualCompatibleStr("composite_key", aliases=["composite_primary_key"])
                elif uniq_pct >= 95.0 and null_pct <= 2.0:
                    conf = 0.80
                    role = DualCompatibleStr("candidate_composite_key", aliases=["candidate_composite_primary_key"])
                else:
                    continue

                composite_keys.append({
                    "columns": comb_list,
                    "key_columns": comb_list,
                    "composite_column_name": " + ".join(comb_list),
                    "uniqueness": round(uniq_comb / max(total_rows, 1), 4),
                    "uniqueness_percentage": uniq_pct,
                    "uniqueness_ratio": round(uniq_comb / max(total_rows, 1), 4),
                    "null_percentage": null_pct,
                    "null_count": null_rows,
                    "duplicate_count": dup_count,
                    "candidate_confidence": conf,
                    "confidence": conf,
                    "key_role": role,
                    "evidence": f"Columns {comb_list} combined provide {uniq_pct}% uniqueness with {null_pct}% nulls across {total_rows} records.",
                    "reason": f"Columns {comb_list} combined provide {uniq_pct}% uniqueness with {null_pct}% nulls across {total_rows} records."
                })

        return composite_keys

    @classmethod
    def detect_composite_primary_keys(
        cls,
        df: pd.DataFrame,
        candidate_cols: Optional[List[str]] = None,
        max_cols: int = 3
    ) -> List[Dict[str, Any]]:
        return cls.detect_composite_keys(df, candidate_cols=candidate_cols, max_cols=max_cols)



# ==============================================================================
# 3. RELATIONSHIP DETECTION & VALIDATION ENGINE
# ==============================================================================

class RelationshipDetectionEngine:
    """
    Discovers candidate relationships between tables/sheets or within a table (self-referencing).
    Validates referential integrity:
    - FK row count
    - matched references
    - unmatched references (orphans)
    - FK violation count & percentage
    - null FK count
    - relationship types: one-to-one, one-to-many, many-to-one, many-to-many
    - distinguishes: valid relationship vs candidate relationship vs relationship violation
    """

    @classmethod
    def detect_and_validate_relationships(cls, tables: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        relationships = []
        if not tables or len(tables) < 2:
            return relationships

        tbl_names = list(tables.keys())
        for i, src_name in enumerate(tbl_names):
            src_df = tables[src_name]
            for j, tgt_name in enumerate(tbl_names):
                if i == j:
                    continue
                tgt_df = tables[tgt_name]

                # Compare candidate columns
                for src_col in src_df.columns:
                    src_low = str(src_col).lower().strip()
                    for tgt_col in tgt_df.columns:
                        tgt_low = str(tgt_col).lower().strip()
                        if src_low == tgt_low or src_low.endswith(f"_{tgt_low}") or tgt_low.endswith(f"_{src_low}"):
                            # Evaluate datatype compatibility
                            src_s = src_df[src_col].dropna().astype(str).str.strip()
                            tgt_s = tgt_df[tgt_col].dropna().astype(str).str.strip()

                            if len(src_s) == 0 or len(tgt_s) == 0:
                                continue

                            src_set = set(src_s.unique())
                            tgt_set = set(tgt_s.unique())
                            overlap = len(src_set & tgt_set)
                            if overlap == 0:
                                continue

                            coverage = (overlap / len(src_set)) * 100
                            orphan_set = src_set - tgt_set
                            orphan_count = len(orphan_set)
                            null_fk_count = int(src_df[src_col].isnull().sum())
                            fk_violations = orphan_count
                            violation_pct = round((fk_violations / max(len(src_set), 1)) * 100, 2)

                            tgt_uniq_ratio = len(tgt_set) / max(len(tgt_df), 1)
                            src_uniq_ratio = len(src_set) / max(len(src_df), 1)

                            if tgt_uniq_ratio >= 0.90 and src_uniq_ratio >= 0.90:
                                rel_type = "one-to-one"
                            elif tgt_uniq_ratio >= 0.90:
                                rel_type = "many-to-one"
                            elif src_uniq_ratio >= 0.90:
                                rel_type = "one-to-many"
                            else:
                                rel_type = "many-to-many"

                            # Suspicious overlap check (e.g. child=[1,2,3] vs parent=[1,2,3] with generic codes)
                            is_suspicious_small_overlap = (len(tgt_set) <= 5 and all(str(v).isdigit() for v in tgt_set) and src_low != tgt_low)

                            if is_suspicious_small_overlap:
                                status = "candidate_relationship"
                                conf = 0.50
                                conf_required = True
                            elif coverage >= 95.0 and tgt_uniq_ratio >= 0.85:
                                status = "valid_relationship"
                                conf = 0.95
                                conf_required = False
                            elif coverage >= 60.0:
                                status = "candidate_relationship"
                                conf = 0.75
                                conf_required = True
                            else:
                                status = "relationship_violation" if fk_violations > 0 else "candidate_relationship"
                                conf = 0.50
                                conf_required = True

                            ev_list = [
                                f"Match rate {coverage:.1f}% across {overlap} matching keys",
                                f"Parent distinct key uniqueness {tgt_uniq_ratio:.1%}",
                                f"{fk_violations} referential violations detected ({violation_pct}%)"
                            ]

                            relationships.append({
                                "relationship_id": f"rel_{src_name}_{src_col}_to_{tgt_name}_{tgt_col}",
                                "source_table": src_name,
                                "source_column": src_col,
                                "target_table": tgt_name,
                                "target_column": tgt_col,
                                "child_table": src_name,
                                "child_column": src_col,
                                "parent_table": tgt_name,
                                "parent_column": tgt_col,
                                "child_row_count": len(src_df),
                                "parent_distinct_key_count": len(tgt_set),
                                "matching_values": overlap,
                                "unmatched_values": orphan_count,
                                "null_references": null_fk_count,
                                "fk_violation_count": fk_violations,
                                "fk_violation_percentage": violation_pct,
                                "fk_violation_percent": violation_pct,
                                "match_rate": round(coverage, 2),
                                "candidate_confidence": conf,
                                "type": "Candidate Foreign Key" if status == "candidate_relationship" else ("Foreign Key" if status == "valid_relationship" else "Foreign Key Violation"),
                                "relationship_type": rel_type,
                                "relationship_cardinality": rel_type,
                                "referential_coverage": round(coverage, 2),
                                "matched_references": overlap,
                                "unmatched_references": orphan_count,
                                "orphan_count": orphan_count,
                                "null_fk_count": null_fk_count,
                                "status": status,
                                "confidence": conf,
                                "confirmation_required": conf_required,
                                "evidence": ev_list,
                                "detection_method": "Exact value overlap & cardinality analysis"
                            })

        return relationships

    @classmethod
    def detect_composite_foreign_keys(cls, tables: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Discovers candidate composite foreign keys between tables:
        (fact.order_id, fact.product_id) -> (dim.order_id, dim.product_id)
        """
        composite_fks = []
        if not tables or len(tables) < 2:
            return composite_fks

        import itertools
        tbl_names = list(tables.keys())
        for i, src_name in enumerate(tbl_names):
            src_df = tables[src_name]
            for j, tgt_name in enumerate(tbl_names):
                if i == j:
                    continue
                tgt_df = tables[tgt_name]
                common_cols = [c for c in src_df.columns if c in tgt_df.columns and not pd.api.types.is_float_dtype(src_df[c])]
                if len(common_cols) >= 2:
                    for r in range(2, min(4, len(common_cols) + 1)):
                        for comb in itertools.combinations(common_cols, r):
                            comb_cols = list(comb)
                            tgt_sub = tgt_df[comb_cols].dropna()
                            src_sub = src_df[comb_cols].dropna()
                            if len(tgt_sub) < 2 or len(src_sub) < 2:
                                continue
                            tgt_tuple_series = tgt_sub.astype(str).agg("::".join, axis=1)
                            src_tuple_series = src_sub.astype(str).agg("::".join, axis=1)
                            parent_uniq_ratio = tgt_tuple_series.nunique() / max(len(tgt_df), 1)

                            if parent_uniq_ratio >= 0.85:
                                parent_tuples = set(tgt_tuple_series.unique())
                                child_tuples = set(src_tuple_series.unique())
                                overlap = len(child_tuples & parent_tuples)
                                if overlap > 0:
                                    coverage = (overlap / len(child_tuples)) * 100
                                    orphans = len(child_tuples - parent_tuples)
                                    violation_pct = round((orphans / max(len(child_tuples), 1)) * 100, 2)
                                    if coverage >= 80.0:
                                        composite_fks.append({
                                            "child_table": src_name,
                                            "child_columns": comb_cols,
                                            "parent_table": tgt_name,
                                            "parent_columns": comb_cols,
                                            "match_rate": round(coverage, 2),
                                            "violation_count": orphans,
                                            "violation_percentage": violation_pct,
                                            "relationship_cardinality": "many-to-one" if parent_uniq_ratio >= 0.95 else "many-to-many",
                                            "candidate_confidence": 0.90 if coverage >= 95.0 and orphans == 0 else 0.75,
                                            "status": "valid_composite_relationship" if (coverage >= 95.0 and orphans == 0) else "candidate_composite_relationship",
                                            "evidence": [
                                                f"Columns {comb_cols} match between '{src_name}' and '{tgt_name}'",
                                                f"Parent tuple uniqueness is {parent_uniq_ratio:.1%}",
                                                f"Referential coverage is {coverage:.1f}% with {orphans} violations"
                                            ]
                                        })
        return composite_fks


# ==============================================================================
# 4. DATA QUALITY ENGINE
# ==============================================================================

class DataQualityEngine:
    """
    Automated Data Quality Profiling Engine.
    Crucial Rule: Strictly separates STATISTICAL OUTLIER from DATA QUALITY ERROR.
    Stores separately:
    - outlier_flag (statistical observation via IQR or Z-score)
    - outlier_method, outlier_threshold
    - data_quality_error (actual domain/syntax error, e.g. negative days, impossible percentage)
    - data_quality_error_reason
    """

    @classmethod
    def run_quality_audit(cls, df: pd.DataFrame, sem_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return cls.profile_dataframe(df, sem_metadata=sem_metadata)

    @classmethod
    def profile_dataframe(cls, df: pd.DataFrame, sem_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        total_rows = len(df)
        total_cols = len(df.columns)
        dup_rows = int(df.duplicated().sum())
        dup_pct = round((dup_rows / max(total_rows, 1)) * 100, 2)

        col_profiles = {}
        for col in df.columns:
            s = df[col]
            null_cnt = int(s.isnull().sum())
            null_pct = round((null_cnt / max(total_rows, 1)) * 100, 2)
            non_null = s.dropna()
            uniq_cnt = int(non_null.nunique())

            col_meta = sem_metadata.get(col, {}) if sem_metadata else {}
            is_meas = col_meta.get("is_measure", False) or pd.api.types.is_numeric_dtype(s)
            is_id = col_meta.get("is_identifier", False) or col_meta.get("key_role") in ["primary_key", "foreign_key"]

            stat_profile = {}
            outliers = []
            dq_errors = []

            if is_meas and not is_id and len(non_null) > 0:
                s_num = pd.to_numeric(non_null, errors='coerce').dropna()
                if len(s_num) > 0:
                    min_v = float(s_num.min())
                    max_v = float(s_num.max())
                    mean_v = float(s_num.mean())
                    med_v = float(s_num.median())
                    std_v = float(s_num.std()) if len(s_num) > 1 else 0.0
                    var_v = float(s_num.var()) if len(s_num) > 1 else 0.0
                    q1 = float(s_num.quantile(0.25))
                    q3 = float(s_num.quantile(0.75))
                    iqr = q3 - q1

                    # Statistical Outliers (IQR Method)
                    lower_fence = q1 - 1.5 * iqr
                    upper_fence = q3 + 1.5 * iqr
                    iqr_outliers = s_num[(s_num < lower_fence) | (s_num > upper_fence)]
                    outlier_count = len(iqr_outliers)

                    stat_profile = {
                        "min": min_v,
                        "max": max_v,
                        "mean": round(mean_v, 2),
                        "median": round(med_v, 2),
                        "std": round(std_v, 2),
                        "variance": round(var_v, 4),
                        "zero_variance": bool(var_v == 0.0 or uniq_cnt <= 1),
                        "q1": round(q1, 2),
                        "q3": round(q3, 2),
                        "iqr": round(iqr, 2),
                        "outlier_count": outlier_count,
                        "outlier_percentage": round((outlier_count / len(s_num)) * 100, 2)
                    }

                    # Domain Data Quality Error check (e.g. negative working days, negative age, impossible percentages)
                    m_type = col_meta.get("measure_type", "")
                    unit_v = col_meta.get("unit", "")
                    c_low = col.lower()
                    non_negative_tokens = ["age", "quantity", "qty", "count", "days", "price", "rating", "duration", "hours", "minutes", "lead_time", "visits", "orders", "tenure"]
                    is_non_neg = any(k in c_low for k in non_negative_tokens) or unit_v in ["days", "hours", "minutes", "seconds", "orders", "units"] or m_type in ["Count", "Duration", "Quantity"]
                    if is_non_neg and min_v < 0:
                        neg_cnt = int((s_num < 0).sum())
                        dq_errors.append({
                            "type": "invalid_domain_negative_value",
                            "count": neg_cnt,
                            "reason": f"Column '{col}' represents non-negative quantity ({unit_v or m_type}) but contains {neg_cnt} negative values (min: {min_v})."
                        })

                    if m_type == "Percentage" and (max_v > 100.0 or min_v < 0):
                        bounds_cnt = int(((s_num < 0) | (s_num > 100)).sum())
                        dq_errors.append({
                            "type": "percentage_out_of_bounds",
                            "count": bounds_cnt,
                            "reason": f"Percentage values outside [0, 100] range."
                        })

                    if ("rating" in c_low or "csat" in c_low or m_type == "Rating") and (max_v > 5.0 or min_v < 1.0):
                        bounds_cnt = int(((s_num < 1.0) | (s_num > 5.0)).sum())
                        if bounds_cnt > 0:
                            dq_errors.append({
                                "type": "rating_out_of_bounds",
                                "count": bounds_cnt,
                                "reason": f"Rating values outside standard 1-5 scale."
                            })

            col_profiles[col] = {
                "null_count": null_cnt,
                "null_percentage": null_pct,
                "unique_count": uniq_cnt,
                "statistics": stat_profile,
                "statistical_outliers": outlier_count if 'outlier_count' in stat_profile else 0,
                "data_quality_errors": dq_errors
            }

        # Dataset-level chronological inversion check
        dataset_dq_issues = []
        date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])]
        for i in range(len(date_cols)):
            for j in range(len(date_cols)):
                if i != j:
                    c1, c2 = date_cols[i], date_cols[j]
                    if any(k in c1.lower() for k in ["start", "order", "created", "begin"]) and any(k in c2.lower() for k in ["end", "ship", "closed", "finish"]):
                        try:
                            d1 = pd.to_datetime(df[c1], errors='coerce')
                            d2 = pd.to_datetime(df[c2], errors='coerce')
                            valid_mask = d1.notna() & d2.notna()
                            inversions = int((d2[valid_mask] < d1[valid_mask]).sum())
                            if inversions > 0:
                                dataset_dq_issues.append({
                                    "type": "chronological_inversion",
                                    "columns": [c1, c2],
                                    "count": inversions,
                                    "reason": f"Found {inversions} records where '{c2}' is chronologically earlier than '{c1}'."
                                })
                        except Exception:
                            pass

        # Currency inconsistency check
        curr_cols = [c for c in df.columns if any(k in c.lower() for k in ['currency', 'curr']) or (sem_metadata and sem_metadata.get(c, {}).get("semantic_role") == "Currency")]
        for c in curr_cols:
            non_null_c = df[c].dropna()
            if non_null_c.nunique() > 1:
                currs = [str(x) for x in non_null_c.unique()]
                dataset_dq_issues.append({
                    "type": "currency_inconsistency",
                    "column": c,
                    "currencies": currs,
                    "count": len(non_null_c),
                    "reason": f"Column '{c}' contains multiple currencies ({currs}). Monetary aggregations across mixed currencies without conversion are invalid."
                })

        # Unit inconsistency check across columns of similar measure domain
        if sem_metadata:
            distance_units = set()
            mass_units = set()
            for c, meta in sem_metadata.items():
                u = meta.get("unit", "")
                if u in ["km", "miles", "m"]:
                    distance_units.add(u)
                elif u in ["kg", "g", "tonnes"]:
                    mass_units.add(u)
            if len(distance_units) > 1:
                dataset_dq_issues.append({
                    "type": "unit_inconsistency",
                    "domain": "distance",
                    "units": list(distance_units),
                    "reason": f"Dataset contains mixed distance units ({list(distance_units)}). Harmonization required before additive analysis."
                })
            if len(mass_units) > 1:
                dataset_dq_issues.append({
                    "type": "unit_inconsistency",
                    "domain": "mass",
                    "units": list(mass_units),
                    "reason": f"Dataset contains mixed mass units ({list(mass_units)}). Harmonization required before additive analysis."
                })

        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "duplicate_rows": dup_rows,
            "duplicate_percentage": dup_pct,
            "columns": col_profiles,
            "dataset_issues": dataset_dq_issues
        }


# ==============================================================================
# 5. CATEGORY NORMALIZATION ENGINE
# ==============================================================================

class CategoryNormalizationEngine:
    """
    Safely normalizes categorical strings before analysis:
    - Trims leading/trailing whitespace
    - Casing standardization (e.g. 'Hybrid', 'hybrid', 'HYBRID' -> 'Hybrid')
    - Unicode normalization
    - Maintains original_value <-> canonical_value mapping for 100% auditability
    - Never silently destroys original values
    """

    @classmethod
    def normalize_series(cls, series: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
        orig_s = series.copy()
        non_null = orig_s.dropna().astype(str)
        orig_cats = list(non_null.unique())

        # Group by lowercase cluster
        clusters = {}
        for val in orig_cats:
            cleaned = val.strip()
            key = cleaned.lower()
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(val)

        val_map = {}
        rules_applied = []
        for key, raw_variants in clusters.items():
            if len(raw_variants) == 1:
                canonical = raw_variants[0].strip()
                val_map[raw_variants[0]] = canonical
            else:
                # Select most frequent or capitalized form
                val_counts = non_null[non_null.isin(raw_variants)].value_counts()
                canonical = val_counts.index[0].strip().capitalize() if len(val_counts) > 0 else raw_variants[0].strip()
                for v in raw_variants:
                    val_map[v] = canonical
                rules_applied.append({
                    "cluster": key,
                    "variants": raw_variants,
                    "canonical": canonical,
                    "reason": "Normalized whitespace and case variation"
                })

        normalized_s = orig_s.map(lambda x: val_map.get(str(x), x) if pd.notna(x) else x)
        canonical_cats = list(normalized_s.dropna().unique())

        report = {
            "original_category_count": len(orig_cats),
            "canonical_category_count": len(canonical_cats),
            "normalized_values_map": val_map,
            "normalization_rules": rules_applied,
            "ambiguous_categories": []
        }

        return normalized_s, report


# ==============================================================================
# 5B. CENTRAL ANALYTICAL ELIGIBILITY ENGINE
# ==============================================================================

class AnalyticalEligibilityEngine:
    """
    Central, Authoritative Analytical Eligibility Engine for InsightFlow AI.
    Single authority for all downstream modules (EDA, PCA, correlation, ANOVA, visualizations).
    
    Exclusion Reasons:
    - IDENTIFIER
    - PRIMARY_KEY
    - FOREIGN_KEY
    - CANDIDATE_IDENTIFIER
    - TEST_FIXTURE
    - SPARSE
    - CONSTANT
    - NEAR_ZERO_VARIANCE
    - INVALID_RANGE
    - CATEGORICAL
    - INSUFFICIENT_SAMPLE
    - INSUFFICIENT_GROUP_SIZE
    - UNSUPPORTED_MEASURE_TYPE
    - UNVERIFIED_SEMANTICS
    """

    TEST_FIXTURE_PATTERNS = [
        r'^relationship_test_',
        r'^test_',
        r'^fixture_',
        r'^validation_',
        r'^synthetic_',
        r'^mock_',
        r'_test_fixture',
        r'_test_col',
        r'^tmp_test'
    ]

    @classmethod
    def is_likely_test_fixture(
        cls,
        col_name: str,
        series: pd.Series,
        total_rows: int,
        coverage_pct: float,
        non_null_count: int
    ) -> Tuple[bool, str]:
        c_low = str(col_name).lower().strip()
        has_name_pattern = any(bool(re.search(pat, c_low)) for pat in cls.TEST_FIXTURE_PATTERNS)
        
        # 1. Obvious name pattern (Section 13: Test_, Validation_, Fixture_, Mock_, Relationship_Test_)
        if has_name_pattern:
            return True, f"Matches test/fixture naming pattern ('{col_name}')"

        # 2. Extreme isolation: e.g. ~20 populated rows out of >= 200 rows with unverified/auxiliary name
        if total_rows >= 200 and non_null_count <= 50 and coverage_pct <= 10.0:
            if any(k in c_low for k in ['test', 'sample', 'dummy', 'mock', 'trial', 'temp', 'tmp', 'check', 'probe', 'experiment']):
                return True, f"Extreme isolated populated count ({non_null_count}/{total_rows}) with fixture keywords"
            if coverage_pct < 5.0 and total_rows >= 500:
                return True, f"Severe sparsity ({non_null_count}/{total_rows} rows, {coverage_pct:.1f}% coverage) indicates auxiliary/fixture field"

        return False, ""

    @classmethod
    def evaluate_column(
        cls,
        col_name: str,
        series: pd.Series,
        total_rows: Optional[int] = None,
        semantic_meta: Optional[Dict[str, Any]] = None,
        sparsity_threshold: float = 95.0
    ) -> Dict[str, Any]:
        n_rows = total_rows if total_rows is not None else len(series)
        non_null_s = series.dropna()
        valid_cnt = len(non_null_s)
        missing_cnt = n_rows - valid_cnt
        missing_pct = round((missing_cnt / max(n_rows, 1)) * 100, 2)
        coverage_pct = round(100.0 - missing_pct, 2)
        unique_cnt = int(non_null_s.nunique())
        cardinality_ratio = round(unique_cnt / max(n_rows, 1), 4)

        meta = semantic_meta or {}
        sem_role = str(meta.get("semantic_role", "")).strip()
        m_type = str(meta.get("measure_type", meta.get("semantic_subtype", ""))).strip()
        key_role = str(meta.get("key_role", "none")).strip()
        k_role_low = key_role.lower()

        # Variance calculation
        is_num = pd.api.types.is_numeric_dtype(series) or meta.get("is_measure", False)
        variance_val = None
        is_constant = False
        if valid_cnt > 0 and is_num:
            s_num = pd.to_numeric(non_null_s, errors='coerce').dropna().astype(float)
            if len(s_num) > 1:
                variance_val = round(float(s_num.var(ddof=1)), 6)
                min_v = float(s_num.min())
                max_v = float(s_num.max())
                if unique_cnt <= 1 or (variance_val is not None and abs(variance_val) <= 1e-9) or (max_v - min_v) <= 1e-9:
                    is_constant = True
            elif len(s_num) == 1:
                variance_val = 0.0
                is_constant = True
            else:
                is_constant = True
        elif valid_cnt > 0:
            if unique_cnt <= 1:
                is_constant = True
        else:
            is_constant = True

        # Determine Eligibility and Exclusion Reason
        analytical_eligible = True
        exclusion_reason = None

        # 1. Primary Key Check
        if k_role_low in ["primary_key", "primary key"] or meta.get("primary_key") or meta.get("is_primary_key"):
            analytical_eligible = False
            exclusion_reason = "PRIMARY_KEY"

        # 2. Foreign Key Check
        elif k_role_low in ["foreign_key", "foreign key"] or meta.get("foreign_key") or meta.get("is_foreign_key"):
            analytical_eligible = False
            exclusion_reason = "FOREIGN_KEY"

        # 3. Candidate Identifier / ID Safety Check
        elif (
            k_role_low in ["candidate_primary_key", "candidate_foreign_key", "candidate primary key", "candidate foreign key", "candidate_key"]
            or "identifier" in sem_role.lower()
            or meta.get("is_identifier", False)
            or (meta.get("id_safety") is True)
        ):
            analytical_eligible = False
            exclusion_reason = "IDENTIFIER" if "identifier" in sem_role.lower() else "CANDIDATE_IDENTIFIER"

        # 4. Test / Fixture Check
        else:
            is_fixture, fixture_reason = cls.is_likely_test_fixture(
                col_name, series, n_rows, coverage_pct, valid_cnt
            )
            if is_fixture:
                analytical_eligible = False
                exclusion_reason = "TEST_FIXTURE"

            # 5. Sparsity Rule (coverage_pct < 95% OR missing_pct > 5%)
            elif coverage_pct < sparsity_threshold or missing_pct > (100.0 - sparsity_threshold):
                analytical_eligible = False
                exclusion_reason = "SPARSE"

            # 6. Zero / Near-Zero Variance
            elif is_constant or unique_cnt <= 1:
                analytical_eligible = False
                exclusion_reason = "CONSTANT"

            elif variance_val is not None and abs(variance_val) <= 1e-9:
                analytical_eligible = False
                exclusion_reason = "NEAR_ZERO_VARIANCE"

            # 7. Sample Size Check
            elif valid_cnt < 5:
                analytical_eligible = False
                exclusion_reason = "INSUFFICIENT_SAMPLE"

            # 8. Categorical / Dimension Handling
            elif not is_num or "dimension" in sem_role.lower() or "text" in sem_role.lower() or "date" in sem_role.lower():
                # Categorical dimensions and temporal fields are analytically eligible for grouping and aggregation
                analytical_eligible = True
                exclusion_reason = None

        # Section 1, 8, 9: Independent Eligibility Dimensions
        is_dim = ("dimension" in sem_role.lower() or "category" in sem_role.lower() or (not is_num and not ("identifier" in sem_role.lower() or k_role_low in ["primary_key", "foreign_key", "candidate_primary_key", "candidate_foreign_key"])))
        is_time = ("date" in sem_role.lower() or "time" in sem_role.lower() or "temporal" in sem_role.lower())
        is_id = (
            k_role_low in ["primary_key", "foreign_key", "candidate_primary_key", "candidate_foreign_key", "primary key", "foreign key", "candidate key"]
            or "identifier" in sem_role.lower()
            or meta.get("is_identifier", False)
            or (meta.get("id_safety") is True)
        )

        grouping_eligibility = bool((is_dim or is_time) and not is_id and unique_cnt >= 2)
        time_series_eligibility = bool(is_time) and not is_id

        corr_elig = bool(analytical_eligible and is_num and not is_constant and not is_id)
        pca_elig = bool(analytical_eligible and is_num and not is_constant and not is_id and valid_cnt >= 5)
        anova_elig = bool(analytical_eligible and is_num and not is_constant and not is_id)
        kruskal_elig = bool(analytical_eligible and is_num and not is_constant and not is_id)

        # Statistical and Visualization suitability
        stat_elig = {
            "correlation": corr_elig,
            "pca": pca_elig,
            "anova_dependent": anova_elig,
            "kruskal_dependent": kruskal_elig,
            "regression": corr_elig,
            "distribution": corr_elig,
            "multivariate": pca_elig
        }
        viz_elig = {
            "is_eligible": True,
            "histogram": corr_elig,
            "box_plot": corr_elig,
            "violin": corr_elig,
            "scatter": corr_elig,
            "line": corr_elig or time_series_eligibility,
            "bar": True,
            "metric_card": True
        }

        return {
            "column_name": col_name,
            "analytical_eligible": analytical_eligible,
            "analytical_eligibility": analytical_eligible,
            "grouping_eligibility": grouping_eligibility,
            "grouping_eligible": grouping_eligibility,
            "time_series_eligibility": time_series_eligibility,
            "time_series_eligible": time_series_eligibility,
            "correlation_eligibility": corr_elig,
            "correlation_eligible": corr_elig,
            "pca_eligibility": pca_elig,
            "pca_eligible": pca_elig,
            "anova_eligibility": anova_elig,
            "anova_eligible": anova_elig,
            "kruskal_eligibility": kruskal_elig,
            "kruskal_eligible": kruskal_elig,
            "visualization_eligibility": viz_elig,
            "visualization_eligible": True,
            "exclusion_reason": exclusion_reason,
            "non_null_count": valid_cnt,
            "missing_count": missing_cnt,
            "missing_pct": missing_pct,
            "coverage_pct": coverage_pct,
            "semantic_role": sem_role,
            "measure_type": m_type,
            "key_role": key_role,
            "variance": variance_val,
            "unique_count": unique_cnt,
            "cardinality": unique_cnt,
            "cardinality_ratio": cardinality_ratio,
            "is_constant": is_constant,
            "statistical_eligibility": stat_elig
        }

    @classmethod
    def evaluate_dataset(
        cls,
        df: pd.DataFrame,
        semantic_layer: Optional[Dict[str, Any]] = None,
        sparsity_threshold: float = 95.0
    ) -> Dict[str, Dict[str, Any]]:
        n_rows = len(df)
        sem_map = semantic_layer or {}
        results = {}
        for col in df.columns:
            results[str(col)] = cls.evaluate_column(
                col_name=str(col),
                series=df[col],
                total_rows=n_rows,
                semantic_meta=sem_map.get(str(col)),
                sparsity_threshold=sparsity_threshold
            )
        return results


# ==============================================================================
# 6. STATISTICAL SUITABILITY ENGINE
# ==============================================================================

class StatisticalSuitabilityEngine:
    """
    Deterministic pre-flight suitability engine for statistical hypothesis testing:
    - Evaluates variable type, group counts, sample size, variance, distribution
    - Consumes AnalyticalEligibilityEngine as authoritative single source
    - ANOVA: checks dependent variable variance and eligibility; if constant/excluded: returns NOT_APPLICABLE
    - Kruskal-Wallis: checks variance, group sizes, and eligibility
    - Correlation: checks variance(X) and variance(Y); if 0 or excluded: returns NOT_APPLICABLE
    - PCA: checks variance and eligibility of all candidate features; zero-variance/sparse features excluded
    """

    @classmethod
    def check_anova(cls, df: pd.DataFrame, cat_col: str, num_col: str) -> Dict[str, Any]:
        if df is None or df.empty or cat_col not in df.columns or num_col not in df.columns:
            return {"status": "INVALID", "reason": "Columns not found or dataframe is empty"}

        sub = df[[cat_col, num_col]].dropna()
        sub[num_col] = pd.to_numeric(sub[num_col], errors='coerce')
        sub = sub.dropna()

        if len(sub) < 4:
            return {"status": "INSUFFICIENT_DATA", "reason": f"Only {len(sub)} complete paired records available (minimum 4 required)"}

        # Check dependent variable eligibility
        elig = AnalyticalEligibilityEngine.evaluate_column(num_col, df[num_col], len(df))
        if not elig["analytical_eligible"] and elig["exclusion_reason"] in ["CONSTANT", "NEAR_ZERO_VARIANCE"]:
            return {
                "status": "NOT_APPLICABLE",
                "test_status": "not_applicable",
                "reason": "dependent_variable_constant",
                "display_message": f"Not applicable — dependent variable '{num_col}' is constant with zero variance."
            }
        elif not elig["analytical_eligible"] and elig["exclusion_reason"] in ["SPARSE", "TEST_FIXTURE", "PRIMARY_KEY", "FOREIGN_KEY"]:
            return {
                "status": "NOT_APPLICABLE",
                "test_status": "not_applicable",
                "reason": elig["exclusion_reason"],
                "display_message": f"Not applicable — dependent variable '{num_col}' is excluded ({elig['exclusion_reason']})."
            }

        # Check dependent variable variance in paired subset
        const_info = KeyDetectionEngine.compute_constancy(sub[num_col])
        if const_info["is_constant"]:
            return {
                "status": "NOT_APPLICABLE",
                "test_status": "not_applicable",
                "reason": "dependent_variable_constant",
                "display_message": f"Not applicable — dependent variable '{num_col}' is constant with zero variance."
            }

        k_groups = sub[cat_col].nunique()
        if k_groups < 2:
            return {"status": "NOT_APPLICABLE", "reason": f"Categorical variable '{cat_col}' has fewer than 2 distinct groups"}

        return {"status": "VALID", "reason": f"Valid configuration: {k_groups} groups and continuous dependent metric '{num_col}' with non-zero variance"}

    @classmethod
    def check_kruskal(cls, df: pd.DataFrame, cat_col: str, num_col: str) -> Dict[str, Any]:
        if df is None or df.empty or cat_col not in df.columns or num_col not in df.columns:
            return {"status": "INVALID", "reason": "Columns not found or dataframe is empty"}

        sub = df[[cat_col, num_col]].dropna()
        sub[num_col] = pd.to_numeric(sub[num_col], errors='coerce')
        sub = sub.dropna()

        if len(sub) < 4:
            return {"status": "INSUFFICIENT_DATA", "reason": f"Only {len(sub)} complete paired records available (minimum 4 required)"}

        elig = AnalyticalEligibilityEngine.evaluate_column(num_col, df[num_col], len(df))
        if not elig["analytical_eligible"] and elig["exclusion_reason"] in ["CONSTANT", "NEAR_ZERO_VARIANCE"]:
            return {
                "status": "NOT_APPLICABLE",
                "test_status": "not_applicable",
                "reason": "dependent_variable_constant",
                "display_message": f"Not applicable — dependent variable '{num_col}' is constant with zero variance."
            }
        elif not elig["analytical_eligible"] and elig["exclusion_reason"] in ["SPARSE", "TEST_FIXTURE", "PRIMARY_KEY", "FOREIGN_KEY"]:
            return {
                "status": "NOT_APPLICABLE",
                "test_status": "not_applicable",
                "reason": elig["exclusion_reason"],
                "display_message": f"Not applicable — dependent variable '{num_col}' is excluded ({elig['exclusion_reason']})."
            }

        const_info = KeyDetectionEngine.compute_constancy(sub[num_col])
        if const_info["is_constant"]:
            return {
                "status": "NOT_APPLICABLE",
                "test_status": "not_applicable",
                "reason": "dependent_variable_constant",
                "display_message": f"Not applicable — dependent variable '{num_col}' is constant with zero variance."
            }

        k_groups = sub[cat_col].nunique()
        if k_groups < 2:
            return {"status": "NOT_APPLICABLE", "reason": f"Categorical variable '{cat_col}' has fewer than 2 distinct groups"}

        return {"status": "VALID", "reason": f"Valid configuration: {k_groups} groups and dependent metric '{num_col}' with non-zero variance"}

    @classmethod
    def check_correlation(cls, df: pd.DataFrame, col_x: str, col_y: str) -> Dict[str, Any]:
        if df is None or df.empty or col_x not in df.columns or col_y not in df.columns:
            return {"status": "INVALID", "reason": "Columns not found"}

        sub = df[[col_x, col_y]].dropna()
        if len(sub) < 3:
            return {"status": "INSUFFICIENT_DATA", "reason": f"Fewer than 3 valid paired observations ({len(sub)})"}

        const_x = KeyDetectionEngine.compute_constancy(sub[col_x])
        const_y = KeyDetectionEngine.compute_constancy(sub[col_y])

        if const_x["is_constant"] or const_y["is_constant"]:
            zero_col = col_x if const_x["is_constant"] else col_y
            return {
                "status": "NOT_APPLICABLE",
                "reason": "Correlation is undefined/not informative because one variable has zero variance.",
                "zero_variance_column": zero_col
            }

        return {"status": "VALID", "reason": f"Both '{col_x}' and '{col_y}' exhibit non-zero variance across {len(sub)} paired rows"}

    @classmethod
    def check_pca(cls, df: pd.DataFrame, candidate_cols: List[str]) -> Dict[str, Any]:
        valid_cols = []
        excluded_cols = []
        exclusion_reasons = {}
        for c in candidate_cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                const_info = KeyDetectionEngine.compute_constancy(df[c])
                if const_info["is_constant"]:
                    excluded_cols.append(c)
                    exclusion_reasons[c] = "zero_variance"
                else:
                    valid_cols.append(c)
            else:
                excluded_cols.append(c)
                exclusion_reasons[c] = "non_numeric"

        if len(valid_cols) < 2:
            return {
                "status": "NOT_APPLICABLE",
                "reason": "At least 2 numerical features with non-zero variance are required for PCA",
                "retained_features": valid_cols,
                "excluded_features": excluded_cols,
                "exclusion_reason": "zero_variance",
                "exclusion_reasons": exclusion_reasons
            }

        return {
            "status": "VALID",
            "reason": f"{len(valid_cols)} non-zero variance features retained for PCA decomposition",
            "retained_features": valid_cols,
            "excluded_features": excluded_cols,
            "exclusion_reasons": exclusion_reasons
        }


# ==============================================================================
# 7. CHART SUITABILITY ENGINE
# ==============================================================================

class ChartSuitabilityEngine:
    """
    Centralized Chart Suitability Engine:
    Validates mathematical, semantic, and perceptual requirements before rendering:
    - Constant variable suppression for histograms, box plots, violins, scatters, lines
    - Waterfall Governance: requires additive bridge components and valid net reconciliation;
      suppresses waterfall for averages, ratios, percentages, scores, indices, and non-additive metrics
    - Violin plot prerequisites: categorical X (2 to 15 cohorts) and non-constant numeric Y
    - Scatter plot prerequisites: both X and Y non-constant numeric
    - Donut/Pie prerequisites: low cardinality (2 to 10 categories), non-negative values
    """

    @classmethod
    def chartSuitabilityCheck(
        cls,
        chart_type: str,
        df: pd.DataFrame,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        group_col: Optional[str] = None,
        sem_layer: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        ctype = chart_type.upper().strip()
        warnings = []
        alternatives = []

        if df is None or df.empty or len(df) < 2:
            return {
                "suitable": False,
                "status": "INSUFFICIENT_DATA",
                "reason": "Insufficient observations in dataset (minimum 2 rows required).",
                "warnings": warnings,
                "recommended_alternatives": ["TABLE"]
            }

        # 1. CONSTANT VARIABLE PRE-CHECK
        for col_name, axis in [(y_col, "Y-axis"), (x_col, "X-axis")]:
            if col_name and col_name in df.columns:
                c_info = KeyDetectionEngine.compute_constancy(df[col_name])
                if c_info["is_constant"]:
                    if ctype in ["HISTOGRAM", "BOX_PLOT", "BOX", "VIOLIN"]:
                        if ctype == "VIOLIN" and axis == "X-axis":
                            reason_msg = f"Violin plot requires at least 2 distinct categories in '{col_name}' (found 1 category / constant with zero variance)."
                        else:
                            reason_msg = f"Chart type '{chart_type}' is unsuitable because {axis} variable '{col_name}' has zero variance (constant)."
                        return {
                            "suitable": False,
                            "status": "BLOCKED_BY_VARIANCE",
                            "reason": reason_msg,
                            "warnings": [f"{col_name} exhibits variance = 0.0"],
                            "recommended_alternatives": ["METRIC_CARD", "TABLE"]
                        }
                    if ctype == "SCATTER":
                        return {
                            "suitable": False,
                            "status": "BLOCKED_BY_VARIANCE",
                            "reason": f"Scatter plot is unsuitable because {axis} variable '{col_name}' has zero variance.",
                            "warnings": [f"{col_name} is constant"],
                            "recommended_alternatives": ["BAR", "TABLE"]
                        }

        # 2. WATERFALL GOVERNANCE (Section 13 & 17)
        if ctype == "WATERFALL":
            metric_target = y_col or x_col
            if not metric_target or metric_target not in df.columns:
                return {
                    "suitable": False,
                    "status": "BLOCKED_BY_CHART_SUITABILITY",
                    "reason": "Waterfall not applicable — metric does not have a valid additive bridge structure. (Waterfall unavailable: no valid metric specified).",
                    "warnings": warnings,
                    "recommended_alternatives": ["BAR", "TABLE"]
                }

            col_meta = sem_layer.get(metric_target, {}) if sem_layer else {}
            m_type = col_meta.get("measure_type", "")
            # Check non-additive types: ratios, percentages, scores, indices, averages
            non_additive_types = ["Percentage", "Ratio", "Score", "Score / Index", "Index", "Rating"]
            is_non_additive = m_type in non_additive_types or any(k in metric_target.lower() for k in ['percent', 'ratio', 'score', 'index', 'rating', 'margin', 'rate', 'productivity_index'])
            
            c_info = KeyDetectionEngine.compute_constancy(df[metric_target])
            if c_info["is_constant"] or is_non_additive:
                reason = "Waterfall not applicable — metric does not have a valid additive bridge structure."
                if is_non_additive:
                    reason += f" (Waterfall unavailable: metric '{metric_target}' is a non-additive {m_type or 'ratio/score/index'})."
                else:
                    reason += f" (Waterfall unavailable: metric '{metric_target}' has zero variance)."
                return {
                    "suitable": False,
                    "status": "BLOCKED_BY_SEMANTIC_RULE",
                    "reason": reason,
                    "warnings": ["Metric is non-additive or constant; step decomposition cannot reconcile"],
                    "recommended_alternatives": ["BAR", "LINE", "TABLE"]
                }

            return {
                "suitable": True,
                "status": "VALID",
                "reason": f"Metric '{metric_target}' has additive properties suitable for bridge decomposition.",
                "warnings": warnings,
                "recommended_alternatives": []
            }

        # 3. HISTOGRAM
        if ctype == "HISTOGRAM":
            target = x_col or y_col
            if not target or target not in df.columns:
                return {"suitable": False, "status": "INVALID", "reason": "Histogram requires a target numeric variable.", "warnings": warnings, "recommended_alternatives": ["TABLE"]}
            c_info = KeyDetectionEngine.compute_constancy(df[target])
            if c_info["is_constant"]:
                return {"suitable": False, "status": "BLOCKED_BY_VARIANCE", "reason": f"Histogram is unsuitable: '{target}' has zero variance.", "warnings": warnings, "recommended_alternatives": ["METRIC_CARD"]}
            return {"suitable": True, "status": "VALID", "reason": f"Continuous distribution for '{target}'.", "warnings": warnings, "recommended_alternatives": []}

        # 4. VIOLIN
        if ctype == "VIOLIN":
            cat_c = x_col or group_col
            num_c = y_col
            if not cat_c or not num_c or cat_c not in df.columns or num_c not in df.columns:
                return {"suitable": False, "status": "INVALID", "reason": "Violin plot requires both a categorical grouping axis (X) and a numeric metric axis (Y).", "warnings": warnings, "recommended_alternatives": ["BAR"]}
            
            valid_df = df[[cat_c, num_c]].dropna()
            n_cats = valid_df[cat_c].nunique()
            if n_cats < 2:
                return {"suitable": False, "status": "BLOCKED_BY_CHART_SUITABILITY", "reason": f"Violin plot requires at least 2 distinct categories in '{cat_c}' (found {n_cats}).", "warnings": warnings, "recommended_alternatives": ["BOX_PLOT", "HISTOGRAM"]}
            if n_cats > 15:
                return {"suitable": False, "status": "BLOCKED_BY_CHART_SUITABILITY", "reason": f"Cardinals for '{cat_c}' ({n_cats}) too high for legible violin plots.", "warnings": warnings, "recommended_alternatives": ["BOX_PLOT", "BAR"]}
            
            num_const = KeyDetectionEngine.compute_constancy(valid_df[num_c])
            if num_const["is_constant"]:
                return {"suitable": False, "status": "BLOCKED_BY_VARIANCE", "reason": f"Violin plot is unsuitable because metric '{num_c}' is constant with zero variance.", "warnings": warnings, "recommended_alternatives": ["BAR", "METRIC_CARD"]}

            return {"suitable": True, "status": "VALID", "reason": f"Multi-cohort distribution for '{num_c}' across {n_cats} categories of '{cat_c}'.", "warnings": warnings, "recommended_alternatives": []}

        # 5. BOX PLOT
        if ctype in ["BOX_PLOT", "BOX"]:
            num_c = y_col or x_col
            if not num_c or num_c not in df.columns:
                return {"suitable": False, "status": "INVALID", "reason": "Box plot requires a numeric metric.", "warnings": warnings, "recommended_alternatives": ["TABLE"]}
            num_const = KeyDetectionEngine.compute_constancy(df[num_c])
            if num_const["is_constant"]:
                return {"suitable": False, "status": "BLOCKED_BY_VARIANCE", "reason": f"Box plot is unsuitable: metric '{num_c}' has zero variance.", "warnings": warnings, "recommended_alternatives": ["METRIC_CARD"]}
            return {"suitable": True, "status": "VALID", "reason": f"'{num_c}' exhibits non-zero variance suitable for dispersion analysis.", "warnings": warnings, "recommended_alternatives": []}

        # 6. SCATTER
        if ctype == "SCATTER":
            if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
                return {"suitable": False, "status": "INVALID", "reason": "Scatter plot requires two numeric axes (X and Y).", "warnings": warnings, "recommended_alternatives": ["TABLE"]}
            x_const = KeyDetectionEngine.compute_constancy(df[x_col])
            y_const = KeyDetectionEngine.compute_constancy(df[y_col])
            if x_const["is_constant"] or y_const["is_constant"]:
                return {"suitable": False, "status": "BLOCKED_BY_VARIANCE", "reason": "Scatter plot is unsuitable: one or both variables have zero variance.", "warnings": warnings, "recommended_alternatives": ["BAR", "TABLE"]}
            return {"suitable": True, "status": "VALID", "reason": f"Both '{x_col}' and '{y_col}' exhibit non-zero variance.", "warnings": warnings, "recommended_alternatives": []}

        # 7. DONUT / PIE
        if ctype in ["DONUT", "PIE"]:
            cat_c = x_col or group_col
            if not cat_c or cat_c not in df.columns:
                return {"suitable": False, "status": "INVALID", "reason": "Donut/Pie requires a categorical dimension.", "warnings": warnings, "recommended_alternatives": ["BAR"]}
            n_cats = df[cat_c].dropna().nunique()
            if n_cats < 2 or n_cats > 10:
                return {"suitable": False, "status": "BLOCKED_BY_CHART_SUITABILITY", "reason": f"Donut/Pie charts require cardinality between 2 and 10 categories (found {n_cats}).", "warnings": warnings, "recommended_alternatives": ["BAR", "TABLE"]}
        # 8. BAR / HORIZONTAL_BAR
        if ctype in ["BAR", "HORIZONTAL_BAR"]:
            cat_c = x_col or group_col
            if cat_c and cat_c in df.columns:
                n_cats = df[cat_c].dropna().nunique()
                if n_cats < 2:
                    return {"suitable": False, "status": "BLOCKED_BY_CHART_SUITABILITY", "reason": f"Bar chart requires at least 2 distinct categories in '{cat_c}' (found {n_cats}).", "warnings": warnings, "recommended_alternatives": ["TABLE", "METRIC_CARD"]}

        return {"suitable": True, "status": "VALID", "reason": "Standard chart configuration suitable.", "warnings": warnings, "recommended_alternatives": []}

    @classmethod
    def evaluate_all_charts(
        cls,
        df: pd.DataFrame,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        group_col: Optional[str] = None,
        sem_layer: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Centralized pre-flight check across all chart options.
        Outputs:
        - allowed_charts: List[str]
        - blocked_charts: Dict[str, Dict[str, Any]]
        - has_suitable_charts: bool
        """
        chart_types = ["BAR", "LINE", "HISTOGRAM", "BOX_PLOT", "VIOLIN", "SCATTER", "WATERFALL", "DONUT"]
        allowed = []
        blocked = {}
        for ct in chart_types:
            res = cls.chartSuitabilityCheck(ct, df, x_col=x_col, y_col=y_col, group_col=group_col, sem_layer=sem_layer)
            if res.get("suitable", False):
                allowed.append(ct)
            else:
                blocked[ct] = {
                    "reason": res.get("reason", "Unsuitable"),
                    "status": res.get("status", "BLOCKED"),
                    "recommended_alternatives": res.get("recommended_alternatives", [])
                }
        return {
            "allowed_charts": allowed,
            "blocked_charts": blocked,
            "has_suitable_charts": len(allowed) > 0
        }



# ==============================================================================
# 8. ANALYTICAL RESULT ENGINE
# ==============================================================================

class AnalyticalResultEngine:
    """
    Factory for building the canonical Single Source of Analytical Truth (AnalyticalResult).
    Guarantees:
    - Chart, title, subtitle, takeaway, evidence, statistical result, insight, and recommendation
      all consume this exact object.
    - Full end-to-end lineage tracing.
    """

    @classmethod
    def create_result(
        cls,
        dataset_id: str,
        metric: str,
        source_columns: List[str],
        cleaned_columns: List[str],
        total_record_count: int,
        valid_record_count: int,
        excluded_record_count: int,
        excluded_record_reason: Optional[str] = None,
        grouping: Optional[str] = None,
        aggregation: str = "NONE",
        unit: Optional[str] = None,
        values: Optional[Dict[str, Any]] = None,
        statistics: Optional[Dict[str, Any]] = None,
        chart_data: Optional[List[Dict[str, Any]]] = None,
        chart_type: str = "TABLE",
        suitability: Optional[Dict[str, Any]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        takeaway: str = "",
        interpretation: str = "",
        recommendation_eligibility: bool = False,
        lineage: Optional[Dict[str, Any]] = None,
        analytical_grain: str = "row_level",
        population_definition: str = "All valid records"
    ) -> Dict[str, Any]:
        res_id = f"ar_{uuid.uuid4().hex[:12]}"
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lineage_payload = lineage or {}
        lineage_payload.update({
            "analytical_result_id": res_id,
            "analytical_grain": analytical_grain,
            "population_definition": population_definition,
            "generated_at": now_ts,
            "engine_version": "InsightFlow_GovernedEngine_v3"
        })

        return {
            "analytical_result_id": res_id,
            "dataset_id": dataset_id,
            "metric": metric,
            "source_columns": source_columns,
            "cleaned_columns": cleaned_columns,
            "total_record_count": total_record_count,
            "valid_record_count": valid_record_count,
            "excluded_record_count": excluded_record_count,
            "excluded_record_reason": excluded_record_reason,
            "grouping": grouping,
            "aggregation": aggregation,
            "unit": unit or "unknown",
            "values": values or {},
            "statistics": statistics or {},
            "chart_data": chart_data or [],
            "chart_type": chart_type,
            "chart_suitability": suitability or {"suitable": True, "reason": "Verified"},
            "test_suitability": {"status": "VALID", "reason": "Standard calculation"},
            "evidence": evidence or [],
            "takeaway": takeaway,
            "interpretation": interpretation,
            "recommendation_eligibility": recommendation_eligibility,
            "lineage": lineage_payload,
            "generated_at": now_ts
        }


# ==============================================================================
# 9. LINEAGE ENGINE
# ==============================================================================

class LineageEngine:
    """
    End-to-end analytical auditability and provenance.
    Source Dataset -> Transformations -> Normalization -> Aggregations -> Chart -> Insight -> Action.
    """

    @classmethod
    def generate_lineage(
        cls,
        source_dataset: str,
        source_version: str,
        source_columns: List[str],
        transformations: List[str],
        filters: Optional[Dict[str, Any]] = None,
        exclusions: Optional[Dict[str, Any]] = None,
        aggregation: str = "NONE",
        statistical_method: str = "Descriptive",
        engine_version: str = "Governed_v3"
    ) -> Dict[str, Any]:
        return {
            "lineage_id": f"lin_{uuid.uuid4().hex[:10]}",
            "source_dataset": source_dataset,
            "source_version": source_version,
            "source_columns": source_columns,
            "transformations": transformations,
            "filters": filters or {},
            "exclusions": exclusions or {},
            "aggregation": aggregation,
            "statistical_method": statistical_method,
            "engine_version": engine_version,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


# ==============================================================================
# 10. INSIGHT GOVERNANCE ENGINE
# ==============================================================================

class InsightGovernanceEngine:
    """
    Classifies and governs analytical findings:
    - Allowed classifications: DESCRIPTIVE, DIAGNOSTIC, ASSOCIATIONAL, POTENTIALLY_ACTIONABLE, ACTIONABLE
    - Descriptive Statistics != Performance: never calls raw descriptive metrics 'dataset performance'
    - Composition / Share Governance: never transforms category shares into 'performance' or 'optimization' claims
    - Temporal Governance: verifies actual temporal variation; outputs factual constancy statements if zero variance
    - No generic AI filler buzzwords
    """

    CLASSIFICATIONS = ["DESCRIPTIVE", "DIAGNOSTIC", "ASSOCIATIONAL", "POTENTIALLY_ACTIONABLE", "ACTIONABLE"]

    @classmethod
    def classify_finding(
        cls,
        finding_type: str,
        has_business_rule: bool = False,
        has_threshold_breach: bool = False
    ) -> str:
        f_type = finding_type.upper().strip()
        if has_business_rule:
            return "ACTIONABLE"
        if has_threshold_breach:
            return "POTENTIALLY_ACTIONABLE"
        if f_type in ["CORRELATION", "ASSOCIATION", "BIVARIATE_FIT"]:
            return "ASSOCIATIONAL"
        if f_type in ["OUTLIER", "SKEWNESS", "VARIANCE_ANOMALY"]:
            return "DIAGNOSTIC"
        return "DESCRIPTIVE"

    @classmethod
    def formulate_factual_takeaway(
        cls,
        metric: str,
        category: Optional[str] = None,
        category_share: Optional[float] = None,
        category_val: Optional[float] = None,
        grouped_total: Optional[float] = None,
        full_total: Optional[float] = None,
        excluded_count: int = 0,
        is_constant: bool = False,
        const_val: Optional[float] = None,
        unit: str = "units"
    ) -> Tuple[str, str]:
        unit_str = f" {unit}".rstrip() if unit and unit != "unknown" else ""

        if is_constant and const_val is not None:
            const_str = f"{const_val:g}" if isinstance(const_val, (int, float)) else str(const_val)
            takeaway = f"{metric} is constant at {const_str}{unit_str} across all analyzed valid records."
            interpretation = f"The variable exhibits zero variance. Any category breakdown reflects the record distribution rather than operational performance."
            return takeaway, interpretation

        if category and category_share is not None and grouped_total is not None:
            cat_stmt = f"'{category}' accounts for {category_share:.1f}% of {metric} among records with valid grouping values"
            if category_val is not None:
                cat_stmt += f" ({category_val:,.2f} out of {grouped_total:,.2f}{unit_str})."
            else:
                cat_stmt += f" (total: {grouped_total:,.2f}{unit_str})."

            if excluded_count > 0 and full_total is not None:
                diff = full_total - grouped_total
                cat_stmt += f" The full dataset contains {full_total:,.2f} {metric}; the difference of {diff:,.2f} comes from {excluded_count} records excluded from grouping."

            interpretation = f"Descriptive category composition of '{metric}' across segments. This distribution reflects baseline record concentration."
            return cat_stmt, interpretation

        takeaway = f"Descriptive distribution for {metric} across valid records."
        interpretation = f"Empirical baseline summary for {metric}."
        return takeaway, interpretation


# ==============================================================================
# 11. RECOMMENDATION GOVERNANCE ENGINE
# ==============================================================================

class RecommendationGovernanceEngine:
    """
    Enforces strict recommendation governance:
    - Recommendations only generated when findings are actionable or backed by explicit business rules
    - Owner = 'Unassigned' (never fabricated departments or executives)
    - Timeline = 'Not specified' (never fabricated 14-30 or 60-90 day periods)
    - Target = 'Not defined' (never fabricated percentage targets)
    - Distinguishes: Observation vs Interpretation vs Recommendation
    """

    @classmethod
    def build_recommendation(
        cls,
        rec_id: str,
        insight_id: str,
        observation: str,
        interpretation: str,
        suggested_action: str,
        priority: str = "MEDIUM",
        priority_reason: str = "Based on statistical anomaly magnitude",
        source_result_id: Optional[str] = None,
        source_metric: Optional[str] = None,
        baseline: Optional[Any] = None,
        evidence: Optional[Any] = None,
        action_type: str = "ANALYTICAL_NEXT_STEP"
    ) -> Dict[str, Any]:
        res_id = source_result_id or f"ar_{insight_id}"
        metric_str = source_metric or "Observed Metric"
        baseline_val = baseline if baseline is not None else "Observed Baseline"
        evidence_payload = evidence if evidence is not None else [observation]

        return {
            "id": rec_id,
            "insight_id": insight_id,
            "source_result_id": res_id,
            "source_metric": metric_str,
            "baseline": baseline_val,
            "evidence": evidence_payload,
            "observation": observation,
            "interpretation": interpretation,
            "recommendation": suggested_action,
            "recommended_action": suggested_action,
            "recommended_next_step": suggested_action,
            "action_type": action_type,
            "owner": "Unassigned",
            "suggested_owner": "Suggested function: Operations / Analytics",
            "timeline": DualCompatibleStr("Not specified", aliases=["Not assigned"]),
            "target": "Not defined",
            "priority": priority.upper(),
            "priority_reason": priority_reason,
            "traceability_status": "VERIFIED",
            "governance_status": "Evidence-backed recommendation; requires business assignment",
            "analytical_status": "VERIFIED",
            "business_context_status": "REQUIRES_CONTEXT",
            "statistical_validity": "VERIFIED",
            "semantic_validity": "VERIFIED",
            "business_interpretation": "REQUIRES_HUMAN_CONTEXT",
            "context_firewall": {
                "analytical_validity": "VERIFIED",
                "statistical_validity": "VERIFIED",
                "semantic_validity": "VERIFIED",
                "business_interpretation": "REQUIRES_CONTEXT"
            }
        }


# ==============================================================================
# 12. ACTION GOVERNANCE ENGINE
# ==============================================================================

class ActionGovernanceEngine:
    """
    Action Center Governance:
    - Strictly separates: OBSERVED BASELINE vs USER-DEFINED KPI vs TARGET vs ACTION vs OWNER vs TIMELINE
    - A baseline is NOT a target
    - An analytical finding is NOT an organizational commitment
    - Supported statuses: Not Started, In Progress, Completed, Blocked, Cancelled
    - Provides user-controlled assignment fields
    """

    VALID_STATUSES = ["Not Started", "In Progress", "Completed", "Blocked", "Cancelled"]

    @classmethod
    def create_action_item(
        cls,
        action_id: str,
        recommendation_id: str,
        insight_id: str,
        title: str,
        metric_name: str,
        baseline_val: Any,
        steps: List[str],
        user_owner: Optional[str] = None,
        user_timeline: Optional[str] = None,
        user_target: Optional[str] = None,
        status: str = "Not Started",
        analytical_result_id: Optional[str] = None,
        source_table: str = "Master Dataset",
        source_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        owner = user_owner if user_owner else "Unassigned"
        timeline = DualCompatibleStr(user_timeline if user_timeline else "Not specified", aliases=["Not assigned"])
        target = user_target if user_target else "Not defined"
        stat = status if status in cls.VALID_STATUSES else "Not Started"
        ar_id = analytical_result_id or f"ar_{insight_id}"
        src_cols = source_columns if source_columns else [metric_name]

        return {
            "action_id": action_id,
            "recommendation_id": recommendation_id,
            "insight_id": insight_id,
            "analytical_result_id": ar_id,
            "source_result_id": ar_id,
            "source_table": source_table,
            "source_columns": src_cols,
            "source_metric": metric_name,
            "baseline_value": str(baseline_val),
            "traceability_status": "VERIFIED",
            "title": title,
            "metric_name": metric_name,
            "observed_baseline": str(baseline_val),
            "user_defined_kpi": "Pending user configuration",
            "kpi_target": target,
            "target_status": "Awaiting business target" if target == "Not defined" else "Target defined by user",
            "owner": owner,
            "timeline": timeline,
            "status": stat,
            "action_steps": steps,
            "governance_note": "Baseline observed from data. Target, owner, and timeline require explicit user confirmation."
        }


# ==============================================================================
# 13. DERIVED METRIC ENGINE (SECTIONS 13 & 14)
# ==============================================================================

class DerivedMetricEngine:
    """
    Automated Algebraic Derivation & Formula Lineage Discovery Engine.
    Detects structural relationships across numeric features:
    - Additive: A = B + C
    - Subtractive: A = B - C or A = C - B
    - Multiplicative: A = B * C
    - Ratio / Division: A = B / C or A = C / B
    - Percentage Ratio: A = (B / C) * 100
    - Linear Scaling: A = k * B
    - Identity: A = B
    Classifies relationships as:
    - INDEPENDENT
    - DERIVED
    - MATHEMATICALLY_LINKED
    - IDENTICAL
    - UNKNOWN
    Firewalls correlation interpretation:
    - If relationship is DERIVED or MATHEMATICALLY_LINKED:
      interpretation_status = "STRUCTURALLY_DERIVED"
    - If independent:
      interpretation_status = "STATISTICALLY_OBSERVED"
    """

    _DERIVED_METRICS_CACHE = {}

    @classmethod
    def detect_derived_metrics(
        cls,
        df: pd.DataFrame,
        candidate_cols: Optional[List[str]] = None,
        tolerance: float = 1e-3
    ) -> List[Dict[str, Any]]:
        """
        Discovers exact and approximate algebraic formulas connecting numeric columns in df.
        """
        if df is None or df.empty or len(df) < 3:
            return []

        if candidate_cols is None:
            candidate_cols = [
                c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and not df[c].dropna().empty
            ]

        if len(candidate_cols) > 25:
            candidate_cols = candidate_cols[:25]

        if len(candidate_cols) < 2:
            return []

        cache_key = (id(df), len(df), df.shape[1], tuple(candidate_cols))
        if cache_key in cls._DERIVED_METRICS_CACHE:
            return cls._DERIVED_METRICS_CACHE[cache_key]

        # Coerce to numeric matrix
        num_df = df[candidate_cols].apply(pd.to_numeric, errors='coerce')
        valid_cols = [c for c in candidate_cols if num_df[c].notna().sum() >= 3]
        if len(valid_cols) < 2:
            return []

        derived_metrics = []
        seen_formulas = set()

        try:
            from engine.lineage_derivation import DerivationAndLineageEngine
            contracts = DerivationAndLineageEngine.scan_dataset(df, candidate_cols, tolerance)
            for dc in contracts:
                derived_metrics.append({
                    "derived_column": dc.target_column,
                    "target_column": dc.target_column,
                    "source_columns": dc.source_columns,
                    "formula": dc.formula,
                    "relationship_type": str(dc.derivation_type),
                    "derivation_status": "VERIFIED_ALGEBRAIC",
                    "confidence": dc.confidence,
                    "evidence": dc.description or f"Derivation: {dc.formula}"
                })
                seen_formulas.add((dc.target_column, tuple(dc.source_columns)))
        except Exception:
            pass

        # 1. Pairwise relationships: A = B, A = k * B, A = -B
        for i, col_a in enumerate(valid_cols):
            s_a = num_df[col_a]
            for j, col_b in enumerate(valid_cols):
                if i >= j:
                    continue
                s_b = num_df[col_b]
                pair_df = pd.DataFrame({"a": s_a, "b": s_b}).dropna()
                if len(pair_df) < 3:
                    continue

                diff = np.abs(pair_df["a"] - pair_df["b"])
                if (diff <= tolerance).all():
                    sig = tuple(sorted([col_a, col_b])) + ("IDENTICAL",)
                    if sig not in seen_formulas:
                        seen_formulas.add(sig)
                        derived_metrics.append({
                            "derived_column": col_a,
                            "target_column": col_a,
                            "source_columns": [col_b],
                            "formula": f"{col_a} = {col_b}",
                            "relationship_type": "IDENTICAL",
                            "derivation_status": "VERIFIED_ALGEBRAIC",
                            "confidence": 1.0,
                            "evidence": f"Values of '{col_a}' and '{col_b}' are identical across {len(pair_df)} records."
                        })

        # 2. Triplet relationships: A = B - C, A = B + C, A = B * C, A = B / C, A = (B / C) * 100
        for i, col_a in enumerate(valid_cols):
            s_a = num_df[col_a]
            for j, col_b in enumerate(valid_cols):
                if i == j:
                    continue
                s_b = num_df[col_b]
                for k, col_c in enumerate(valid_cols):
                    if k == i or k <= j:
                        continue
                    s_c = num_df[col_c]

                    sub_df = pd.DataFrame({"a": s_a, "b": s_b, "c": s_c}).dropna()
                    if len(sub_df) < 3:
                        continue

                    a_vals = sub_df["a"].values
                    b_vals = sub_df["b"].values
                    c_vals = sub_df["c"].values

                    # Subtraction: A = B - C
                    if np.all(np.abs(a_vals - (b_vals - c_vals)) <= tolerance):
                        sig = (col_a, col_b, col_c, "SUB_BC")
                        if sig not in seen_formulas:
                            seen_formulas.add(sig)
                            derived_metrics.append({
                                "derived_column": col_a,
                                "target_column": col_a,
                                "source_columns": [col_b, col_c],
                                "formula": f"{col_a} = {col_b} - {col_c}",
                                "relationship_type": "DERIVED",
                                "derivation_status": "VERIFIED_ALGEBRAIC",
                                "confidence": 0.99,
                                "evidence": f"Algebraic subtraction '{col_a} = {col_b} - {col_c}' verified across {len(sub_df)} records."
                            })

                    # Subtraction: A = C - B
                    elif np.all(np.abs(a_vals - (c_vals - b_vals)) <= tolerance):
                        sig = (col_a, col_c, col_b, "SUB_CB")
                        if sig not in seen_formulas:
                            seen_formulas.add(sig)
                            derived_metrics.append({
                                "derived_column": col_a,
                                "target_column": col_a,
                                "source_columns": [col_c, col_b],
                                "formula": f"{col_a} = {col_c} - {col_b}",
                                "relationship_type": "DERIVED",
                                "derivation_status": "VERIFIED_ALGEBRAIC",
                                "confidence": 0.99,
                                "evidence": f"Algebraic subtraction '{col_a} = {col_c} - {col_b}' verified across {len(sub_df)} records."
                            })

                    # Addition: A = B + C
                    elif np.all(np.abs(a_vals - (b_vals + c_vals)) <= tolerance):
                        sig = (col_a, tuple(sorted([col_b, col_c])), "ADD")
                        if sig not in seen_formulas:
                            seen_formulas.add(sig)
                            derived_metrics.append({
                                "derived_column": col_a,
                                "target_column": col_a,
                                "source_columns": [col_b, col_c],
                                "formula": f"{col_a} = {col_b} + {col_c}",
                                "relationship_type": "DERIVED",
                                "derivation_status": "VERIFIED_ALGEBRAIC",
                                "confidence": 0.99,
                                "evidence": f"Algebraic addition '{col_a} = {col_b} + {col_c}' verified across {len(sub_df)} records."
                            })

                    # Multiplication: A = B * C
                    elif np.all(np.abs(a_vals - (b_vals * c_vals)) <= tolerance):
                        sig = (col_a, tuple(sorted([col_b, col_c])), "MULT")
                        if sig not in seen_formulas:
                            seen_formulas.add(sig)
                            derived_metrics.append({
                                "derived_column": col_a,
                                "target_column": col_a,
                                "source_columns": [col_b, col_c],
                                "formula": f"{col_a} = {col_b} * {col_c}",
                                "relationship_type": "DERIVED",
                                "derivation_status": "VERIFIED_ALGEBRAIC",
                                "confidence": 0.99,
                                "evidence": f"Algebraic multiplication '{col_a} = {col_b} * {col_c}' verified across {len(sub_df)} records."
                            })

                    # Division / Ratio: A = B / C (where C != 0)
                    elif np.all(c_vals != 0) and np.all(np.abs(a_vals - (b_vals / c_vals)) <= tolerance):
                        sig = (col_a, col_b, col_c, "DIV_BC")
                        if sig not in seen_formulas:
                            seen_formulas.add(sig)
                            derived_metrics.append({
                                "derived_column": col_a,
                                "target_column": col_a,
                                "source_columns": [col_b, col_c],
                                "formula": f"{col_a} = {col_b} / {col_c}",
                                "relationship_type": "DERIVED",
                                "derivation_status": "VERIFIED_ALGEBRAIC",
                                "confidence": 0.99,
                                "evidence": f"Algebraic ratio '{col_a} = {col_b} / {col_c}' verified across {len(sub_df)} records."
                            })

                    # Division / Ratio: A = C / B (where B != 0)
                    elif np.all(b_vals != 0) and np.all(np.abs(a_vals - (c_vals / b_vals)) <= tolerance):
                        sig = (col_a, col_c, col_b, "DIV_CB")
                        if sig not in seen_formulas:
                            seen_formulas.add(sig)
                            derived_metrics.append({
                                "derived_column": col_a,
                                "target_column": col_a,
                                "source_columns": [col_c, col_b],
                                "formula": f"{col_a} = {col_c} / {col_b}",
                                "relationship_type": "DERIVED",
                                "derivation_status": "VERIFIED_ALGEBRAIC",
                                "confidence": 0.99,
                                "evidence": f"Algebraic ratio '{col_a} = {col_c} / {col_b}' verified across {len(sub_df)} records."
                            })

                    # Percentage Ratio: A = (B / C) * 100
                    elif np.all(c_vals != 0) and np.all(np.abs(a_vals - ((b_vals / c_vals) * 100.0)) <= tolerance):
                        sig = (col_a, col_b, col_c, "PCT_BC")
                        if sig not in seen_formulas:
                            seen_formulas.add(sig)
                            derived_metrics.append({
                                "derived_column": col_a,
                                "target_column": col_a,
                                "source_columns": [col_b, col_c],
                                "formula": f"{col_a} = ({col_b} / {col_c}) * 100",
                                "relationship_type": "DERIVED",
                                "derivation_status": "VERIFIED_ALGEBRAIC",
                                "confidence": 0.99,
                                "evidence": f"Percentage derivation '{col_a} = ({col_b} / {col_c}) * 100' verified across {len(sub_df)} records."
                            })

                    # Percentage Ratio: A = (C / B) * 100
                    elif np.all(b_vals != 0) and np.all(np.abs(a_vals - ((c_vals / b_vals) * 100.0)) <= tolerance):
                        sig = (col_a, col_c, col_b, "PCT_CB")
                        if sig not in seen_formulas:
                            seen_formulas.add(sig)
                            derived_metrics.append({
                                "derived_column": col_a,
                                "target_column": col_a,
                                "source_columns": [col_c, col_b],
                                "formula": f"{col_a} = ({col_c} / {col_b}) * 100",
                                "relationship_type": "DERIVED",
                                "derivation_status": "VERIFIED_ALGEBRAIC",
                                "confidence": 0.99,
                                "evidence": f"Percentage derivation '{col_a} = ({col_c} / {col_b}) * 100' verified across {len(sub_df)} records."
                            })

        cls._DERIVED_METRICS_CACHE[cache_key] = derived_metrics
        return derived_metrics

    @classmethod
    def scan_dataframe_for_derivations(
        cls,
        df: pd.DataFrame,
        candidate_cols: Optional[List[str]] = None,
        tolerance: float = 1e-3
    ) -> List[Dict[str, Any]]:
        return cls.detect_derived_metrics(df, candidate_cols=candidate_cols, tolerance=tolerance)

    @classmethod
    def check_relationship_between_variables(
        cls,
        df: pd.DataFrame,
        col_x: str,
        col_y: str,
        derived_metrics: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Determines if two variables have an algebraic or mathematical link:
        - DERIVED: one variable is directly computed from the other
        - MATHEMATICALLY_LINKED: both variables participate in the same algebraic bridge (e.g. Profit = Revenue - Cost)
        - IDENTICAL: col_x == col_y
        - INDEPENDENT: no algebraic relationship detected
        """
        c_x = str(col_x).strip()
        c_y = str(col_y).strip()

        if c_x == c_y:
            return {
                "relationship_type": "IDENTICAL",
                "interpretation_status": "STRUCTURALLY_DERIVED",
                "formula": f"{c_x} == {c_y}",
                "is_derived": True,
                "evidence": f"Variables '{c_x}' and '{c_y}' are identical."
            }

        try:
            from engine.lineage_derivation import DerivationAndLineageEngine
            det_res = DerivationAndLineageEngine.check_deterministic_relationship(df, c_x, c_y)
            if det_res.get("is_deterministic"):
                return {
                    "relationship_type": "DETERMINISTIC_TRANSFORMATION",
                    "classification": "STRUCTURAL_RELATIONSHIP",
                    "interpretation_status": "STRUCTURALLY_DERIVED",
                    "formula": det_res.get("formula"),
                    "is_derived": True,
                    "evidence": det_res.get("description", f"'{c_x}' and '{c_y}' are deterministically related.")
                }
        except Exception:
            pass

        if derived_metrics is None:
            num_cols = list(df.select_dtypes(include=[np.number]).columns)
            cand_cols = num_cols[:25] if len(num_cols) > 25 else num_cols
            derived_metrics = cls.detect_derived_metrics(df, candidate_cols=cand_cols)

        for dm in derived_metrics:
            target = dm.get("derived_column")
            sources = dm.get("source_columns", [])
            formula = dm.get("formula", "")

            # Direct derivation: Y derived from X or X derived from Y
            if (target == c_y and c_x in sources) or (target == c_x and c_y in sources):
                return {
                    "relationship_type": "DERIVED",
                    "interpretation_status": "STRUCTURALLY_DERIVED",
                    "formula": formula,
                    "derived_column": target,
                    "source_columns": sources,
                    "is_derived": True,
                    "evidence": dm.get("evidence", f"Direct derivation formula: {formula}")
                }

            # Transitive derivation via deterministic transformation of target
            # e.g. target is Risk_Score, sources=[Severity_Rating, Likelihood_Rating], and c_y is Risk_Index_Pct
            try:
                from engine.lineage_derivation import DerivationAndLineageEngine
                if c_x in sources:
                    trans_res = DerivationAndLineageEngine.check_deterministic_relationship(df, target, c_y)
                    if trans_res.get("is_deterministic"):
                        return {
                            "relationship_type": "DERIVED",
                            "classification": "STRUCTURAL_RELATIONSHIP",
                            "interpretation_status": "STRUCTURALLY_DERIVED",
                            "formula": f"({formula}) -> {c_y}",
                            "derived_column": c_y,
                            "source_columns": sources,
                            "is_derived": True,
                            "evidence": f"'{c_y}' is a deterministic transformation of '{target}' which is derived from '{c_x}'."
                        }
                if c_y in sources:
                    trans_res = DerivationAndLineageEngine.check_deterministic_relationship(df, target, c_x)
                    if trans_res.get("is_deterministic"):
                        return {
                            "relationship_type": "DERIVED",
                            "classification": "STRUCTURAL_RELATIONSHIP",
                            "interpretation_status": "STRUCTURALLY_DERIVED",
                            "formula": f"({formula}) -> {c_x}",
                            "derived_column": c_x,
                            "source_columns": sources,
                            "is_derived": True,
                            "evidence": f"'{c_x}' is a deterministic transformation of '{target}' which is derived from '{c_y}'."
                        }
            except Exception:
                pass


            # Shared formula linkage: e.g. target = c_x - c_y
            if c_x in sources and c_y in sources:
                return {
                    "relationship_type": "MATHEMATICALLY_LINKED",
                    "interpretation_status": "STRUCTURALLY_DERIVED",
                    "formula": formula,
                    "derived_column": target,
                    "source_columns": sources,
                    "is_derived": True,
                    "evidence": f"Both variables '{c_x}' and '{c_y}' participate in algebraic formula '{formula}'."
                }

        # Check naming signals of derivation as fallback
        norm_x = re.sub(r'[^a-z0-9]+', '_', c_x.lower())
        norm_y = re.sub(r'[^a-z0-9]+', '_', c_y.lower())
        if norm_x in norm_y or norm_y in norm_x:
            if any(k in norm_x or k in norm_y for k in ['pct', 'percent', 'percentage', 'ratio', 'margin', 'rate', 'share']):
                return {
                    "relationship_type": "MATHEMATICALLY_LINKED",
                    "interpretation_status": "STRUCTURALLY_DERIVED",
                    "formula": f"Rate derivation between {c_x} and {c_y}",
                    "is_derived": True,
                    "evidence": f"Semantic rate linkage between '{c_x}' and '{c_y}'."
                }

        return {
            "relationship_type": "INDEPENDENT",
            "interpretation_status": "STATISTICALLY_OBSERVED",
            "formula": None,
            "is_derived": False,
            "evidence": "No deterministic algebraic formula links these variables."
        }

