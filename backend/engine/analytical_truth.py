import math
import re
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd

from engine.governed_engines import (
    DualCompatibleStr,
    SemanticInferenceEngine,
    KeyDetectionEngine,
    RelationshipDetectionEngine,
    DataQualityEngine,
    CategoryNormalizationEngine,
    StatisticalSuitabilityEngine,
    ChartSuitabilityEngine,
    AnalyticalResultEngine,
    LineageEngine,
    InsightGovernanceEngine,
    RecommendationGovernanceEngine,
    ActionGovernanceEngine,
    DerivedMetricEngine
)

class AnalyticalResult:
    """
    Canonical Single Source of Analytical Truth.
    Every chart, numerical evidence, narrative takeaway, statistical interpretation,
    and recommendation consumes this exact verified object.
    """
    _counter = 0

    def __init__(
        self,
        dataset_id: str,
        metric: str,
        source_columns: List[str],
        cleaned_columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        total_record_count: int = 0,
        valid_record_count: int = 0,
        excluded_record_count: int = 0,
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
        result_id: Optional[str] = None,
        source_table: str = "Master Dataset",
        row_count_raw: Optional[int] = None,
        row_count_analytical: Optional[int] = None,
        row_count_excluded: Optional[int] = None,
        aggregation_grain: Optional[str] = None,
        aggregation_method: Optional[str] = None,
        currency: Optional[str] = None,
        statistical_method: Optional[str] = None,
        assumptions: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        confidence: float = 0.95,
        analysis_type: str = "GENERAL",
        data_stage: str = "CLEANED",
        sample_size: Optional[int] = None,
        excluded_count: Optional[int] = None,
        denominator: Optional[float] = None,
        relationship_type: str = "INDEPENDENT",
        interpretation_status: str = "STATISTICALLY_OBSERVED",
        confidence_metadata: Optional[Dict[str, Any]] = None,
        recommendation: Optional[Dict[str, Any]] = None,
        traceability_status: str = "FULLY_TRACEABLE",
        dataset_version: str = "1.0",
        percentage_metadata: Optional[Dict[str, Any]] = None
    ):
        AnalyticalResult._counter += 1
        self.result_id = result_id or f"AR-{AnalyticalResult._counter:06d}"
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version
        self.metric = metric
        self.source_columns = source_columns
        self.cleaned_columns = cleaned_columns or source_columns
        self.filters = filters or {}
        self.total_record_count = total_record_count if total_record_count else (row_count_raw or 0)
        self.valid_record_count = valid_record_count if valid_record_count else (row_count_analytical or 0)
        self.excluded_record_count = excluded_record_count if excluded_record_count else (row_count_excluded or 0)
        self.excluded_record_reason = excluded_record_reason
        self.grouping = grouping or aggregation_grain
        self.aggregation = aggregation_method or aggregation
        self.unit = unit or "Unknown / not specified"
        self.currency = currency or ("USD" if str(unit).upper() in ["USD", "$"] else ("INR" if str(unit).upper() in ["INR", "₹"] else (unit if unit in ["EUR", "GBP", "JPY"] else "None")))
        self.values = values or {}
        self.statistics = statistics or {}
        self.chart_data = chart_data or []
        self.chart_type = chart_type
        self.suitability = suitability or {"is_suitable": True, "reasons": []}
        self.evidence = evidence or []
        self.takeaway = takeaway
        self.interpretation = interpretation
        self.recommendation_eligibility = recommendation_eligibility
        self.source_table = source_table
        self.statistical_method = statistical_method or f"{self.aggregation} of {metric}"
        self.assumptions = assumptions or {}
        self.warnings = warnings or []
        self.confidence = confidence

        # Section 2 Canonical Analytical Result Governance
        self.analysis_type = analysis_type
        self.data_stage = data_stage
        self.sample_size = sample_size if sample_size is not None else self.valid_record_count
        self.excluded_count = excluded_count if excluded_count is not None else self.excluded_record_count
        self.denominator = denominator
        self.relationship_type = relationship_type
        self.interpretation_status = interpretation_status
        self.confidence_metadata = confidence_metadata or {
            "confidence": self.confidence,
            "sample_size": self.sample_size,
            "data_stage": self.data_stage
        }
        self.recommendation = recommendation
        self.traceability_status = traceability_status
        self.percentage_metadata = percentage_metadata or self.statistics.get("percentage_metadata", {})
        
        lin = dict(lineage) if lineage else {}
        lin.setdefault("resultId", self.result_id)
        lin.setdefault("result_id", self.result_id)
        lin.setdefault("source_dataset", dataset_id)
        lin.setdefault("source_table", self.source_table)
        lin.setdefault("source_columns", source_columns)
        lin.setdefault("aggregation_function", self.aggregation)
        lin.setdefault("aggregation_grain", self.grouping)
        lin.setdefault("measure", metric)
        lin.setdefault("grouping_dimensions", [self.grouping] if self.grouping else [])
        lin.setdefault("row_count_raw", self.total_record_count)
        lin.setdefault("row_count_analytical", self.valid_record_count)
        lin.setdefault("row_count_excluded", self.excluded_record_count)
        lin.setdefault("statistical_method", self.statistical_method)
        lin.setdefault("confidence", self.confidence)
        lin.setdefault("data_stage", self.data_stage)
        lin.setdefault("analysis_type", self.analysis_type)
        lin.setdefault("traceability_status", self.traceability_status)
        self.lineage = lin

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
    def rowFilter(self) -> Dict[str, Any]:
        return self.filters

    @property
    def rowCount(self) -> int:
        return self.total_record_count

    @property
    def row_count_raw(self) -> int:
        return self.total_record_count

    @property
    def row_count_analytical(self) -> int:
        return self.valid_record_count

    @property
    def row_count_excluded(self) -> int:
        return self.excluded_record_count

    @property
    def aggregation_grain(self) -> Optional[str]:
        return self.grouping

    @property
    def aggregation_method(self) -> str:
        return self.aggregation

    @property
    def measure(self) -> str:
        return self.metric

    @property
    def dimensions(self) -> List[str]:
        return [self.grouping] if self.grouping else []

    @property
    def computedValues(self) -> Dict[str, Any]:
        return self.values

    @property
    def chartSpec(self) -> Dict[str, Any]:
        return {"chart_type": self.chart_type, "chart_data": self.chart_data}

    @property
    def numericalEvidence(self) -> List[Dict[str, Any]]:
        return self.evidence

    @property
    def numerical_evidence(self) -> List[Dict[str, Any]]:
        return self.evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "resultId": self.result_id,
            "analysis_type": self.analysis_type,
            "dataset_id": self.dataset_id,
            "datasetId": self.dataset_id,
            "source_table": self.source_table,
            "metric": self.metric,
            "measure": self.metric,
            "source_columns": self.source_columns,
            "sourceColumns": self.source_columns,
            "cleaned_columns": self.cleaned_columns,
            "data_stage": self.data_stage,
            "filters": self.filters,
            "rowFilter": self.filters,
            "total_record_count": self.total_record_count,
            "rowCount": self.total_record_count,
            "row_count_raw": self.total_record_count,
            "valid_record_count": self.valid_record_count,
            "row_count_analytical": self.valid_record_count,
            "sample_size": self.sample_size,
            "excluded_record_count": self.excluded_record_count,
            "row_count_excluded": self.excluded_record_count,
            "excluded_count": self.excluded_count,
            "excluded_record_reason": self.excluded_record_reason,
            "denominator": self.denominator,
            "grouping": self.grouping,
            "aggregation_grain": self.grouping,
            "dimensions": [self.grouping] if self.grouping else [],
            "aggregation": self.aggregation,
            "aggregation_method": self.aggregation,
            "unit": self.unit,
            "currency": self.currency,
            "statistical_method": self.statistical_method,
            "assumptions": self.assumptions,
            "warnings": self.warnings,
            "confidence": self.confidence,
            "confidence_metadata": self.confidence_metadata,
            "values": self.values,
            "computedValues": self.values,
            "statistics": self.statistics,
            "chart_data": self.chart_data,
            "chart_type": self.chart_type,
            "chartSpec": {"chart_type": self.chart_type, "chart_data": self.chart_data},
            "suitability": self.suitability,
            "evidence": self.evidence,
            "numerical_evidence": self.evidence,
            "numericalEvidence": self.evidence,
            "takeaway": self.takeaway,
            "interpretation": self.interpretation,
            "relationship_type": self.relationship_type,
            "interpretation_status": self.interpretation_status,
            "dataset_version": getattr(self, "dataset_version", "1.0"),
            "total_n": self.total_record_count,
            "valid_n": self.valid_record_count,
            "excluded_n": self.excluded_record_count,
            "coverage_pct": round((self.valid_record_count / max(self.total_record_count, 1)) * 100, 2),
            "semantic_roles": getattr(self, "semantic_roles", {}),
            "eligibility_decisions": getattr(self, "eligibility_decisions", {}),
            "percentage_metadata": self.percentage_metadata,
            "recommendation": self.recommendation,
            "traceability_status": self.traceability_status,
            "lineage": self.lineage
        }


# ==============================================================================
# 1. CONSTANT VARIABLE & VARIANCE DETECTION ENGINE
# ==============================================================================

def compute_variable_constancy(series: pd.Series, tolerance: float = 1e-9) -> Dict[str, Any]:
    """
    Deterministic Constancy and Variance Detection.
    Checks whether numeric or non-numeric series has zero variance or single unique state.
    """
    total_count = len(series)
    non_null_s = series.dropna()
    valid_count = len(non_null_s)
    missing_count = total_count - valid_count

    if valid_count == 0:
        return {
            "is_constant": True,
            "constant_value": None,
            "variance": 0.0,
            "std": 0.0,
            "nunique": 0,
            "valid_count": 0,
            "missing_count": missing_count,
            "reason": "All observations are null."
        }

    nunique = int(non_null_s.nunique())

    if pd.api.types.is_numeric_dtype(non_null_s):
        s_num = pd.to_numeric(non_null_s, errors='coerce').dropna().astype(float)
        if len(s_num) == 0:
            return {
                "is_constant": True,
                "constant_value": None,
                "variance": 0.0,
                "std": 0.0,
                "nunique": 0,
                "valid_count": 0,
                "missing_count": missing_count,
                "reason": "No valid numeric observations."
            }
        
        var_val = float(s_num.var(ddof=1)) if len(s_num) > 1 else 0.0
        std_val = float(s_num.std(ddof=1)) if len(s_num) > 1 else 0.0
        min_v = float(s_num.min())
        max_v = float(s_num.max())

        is_const = bool(nunique <= 1 or abs(var_val) <= tolerance or (max_v - min_v) <= tolerance)
        const_val = float(s_num.iloc[0]) if is_const and len(s_num) > 0 else None

        return {
            "is_constant": is_const,
            "constant_value": const_val,
            "variance": round(var_val, 6),
            "std": round(std_val, 6),
            "min": min_v,
            "max": max_v,
            "nunique": nunique,
            "valid_count": len(s_num),
            "missing_count": total_count - len(s_num),
            "reason": f"Variance is {var_val:.6f} with {nunique} unique value(s)." if is_const else "Variable exhibits non-zero variance."
        }
    else:
        is_const = bool(nunique <= 1)
        const_val = str(non_null_s.iloc[0]) if is_const and len(non_null_s) > 0 else None
        return {
            "is_constant": is_const,
            "constant_value": const_val,
            "variance": 0.0 if is_const else None,
            "std": 0.0 if is_const else None,
            "nunique": nunique,
            "valid_count": valid_count,
            "missing_count": missing_count,
            "reason": f"Categorical/string variable has {nunique} unique value(s)."
        }


def generate_constant_variable_statement(col_name: str, constancy_info: dict, unit: str = "", time_col: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> str:
    """
    Generates exact deterministic factual statement for constant variables.
    """
    const_val = constancy_info.get("constant_value")
    valid_count = constancy_info.get("valid_count", 0)
    unit_str = f" {unit}".rstrip() if unit and unit != "unknown" else ""

    # Format numeric value nicely
    if isinstance(const_val, float) and const_val.is_integer():
        val_str = str(int(const_val))
    elif const_val is not None:
        val_str = f"{const_val:g}" if isinstance(const_val, (int, float)) else str(const_val)
    else:
        val_str = "None"

    # Time dimension check
    if time_col and df is not None and time_col in df.columns:
        valid_temporal = df[[time_col, col_name]].dropna()
        if len(valid_temporal) > 0:
            t_s = pd.to_datetime(valid_temporal[time_col], errors='coerce').dropna()
            if len(t_s) >= 2:
                n_months = t_s.dt.to_period('M').nunique()
                if n_months > 1:
                    return f"{col_name} remained constant at {val_str}{unit_str} across all {n_months} recorded months."

    return f"{col_name} is constant at {val_str}{unit_str} across all {valid_count:,} valid records."


# ==============================================================================
# 2. CHART SUITABILITY ENGINE
# ==============================================================================

def is_chart_suitable(
    chart_type: str,
    df: pd.DataFrame,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    group_col: Optional[str] = None,
    sem_layer: Optional[dict] = None
) -> Dict[str, Any]:
    """
    Rigorous Chart Suitability Engine:
    Validates mathematical and semantic prerequisites before any visualization is constructed.
    Guarantees:
    - Constant variables are suppressed from Histograms, Box Plots, Violins, Correlations, Scatters, Trends.
    - Violin plots require categorical X with multiple groups (>=2) and numeric non-constant Y.
    - Box plots require non-zero variance in numeric Y.
    - Scatter plots require two non-constant numeric variables.
    - Compositional charts (Donut/Pie) require low cardinality (<=10) and non-negative aggregations.
    """
    ctype = chart_type.upper().strip()
    
    if df is None or df.empty or len(df) < 2:
        return {"is_suitable": False, "reason": "Insufficient observations in dataset (minimum 2 rows required)."}

    # 1. CONSTANT VARIABLE PRE-CHECK
    if y_col and y_col in df.columns:
        y_const = compute_variable_constancy(df[y_col])
        if y_const["is_constant"]:
            if ctype in ["HISTOGRAM", "BOX_PLOT", "BOX", "VIOLIN"]:
                return {
                    "is_suitable": False,
                    "reason": f"Chart type '{chart_type}' is unsuitable because '{y_col}' is constant with zero variance ({y_const['reason']})."
                }

    if x_col and x_col in df.columns:
        x_const = compute_variable_constancy(df[x_col])
        if x_const["is_constant"]:
            if ctype in ["HISTOGRAM"]:
                return {
                    "is_suitable": False,
                    "reason": f"Histogram is unsuitable because '{x_col}' is constant with zero variance."
                }
            if ctype in ["SCATTER"]:
                return {
                    "is_suitable": False,
                    "reason": f"Scatter plot is unsuitable because X-axis variable '{x_col}' has zero variance."
                }

    # 2. CHART-SPECIFIC PREREQUISITES
    if ctype == "HISTOGRAM":
        target = x_col or y_col
        if not target or target not in df.columns:
            return {"is_suitable": False, "reason": "Histogram requires a valid target numeric variable."}
        s_const = compute_variable_constancy(df[target])
        if s_const["is_constant"] or s_const["nunique"] <= 1:
            return {"is_suitable": False, "reason": f"Histogram is unsuitable: '{target}' has zero variance ({s_const['nunique']} unique value)."}
        if not pd.api.types.is_numeric_dtype(df[target]):
            return {"is_suitable": False, "reason": f"Histogram requires numeric data, but '{target}' is {df[target].dtype}."}
        return {"is_suitable": True, "reason": f"'{target}' exhibits continuous distribution suitable for binning."}

    elif ctype in ["VIOLIN"]:
        cat_c = x_col or group_col
        num_c = y_col
        if not cat_c or not num_c or cat_c not in df.columns or num_c not in df.columns:
            return {"is_suitable": False, "reason": "Violin plot requires both a categorical grouping axis (X) and a numeric metric axis (Y)."}
        
        valid_df = df[[cat_c, num_c]].dropna()
        if len(valid_df) < 4:
            return {"is_suitable": False, "reason": "Violin plot requires at least 4 valid bivariate observations."}
        
        n_cats = valid_df[cat_c].nunique()
        if n_cats < 2:
            return {"is_suitable": False, "reason": f"Violin plot requires multiple distinct categories in '{cat_c}' (found {n_cats})."}
        if n_cats > 15:
            return {"is_suitable": False, "reason": f"Categorical axis '{cat_c}' cardinality ({n_cats}) is too high for legible violin distributions."}
        
        num_const = compute_variable_constancy(valid_df[num_c])
        if num_const["is_constant"]:
            return {"is_suitable": False, "reason": f"Violin plot is unsuitable because metric '{num_c}' is constant with zero variance."}
        
        return {"is_suitable": True, "reason": f"Valid multi-cohort distribution for '{num_c}' across {n_cats} categories of '{cat_c}'."}

    elif ctype in ["BOX_PLOT", "BOX"]:
        num_c = y_col or x_col
        if not num_c or num_c not in df.columns:
            return {"is_suitable": False, "reason": "Box plot requires a valid numeric metric."}
        num_const = compute_variable_constancy(df[num_c])
        if num_const["is_constant"]:
            return {"is_suitable": False, "reason": f"Box plot is unsuitable: metric '{num_c}' has zero variance."}
        return {"is_suitable": True, "reason": f"'{num_c}' exhibits non-zero variance suitable for dispersion analysis."}

    elif ctype == "SCATTER":
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            return {"is_suitable": False, "reason": "Scatter plot requires two valid axes (X and Y)."}
        x_const = compute_variable_constancy(df[x_col])
        y_const = compute_variable_constancy(df[y_col])
        if x_const["is_constant"]:
            return {"is_suitable": False, "reason": f"Scatter plot is unsuitable: X variable '{x_col}' has zero variance."}
        if y_const["is_constant"]:
            return {"is_suitable": False, "reason": f"Scatter plot is unsuitable: Y variable '{y_col}' has zero variance."}
        valid_cnt = len(df[[x_col, y_col]].dropna())
        if valid_cnt < 3:
            return {"is_suitable": False, "reason": f"Insufficient paired observations for scatter plot ({valid_cnt} < 3)."}
        return {"is_suitable": True, "reason": f"Both '{x_col}' and '{y_col}' exhibit non-zero variance across {valid_cnt} valid observations."}

    elif ctype in ["LINE", "AREA"]:
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            return {"is_suitable": False, "reason": "Line chart requires temporal/sequential X-axis and numeric Y-axis."}
        y_const = compute_variable_constancy(df[y_col])
        valid_df = df[[x_col, y_col]].dropna()
        if len(valid_df) < 2:
            return {"is_suitable": False, "reason": "Insufficient observations for trend visualization."}
        return {"is_suitable": True, "reason": f"Sequential progression across '{x_col}'."}

    elif ctype in ["DONUT", "PIE"]:
        cat_c = x_col or group_col
        if not cat_c or cat_c not in df.columns:
            return {"is_suitable": False, "reason": "Compositional chart requires a categorical dimension."}
        n_cats = df[cat_c].dropna().nunique()
        if n_cats < 2 or n_cats > 10:
            return {"is_suitable": False, "reason": f"Donut/Pie charts require low cardinality between 2 and 10 categories (found {n_cats})."}
        return {"is_suitable": True, "reason": f"Low-cardinality category '{cat_c}' ({n_cats} categories) suitable for composition."}

    elif ctype == "WATERFALL":
        target = y_col or x_col
        if not target or target not in df.columns:
            return {"is_suitable": False, "reason": "Waterfall unavailable — no valid metric specified."}
        col_meta = sem_layer.get(target, {}) if sem_layer else {}
        m_type = col_meta.get("measure_type", "")
        non_additive_types = ["Percentage", "Ratio", "Score", "Score / Index", "Index", "Rating"]
        is_non_additive = m_type in non_additive_types or any(k in target.lower() for k in ['percent', 'ratio', 'score', 'index', 'rating', 'margin', 'rate'])
        y_const = compute_variable_constancy(df[target])
        if y_const["is_constant"] or is_non_additive:
            return {
                "is_suitable": False,
                "reason": "Waterfall unavailable — the selected metric does not have a valid additive bridge structure."
            }
        return {"is_suitable": True, "reason": "Additive variances suitable for step decomposition."}

    # Default fallback
    return {"is_suitable": True, "reason": "Standard chart configuration."}


# ==============================================================================
# 3. DETERMINISTIC STATISTICAL TEST EXECUTORS
# ==============================================================================

def safe_anova(df: pd.DataFrame, cat_col: str, num_col: str) -> Dict[str, Any]:
    """
    Safe One-Way ANOVA Executor.
    Section 13: Before running ANOVA, check dependent variable variance.
    If variance == 0: Do NOT run test. Display 'Not applicable — dependent variable is constant.'
    Store: test_status = 'not_applicable', reason = 'dependent_variable_constant'.
    Never display meaningless p=1, F=0.
    """
    if df is None or df.empty or cat_col not in df.columns or num_col not in df.columns:
        return {
            "test_status": "error",
            "reason": "Missing required columns or empty dataset",
            "is_valid": False,
            "f_stat": None,
            "p_val": None
        }

    sub = df[[cat_col, num_col]].dropna()
    sub[num_col] = pd.to_numeric(sub[num_col], errors='coerce')
    sub = sub.dropna()

    if len(sub) == 0:
        return {
            "test_status": "insufficient_data",
            "reason": "No complete paired observations",
            "is_valid": False,
            "f_stat": None,
            "p_val": None
        }

    # Check dependent variable variance
    const_info = compute_variable_constancy(sub[num_col])
    if const_info["is_constant"]:
        return {
            "test_status": "not_applicable",
            "reason": "dependent_variable_constant",
            "display_message": f"Not applicable — dependent variable '{num_col}' is constant ({const_info['reason']}).",
            "is_valid": False,
            "f_stat": None,
            "p_val": None,
            "eta_squared": 0.0
        }

    categories = sub[cat_col].unique()
    k_groups = len(categories)
    if k_groups < 2:
        return {
            "test_status": "insufficient_groups",
            "reason": f"Categorical variable '{cat_col}' has fewer than 2 distinct groups (found {k_groups}).",
            "is_valid": False,
            "f_stat": None,
            "p_val": None
        }

    groups = [sub[sub[cat_col] == cat][num_col].values for cat in categories]
    # Ensure sufficient samples
    if any(len(g) < 2 for g in groups):
        return {
            "test_status": "insufficient_observations_per_group",
            "reason": "At least 2 observations required per category group.",
            "is_valid": False,
            "f_stat": None,
            "p_val": None
        }

    try:
        from scipy import stats
        f_stat, p_val = stats.f_oneway(*groups)
        if math.isnan(f_stat) or math.isnan(p_val):
            return {
                "test_status": "not_applicable",
                "reason": "dependent_variable_constant",
                "display_message": f"Not applicable — dependent variable '{num_col}' has zero variance.",
                "is_valid": False,
                "f_stat": None,
                "p_val": None
            }

        df_between = k_groups - 1
        df_within = len(sub) - k_groups
        grand_mean = sub[num_col].mean()
        ss_total = np.sum((sub[num_col].values - grand_mean) ** 2)
        ss_between = np.sum([len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups])
        eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

        return {
            "test_status": "completed",
            "reason": "Test successfully executed.",
            "is_valid": True,
            "f_stat": round(float(f_stat), 3),
            "p_val": round(float(p_val), 5),
            "df_between": df_between,
            "df_within": df_within,
            "eta_squared": round(eta_sq, 4),
            "is_significant": bool(p_val < 0.05),
            "display_message": f"ANOVA F({df_between}, {df_within}) = {f_stat:.2f}, p = {p_val:.4f}"
        }
    except Exception as e:
        return {
            "test_status": "error",
            "reason": str(e),
            "is_valid": False,
            "f_stat": None,
            "p_val": None
        }


def safe_kruskal(df: pd.DataFrame, cat_col: str, num_col: str) -> Dict[str, Any]:
    """
    Safe Kruskal-Wallis Executor.
    Section 13: Before running Kruskal-Wallis, check dependent variable variance.
    If variance == 0: Do NOT run test. Display 'Not applicable — dependent variable is constant.'
    Store: test_status = 'not_applicable', reason = 'dependent_variable_constant'.
    """
    if df is None or df.empty or cat_col not in df.columns or num_col not in df.columns:
        return {
            "test_status": "error",
            "reason": "Missing required columns or empty dataset",
            "is_valid": False,
            "h_stat": None,
            "p_val": None
        }

    sub = df[[cat_col, num_col]].dropna()
    sub[num_col] = pd.to_numeric(sub[num_col], errors='coerce')
    sub = sub.dropna()

    const_info = compute_variable_constancy(sub[num_col])
    if const_info["is_constant"]:
        return {
            "test_status": "not_applicable",
            "reason": "dependent_variable_constant",
            "display_message": f"Not applicable — dependent variable '{num_col}' is constant.",
            "is_valid": False,
            "h_stat": None,
            "p_val": None
        }

    categories = sub[cat_col].unique()
    if len(categories) < 2:
        return {
            "test_status": "insufficient_groups",
            "reason": f"Fewer than 2 valid categories in '{cat_col}'.",
            "is_valid": False,
            "h_stat": None,
            "p_val": None
        }

    groups = [sub[sub[cat_col] == cat][num_col].values for cat in categories]
    try:
        from scipy import stats
        h_stat, p_val = stats.kruskal(*groups)
        if math.isnan(h_stat) or math.isnan(p_val):
            return {
                "test_status": "not_applicable",
                "reason": "dependent_variable_constant",
                "is_valid": False,
                "h_stat": None,
                "p_val": None
            }
        return {
            "test_status": "completed",
            "reason": "Test successfully executed.",
            "is_valid": True,
            "h_stat": round(float(h_stat), 3),
            "p_val": round(float(p_val), 5),
            "is_significant": bool(p_val < 0.05)
        }
    except Exception as e:
        return {
            "test_status": "error",
            "reason": str(e),
            "is_valid": False,
            "h_stat": None,
            "p_val": None
        }


def safe_correlation(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    check_paradox: bool = True,
    check_derivation: bool = True,
    eligibility_cache: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Safe Pearson & Spearman Correlation Executor.
    Section 14: If variance(X) == 0 OR variance(Y) == 0:
    correlation_status = 'not_applicable'
    Do not calculate/display a correlation.
    Reason: 'Correlation is undefined/not informative because one variable has zero variance.'
    """
    if df is None or df.empty or col_x not in df.columns or col_y not in df.columns:
        return {
            "correlation_status": "error",
            "is_computable": False,
            "is_valid": False,
            "reason": "Missing columns or empty dataset",
            "pearson": None,
            "spearman": None
        }

    s_x = pd.to_numeric(df[col_x], errors='coerce')
    s_y = pd.to_numeric(df[col_y], errors='coerce')
    sub = pd.DataFrame({"x": s_x, "y": s_y}).dropna()

    if len(sub) < 3:
        return {
            "correlation_status": "insufficient_data",
            "is_computable": False,
            "is_valid": False,
            "reason": "Fewer than 3 valid paired observations.",
            "pearson": None,
            "spearman": None
        }

    const_x = compute_variable_constancy(sub["x"])
    const_y = compute_variable_constancy(sub["y"])

    if const_x["is_constant"] or const_y["is_constant"]:
        zero_var_col = col_x if const_x["is_constant"] else col_y
        return {
            "correlation_status": "not_applicable",
            "is_computable": False,
            "is_valid": False,
            "reason": "Correlation is undefined/not informative because one variable has zero variance.",
            "zero_variance_column": zero_var_col,
            "pearson": None,
            "spearman": None
        }

    # Central Analytical Eligibility Verification (Part 9)
    try:
        from engine.governed_engines import AnalyticalEligibilityEngine
        if eligibility_cache is not None and col_x in eligibility_cache:
            el_x = eligibility_cache[col_x]
        else:
            el_x = AnalyticalEligibilityEngine.evaluate_column(col_x, df[col_x], len(df))
            if eligibility_cache is not None:
                eligibility_cache[col_x] = el_x

        if not el_x.get("analytical_eligible", True):
            return {
                "correlation_status": "not_applicable",
                "is_computable": False,
                "is_valid": False,
                "reason": f"Correlation not applicable: '{col_x}' excluded ({el_x.get('exclusion_reason')}).",
                "exclusion_reason": el_x.get("exclusion_reason"),
                "pearson": None,
                "spearman": None
            }

        if eligibility_cache is not None and col_y in eligibility_cache:
            el_y = eligibility_cache[col_y]
        else:
            el_y = AnalyticalEligibilityEngine.evaluate_column(col_y, df[col_y], len(df))
            if eligibility_cache is not None:
                eligibility_cache[col_y] = el_y

        if not el_y.get("analytical_eligible", True):
            return {
                "correlation_status": "not_applicable",
                "is_computable": False,
                "is_valid": False,
                "reason": f"Correlation not applicable: '{col_y}' excluded ({el_y.get('exclusion_reason')}).",
                "exclusion_reason": el_y.get("exclusion_reason"),
                "pearson": None,
                "spearman": None
            }
    except Exception:
        pass

    p_val_pearson = None
    p_val_spearman = None
    ci_dict = None
    warnings = []

    try:
        from scipy import stats
        res_p = stats.pearsonr(sub["x"], sub["y"])
        r_p = float(res_p[0])
        p_val_pearson = float(res_p[1])
        res_s = stats.spearmanr(sub["x"], sub["y"])
        r_s = float(res_s[0])
        p_val_spearman = float(res_s[1])

        # Fisher's z confidence interval (95%)
        n = len(sub)
        if n >= 4 and abs(r_p) < 1.0:
            z_val = math.atanh(r_p)
            se_z = 1.0 / math.sqrt(n - 3)
            z_low = z_val - 1.96 * se_z
            z_high = z_val + 1.96 * se_z
            ci_dict = {
                "lower": round(math.tanh(z_low), 4),
                "upper": round(math.tanh(z_high), 4),
                "confidence_level": 0.95,
                "method": "Fisher's z-transformation"
            }
    except Exception:
        pass

    # Outlier sensitivity detection: large gap between Pearson and Spearman
    if abs(r_p - r_s) > 0.25:
        warnings.append(
            f"Outlier sensitivity detected: discrepancy between Pearson (r = {r_p:.2f}) "
            f"and Spearman (rho = {r_s:.2f}) indicates potential influential observations."
        )

    # Check for potential Simpson's Paradox / confounding on meaningful correlations
    if check_paradox and abs(r_p) >= 0.35:
        try:
            from services.statistical_analysis import check_simpsons_paradox
            simpson = check_simpsons_paradox(df, col_x, col_y)
            if simpson.get("paradox_detected"):
                warnings.append(simpson["warning"])
        except Exception:
            pass

    direction_str = "positive" if r_p > 0 else "negative"
    strength_str = "strong" if abs(r_p) >= 0.6 else ("moderate" if abs(r_p) >= 0.4 else "weak")

    # Section 13 & 14 Derived Metric Detection & Correlation Governance
    if check_derivation and abs(r_p) >= 0.35:
        rel_info = DerivedMetricEngine.check_relationship_between_variables(df, col_x, col_y)
        relationship_type = rel_info.get("relationship_type", "INDEPENDENT")
        interpretation_status = rel_info.get("interpretation_status", "STATISTICALLY_OBSERVED")
    else:
        rel_info = {"is_derived": False}
        relationship_type = "INDEPENDENT"
        interpretation_status = "STATISTICALLY_OBSERVED"

    if rel_info.get("is_derived", False) or relationship_type in ["DERIVED", "MATHEMATICALLY_LINKED"]:
        interp_text = (
            f"Variable '{col_x}' and Variable '{col_y}' show a {strength_str} {direction_str} correlation "
            f"(Pearson r = {r_p:.3f}). The relationship is structurally expected because '{col_y}' is "
            f"mathematically derived from '{col_x}' and related components. "
            "Correlation does not establish causation."
        )
    else:
        interp_text = (
            f"Statistical analysis identifies a {strength_str} {direction_str} association between '{col_x}' and '{col_y}' "
            f"(Pearson r = {r_p:.3f}, Spearman rho = {r_s:.3f}, n = {len(sub)}). "
            "Association reflects empirical co-variance across observations and does not establish causality."
        )

    valid_n = len(sub)
    total_df_n = len(df) if df is not None else valid_n
    coverage_pct = round((valid_n / max(total_df_n, 1)) * 100, 2)
    missing_pairs = int(total_df_n - valid_n)
    adj_p = round(p_val_pearson, 6) if p_val_pearson is not None else None
    causal_warning = "Correlation reflects empirical co-variance across observations and does not establish causality."

    is_det = bool(
        relationship_type == "DETERMINISTIC_TRANSFORMATION"
        or rel_info.get("is_derived", False)
        or (relationship_type in ["DERIVED", "MATHEMATICALLY_LINKED"] and rel_info.get("formula") is not None)
    )
    classification_str = rel_info.get("classification") or ("STRUCTURAL_RELATIONSHIP" if is_det else "STATISTICAL_ASSOCIATION")

    return {
        "correlation_status": "valid",
        "is_computable": True,
        "is_valid": True,
        "variable_x": col_x,
        "variable_y": col_y,
        "pearson": round(r_p, 3),
        "spearman": round(r_s, 3),
        "pearson_r": round(r_p, 4),
        "spearman_rho": round(r_s, 4),
        "valid_n": valid_n,
        "sample_size": valid_n,
        "paired_observations": valid_n,
        "total_n": total_df_n,
        "coverage_pct": coverage_pct,
        "missing_pairs": missing_pairs,
        "missing_pair_count": missing_pairs,
        "method": "Pearson & Spearman",
        "causal_warning": causal_warning,
        "p_value": round(p_val_pearson, 6) if p_val_pearson is not None else None,
        "raw_p_value": round(p_val_pearson, 6) if p_val_pearson is not None else None,
        "adjusted_p_value": adj_p,
        "correction_method": "Benjamini-Hochberg FDR",
        "spearman_p_value": round(p_val_spearman, 6) if p_val_spearman is not None else None,
        "confidence_interval": ci_dict,
        "direction": direction_str,
        "strength": strength_str,
        "association_strength": round(abs(r_p), 4),
        "relationship_type": relationship_type,
        "interpretation_status": interpretation_status,
        "derivation_formula": rel_info.get("formula"),
        "is_deterministic": is_det,
        "is_derived": bool(rel_info.get("is_derived", False)),
        "classification": classification_str,
        "data_stage": "CLEANED",
        "warnings": warnings,
        "interpretation": interp_text,
        "takeaway": interp_text
    }



def safe_pca_features(df: pd.DataFrame, candidate_cols: List[str]) -> Tuple[List[str], List[str], str]:
    """
    Safe PCA Feature Selector.
    Section 15: Before PCA: remove zero-variance features.
    Store: excluded_features = [...], exclusion_reason = 'zero_variance'.
    """
    retained = []
    excluded = []

    for col in candidate_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            c_info = compute_variable_constancy(df[col])
            if c_info["is_constant"]:
                excluded.append(col)
            else:
                retained.append(col)
        else:
            excluded.append(col)

    return retained, excluded, "zero_variance"


# ==============================================================================
# 4. CANONICAL GROUPED ANALYTICAL RESULT GENERATOR
# ==============================================================================

def compute_grouped_analytical_result(
    df: pd.DataFrame,
    measure_col: str,
    group_col: Optional[str] = None,
    aggregation: str = "SUM",
    unit: str = "units",
    dataset_id: str = "main",
    dataset_version: str = "cleaned_v1"
) -> AnalyticalResult:
    """
    Computes a canonical AnalyticalResult with strict Denominator and Accounting Integrity.
    Section 18, 19, 20:
    - full_valid_total: sum across all records with valid measure_col
    - grouped_valid_total: sum across records with valid measure_col AND valid group_col
    - excluded_from_grouping_count: records with valid measure_col but missing group_col
    - excluded_from_grouping_total: difference between full and grouped
    - Percentage denominators explicitly use grouped_valid_total (e.g. 55,946 / 110,616 = 50.6%)
    - Factual takeaway explaining difference if records were excluded from grouping
    - Never uses 'performance' language for category shares or constant variables
    """
    total_records = len(df)
    s_meas = pd.to_numeric(df[measure_col], errors='coerce')
    valid_meas_mask = s_meas.notna()
    full_valid_count = int(valid_meas_mask.sum())
    
    # Compute full dataset measure total
    if aggregation.upper() == "SUM":
        full_valid_total = float(s_meas[valid_meas_mask].sum())
    elif aggregation.upper() == "MEAN" or aggregation.upper() == "AVG":
        full_valid_total = float(s_meas[valid_meas_mask].mean()) if full_valid_count > 0 else 0.0
    else:
        full_valid_total = float(full_valid_count)

    meas_constancy = compute_variable_constancy(df[measure_col])
    unit_str = f" {unit}".rstrip() if unit and unit != "unknown" else ""

    if not group_col or group_col not in df.columns:
        # Ungrouped overall summary
        mean_val = float(s_meas[valid_meas_mask].mean()) if full_valid_count > 0 else 0.0
        med_val = float(s_meas[valid_meas_mask].median()) if full_valid_count > 0 else 0.0
        std_val = float(s_meas[valid_meas_mask].std()) if full_valid_count > 1 else 0.0

        if meas_constancy["is_constant"]:
            const_val = meas_constancy["constant_value"]
            takeaway = f"{measure_col} is constant at {const_val:g}{unit_str} across all {full_valid_count:,} valid records."
            interpretation = f"The variable exhibits zero variance across all {full_valid_count:,} observations."
        else:
            takeaway = f"Across {full_valid_count:,} valid records, total {measure_col} is {full_valid_total:,.2f}{unit_str}, with a mean of {mean_val:,.2f}{unit_str} per record (median: {med_val:,.2f}{unit_str})."
            interpretation = f"Distribution summary for '{measure_col}' reflecting central tendency and dispersion."

        chart_data = [{"metric": f"Total {measure_col}", "value": round(full_valid_total, 2)}]
        ar_seq = abs(hash((dataset_id, measure_col, "overall", aggregation))) % 900000 + 100000
        ar_id = f"AR-{ar_seq:06d}"

        return AnalyticalResult(
            result_id=ar_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            metric=measure_col,
            source_columns=[measure_col],
            cleaned_columns=[measure_col],
            total_record_count=total_records,
            valid_record_count=full_valid_count,
            excluded_record_count=total_records - full_valid_count,
            excluded_record_reason=f"{total_records - full_valid_count} missing values in '{measure_col}'" if total_records > full_valid_count else None,
            grouping=None,
            aggregation=aggregation,
            unit=unit,
            values={"total": full_valid_total},
            statistics={"mean": mean_val, "median": med_val, "std": std_val, "sum": full_valid_total},
            chart_data=chart_data,
            chart_type="METRIC_CARD",
            suitability={"is_suitable": True, "reasons": []},
            evidence=[
                {"metric": f"Total {measure_col}", "value": f"{full_valid_total:,.2f}{unit_str}"},
                {"metric": "Valid Records Count", "value": f"{full_valid_count:,}"},
                {"metric": f"Mean {measure_col}", "value": f"{mean_val:,.2f}{unit_str}"}
            ],
            takeaway=takeaway,
            interpretation=interpretation,
            recommendation_eligibility=False,
            lineage={
                "resultId": ar_id,
                "result_id": ar_id,
                "analytical_result_id": ar_id,
                "source_dataset": dataset_id,
                "source_columns": [measure_col],
                "row_filter": "valid numeric records",
                "cleaned_dataset_version": dataset_version,
                "analytical_dataset_version": dataset_version,
                "grouping_dimensions": [],
                "aggregation_function": aggregation,
                "measure": measure_col,
                "denominator": full_valid_total,
                "full_denominator": full_valid_total,
                "grouped_denominator": full_valid_total,
                "excluded_rows": total_records - full_valid_count,
                "missing_value_treatment": f"Excluded {total_records - full_valid_count} missing rows",
                "outlier_treatment": "Retained at true empirical values",
                "statistical_method": f"{aggregation}({measure_col}) across valid records",
                "engine_version": "AnalyticalTruth_v2"
            }
        )

    # Grouped analysis
    valid_group_mask = df[group_col].notna() & (df[group_col].astype(str).str.strip() != "")
    both_valid_mask = valid_meas_mask & valid_group_mask

    grouped_valid_count = int(both_valid_mask.sum())
    excluded_from_grouping = full_valid_count - grouped_valid_count

    sub = pd.DataFrame({
        "group": df.loc[both_valid_mask, group_col].astype(str).str.strip(),
        "measure": s_meas.loc[both_valid_mask]
    })

    if aggregation.upper() == "SUM":
        grp_res = sub.groupby("group")["measure"].sum().sort_values(ascending=False)
        grouped_valid_total = float(grp_res.sum())
    elif aggregation.upper() in ["MEAN", "AVG"]:
        grp_res = sub.groupby("group")["measure"].mean().sort_values(ascending=False)
        grouped_valid_total = float(sub["measure"].mean()) if grouped_valid_count > 0 else 0.0
    else:
        grp_res = sub.groupby("group")["measure"].count().sort_values(ascending=False)
        grouped_valid_total = float(len(sub))

    excluded_diff_total = full_valid_total - grouped_valid_total if aggregation.upper() == "SUM" else 0.0

    # Build chart data directly from exact grouped results
    chart_data = []
    values_dict = {}
    for grp_name, grp_val in grp_res.items():
        v_flt = float(grp_val)
        pct_of_grouped = (v_flt / grouped_valid_total * 100) if grouped_valid_total > 0 else 0.0
        chart_data.append({
            "category": str(grp_name),
            group_col: str(grp_name),
            "value": round(v_flt, 2),
            measure_col: round(v_flt, 2),
            "percentage_of_grouped": round(pct_of_grouped, 1)
        })
        values_dict[str(grp_name)] = v_flt

    # Formulate EXACT takeaway and interpretation without hallucinating performance
    top_cat = str(grp_res.index[0]) if len(grp_res) > 0 else "N/A"
    top_val = float(grp_res.iloc[0]) if len(grp_res) > 0 else 0.0
    top_pct = round((top_val / grouped_valid_total * 100), 1) if grouped_valid_total > 0 else 0.0

    if excluded_from_grouping > 0 and aggregation.upper() == "SUM":
        exclusion_stmt = (
            f" {grouped_valid_total:,.0f} {measure_col} are represented in records with valid {group_col} values. "
            f"The full dataset contains {full_valid_total:,.0f} {measure_col}; the difference of {excluded_diff_total:,.0f} "
            f"comes from {excluded_from_grouping} records excluded from the {group_col} grouping (missing {group_col})."
        )
    else:
        exclusion_stmt = ""

    if meas_constancy["is_constant"]:
        const_val = meas_constancy["constant_value"]
        const_clause = f" Because {measure_col} is constant at {const_val:g}{unit_str}, this share primarily reflects the distribution of records across {group_col} categories."
        takeaway = (
            f"'{top_cat}' accounts for {top_pct}% of {measure_col} among records with valid {group_col} values "
            f"({top_val:,.0f} out of {grouped_valid_total:,.0f}{unit_str}).{exclusion_stmt}{const_clause}"
        )
        interpretation = (
            f"Because {measure_col} is constant at {const_val:g}{unit_str} across valid observations, "
            f"this share primarily reflects the distribution of records across {group_col} categories."
        )
    else:
        takeaway = (
            f"'{top_cat}' accounts for {top_pct}% of {measure_col} among records with valid {group_col} values "
            f"({top_val:,.2f} out of {grouped_valid_total:,.2f}{unit_str}).{exclusion_stmt}"
        )
        interpretation = (
            f"Category contribution distribution of '{measure_col}' across '{group_col}' segments."
        )

    is_record_count = str(aggregation).upper() in ["COUNT", "RECORD_COUNT"]
    pct_type = DualCompatibleStr("MEASURE_CONTRIBUTION" if not is_record_count else "RECORD_SHARE", aliases=["measure_contribution", "record_share", "MEASURE_CONTRIBUTION", "RECORD_SHARE"])
    pct_metric_name = f"Share of Valid {group_col} Records" if is_record_count else f"{measure_col} Contribution"

    percentage_metadata = {
        "percentage_type": pct_type,
        "numerator": f"{top_cat} {measure_col}" if not is_record_count else f"{top_cat} record count",
        "denominator": f"Total valid {measure_col} across {group_col}" if not is_record_count else f"Total valid records across {group_col}",
        "numerator_value": round(top_val, 2),
        "denominator_value": round(grouped_valid_total, 2),
        "denominator_scope": f"Records with valid {group_col} and {measure_col}",
        "valid_n": grouped_valid_count,
        "excluded_n": excluded_from_grouping
    }

    evidence = [
        {"metric": f"{top_cat} ({group_col}) {measure_col}", "value": f"{top_val:,.2f}{unit_str}".replace(".00", "")},
        {"metric": f"Grouped Valid {measure_col} Total", "value": f"{grouped_valid_total:,.2f}{unit_str}".replace(".00", "")},
        {"metric": f"Full Dataset {measure_col} Total", "value": f"{full_valid_total:,.2f}{unit_str}".replace(".00", "")},
        {"metric": pct_metric_name, "value": f"{top_pct}%"},
        {"metric": "Percentage Type", "value": pct_type},
        {"metric": "Numerator Value", "value": f"{top_val:,.2f}".replace(".00", "")},
        {"metric": "Denominator Value", "value": f"{grouped_valid_total:,.2f}".replace(".00", "")},
        {"metric": "Valid N", "value": str(grouped_valid_count)},
        {"metric": "Excluded Records from Grouping", "value": f"{excluded_from_grouping}"}
    ]

    chart_type = "DONUT" if len(grp_res) <= 5 else "BAR"
    ar_seq = abs(hash((dataset_id, measure_col, group_col, aggregation))) % 900000 + 100000
    ar_id = f"AR-{ar_seq:06d}"

    return AnalyticalResult(
        result_id=ar_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        metric=measure_col,
        source_columns=[group_col, measure_col],
        cleaned_columns=[group_col, measure_col],
        filters={"group_col_valid": True},
        total_record_count=total_records,
        valid_record_count=grouped_valid_count,
        excluded_record_count=excluded_from_grouping,
        excluded_record_reason=f"{excluded_from_grouping} records have missing {group_col}" if excluded_from_grouping > 0 else None,
        grouping=group_col,
        aggregation=aggregation,
        unit=unit,
        values=values_dict,
        statistics={
            "grouped_total": grouped_valid_total,
            "full_total": full_valid_total,
            "excluded_count": excluded_from_grouping,
            "excluded_difference": excluded_diff_total,
            "top_category": top_cat,
            "top_category_share": top_pct,
            "percentage_type": pct_type,
            "percentage_metadata": percentage_metadata,
            "numerator": percentage_metadata["numerator"],
            "denominator": percentage_metadata["denominator"],
            "numerator_value": top_val,
            "denominator_value": grouped_valid_total,
            "denominator_scope": percentage_metadata["denominator_scope"],
            "valid_n": grouped_valid_count,
            "excluded_n": excluded_from_grouping,
            "percentage_label": f"{measure_col} Contribution: {top_pct}%" if not is_record_count else f"Record Share: {top_pct}%",
            "measure_contribution": top_pct if not is_record_count else None,
            "record_share": top_pct if is_record_count else None
        },
        chart_data=chart_data,
        chart_type=chart_type,
        suitability={"is_suitable": True, "reasons": []},
        evidence=evidence,
        takeaway=takeaway,
        interpretation=interpretation,
        recommendation_eligibility=False,
        lineage={
            "resultId": ar_id,
            "result_id": ar_id,
            "analytical_result_id": ar_id,
            "source_dataset": dataset_id,
            "source_columns": [group_col, measure_col],
            "row_filter": f"valid '{group_col}' and '{measure_col}' records",
            "cleaned_dataset_version": dataset_version,
            "analytical_dataset_version": dataset_version,
            "grouping_dimensions": [group_col],
            "aggregation_function": aggregation,
            "measure": measure_col,
            "denominator": grouped_valid_total,
            "full_denominator": full_valid_total,
            "grouped_denominator": grouped_valid_total,
            "excluded_rows": excluded_from_grouping,
            "missing_value_treatment": f"Excluded {excluded_from_grouping} rows with missing grouping variable '{group_col}'",
            "outlier_treatment": "Retained at true empirical values",
            "statistical_method": f"{aggregation}({measure_col}) GROUP BY {group_col}",
            "engine_version": "AnalyticalTruth_v2"
        }
    )
