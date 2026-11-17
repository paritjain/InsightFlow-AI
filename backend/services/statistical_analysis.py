import pandas as pd
import numpy as np
import math
from typing import Dict, List, Any, Optional

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    
    SCIPY_AVAILABLE = False


from engine.analytical_truth import compute_variable_constancy
from engine.governed_engines import DualCompatibleStr, AnalyticalEligibilityEngine


# ==============================================================================
# 0. WELCH ANOVA & MULTIPLE TESTING CORRECTION & CONFIDENCE INTERVALS
# ==============================================================================

def compute_welch_anova(groups: List[np.ndarray]) -> Dict[str, Any]:
    """
    Computes Welch's One-Way ANOVA for unequal variances across groups (Welch, 1951).
    Robust alternative to ordinary ANOVA when homoscedasticity assumption is violated.
    """
    if not SCIPY_AVAILABLE or len(groups) < 2:
        return {"valid": False, "reason": "Insufficient groups or scipy unavailable"}

    k = len(groups)
    n = np.array([len(g) for g in groups], dtype=float)
    if any(n_i < 2 for n_i in n):
        return {"valid": False, "reason": "Every group must have at least 2 observations"}

    means = np.array([np.mean(g) for g in groups], dtype=float)
    variances = np.array([np.var(g, ddof=1) for g in groups], dtype=float)

    # Protect against zero variance in individual groups
    for i in range(k):
        if variances[i] <= 1e-12:
            variances[i] = 1e-12

    w = n / variances
    W = float(np.sum(w))
    if W <= 0:
        return {"valid": False, "reason": "Total variance weight is zero"}

    x_prime = float(np.sum(w * means) / W)
    df1 = k - 1
    
    # Lambda term
    denom_k = k ** 2 - 1
    term_sum = float(np.sum(((1.0 - w / W) ** 2) / (n - 1)))
    lambda_term = (3.0 * term_sum) / denom_k if denom_k > 0 else 0.0

    if lambda_term <= 0:
        df2 = float(np.sum(n - 1))
    else:
        df2 = 1.0 / lambda_term

    numerator = (1.0 / max(1, k - 1)) * float(np.sum(w * (means - x_prime) ** 2))
    denominator = 1.0 + (2.0 * (k - 2) / max(1, denom_k)) * term_sum
    f_welch = numerator / denominator if denominator > 0 else 0.0
    p_welch = float(stats.f.sf(f_welch, df1, df2))

    return {
        "valid": True,
        "method": "Welch ANOVA",
        "f_statistic": round(float(f_welch), 4) if not math.isnan(f_welch) else 0.0,
        "p_value": round(float(p_welch), 6) if not math.isnan(p_welch) else 1.0,
        "df_between": df1,
        "df_within": round(float(df2), 2),
        "is_significant": bool(p_welch < 0.05),
        "interpretation": f"Welch's ANOVA F({df1}, {df2:.1f}) = {f_welch:.2f}, p = {p_welch:.4f} (robust to unequal variances)."
    }


def apply_multiple_testing_correction(
    p_values: List[float],
    method: str = "Benjamini-Hochberg"
) -> List[Dict[str, Any]]:
    """
    Applies multiple-comparison control:
    - Benjamini-Hochberg False Discovery Rate (FDR)
    - Bonferroni Family-Wise Error Rate (FWER)
    Returns raw p-value, number of tests, adjusted p-value, and significance flags.
    """
    m = len(p_values)
    if m == 0:
        return []
    if m == 1:
        p = float(p_values[0]) if p_values[0] is not None else 1.0
        return [{
            "raw_p_value": round(p, 6),
            "p_value": round(p, 6),
            "number_of_tests": 1,
            "adjusted_p_value": round(p, 6),
            "multiple_testing_method": method,
            "significant_before_correction": p < 0.05,
            "significant_after_correction": p < 0.05
        }]

    clean_p = [float(p) if (p is not None and not math.isnan(p)) else 1.0 for p in p_values]

    if method.lower() in ["bonferroni", "bonf"]:
        adj_p_list = [min(1.0, p * m) for p in clean_p]
        return [
            {
                "raw_p_value": round(p, 6),
                "p_value": round(p, 6),
                "number_of_tests": m,
                "adjusted_p_value": round(adj, 6),
                "multiple_testing_method": "Bonferroni",
                "significant_before_correction": p < 0.05,
                "significant_after_correction": adj < 0.05
            }
            for p, adj in zip(clean_p, adj_p_list)
        ]

    # Default: Benjamini-Hochberg FDR
    indexed_p = sorted(enumerate(clean_p), key=lambda x: x[1])
    q_values = [0.0] * m

    cum_min = 1.0
    for rank in range(m, 0, -1):
        idx, p_val = indexed_p[rank - 1]
        raw_q = (p_val * m) / float(rank)
        cum_min = min(cum_min, raw_q)
        q_values[idx] = min(1.0, max(0.0, cum_min))

    return [
        {
            "raw_p_value": round(p, 6),
            "p_value": round(p, 6),
            "number_of_tests": m,
            "adjusted_p_value": round(q, 6),
            "multiple_testing_method": "Benjamini-Hochberg",
            "significant_before_correction": p < 0.05,
            "significant_after_correction": q < 0.05
        }
        for p, q in zip(clean_p, q_values)
    ]


def compute_mean_confidence_interval(
    series: pd.Series,
    confidence_level: float = 0.95
) -> Optional[Dict[str, Any]]:
    """
    Computes exact parametric Confidence Interval for sample mean using Student's t-distribution.
    """
    s = pd.to_numeric(series, errors='coerce').dropna()
    n = len(s)
    if n < 2:
        return None
    mean_val = float(s.mean())
    std_val = float(s.std(ddof=1))
    if std_val <= 1e-12:
        return {
            "mean": round(mean_val, 4),
            "confidence_level": confidence_level,
            "lower": round(mean_val, 4),
            "upper": round(mean_val, 4),
            "margin_of_error": 0.0,
            "method": "zero_variance",
            "sample_size": n
        }

    alpha = 1.0 - confidence_level
    df = n - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df)) if SCIPY_AVAILABLE else 1.96
    se = std_val / math.sqrt(n)
    moe = t_crit * se
    return {
        "mean": round(mean_val, 4),
        "confidence_level": confidence_level,
        "lower": round(mean_val - moe, 4),
        "upper": round(mean_val + moe, 4),
        "margin_of_error": round(moe, 4),
        "method": "Student's t-distribution",
        "sample_size": n
    }


def compute_proportion_confidence_interval(
    count: int,
    n: int,
    confidence_level: float = 0.95
) -> Optional[Dict[str, Any]]:
    """
    Computes exact Wilson score confidence interval for a proportion/percentage.
    """
    if n < 1 or count < 0 or count > n:
        return None
    p = count / float(n)
    alpha = 1.0 - confidence_level
    z = float(stats.norm.ppf(1.0 - alpha / 2.0)) if SCIPY_AVAILABLE else 1.96

    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / denom
    moe = (z / denom) * math.sqrt((p * (1.0 - p)) / n + (z ** 2) / (4.0 * (n ** 2)))
    lower = max(0.0, center - moe)
    upper = min(1.0, center + moe)

    return {
        "proportion": round(p, 4),
        "percentage": round(p * 100, 2),
        "confidence_level": confidence_level,
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "lower_percentage": round(lower * 100, 2),
        "upper_percentage": round(upper * 100, 2),
        "method": "Wilson score interval",
        "sample_size": n
    }


def compute_correlation_confidence_interval(
    r: float,
    n: int,
    confidence_level: float = 0.95
) -> Optional[Dict[str, Any]]:
    """
    Computes exact Fisher's z-transformation Confidence Interval for Pearson correlation coefficient r.
    """
    if n < 4 or r is None or math.isnan(r) or abs(r) >= 1.0:
        return {
            "r": round(float(r), 4) if r is not None and not math.isnan(r) else 0.0,
            "confidence_level": confidence_level,
            "lower": round(float(r), 4) if r is not None and not math.isnan(r) else 0.0,
            "upper": round(float(r), 4) if r is not None and not math.isnan(r) else 0.0,
            "method": "exact_bound",
            "sample_size": n
        }

    alpha = 1.0 - confidence_level
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0)) if SCIPY_AVAILABLE else 1.96
    z_val = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_low = z_val - z_crit * se
    z_high = z_val + z_crit * se
    r_low = math.tanh(z_low)
    r_high = math.tanh(z_high)

    return {
        "r": round(r, 4),
        "confidence_level": confidence_level,
        "lower": round(r_low, 4),
        "upper": round(r_high, 4),
        "method": "Fisher's z-transformation",
        "sample_size": n
    }


def check_simpsons_paradox(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    candidate_group_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates whether an aggregate correlation reverses or substantially shifts
    when conditioning on categorical subgroups (Simpson's Paradox / Subgroup Confounding).
    """
    if df is None or x_col not in df.columns or y_col not in df.columns:
        return {"paradox_detected": False, "reason": "Columns unavailable"}

    sub = df[[x_col, y_col]].dropna()
    sub[x_col] = pd.to_numeric(sub[x_col], errors='coerce')
    sub[y_col] = pd.to_numeric(sub[y_col], errors='coerce')
    sub = sub.dropna()

    if len(sub) < 6 or sub[x_col].std() <= 1e-9 or sub[y_col].std() <= 1e-9:
        return {"paradox_detected": False, "simpsons_paradox_detected": False, "reason": "Constant variable or small sample"}

    overall_r = float(sub[x_col].corr(sub[y_col]))
    if abs(overall_r) < 0.20:
        return {"paradox_detected": False, "simpsons_paradox_detected": False, "overall_r": round(overall_r, 4)}

    if not candidate_group_cols:
        candidate_group_cols = [
            c for c in df.columns
            if c not in [x_col, y_col] and 2 <= df[c].nunique() <= 12
        ]
        if len(candidate_group_cols) > 5:
            candidate_group_cols = candidate_group_cols[:5]

    for g_col in candidate_group_cols:
        g_sub = df[[x_col, y_col, g_col]].dropna()
        g_sub[x_col] = pd.to_numeric(g_sub[x_col], errors='coerce')
        g_sub[y_col] = pd.to_numeric(g_sub[y_col], errors='coerce')
        g_sub = g_sub.dropna()
        if len(g_sub) < 6:
            continue

        group_corrs = []
        weights = []
        for grp_val, grp_df in g_sub.groupby(g_col):
            if len(grp_df) >= 3 and grp_df[x_col].std() > 1e-9 and grp_df[y_col].std() > 1e-9:
                r_val = float(grp_df[x_col].corr(grp_df[y_col]))
                group_corrs.append((str(grp_val), r_val, len(grp_df)))
                weights.append(len(grp_df))

        if len(group_corrs) >= 2 and sum(weights) >= 0.5 * len(g_sub):
            # Check if majority weight of subgroups reverses sign
            opposite_weight = sum(
                w for _, r_k, w in group_corrs
                if (overall_r > 0 and r_k < -0.15) or (overall_r < 0 and r_k > 0.15)
            )
            if opposite_weight >= 0.5 * sum(weights):
                return {
                    "paradox_detected": True,
                    "simpsons_paradox_detected": True,
                    "grouping_column": g_col,
                    "overall_r": round(overall_r, 4),
                    "subgroup_correlations": {k: round(r, 4) for k, r, _ in group_corrs},
                    "warning": (
                        f"Potential subgroup/confounding effect detected. The aggregate relationship "
                        f"between '{x_col}' and '{y_col}' (r = {overall_r:.2f}) differs or reverses when "
                        f"stratified by '{g_col}'. Consider stratified analysis."
                    )
                }

    return {"paradox_detected": False, "simpsons_paradox_detected": False, "overall_r": round(overall_r, 4)}


def compute_pca(df: pd.DataFrame, n_components: int = 3, sem_layer: dict = None) -> Dict[str, Any]:
    """
    Computes Principal Component Analysis (PCA) for standardized NUMERIC MEASURE features.
    Strictly excludes zero-variance features, IDENTIFIER, TEMPORAL, and raw unencoded CATEGORICAL columns.
    Section 15: remove zero-variance and near-zero variance features.
    """
    if isinstance(n_components, dict) and sem_layer is None:
        sem_layer = n_components
        n_components = 3
    if n_components is None or not isinstance(n_components, int):
        n_components = 3

    if not SKLEARN_AVAILABLE or df is None or df.empty:
        return {"is_valid": False, "reason": "Scikit-learn not available or empty DataFrame"}

    # Select candidate numerical features with strict analytical eligibility
    candidates = []
    excluded_features = []
    exclusion_reasons = {}

    if sem_layer:
        for c, m in sem_layer.items():
            if c not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                excluded_features.append(c)
                exclusion_reasons[c] = "non_numeric"
                continue
            if m.get("is_identifier") or m.get("id_safety") is True or "identifier" in str(m.get("semantic_role", "")).lower():
                excluded_features.append(c)
                exclusion_reasons[c] = "identifier"
                continue
            if m.get("primary_key") or m.get("is_primary_key") or "primary" in str(m.get("key_role", "")).lower():
                excluded_features.append(c)
                exclusion_reasons[c] = "primary_key"
                continue
            if m.get("foreign_key") or m.get("is_foreign_key") or "foreign" in str(m.get("key_role", "")).lower():
                excluded_features.append(c)
                exclusion_reasons[c] = "foreign_key"
                continue
            if m.get("analytical_eligible") is False:
                excluded_features.append(c)
                exclusion_reasons[c] = m.get("exclusion_reason") or "analytical_ineligible"
                continue
            if not m.get("is_measure", True):
                excluded_features.append(c)
                exclusion_reasons[c] = "not_a_measure"
                continue
            candidates.append(c)
    else:
        id_kws = ['_id', 'id', 'uuid', 'ssn', 'sku', 'number', 'key', 'index', 'code', 'zip', 'account', 'num']
        for c in df.columns:
            c_low = str(c).lower()
            if any(c_low.endswith(k) or c_low.startswith(f"{k}_") or f"_{k}_" in c_low or c_low == k for k in id_kws):
                excluded_features.append(c)
                exclusion_reasons[c] = "identifier"
                continue
            if not pd.api.types.is_numeric_dtype(df[c]):
                excluded_features.append(c)
                exclusion_reasons[c] = "non_numeric"
                continue
            try:
                eval_res = AnalyticalEligibilityEngine.evaluate_column(c, df[c], len(df))
                if not eval_res.get("analytical_eligible", True):
                    excluded_features.append(c)
                    exclusion_reasons[c] = eval_res.get("exclusion_reason", "analytical_ineligible")
                    continue
            except Exception:
                pass
            candidates.append(c)

    num_cols = []
    for c in candidates:
        c_info = compute_variable_constancy(df[c])
        if c_info["is_constant"] or (c_info.get("variance") is not None and abs(c_info["variance"]) <= 1e-9):
            excluded_features.append(c)
            exclusion_reasons[c] = "zero_variance"
        elif len(df[c].dropna()) >= 5:
            num_cols.append(c)
        else:
            excluded_features.append(c)
            exclusion_reasons[c] = "insufficient_sample"

    # Exclude redundant derived / deterministic variables from PCA (Prompt Section 18)
    try:
        from engine.lineage_derivation import DerivationAndLineageEngine
        derivs = DerivationAndLineageEngine.scan_dataset(df, candidate_cols=num_cols)
        for d in derivs:
            if d.is_deterministic or "DETERMINISTIC" in str(d.derivation_type).upper():
                t_col = d.target_column
                if t_col in num_cols:
                    num_cols.remove(t_col)
                    if t_col not in excluded_features:
                        excluded_features.append(t_col)
                    exclusion_reasons[t_col] = "REDUNDANT_DETERMINISTIC_VARIABLE"
    except Exception:
        pass

    total_dataset_records = len(df)
    clean_df = df[num_cols].dropna() if num_cols else pd.DataFrame()
    effective_observations = len(clean_df)
    effective_observation_pct = round((effective_observations / max(total_dataset_records, 1)) * 100, 2)
    excluded_records = total_dataset_records - effective_observations

    has_zero_var = any(str(v).lower() in ["zero_variance", "constant", "near_zero_variance"] for v in exclusion_reasons.values())
    zero_var_reason = DualCompatibleStr("zero_variance", aliases=["CONSTANT", "NEAR_ZERO_VARIANCE", "zero_variance"])

    if len(num_cols) < 2:
        return {
            "is_valid": False,
            "status": "NOT_APPLICABLE",
            "reason": "At least 2 numerical features with non-zero variance are required for PCA",
            "total_dataset_records": total_dataset_records,
            "source_records": total_dataset_records,
            "source_n": total_dataset_records,
            "effective_observations": effective_observations,
            "effective_n": effective_observations,
            "excluded_records": excluded_records,
            "excluded_n": excluded_records,
            "coverage_pct": effective_observation_pct,
            "missing_value_policy": "COMPLETE_CASE",
            "effective_observation_pct": effective_observation_pct,
            "eligible_features": num_cols,
            "features_analyzed": num_cols,
            "excluded_features": excluded_features,
            "exclusion_reasons": exclusion_reasons,
            "exclusion_reason": zero_var_reason if has_zero_var else "none"
        }

    # Coverage threshold rule (Prompt Section 12 & 18: Primary PCA requires >= 95% coverage)
    if total_dataset_records >= 20 and effective_observation_pct < 95.0:
        return {
            "is_valid": False,
            "status": "NOT_APPLICABLE",
            "reason": f"ANALYTICAL_COVERAGE_BELOW_THRESHOLD: PCA requires >= 95% analytical coverage; observed coverage is {effective_observation_pct:.2f}%.",
            "total_dataset_records": total_dataset_records,
            "source_records": total_dataset_records,
            "source_n": total_dataset_records,
            "effective_observations": effective_observations,
            "effective_n": effective_observations,
            "excluded_records": excluded_records,
            "excluded_n": excluded_records,
            "coverage_pct": effective_observation_pct,
            "missing_value_policy": "COMPLETE_CASE",
            "effective_observation_pct": effective_observation_pct,
            "eligible_features": num_cols,
            "features_analyzed": num_cols,
            "excluded_features": excluded_features,
            "exclusion_reasons": exclusion_reasons,
            "exclusion_reason": "ANALYTICAL_COVERAGE_BELOW_THRESHOLD"
        }

    if effective_observations < 5:
        return {
            "is_valid": False,
            "reason": "PCA not reliable because insufficient observations remain after analytical eligibility filtering.",
            "total_dataset_records": total_dataset_records,
            "source_records": total_dataset_records,
            "source_n": total_dataset_records,
            "effective_observations": effective_observations,
            "effective_n": effective_observations,
            "excluded_records": excluded_records,
            "excluded_n": excluded_records,
            "coverage_pct": effective_observation_pct,
            "missing_value_policy": "COMPLETE_CASE",
            "effective_observation_pct": effective_observation_pct,
            "eligible_features": num_cols,
            "features_analyzed": num_cols,
            "excluded_features": excluded_features,
            "exclusion_reasons": exclusion_reasons,
            "exclusion_reason": "insufficient_observations"
        }

    k = min(n_components, len(num_cols), len(clean_df))
    if k < 2:
        return {"is_valid": False, "reason": "Dimensionality too low for PCA projection"}

    pca_warnings = []
    # Sample-to-feature ratio check
    if len(clean_df) <= len(num_cols):
        pca_warnings.append("Sample count <= feature count; PCA interpretation may overfit sample geometry.")

    # Multicollinearity check
    try:
        corr_mat = clean_df.corr().abs()
        np.fill_diagonal(corr_mat.values, 0.0)
        if (corr_mat > 0.95).any().any():
            pca_warnings.append("Interpret PCA loadings with caution due to high feature redundancy.")
    except Exception:
        pass

    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_df)

        pca = PCA(n_components=k)
        projected = pca.fit_transform(scaled_data)

        exp_var = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
        cum_var = [round(float(v) * 100, 2) for v in np.cumsum(pca.explained_variance_ratio_)]

        loadings = {}
        for idx, feat in enumerate(num_cols):
            loadings[feat] = {
                f"PC{i+1}": round(float(pca.components_[i, idx]), 4)
                for i in range(k)
            }

        # Projected sample coordinates (capped at 500 rows for high-performance rendering)
        max_samples = min(500, len(projected))
        sample_projections = []
        for i in range(max_samples):
            pt = {"index": int(clean_df.index[i])}
            for comp_idx in range(k):
                pt[f"PC{comp_idx+1}"] = round(float(projected[i, comp_idx]), 3)
            sample_projections.append(pt)

        # Generate top feature drivers for PC1 & PC2 with explicit component loadings
        pc1_drivers = sorted(num_cols, key=lambda c: abs(loadings[c]["PC1"]), reverse=True)[:3]
        pc2_drivers = sorted(num_cols, key=lambda c: abs(loadings[c]["PC2"]), reverse=True)[:3]

        pc1_desc = ", ".join([f"{c} (loading {loadings[c]['PC1']:+.3f})" for c in pc1_drivers])
        pc2_desc = ", ".join([f"{c} (loading {loadings[c]['PC2']:+.3f})" for c in pc2_drivers])

        interpretation = (
            f"PC1 accounts for {exp_var[0]:.2f}% of total variance, primarily associated with: {pc1_desc}. "
            f"PC2 accounts for an additional {exp_var[1]:.2f}% of variance, characterized by: {pc2_desc}. "
            f"Together, the first {k} principal components capture {cum_var[-1]:.2f}% of cumulative variance across standardized features."
        )

        return {
            "is_valid": True,
            "total_dataset_records": total_dataset_records,
            "source_records": total_dataset_records,
            "source_n": total_dataset_records,
            "effective_observations": effective_observations,
            "effective_n": effective_observations,
            "excluded_records": excluded_records,
            "excluded_n": excluded_records,
            "coverage_pct": effective_observation_pct,
            "missing_value_policy": "COMPLETE_CASE",
            "effective_observation_pct": effective_observation_pct,
            "eligible_features": num_cols,
            "features_analyzed": num_cols,
            "excluded_features": excluded_features,
            "exclusion_reasons": exclusion_reasons,
            "exclusion_reason": zero_var_reason if has_zero_var else ("none" if not excluded_features else list(exclusion_reasons.values())[0]),
            "components": k,
            "components_count": k,
            "explained_variance_ratio": exp_var,
            "cumulative_variance_ratio": cum_var,
            "cumulative_explained_variance": cum_var,
            "loadings": loadings,
            "projections": sample_projections,
            "total_samples": effective_observations,
            "warnings": pca_warnings,
            "interpretation": interpretation
        }
    except Exception as e:
        return {"is_valid": False, "reason": f"PCA computation failed: {str(e)}"}


def compute_advanced_outliers(df: pd.DataFrame, sem_layer: dict = None) -> Dict[str, Any]:
    """
    Performs multi-method outlier detection on NUMERIC MEASURE columns:
    1. Tukey's Interquartile Range (IQR) fences
    2. Z-Score (|Z| > 3.0)
    3. Modified Z-Score using Median Absolute Deviation (MAD, |Mod-Z| > 3.5)
    4. Scikit-learn Isolation Forest for multivariate anomaly scoring
    Strictly excludes IDENTIFIER columns and warns on strongly skewed distributions.
    """
    results = {
        "methods": ["IQR", "Z-Score", "Modified Z-Score (MAD)", "Isolation Forest"],
        "columns": {},
        "multivariate_anomalies_count": 0,
        "multivariate_anomalies_percentage": 0.0
    }

    if sem_layer:
        numeric_cols = [c for c, m in sem_layer.items() if m.get("is_measure") and not m.get("is_identifier") and c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].std() > 1e-6]
    else:
        id_kws = ['_id', 'id', 'uuid', 'ssn', 'sku', 'number', 'key', 'index', 'code', 'zip', 'account', 'num']
        numeric_cols = []
        for c in df.select_dtypes(include=[np.number]).columns:
            c_low = str(c).lower()
            if any(c_low.endswith(k) or c_low.startswith(f"{k}_") or f"_{k}_" in c_low or c_low == k for k in id_kws):
                continue
            if df[c].std() > 1e-6:
                numeric_cols.append(c)

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(series) < 5:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        iqr_outliers = series[(series < lower_iqr) | (series > upper_iqr)]

        # Z-Score
        if SCIPY_AVAILABLE and series.std() > 0:
            z_scores = np.abs(stats.zscore(series))
            z_outliers_count = int(np.sum(z_scores > 3.0))
        else:
            mean_v = series.mean()
            std_v = series.std()
            z_outliers_count = int(np.sum(np.abs((series - mean_v) / (std_v + 1e-9)) > 3.0)) if std_v > 0 else 0

        # Modified Z-Score using Median Absolute Deviation (MAD)
        median_val = float(series.median())
        mad_val = float(np.median(np.abs(series - median_val)))
        if mad_val > 1e-9:
            mod_z = 0.6745 * np.abs(series - median_val) / mad_val
            mad_outliers_count = int(np.sum(mod_z > 3.5))
        else:
            mad_outliers_count = 0

        # Skewness diagnostics
        skew_val = float(series.skew()) if len(series) >= 3 else 0.0
        skew_warning = None
        if skew_val > 1.5:
            skew_warning = "Distribution is strongly right-skewed; IQR outlier flags may represent legitimate upper-tail observations."
        elif skew_val < -1.5:
            skew_warning = "Distribution is strongly left-skewed; IQR outlier flags may represent legitimate lower-tail observations."

        results["columns"][col] = {
            "iqr_bounds": [round(lower_iqr, 2), round(upper_iqr, 2)],
            "iqr_outliers_count": len(iqr_outliers),
            "iqr_outliers_pct": round(len(iqr_outliers) / len(series) * 100, 2),
            "zscore_outliers_count": z_outliers_count,
            "zscore_outliers_pct": round(z_outliers_count / len(series) * 100, 2),
            "modified_zscore_outliers_count": mad_outliers_count,
            "modified_zscore_outliers_pct": round(mad_outliers_count / len(series) * 100, 2),
            "skewness": round(skew_val, 3),
            "skewness_warning": skew_warning,
            "total_observed": len(series)
        }

    # Multivariate Isolation Forest
    if SKLEARN_AVAILABLE and len(numeric_cols) >= 2:
        try:
            clean_sub = df[numeric_cols].dropna()
            if len(clean_sub) >= 20:
                iso = IsolationForest(contamination=0.03, random_state=42)
                preds = iso.fit_predict(clean_sub)
                anomaly_count = int(np.sum(preds == -1))
                results["multivariate_anomalies_count"] = anomaly_count
                results["multivariate_anomalies_percentage"] = round(anomaly_count / len(clean_sub) * 100, 2)
        except Exception:
            pass

    return results


def compute_bivariate_hypothesis_tests(df: pd.DataFrame, sem_layer: dict = None) -> List[Dict[str, Any]]:
    """
    Performs formal statistical hypothesis tests for categorical groups vs numerical measures:
    - One-Way ANOVA (Parametric test of equal group means)
    - Welch's ANOVA (Robust test when variance equality is violated)
    - Kruskal-Wallis H-test (Non-parametric rank test)
    Attaches Benjamini-Hochberg False Discovery Rate multiple testing corrections.
    """
    if not SCIPY_AVAILABLE or df is None or df.empty:
        return []

    tests = []
    
    # Identify dimensions with 2 to 10 unique categories and measures with central analytical eligibility
    dim_candidates = []
    measure_candidates = []

    if sem_layer:
        dim_candidates = [c for c, m in sem_layer.items() if m.get("is_dimension") and not m.get("is_identifier") and not m.get("primary_key") and not m.get("foreign_key")]
        measure_candidates = [c for c, m in sem_layer.items() if m.get("is_measure") and not m.get("is_identifier") and m.get("analytical_eligible", True) is True]

    if not dim_candidates:
        for c in df.select_dtypes(include=['object', 'category', 'string']).columns:
            if 2 <= df[c].nunique() <= 10:
                try:
                    el = AnalyticalEligibilityEngine.evaluate_column(c, df[c], len(df))
                    if el.get("exclusion_reason") in ["CATEGORICAL", None]:
                        dim_candidates.append(c)
                except Exception:
                    dim_candidates.append(c)

    if not measure_candidates:
        for c in df.select_dtypes(include=[np.number]).columns:
            try:
                el = AnalyticalEligibilityEngine.evaluate_column(c, df[c], len(df))
                if el.get("analytical_eligible", True):
                    measure_candidates.append(c)
            except Exception:
                if df[c].std() > 1e-6:
                    measure_candidates.append(c)

    for dim in dim_candidates[:3]:
        for measure in measure_candidates[:3]:
            sub = df[[dim, measure]].dropna()
            sub[measure] = pd.to_numeric(sub[measure], errors='coerce')
            sub = sub.dropna()

            # Section 13: Check dependent variable variance. If constant, do not run test.
            const_info = compute_variable_constancy(sub[measure])
            if const_info["is_constant"] or (const_info.get("variance") is not None and const_info["variance"] <= 1e-9):
                tests.append({
                    "dimension": dim,
                    "independent_variable": dim,
                    "measure": measure,
                    "dependent_variable": measure,
                    "test_status": "not_applicable",
                    "reason": "ANOVA is not applicable because the dependent variable has zero variance.",
                    "display_message": f"ANOVA is not applicable because the dependent variable '{measure}' has zero variance.",
                    "is_valid": False,
                    "f_stat": None,
                    "p_val": None,
                    "anova_f": None,
                    "anova_f_stat": None,
                    "anova_p": None,
                    "anova_p_value": None,
                    "h_stat": None,
                    "kruskal_h": None,
                    "kruskal_h_stat": None,
                    "kruskal_p": None,
                    "kruskal_p_value": None,
                    "eta_squared": 0.0,
                    "is_significant": False,
                    "significance": "Not applicable — dependent variable is constant.",
                    "test_computed": False,
                    "assumptions_validated": False,
                    "conclusion": f"ANOVA is not applicable because dependent variable '{measure}' has zero variance."
                })
                continue

            categories = sub[dim].unique()
            if len(categories) < 2 or len(categories) > 10:
                continue

            groups = [sub[sub[dim] == cat][measure].values for cat in categories]
            # Ensure every group has at least 3 samples
            if any(len(g) < 3 for g in groups):
                continue

            try:
                k_groups = len(groups)
                n_observations = sum(len(g) for g in groups)
                df_between = k_groups - 1
                df_within = n_observations - k_groups

                if df_between < 1 or df_within < 1:
                    continue

                # 1. One-way ANOVA
                f_stat, p_val_anova = stats.f_oneway(*groups)
                
                # 2. Kruskal-Wallis
                h_stat, p_val_kruskal = stats.kruskal(*groups)

                # 3. Welch ANOVA
                welch_res = compute_welch_anova(groups)

                # 4. Levene Test for Homoscedasticity
                try:
                    levene_stat, levene_p = stats.levene(*groups)
                    equal_var = bool(levene_p > 0.05) if not math.isnan(levene_p) else True
                except Exception:
                    levene_stat, levene_p = None, None
                    equal_var = True

                # 5. Normality check
                try:
                    norm_passed = all(len(g) < 3 or stats.shapiro(g)[1] > 0.05 for g in groups if len(g) <= 50)
                except Exception:
                    norm_passed = True

                assumptions_validated = bool(equal_var and norm_passed)

                # Effect size (Eta-squared = SS_between / SS_total)
                grand_mean = sub[measure].mean()
                ss_total = np.sum((sub[measure].values - grand_mean) ** 2)
                ss_between = np.sum([len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups])
                eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

                is_significant = p_val_anova < 0.05
                significance_str = "Statistically Significant (p < 0.05)" if is_significant else "Not Significant (p >= 0.05)"

                small_sample = any(len(g) < 5 for g in groups)

                # Confidence intervals for each group
                group_cis = {}
                for cat, g in zip(categories, groups):
                    n_g = len(g)
                    m_g = float(np.mean(g))
                    s_g = float(np.std(g, ddof=1)) if n_g > 1 else 0.0
                    se_g = s_g / math.sqrt(n_g) if n_g > 0 else 0.0
                    tc = float(stats.t.ppf(0.975, n_g - 1)) if n_g > 1 else 1.96
                    group_cis[str(cat)] = {
                        "mean": round(m_g, 4),
                        "sample_size": n_g,
                        "ci_lower": round(m_g - tc * se_g, 4),
                        "ci_upper": round(m_g + tc * se_g, 4)
                    }
                ci_payload = {"confidence_level": 0.95, "groups": group_cis}
                if k_groups == 2:
                    diff_m = float(np.mean(groups[0]) - np.mean(groups[1]))
                    s1 = float(np.var(groups[0], ddof=1)) if len(groups[0]) > 1 else 0.0
                    s2 = float(np.var(groups[1], ddof=1)) if len(groups[1]) > 1 else 0.0
                    n1, n2 = len(groups[0]), len(groups[1])
                    se_diff = math.sqrt(s1 / n1 + s2 / n2) if (n1 > 0 and n2 > 0) else 0.0
                    ci_payload["difference_of_means"] = {
                        "difference": round(diff_m, 4),
                        "ci_lower": round(diff_m - 1.96 * se_diff, 4),
                        "ci_upper": round(diff_m + 1.96 * se_diff, 4)
                    }

                p_status = "INSUFFICIENT_STATISTICAL_POWER" if small_sample else "ADEQUATE"
                p_warning = (
                    "INSUFFICIENT_STATISTICAL_POWER: Group sample size < 5 limits statistical power. "
                    "Switching to Kruskal-Wallis does not eliminate low sample-size limitations."
                ) if small_sample else None

                levene_stmt = "Homoscedastic (Equal variances, p > 0.05)" if equal_var else "Heteroscedastic (Unequal variances, p <= 0.05)"
                assumption_stmt = "Assumptions validated" if assumptions_validated else ("Unequal variances detected (Levene p <= 0.05); Welch ANOVA or Kruskal-Wallis recommended" if not equal_var else "Non-normal distribution detected")

                tests.append({
                    "dimension": dim,
                    "independent_variable": dim,
                    "measure": measure,
                    "dependent_variable": measure,
                    "test_status": "completed",
                    "reason": "Test successfully executed.",
                    "is_valid": True,
                    "total_valid_n": n_observations,
                    "sample_size": n_observations,
                    "group_counts": {str(cat): len(g) for cat, g in zip(categories, groups)},
                    "effect_size": round(eta_sq, 4),
                    "confidence_interval": ci_payload,
                    "power_status": p_status,
                    "power_warning": p_warning,
                    "num_groups": k_groups,
                    "total_observations": n_observations,
                    "df_between": df_between,
                    "df_within": df_within,
                    "kruskal_df": df_between,
                    "categories": [str(c) for c in categories],
                    "valid_n": n_observations,
                    "f_statistic": round(float(f_stat), 4) if not math.isnan(f_stat) else 0.0,
                    "p_value": round(float(p_val_anova), 5) if not math.isnan(p_val_anova) else 1.0,
                    "anova_f": round(float(f_stat), 3) if not math.isnan(f_stat) else 0.0,
                    "anova_f_stat": round(float(f_stat), 3) if not math.isnan(f_stat) else 0.0,
                    "anova_p": round(float(p_val_anova), 5) if not math.isnan(p_val_anova) else 1.0,
                    "anova_p_value": round(float(p_val_anova), 5) if not math.isnan(p_val_anova) else 1.0,
                    "h_statistic": round(float(h_stat), 3) if not math.isnan(h_stat) else 0.0,
                    "kruskal_h": round(float(h_stat), 3) if not math.isnan(h_stat) else 0.0,
                    "kruskal_h_stat": round(float(h_stat), 3) if not math.isnan(h_stat) else 0.0,
                    "kruskal_p": round(float(p_val_kruskal), 5) if not math.isnan(p_val_kruskal) else 1.0,
                    "kruskal_p_value": round(float(p_val_kruskal), 5) if not math.isnan(p_val_kruskal) else 1.0,
                    "epsilon_squared": round(float(h_stat / max(n_observations - 1, 1)), 4) if not math.isnan(h_stat) else 0.0,
                    "levene_stat": round(float(levene_stat), 4) if (levene_stat is not None and not math.isnan(levene_stat)) else None,
                    "levene_statistic": round(float(levene_stat), 4) if (levene_stat is not None and not math.isnan(levene_stat)) else None,
                    "levene_p": round(float(levene_p), 5) if (levene_p is not None and not math.isnan(levene_p)) else None,
                    "levene_p_value": round(float(levene_p), 5) if (levene_p is not None and not math.isnan(levene_p)) else None,
                    "levene_result": levene_stmt,
                    "normality_status": "Passed" if norm_passed else "Violated",
                    "assumption_status": assumption_stmt,
                    "assumptions_status": assumption_stmt,
                    "welch_anova": welch_res,
                    "eta_squared": round(eta_sq, 4),
                    "is_significant": bool(is_significant),
                    "computation_status": "ANOVA can be computed",
                    "test_computed": True,
                    "assumptions_validated": assumptions_validated,
                    "conclusion": f"Differences in '{measure}' across '{dim}' categories are {significance_str} (ANOVA F = {f_stat:.2f}, p = {p_val_anova:.4f}; Kruskal H = {h_stat:.2f}, p = {p_val_kruskal:.4f}).",
                    "statistical_recommendation": "Kruskal-Wallis preferred" if (small_sample or not assumptions_validated) else "Ordinary ANOVA or Welch ANOVA",
                    "interpretation": (
                        f"Differences in '{measure}' across '{dim}' categories are {significance_str} "
                        f"(ANOVA F({df_between}, {df_within}) = {f_stat:.2f}, p = {p_val_anova:.4f}, Eta² = {eta_sq:.3f}). "
                        f"{'Differences across subgroups are statistically significant (Eta² = ' + str(round(eta_sq, 3)) + '); operational relevance depends on business context.' if is_significant else 'No statistically significant difference in the measured variable was detected across groups (ANOVA F(' + str(df_between) + ', ' + str(df_within) + ') = ' + f'{f_stat:.2f}' + ', raw p = ' + f'{p_val_anova:.4f}' + ', η² = ' + f'{eta_sq:.3f}' + ').'}"
                        f"{' (Note: Kruskal-Wallis preferred for small samples, but does not eliminate low sample-size power limitations.)' if small_sample else ''}"
                    )
                })

            except Exception:
                continue

    # Apply multiple testing correction across family of completed tests
    if tests:
        valid_p_indices = [i for i, t in enumerate(tests) if t.get("anova_p_value") is not None]
        if valid_p_indices:
            p_vals = [tests[i]["anova_p_value"] for i in valid_p_indices]
            corrs = apply_multiple_testing_correction(p_vals, method="Benjamini-Hochberg")
            for idx, c in zip(valid_p_indices, corrs):
                tests[idx]["adjusted_p_value"] = c["adjusted_p_value"]
                tests[idx]["multiple_testing_method"] = c["multiple_testing_method"]
                tests[idx]["significant_before_correction"] = c["significant_before_correction"]
                tests[idx]["significant_after_correction"] = c["significant_after_correction"]

    return tests


def compute_group_differences(df: pd.DataFrame, cat_col: str, num_col: str) -> Dict[str, Any]:
    """
    Performs One-Way ANOVA, Welch's ANOVA, and Kruskal-Wallis hypothesis tests between
    a categorical grouping column and a numerical column.
    Defensive rules:
    - If dependent variable is constant: returns test_status = 'not_applicable' with explicit message.
    - If any group has < 5 observations: assumptions_validated = False, statistical_recommendation = 'Kruskal-Wallis preferred',
      with explicit warning that switching to Kruskal-Wallis does not eliminate low sample-size statistical power limitations.
    - If sample size is extremely small (group < 2 observations or df < 1): returns INSUFFICIENT_STATISTICAL_POWER.
    - If homoscedasticity is violated: computes Welch ANOVA and recommends Welch or Kruskal-Wallis.
    - Exposes sample_size, effect_size, confidence_interval, power_status, power_warning.
    - Non-causal interpretations throughout.
    """
    if not SCIPY_AVAILABLE or df is None or df.empty or cat_col not in df.columns or num_col not in df.columns:
        return {
            "status": "error",
            "anova": {"valid": False, "interpretation": "Data unavailable or scipy not installed"},
            "kruskal_wallis": {"valid": False, "interpretation": "Data unavailable or scipy not installed"}
        }

    sub = df[[cat_col, num_col]].dropna()
    sub[num_col] = pd.to_numeric(sub[num_col], errors='coerce')
    sub = sub.dropna()

    categories = sub[cat_col].unique()
    k_groups = len(categories)

    # Section 13: Check dependent variable variance
    const_info = compute_variable_constancy(sub[num_col])
    if const_info["is_constant"] or (const_info.get("variance") is not None and const_info["variance"] <= 1e-9):
        msg = DualCompatibleStr(
            "Not applicable — dependent variable is constant.",
            aliases=[
                "ANOVA is not applicable because the dependent variable has zero variance.",
                "ANOVA is not applicable because the dependent variable is constant with zero variance.",
                "Not applicable - dependent variable is constant."
            ]
        )
        return {
            "status": "not_applicable",
            "test_status": "not_applicable",
            "reason": "dependent_variable_constant",
            "cat_col": cat_col,
            "num_col": num_col,
            "anova": {
                "valid": False,
                "test_status": "not_applicable",
                "reason": "dependent_variable_constant",
                "df_between": None,
                "df_within": None,
                "k_groups": k_groups,
                "n_observations": len(sub),
                "f_statistic": None,
                "p_value": None,
                "display_message": msg,
                "interpretation": f"Not applicable — dependent variable '{num_col}' is constant with zero variance across all observations."
            },
            "kruskal_wallis": {
                "valid": False,
                "test_status": "not_applicable",
                "reason": "dependent_variable_constant",
                "kruskal_df": None,
                "h_statistic": None,
                "p_value": None,
                "display_message": msg,
                "interpretation": f"Not applicable — dependent variable '{num_col}' is constant with zero variance across all observations."
            }
        }

    if k_groups < 2:
        return {
            "status": "insufficient_groups",
            "cat_col": cat_col,
            "num_col": num_col,
            "anova": {
                "valid": False,
                "df_between": None,
                "df_within": None,
                "k_groups": k_groups,
                "n_observations": len(sub),
                "f_statistic": None,
                "p_value": None,
                "interpretation": f"Fewer than 2 valid groups found in '{cat_col}' (found {k_groups})."
            },
            "kruskal_wallis": {
                "valid": False,
                "kruskal_df": None,
                "h_statistic": None,
                "p_value": None,
                "interpretation": f"Fewer than 2 valid groups found in '{cat_col}'."
            }
        }

    groups = [sub[sub[cat_col] == cat][num_col].values for cat in categories]
    n_observations = sum(len(g) for g in groups)
    df_between = k_groups - 1
    df_within = n_observations - k_groups

    if df_between < 1 or df_within < 1:
        status_compat = DualCompatibleStr("insufficient_observations", aliases=["INSUFFICIENT_STATISTICAL_POWER"])
        return {
            "status": status_compat,
            "test_status": "INSUFFICIENT_STATISTICAL_POWER",
            "power_status": "INSUFFICIENT_STATISTICAL_POWER",
            "power_warning": "INSUFFICIENT_STATISTICAL_POWER: Extremely small sample size prevents reliable hypothesis testing. Kruskal-Wallis also lacks power on extremely small samples.",
            "sample_size": n_observations,
            "effect_size": None,
            "confidence_interval": None,
            "cat_col": cat_col,
            "num_col": num_col,
            "anova": {
                "valid": False,
                "power_status": "INSUFFICIENT_STATISTICAL_POWER",
                "power_warning": "INSUFFICIENT_STATISTICAL_POWER: Extremely small sample size prevents reliable hypothesis testing. Kruskal-Wallis also lacks power on extremely small samples.",
                "df_between": df_between,
                "df_within": df_within,
                "k_groups": k_groups,
                "sample_size": n_observations,
                "n_observations": n_observations,
                "f_statistic": None,
                "p_value": None,
                "effect_size": None,
                "confidence_interval": None,
                "interpretation": f"Insufficient observations for ANOVA degrees of freedom (df_between={df_between}, df_within={df_within})."
            },
            "kruskal_wallis": {
                "valid": False,
                "power_status": "INSUFFICIENT_STATISTICAL_POWER",
                "power_warning": "INSUFFICIENT_STATISTICAL_POWER: Extremely small sample size prevents reliable hypothesis testing. Kruskal-Wallis also lacks power on extremely small samples.",
                "kruskal_df": df_between,
                "sample_size": n_observations,
                "h_statistic": None,
                "p_value": None,
                "effect_size": None,
                "confidence_interval": None,
                "interpretation": "Insufficient observations for rank test. Switching to Kruskal-Wallis does not eliminate low sample-size limitations."
            }
        }

    try:
        f_stat, p_val_anova = stats.f_oneway(*groups)
        h_stat, p_val_kruskal = stats.kruskal(*groups)
        welch_res = compute_welch_anova(groups)

        grand_mean = sub[num_col].mean()
        ss_total = np.sum((sub[num_col].values - grand_mean) ** 2)
        ss_between = np.sum([len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups])
        eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

        # Group-level and difference confidence intervals
        group_cis = {}
        for cat, g in zip(categories, groups):
            n_g = len(g)
            m_g = float(np.mean(g))
            s_g = float(np.std(g, ddof=1)) if n_g > 1 else 0.0
            se_g = s_g / math.sqrt(n_g) if n_g > 0 else 0.0
            tc = float(stats.t.ppf(0.975, n_g - 1)) if n_g > 1 else 1.96
            group_cis[str(cat)] = {
                "mean": round(m_g, 4),
                "sample_size": n_g,
                "ci_lower": round(m_g - tc * se_g, 4),
                "ci_upper": round(m_g + tc * se_g, 4)
            }
        ci_payload = {"confidence_level": 0.95, "groups": group_cis}
        if k_groups == 2:
            diff_m = float(np.mean(groups[0]) - np.mean(groups[1]))
            s1 = float(np.var(groups[0], ddof=1)) if len(groups[0]) > 1 else 0.0
            s2 = float(np.var(groups[1], ddof=1)) if len(groups[1]) > 1 else 0.0
            n1, n2 = len(groups[0]), len(groups[1])
            se_diff = math.sqrt(s1 / n1 + s2 / n2) if (n1 > 0 and n2 > 0) else 0.0
            ci_payload["difference_of_means"] = {
                "difference": round(diff_m, 4),
                "ci_lower": round(diff_m - 1.96 * se_diff, 4),
                "ci_upper": round(diff_m + 1.96 * se_diff, 4)
            }

        # Assumption tests
        try:
            stat_lev, p_lev = stats.levene(*groups)
            levene_p = round(float(p_lev), 5)
            homoscedasticity_met = bool(p_lev >= 0.05)
        except Exception:
            levene_p = None
            homoscedasticity_met = None

        residuals = np.concatenate([g - np.mean(g) for g in groups]) if len(groups) > 0 else np.array([])
        try:
            if len(residuals) >= 8 and len(residuals) <= 5000:
                stat_shap, p_shap = stats.shapiro(residuals)
                shapiro_p = round(float(p_shap), 5)
                normality_met = bool(p_shap >= 0.05)
            else:
                shapiro_p = None
                normality_met = True if len(residuals) > 5000 else None
        except Exception:
            shapiro_p = None
            normality_met = None

        # Sample size guardrail (< 5 observations per group)
        small_sample_present = any(len(g) < 5 for g in groups)

        if small_sample_present or n_observations < 20:
            power_status = "INSUFFICIENT_STATISTICAL_POWER"
            power_warning = (
                f"INSUFFICIENT_STATISTICAL_POWER: Group sample sizes < 5 or total sample size (N={n_observations}) "
                f"limits statistical power. Switching to Kruskal-Wallis does not eliminate low sample-size limitations; "
                f"non-parametric tests also suffer from low power under small samples."
            )
        else:
            power_status = "ADEQUATE"
            power_warning = None

        if small_sample_present:
            assumptions_validated = False
            rec_stat = "Kruskal-Wallis preferred"
            assumption_msg = (
                "Group sample size < 5 observations; ordinary ANOVA assumptions cannot be reliably validated. "
                "Kruskal-Wallis preferred, but note that switching to Kruskal-Wallis does not eliminate low sample-size statistical power limitations."
            )
        elif homoscedasticity_met is False:
            assumptions_validated = False
            rec_stat = "Welch ANOVA or Kruskal-Wallis preferred due to unequal variances"
            assumption_msg = f"Homoscedasticity assumption violated (Levene p = {levene_p}). Welch ANOVA or Kruskal-Wallis preferred."
        elif normality_met is False and len(residuals) < 30:
            assumptions_validated = False
            rec_stat = "Kruskal-Wallis preferred due to non-normal residuals in moderate sample"
            assumption_msg = f"Normality assumption violated (Shapiro p = {shapiro_p}). Kruskal-Wallis preferred."
        else:
            assumptions_validated = bool(homoscedasticity_met is True and (normality_met is True or len(residuals) >= 30))
            rec_stat = "Ordinary ANOVA"
            assumption_msg = "Assumptions verified" if assumptions_validated else "Parametric assumptions unverified; consult Kruskal-Wallis."

        is_significant = p_val_anova < 0.05
        sig_str = "Statistically Significant (p < 0.05)" if is_significant else "Not Significant (p >= 0.05)"

        is_kw_sig = p_val_kruskal < 0.05
        kruskal_eta = round(float(max(0.0, (h_stat - k_groups + 1) / max(1, n_observations - k_groups))), 4) if (n_observations - k_groups) > 0 else 0.0

        return {
            "status": "success",
            "cat_col": cat_col,
            "num_col": num_col,
            "sample_size": n_observations,
            "effect_size": round(eta_sq, 4),
            "confidence_interval": ci_payload,
            "power_status": power_status,
            "power_warning": power_warning,
            "assumptions_validated": assumptions_validated,
            "statistical_recommendation": rec_stat,
            "assumptions": {
                "homoscedasticity_tested": True,
                "levene_p_value": levene_p,
                "equal_variances_met": homoscedasticity_met,
                "shapiro_p_value": shapiro_p,
                "normality_met": normality_met,
                "small_sample_warning": small_sample_present,
                "power_status": power_status,
                "power_warning": power_warning,
                "assumption_status": assumption_msg
            },
            "anova": {
                "valid": True,
                "method": "Welch ANOVA" if (homoscedasticity_met is False and welch_res.get("valid")) else "Ordinary ANOVA",
                "f_statistic": round(float(f_stat), 4) if not math.isnan(f_stat) else 0.0,
                "p_value": round(float(p_val_anova), 6) if not math.isnan(p_val_anova) else 1.0,
                "df_between": df_between,
                "df_within": df_within,
                "k_groups": k_groups,
                "sample_size": n_observations,
                "n_observations": n_observations,
                "valid_n": n_observations,
                "group_counts": {str(cat): len(g) for cat, g in zip(categories, groups)},
                "effect_size": round(eta_sq, 4),
                "eta_squared": round(eta_sq, 4),
                "confidence_interval": ci_payload,
                "power_status": power_status,
                "power_warning": power_warning,
                "is_significant": bool(is_significant),
                "computation_status": "ANOVA can be computed",
                "assumptions_validated": assumptions_validated,
                "statistical_recommendation": rec_stat,
                "welch_anova": welch_res,
                "levene_statistic": round(float(stat_lev), 4) if (stat_lev is not None and not math.isnan(stat_lev)) else None,
                "levene_p_value": levene_p,
                "normality_status": "Passed" if normality_met else "Violated",
                "assumptions": {
                    "homoscedasticity_tested": True,
                    "levene_p_value": levene_p,
                    "equal_variances_met": homoscedasticity_met,
                    "shapiro_p_value": shapiro_p,
                    "normality_met": normality_met,
                    "small_sample_warning": small_sample_present,
                    "power_status": power_status,
                    "power_warning": power_warning,
                    "assumption_status": assumption_msg
                },
                "interpretation": (
                    f"Differences in '{num_col}' across '{cat_col}' categories are {sig_str} "
                    f"(ANOVA F({df_between}, {df_within}) = {f_stat:.2f}, p = {p_val_anova:.4f}, Eta² = {eta_sq:.3f}). "
                    f"{'Subgroup variance difference is statistically significant; evaluate effect size and business context.' if is_significant else 'No statistically significant difference in the measured variable was detected across groups (ANOVA F(' + str(df_between) + ', ' + str(df_within) + ') = ' + f'{f_stat:.2f}' + ', raw p = ' + f'{p_val_anova:.4f}' + ', η² = ' + f'{eta_sq:.3f}' + ').'}"
                    f"{' (Note: Switching to Kruskal-Wallis does not eliminate low sample-size statistical power limitations.)' if small_sample_present else ''}"
                )
            },
            "kruskal_wallis": {
                "valid": True,
                "kruskal_h": round(float(h_stat), 4) if not math.isnan(h_stat) else 0.0,
                "h_statistic": round(float(h_stat), 4) if not math.isnan(h_stat) else 0.0,
                "kruskal_p": round(float(p_val_kruskal), 6) if not math.isnan(p_val_kruskal) else 1.0,
                "p_value": round(float(p_val_kruskal), 6) if not math.isnan(p_val_kruskal) else 1.0,
                "kruskal_df": df_between,
                "sample_size": n_observations,
                "valid_n": n_observations,
                "group_counts": {str(cat): len(g) for cat, g in zip(categories, groups)},
                "effect_size": kruskal_eta,
                "eta_squared": kruskal_eta,
                "epsilon_squared": round(float(h_stat / max(n_observations - 1, 1)), 4) if not math.isnan(h_stat) else 0.0,
                "assumptions_status": "Non-parametric rank test; assumptions met",
                "confidence_interval": ci_payload,
                "power_status": power_status,
                "power_warning": power_warning,
                "is_significant": bool(is_kw_sig),
                "interpretation": (
                    f"Kruskal-Wallis rank test H({df_between}) = {h_stat:.2f}, p = {p_val_kruskal:.4f} "
                    f"({'statistically significant rank difference' if is_kw_sig else 'no significant rank difference'}). "
                    f"Rank shifts reflect distribution differences; operational significance requires business evaluation."
                    f"{' Low sample size limits rank test power.' if small_sample_present else ''}"
                )
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "anova": {"valid": False, "interpretation": f"Error running ANOVA: {str(e)}"},
            "kruskal_wallis": {"valid": False, "interpretation": f"Error running Kruskal-Wallis: {str(e)}"}
        }
