import pandas as pd
import numpy as np
import math
from engine.profiler import infer_column_semantic_metadata, parse_flexible_datetime
from engine.cleaner import coerce_numeric_columns
from engine.analytical_truth import (
    compute_variable_constancy,
    generate_constant_variable_statement,
    is_chart_suitable,
    safe_correlation
)
from engine.governed_engines import calculate_canonical_dataset_health

def detect_domain(df: pd.DataFrame) -> str:
    """
    Dataset-Agnostic Profile Description.
    Summarizes dataset structure without hardcoding domain assumptions.
    """
    n_rows, n_cols = df.shape
    return f"Structured Dataset Profile ({n_rows:,} records, {n_cols} variables)"

def validate_chart_spec(chart_type: str, df: pd.DataFrame, params: dict) -> dict:
    """
    Deterministic Chart Selection & Validation Engine:
    Validates:
    - required column types exist
    - sufficient non-null observations exist (>= 2)
    - sufficient category cardinality exists (>= 2)
    - aggregation is mathematically meaningful (sum for additive volume, mean for non-additive rates/ratings)
    - non-negative requirements for compositional charts (Pie/Donut)
    - standard Tukey fences for box plots
    - adaptive temporal binning (daily vs monthly) for line charts
    - the resulting data is non-empty and finite
    Returns spec with 'is_valid': True/False, 'fallback_message', and computed 'data'.
    """
    spec = {
        "chart_type": chart_type,
        "is_valid": False,
        "fallback_message": "Insufficient data for this visualization",
        "data": []
    }

    try:
        if df is None or df.empty or len(df) < 2:
            return spec

        if chart_type in ["LINE", "AREA"]:
            x_col = params.get("x_axis")
            y_col = params.get("y_axis")
            agg_method = params.get("agg", "sum")
            if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
                return spec
            
            s_x = parse_flexible_datetime(df[x_col], errors='coerce')
            s_y = pd.to_numeric(df[y_col], errors='coerce')
            valid_df = pd.DataFrame({"x": s_x, "y": s_y}).dropna()
            
            if len(valid_df) >= 2 and valid_df["y"].apply(lambda v: math.isfinite(v)).all():
                date_span = (valid_df["x"].max() - valid_df["x"].min()).days
                if date_span <= 60:
                    valid_df["period"] = valid_df["x"].dt.strftime('%Y-%m-%d')
                else:
                    valid_df["period"] = valid_df["x"].dt.strftime('%Y-%m')

                if agg_method == "mean":
                    grouped = valid_df.groupby("period")["y"].mean()
                else:
                    grouped = valid_df.groupby("period")["y"].sum()

                data_points = [{"x": str(k), "y": round(float(v), 2), "period": str(k), "value": round(float(v), 2)} for k, v in grouped.items() if math.isfinite(v)]
                if len(data_points) >= 2:
                    spec["data"] = data_points
                    spec["is_valid"] = True
                    spec["fallback_message"] = None

        elif chart_type in ["BAR", "HORIZONTAL_BAR"]:
            dim_col = params.get("dim_col") or params.get("x_axis") or params.get("y_axis")
            m_col = params.get("m_col") or params.get("y_axis") or params.get("x_axis")
            agg_method = params.get("agg", "sum")
            if not dim_col or not m_col or dim_col not in df.columns or m_col not in df.columns:
                return spec
            
            valid_df = df[[dim_col, m_col]].dropna()
            valid_df[m_col] = pd.to_numeric(valid_df[m_col], errors='coerce')
            valid_df = valid_df.dropna()

            if len(valid_df) >= 2 and valid_df[dim_col].nunique() >= 2:
                if agg_method == "mean":
                    grouped = valid_df.groupby(dim_col)[m_col].mean().sort_values(ascending=False).head(10)
                else:
                    grouped = valid_df.groupby(dim_col)[m_col].sum().sort_values(ascending=False).head(10)

                data_points = [{"category": str(k), "value": round(float(v), 2), "x": str(k), "y": round(float(v), 2)} for k, v in grouped.items() if math.isfinite(v)]
                if len(data_points) >= 2:
                    spec["data"] = data_points
                    spec["is_valid"] = True
                    spec["fallback_message"] = None

        elif chart_type in ["STACKED_BAR", "STACKED_BAR_100", "GROUPED_BAR"]:
            dim1 = params.get("dim1") or params.get("x_axis")
            dim2 = params.get("dim2")
            m_col = params.get("measure") or params.get("m_col")
            if not dim1 or not dim2 or not m_col or dim1 not in df.columns or dim2 not in df.columns or m_col not in df.columns:
                return spec
            
            valid_df = df[[dim1, dim2, m_col]].dropna()
            valid_df[m_col] = pd.to_numeric(valid_df[m_col], errors='coerce')
            valid_df = valid_df.dropna()

            if valid_df[dim1].nunique() >= 2 and valid_df[dim2].nunique() >= 2:
                grouped = valid_df.groupby([dim1, dim2])[m_col].sum().head(20)
                data_points = [{"x": str(cat1), "group": str(cat2), "value": round(float(v), 2)} for (cat1, cat2), v in grouped.items() if math.isfinite(v)]
                if len(data_points) >= 2:
                    spec["data"] = data_points
                    spec["is_valid"] = True
                    spec["fallback_message"] = None

        elif chart_type in ["DONUT", "PIE"]:
            dim_col = params.get("dim_col") or params.get("x_axis")
            m_col = params.get("m_col") or params.get("y_axis")
            if not dim_col or dim_col not in df.columns:
                return spec
            
            nunique = df[dim_col].nunique()
            if 2 <= nunique <= 6:
                if m_col and m_col in df.columns:
                    valid_df = df[[dim_col, m_col]].dropna()
                    valid_df[m_col] = pd.to_numeric(valid_df[m_col], errors='coerce')
                    grouped = valid_df.dropna().groupby(dim_col)[m_col].sum()
                else:
                    grouped = df[dim_col].value_counts()

                # Compositional pie requires non-negative values and non-zero sum
                if (grouped < 0).any() or grouped.sum() <= 0:
                    return spec

                data_points = [{"category": str(k), "value": round(float(v), 2)} for k, v in grouped.items() if math.isfinite(v)]
                if len(data_points) >= 2:
                    spec["data"] = data_points
                    spec["is_valid"] = True
                    spec["fallback_message"] = None

        elif chart_type == "SCATTER":
            x_col = params.get("x_axis")
            y_col = params.get("y_axis")
            if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
                return spec
            
            s_x = pd.to_numeric(df[x_col], errors='coerce')
            s_y = pd.to_numeric(df[y_col], errors='coerce')
            valid_df = pd.DataFrame({"x": s_x, "y": s_y}).dropna()

            if len(valid_df) >= 3 and valid_df["x"].std() > 0 and valid_df["y"].std() > 0:
                data_points = [{"x": round(float(r["x"]), 2), "y": round(float(r["y"]), 2)} for _, r in valid_df.head(100).iterrows() if math.isfinite(r["x"]) and math.isfinite(r["y"])]
                if len(data_points) >= 3:
                    spec["data"] = data_points
                    spec["is_valid"] = True
                    spec["fallback_message"] = None

        elif chart_type == "CORRELATION_HEATMAP":
            vars_list = params.get("variables", [])
            valid_cols = [c for c in vars_list if c in df.columns]
            if len(valid_cols) >= 2:
                num_df = df[valid_cols].apply(pd.to_numeric, errors='coerce')
                valid_vars = [c for c in valid_cols if num_df[c].notnull().sum() >= 3]
                if len(valid_vars) >= 2:
                    p_matrix = num_df[valid_vars].corr(method='pearson').fillna(0.0)
                    s_matrix = num_df[valid_vars].corr(method='spearman').fillna(0.0)
                    stds = num_df[valid_vars].std()
                    data_points = []
                    for c1 in valid_vars:
                        for c2 in valid_vars:
                            is_const = (stds.get(c1, 0) == 0 or stds.get(c2, 0) == 0)
                            if is_const and c1 != c2:
                                p_val = 0.0
                                s_val = 0.0
                            elif c1 == c2:
                                p_val = 1.0
                                s_val = 1.0
                            else:
                                p_val = round(float(p_matrix.loc[c1, c2]), 3)
                                s_val = round(float(s_matrix.loc[c1, c2]), 3)
                            data_points.append({
                                "x": str(c1),
                                "y": str(c2),
                                "value": p_val,
                                "pearson": p_val,
                                "spearman": s_val,
                                "is_constant": is_const
                            })
                    if len(data_points) >= 4:
                        spec["data"] = data_points
                        spec["is_valid"] = True
                        spec["fallback_message"] = None

        elif chart_type == "HISTOGRAM":
            var_col = params.get("variable")
            if not var_col or var_col not in df.columns:
                return spec
            s = pd.to_numeric(df[var_col], errors='coerce').dropna()
            if s.empty or s.nunique() <= 1 or (len(s) > 1 and s.std() == 0):
                spec["fallback_message"] = f"Not suitable — variable '{var_col}' is constant with zero variance."
                return spec
            if len(s) >= 4 and s.nunique() >= 2:
                num_bins = min(10, max(3, s.nunique()))
                counts, bin_edges = np.histogram(s, bins=num_bins)
                data_points = [
                    {
                        "bin": f"{round(float(b_start), 2)} - {round(float(b_end), 2)}",
                        "count": int(cnt),
                        "value": int(cnt),
                        "x": f"{round(float(b_start), 1)}-{round(float(b_end), 1)}",
                        "y": int(cnt)
                    }
                    for cnt, b_start, b_end in zip(counts, bin_edges[:-1], bin_edges[1:])
                ]
                if len(data_points) >= 2:
                    spec["data"] = data_points
                    spec["is_valid"] = True
                    spec["fallback_message"] = None

        elif chart_type == "BOX_PLOT":
            cat_col = params.get("category")
            m_col = params.get("metric")
            if not cat_col or not m_col or cat_col not in df.columns or m_col not in df.columns:
                return spec
            valid_df = df[[cat_col, m_col]].dropna()
            valid_df[m_col] = pd.to_numeric(valid_df[m_col], errors='coerce')
            valid_df = valid_df.dropna()

            if valid_df.empty or valid_df[m_col].nunique() <= 1 or (len(valid_df) > 1 and valid_df[m_col].std() == 0):
                spec["fallback_message"] = f"Not suitable — metric '{m_col}' has zero variance."
                return spec

            if len(valid_df) >= 4 and valid_df[cat_col].nunique() >= 2:
                box_data = []
                for cat_val, grp in valid_df.groupby(cat_col):
                    vals = grp[m_col].values
                    if len(vals) >= 2:
                        q1, med, q3 = np.percentile(vals, [25, 50, 75])
                        iqr = q3 - q1
                        lower_fence = max(float(vals.min()), q1 - 1.5 * iqr)
                        upper_fence = min(float(vals.max()), q3 + 1.5 * iqr)
                        outliers = [float(x) for x in vals if x < lower_fence or x > upper_fence]
                        box_data.append({
                            "category": str(cat_val),
                            "x": str(cat_val),
                            "min": round(lower_fence, 2),
                            "q1": round(float(q1), 2),
                            "median": round(float(med), 2),
                            "q3": round(float(q3), 2),
                            "max": round(upper_fence, 2),
                            "outliers_count": len(outliers)
                        })
                if len(box_data) >= 2:
                    spec["data"] = box_data
                    spec["is_valid"] = True
                    spec["fallback_message"] = None

        elif chart_type == "VIOLIN":
            cat_col = params.get("category") or params.get("x_axis")
            m_col = params.get("metric") or params.get("y_axis")
            if not cat_col or not m_col or cat_col not in df.columns or m_col not in df.columns:
                return spec
            valid_df = df[[cat_col, m_col]].dropna()
            valid_df[m_col] = pd.to_numeric(valid_df[m_col], errors='coerce')
            valid_df = valid_df.dropna()

            if valid_df.empty or valid_df[m_col].nunique() <= 1 or (len(valid_df) > 1 and valid_df[m_col].std() == 0):
                spec["fallback_message"] = f"Not suitable — metric '{m_col}' has zero variance."
                return spec

            if len(valid_df) >= 4 and valid_df[cat_col].nunique() >= 2:
                spec["data"] = valid_df.head(500).to_dict(orient="records")
                spec["is_valid"] = True
                spec["fallback_message"] = None

        elif chart_type == "WATERFALL":
            data = params.get("data", [])
            if len(data) >= 2:
                spec["data"] = data
                spec["is_valid"] = True
                spec["fallback_message"] = None

    except Exception:
        pass

    return spec

def recommend_visualizations(df: pd.DataFrame, sem_layer: dict) -> list:
    """
    Visualization Recommendation Engine strictly tied to Data Types, Roles, and Statistical Characteristics.
    Excludes Identifier columns from recommendations.
    Dynamically generates appropriate mix of visualizations with non-empty validated data,
    takeaways, and defensible suitability reasons.
    """
    recommendations = []

    # Exclude identifiers and non-aggregatable fields from measure list
    measures = [c for c, m in sem_layer.items() if m.get("is_measure") and not m.get("is_identifier") and m.get("aggregation_eligibility", True)]
    dimensions = [c for c, m in sem_layer.items() if (m.get("is_dimension") or m.get("semantic_role") == "Geographic") and m.get("grouping_eligibility", True) and not m.get("is_identifier")]
    time_cols = [c for c, m in sem_layer.items() if m.get("is_time") or m.get("semantic_role") == "Date/Time"]

    measures_sorted = sorted(measures, key=lambda c: sem_layer[c].get("impact_score", 0), reverse=True)
    dims_sorted = sorted(dimensions, key=lambda c: sem_layer[c].get("impact_score", 0), reverse=True)

    # Separate non-constant measures from constant measures
    non_constant_measures = [m for m in measures_sorted if not compute_variable_constancy(df[m])["is_constant"]]

    chart_counter = 1

    # 1. LINE / AREA CHART: Time + Measure (Adaptive Daily/Monthly & Additive/Mean)
    if time_cols and measures_sorted:
        date_col = time_cols[0]
        m_col = measures_sorted[0]
        is_add = sem_layer.get(m_col, {}).get("is_additive", True) and sem_layer.get(m_col, {}).get("semantic_subtype") not in ["Rating", "Percentage", "Duration"]
        agg_method = "sum" if is_add else "mean"
        params = {"x_axis": date_col, "y_axis": m_col, "agg": agg_method}
        val_res = validate_chart_spec("LINE", df, params)

        if val_res["is_valid"]:
            pts = val_res["data"]
            m_const = compute_variable_constancy(df[m_col])
            metric_label = f"Total {m_col}" if is_add else f"Average {m_col}"

            if m_const["is_constant"]:
                takeaway = generate_constant_variable_statement(m_col, m_const, unit=sem_layer.get(m_col, {}).get("unit", ""), time_col=date_col, df=df)
                desc = f"{m_col} remained constant across temporal axis '{date_col}'."
            else:
                max_pt = max(pts, key=lambda p: p["value"]) if pts else {"x": "N/A", "value": 0}
                min_pt = min(pts, key=lambda p: p["value"]) if pts else {"x": "N/A", "value": 0}
                takeaway = f"Peak {metric_label.lower()} of {max_pt['value']:,} recorded in {max_pt['x']}, while lowest value of {min_pt['value']:,} occurred in {min_pt['x']}."
                desc = f"Progression of {m_col} ({agg_method} aggregation) over temporal axis '{date_col}'."
            
            recommendations.append({
                "id": f"chart_{chart_counter}",
                "chart_type": "LINE",
                "title": f"{m_col} Trend Over Time",
                "description": desc,
                "x_axis": date_col,
                "y_axis": m_col,
                "is_valid": True,
                "fallback_message": None,
                "data": val_res["data"],
                "takeaway": takeaway,
                "suitability_reason": f"Temporal field '{date_col}' combined with measure '{m_col}'. Ideal for observing trend trajectories and seasonality.",
                "reasoning": f"Temporal dimension matched with continuous measure ({agg_method} aggregation). Visualizes progression and trends.",
                "source": f"Master Dataset -> {m_col} -> {date_col}"
            })
            chart_counter += 1

    # 2. BAR CHART / HORIZONTAL BAR CHART: Dimension vs Measure
    if dims_sorted and measures_sorted:
        dim_col = dims_sorted[0]
        m_col = measures_sorted[0]
        is_add = sem_layer.get(m_col, {}).get("is_additive", True) and sem_layer.get(m_col, {}).get("semantic_subtype") not in ["Rating", "Percentage", "Duration"]
        agg_method = "sum" if is_add else "mean"
        unique_cnt = df[dim_col].nunique()
        chart_type = "HORIZONTAL_BAR" if unique_cnt > 5 else "BAR"
        params = {"dim_col": dim_col, "m_col": m_col, "agg": agg_method}
        val_res = validate_chart_spec(chart_type, df, params)

        if val_res["is_valid"]:
            pts = val_res["data"]
            top_cat = pts[0] if pts else {"category": "N/A", "value": 0}
            metric_label = f"total {m_col}" if is_add else f"average {m_col}"
            tot_sum = sum(p['value'] for p in pts)
            if is_add and all(p['value'] >= 0 for p in pts) and tot_sum > 0:
                pct_share = f" ({round((top_cat['value'] / tot_sum)*100, 1)}% of top total)"
            else:
                pct_share = ""
            
            m_const = compute_variable_constancy(df[m_col])
            if m_const["is_constant"]:
                takeaway = f"Top segment is '{top_cat['category']}' contributing {metric_label} of {top_cat['value']:,}{pct_share}. Because '{m_col}' is constant, this distribution primarily reflects the count of observations per '{dim_col}' category."
            else:
                takeaway = f"Top leading segment is '{top_cat['category']}' contributing {metric_label} of {top_cat['value']:,}{pct_share}."

            recommendations.append({
                "id": f"chart_{chart_counter}",
                "chart_type": chart_type,
                "title": f"{m_col} by {dim_col}",
                "description": f"Segmentation ranking of {dim_col} evaluated by {metric_label}.",
                "x_axis": dim_col if chart_type == "BAR" else m_col,
                "y_axis": m_col if chart_type == "BAR" else dim_col,
                "is_valid": True,
                "fallback_message": None,
                "data": val_res["data"],
                "takeaway": takeaway,
                "suitability_reason": f"Categorical field '{dim_col}' with {unique_cnt} distinct categories evaluated against continuous metric '{m_col}'.",
                "reasoning": f"Categorical dimension with {unique_cnt} unique categories. Ranks {metric_label} contribution.",
                "source": f"Master Dataset -> {m_col} -> {dim_col}"
            })
            chart_counter += 1

    # 3. DONUT / PIE CHART: Only when low cardinality (2 to 6) and non-negative
    low_cardinality_dims = [d for d in dims_sorted if 2 <= df[d].nunique() <= 6]
    if low_cardinality_dims and measures_sorted:
        pie_dim = low_cardinality_dims[0]
        # Prefer additive measure for pie chart
        add_measures = [m for m in measures_sorted if sem_layer.get(m, {}).get("is_additive", True) and sem_layer.get(m, {}).get("semantic_subtype") not in ["Rating", "Percentage", "Duration"]]
        m_col = add_measures[0] if add_measures else measures_sorted[0]
        params = {"dim_col": pie_dim, "m_col": m_col}
        val_res = validate_chart_spec("DONUT", df, params)

        if val_res["is_valid"]:
            pts = val_res["data"]
            top_pie = max(pts, key=lambda p: p["value"]) if pts else {"category": "N/A", "value": 0}
            m_const = compute_variable_constancy(df[m_col])
            if m_const["is_constant"]:
                takeaway = f"'{top_pie['category']}' represents {top_pie['value']:,} of total {m_col}. Because '{m_col}' is constant, this proportion reflects category observation frequencies."
            else:
                takeaway = f"'{top_pie['category']}' dominates segment composition representing {top_pie['value']:,} of total {m_col}."

            recommendations.append({
                "id": f"chart_{chart_counter}",
                "chart_type": "DONUT",
                "title": f"Proportional Composition of {m_col} by {pie_dim}",
                "description": f"Donut chart displaying percentage share of {pie_dim} for {m_col}.",
                "x_axis": pie_dim,
                "y_axis": m_col,
                "is_valid": True,
                "fallback_message": None,
                "data": val_res["data"],
                "takeaway": takeaway,
                "suitability_reason": f"Low-cardinality discrete dimension ({df[pie_dim].nunique()} categories). Optimal for proportional composition breakdown.",
                "reasoning": f"Ideal low-cardinality dimension ({df[pie_dim].nunique()} categories) suitable for composition visualization.",
                "source": f"Master Dataset -> {pie_dim}"
            })
            chart_counter += 1

    # 4. STACKED BAR CHART: Two Dimensions + Measure
    if len(dims_sorted) >= 2 and measures_sorted:
        dim1, dim2 = dims_sorted[0], dims_sorted[1]
        m_col = measures_sorted[0]
        params = {"dim1": dim1, "dim2": dim2, "measure": m_col}
        val_res = validate_chart_spec("STACKED_BAR", df, params)

        if val_res["is_valid"]:
            recommendations.append({
                "id": f"chart_{chart_counter}",
                "chart_type": "STACKED_BAR",
                "title": f"{m_col} by {dim1} and {dim2}",
                "description": f"Multi-dimensional composition of {m_col} split across {dim1} and {dim2}.",
                "x_axis": dim1,
                "y_axis": m_col,
                "is_valid": True,
                "fallback_message": None,
                "data": val_res["data"],
                "takeaway": f"Visualizes cross-segmentation decomposition of '{m_col}' across primary dimension '{dim1}' and sub-segment '{dim2}'.",
                "suitability_reason": f"Dual categorical dimensions ({dim1}, {dim2}) paired with quantitative measure '{m_col}'.",
                "reasoning": "Two categorical dimensions matched with top measure for cross-segment decomposition.",
                "source": f"Master Dataset -> {dim1} x {dim2}"
            })
            chart_counter += 1

    # 5. HISTOGRAM: Continuous Numeric Distribution (Non-Constant Only)
    if non_constant_measures:
        target_m = non_constant_measures[0]
        params = {"variable": target_m}
        val_res = validate_chart_spec("HISTOGRAM", df, params)

        if val_res["is_valid"]:
            pts = val_res["data"]
            max_bin = max(pts, key=lambda p: p["count"]) if pts else {"bin": "N/A", "count": 0}
            takeaway = f"Distribution for '{target_m}' peaks in bin range [{max_bin['bin']}] with {max_bin['count']} observations."

            recommendations.append({
                "id": f"chart_{chart_counter}",
                "chart_type": "HISTOGRAM",
                "title": f"Distribution of {target_m}",
                "description": f"Frequency distribution displaying skewness and spread for {target_m}.",
                "variable": target_m,
                "x_axis": target_m,
                "y_axis": "Frequency Count",
                "is_valid": True,
                "fallback_message": None,
                "data": val_res["data"],
                "takeaway": takeaway,
                "suitability_reason": f"Continuous numeric metric '{target_m}' with non-zero variance. Evaluates skewness and empirical distribution spread.",
                "reasoning": "Continuous numerical measure. Evaluates distributional symmetry and spread.",
                "source": f"Master Dataset -> {target_m}"
            })
            chart_counter += 1

    # 6. SCATTER PLOT: 2 Non-Constant Measures with strongest linear association
    if len(non_constant_measures) >= 2:
        best_pair = (non_constant_measures[0], non_constant_measures[1])
        best_abs_r = -1.0
        for i in range(min(6, len(non_constant_measures))):
            for j in range(i + 1, min(6, len(non_constant_measures))):
                c_a, c_b = non_constant_measures[i], non_constant_measures[j]
                s_a, s_b = pd.to_numeric(df[c_a], errors='coerce'), pd.to_numeric(df[c_b], errors='coerce')
                valid_p = pd.DataFrame({c_a: s_a, c_b: s_b}).dropna()
                if len(valid_p) >= 3 and valid_p[c_a].std() > 0 and valid_p[c_b].std() > 0:
                    curr_r = abs(float(valid_p[c_a].corr(valid_p[c_b])))
                    if curr_r > best_abs_r:
                        best_abs_r = curr_r
                        best_pair = (c_a, c_b)

        col1, col2 = best_pair
        params = {"x_axis": col1, "y_axis": col2}
        val_res = validate_chart_spec("SCATTER", df, params)

        if val_res["is_valid"]:
            s1, s2 = pd.to_numeric(df[col1], errors='coerce'), pd.to_numeric(df[col2], errors='coerce')
            valid_pair = pd.DataFrame({col1: s1, col2: s2}).dropna()
            r_val = round(float(valid_pair[col1].corr(valid_pair[col2])), 3) if not valid_pair.empty else 0.0
            direction = "positive" if r_val > 0 else "negative"
            strength = "strong" if abs(r_val) >= 0.6 else ("moderate" if abs(r_val) >= 0.3 else "weak")
            takeaway = f"Pearson correlation coefficient r = {r_val} indicates a {strength} {direction} linear relationship between '{col1}' and '{col2}'."

            recommendations.append({
                "id": f"chart_{chart_counter}",
                "chart_type": "SCATTER",
                "title": f"{col1} vs {col2} Correlation Scatter Plot",
                "description": f"Co-variance relationship analysis between {col1} and {col2}.",
                "x_axis": col1,
                "y_axis": col2,
                "is_valid": True,
                "fallback_message": None,
                "data": val_res["data"],
                "takeaway": takeaway,
                "suitability_reason": f"Two quantitative measures with variance ({col1}, {col2}). Evaluates co-variance and bivariate linearity.",
                "reasoning": f"Two continuous numerical measures (r = {r_val}). Evaluates correlation coefficient and co-variance.",
                "source": f"Master Dataset -> {col1} vs {col2}"
            })
            chart_counter += 1

    # 7. CORRELATION HEATMAP: Multi-variable Matrix (Non-Constant Measures)
    if len(non_constant_measures) >= 2:
        params = {"variables": non_constant_measures[:6]}
        val_res = validate_chart_spec("CORRELATION_HEATMAP", df, params)

        if val_res["is_valid"]:
            recommendations.append({
                "id": f"chart_{chart_counter}",
                "chart_type": "CORRELATION_HEATMAP",
                "title": "Correlation Matrix of Numeric Measures",
                "description": "Pearson & Spearman correlation matrix across numerical measures.",
                "variables": non_constant_measures[:6],
                "x_axis": "Variable 1",
                "y_axis": "Variable 2",
                "is_valid": True,
                "fallback_message": None,
                "data": val_res["data"],
                "takeaway": f"Pairwise correlation matrix evaluating linear (Pearson) and monotonic (Spearman) co-variance across {len(non_constant_measures[:6])} quantitative metrics.",
                "suitability_reason": f"Multiple continuous numerical measures with variance ({len(non_constant_measures[:6])} columns). Displays full co-variance topology.",
                "reasoning": "Multi-variable linear co-variance matrix.",
                "source": "Master Dataset -> Pearson/Spearman Matrix"
            })
            chart_counter += 1

    # 8. BOX PLOT: Outliers & Distribution across Dimension (Non-Constant Metric)
    if dims_sorted and non_constant_measures:
        params = {"category": dims_sorted[0], "metric": non_constant_measures[0]}
        val_res = validate_chart_spec("BOX_PLOT", df, params)

        if val_res["is_valid"]:
            recommendations.append({
                "id": f"chart_{chart_counter}",
                "chart_type": "BOX_PLOT",
                "title": f"{non_constant_measures[0]} Spread across {dims_sorted[0]}",
                "description": f"Median, IQR spread, and extreme outlier values for {non_constant_measures[0]} grouped by {dims_sorted[0]}.",
                "category": dims_sorted[0],
                "metric": non_constant_measures[0],
                "x_axis": dims_sorted[0],
                "y_axis": non_constant_measures[0],
                "is_valid": True,
                "fallback_message": None,
                "data": val_res["data"],
                "takeaway": f"Grouped distribution analysis illustrating median, IQR percentiles (Q1, Q3), and variance spread of '{non_constant_measures[0]}' by '{dims_sorted[0]}'.",
                "suitability_reason": f"Categorical grouping variable '{dims_sorted[0]}' paired with numeric metric '{non_constant_measures[0]}'.",
                "reasoning": "Highlights Q1, median, Q3, and IQR outlier boundaries across categorical groups.",
                "source": f"Master Dataset -> {non_constant_measures[0]} by {dims_sorted[0]}"
            })
            chart_counter += 1

    # 9. WATERFALL: Additive volume transition / category breakdown
    wf = generate_waterfall_chart(df, sem_layer)
    if wf.get("is_valid") and len(wf.get("data", [])) >= 2:
        recommendations.append({
            "id": f"chart_{chart_counter}",
            "chart_type": "WATERFALL",
            "title": wf.get("title", "Volume Bridge Decomposition"),
            "description": wf.get("subtitle", "Step-by-step contribution transition across components"),
            "x_axis": "Component / Period",
            "y_axis": "Value",
            "is_valid": True,
            "fallback_message": None,
            "data": wf.get("data", []),
            "takeaway": f"Waterfall bridge representing additive category contribution to total." if "Cumulative Contribution" in wf.get("title", "") else f"Waterfall bridge representing additive volume bridge across recorded periods.",
            "suitability_reason": "Visualizes step-by-step additive bridge and cumulative total transition.",
            "reasoning": "Additive metric decomposition.",
            "source": "Master Dataset -> Waterfall Decomposition"
        })
        chart_counter += 1

    # 10. EVENT LOG / OBSERVATIONAL / CATEGORICAL FREQUENCY CHARTS (When no continuous measures exist)
    if not measures_sorted and (dims_sorted or time_cols):
        # 10a. Temporal Trend of Event / Observation Frequency
        if time_cols:
            date_col = time_cols[0]
            s_date = parse_flexible_datetime(df[date_col], errors='coerce').dropna()
            if len(s_date) >= 2:
                span_days = (s_date.max() - s_date.min()).days
                if span_days <= 60:
                    period_series = s_date.dt.strftime('%Y-%m-%d')
                else:
                    period_series = s_date.dt.strftime('%Y-%m')
                
                daily_counts = period_series.value_counts().sort_index()
                data_points = [{"x": str(k), "y": int(v), "period": str(k), "value": int(v)} for k, v in daily_counts.items()]
                if len(data_points) >= 2:
                    max_pt = max(data_points, key=lambda p: p["value"])
                    min_pt = min(data_points, key=lambda p: p["value"])
                    takeaway = f"Peak observation volume of {max_pt['value']:,} recorded on {max_pt['x']}, while lowest volume of {min_pt['value']:,} occurred on {min_pt['x']}."
                    recommendations.append({
                        "id": f"chart_{chart_counter}",
                        "chart_type": "LINE",
                        "title": f"Daily Observation Volume Trend ({date_col})",
                        "description": f"Progression of recorded observation frequency over temporal axis '{date_col}'.",
                        "x_axis": date_col,
                        "y_axis": "Observation Count",
                        "is_valid": True,
                        "fallback_message": None,
                        "data": data_points,
                        "takeaway": takeaway,
                        "suitability_reason": f"Temporal field '{date_col}' evaluated against event observation volume. Highlights reporting cadence and temporal peaks.",
                        "reasoning": f"Temporal progression of observation counts over '{date_col}'.",
                        "source": f"Master Dataset -> {date_col} (Event Count)"
                    })
                    chart_counter += 1

        # 10b-1. Dedicated High-Risk / Fatal Cases by Plant Location (Management Alert)
        impact_cols = [c for c in df.columns if any(k in c.lower() for k in ['impact', 'severity', 'incident', 'potential', 'classification', 'injury'])]
        loc_cols = [c for c in df.columns if any(k in c.lower() for k in ['location', 'plant', 'area', 'unit', 'site', 'facility', 'department'])]
        
        if impact_cols and loc_cols:
            imp_c = impact_cols[0]
            loc_c = loc_cols[0]
            fatal_mask = df[imp_c].fillna('').astype(str).str.contains('fatal', case=False)
            tot_fatal = int(fatal_mask.sum())
            if tot_fatal > 0:
                fatal_df = df[fatal_mask]
                fatal_by_loc = fatal_df[loc_c].dropna().astype(str).value_counts()
                if len(fatal_by_loc) > 0:
                    fatal_data = [{"category": str(k), "value": int(v), "x": str(k), "y": int(v), loc_c: str(k), "count": int(v)} for k, v in fatal_by_loc.items()]
                    top_loc = fatal_data[0]["category"]
                    top_loc_cnt = fatal_data[0]["value"]
                    takeaway = f"Critical Safety Alert: Total {tot_fatal} Fatal case(s) recorded. Highest concentration is in '{top_loc}' with {top_loc_cnt} fatal cases ({round(top_loc_cnt/tot_fatal*100, 1)}% of all fatal cases)."
                    recommendations.append({
                        "id": f"chart_{chart_counter}",
                        "chart_type": "HORIZONTAL_BAR" if len(fatal_data) > 3 else "BAR",
                        "title": f"Fatal Cases by Plant Location ({loc_c})",
                        "description": f"Management Risk Analysis: Exact distribution of {tot_fatal} Fatal case(s) across plant locations.",
                        "x_axis": "Fatal Count" if len(fatal_data) > 3 else loc_c,
                        "y_axis": loc_c if len(fatal_data) > 3 else "Fatal Count",
                        "is_valid": True,
                        "fallback_message": None,
                        "data": fatal_data,
                        "takeaway": takeaway,
                        "suitability_reason": f"High-risk exception isolation: Highlights {tot_fatal} critical Fatal cases across operational locations without drowning in safe observations.",
                        "reasoning": f"Critical risk stratification isolating {tot_fatal} Fatal cases by {loc_c}.",
                        "source": f"Master Dataset -> {loc_c} (Filtered: {imp_c} = 'Fatal')"
                    })
                    chart_counter += 1

        # 10b. Categorical Rankings (Horizontal Bar / Bar)
        for dim_col in dims_sorted[:4]:
            s_dim = df[dim_col].dropna().astype(str)
            s_dim = s_dim[~s_dim.str.strip().isin(['', 'nan', 'none', 'null', '\xa0'])]
            unique_cnt = s_dim.nunique()
            if unique_cnt >= 2:
                top_counts = s_dim.value_counts().head(10)
                data_points = [{"category": str(k), "value": int(v), "x": str(k), "y": int(v)} for k, v in top_counts.items()]
                if len(data_points) >= 2:
                    chart_type = "HORIZONTAL_BAR" if unique_cnt > 5 else "BAR"
                    top_cat = data_points[0]["category"]
                    top_val = data_points[0]["value"]
                    tot_val = len(s_dim)
                    pct_share = round((top_val / max(tot_val, 1)) * 100, 1)
                    takeaway = f"Leading segment in '{dim_col}' is '{top_cat}' with {top_val:,} observations ({pct_share}% of {tot_val:,} valid records)."
                    recommendations.append({
                        "id": f"chart_{chart_counter}",
                        "chart_type": chart_type,
                        "title": f"Observations by {dim_col}",
                        "description": f"Frequency ranking across top distinct '{dim_col}' categories.",
                        "x_axis": dim_col if chart_type == "BAR" else "Observation Count",
                        "y_axis": "Observation Count" if chart_type == "BAR" else dim_col,
                        "is_valid": True,
                        "fallback_message": None,
                        "data": data_points,
                        "takeaway": takeaway,
                        "suitability_reason": f"Discrete categorical dimension '{dim_col}' with {unique_cnt} distinct categories evaluated by observation frequency.",
                        "reasoning": f"Ranks observation volume by '{dim_col}'.",
                        "source": f"Master Dataset -> {dim_col}"
                    })
                    chart_counter += 1

        # 10c. Proportional Composition Donut (Low Cardinality: 2-6 categories)
        low_card_dims = [d for d in dims_sorted if 2 <= df[d].dropna().astype(str).str.strip().nunique() <= 6]
        for pie_dim in low_card_dims[:2]:
            s_pie = df[pie_dim].dropna().astype(str)
            s_pie = s_pie[~s_pie.str.strip().isin(['', 'nan', 'none', 'null', '\xa0'])]
            if 2 <= s_pie.nunique() <= 6:
                counts = s_pie.value_counts()
                data_points = [{"category": str(k), "value": int(v), "x": str(k), "y": int(v)} for k, v in counts.items()]
                top_pie = data_points[0]
                pct_share = round((top_pie["value"] / max(len(s_pie), 1)) * 100, 1)
                takeaway = f"'{top_pie['category']}' represents {top_pie['value']:,} observations ({pct_share}% of total {len(s_pie):,})."
                recommendations.append({
                    "id": f"chart_{chart_counter}",
                    "chart_type": "DONUT",
                    "title": f"Proportional Composition of Observations by {pie_dim}",
                    "description": f"Donut chart displaying percentage share of observations across '{pie_dim}'.",
                    "x_axis": pie_dim,
                    "y_axis": "Observation Count",
                    "is_valid": True,
                    "fallback_message": None,
                    "data": data_points,
                    "takeaway": takeaway,
                    "suitability_reason": f"Low-cardinality discrete dimension ({s_pie.nunique()} categories). Optimal for proportional composition breakdown.",
                    "reasoning": f"Ideal low-cardinality dimension ({s_pie.nunique()} categories) suitable for composition visualization.",
                    "source": f"Master Dataset -> {pie_dim}"
                })
                chart_counter += 1

        # 10d. Multi-Dimensional Stacked Bar
        if len(dims_sorted) >= 2:
            cand_dim2 = [d for d in dims_sorted if 2 <= df[d].dropna().astype(str).str.strip().nunique() <= 6]
            cand_dim1 = [d for d in dims_sorted if 3 <= df[d].dropna().astype(str).str.strip().nunique() <= 50 and d not in cand_dim2] or [d for d in dims_sorted if d not in cand_dim2]
            if cand_dim1 and cand_dim2:
                dim1 = cand_dim1[0]
                dim2 = cand_dim2[0]
                top_d1 = df[dim1].value_counts().head(8).index
                sub_df = df[df[dim1].isin(top_d1)]
                pivot = pd.crosstab(sub_df[dim1], sub_df[dim2])
                pivot_dict = pivot.reset_index().to_dict(orient='records')
                for r in pivot_dict:
                    r["x"] = str(r.get(dim1, ""))
                recommendations.append({
                    "id": f"chart_{chart_counter}",
                    "chart_type": "STACKED_BAR",
                    "title": f"Observations by {dim1} Split by {dim2}",
                    "description": f"Cross-dimensional frequency composition of '{dim2}' across top '{dim1}' categories.",
                    "x_axis": dim1,
                    "y_axis": "Observation Count",
                    "series_keys": [str(c) for c in pivot.columns],
                    "is_valid": True,
                    "fallback_message": None,
                    "data": pivot_dict,
                    "takeaway": f"Cross-segment distribution illustrates relative severity / classification mix of '{dim2}' across top categories in '{dim1}'.",
                    "suitability_reason": f"Multi-dimensional categorical composition ({dim1} x {dim2}). Decomposes segment volume.",
                    "reasoning": f"Two categorical dimensions matched for cross-segment decomposition.",
                    "source": f"Master Dataset -> {dim1} x {dim2}"
                })
                chart_counter += 1

    return recommendations

def generate_waterfall_chart(df: pd.DataFrame, sem_layer: dict) -> dict:
    """
    Generates a Governed Financial/Additive Waterfall Chart.
    Strict Waterfall Governance:
    - START +/- ADDITIVE COMPONENTS = END within 0.01 tolerance
    - Suppresses waterfall for percentages, ratios, averages, medians, scores, indices, ratings, and margins
    - Blocks calculation when multi-currency mismatch is detected without harmonization
    - Validates both Category Breakdown Bridge and Temporal Volume Bridge
    """
    measures = [c for c, m in sem_layer.items() if m.get("is_measure") and not m.get("is_identifier")]
    dimensions = [c for c, m in sem_layer.items() if m.get("is_dimension") and not m.get("is_identifier") and m.get("grouping_eligibility", True)]
    time_cols = [c for c, m in sem_layer.items() if m.get("is_time") or m.get("semantic_role") == "Date/Time"]

    # 0. Check Currency Consistency across dataset
    currency_cols = [c for c in df.columns if any(k in c.lower() for k in ['curr', 'currency']) or sem_layer.get(c, {}).get("semantic_role") == "Currency"]
    for c_col in currency_cols:
        if df[c_col].dropna().nunique() > 1:
            return {
                "chart_type": "WATERFALL",
                "title": "Waterfall Chart",
                "is_valid": False,
                "waterfall_status": "INVALID",
                "currency_mismatch": True,
                "reason": "Waterfall blocked — multi-currency dataset detected without harmonization. Mixed currency values cannot be summed.",
                "fallback_message": "Waterfall blocked — multi-currency dataset detected without harmonization. Mixed currency values cannot be summed.",
                "data": []
            }

    # 1. Category Component Decomposition Bridge (Top Category contributions to Total)
    if dimensions and measures:
        dim = dimensions[0]
        m_col = measures[0]
        col_meta = sem_layer.get(m_col, {})
        m_type = col_meta.get("measure_type", "")
        m_low = m_col.lower()

        # Non-additive metrics rule
        non_additive_types = ["Percentage", "Ratio", "Score", "Score / Index", "Index", "Rating"]
        non_additive_tokens = ['pct', 'percent', 'ratio', 'margin', 'rate', 'score', 'index', 'rating', 'average', 'avg', 'median']
        is_non_add = (
            m_type in non_additive_types or
            any(k in m_low for k in non_additive_tokens) or
            col_meta.get("semantic_subtype") in ["Rating", "Percentage", "Duration"] or
            not col_meta.get("is_additive", True)
        )

        if is_non_add:
            return {
                "chart_type": "WATERFALL",
                "title": f"Waterfall Decomposition ({m_col})",
                "is_valid": False,
                "waterfall_status": "INVALID",
                "reason": f"Waterfall unavailable — metric '{m_col}' is non-additive (percentages, ratios, scores, ratings, margins, and medians cannot be decomposed additively).",
                "fallback_message": f"Waterfall unavailable — metric '{m_col}' is non-additive (percentages, ratios, scores, ratings, margins, and medians cannot be decomposed additively).",
                "data": []
            }

        try:
            valid_df = df[[dim, m_col]].dropna()
            valid_df[m_col] = pd.to_numeric(valid_df[m_col], errors='coerce')
            valid_df = valid_df.dropna()
            if len(valid_df) >= 3 and valid_df[dim].nunique() >= 2:
                grp = valid_df.groupby(dim)[m_col].sum().sort_values(ascending=False)
                if grp.sum() > 0:
                    top_items = grp.head(4)
                    other_val = grp.iloc[4:].sum() if len(grp) > 4 else 0.0
                    tot_val = float(grp.sum())

                    items = []
                    running = 0.0
                    for cat, v in top_items.items():
                        v_float = round(float(v), 2)
                        items.append({
                            "category": str(cat),
                            "value": v_float,
                            "type": "addition" if v_float >= 0 else "subtraction",
                            "start": round(running, 2),
                            "end": round(running + v_float, 2)
                        })
                        running += v_float
                    if other_val > 0:
                        v_other = round(float(other_val), 2)
                        items.append({
                            "category": "Other Segments",
                            "value": v_other,
                            "type": "addition",
                            "start": round(running, 2),
                            "end": round(running + v_other, 2)
                        })
                        running += v_other
                    items.append({
                        "category": f"Total {m_col}",
                        "value": round(tot_val, 2),
                        "type": "total",
                        "start": 0.0,
                        "end": round(tot_val, 2)
                    })

                    # Strict numerical additivity check: START + sum(components) == END
                    comp_sum = sum(item["value"] for item in items if item["type"] != "total")
                    if abs(comp_sum - tot_val) > 0.01:
                        return {
                            "chart_type": "WATERFALL",
                            "title": f"{m_col} Cumulative Contribution by {dim}",
                            "is_valid": False,
                            "waterfall_status": "INVALID",
                            "reason": f"Waterfall invalid: START + components != END (sum of components {comp_sum:.2f} != total {tot_val:.2f}).",
                            "fallback_message": "Waterfall invalid: START + components != END.",
                            "data": []
                        }

                    return {
                        "chart_type": "WATERFALL",
                        "title": f"{m_col} Cumulative Contribution by {dim}",
                        "subtitle": f"Additive decomposition of total {m_col} across {dim} categories",
                        "priority": "HIGH",
                        "is_valid": True,
                        "waterfall_status": "VALID",
                        "reconciliation_tolerance": 0.01,
                        "fallback_message": None,
                        "data": items
                    }
        except Exception:
            pass

    # 2. Temporal Volume Bridge
    if time_cols and measures:
        try:
            date_col = time_cols[0]
            m_col = measures[0]
            col_meta = sem_layer.get(m_col, {})
            m_type = col_meta.get("measure_type", "")
            m_low = m_col.lower()

            non_additive_types = ["Percentage", "Ratio", "Score", "Score / Index", "Index", "Rating"]
            non_additive_tokens = ['pct', 'percent', 'ratio', 'margin', 'rate', 'score', 'index', 'rating', 'average', 'avg', 'median']
            if (
                m_type in non_additive_types or
                any(k in m_low for k in non_additive_tokens) or
                col_meta.get("semantic_subtype") in ["Rating", "Percentage", "Duration"] or
                not col_meta.get("is_additive", True)
            ):
                return {
                    "chart_type": "WATERFALL",
                    "title": f"Temporal Volume Bridge ({m_col})",
                    "is_valid": False,
                    "waterfall_status": "INVALID",
                    "reason": f"Waterfall unavailable — metric '{m_col}' is non-additive.",
                    "fallback_message": f"Waterfall unavailable — metric '{m_col}' is non-additive.",
                    "data": []
                }

            t_df = df[[date_col, m_col]].copy()
            t_df[date_col] = pd.to_datetime(t_df[date_col], errors='coerce')
            t_df[m_col] = pd.to_numeric(t_df[m_col], errors='coerce')
            t_df = t_df.dropna().sort_values(date_col)
            if len(t_df) >= 2:
                date_span = (t_df[date_col].max() - t_df[date_col].min()).days
                if date_span <= 60:
                    t_df['Period'] = t_df[date_col].dt.strftime('%Y-%m-%d')
                else:
                    t_df['Period'] = t_df[date_col].dt.to_period('M').astype(str)

                monthly = t_df.groupby('Period')[m_col].sum().reset_index()
                if len(monthly) >= 2:
                    first_m = monthly.iloc[0]
                    last_m = monthly.iloc[-1]
                    start_val = round(float(first_m[m_col]), 2)
                    end_val = round(float(last_m[m_col]), 2)
                    diff = round(end_val - start_val, 2)

                    # Strict numerical additivity check: START + shift == END
                    if abs((start_val + diff) - end_val) > 0.01:
                        return {
                            "chart_type": "WATERFALL",
                            "title": f"Temporal Volume Bridge ({first_m['Period']} to {last_m['Period']})",
                            "is_valid": False,
                            "waterfall_status": "INVALID",
                            "reason": "Waterfall invalid: START + shift != END.",
                            "fallback_message": "Waterfall invalid: START + shift != END.",
                            "data": []
                        }

                    items = [
                        {"category": f"Start ({first_m['Period']})", "value": start_val, "type": "total", "start": 0.0, "end": start_val},
                        {"category": "Net Period Shift", "value": diff, "type": "addition" if diff >= 0 else "subtraction", "start": start_val, "end": end_val},
                        {"category": f"Current ({last_m['Period']})", "value": end_val, "type": "total", "start": 0.0, "end": end_val}
                    ]
                    return {
                        "chart_type": "WATERFALL",
                        "title": f"Temporal Volume Bridge ({first_m['Period']} to {last_m['Period']})",
                        "subtitle": f"Period-over-period net shift in {m_col}",
                        "priority": "HIGH",
                        "is_valid": True,
                        "waterfall_status": "VALID",
                        "reconciliation_tolerance": 0.01,
                        "fallback_message": None,
                        "data": items
                    }
        except Exception:
            pass

    return {
        "chart_type": "WATERFALL",
        "title": "Waterfall Chart",
        "is_valid": False,
        "waterfall_status": "NOT_APPLICABLE",
        "fallback_message": "Insufficient data for this visualization",
        "data": []
    }

def generate_stacked_bar_charts(df: pd.DataFrame, sem_layer: dict) -> list:
    """
    Generates Stacked Bar and 100% Stacked Bar Charts for multi-dimensional categorical performance.
    """
    dimensions = [c for c, m in sem_layer.items() if m.get("is_dimension") and m.get("grouping_eligibility") and not m.get("is_identifier")]
    measures = [c for c, m in sem_layer.items() if m.get("is_measure") and not m.get("is_identifier")]

    stacked_charts = []
    if len(dimensions) >= 2:
        if measures:
            dim1 = dimensions[0]
            dim2 = dimensions[1]
            m_col = measures[0]

            try:
                pivot = df.groupby([dim1, dim2])[m_col].sum().unstack().fillna(0.0)
                if len(pivot) <= 10 and len(pivot.columns) <= 6:
                    pivot_dict = pivot.reset_index().to_dict(orient='records')
                    for r in pivot_dict:
                        r["x"] = str(r.get(dim1, ""))
                    
                    # 1. Standard Stacked Bar
                    stacked_charts.append({
                        "chart_type": "STACKED_BAR",
                        "title": f"{m_col} by {dim1} Split by {dim2}",
                        "subtitle": f"Cross-dimensional contribution of {dim2} across {dim1}",
                        "priority": "HIGH",
                        "x_axis": dim1,
                        "x_dimension": dim1,
                        "series_dimension": dim2,
                        "measure": m_col,
                        "series_keys": [str(c) for c in pivot.columns],
                        "grouped_values": pivot_dict,
                        "total_values": {str(k): round(float(v), 2) for k, v in pivot.sum(axis=1).items()},
                        "source_result_id": f"res_stacked_{dim1}_{dim2}_{m_col}",
                        "is_valid": True,
                        "fallback_message": None,
                        "data": pivot_dict
                    })

                    # 2. 100% Stacked Bar
                    pivot_sum = pivot.sum(axis=1)
                    pivot_sum[pivot_sum == 0] = 1.0
                    pct_pivot = pivot.div(pivot_sum, axis=0).mul(100).round(1)
                    pct_dict = pct_pivot.reset_index().to_dict(orient='records')
                    for r in pct_dict:
                        r["x"] = str(r.get(dim1, ""))
                    stacked_charts.append({
                        "chart_type": "STACKED_BAR_100",
                        "title": f"Percentage Mix of {dim2} across {dim1}",
                        "subtitle": f"Proportional composition of {dim2} within each {dim1} segment",
                        "priority": "MEDIUM",
                        "x_axis": dim1,
                        "x_dimension": dim1,
                        "series_dimension": dim2,
                        "measure": m_col,
                        "series_keys": [str(c) for c in pivot.columns],
                        "grouped_values": pct_dict,
                        "total_values": {str(k): 100.0 for k in pct_pivot.index},
                        "source_result_id": f"res_stacked_100_{dim1}_{dim2}_{m_col}",
                        "is_valid": True,
                        "fallback_message": None,
                        "data": pct_dict
                    })
            except Exception:
                pass
        else:
            try:
                cand_dim2 = [d for d in dimensions if 2 <= df[d].dropna().astype(str).str.strip().nunique() <= 6]
                cand_dim1 = [d for d in dimensions if 3 <= df[d].dropna().astype(str).str.strip().nunique() <= 50 and d not in cand_dim2] or [d for d in dimensions if d not in cand_dim2]
                if cand_dim1 and cand_dim2:
                    dim1 = cand_dim1[0]
                    dim2 = cand_dim2[0]
                    top_d1 = df[dim1].value_counts().head(8).index
                    sub_df = df[df[dim1].isin(top_d1)]
                    pivot = pd.crosstab(sub_df[dim1], sub_df[dim2])
                    if len(pivot) <= 10 and len(pivot.columns) <= 6:
                        pivot_dict = pivot.reset_index().to_dict(orient='records')
                        for r in pivot_dict:
                            r["x"] = str(r.get(dim1, ""))
                        
                        stacked_charts.append({
                            "chart_type": "STACKED_BAR",
                            "title": f"Observations by {dim1} Split by {dim2}",
                            "subtitle": f"Cross-dimensional frequency of {dim2} across top {dim1} categories",
                            "priority": "HIGH",
                            "x_axis": dim1,
                            "x_dimension": dim1,
                            "series_dimension": dim2,
                            "measure": "Observation_Count",
                            "series_keys": [str(c) for c in pivot.columns],
                            "grouped_values": pivot_dict,
                            "total_values": {str(k): round(float(v), 2) for k, v in pivot.sum(axis=1).items()},
                            "source_result_id": f"res_stacked_{dim1}_{dim2}_count",
                            "is_valid": True,
                            "fallback_message": None,
                            "data": pivot_dict
                        })

                        pivot_sum = pivot.sum(axis=1)
                        pivot_sum[pivot_sum == 0] = 1.0
                        pct_pivot = pivot.div(pivot_sum, axis=0).mul(100).round(1)
                        pct_dict = pct_pivot.reset_index().to_dict(orient='records')
                        for r in pct_dict:
                            r["x"] = str(r.get(dim1, ""))
                        stacked_charts.append({
                            "chart_type": "STACKED_BAR_100",
                            "title": f"Percentage Mix of {dim2} across {dim1}",
                            "subtitle": f"Proportional composition of {dim2} within each {dim1} category",
                            "priority": "MEDIUM",
                            "x_axis": dim1,
                            "x_dimension": dim1,
                            "series_dimension": dim2,
                            "measure": "Observation_Count",
                            "series_keys": [str(c) for c in pivot.columns],
                            "grouped_values": pct_dict,
                            "total_values": {str(k): 100.0 for k in pct_pivot.index},
                            "source_result_id": f"res_stacked_100_{dim1}_{dim2}_count",
                            "is_valid": True,
                            "fallback_message": None,
                            "data": pct_dict
                        })
            except Exception:
                pass

    return stacked_charts

def run_automated_eda(df: pd.DataFrame, column_overrides: dict = None):
    """
    Automated EDA Engine:
    Runs deterministic exploratory analysis across numerical, categorical, and temporal variables.
    Computes robust statistical metrics (skewness, kurtosis, IQR, zero inflation, outliers, Spearman/Pearson, trends).
    Dataset-agnostic without domain assumptions.
    """
    df = coerce_numeric_columns(df)
    domain = detect_domain(df)
    num_rows = len(df)

    sem_layer = {str(c): infer_column_semantic_metadata(c, df[c], num_rows, override=column_overrides.get(c) if column_overrides else None) for c in df.columns}

    measures = [c for c, m in sem_layer.items() if m.get("is_measure") and not m.get("is_identifier")]
    eligible_measures = [c for c in measures if sem_layer[c].get("analytical_eligible", True) is True]
    dimensions = [c for c, m in sem_layer.items() if m.get("is_dimension") and m.get("grouping_eligibility") and not m.get("is_identifier")]
    time_cols = [c for c, m in sem_layer.items() if m.get("is_time") or m.get("semantic_role") in ["Date/Time", "Time"]]

    # 1. Dataset Overview & Data Health Summary
    canonical_health = calculate_canonical_dataset_health(df)
    total_cells = num_rows * len(df.columns) if num_rows * len(df.columns) > 0 else 1
    total_nulls = int(df.isnull().sum().sum())
    total_duplicates = int(df.duplicated().sum())

    health_score = float(canonical_health.health_score)

    overview = {
        "num_rows": num_rows,
        "num_columns": len(df.columns),
        "total_nulls": total_nulls,
        "total_duplicates": total_duplicates,
        "health_score": health_score,
        "measures_count": len(measures),
        "dimensions_count": len(dimensions),
        "time_columns_count": len(time_cols)
    }

    # 2. Numeric Variables Summary & Zero-Inflation Outlier Logic
    numeric_summary = {}
    for col in measures:
        s = pd.to_numeric(df[col], errors='coerce').dropna().astype(float)
        if s.empty:
            total_count = len(df[col])
            numeric_summary[col] = {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "mode": None,
                "modes": [],
                "is_multimodal": False,
                "std": 0.0,
                "variance": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p05": 0.0,
                "p25": 0.0,
                "q1": 0.0,
                "p75": 0.0,
                "q3": 0.0,
                "p95": 0.0,
                "iqr": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "outliers_count": 0,
                "is_zero_inflated": False,
                "zero_percentage": 0.0,
                "nonzero_percentage": 0.0,
                "missing_count": int(df[col].isnull().sum()),
                "missing_percentage": 100.0
            }
            continue
        
        total_count = len(df[col])
        zero_count = int((s == 0).sum())
        zero_ratio = zero_count / max(total_count, 1)
        is_zero_heavy = (zero_ratio >= 0.4)
        is_zero_inflated = is_zero_heavy

        # Non-zero cohort analysis
        non_zero_s = s[s != 0]
        non_zero_mean = round(float(non_zero_s.mean()), 4) if not non_zero_s.empty else None
        non_zero_count = int(len(non_zero_s))

        # Standard Tukey's Interquartile Range (IQR) on the complete metric distribution
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        if iqr > 0:
            outlier_bound_lower = q1 - 1.5 * iqr
            outlier_bound_upper = q3 + 1.5 * iqr
            outlier_count = int(((s < outlier_bound_lower) | (s > outlier_bound_upper)).sum())
        elif s.std() > 0:
            # Fallback to 3-sigma rule when IQR is zero
            m_s, std_s = float(s.mean()), float(s.std())
            outlier_count = int((abs(s - m_s) > 3.0 * std_s).sum())
            outlier_bound_lower = m_s - 3.0 * std_s
            outlier_bound_upper = m_s + 3.0 * std_s
        else:
            outlier_count = 0
            outlier_bound_lower = float(s.min()) if not s.empty else 0.0
            outlier_bound_upper = float(s.max()) if not s.empty else 0.0

        outliers_percentage = round((outlier_count / max(total_count, 1)) * 100, 2)

        skew_val = float(s.skew()) if len(s) > 2 and s.std() > 0 else 0.0
        kurt_val = float(s.kurtosis()) if len(s) > 3 and s.std() > 0 else 0.0
        skewness = round(skew_val, 2) if math.isfinite(skew_val) else 0.0
        kurtosis = round(kurt_val, 2) if math.isfinite(kurt_val) else 0.0

        # Empirical Mode calculation (unimodal, multimodal, constant, empty)
        modes_series = s.mode()
        if not modes_series.empty:
            modes_list = [round(float(m), 2) for m in modes_series.dropna().tolist() if math.isfinite(m)]
            primary_mode = modes_list[0] if modes_list else None
            is_multimodal = bool(len(modes_list) > 1 and len(modes_list) < len(s))
        else:
            modes_list = []
            primary_mode = None
            is_multimodal = False

        constancy = compute_variable_constancy(df[col])
        is_const = constancy["is_constant"]
        const_stmt = generate_constant_variable_statement(col, constancy, unit=sem_layer.get(col, {}).get("unit", ""), time_col=time_cols[0] if time_cols else None, df=df) if is_const else None

        numeric_summary[col] = {
            "count": int(s.count()),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 2),
            "mode": primary_mode,
            "modes": modes_list[:5],
            "is_multimodal": is_multimodal,
            "std": round(float(constancy["std"]), 4) if not is_const and len(s) > 1 else 0.0,
            "variance": round(float(constancy["variance"]), 4) if not is_const and len(s) > 1 else 0.0,
            "constant_variable": is_const,
            "constant_statement": const_stmt,
            "unique_count": int(s.nunique()),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "p05": round(float(s.quantile(0.05)), 2),
            "p25": round(q1, 3),
            "q1": round(q1, 3),
            "p75": round(q3, 3),
            "q3": round(q3, 3),
            "p95": round(float(s.quantile(0.95)), 2),
            "iqr": round(iqr, 3),
            "iqr_lower_bound": round(outlier_bound_lower, 3),
            "iqr_upper_bound": round(outlier_bound_upper, 3),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "outliers_count": outlier_count,
            "outliers_percentage": outliers_percentage,
            "is_zero_inflated": is_zero_heavy,
            "is_zero_heavy": is_zero_heavy,
            "distribution_type": "zero-heavy distribution" if is_zero_heavy else ("normal" if abs(skewness) < 0.5 else "skewed"),
            "zero_count": zero_count,
            "zero_percentage": round(zero_ratio * 100, 2),
            "nonzero_percentage": round((1.0 - zero_ratio) * 100, 2),
            "non_zero_count": non_zero_count,
            "non_zero_mean": non_zero_mean,
            "missing_count": int(df[col].isnull().sum()),
            "missing_percentage": round((int(df[col].isnull().sum()) / max(total_count, 1)) * 100, 2)
        }

    # 2b. Identifier Variables Summary (Constrained Non-Aggregatable Metrics ONLY)
    identifiers = [c for c, m in sem_layer.items() if m.get("is_identifier")]
    identifier_summary = {}
    for col in identifiers:
        s = df[col]
        total_cnt = len(s)
        missing_cnt = int(s.isnull().sum())
        unique_cnt = int(s.nunique())
        uniqueness_pct = round((unique_cnt / max(total_cnt, 1)) * 100, 2)
        identifier_summary[col] = {
            "total_count": total_cnt,
            "unique_count": unique_cnt,
            "missing_count": missing_cnt,
            "uniqueness_percentage": uniqueness_pct
        }

    # 3. Categorical Variables Summary
    categorical_summary = {}
    for col in dimensions:
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        top_counts = s.value_counts().head(5).to_dict()
        total_rows = len(df)
        valid_rows = len(s)
        missing_cnt = int(df[col].isnull().sum())
        missing_pct = round((missing_cnt / max(total_rows, 1)) * 100, 1)

        top_cat = str(list(top_counts.keys())[0]) if top_counts else "N/A"
        top_cat_count = int(list(top_counts.values())[0]) if top_counts else 0
        top_cat_pct = round((top_cat_count / max(total_rows, 1)) * 100, 1)

        top_pcts = {str(k): round((v / max(valid_rows, 1)) * 100, 1) for k, v in top_counts.items()}
        conc_ratio = round((top_cat_count / max(valid_rows, 1)) * 100, 1) if top_counts else 0.0

        categorical_summary[col] = {
            "column": col,
            "unique_categories": int(s.nunique()),
            "unique_count": int(s.nunique()),
            "top_category": top_cat,
            "top_category_count": top_cat_count,
            "top_category_pct": top_cat_pct,
            "dominant_category": top_cat,
            "dominant_category_pct": top_cat_pct,
            "top_counts": {str(k): int(v) for k, v in top_counts.items()},
            "top_percentages": top_pcts,
            "concentration_ratio": conc_ratio,
            "missing_values": missing_cnt,
            "missing_count": missing_cnt,
            "missing_percentage": missing_pct,
            "total_records": total_rows,
            "valid_records": valid_rows
        }

    # 4. Relationships & Associations (Pearson & Spearman) on Eligible Measures Only
    relationships = {
        "status": "sufficient_data" if len(eligible_measures) >= 2 else "insufficient_data",
        "reason": "Sufficient numeric measures present" if len(eligible_measures) >= 2 else "Dataset contains fewer than 2 continuous numeric measures for correlation analysis.",
        "pearson_matrix": {},
        "spearman_matrix": {},
        "strong_associations": []
    }

    if len(eligible_measures) >= 2:
        measures_df = df[eligible_measures].apply(pd.to_numeric, errors='coerce')
        constancy_map = {m: compute_variable_constancy(measures_df[m]) for m in eligible_measures}
        non_const_measures = [m for m in eligible_measures if not constancy_map[m]["is_constant"]]

        pearson_matrix = {m: {m: (1.0 if not constancy_map[m]["is_constant"] else 0.0)} for m in eligible_measures}
        spearman_matrix = {m: {m: (1.0 if not constancy_map[m]["is_constant"] else 0.0)} for m in eligible_measures}
        el_cache = {m: {"analytical_eligible": True} for m in eligible_measures}

        for i in range(len(eligible_measures)):
            m1 = eligible_measures[i]
            for j in range(i + 1, len(eligible_measures)):
                m2 = eligible_measures[j]
                if constancy_map[m1]["is_constant"] or constancy_map[m2]["is_constant"]:
                    p_val = 0.0
                    s_val = 0.0
                else:
                    corr_res = safe_correlation(
                        measures_df, m1, m2,
                        check_paradox=False,
                        check_derivation=False,
                        eligibility_cache=el_cache
                    )
                    p_val = corr_res.get("pearson") if corr_res.get("pearson") is not None else 0.0
                    s_val = corr_res.get("spearman") if corr_res.get("spearman") is not None else 0.0

                pearson_matrix[m1][m2] = pearson_matrix[m2][m1] = p_val
                spearman_matrix[m1][m2] = spearman_matrix[m2][m1] = s_val

        relationships["pearson_matrix"] = pearson_matrix
        relationships["spearman_matrix"] = spearman_matrix

        candidate_pairs = []
        for i in range(len(non_const_measures)):
            for j in range(i + 1, len(non_const_measures)):
                m1, m2 = non_const_measures[i], non_const_measures[j]
                r_val = float(pearson_matrix[m1][m2])
                s_val = float(spearman_matrix[m1][m2])
                if abs(r_val) >= 0.4:
                    candidate_pairs.append((abs(r_val), m1, m2, r_val, s_val))

        candidate_pairs.sort(key=lambda x: x[0], reverse=True)

        assoc_list = []
        for idx, (_, m1, m2, r_val, s_val) in enumerate(candidate_pairs):
            check_deep = (idx < 25)
            full_res = safe_correlation(
                measures_df, m1, m2,
                check_paradox=check_deep,
                check_derivation=check_deep,
                eligibility_cache=el_cache
            )
            val_n = full_res.get("valid_n", len(df[[m1, m2]].dropna()) if m1 in df.columns and m2 in df.columns else len(df))
            tot_n = len(df)
            cov_pct = full_res.get("coverage_pct", round((val_n / max(tot_n, 1)) * 100, 2))
            is_det = bool(full_res.get("is_deterministic") or full_res.get("relationship_type") in ["DETERMINISTIC_TRANSFORMATION", "DERIVED"])
            rel_type = full_res.get("relationship_type", "INDEPENDENT")
            classif = full_res.get("classification") or ("STRUCTURAL_RELATIONSHIP" if is_det else "STATISTICAL_ASSOCIATION")
            interp = full_res.get("interpretation") or (
                f"Statistical analysis shows a {('strong' if abs(r_val) >= 0.6 else 'moderate')} {('positive' if r_val > 0 else 'negative')} linear association between '{m1}' and '{m2}' (Pearson r = {r_val}). Note: Association reflects statistical co-variance and does not establish causation."
            )
            assoc_list.append({
                "metric_1": m1,
                "metric_2": m2,
                "variable_x": m1,
                "variable_y": m2,
                "pearson_r": r_val,
                "spearman_r": s_val,
                "spearman_rho": s_val,
                "valid_n": val_n,
                "total_n": tot_n,
                "coverage_pct": cov_pct,
                "direction": "positive" if r_val > 0 else "negative",
                "strength": "strong" if abs(r_val) >= 0.6 else "moderate",
                "is_deterministic": is_det,
                "relationship_type": rel_type,
                "classification": classif,
                "derivation_formula": full_res.get("derivation_formula"),
                "interpretation": interp
            })

        strategic_assocs = [a for a in assoc_list if not a.get("is_deterministic") and a.get("classification") != "STRUCTURAL_RELATIONSHIP"]
        structural_rels = [a for a in assoc_list if a.get("is_deterministic") or a.get("classification") == "STRUCTURAL_RELATIONSHIP"]

        strategic_assocs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)
        structural_rels.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

        relationships["strategic_associations"] = strategic_assocs
        relationships["structural_relationships"] = structural_rels
        relationships["strong_associations"] = strategic_assocs + structural_rels

    # 5. Time Series Analysis
    time_series_analysis = {
        "status": "sufficient_data" if (time_cols and (measures or len(df) >= 2)) else "insufficient_data",
        "reason": "Temporal column present" if (time_cols and (measures or len(df) >= 2)) else "Dataset lacks required datetime columns for temporal trend analysis.",
        "interpolation_enabled": False,
        "interpolation_method": None,
        "interpolated_periods": [],
        "original_observations": 0,
        "interpolation_warning": None,
        "missing_periods": [],
        "duplicate_timestamps": 0,
        "irregular_intervals": False,
        "frequency": "Unknown",
        "volatility": 0.0,
        "seasonality_detected": False,
        "structural_break_detected": False
    }
    if time_cols:
        date_col = time_cols[0]
        m_col = measures[0] if measures else "Observation_Count"
        try:
            if measures:
                t_df = df[[date_col, m_col]].copy()
                t_df[date_col] = parse_flexible_datetime(t_df[date_col], errors='coerce')
                t_df[m_col] = pd.to_numeric(t_df[m_col], errors='coerce')
                t_df = t_df.dropna().sort_values(date_col)
            else:
                t_df = df[[date_col]].copy()
                t_df[date_col] = parse_flexible_datetime(t_df[date_col], errors='coerce')
                t_df[m_col] = 1
                t_df = t_df.dropna().sort_values(date_col)

            duplicate_ts_cnt = int(t_df[date_col].duplicated().sum())

            if len(t_df) >= 2:
                date_span = (t_df[date_col].max() - t_df[date_col].min()).days
                is_daily = (date_span <= 60)
                if is_daily:
                    t_df['YearMonth'] = t_df[date_col].dt.strftime('%Y-%m-%d')
                    period_name = "Daily"
                    freq_code = "D"
                else:
                    t_df['YearMonth'] = t_df[date_col].dt.to_period('M').astype(str)
                    period_name = "Monthly"
                    freq_code = "M"

                if measures:
                    is_add = sem_layer.get(m_col, {}).get("is_additive", True) and sem_layer.get(m_col, {}).get("semantic_subtype") not in ["Rating", "Percentage", "Duration"]
                    if is_add:
                        monthly = t_df.groupby('YearMonth')[m_col].sum().reset_index()
                    else:
                        monthly = t_df.groupby('YearMonth')[m_col].mean().round(2).reset_index()
                else:
                    is_add = True
                    monthly = t_df.groupby('YearMonth')[m_col].sum().reset_index()

                monthly['value'] = monthly[m_col]
                monthly['y'] = monthly[m_col]
                pct_col_name = 'DoD_Growth_%' if is_daily else 'MoM_Growth_%'
                monthly[pct_col_name] = (monthly[m_col].pct_change() * 100).round(2).fillna(0.0)
                monthly['MoM_Growth_%'] = monthly[pct_col_name]
                rolling_window = min(3, len(monthly))
                monthly['Rolling_Avg_3M'] = monthly[m_col].rolling(rolling_window, min_periods=1).mean().round(2)

                # Missing periods & irregular intervals detection (No silent interpolation)
                observed_periods = list(monthly['YearMonth'])
                missing_periods = []
                try:
                    if is_daily and len(t_df) >= 2:
                        start_d = t_df[date_col].min().date()
                        end_d = t_df[date_col].max().date()
                        full_calendar = [str(d.date()) for d in pd.date_range(start=start_d, end=end_d, freq='D')]
                        missing_periods = [p for p in full_calendar if p not in set(observed_periods)]
                    elif not is_daily and len(monthly) >= 2:
                        p_start = pd.Period(observed_periods[0], freq='M')
                        p_end = pd.Period(observed_periods[-1], freq='M')
                        full_calendar = [str(p) for p in pd.period_range(start=p_start, end=p_end, freq='M')]
                        missing_periods = [p for p in full_calendar if p not in set(observed_periods)]
                except Exception:
                    missing_periods = []

                irregular_intervals = len(missing_periods) > 0
                interp_warning = "Observed-only data used; missing periods are retained without artificial interpolation." if irregular_intervals else None

                # Volatility (Coefficient of Variation)
                mean_val = float(monthly[m_col].mean())
                std_val = float(monthly[m_col].std()) if len(monthly) > 1 else 0.0
                volatility = round(std_val / (abs(mean_val) + 1e-9), 4) if mean_val != 0 else 0.0

                m_const = compute_variable_constancy(df[m_col]) if m_col in df.columns else {"is_constant": False}
                if m_const["is_constant"]:
                    overall_trend = "constant"
                    const_stmt = generate_constant_variable_statement(m_col, m_const, unit=sem_layer.get(m_col, {}).get("unit", ""), time_col=date_col, df=df)
                    time_series_analysis.update({
                        "date_column": date_col,
                        "measure_column": m_col,
                        "granularity": period_name,
                        "aggregation": "sum" if is_add else "mean",
                        "overall_trend": overall_trend,
                        "is_constant": True,
                        "constant_statement": const_stmt,
                        "takeaway": const_stmt,
                        "monthly_trend": monthly.to_dict(orient='records'),
                        "rolling_average_3m": monthly['Rolling_Avg_3M'].tolist(),
                        "peak_period": str(monthly.iloc[0]['YearMonth']) if not monthly.empty else "N/A",
                        "lowest_period": str(monthly.iloc[-1]['YearMonth']) if not monthly.empty else "N/A",
                        "anomalies": [],
                        "seasonality_detected": False,
                        "structural_break_detected": False,
                        "missing_periods": missing_periods,
                        "duplicate_timestamps": duplicate_ts_cnt,
                        "irregular_intervals": irregular_intervals,
                        "interpolation_enabled": False,
                        "interpolation_method": None,
                        "interpolated_periods": [],
                        "original_observations": len(monthly),
                        "interpolation_warning": interp_warning,
                        "volatility": 0.0
                    })
                else:
                    anomalies = monthly[abs(monthly[m_col] - mean_val) > (2 * std_val)]['YearMonth'].tolist() if (std_val > 0 and len(monthly) >= 4) else []

                    # Linear slope and trend direction
                    if len(monthly) >= 2:
                        slope = float(np.polyfit(np.arange(len(monthly)), monthly[m_col].values, 1)[0])
                        overall_trend = "upward" if slope > 0.001 else ("downward" if slope < -0.001 else "stable")
                    else:
                        slope = 0.0
                        overall_trend = "stable"

                    # Seasonality: require >= 2 repeated cycles with significant autocorrelation
                    series_vals = pd.Series(monthly[m_col].values)
                    seasonality_detected = False
                    seasonality_period = None
                    seasonality_strength = 0.0
                    seasonality_method = "Autocorrelation"
                    seasonality_conf = 0.0

                    if is_daily and len(monthly) >= 14:
                        ac_7 = float(series_vals.autocorr(lag=7))
                        if not math.isnan(ac_7) and ac_7 >= 0.35:
                            seasonality_detected = True
                            seasonality_period = 7
                            seasonality_strength = round(ac_7, 3)
                            seasonality_method = "Autocorrelation (Lag 7 - Weekly Cycle)"
                            seasonality_conf = round(min(1.0, ac_7 + 0.1), 2)
                    elif not is_daily and len(monthly) >= 24:
                        ac_12 = float(series_vals.autocorr(lag=12))
                        if not math.isnan(ac_12) and ac_12 >= 0.35:
                            seasonality_detected = True
                            seasonality_period = 12
                            seasonality_strength = round(ac_12, 3)
                            seasonality_method = "Autocorrelation (Lag 12 - Annual Cycle)"
                            seasonality_conf = round(min(1.0, ac_12 + 0.1), 2)
                    elif not is_daily and len(monthly) >= 8:
                        ac_4 = float(series_vals.autocorr(lag=4))
                        if not math.isnan(ac_4) and ac_4 >= 0.40:
                            seasonality_detected = True
                            seasonality_period = 4
                            seasonality_strength = round(ac_4, 3)
                            seasonality_method = "Autocorrelation (Lag 4 - Quarterly Cycle)"
                            seasonality_conf = round(min(1.0, ac_4 + 0.1), 2)

                    # Structural break detection: test for shift in mean across split points
                    structural_break_detected = False
                    structural_break_period = None
                    structural_break_stmt = None
                    if len(monthly) >= 12 and std_val > 0:
                        best_shift = 0.0
                        best_k = None
                        best_pre = 0.0
                        best_post = 0.0
                        for k in range(3, len(monthly) - 3):
                            pre = monthly[m_col].iloc[:k]
                            post = monthly[m_col].iloc[k:]
                            pre_m, post_m = float(pre.mean()), float(post.mean())
                            pre_v, post_v = float(pre.var()), float(post.var())
                            pooled = math.sqrt(((len(pre) - 1) * pre_v + (len(post) - 1) * post_v) / max(len(monthly) - 2, 1))
                            if pooled > 0:
                                shift_score = abs(post_m - pre_m) / pooled
                                if shift_score > best_shift:
                                    best_shift = shift_score
                                    best_k = k
                                    best_pre = pre_m
                                    best_post = post_m
                        if best_shift >= 2.0 and best_k is not None:
                            structural_break_detected = True
                            structural_break_period = str(monthly.iloc[best_k]['YearMonth'])
                            structural_break_stmt = f"Potential structural change detected around {structural_break_period} (mean shifted from {best_pre:.2f} to {best_post:.2f}; shift score: {best_shift:.2f})."

                    n_periods = len(monthly)
                    y_vals = monthly[m_col].values
                    x_idx = np.arange(n_periods)
                    if n_periods < 3:
                        temporal_class = "INSUFFICIENT_OBSERVATIONS"
                        r_squared = 0.0
                        direction_reversals = 0
                    else:
                        diffs = np.diff(y_vals)
                        direction_reversals = int(np.sum(diffs[:-1] * diffs[1:] < 0)) if len(diffs) > 1 else 0
                        corr_mat = np.corrcoef(x_idx, y_vals)
                        r_squared = float(corr_mat[0, 1] ** 2) if not np.isnan(corr_mat[0, 1]) else 0.0

                        if slope > 0.001 and r_squared >= 0.65 and direction_reversals <= 1:
                            temporal_class = "INCREASING"
                        elif slope < -0.001 and r_squared >= 0.65 and direction_reversals <= 1:
                            temporal_class = "DECREASING"
                        elif structural_break_detected:
                            temporal_class = "STRUCTURAL_BREAK"
                        elif seasonality_detected:
                            temporal_class = "SEASONAL"
                        elif volatility > 0.40 and direction_reversals >= n_periods // 3:
                            temporal_class = "VOLATILE"
                        elif volatility <= 0.05:
                            temporal_class = "STABLE"
                        else:
                            temporal_class = "FLUCTUATING"

                    max_idx = monthly[m_col].idxmax()
                    min_idx = monthly[m_col].idxmin()
                    max_period = str(monthly.loc[max_idx]['YearMonth'])
                    max_val = float(monthly.loc[max_idx][m_col])
                    min_period = str(monthly.loc[min_idx]['YearMonth'])
                    min_val = float(monthly.loc[min_idx][m_col])

                    action_verb = {
                        "INCREASING": "trended upward",
                        "DECREASING": "trended downward",
                        "STABLE": "remained stable",
                        "VOLATILE": "exhibited high volatility",
                        "FLUCTUATING": "fluctuated",
                        "SEASONAL": "exhibited seasonal variation",
                        "STRUCTURAL_BREAK": "exhibited a structural shift",
                        "INSUFFICIENT_OBSERVATIONS": "contains limited observations"
                    }.get(temporal_class, "fluctuated")

                    takeaway_text = (
                        f"The measured variable '{m_col}' {action_verb} across {n_periods} recorded periods, "
                        f"with a maximum of {max_val:,.2f} in period {max_period} and minimum of {min_val:,.2f} in period {min_period}."
                    )
                    if missing_periods:
                        takeaway_text += f" Irregular time intervals detected with {len(missing_periods)} missing calendar period(s) between {missing_periods[0]} and {missing_periods[-1]}."

                    time_series_analysis.update({
                        "date_column": date_col,
                        "measure_column": m_col,
                        "granularity": period_name,
                        "aggregation": "sum" if is_add else "mean",
                        "overall_trend": overall_trend,
                        "temporal_classification": temporal_class,
                        "r_squared": round(r_squared, 4),
                        "direction_reversals": direction_reversals,
                        "slope": round(slope, 4),
                        "is_constant": False,
                        "monthly_trend": monthly.to_dict(orient='records'),
                        "rolling_average_3m": monthly['Rolling_Avg_3M'].tolist(),
                        "peak_period": max_period,
                        "lowest_period": min_period,
                        "anomalies": anomalies,
                        "volatility": volatility,
                        "missing_periods": missing_periods,
                        "duplicate_timestamps": duplicate_ts_cnt,
                        "irregular_intervals": irregular_intervals,
                        "interpolation_enabled": False,
                        "interpolation_method": None,
                        "interpolated_periods": [],
                        "original_observations": len(monthly),
                        "interpolation_warning": interp_warning,
                        "seasonality_detected": seasonality_detected,
                        "seasonality_period": seasonality_period,
                        "seasonality_strength": seasonality_strength,
                        "seasonality_method": seasonality_method,
                        "seasonality_confidence": seasonality_conf,
                        "structural_break_detected": structural_break_detected,
                        "structural_break_period": structural_break_period,
                        "structural_break_statement": structural_break_stmt,
                        "takeaway": takeaway_text
                    })
        except Exception as e:
            time_series_analysis = {"status": "error", "reason": str(e)}

    # 6. Histograms (Eligible Non-Constant Measures Only)
    histograms = {}
    for col in eligible_measures[:6]:
        if compute_variable_constancy(df[col])["is_constant"]:
            continue
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if not s.empty and s.nunique() >= 2:
            num_bins = min(8, max(3, s.nunique()))
            counts, bin_edges = np.histogram(s, bins=num_bins)
            histograms[col] = {
                "bin_labels": [f"{round(bin_edges[i],1)}-{round(bin_edges[i+1],1)}" for i in range(len(counts))],
                "counts": [int(c) for c in counts],
                "skewness": numeric_summary.get(col, {}).get("skewness", 0.0),
                "outliers_count": numeric_summary.get(col, {}).get("outliers_count", 0)
            }

    # 7. Scatter Plots with Trendline & Pearson r (Eligible Non-Constant Pairs Only)
    scatter_plots = []
    non_const_meas = [m for m in eligible_measures if not compute_variable_constancy(df[m])["is_constant"]]
    if len(non_const_meas) >= 2:
        if relationships.get("strong_associations"):
            best_assoc = relationships["strong_associations"][0]
            c1, c2 = best_assoc["metric_1"], best_assoc["metric_2"]
        else:
            c1, c2 = non_const_meas[0], non_const_meas[1]

        s1, s2 = pd.to_numeric(df[c1], errors='coerce'), pd.to_numeric(df[c2], errors='coerce')
        valid = pd.DataFrame({c1: s1, c2: s2}).dropna()
        if len(valid) >= 3 and valid[c1].std() > 0 and valid[c2].std() > 0:
            r_val = round(float(valid[c1].corr(valid[c2])), 3)
            m, b = np.polyfit(valid[c1], valid[c2], 1)
            scatter_plots.append({
                "metric_x": c1,
                "metric_y": c2,
                "pearson_r": r_val,
                "trend_slope": round(float(m), 4),
                "trend_intercept": round(float(b), 4),
                "data": [
                    {
                        "category": f"Record #{idx+1}",
                        "x": round(float(r[c1]), 2),
                        "y": round(float(r[c2]), 2),
                        c1: round(float(r[c1]), 2),
                        c2: round(float(r[c2]), 2)
                    }
                    for idx, r in valid.head(100).iterrows()
                ]
            })

    # 8. Box Plots (Non-Constant Metrics Only)
    box_plots = {}
    if dimensions and non_const_meas:
        dim_c, num_c = dimensions[0], non_const_meas[0]
        grouped = df.groupby(dim_c)[num_c]
        bp_dict = {}
        for name, group in grouped:
            g_clean = pd.to_numeric(group, errors='coerce').astype(float).dropna()
            if len(g_clean) >= 2 and g_clean.std() > 0:
                q1 = float(g_clean.quantile(0.25))
                med = float(g_clean.median())
                q3 = float(g_clean.quantile(0.75))
                iqr = q3 - q1
                min_v = float(max(g_clean.min(), q1 - 1.5 * iqr))
                max_v = float(min(g_clean.max(), q3 + 1.5 * iqr))
                outliers = [float(x) for x in g_clean if x < min_v or x > max_v]
                bp_dict[str(name)] = {
                    "min": round(min_v, 2),
                    "q1": round(q1, 2),
                    "median": round(med, 2),
                    "q3": round(q3, 2),
                    "max": round(max_v, 2),
                    "outliers_count": len(outliers)
                }
        if bp_dict:
            box_plots[f"{num_c}_by_{dim_c}"] = bp_dict

    # 9. Waterfall Chart
    waterfall_chart = generate_waterfall_chart(df, sem_layer)

    # 10. Stacked Bar Charts
    stacked_bar_charts = generate_stacked_bar_charts(df, sem_layer)

    # 11. Recommended Visualizations (Dynamic Chart Selection Mix)
    visualizations = recommend_visualizations(df, sem_layer)

    # 12. Advanced Statistical Analytics: PCA, Multi-Method Outliers & ANOVA Hypothesis Testing
    try:
        from services.statistical_analysis import compute_pca, compute_advanced_outliers, compute_bivariate_hypothesis_tests
        pca_analysis = compute_pca(df, sem_layer=sem_layer)
        advanced_outliers = compute_advanced_outliers(df, sem_layer=sem_layer)
        group_hypothesis_tests = compute_bivariate_hypothesis_tests(df, sem_layer)
    except Exception:
        pca_analysis = {"is_valid": False, "reason": "Advanced statistical module unavailable"}
        advanced_outliers = {"methods": ["IQR"], "columns": {}}
        group_hypothesis_tests = []

    # 13. Operational 5-Question Industrial EDA Framework (Prompt Section 3 & 8)
    operational_eda = build_operational_eda(
        df=df,
        sem_layer=sem_layer,
        num_summary=numeric_summary,
        cat_summary=categorical_summary,
        ts_analysis=time_series_analysis,
        relationships=relationships
    )

    return {
        "domain": domain,
        "overview": overview,
        "operational_eda": operational_eda,
        "numeric_summary": numeric_summary,
        "identifier_summary": identifier_summary,
        "categorical_summary": categorical_summary,
        "relationships": relationships,
        "correlation_matrix": relationships.get("pearson_matrix", {}),
        "time_series_analysis": time_series_analysis,
        "time_series": time_series_analysis,
        "histograms": histograms,
        "scatter_plots": scatter_plots,
        "box_plots": box_plots,
        "waterfall_chart": waterfall_chart,
        "stacked_bar_charts": stacked_bar_charts,
        "recommended_visualizations": visualizations,
        "recommended_charts": visualizations,
        "pca_analysis": pca_analysis,
        "advanced_outliers": advanced_outliers,
        "group_hypothesis_tests": group_hypothesis_tests
    }


def build_operational_eda(df: pd.DataFrame, sem_layer: dict, num_summary: dict, cat_summary: dict, ts_analysis: dict, relationships: dict) -> dict:
    """
    Constructs the operational 5-question EDA layer (Prompt Section 3 & 8).
    Answers:
    1. What is happening?
    2. Where is it happening?
    3. Why might it be happening?
    4. What is lacking or deviating?
    5. What should we do next?
    """
    from engine.profiler import detect_observation_grain
    grain_info = detect_observation_grain(df, sem_layer)
    total_records = len(df)

    # 1. WHAT IS HAPPENING?
    time_span = "Current Validated Dataset"
    if ts_analysis and ts_analysis.get("status") == "sufficient_data":
        pk = ts_analysis.get("peak_period")
        lw = ts_analysis.get("lowest_period")
        gran = ts_analysis.get("granularity", "Period")
        trend = ts_analysis.get("overall_trend", "stable")
        time_span = f"Recorded periods from {lw} to {pk} ({gran} intervals, overall trend: {trend})"

    key_kpis = []
    for col, stats in list(num_summary.items())[:5]:
        unit = sem_layer.get(col, {}).get("unit", "")
        unit_str = f" {unit}".rstrip() if unit and unit != "unknown" else ""
        key_kpis.append({
            "name": col,
            "mean": stats.get("mean"),
            "median": stats.get("median"),
            "min": stats.get("min"),
            "max": stats.get("max"),
            "unit": unit,
            "display": f"Mean: {stats.get('mean')}{unit_str} | Median: {stats.get('median')}{unit_str} (Range: [{stats.get('min')}, {stats.get('max')}])"
        })

    if key_kpis:
        top_kpi = key_kpis[0]
        what_summary = (
            f"Dataset encompasses {total_records:,} verified records at an observation grain of '{grain_info.get('grain_description')}'. "
            f"Primary metric '{top_kpi['name']}' demonstrates central tendency of {top_kpi['mean']} (median: {top_kpi['median']}). "
            f"Timeframe: {time_span}."
        )
    else:
        what_summary = f"Dataset encompasses {total_records:,} verified records at an observation grain of '{grain_info.get('grain_description')}'. Categorical dimensions dominate."

    what_is_happening = {
        "summary": what_summary,
        "observation_grain": grain_info.get("grain_description"),
        "grain_type": grain_info.get("grain_type"),
        "total_records": total_records,
        "time_span": time_span,
        "key_kpis": key_kpis
    }

    # 2. WHERE IS IT HAPPENING?
    group_comparisons = []
    highest_performing = None
    lowest_performing = None
    pareto_findings = []

    for dim, c_stats in list(cat_summary.items())[:4]:
        top_counts = c_stats.get("top_counts", {})
        top_pcts = c_stats.get("top_percentages", {})
        dominant_cat = c_stats.get("top_category")
        dominant_pct = c_stats.get("top_category_pct")
        
        group_comparisons.append({
            "dimension": dim,
            "unique_categories_count": c_stats.get("unique_categories"),
            "dominant_segment": dominant_cat,
            "dominant_share_pct": dominant_pct,
            "breakdown": [{"category": str(k), "count": int(v), "percentage": top_pcts.get(k, 0.0)} for k, v in list(top_counts.items())[:6]]
        })

        if top_counts:
            sorted_items = sorted(top_counts.items(), key=lambda x: x[1], reverse=True)
            if not highest_performing:
                highest_performing = {"dimension": dim, "segment": sorted_items[0][0], "count": sorted_items[0][1], "share_pct": dominant_pct}
            if len(sorted_items) > 1 and not lowest_performing:
                lowest_performing = {"dimension": dim, "segment": sorted_items[-1][0], "count": sorted_items[-1][1]}

            if len(sorted_items) >= 2:
                top_2_sum = sorted_items[0][1] + sorted_items[1][1]
                top_2_pct = round((top_2_sum / max(total_records, 1)) * 100, 1)
                if top_2_pct >= 60.0:
                    pareto_findings.append(f"Top 2 categories in '{dim}' ('{sorted_items[0][0]}' and '{sorted_items[1][0]}') represent {top_2_pct}% of total recorded observations.")

    # Check for critical safety / fatal case hotspots across plant
    impact_cols = [c for c in df.columns if any(k in c.lower() for k in ['impact', 'severity', 'incident', 'potential', 'classification', 'injury'])]
    loc_cols = [c for c in df.columns if any(k in c.lower() for k in ['location', 'plant', 'area', 'unit', 'site', 'facility', 'department'])]
    has_fatal = False
    top_fatal_loc = ""
    top_fatal_cnt = 0
    tot_fatal_cnt = 0
    if impact_cols and loc_cols:
        fatal_m = df[impact_cols[0]].fillna('').astype(str).str.contains('fatal', case=False)
        tot_fatal_cnt = int(fatal_m.sum())
        if tot_fatal_cnt > 0:
            has_fatal = True
            fatal_locs = df[fatal_m][loc_cols[0]].dropna().astype(str).value_counts()
            if len(fatal_locs) > 0:
                top_fatal_loc = fatal_locs.index[0]
                top_fatal_cnt = int(fatal_locs.iloc[0])
                breakdown_text = ", ".join([f"{loc}: {cnt}" for loc, cnt in fatal_locs.items()])
                pareto_findings.insert(0, f"CRITICAL SAFETY HOTSPOT: {tot_fatal_cnt} Fatal case(s) detected across plant: {breakdown_text}. Primary concentration at '{top_fatal_loc}' ({top_fatal_cnt} cases).")

    where_is_it_happening = {
        "group_comparisons": group_comparisons,
        "highest_concentration_unit": highest_performing,
        "lowest_concentration_unit": lowest_performing,
        "pareto_findings": pareto_findings
    }

    # 3. WHY MIGHT IT BE HAPPENING?
    contributing_factors = []
    strategic_assocs = relationships.get("strategic_associations", [])
    for a in strategic_assocs[:4]:
        contributing_factors.append({
            "variable_1": a.get("metric_1"),
            "variable_2": a.get("metric_2"),
            "association_strength": a.get("strength"),
            "direction": a.get("direction"),
            "correlation_coefficient": a.get("pearson_r"),
            "interpretation": a.get("interpretation")
        })

    why_might_it_be_happening = {
        "contributing_factors": contributing_factors,
        "causality_disclaimer": "Observed correlations reflect statistical co-variance in available sample data and do not establish engineering causality.",
        "operational_hypotheses": [
            f"Observed variation in {a.get('metric_1')} corresponds with changes in {a.get('metric_2')} (r = {a.get('pearson_r')}); investigate operational alignment between these processes."
            for a in strategic_assocs[:2]
        ] if strategic_assocs else ["Insufficient continuous metric pairs to compute co-variance associations."]
    }

    # 4. WHAT IS LACKING OR DEVIATING?
    baseline_exceptions = []
    for col, stats in list(num_summary.items())[:4]:
        outliers_cnt = stats.get("outliers_count", 0)
        outliers_pct = stats.get("outliers_percentage", 0.0)
        if outliers_cnt > 0:
            baseline_exceptions.append({
                "metric": col,
                "deviating_records_count": outliers_cnt,
                "deviating_percentage": outliers_pct,
                "normal_operating_range": f"[{stats.get('iqr_lower_bound')}, {stats.get('iqr_upper_bound')}]",
                "finding": f"{outliers_cnt} record(s) ({outliers_pct}%) deviate significantly from the empirical operating baseline for '{col}'."
            })

    recurring_issues = []
    if has_fatal:
        recurring_issues.append(f"Critical Safety Exception: {tot_fatal_cnt} Fatal cases recorded across plant locations requiring immediate supervisor inspection and CAPA review.")

    for dim, c_stats in list(cat_summary.items())[:3]:
        dom = c_stats.get("top_category")
        pct = c_stats.get("top_category_pct", 0.0)
        if pct >= 40.0:
            recurring_issues.append(f"Recurring concentration in '{dim}': '{dom}' accounts for {pct}% of all recorded events.")

    what_is_lacking_or_deviating = {
        "baseline_exceptions": baseline_exceptions,
        "recurring_issues": recurring_issues,
        "missing_records_summary": f"Detected across {total_records} records with empirical Tukey fences and variance checks."
    }

    # 5. WHAT SHOULD WE DO NEXT?
    actions = []
    if has_fatal and top_fatal_loc:
        actions.append({
            "priority": "HIGH",
            "action": f"Immediate Safety Inspection at '{top_fatal_loc}'",
            "objective": f"Audit plant conditions and implement controls for {top_fatal_cnt} Fatal case(s) concentrated in this area (out of {tot_fatal_cnt} plant-wide)."
        })
    if baseline_exceptions:
        top_ex = baseline_exceptions[0]
        actions.append({
            "priority": "HIGH",
            "action": f"Investigate root cause of {top_ex['deviating_records_count']} outlier deviation(s) in '{top_ex['metric']}'.",
            "objective": f"Restore '{top_ex['metric']}' measurements to normal baseline [{top_ex['normal_operating_range']}]."
        })
    if pareto_findings:
        actions.append({
            "priority": "MEDIUM",
            "action": "Focus operational mitigation on dominant category segments identified in Pareto distribution.",
            "objective": "Target the top 20% segment drivers responsible for majority volume concentration."
        })
    if not actions:
        actions.append({
            "priority": "MEDIUM",
            "action": "Establish permanent engineering operating limits and recurring shift monitoring.",
            "objective": "Ensure ongoing data logging integrity and early deviation warning."
        })

    what_should_we_do_next = {
        "recommended_operational_actions": actions
    }

    return {
        "what_is_happening": what_is_happening,
        "where_is_it_happening": where_is_it_happening,
        "why_might_it_be_happening": why_might_it_be_happening,
        "what_is_lacking_or_deviating": what_is_lacking_or_deviating,
        "what_should_we_do_next": what_should_we_do_next
    }

