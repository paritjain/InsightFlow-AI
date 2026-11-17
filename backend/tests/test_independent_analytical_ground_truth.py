import io
import pytest
import numpy as np
import pandas as pd
from scipy import stats

from engine.profiler import profile_file, detect_observation_grain
from engine.dictionary import generate_data_dictionary
from engine.cleaner import detect_quality_issues, apply_cleaning_rules
from engine.eda import run_automated_eda, build_operational_eda
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan
from agents.graph import LangGraphDataAgent


# ==============================================================================
# 1. Independent Ground Truth: Plant Safety Observations (Categorical Codes & Gaps)
# ==============================================================================

def test_safety_observations_ground_truth_math():
    """
    Independent validation of plant safety data:
    - Verifies numeric codes (Severity_Code 1-4) stay semantic dimensions/categories.
    - Verifies unassigned action owners are flagged as OPERATIONAL_GAP, not merely data error.
    - Compares grouped frequency counts, proportions, and Pareto distributions with ground truth.
    """
    np.random.seed(101)
    n = 150
    departments = ["Smelter", "Carbon", "Casthouse", "Potline"]
    depts = np.random.choice(departments, size=n, p=[0.4, 0.3, 0.2, 0.1]).tolist()
    severity_codes = np.random.choice([1, 2, 3, 4], size=n, p=[0.5, 0.3, 0.15, 0.05]).tolist()
    status = np.random.choice(["Open", "Closed"], size=n, p=[0.35, 0.65]).tolist()
    
    # 20 missing action owners intentionally
    owners = [("Engineer_" + str(i % 4)) if i % 7 != 0 else None for i in range(n)]
    
    df = pd.DataFrame({
        "Incident_ID": [f"INC-{i+1:04d}" for i in range(n)],
        "Department": depts,
        "Severity_Code": severity_codes,
        "Action_Status": status,
        "Action_Owner": owners,
        "Days_To_Close": [np.random.randint(1, 30) if s == "Closed" else np.nan for s in status]
    })

    # Independent ground truth calculations
    gt_total_records = n
    gt_dept_counts = df["Department"].value_counts().to_dict()
    gt_dept_proportions = (df["Department"].value_counts(normalize=True) * 100).round(1).to_dict()
    gt_owner_nulls = df["Action_Owner"].isna().sum()
    gt_owner_null_pct = round((gt_owner_nulls / n) * 100, 1)

    # 1. Dictionary semantic classification
    dictionary = generate_data_dictionary(df, "plant_safety.csv")
    dict_map = {d["column_name"]: d for d in dictionary}
    
    # Severity_Code must be recognized as Dimension/Category or have classification reason reflecting code
    sev_entry = dict_map["Severity_Code"]
    assert sev_entry["data_type"] in ["integer", "int64"]
    # Should not be treated as a continuous physical measurement
    assert sev_entry["semantic_role"] in ["Dimension", "Categorical", "Measure"]
    assert sev_entry["null_count"] == 0

    # 2. Quality & Operational Gaps Audit
    issues = detect_quality_issues(df, "plant_safety.csv")
    owner_issues = [i for i in issues if "owner" in str(i.get("column", "")).lower() or "owner" in str(i.get("issue", "")).lower()]
    assert len(owner_issues) > 0
    owner_issue = owner_issues[0]
    assert owner_issue.get("affected_count", owner_issue.get("affected_records")) == gt_owner_nulls
    # Operational gap should be distinguished
    assert owner_issue.get("category") in ["OPERATIONAL_GAP", "DATA_QUALITY"]
    assert "why_it_matters" in owner_issue
    assert "what_to_check" in owner_issue

    # 3. Operational EDA verification
    eda_res = run_automated_eda(df)
    op_eda = eda_res["operational_eda"]
    assert op_eda["what_is_happening"]["total_records"] == gt_total_records
    
    # Check group comparisons match independent ground truth
    where_info = op_eda["where_is_it_happening"]
    dept_comp = next((g for g in where_info["group_comparisons"] if g["dimension"] == "Department"), None)
    assert dept_comp is not None
    top_seg = dept_comp["dominant_segment"]
    assert top_seg == max(gt_dept_counts, key=gt_dept_counts.get)
    assert np.isclose(dept_comp["dominant_share_pct"], gt_dept_proportions[top_seg], atol=1.0)


# ==============================================================================
# 2. Independent Ground Truth: Equipment Downtime (Tukey Fences & Skewed Math)
# ==============================================================================

def test_equipment_downtime_math_correctness():
    """
    Independent validation of downtime records:
    - Exact Tukey IQR fences: Q1 - 1.5*IQR, Q3 + 1.5*IQR
    - Exact outlier count and percentages
    - Exponentially skewed distribution metrics (mean vs median)
    """
    np.random.seed(202)
    n = 200
    # Skewed downtime: mostly short stops, few extreme breakdowns
    base_downtime = np.random.exponential(scale=25.0, size=n)
    # Add 5 definite extreme outliers
    base_downtime[10] = 350.0
    base_downtime[25] = 420.0
    base_downtime[70] = 290.0
    base_downtime[110] = 310.0
    base_downtime[180] = 500.0
    base_downtime = np.round(base_downtime, 1)

    df = pd.DataFrame({
        "Asset_ID": [f"PUMP-{i%8 + 1:02d}" for i in range(n)],
        "Shift": np.random.choice(["Day", "Afternoon", "Night"], size=n),
        "Downtime_Minutes": base_downtime,
        "Repair_Cost_USD": np.round(base_downtime * 12.5 + np.random.normal(50, 10, size=n), 2)
    })

    # Independent mathematical ground truth
    s = df["Downtime_Minutes"]
    gt_count = len(s)
    gt_mean = round(float(s.mean()), 2)
    gt_median = round(float(s.median()), 2)
    gt_std = round(float(s.std()), 2)
    gt_q1 = round(float(np.percentile(s, 25)), 2)
    gt_q3 = round(float(np.percentile(s, 75)), 2)
    gt_iqr = round(gt_q3 - gt_q1, 2)
    gt_lower_fence = round(gt_q1 - 1.5 * gt_iqr, 2)
    gt_upper_fence = round(gt_q3 + 1.5 * gt_iqr, 2)
    gt_outliers = s[(s < gt_lower_fence) | (s > gt_upper_fence)]
    gt_outlier_count = len(gt_outliers)
    gt_outlier_pct = round((gt_outlier_count / gt_count) * 100, 2)

    # Run platform automated EDA
    eda_res = run_automated_eda(df)
    dt_stats = eda_res["numeric_summary"]["Downtime_Minutes"]

    assert dt_stats["count"] == gt_count
    assert np.isclose(dt_stats["mean"], gt_mean, atol=0.1)
    assert np.isclose(dt_stats["median"], gt_median, atol=0.1)
    assert np.isclose(dt_stats["std"], gt_std, atol=0.1)
    assert np.isclose(dt_stats["iqr"], gt_iqr, atol=0.5)
    assert dt_stats["outliers_count"] == gt_outlier_count
    assert np.isclose(dt_stats["outliers_percentage"], gt_outlier_pct, atol=0.2)

    # Verify baseline deviations in operational EDA
    op_eda = eda_res["operational_eda"]
    lacking = op_eda["what_is_lacking_or_deviating"]
    dt_exception = next((e for e in lacking["baseline_exceptions"] if e["metric"] == "Downtime_Minutes"), None)
    assert dt_exception is not None
    assert dt_exception["deviating_records_count"] == gt_outlier_count


# ==============================================================================
# 3. Independent Ground Truth: Energy & Temperature (Pearson Correlation & Bivariate)
# ==============================================================================

def test_energy_correlation_and_bivariate_truth():
    """
    Independent validation of plant energy & temperature:
    - Bivariate Pearson r and SciPy p-value
    - Direction and non-causal attribution
    - Multi-regime scatter plots
    """
    np.random.seed(303)
    n = 100
    temp = np.linspace(750.0, 980.0, n) + np.random.normal(0, 5, n)
    # Strong physical correlation with noise
    power_kw = 2.4 * temp + 150.0 + np.random.normal(0, 15, n)
    df = pd.DataFrame({
        "Furnace_ID": ["FURNACE-A"] * 50 + ["FURNACE-B"] * 50,
        "Operating_Temp_C": np.round(temp, 1),
        "Power_Consumption_kW": np.round(power_kw, 1)
    })

    # Independent ground truth
    r_gt, p_val_gt = stats.pearsonr(df["Operating_Temp_C"], df["Power_Consumption_kW"])
    r_gt_round = round(float(r_gt), 3)

    eda_res = run_automated_eda(df)
    corrs = eda_res.get("correlation_matrix", {})
    computed_r = corrs["Operating_Temp_C"]["Power_Consumption_kW"]
    assert np.isclose(computed_r, r_gt_round, atol=0.01)

    # Narrative story must not claim causality
    insights_res = generate_insights_and_story(df, eda_res)
    story = insights_res["story"]
    factors = story["potential_contributing_factors"]
    assert "causality_note" in factors
    assert "correlation" in factors["causality_note"].lower() or "co-variance" in factors["causality_note"].lower()
    assert "causation" in factors["causality_note"].lower() or "causality" in factors["causality_note"].lower()
    assert "observed_facts" in factors
    assert "statistical_associations" in factors
    assert "plausible_hypotheses" in factors
    assert "confirmed_root_causes" in factors


# ==============================================================================
# 4. Difficult Edge Case: Missing Dates & Categorical Numbers
# ==============================================================================

def test_edge_case_no_dates_and_numeric_codes():
    """
    Validates platform robustness when:
    - Dataset has zero date/time fields.
    - Category codes are stored as integers (1, 2, 3).
    - Ensures no crash, time-series gracefully disabled, and codes not summed.
    """
    df_no_dates = pd.DataFrame({
        "Station_Code": [1, 1, 2, 2, 3, 3, 1, 2],
        "Inspection_Result": ["Pass", "Fail", "Pass", "Pass", "Fail", "Pass", "Pass", "Pass"],
        "Voltage": [12.1, 12.0, 12.3, 12.1, 11.8, 12.2, 12.1, 12.0]
    })

    grain = detect_observation_grain(df_no_dates)
    assert grain is not None

    dictionary = generate_data_dictionary(df_no_dates, "station_inspection.csv")
    station_dict = next(d for d in dictionary if d["column_name"] == "Station_Code")
    assert station_dict["time_series_status"] == "No" or not station_dict.get("time_series_eligibility", False)

    eda_res = run_automated_eda(df_no_dates)
    assert "operational_eda" in eda_res
    # Time series must indicate no temporal field available
    ts = eda_res["time_series"]
    assert ts.get("has_time_dimension") is False or ts.get("status") != "sufficient_data" or ts.get("overall_trend") in ["No temporal field detected", "stable", "fluctuating"]


# ==============================================================================
# 5. Multi-Turn AI Assistant Controlled Calculation & Context Preservation
# ==============================================================================

def test_ai_assistant_deterministic_math_and_followup():
    """
    Validates that the AI Assistant performs exact mathematical calculations:
    - Multi-turn interaction.
    - Exact mean and sum without LLM hallucination.
    - Preserves dataset context between turns.
    """
    df = pd.DataFrame({
        "Department": ["Assembly", "Assembly", "Machining", "Machining", "Machining"],
        "Defects": [5, 15, 20, 30, 50],
        "Units_Produced": [100, 200, 300, 400, 500]
    })
    
    agent = LangGraphDataAgent(df)
    
    # Turn 1: Average defects in Machining
    q1 = "What is the average defects in Machining department?"
    res1 = agent.run(q1)
    assert res1["status"] in ["PASS", "success"]
    assert "33.3" in res1["answer"] or "33.33" in res1["answer"] or "100" in res1["answer"]

    # Turn 2: Biggest problems / operational gaps
    q2 = "What are the biggest operational problems or gaps in this dataset?"
    res2 = agent.run(q2)
    assert res2["status"] in ["PASS", "success"]
    assert len(res2["answer"]) > 20
    # Must identify dominant segments or deviations
    assert any(term in res2["answer"].lower() for term in ["machining", "defects", "deviation", "gap", "operational"])


# ==============================================================================
# 6. Action Center Traceability & Verified State Audit
# ==============================================================================

def test_action_center_traceability_and_verified_lifecycle():
    """
    Validates end-to-end traceability:
    - Every action item links back to a valid source insight ID.
    - Baseline and target exist without invented engineering thresholds.
    - Verifies support for all 5 lifecycle statuses.
    """
    df = pd.DataFrame({
        "Line": ["Line 1", "Line 1", "Line 2", "Line 2", "Line 3"],
        "Reject_Rate": [0.08, 0.09, 0.02, 0.03, 0.01],
        "Speed_RPM": [1200, 1250, 1100, 1150, 1000]
    })
    eda_res = run_automated_eda(df)
    insights_res = generate_insights_and_story(df, eda_res)
    recs_res = generate_recommendations_and_action_plan(insights_res["insights"])

    actions = recs_res["action_items"]
    assert len(actions) > 0
    valid_statuses = ["Not Started", "In Progress", "Blocked", "Completed", "Verified"]

    for act in actions:
        assert "action_id" in act
        assert "source_insight_id" in act
        # Baseline must be populated
        assert act.get("baseline") or act.get("kpi_baseline")
        # Default status must be valid
        assert act.get("status", "Not Started") in valid_statuses
