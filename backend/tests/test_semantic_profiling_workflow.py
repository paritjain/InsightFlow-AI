import pytest
import pandas as pd
import numpy as np

from engine.profiler import infer_column_semantic_metadata, profile_file
from engine.eda import run_automated_eda
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan
from services.statistical_analysis import compute_pca, compute_advanced_outliers


def test_step1_semantic_role_assignment_and_heuristics():
    """
    Step 1: Verify semantic role mapping into:
    IDENTIFIER, CATEGORICAL DIMENSION, TEMPORAL, NUMERIC MEASURE.
    Verify identifier heuristics and eligibility flags.
    """
    df = pd.DataFrame({
        "CustomerID": [101, 102, 103, 104, 105],
        "order_num": ["ORD-1", "ORD-2", "ORD-3", "ORD-4", "ORD-5"],
        "account_code": ["ACC1", "ACC2", "ACC3", "ACC4", "ACC5"],
        "ZipCode": ["10001", "10002", "10003", "10004", "10005"],
        "SKU": ["SKU-A", "SKU-B", "SKU-C", "SKU-D", "SKU-E"],
        "high_unique_int": [991, 992, 993, 994, 995],  # 100% unique integer
        "Region": ["North", "South", "East", "West", "North"],
        "OrderDate": pd.date_range("2024-01-01", periods=5, freq="D"),
        "Sales": [150.0, 250.0, 350.0, 450.0, 550.0],
        "Quantity": [2, 5, 3, 8, 4]
    })

    # Test Identifier columns
    id_cols = ["CustomerID", "order_num", "account_code", "ZipCode", "SKU", "high_unique_int"]
    for col in id_cols:
        meta = infer_column_semantic_metadata(col, df[col], len(df))
        assert meta["is_identifier"] is True, f"Column '{col}' was not detected as IDENTIFIER"
        assert meta["aggregation_eligibility"] is False, f"Aggregation eligibility not False for '{col}'"
        assert meta["correlation_eligibility"] is False, f"Correlation eligibility not False for '{col}'"
        assert meta["suitable_for_correlation"] is False
        assert meta["suitable_for_sum"] is False

    # Test Categorical Dimension
    meta_region = infer_column_semantic_metadata("Region", df["Region"], len(df))
    assert meta_region["is_dimension"] is True
    assert meta_region["grouping_eligibility"] is True
    assert meta_region["aggregation_eligibility"] is False

    # Test Temporal
    meta_date = infer_column_semantic_metadata("OrderDate", df["OrderDate"], len(df))
    assert meta_date["is_time"] is True
    assert meta_date["time_series_eligibility"] is True

    # Test Numeric Measures
    for col in ["Sales", "Quantity"]:
        meta_num = infer_column_semantic_metadata(col, df[col], len(df))
        assert meta_num["is_measure"] is True
        assert meta_num["is_identifier"] is False
        assert meta_num["aggregation_eligibility"] is True
        assert meta_num["correlation_eligibility"] is True


def test_step2_constrained_eda_and_statistical_pipeline():
    """
    Step 2: Verify constrained calculations:
    - Descriptive stats computed ONLY for NUMERIC MEASURE
    - IDENTIFIERS report ONLY count, unique count, missing count, uniqueness percentage
    - Correlations ONLY between NUMERIC MEASURE columns
    - PCA feature space consists ONLY of standardized NUMERIC MEASURE features
    """
    df = pd.DataFrame({
        "Trans_ID": [1001, 1002, 1003, 1004, 1005, 1006],
        "Customer_Code": ["C1", "C2", "C3", "C4", "C5", "C6"],
        "Category": ["Electronics", "Clothing", "Electronics", "Home", "Clothing", "Home"],
        "Tx_Date": pd.date_range("2024-01-01", periods=6, freq="D"),
        "Revenue": [1200.0, 350.0, 800.0, 450.0, 290.0, 950.0],
        "Discount": [0.05, 0.10, 0.0, 0.15, 0.0, 0.20],
        "Profit": [300.0, 50.0, 200.0, 80.0, 40.0, 220.0]
    })

    eda_res = run_automated_eda(df)

    # 1. Descriptive stats: Mean, std, median, IQR ONLY for measures
    assert "Revenue" in eda_res["numeric_summary"]
    assert "Discount" in eda_res["numeric_summary"]
    assert "Profit" in eda_res["numeric_summary"]
    assert "Trans_ID" not in eda_res["numeric_summary"], "Trans_ID should NOT be in numeric_summary"
    assert "Customer_Code" not in eda_res["numeric_summary"]

    # 2. Identifier summary: Only total_count, unique_count, missing_count, uniqueness_percentage
    id_sum = eda_res.get("identifier_summary", {})
    assert "Trans_ID" in id_sum
    assert "total_count" in id_sum["Trans_ID"]
    assert "unique_count" in id_sum["Trans_ID"]
    assert "missing_count" in id_sum["Trans_ID"]
    assert "uniqueness_percentage" in id_sum["Trans_ID"]
    assert "mean" not in id_sum["Trans_ID"]
    assert "median" not in id_sum["Trans_ID"]
    assert "std" not in id_sum["Trans_ID"]
    assert "iqr" not in id_sum["Trans_ID"]

    # 3. Correlation Matrix: ONLY between numeric measures
    corr_matrix = eda_res["correlation_matrix"]
    assert "Trans_ID" not in corr_matrix
    assert "Customer_Code" not in corr_matrix
    for pair in eda_res["relationships"]["strong_associations"]:
        assert pair["metric_1"] in ["Revenue", "Discount", "Profit"]
        assert pair["metric_2"] in ["Revenue", "Discount", "Profit"]
        assert pair["metric_1"] != "Trans_ID"
        assert pair["metric_2"] != "Trans_ID"

    # 4. PCA: Feature space contains ONLY NUMERIC MEASURES
    pca_analysis = eda_res["pca_analysis"]
    assert pca_analysis["is_valid"] is True
    for feat in pca_analysis["features_analyzed"]:
        assert feat in ["Revenue", "Discount", "Profit"]
        assert feat not in ["Trans_ID", "Customer_Code", "Tx_Date", "Category"]


def test_step3_insight_and_recommendation_guardrails():
    """
    Step 3: Verify sanity rule and self-correction checks:
    - Never base recommendations on sum/avg/total of IDENTIFIER
    - Discard any action treating an ID as a measurable KPI
    """
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "category": ["A", "B", "A", "B", "A"],
        "revenue": [100.0, 200.0, 300.0, 400.0, 500.0]
    })
    eda_res = run_automated_eda(df)
    ins_res = generate_insights_and_story(df, eda_res)
    recs_res = generate_recommendations_and_action_plan(ins_res["insights"])

    for rec in recs_res["recommendations"]:
        kpi = str(rec.get("kpi", "")).lower()
        title = str(rec.get("title", "")).lower()
        target_m = str(rec.get("target_metric", "")).lower()

        # Ensure no recommendation targets an identifier
        assert "order_id" not in kpi
        assert "order_id" not in target_m
        assert "baseline volume for order_id" not in title
