import os
import sys
import io
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from main import app, BoundedProjectCache, MAX_UPLOAD_SIZE_BYTES
from services.statistical_analysis import compute_group_differences
from engine.eda import run_automated_eda
from agents.graph import LangGraphDataAgent, session_manager


# =====================================================================
# 1. Statistical Analysis Degrees of Freedom Tests
# =====================================================================

def test_anova_degrees_of_freedom_valid():
    """Verify df_between = k - 1 and df_within = N - k on valid multi-group data."""
    # 3 groups: A (5 obs), B (5 obs), C (5 obs) -> k=3, N=15
    # df_between = 3 - 1 = 2
    # df_within = 15 - 3 = 12
    df = pd.DataFrame({
        "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        "score": [10, 12, 11, 13, 12, 20, 22, 21, 23, 22, 30, 31, 32, 30, 33]
    })
    result = compute_group_differences(df, cat_col="group", num_col="score")
    assert result["status"] == "success"
    anova = result["anova"]
    assert anova["valid"] is True
    assert anova["df_between"] == 2
    assert anova["df_within"] == 12
    assert anova["k_groups"] == 3
    assert anova["n_observations"] == 15
    assert anova["f_statistic"] > 0
    assert anova["p_value"] < 0.05
    assert "F(2, 12)" in anova["interpretation"]

    # Kruskal-Wallis df = k - 1 = 2
    kruskal = result["kruskal_wallis"]
    assert kruskal["valid"] is True
    assert kruskal["kruskal_df"] == 2


def test_anova_degrees_of_freedom_with_missing_data():
    """Verify N calculation properly excludes missing values in categorical or numeric columns."""
    # 3 groups: A (5 obs), B (5 obs), C (5 obs with 2 NaNs) -> valid N = 13, k = 3
    # df_between = 3 - 1 = 2
    # df_within = 13 - 3 = 10
    df = pd.DataFrame({
        "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        "score": [10, 12, 11, 13, 12, 20, 22, 21, 23, 22, 30, None, np.nan, 30, 33]
    })
    result = compute_group_differences(df, cat_col="group", num_col="score")
    assert result["status"] == "success"
    anova = result["anova"]
    assert anova["valid"] is True
    assert anova["df_between"] == 2
    assert anova["df_within"] == 10
    assert anova["n_observations"] == 13
    assert anova["k_groups"] == 3
    assert "F(2, 10)" in anova["interpretation"]


def test_anova_insufficient_groups():
    """Verify graceful handling when fewer than 2 valid groups exist."""
    df_single = pd.DataFrame({
        "group": ["A", "A", "A", "A"],
        "score": [10, 11, 12, 13]
    })
    res_single = compute_group_differences(df_single, cat_col="group", num_col="score")
    assert res_single["status"] == "insufficient_groups"
    assert res_single["anova"]["valid"] is False
    assert "Fewer than 2 valid groups" in res_single["anova"]["interpretation"]


# =====================================================================
# 2. Empirical Numeric Mode Tests in EDA
# =====================================================================

def test_numeric_mode_single_mode():
    """Verify single empirical mode detection."""
    df = pd.DataFrame({
        "values": [1.0, 2.0, 2.0, 3.0, 4.0, 2.0]
    })
    eda = run_automated_eda(df)
    col_summary = eda["numeric_summary"]["values"]
    assert col_summary["mode"] == 2.0
    assert col_summary["is_multimodal"] is False
    assert col_summary["modes"] == [2.0]


def test_numeric_mode_multimodal():
    """Verify bimodal/multimodal empirical mode detection."""
    df = pd.DataFrame({
        "bimodal": [10.0, 10.0, 20.0, 20.0, 30.0]
    })
    eda = run_automated_eda(df)
    col_summary = eda["numeric_summary"]["bimodal"]
    assert col_summary["is_multimodal"] is True
    assert sorted(col_summary["modes"]) == [10.0, 20.0]
    assert col_summary["mode"] in [10.0, 20.0]


def test_numeric_mode_constant_and_missing():
    """Verify mode on constant column and column with NaNs."""
    df = pd.DataFrame({
        "constant": [42.0, 42.0, 42.0, 42.0, 42.0, 42.0],
        "with_nans": [5.0, np.nan, 5.0, 8.0, None, 5.0],
        "all_nans": pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], dtype=float)
    })
    eda = run_automated_eda(df)
    
    assert eda["numeric_summary"]["constant"]["mode"] == 42.0
    assert eda["numeric_summary"]["constant"]["is_multimodal"] is False
    
    assert eda["numeric_summary"]["with_nans"]["mode"] == 5.0
    assert eda["numeric_summary"]["with_nans"]["is_multimodal"] is False

    assert eda["numeric_summary"]["all_nans"]["mode"] is None
    assert eda["numeric_summary"]["all_nans"]["modes"] == []


# =====================================================================
# 3. Upload File Size Validation Tests
# =====================================================================

def test_upload_file_size_below_limit():
    """Verify uploading a normal-sized CSV file succeeds (200 OK)."""
    client = TestClient(app)
    csv_content = b"id,name,value\n1,Alpha,100\n2,Beta,200\n3,Gamma,300\n"
    files = [("files", ("small_dataset.csv", io.BytesIO(csv_content), "text/csv"))]
    resp = client.post("/api/projects/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "project_id" in data


def test_upload_file_size_above_limit_rejected_413():
    """Verify uploading a file exceeding 50 MB is rejected with 413 Payload Too Large."""
    client = TestClient(app)

    class OversizedStream(io.RawIOBase):
        """Streams more than 50 MB without allocating 50+ MB memory."""
        def __init__(self, total_bytes):
            self.total_bytes = total_bytes
            self.bytes_read = 0
            self._chunk = b"A" * 65536

        def read(self, size=-1):
            if self.bytes_read >= self.total_bytes:
                return b""
            chunk_size = len(self._chunk) if size < 0 else min(size, len(self._chunk))
            remaining = self.total_bytes - self.bytes_read
            to_send = min(chunk_size, remaining)
            self.bytes_read += to_send
            return self._chunk[:to_send]

    # 51 MB stream
    oversized_data = OversizedStream(51 * 1024 * 1024)
    files = [("files", ("oversized_data.csv", oversized_data, "text/csv"))]
    resp = client.post("/api/projects/upload", files=files)
    assert resp.status_code == 413
    assert "50 MB" in resp.json()["detail"]


# =====================================================================
# 4. LangGraph Orchestration and Tools Tests
# =====================================================================

def test_langgraph_agent_deterministic_tools():
    """Verify LangGraph StateGraph agent correctly routes and executes deterministic tools."""
    df = pd.DataFrame({
        "category": ["Electronics", "Clothing", "Home", "Electronics", "Home"],
        "revenue": [500.0, 150.0, 300.0, 750.0, 450.0],
        "units": [5, 2, 4, 8, 6]
    })
    eda_results = run_automated_eda(df)
    insights = [{"title": "Electronics Leads Revenue", "description": "High value items drive sales"}]

    agent = LangGraphDataAgent(
        df=df,
        semantic_layer=None,
        eda_results=eda_results,
        insights=insights,
        dataset_name="Test Sales"
    )

    # 1. DatasetQueryTool
    ans_query = agent.run("Show summary for revenue")
    assert ans_query["status"] == "success"
    assert "revenue" in ans_query["answer"].lower()

    # 2. ChartTool
    ans_chart = agent.run("Plot a bar chart of revenue by category")
    assert ans_chart["status"] == "success"
    assert ans_chart["tool_used"] in ["ChartGenerationTool", "ChartTool"]
    assert "plotly_spec" in ans_chart["tool_data"] or "chart_spec" in ans_chart

    # 3. InsightTool
    ans_insight = agent.run("What are the key insights and anomalies?")
    assert ans_insight["status"] == "success"
    assert ans_insight["tool_used"] == "InsightTool"

    # 4. EDATool
    ans_eda = agent.run("Show missing values and correlations")
    assert ans_eda["status"] == "success"
    assert ans_eda["tool_used"] == "EDATool"


# =====================================================================
# 5. Bounded Project Cache LRU Tests
# =====================================================================

def test_bounded_project_cache_eviction():
    """Verify BoundedProjectCache enforces max_size and preserves protected demo projects."""
    cache = BoundedProjectCache(max_size=3)

    # Always preserve demo project
    cache["proj-demo-q3-sales"] = {"name": "Demo Project"}
    cache["proj-1"] = {"name": "Project 1"}
    cache["proj-2"] = {"name": "Project 2"}
    assert len(cache) == 3

    # Adding a 4th project should evict the oldest non-demo project (proj-1)
    cache["proj-3"] = {"name": "Project 3"}
    assert len(cache) == 3
    assert "proj-demo-q3-sales" in cache  # Protected from eviction
    assert "proj-1" not in cache          # Oldest evicted
    assert "proj-2" in cache
    assert "proj-3" in cache

    # Accessing proj-2 should refresh its LRU status
    _ = cache.get("proj-2")
    cache["proj-4"] = {"name": "Project 4"}
    assert len(cache) == 3
    assert "proj-demo-q3-sales" in cache
    assert "proj-3" not in cache          # Now proj-3 was oldest, so evicted
    assert "proj-2" in cache
    assert "proj-4" in cache
