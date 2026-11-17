import io
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from main import app, PROJECTS_DB, STORE_DIR
from engine.profiler import detect_observation_grain, profile_file
from engine.dictionary import generate_data_dictionary
from engine.cleaner import detect_quality_issues
from engine.eda import run_automated_eda, build_operational_eda
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan

client = TestClient(app)

def test_plant_industrial_refactor_end_to_end():
    """
    Validates complete 10-point industrial analytics workflow on a synthetic plant dataset:
    - Smelter potline / maintenance / safety operations data
    - Grain detection, profiling summary, data quality vs operational gaps
    - 5 operational questions
    - 7 management story sections
    - 5 recommendation categories with Action Center traceability
    """
    # 1. Generate realistic industrial plant dataset
    np.random.seed(42)
    n = 120
    shifts = ["Shift A", "Shift B", "Shift C"]
    departments = ["Smelter Potline 1", "Smelter Potline 2", "Casthouse", "Carbon Plant"]
    statuses = ["Open", "Closed", "Pending Parts", "In Progress"]
    severities = [1, 2, 3, 4, 5]

    data = {
        "Observation_ID": [f"OBS-{1000 + i}" for i in range(n)],
        "Date": pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y-%m-%d").tolist(),
        "Shift": np.random.choice(shifts, size=n).tolist(),
        "Department": np.random.choice(departments, size=n).tolist(),
        "Operating_Temperature_C": np.random.normal(960.0, 15.0, size=n).round(1).tolist(),
        "Bath_Voltage_V": np.random.normal(4.2, 0.25, size=n).round(2).tolist(),
        "Downtime_Minutes": np.random.exponential(18.0, size=n).round(1).tolist(),
        "Action_Status": np.random.choice(statuses, size=n).tolist(),
        "Action_Owner": [("Supervisor " + str(i % 5)) if i % 4 != 0 else None for i in range(n)], # intentional gaps
        "Severity_Grade": np.random.choice(severities, size=n).tolist(),
    }
    df = pd.DataFrame(data)

    # 2. Test Observation Grain Detection
    grain = detect_observation_grain(df)
    assert grain is not None
    assert "grain_description" in grain
    assert len(grain["grain_description"]) > 5

    # 3. Test Profiling Summary (Prompt Section 5)
    profile = profile_file("plant_potline_data.csv", df=df)
    summary = profile.get("profiling_summary", {})
    assert "what_was_uploaded" in summary
    assert "observation_grain" in summary
    assert "reliable_analyses" in summary
    assert len(summary["reliable_analyses"]) >= 2
    assert "additional_information_needed" in summary

    # 4. Test Data Dictionary Generic Semantic Roles & Suggested Uses
    dictionary = generate_data_dictionary(df, "plant_potline_data.csv")
    col_names = [d["column_name"] for d in dictionary]
    assert "Observation_ID" in col_names
    assert "Operating_Temperature_C" in col_names
    obs_id_dict = next(d for d in dictionary if d["column_name"] == "Observation_ID")
    assert obs_id_dict["semantic_role"] in ["Dimension", "Identifier"]
    assert "identification" in obs_id_dict["suggested_analytical_use"].lower()

    temp_dict = next(d for d in dictionary if d["column_name"] == "Operating_Temperature_C")
    assert temp_dict["semantic_role"] == "Measure"
    assert "distribution" in temp_dict["suggested_analytical_use"].lower()

    # 5. Test Data Quality vs Operational Process Gaps
    issues = detect_quality_issues(df, "plant_potline_data.csv")
    assert len(issues) > 0
    categories = [i.get("category") for i in issues]
    assert "OPERATIONAL_GAP" in categories or "DATA_QUALITY" in categories
    for issue in issues:
        assert "why_it_matters" in issue
        assert "what_to_check" in issue
        assert "confirmation_status" in issue

    # 6. Test Automated EDA & 5 Operational Questions
    eda_results = run_automated_eda(df)
    assert "operational_eda" in eda_results
    op_eda = eda_results["operational_eda"]
    assert "what_is_happening" in op_eda
    assert "where_is_it_happening" in op_eda
    assert "why_might_it_be_happening" in op_eda
    assert "what_is_lacking_or_deviating" in op_eda
    assert "what_should_we_do_next" in op_eda

    assert op_eda["what_is_happening"]["total_records"] == n
    assert len(op_eda["where_is_it_happening"]["group_comparisons"]) > 0

    # 7. Test Management Story 7 Sections
    insights_res = generate_insights_and_story(df, eda_results, dataset_version="cleaned_v1")
    story = insights_res["story"]
    assert "executive_summary" in story and isinstance(story["executive_summary"], str)
    assert len(story["executive_summary"]) > 20
    assert "current_operating_situation" in story
    assert len(story["major_findings"]) > 0
    assert len(story["performance_gaps"]) > 0
    assert "potential_contributing_factors" in story
    assert len(story["management_implications"]) > 0
    assert len(story["recommended_priorities"]) > 0

    # 8. Test Recommendations & Action Plan
    recs_res = generate_recommendations_and_action_plan(insights_res["insights"])
    recs = recs_res["recommendations"]
    actions = recs_res["action_items"]
    assert len(recs) > 0
    assert len(actions) > 0

    valid_rec_types = [
        "Immediate Corrective Action",
        "Further Investigation",
        "Preventive Action",
        "Data-Quality Improvement",
        "Long-Term Optimization"
    ]
    for rec in recs:
        assert rec["recommendation_category"] in valid_rec_types
        assert "evidence" in rec
        assert "kpi_baseline" in rec
        assert "kpi_target" in rec

    for act in actions:
        assert "action_id" in act
        assert "problem_statement" in act
        assert "status" in act
        assert act["status"] in ["Not Started", "In Progress", "Blocked", "Completed", "Verified"]

    # 9. Test API Upload and Action Status Update Lifecycle
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    upload_resp = client.post(
        "/api/projects/upload",
        files=[("files", ("synthetic_smelter_plant.csv", buf, "text/csv"))]
    )
    assert upload_resp.status_code == 200
    upload_json = upload_resp.json()
    proj_id = upload_json["project_id"]
    assert proj_id in PROJECTS_DB

    # Verify project retrieval contains refactored operational_eda & 7 story sections
    proj_details = client.get(f"/api/project/{proj_id}").json()
    assert "operational_eda" in proj_details["eda_results"]
    assert "executive_summary" in proj_details["story"]

    # Update Action Status with owner and verification notes
    first_action = proj_details["action_items"][0]
    act_id = first_action["id"] if "id" in first_action else first_action["action_id"]

    update_payload = {
        "status": "Verified",
        "owner": "Potline General Manager",
        "due_date": "End of Q2 2026",
        "target": "Zero critical temperature excursions",
        "completion_notes": "Calibrated temperature probes across Potline 1 and 2.",
        "verification_notes": "Empirically verified temperature variance reduced by 42% over 14 operating days."
    }
    status_resp = client.post(f"/api/project/{proj_id}/action-status", json={"action_id": act_id, **update_payload})
    assert status_resp.status_code == 200

    # Verify persistence of action updates
    proj_refreshed = client.get(f"/api/project/{proj_id}").json()
    updated_item = next(a for a in proj_refreshed["action_items"] if (a.get("id") == act_id or a.get("action_id") == act_id))
    assert updated_item["status"] == "Verified"
    assert updated_item["owner"] == "Potline General Manager"
    assert updated_item["due_date"] == "End of Q2 2026"
    assert "temperature excursions" in updated_item["target"]
    assert "Calibrated temperature probes" in updated_item["completion_notes"]
    assert "Empirically verified" in updated_item["verification_notes"]

    # 10. Test AI Assistant Chatbot with plant operational queries
    chat_resp = client.post(
        f"/api/project/{proj_id}/chat",
        json={"message": "What are the biggest operational problems and performance gaps in this dataset?"}
    )
    assert chat_resp.status_code == 200
    chat_json = chat_resp.json()
    answer_text = chat_json.get("answer") or chat_json.get("text") or ""
    assert len(answer_text) > 20
    # The chatbot response should mention operational insights or findings
    assert any(term in answer_text.lower() for term in ["finding", "gap", "operational", "observation", "priority", "temperature", "shift", "status", "action", "baseline", "deviation"])
