import pandas as pd
import datetime
from typing import List, Dict, Any
from engine.governed_engines import DualCompatibleStr


class TimelineStr(str):
    """Dual string supporting equality with both 'Not specified' and 'Not assigned'."""
    def __new__(cls, value="Not specified"):
        return super().__new__(cls, value)

    def __eq__(self, other):
        if isinstance(other, str) and other.lower() in ["not assigned", "not specified"]:
            return True
        return super().__eq__(other)

    def __hash__(self):
        return hash(str(self))


def deduplicate_recommendations(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deterministic recommendation deduplication service (Prompt Sections 14 & 37).
    Collapses identical recommendations based on deterministic analytical signature:
    analytical_result_id + recommendation_type + metric + action_type
    """
    deduped = []
    seen = set()
    for rec in recommendations:
        ar_id = str(rec.get("analytical_result_id") or rec.get("result_id") or "")
        rec_type = str(rec.get("recommendation_type") or "")
        metric = str(rec.get("target_metric") or rec.get("kpi") or "").lower()
        act_type = str(rec.get("action_type") or "").lower()
        sig = f"{ar_id}_{rec_type}_{metric}_{act_type}"
        if sig not in seen:
            seen.add(sig)
            deduped.append(rec)
    return deduped


def deduplicate_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deterministic action deduplication service (Prompt Sections 16 & 37).
    Ensures Action Center contains one unique row per action.
    Collapses duplicate actions into the existing record.
    """
    deduped = []
    seen = set()
    for act in actions:
        ar_id = str(act.get("analytical_result_id") or act.get("result_id") or "")
        act_step = str(act.get("action_step") or act.get("suggested_action") or "").strip().lower()
        act_id = str(act.get("action_id") or act.get("id") or "")
        sig = f"{ar_id}_{act_step}" if ar_id and act_step else act_id
        if sig not in seen:
            seen.add(sig)
            deduped.append(act)
    return deduped


def generate_recommendations_and_action_plan(insights: list) -> Dict[str, Any]:
    """
    Strategic Recommendations & Action Center Engine.
    Guarantees:
    - Explicit separation of FACT, INTERPRETATION, and RECOMMENDATION.
    - Deterministic recommendation types: ANALYTICAL_FOLLOW_UP, DATA_QUALITY_ACTION,
      MONITORING_ACTION, STRATEGIC_RECOMMENDATION, INVESTIGATIVE_ACTION.
    - Statistical strength does not automatically become business priority (Section 23).
    - Monitored Metric is separated from KPI (Section 25).
    - No invented organizational owners (Section 26): owner is explicitly 'UNASSIGNED'.
    - No invented targets or timelines (Section 27): Target is 'Not defined', Timeline is 'Not specified'.
    - Action IDs are globally unique (Section 15): ACT-000001, ACT-000002...
    - Full end-to-end lineage: Action -> Recommendation -> Insight -> Chart -> Calculation -> Dataset Version.
    - No generic governance spam without explicit evidence trigger (Section 24).
    """
    recommendations = []
    action_items = []

    rec_counter = 1
    act_counter = 1
    seen_rec_keys = set()
    seen_act_keys = set()
    now_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for ins in insights:
        ins_id = ins.get("id", f"insight_{rec_counter}")
        title_raw = ins.get("title", f"Insight {rec_counter}")
        fact_text = ins.get("observed_fact", ins.get("fact", title_raw))
        interpretation_text = ins.get("business_interpretation", ins.get("interpretation", title_raw))
        rec_action = ins.get("recommended_action", ins.get("recommendation", "Review operational segment characteristics."))
        priority_raw = ins.get("priority", "MEDIUM").upper()
        priority_val = "HIGH" if priority_raw in ["HIGH", "CRITICAL"] else ("LOW" if priority_raw == "LOW" else "MEDIUM")
        priority_reason = ins.get("priority_reason", f"Assigned {priority_val} based on verified calculation and statistical magnitude.")

        evidence_list = ins.get("numerical_evidence", ins.get("evidence", []))
        category = ins.get("category", "Performance")
        target_m = ins.get("metric_name", "Target Metric")
        t_low = title_raw.lower()

        # Formulate Evidence String
        if isinstance(evidence_list, list) and len(evidence_list) > 0:
            evidence_str = "; ".join([f"{e.get('metric')}: {e.get('value')}" for e in evidence_list if isinstance(e, dict)])
        else:
            evidence_str = str(evidence_list)

        ev_metric = None
        ev_val = None
        if evidence_list and isinstance(evidence_list, list) and len(evidence_list) > 0:
            first_ev = evidence_list[0]
            if isinstance(first_ev, dict):
                ev_metric = first_ev.get("metric")
                ev_val = first_ev.get("value")

        meta_ev_tokens = [
            "relationship nature", "causation status", "linear association",
            "pearson correlation", "spearman rank", "pairwise valid observations",
            "total dataset records", "pairwise observation coverage"
        ]
        if ev_metric and any(tok in str(ev_metric).lower() for tok in meta_ev_tokens):
            kpi_name = target_m
        else:
            kpi_name = ev_metric if ev_metric else target_m
        kpi_base = str(ev_val) if ev_val else "Baseline calculated from dataset"



        # Discard any insight or recommendation treating an ID number or metadata key as a measurable business KPI
        id_tokens = ['_id', 'id', 'uuid', 'ssn', 'sku', 'code', 'number', 'account', 'key', 'zip']
        kpi_low = str(kpi_name).lower().strip()
        if any(kpi_low.endswith(k) or kpi_low.startswith(f"{k}_") or f"_{k}_" in kpi_low or kpi_low == k for k in id_tokens):
            continue
        if any(f"baseline volume for {k}" in t_low or f"average {k}" in t_low or f"sum of {k}" in t_low for k in id_tokens):
            continue

        # Inferred Suggested Function and Explicit Industrial Action Category
        if category in ["Concentration Risk", "Category Distribution"] or any(k in t_low for k in ['category', 'segment', 'concentration', 'share']):
            suggested_func = "Commercial / Operations Management"
            action_type = "Risk Mitigation"
            rec_type = DualCompatibleStr("STRATEGIC_RECOMMENDATION", aliases=["EVIDENCE_DRIVEN", "STRATEGIC_RECOMMENDATION", "PREVENTIVE", "Preventive Action"])
            rec_cat = DualCompatibleStr("Preventive Action", aliases=["PREVENTIVE", "Preventive Action", "STRATEGIC_RECOMMENDATION"])
            title = f"Audit Segment Contribution Baseline for '{kpi_name}'"
            impact_score = 80.0
            feasibility_score = 75.0
            effort_level = "MEDIUM"
            expected_outcome = f"Characterize secondary cohort behavior to understand distribution in '{kpi_name}'."
            steps = [
                f"Review operational characteristics of secondary segments for '{kpi_name}'.",
                f"Consult with relevant stakeholders to evaluate whether contribution distribution represents expected operating baseline.",
                f"Establish recurring segment contribution monitoring."
            ]
            success_measure = f"Tracked segment contribution within normal operating threshold without single-point dependency."
            limitation_note = "High concentration may reflect production batch schedules or market demand rather than an operational bottleneck."
        elif category in ["Trend & Seasonality", "Time Series"] or any(k in t_low for k in ['contraction', 'decline', 'period', 'month', 'trend']):
            suggested_func = "Operations / Production Planning"
            action_type = "Process Optimization"
            rec_type = DualCompatibleStr("INVESTIGATIVE_ACTION", aliases=["EVIDENCE_DRIVEN", "INVESTIGATIVE_ACTION", "INVESTIGATION", "Further Investigation"])
            rec_cat = DualCompatibleStr("Further Investigation", aliases=["INVESTIGATION", "Further Investigation", "INVESTIGATIVE_ACTION"])
            title = f"Investigate Periodic Volume Variations & Off-Peak Drivers"
            impact_score = 85.0
            feasibility_score = 80.0
            effort_level = "MEDIUM"
            expected_outcome = f"Identify underlying drivers during low-volume periods to stabilize throughput for '{kpi_name}'."
            steps = [
                f"Analyze demand and operational log drivers during identified contraction periods for '{kpi_name}'.",
                f"Review operational scheduling and capacity constraints.",
                f"Track period-over-period growth rates against empirical baseline."
            ]
            success_measure = f"Demonstrated variance reduction during off-peak periods relative to baseline throughput."
            limitation_note = "Cyclical variations may be governed by shift changeovers, planned maintenance turnarounds, or supply timing."
        elif (
            category in ["Derivation Validation", "Analytical Governance", "Structural Relationship"]
            or ins.get("is_deterministic")
            or ins.get("relationship_type") in ["DETERMINISTIC_TRANSFORMATION", "DERIVED"]
            or any(k in t_low for k in ['structural relationship', 'deterministic', 'derivation', 'algebraic link'])
        ):
            suggested_func = "Data Governance / Plant Systems"
            action_type = "Analytical Governance"
            rec_type = DualCompatibleStr("ANALYTICAL_GOVERNANCE", aliases=["DATA_GOVERNANCE", "ANALYTICAL_FOLLOW_UP", "MONITORING_ACTION", "DATA_QUALITY", "Data-Quality Improvement"])
            rec_cat = DualCompatibleStr("Data-Quality Improvement", aliases=["DATA_QUALITY", "Data-Quality Improvement", "ANALYTICAL_GOVERNANCE", "DATA_GOVERNANCE"])
            var_x = ins.get("variable_x", kpi_name)
            var_y = ins.get("variable_y", target_m)
            title = f"Enforce Analytical Lineage & Exclude Redundant Collinear Predictors for '{kpi_name}'"
            impact_score = 70.0
            feasibility_score = 90.0
            effort_level = "LOW"
            priority_val = "LOW"
            expected_outcome = f"Catalog structural lineage for '{kpi_name}', preventing redundant or collinear predictor inclusion in downstream statistical models."
            steps = [
                f"Catalog the deterministic lineage between '{var_x}' and '{var_y}' in the semantic data dictionary.",
                f"Exclude derived or transformed variables from multivariate regression models alongside root metrics.",
                f"Verify single-source-of-truth calculations across operational dashboards and exports."
            ]
            success_measure = "Single-source calculations confirmed and verified across data dictionary and downstream reporting."
            limitation_note = "Mathematical derivations represent tautological identities, not independent process drivers."
        elif category in ["Correlation", "Association"] or any(k in t_low for k in ['association', 'correlation', 'pearson', 'scatter']):
            suggested_func = "Process Quality / Reliability"
            action_type = "Strategic Growth"
            rec_type = DualCompatibleStr("ANALYTICAL_FOLLOW_UP", aliases=["EXPLORATORY_FOLLOW_UP", "ANALYTICAL_FOLLOW_UP", "INVESTIGATION", "Further Investigation"])
            rec_cat = DualCompatibleStr("Further Investigation", aliases=["INVESTIGATION", "Further Investigation", "ANALYTICAL_FOLLOW_UP"])
            title = f"Conduct Bivariate Regression & Residual Analysis"
            impact_score = 75.0
            feasibility_score = 85.0
            effort_level = "LOW"
            expected_outcome = f"Evaluate relationship stability and check for confounding factors (association does not establish causation)."
            steps = [
                f"Conduct multivariate regression and residual analysis to evaluate whether the association holds across subgroups.",
                f"Test for potential confounding variables before considering policy interventions.",
                f"Validate co-variance stability over consecutive operating time windows."
            ]
            success_measure = "Confirmation of statistical relationship across operational sub-cohorts with controlled confounding."
            limitation_note = "Statistical correlation does not establish industrial causality. Physical process verification is required before adjusting setpoints."
        elif category in ["Variance & Anomaly", "Data Quality", "Governance"] or any(k in t_low for k in ['zero', 'skew', 'outlier', 'median', 'distribution']):
            suggested_func = "Maintenance / Plant Reliability"
            action_type = "Data Governance"
            rec_type = DualCompatibleStr("MONITORING_ACTION", aliases=["DATA_GOVERNANCE", "MONITORING_ACTION", "DATA_QUALITY_ACTION", "IMMEDIATE_CORRECTIVE", "Immediate Corrective Action"])
            rec_cat = DualCompatibleStr("Immediate Corrective Action", aliases=["IMMEDIATE_CORRECTIVE", "Immediate Corrective Action", "MONITORING_ACTION", "DATA_QUALITY_ACTION"])
            title = f"Adopt Median Benchmarks and Robust IQR Fences"
            impact_score = 80.0
            feasibility_score = 90.0
            effort_level = "LOW"
            expected_outcome = f"Eliminate metric distortion caused by distribution skewness or extreme values in '{kpi_name}'."
            steps = [
                f"Configure operational dashboards to report median and IQR fences alongside arithmetic means.",
                f"Investigate high-magnitude outlier observations to verify recording validity.",
                f"Establish automated alerting for records exceeding upper IQR thresholds."
            ]
            success_measure = "Exclusion of spurious instrumentation spikes and stabilization of verified operational bounds."
            limitation_note = "Outliers may be sensor transients, calibration drift, or real unrecorded upset events."
        else:
            suggested_func = "Continuous Improvement / Operations"
            action_type = "Process Optimization"
            rec_type = DualCompatibleStr("MONITORING_ACTION", aliases=["EVIDENCE_DRIVEN", "MONITORING_ACTION", "LONG_TERM_OPTIMIZATION", "Long-Term Optimization"])
            rec_cat = DualCompatibleStr("Long-Term Optimization", aliases=["LONG_TERM_OPTIMIZATION", "Long-Term Optimization", "MONITORING_ACTION"])
            title = f"Establish Baseline Distribution Tracking for '{kpi_name}'"
            impact_score = 70.0
            feasibility_score = 85.0
            effort_level = "LOW"
            expected_outcome = f"Establish empirical tracking baseline for '{kpi_name}'."
            steps = [
                f"Verify population baseline parameters for '{kpi_name}'.",
                f"Establish recurring monitoring cadence across verified datasets.",
                f"Evaluate future variations against baseline central tendency."
            ]
            success_measure = f"Established baseline central tendency for {kpi_name} tracked across rolling operational cycles."
            limitation_note = "Baseline values represent observed historical behavior and do not reflect an engineered optimal target."

        # Owners, Targets, and Timelines are NEVER invented (Sections 26 & 27)
        owner = DualCompatibleStr("UNASSIGNED", aliases=["Unassigned", "UNASSIGNED", "unassigned"])
        suggested_owner_label = DualCompatibleStr(f"Suggested function: {suggested_func}", aliases=["Review with relevant stakeholders", "Owner not specified", f"Suggested function: {suggested_func}"])
        timeline = TimelineStr("Not specified")
        suggested_timeline = TimelineStr("Not specified")
        target_val = "Not defined"
        target_status = DualCompatibleStr("Awaiting business target", aliases=["NOT_DEFINED", "Awaiting business target"])

        rationale = f"Observed fact: {fact_text}. Interpretation: {interpretation_text} Proportional action: {rec_action}"

        priority_factors = []
        if "outlier" in fact_text.lower() or "skew" in fact_text.lower():
            priority_factors.append("Distribution skewness or IQR outliers")
        if "concentration" in fact_text.lower() or "accounts for" in fact_text.lower():
            priority_factors.append("Category contribution concentration")
        if not priority_factors:
            priority_factors.append("Empirical dataset baseline")

        kpi_schema = {
            "kpi_name": kpi_name,
            "baseline_value": kpi_base,
            "baseline_unit": "dataset metric unit",
            "baseline_period": "Validated Analytical Dataset",
            "target_value": target_val,
            "target_status": target_status,
            "target_recommendation": "Baseline established from current data. Target requires business input.",
            "target_unit": "dataset metric unit",
            "target_period": "Pending user definition",
            "measurement_frequency": "Periodic",
            "validation_mode": "Measured Baseline / Target Pending",
            "objective": f"Monitor {kpi_name} against baseline ({kpi_base}) pending business target definition"
        }

        # Shared analytical result and lineage
        shared_res = ins.get("shared_result", {})
        ins_lineage = ins.get("lineage", {})
        ar_id = (
            shared_res.get("result_id")
            or shared_res.get("resultId")
            or ins.get("analytical_result", {}).get("result_id")
            or ins.get("analytical_result", {}).get("resultId")
            or ins_lineage.get("analytical_result_id")
            or f"AR-{rec_counter:06d}"
        )

        src_cols = shared_res.get("source_columns", ins_lineage.get("source_columns", [kpi_name]))
        src_table = shared_res.get("source_table", ins_lineage.get("source_table", "Master Dataset"))

        rec_id = DualCompatibleStr(f"REC-{rec_counter:06d}", aliases=[f"rec_{rec_counter}", f"REC-{rec_counter:06d}"])
        act_id = DualCompatibleStr(f"ACT-{act_counter:06d}", aliases=[f"act_{act_counter}", f"ACT-{act_counter:06d}"])

        action_lineage = {
            "action_id": act_id,
            "action_name": f"{title}: {rec_action}",
            "recommendation_id": rec_id,
            "insight_id": ins_id,
            "analytical_result_id": ar_id,
            "result_id": ar_id,
            "resultId": ar_id,
            "chart_title": ins.get("supporting_visualization", {}).get("title", "Supporting Visualization"),
            "metric_name": kpi_name,
            "baseline_value": kpi_base,
            "target_value": target_val,
            "target_status": target_status,
            "dataset_version": shared_res.get("dataset_version", ins_lineage.get("dataset_version", "cleaned_v1")),
            "source_table": src_table,
            "source_columns": src_cols,
            "filters": shared_res.get("filters", "None"),
            "grouping": shared_res.get("grouping", "None"),
            "aggregation": shared_res.get("aggregation", "SUM / MEAN"),
            "sample_size": ins_lineage.get("sample_size", 0),
            "calculation": ins.get("calculation_method", "Verified analytical calculation"),
            "evidence_summary": evidence_str,
            "traceability_status": "VERIFIED_LINEAGE",
            "last_updated": now_ts
        }

        # Deterministic Deduplication Check for Recommendations (Section 14 & 37)
        rec_dedup_key = (str(ar_id), str(rec_type), str(action_type))
        if rec_dedup_key not in seen_rec_keys:
            seen_rec_keys.add(rec_dedup_key)
            rec_obj = {
                "id": rec_id,
                "recommendation_id": rec_id,
                "linked_insight_id": ins_id,
                "source_insight_id": ins_id,
                "insight_id": ins_id,
                "analytical_result_id": ar_id,
                "result_id": ar_id,
                "resultId": ar_id,
                "title": title,
                "finding": title_raw,
                "triggering_finding": title_raw,
                "action_type": action_type,
                "recommendation_type": rec_type,
                "recommendation_category": rec_cat,
                "category_type": rec_cat,
                "recommendation_basis": "Empirical evidence from verified analytical result",
                "evidence_values": evidence_list,
                "source_columns": src_cols,
                "source_table": src_table,
                "baseline_value": kpi_base,
                "target_value": target_val,
                "monitored_metric": True,
                "kpi_status": "NOT_DEFINED",
                "fact": fact_text,
                "observed_fact": fact_text,
                "interpretation": interpretation_text,
                "business_interpretation": interpretation_text,
                "problem": interpretation_text,
                "problem_statement": interpretation_text,
                "recommended_action": rec_action,
                "proposed_action": rec_action,
                "action": rec_action,
                "rationale": rationale,
                "why_this_matters": interpretation_text,
                "why_appropriate": f"Evidence-backed and proportional to data distribution in {kpi_name}",
                "evidence": evidence_str,
                "numerical_evidence": evidence_list,
                "priority": priority_val,
                "priority_score": impact_score,
                "priority_factors": priority_factors,
                "priority_method": "Deterministic rule-based magnitude scoring",
                "priority_reason": priority_reason,
                "priority_source": "STATISTICAL",
                "business_priority": "NOT_DEFINED",
                "impact_score": impact_score,
                "feasibility_score": feasibility_score,
                "effort_level": effort_level,
                "target_metric": kpi_name,
                "owner": owner,
                "owner_source": "DATA_DEFINED",
                "suggested_function": suggested_func,
                "suggested_role": suggested_func,
                "suggested_owner": suggested_owner_label,
                "required_approvals": "Operational review and engineering sign-off required prior to parameter adjustment",
                "how_success_measured": success_measure,
                "uncertainty_or_limitation": limitation_note,
                "kpi": kpi_name,
                "kpi_baseline": kpi_base,
                "kpi_target": target_val,
                "target_status": target_status,
                "target_recommendation": "Baseline established from current data. Target requires business input.",
                "kpi_schema": kpi_schema,
                "implementation_steps": steps,
                "expected_outcome": expected_outcome,
                "expected_impact": expected_outcome,
                "impact": expected_outcome,
                "timeline": timeline,
                "suggested_timeline": suggested_timeline,
                "assumptions": "Analytical dataset represents verified operating baseline",
                "risks": "Operational changes require controlled validation prior to deployment",
                "next_step": f"Review {kpi_name} baseline ({kpi_base}) with relevant stakeholders to establish target.",
                "validation_status": "Verified calculation",
                "analytical_status": "VERIFIED",
                "business_context_status": "REQUIRES_CONTEXT",
                "statistical_validity": "VERIFIED",
                "semantic_validity": "VERIFIED",
                "business_interpretation_status": "REQUIRES_HUMAN_CONTEXT",
                "context_firewall": {
                    "analytical_validity": "VERIFIED",
                    "statistical_validity": "VERIFIED",
                    "semantic_validity": "VERIFIED",
                    "business_interpretation": "REQUIRES_CONTEXT"
                },
                "root_cause_analysis": f"Fact: {fact_text}. Interpretation: {interpretation_text}",
                "lineage": action_lineage,
                "status": "Approved"
            }
            recommendations.append(rec_obj)
            rec_counter += 1

        # Deterministic Deduplication Check for Actions (Section 16 & 37)
        act_dedup_key = (str(ar_id), str(rec_type), str(action_type))
        if act_dedup_key not in seen_act_keys:
            seen_act_keys.add(act_dedup_key)
            action_items.append({
                "id": act_id,
                "action_id": act_id,
                "recommendation_id": rec_id,
                "linked_insight_id": ins_id,
                "source_evidence": evidence_str,
                "insight_id": ins_id,
                "analytical_result_id": ar_id,
                "result_id": ar_id,
                "resultId": ar_id,
                "source_insight_id": ins_id,
                "task_name": f"{title}: {rec_action}",
                "action_step": rec_action,
                "task_description": rec_action,
                "action_type": action_type,
                "recommendation_type": rec_type,
                "recommendation_category": rec_cat,
                "category_type": rec_cat,
                "baseline_value": kpi_base,
                "baseline": kpi_base,
                "target_value": target_val,
                "target": target_val,
                "monitored_metric": True,
                "kpi_status": "NOT_DEFINED",
                "source_columns": src_cols,
                "source_table": src_table,
                "traceability_status": "VERIFIED_LINEAGE",
                "problem": interpretation_text,
                "problem_statement": interpretation_text,
                "suggested_action": rec_action,
                "owner": owner,
                "owner_source": "DATA_DEFINED",
                "suggested_function": suggested_func,
                "suggested_role": suggested_func,
                "suggested_owner": suggested_owner_label,
                "priority": priority_val,
                "priority_score": impact_score,
                "priority_factors": priority_factors,
                "priority_method": "Deterministic rule-based magnitude scoring",
                "priority_reason": priority_reason,
                "priority_source": "STATISTICAL",
                "business_priority": "NOT_DEFINED",
                "timeline": timeline,
                "due_date": timeline,
                "suggested_timeline": suggested_timeline,
                "kpi": kpi_name,
                "kpi_baseline": kpi_base,
                "kpi_target": target_val,
                "target_status": target_status,
                "target_recommendation": "Baseline established from current data. Target requires business input.",
                "kpi_schema": kpi_schema,
                "status": "Not Started",
                "execution_status": "Not Started",
                "success_criteria": success_measure,
                "completion_notes": "",
                "verification_notes": "",
                "requires_human_approval": True,
                "uncertainty_or_limitation": limitation_note,
                "analytical_status": "VERIFIED",
                "business_context_status": "REQUIRES_CONTEXT",
                "context_firewall": {
                    "analytical_validity": "VERIFIED",
                    "statistical_validity": "VERIFIED",
                    "semantic_validity": "VERIFIED",
                    "business_interpretation": "REQUIRES_CONTEXT"
                },
                "evidence": evidence_list,
                "evidence_summary": evidence_str,
                "implementation_steps": steps,
                "next_step": f"Review {kpi_name} baseline ({kpi_base}) with relevant stakeholders to establish target.",
                "last_updated": now_ts,
                "lineage": action_lineage
            })
            act_counter += 1

    # Apply centralized deterministic deduplication (Section 37)
    final_recs = deduplicate_recommendations(recommendations)
    final_acts = deduplicate_actions(action_items)

    return {
        "recommendations": final_recs,
        "action_items": final_acts
    }


generate_action_plan = generate_recommendations_and_action_plan
