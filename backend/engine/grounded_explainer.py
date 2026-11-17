"""
Grounded Analytical Explainer Engine.

Transforms deterministic ExecutionResults and QueryPlans into authoritative,
zero-hallucination natural language responses, formatted evidence tables,
and dataset-grounded follow-up recommendations.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np

from engine.contracts.chat_contracts import (
    QueryPlan,
    QueryIntent,
    ExecutionResult,
    ResultValidationStatus,
    ChatResponseContract
)


class GroundedAnalyticalExplainer:
    """
    Zero-Hallucination Grounded Explainer.
    
    Generates natural language responses where every single factual or numerical claim
    is strictly grounded in the deterministic ExecutionResult and QueryPlan.
    """

    def __init__(self, col_metadata: Optional[Dict[str, Dict[str, Any]]] = None):
        self.col_metadata = col_metadata or {}

    def explain(
        self,
        plan: QueryPlan,
        result: ExecutionResult,
        project_id: str = "project_active",
        dataset_id: str = "dataset_active"
    ) -> ChatResponseContract:
        """
        Translates plan and execution result into a complete ChatResponseContract.
        """
        intent = plan.intent
        status = result.status

        # 1. Handle Ambiguity & Clarification Responses
        if status == ResultValidationStatus.AMBIGUOUS or intent == QueryIntent.AMBIGUOUS:
            prompt = plan.clarification_prompt or "Your query matched multiple potential columns. Please clarify."
            return ChatResponseContract(
                project_id=project_id,
                dataset_id=dataset_id,
                question=plan.query_text,
                query=plan.query_text,
                intent=QueryIntent.AMBIGUOUS,
                status=ResultValidationStatus.AMBIGUOUS,
                answer=prompt,
                clarification_required=True,
                warnings=result.warnings,
                query_plan=plan.to_dict(),
                records_evaluated=result.records_evaluated,
                coverage_pct=result.coverage_pct,
                match_suggestions=plan.match_suggestions,
                suggestion_mode=plan.suggestion_mode,
                resolved_field=plan.resolved_field,
                confirmed_field=plan.confirmed_field,
                interactive_options=plan.interactive_options
            )

        # 2. Handle Errors and No-Match Responses
        if status == ResultValidationStatus.ERROR:
            err_msg = result.error_message or "An unexpected error occurred during analytical execution."
            return ChatResponseContract(
                project_id=project_id,
                dataset_id=dataset_id,
                question=plan.query_text,
                query=plan.query_text,
                intent=intent,
                status=ResultValidationStatus.ERROR,
                answer=f"⚠️ **Execution Error**: {err_msg}",
                query_plan=plan.to_dict(),
                warnings=result.warnings
            )

        if intent != QueryIntent.OUTLIER_ANALYSIS and (status == ResultValidationStatus.NO_MATCH or result.rows_returned == 0):
            filter_desc = self._summarize_filters(plan.filters)
            answer = f"No records in the dataset match the criteria: **{filter_desc}**." if filter_desc else "No records matched your request."
            return ChatResponseContract(
                project_id=project_id,
                dataset_id=dataset_id,
                question=plan.query_text,
                query=plan.query_text,
                intent=intent,
                status=ResultValidationStatus.NO_MATCH,
                answer=answer,
                query_plan=plan.to_dict(),
                warnings=result.warnings,
                records_evaluated=result.records_evaluated,
                coverage_pct=result.coverage_pct
            )

        # 3. Generate Intent-Specific Grounded Explanation
        answer, formula = self._build_intent_explanation(plan, result)

        # Append structured warnings if any
        if result.warnings:
            warning_bullets = "\n".join([f"- ⚠️ *{w}*" for w in result.warnings])
            answer += f"\n\n**Data Notes:**\n{warning_bullets}"

        # 4. Generate Dataset-Grounded Follow-Up Suggestions
        follow_ups = self._generate_follow_ups(plan, result)
        if follow_ups:
            follow_up_bullets = "\n".join([f"- {q}" for q in follow_ups])
            answer += f"\n\n**Suggested Next Steps:**\n{follow_up_bullets}"

        # 5. Build dynamic chart specification if evidence is multi-row tabular
        chart_spec = self._build_chart_spec(plan, result)

        return ChatResponseContract(
            project_id=project_id,
            dataset_id=dataset_id,
            question=plan.query_text,
            query=plan.query_text,
            intent=intent,
            status=result.status,
            answer=answer,
            numerical_value=result.scalar_value,
            calculation_formula=formula,
            evidence=result.data_sample,
            chart_spec=chart_spec,
            query_plan=plan.to_dict(),
            warnings=result.warnings,
            records_evaluated=result.records_evaluated,
            coverage_pct=result.coverage_pct,
            match_suggestions=plan.match_suggestions,
            suggestion_mode=plan.suggestion_mode,
            resolved_field=plan.resolved_field,
            confirmed_field=plan.confirmed_field,
            interactive_options=plan.interactive_options
        )

    def _build_intent_explanation(self, plan: QueryPlan, result: ExecutionResult) -> Tuple[str, str]:
        """Builds direct-answer-first natural language text and calculation formula."""
        intent = plan.intent
        metric = plan.metric or "Value"
        dim = plan.dimension or "Category"
        meta = self.col_metadata.get(metric, {})
        subtype = str(meta.get("semantic_subtype", meta.get("measure_type", "")))

        # 1. Global Aggregation
        if intent == QueryIntent.GLOBAL_AGGREGATION:
            val = result.scalar_value
            f_val = self.format_value(val, subtype)
            agg_name = plan.aggregation or (plan.aggregations[0] if plan.aggregations else "MEAN")
            agg_label = "average" if agg_name in ["MEAN", "AVG"] else agg_name.lower()

            if plan.filters:
                filter_desc = self._summarize_filters(plan.filters)
                if not plan.metric or agg_name in ["COUNT", "COUNT_DISTINCT"]:
                    direct_ans = (
                        f"There are **{int(val or 0):,} records** where **{filter_desc}** "
                        f"(accounting for **{result.coverage_pct}%** of the total {result.total_dataset_rows:,} records in the dataset)."
                    )
                    formula = f"SELECT COUNT(*) FROM dataset WHERE {filter_desc}"
                else:
                    direct_ans = (
                        f"The {agg_label} **{metric}** for **{filter_desc}** is **{f_val}** "
                        f"(evaluated across {result.records_evaluated:,} records, {result.coverage_pct}% of the dataset)."
                    )
                    formula = f"SELECT {agg_name}(\"{metric}\") FROM dataset WHERE {filter_desc}"
            else:
                if not plan.metric or agg_name in ["COUNT", "COUNT_DISTINCT"]:
                    direct_ans = f"The total count across the dataset is **{int(val or 0):,} records**."
                    formula = "SELECT COUNT(*) FROM dataset"
                else:
                    direct_ans = f"The {agg_label} **{metric}** across the dataset is **{f_val}** (evaluated across all {result.records_evaluated:,} records)."
                    formula = f"{agg_name}(\"{metric}\")"

            # If multiple aggregations were computed in data_sample
            if result.data_sample and len(result.data_sample) == 1:
                row = result.data_sample[0]
                details = []
                for k, v in row.items():
                    if k != "total_count" and not pd.isna(v):
                        clean_k = k.replace("_", " ").title()
                        details.append(f"**{clean_k}**: {self.format_value(v, subtype)}")
                if len(details) > 1:
                    direct_ans += "\n\n" + " | ".join(details)

            return direct_ans, formula

        # 2. Group Comparison
        elif intent == QueryIntent.GROUP_COMPARISON:
            sample = result.data_sample
            top_row = sample[0]
            bottom_row = sample[-1]
            top_grp = top_row.get(dim, "Unknown")
            bot_grp = bottom_row.get(dim, "Unknown")

            # Determine value key (handles count queries, metric queries, or rank queries)
            val_col = "metric_val"
            if "metric_val" not in top_row:
                for candidate in ["count", "rank_val", "group_count", "value"]:
                    if candidate in top_row:
                        val_col = candidate
                        break
                if val_col == "metric_val":
                    non_dim = [k for k in top_row.keys() if k != dim]
                    if non_dim:
                        val_col = non_dim[0]

            is_count = (val_col == "count" or not plan.metric or plan.aggregation == "COUNT")
            filter_val = str(plan.filters[0].value) if plan.filters and getattr(plan.filters[0], "value", None) else ""
            if is_count:
                metric_label = f"{filter_val} case(s)" if filter_val else "records"
            else:
                agg_label = "average" if plan.aggregation in ["MEAN", "AVG", None] else str(plan.aggregation).lower()
                metric_label = f"{agg_label} {metric}"

            raw_top = top_row.get(val_col, 0)
            raw_bot = bottom_row.get(val_col, 0)
            top_val = self.format_value(raw_top, subtype) if not is_count else f"{int(raw_top):,}"
            bot_val = self.format_value(raw_bot, subtype) if not is_count else f"{int(raw_bot):,}"

            # Calculate total & row percentages
            total_val = sum(r.get(val_col, 0) for r in sample if isinstance(r.get(val_col), (int, float)))
            if total_val > 0:
                for r in sample:
                    if "percentage" not in r and isinstance(r.get(val_col), (int, float)):
                        r["percentage"] = round((r[val_col] / total_val) * 100, 1)

            total_str = f" (Total: **{int(total_val) if is_count else total_val:,.0f}** {metric_label} across all {result.rows_returned} groups)" if total_val > 0 else ""

            direct_ans = (
                f"Across {result.rows_returned} **{dim}** groups, **{top_grp}** has the highest volume of {metric_label} at **{top_val}**, "
                f"while **{bot_grp}** has the lowest at **{bot_val}**{total_str}."
            )
            cols_to_render = [dim, val_col]
            if any("percentage" in r for r in sample):
                cols_to_render.append("percentage")
            table_md = self._render_markdown_table(sample, cols_to_render, subtype)
            direct_ans += f"\n\n{table_md}"
            formula = f"SELECT \"{dim}\", COUNT(*) FROM dataset GROUP BY 1" if is_count else f"SELECT \"{dim}\", AVG(\"{metric}\") FROM dataset GROUP BY 1"
            return direct_ans, formula

        # 3. Ranking Top N
        elif intent == QueryIntent.RANKING_TOP_N:
            sample = result.data_sample
            top_row = sample[0]
            top_grp = top_row.get(dim, "Unknown")
            n_val = result.rows_returned

            val_col = "rank_val"
            if "rank_val" not in top_row:
                for candidate in ["count", "metric_val", "value"]:
                    if candidate in top_row:
                        val_col = candidate
                        break
                if val_col == "rank_val":
                    non_dim = [k for k in top_row.keys() if k != dim]
                    if non_dim:
                        val_col = non_dim[0]

            is_count = (val_col == "count" or not plan.metric or plan.aggregation == "COUNT")
            filter_val = str(plan.filters[0].value) if plan.filters and getattr(plan.filters[0], "value", None) else ""
            metric_label = f"{filter_val} cases" if filter_val and is_count else (metric if not is_count else "records")

            raw_top = top_row.get(val_col, 0)
            top_val = self.format_value(raw_top, subtype) if not is_count else f"{int(raw_top):,}"

            total_val = sum(r.get(val_col, 0) for r in sample if isinstance(r.get(val_col), (int, float)))
            if total_val > 0:
                for r in sample:
                    if "percentage" not in r and isinstance(r.get(val_col), (int, float)):
                        r["percentage"] = round((r[val_col] / total_val) * 100, 1)

            direct_ans = f"The top {n_val} **{dim}** groups ranked by **{metric_label}** are led by **{top_grp}** with **{top_val}**."
            cols_to_render = [dim, val_col]
            if any("percentage" in r for r in sample):
                cols_to_render.append("percentage")
            table_md = self._render_markdown_table(sample, cols_to_render, subtype)
            direct_ans += f"\n\n{table_md}"
            formula = f"SELECT \"{dim}\", COUNT(*) FROM dataset GROUP BY 1 ORDER BY 2 DESC LIMIT {n_val}" if is_count else f"SELECT \"{dim}\", SUM(\"{metric}\") FROM dataset GROUP BY 1 ORDER BY 2 DESC LIMIT {n_val}"
            return direct_ans, formula

        # 4. Correlation Analysis
        elif intent == QueryIntent.CORRELATION_ANALYSIS:
            r_val = float(result.scalar_value or 0.0)
            v1 = plan.metric or (plan.variables[0] if plan.variables else "Metric1")
            v2 = plan.metric2 or (plan.variables[1] if plan.variables and len(plan.variables) > 1 else (plan.variables[0] if plan.variables and plan.variables[0] != v1 else "Secondary Metric"))
            sample_size = result.records_evaluated or result.total_dataset_rows

            abs_r = abs(r_val)
            direction = "positive" if r_val >= 0 else "negative"
            if abs_r >= 0.70:
                strength = "strong"
            elif abs_r >= 0.40:
                strength = "moderate"
            elif abs_r >= 0.20:
                strength = "weak"
            else:
                strength = "negligible"

            direct_ans = (
                f"The Pearson correlation coefficient between **{v1}** and **{v2}** is **{r_val:+.4f}** "
                f"(evaluated on {sample_size:,} observations), indicating a **{strength} {direction} relationship**."
            )
            formula = f"CORR(\"{v1}\", \"{v2}\")"
            return direct_ans, formula

        # 5. Time Series Trend
        elif intent == QueryIntent.TIME_SERIES_TREND:
            sample = result.data_sample
            first_period = str(sample[0].get("period", "start"))[:10]
            last_period = str(sample[-1].get("period", "end"))[:10]
            first_val = float(sample[0].get("avg_val", 0.0))
            last_val = float(sample[-1].get("avg_val", 0.0))
            change = last_val - first_val
            trend_dir = "an upward" if change > 0 else ("a downward" if change < 0 else "a stable")

            direct_ans = (
                f"Over time, **{metric}** shows **{trend_dir} trajectory** across {result.rows_returned} periods, "
                f"starting at **{self.format_value(first_val, subtype)}** in {first_period} and ending at **{self.format_value(last_val, subtype)}** in {last_period}."
            )
            table_md = self._render_markdown_table(sample[:12], ["period", "count", "avg_val"], subtype)
            direct_ans += f"\n\n{table_md}"
            formula = f"SELECT DATE_TRUNC('{plan.time_grain or 'month'}', \"{plan.time_column}\"), AVG(\"{metric}\") FROM dataset GROUP BY 1"
            return direct_ans, formula

        # 6. Distribution Analysis
        elif intent == QueryIntent.DISTRIBUTION_ANALYSIS:
            row = result.data_sample[0]
            mean_v = self.format_value(row.get("mean"), subtype)
            med_v = self.format_value(row.get("median"), subtype)
            min_v = self.format_value(row.get("min"), subtype)
            max_v = self.format_value(row.get("max"), subtype)
            std_v = self.format_value(row.get("stddev"), subtype)
            skew_v = float(row.get("skewness", 0.0))
            skew_desc = "positively skewed (right-tailed)" if skew_v > 0.5 else ("negatively skewed (left-tailed)" if skew_v < -0.5 else "approximately symmetric")

            direct_ans = (
                f"The distribution of **{metric}** has a median of **{med_v}**, mean of **{mean_v}**, "
                f"and standard deviation of **{std_v}**, spanning from **{min_v}** to **{max_v}**. The distribution is {skew_desc} (skewness: {skew_v:.2f})."
            )
            formula = f"SUMMARY_STATISTICS(\"{metric}\")"
            return direct_ans, formula

        # 7. Outlier Analysis
        elif intent == QueryIntent.OUTLIER_ANALYSIS:
            outlier_count = result.rows_returned
            if outlier_count == 0:
                direct_ans = f"No outlier records were identified in **{metric}** using continuous Interquartile Range (1.5 × IQR) boundary criteria."
            else:
                direct_ans = f"Identified **{outlier_count} outlier records** in **{metric}** using continuous Interquartile Range (IQR) boundary criteria."
                top_cols = [c for c in result.tabular_columns if c in [metric, "iqr_distance", dim, "Observation_ID", "Plant_Zone", "Order_ID"]][:4]
                table_md = self._render_markdown_table(result.data_sample[:10], top_cols, subtype)
                direct_ans += f"\n\n{table_md}"
            formula = f"SELECT * FROM dataset WHERE \"{metric}\" < Q1 - 1.5*IQR OR \"{metric}\" > Q3 + 1.5*IQR"
            return direct_ans, formula

        # 8. Distinct Count
        elif intent == QueryIntent.DISTINCT_COUNT:
            cnt = int(result.scalar_value or result.data_sample[0].get("distinct_count", 0))
            direct_ans = f"There are **{cnt:,} distinct values** for **{dim or metric}** in the dataset."
            formula = f"COUNT(DISTINCT \"{dim or metric}\")"
            return direct_ans, formula

        # 9. Filtered Records
        elif intent == QueryIntent.FILTERED_RECORDS:
            cnt = result.rows_returned
            filter_desc = self._summarize_filters(plan.filters)
            direct_ans = f"Found **{cnt} records** matching filter criteria: **{filter_desc}**."
            cols = [c for c in result.tabular_columns if not c.startswith("_")][:5]
            table_md = self._render_markdown_table(result.data_sample[:10], cols, subtype)
            direct_ans += f"\n\n{table_md}"
            formula = f"SELECT * FROM dataset WHERE {filter_desc} LIMIT {plan.limit or 100}"
            return direct_ans, formula

        # 10. Row Lookup
        elif intent == QueryIntent.ROW_LOOKUP:
            row = result.data_sample[0]
            summary_items = [f"**{k}**: {v}" for k, v in list(row.items())[:6]]
            direct_ans = f"Found record details:\n\n" + " | ".join(summary_items)
            formula = "SELECT * FROM dataset WHERE ID = ? LIMIT 1"
            return direct_ans, formula

        # 11. Data Quality & Dataset Summary
        elif intent == QueryIntent.DATA_QUALITY:
            direct_ans = f"Completed data hygiene audit across all {result.total_dataset_rows:,} records."
            formula = "AUDIT_DATA_QUALITY(*)"
            return direct_ans, formula

        else:
            direct_ans = f"The dataset contains **{result.total_dataset_rows:,} records** across {len(result.tabular_columns)} columns."
            formula = "SELECT COUNT(*) FROM dataset"
            return direct_ans, formula

    def format_value(self, val: Any, subtype: Optional[str] = None) -> str:
        """Formats numbers dynamically according to their semantic subtype."""
        if val is None or pd.isna(val):
            return "N/A"
        try:
            num = float(val)
        except (ValueError, TypeError):
            return str(val)

        if subtype == "Currency":
            return f"${num:,.2f}" if num >= 0 else f"-${abs(num):,.2f}"
        elif subtype == "Percentage":
            return f"{num:.2f}%"
        elif subtype in ["Quantity", "Count"]:
            return f"{int(round(num)):,}"
        elif subtype in ["Rating", "Duration"]:
            return f"{num:.2f}"
        else:
            return f"{num:,.2f}" if not num.is_integer() else f"{int(num):,}"

    def _render_markdown_table(self, rows: List[Dict[str, Any]], cols: List[str], subtype: Optional[str]) -> str:
        """Constructs GitHub-flavored markdown table for analytical evidence."""
        if not rows or not cols:
            return ""

        headers = [c.replace("_", " ").title() for c in cols]
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

        data_lines = []
        for r in rows[:10]:
            vals = []
            for c in cols:
                raw_v = r.get(c, "")
                if c == "percentage" and isinstance(raw_v, (int, float)):
                    vals.append(f"{raw_v:.1f}%")
                elif c == "count" and isinstance(raw_v, (int, float)):
                    vals.append(f"{int(raw_v):,}")
                elif isinstance(raw_v, (int, float)) and c in ["metric_val", "rank_val", "avg_val"]:
                    vals.append(self.format_value(raw_v, subtype))
                elif isinstance(raw_v, float):
                    vals.append(f"{raw_v:.2f}")
                else:
                    vals.append(str(raw_v))
            data_lines.append("| " + " | ".join(vals) + " |")

        return "\n".join([header_line, separator_line] + data_lines)

    def _summarize_filters(self, filters: List[Any]) -> str:
        """Formats list of ColumnFilters into clean human-readable text."""
        if not filters:
            return ""
        parts = []
        for f in filters:
            col = getattr(f, "column", str(f))
            op = getattr(f, "operator", "==")
            val = getattr(f, "value", "")
            op_val = getattr(op, "value", str(op))
            op_str = "=" if op_val == "==" else str(op_val)
            parts.append(f"{col} {op_str} '{val}'")
        return ", ".join(parts)

    def _generate_follow_ups(self, plan: QueryPlan, result: ExecutionResult) -> List[str]:
        """Generates grounded, logical follow-up questions."""
        metric = plan.metric
        dim = plan.dimension
        follow_ups = []

        if plan.intent == QueryIntent.GLOBAL_AGGREGATION and metric:
            follow_ups.append(f"How does **{metric}** vary across different categories?")
            follow_ups.append(f"Are there any extreme outliers in **{metric}**?")
        elif plan.intent == QueryIntent.GROUP_COMPARISON and dim and metric:
            if result.data_sample:
                top_group = result.data_sample[0].get(dim)
                if top_group:
                    follow_ups.append(f"What drives the high {metric} in **{top_group}**?")
            follow_ups.append(f"Would you like to see the distribution of **{metric}** across all {dim} groups?")
        elif plan.intent == QueryIntent.CORRELATION_ANALYSIS:
            follow_ups.append("Would you like to analyze if this relationship holds across different categories?")
            follow_ups.append("Should we run a linear regression model on these variables?")
        elif plan.intent == QueryIntent.OUTLIER_ANALYSIS:
            follow_ups.append("Would you like to see the full record details of the top outliers?")

        return follow_ups[:2]

    def _build_chart_spec(self, plan: QueryPlan, result: ExecutionResult) -> Optional[Dict[str, Any]]:
        """Constructs an interactive chart specification when result has multi-row data."""
        sample = result.data_sample
        if not sample or len(sample) < 2:
            return None

        # Do NOT generate aggregation charts for unaggregated raw row dumps
        if plan.intent in [QueryIntent.FILTERED_RECORDS, QueryIntent.ROW_LOOKUP, QueryIntent.DATASET_SUMMARY]:
            return None

        dim = plan.dimension
        metric = plan.metric
        intent = plan.intent

        # Determine value key
        val_key = None
        for candidate in ["metric_val", "count", "rank_val", "value", "avg_val", "distinct_count"]:
            if candidate in sample[0]:
                val_key = candidate
                break

        # Fallback to an explicit numeric column in sample if not standard aggregate key
        if not val_key and dim:
            for k, v in sample[0].items():
                if k != dim and isinstance(v, (int, float)):
                    val_key = k
                    break

        # If no numeric value column exists, do not attempt to chart strings as 0
        if not val_key:
            return None

        # Determine category / x key
        cat_key = dim or ("period" if "period" in sample[0] else list(sample[0].keys())[0])

        if not val_key or not cat_key:
            return None

        # Determine metric label / title
        filter_label = ""
        if plan.filters:
            f_vals = [str(f.value) for f in plan.filters if getattr(f, "value", None)]
            if f_vals:
                filter_label = f"{' '.join(f_vals)} "

        if intent in [QueryIntent.TIME_SERIES_TREND] or "period" in sample[0]:
            chart_type = "LINE"
            title = f"{filter_label}{metric or 'Volume'} Trend over Time"
        elif len(sample) > 5:
            chart_type = "HORIZONTAL_BAR"
            title = f"{filter_label}{metric or 'Cases'} by {cat_key}"
        else:
            chart_type = "BAR"
            title = f"{filter_label}{metric or 'Cases'} by {cat_key}"

        chart_data = []
        for r in sample:
            cat_val = str(r.get(cat_key, "Unknown"))
            val = r.get(val_key, 0)
            if not isinstance(val, (int, float)):
                try:
                    val = float(val)
                except Exception:
                    val = 0
            chart_data.append({
                cat_key: cat_val,
                val_key: val,
                "value": val,
                "category": cat_val,
                "x": cat_val,
                "y": val,
                "percentage": r.get("percentage")
            })

        # Reject charts where all values are 0 (e.g. unaggregated or corrupted data)
        if all(d["value"] == 0 for d in chart_data):
            return None

        top_cat = chart_data[0]["category"]
        top_val = chart_data[0]["value"]

        return {
            "chart_type": chart_type,
            "title": title.title(),
            "subtitle": f"Exact distribution across {len(chart_data)} categories",
            "x_axis": cat_key,
            "y_axis": val_key,
            "data": chart_data,
            "takeaway": f"Dominant concentration at '{top_cat}' with {top_val} {filter_label.lower()}cases/records.",
            "suitability_reason": f"{chart_type.capitalize()} chart accurately reflects discrete distribution across categories."
        }
