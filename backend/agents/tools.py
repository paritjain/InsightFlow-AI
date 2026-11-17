import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional

from engine.cleaner import coerce_numeric_columns
from engine.semantic import SemanticLayer
from engine.ask_data import ask_your_data
from engine.eda import validate_chart_spec
from engine.analytical_engine import UniversalAnalyticalChatEngine


class DatasetQueryTool:
    """
    Dataset Query Tool:
    Calculates aggregates, filtered metrics, rankings, and mathematical operations
    using DuckDB in-memory execution and UniversalAnalyticalChatEngine.
    """

    @staticmethod
    def run(
        df: pd.DataFrame,
        sem_layer: dict = None,
        query: str = "",
        contract: Optional[Any] = None,
        dictionary: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        project_id: str = "project_active",
        confirmed_field: Optional[str] = None
    ) -> Dict[str, Any]:
        if df is None or df.empty:
            return {
                "success": False,
                "error": "Dataset is empty or uninitialized.",
                "answer": "No dataset is currently available for querying.",
                "calculation_formula": "NO_ACTIVE_DATASET"
            }

        try:
            engine = UniversalAnalyticalChatEngine(
                df=df,
                contract=contract,
                dictionary=dictionary,
                sem_layer_dict=sem_layer,
                project_id=project_id
            )
            response = engine.process_query(query=query, context=context, df=df, confirmed_field=confirmed_field)
            resp_dict = response.to_dict()

            return {
                "success": response.status != "ERROR",
                "tool": "DatasetQueryTool",
                "answer": response.answer,
                "calculation_formula": response.calculation_formula,
                "numerical_value": response.numerical_value,
                "plan": response.query_plan,
                "result_table": response.evidence,
                "evidence": response.evidence,
                "chart_spec": response.chart_spec,
                "status": response.status,
                "warnings": response.warnings,
                "records_evaluated": response.records_evaluated,
                "coverage_pct": response.coverage_pct,
                "match_suggestions": response.match_suggestions,
                "suggestion_mode": response.suggestion_mode,
                "resolved_field": response.resolved_field,
                "confirmed_field": response.confirmed_field,
                "interactive_options": response.interactive_options,
                "response_contract": resp_dict
            }
        except Exception as e:
            try:
                result = ask_your_data(df, query)
                return {
                    "success": True,
                    "tool": "DatasetQueryTool",
                    "answer": result.get("answer", ""),
                    "calculation_formula": result.get("calculation_formula", ""),
                    "numerical_value": result.get("numerical_value"),
                    "plan": result.get("query_plan"),
                    "result_table": result.get("evidence", [])
                }
            except Exception as legacy_err:
                return {
                    "success": False,
                    "tool": "DatasetQueryTool",
                    "error": str(e),
                    "answer": f"Unable to compute mathematical query from dataset: {str(e)}",
                    "calculation_formula": "QUERY_EXECUTION_ERROR"
                }


class EDATool:
    """
    EDA Tool:
    Retrieves and runs exploratory data analysis metrics, distributions, correlations,
    outliers, missing data summaries, and PCA dimensionality reduction.
    """

    @staticmethod
    def get_correlations(eda_results: dict, var1: str = None, var2: str = None) -> Dict[str, Any]:
        relationships = eda_results.get("relationships", {})
        strong_assocs = relationships.get("strong_associations", [])
        matrix = relationships.get("pearson_matrix", {})

        if var1 and var2:
            # Check specific pair
            v1_low, v2_low = var1.lower(), var2.lower()
            matching = [
                a for a in strong_assocs
                if (a.get("var1", "").lower() in [v1_low, v2_low] or a.get("metric_1", "").lower() in [v1_low, v2_low])
                and (a.get("var2", "").lower() in [v1_low, v2_low] or a.get("metric_2", "").lower() in [v1_low, v2_low])
            ]
            if matching:
                m = matching[0]
                return {
                    "tool": "EDATool",
                    "type": "specific_correlation",
                    "pair": f"{var1} vs {var2}",
                    "pearson_r": m.get("pearson_r"),
                    "spearman_r": m.get("spearman_r"),
                    "strength": m.get("strength"),
                    "interpretation": m.get("interpretation")
                }

        return {
            "tool": "EDATool",
            "type": "correlation_summary",
            "total_significant_pairs": len(strong_assocs),
            "top_associations": strong_assocs[:5],
            "correlation_matrix": matrix
        }

    @staticmethod
    def get_outliers(eda_results: dict, col: str = None) -> Dict[str, Any]:
        num_summary = eda_results.get("numeric_summary", {})
        adv_outliers = eda_results.get("advanced_outliers", {})

        if col and col in num_summary:
            stat = num_summary[col]
            return {
                "tool": "EDATool",
                "column": col,
                "outliers_count": stat.get("outliers_count", 0),
                "outliers_percentage": stat.get("outliers_percentage", 0),
                "iqr": stat.get("iqr"),
                "median": stat.get("median"),
                "min": stat.get("min"),
                "max": stat.get("max")
            }

        # Global outlier summary
        outlier_cols = [
            {"column": c, "outliers_count": s.get("outliers_count", 0), "pct": s.get("outliers_percentage", 0)}
            for c, s in num_summary.items() if s.get("outliers_count", 0) > 0
        ]
        outlier_cols.sort(key=lambda x: x["outliers_count"], reverse=True)

        return {
            "tool": "EDATool",
            "columns_with_outliers": outlier_cols,
            "multivariate_anomalies": adv_outliers.get("multivariate_anomalies_count", 0)
        }

    @staticmethod
    def get_missing_values(df: pd.DataFrame, eda_results: dict) -> Dict[str, Any]:
        overview = eda_results.get("overview", {})
        null_cols = []
        if df is not None and not df.empty:
            for c in df.columns:
                n_null = int(df[c].isna().sum())
                if n_null > 0:
                    null_cols.append({
                        "column": c,
                        "missing_count": n_null,
                        "missing_pct": round(n_null / len(df) * 100, 2)
                    })
            null_cols.sort(key=lambda x: x["missing_count"], reverse=True)

        return {
            "tool": "EDATool",
            "overall_null_pct": overview.get("overall_null_percentage", 0.0),
            "columns_with_nulls": null_cols,
            "clean_columns_count": len(df.columns) - len(null_cols) if df is not None else 0
        }

    @staticmethod
    def get_pca(eda_results: dict) -> Dict[str, Any]:
        pca = eda_results.get("pca_analysis", {})
        return {
            "tool": "EDATool",
            "pca_analysis": pca
        }


class ChartGenerationTool:
    """
    Chart Generation Tool:
    Generates verified interactive chart specifications (Plotly / Recharts)
    grounded in the active dataset.
    """

    @staticmethod
    def run(df: pd.DataFrame, sem_layer: dict, query: str, x_axis: str = None, y_axis: str = None, chart_type: str = None) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"success": False, "error": "No active dataset."}

        q_low = query.lower()
        sem = sem_layer or {}

        # Discover base types from sem_layer, supplemented with physical types in df
        num_cols = [c for c, m in sem.items() if m.get("is_measure") and not m.get("is_identifier")]
        dim_cols = [c for c, m in sem.items() if m.get("is_dimension") and not m.get("is_identifier")]
        time_cols = [c for c, m in sem.items() if m.get("is_time") or m.get("semantic_role") == "Date/Time"]

        # Ensure physical types in df govern actual numeric reality
        for c in df.columns:
            m = sem.get(c, {})
            is_id = m.get("is_identifier", False) or any(id_w in c.lower() for id_w in ["_id", "id", "uuid", "code", "key", "#"])
            if pd.api.types.is_numeric_dtype(df[c]):
                if not is_id and c not in num_cols:
                    num_cols.append(c)
                # If continuous numeric or measure, ensure it is NOT treated as a dimension
                if c in dim_cols and df[c].nunique() > 10 and not m.get("is_numeric_category_code", False):
                    dim_cols.remove(c)
            elif pd.api.types.is_datetime64_any_dtype(df[c]) or any(t_w in c.lower() for t_w in ["date", "time", "timestamp", "year", "month"]):
                if c not in time_cols:
                    time_cols.append(c)
                if c in dim_cols:
                    dim_cols.remove(c)
            else:
                if not is_id and c not in dim_cols:
                    dim_cols.append(c)
                if c in num_cols:
                    num_cols.remove(c)

        # Resolve requested or inferred chart type
        c_type = chart_type
        if not c_type:
            if any(k in q_low for k in ["scatter", "correlation plot", "vs", "versus"]):
                c_type = "SCATTER"
            elif any(k in q_low for k in ["hist", "histogram", "distribution", "spread"]):
                c_type = "HISTOGRAM"
            elif any(k in q_low for k in ["box", "boxplot", "box plot"]):
                c_type = "BOX_PLOT"
            elif any(k in q_low for k in ["line", "trend", "over time", "monthly"]):
                c_type = "LINE"
            elif any(k in q_low for k in ["donut", "pie", "share", "proportion"]):
                c_type = "DONUT"
            else:
                c_type = "BAR"

        # Match columns explicitly mentioned in the query text
        mentioned_cols = []
        for c in df.columns:
            c_clean = c.lower().replace("_", " ")
            pat = r'\b' + re.escape(c.lower()) + r'\b|\b' + re.escape(c_clean) + r'\b'
            if re.search(pat, q_low):
                mentioned_cols.append(c)

        resolved_x = x_axis
        resolved_y = y_axis

        # Map mentioned columns into x and y based on role and chart type
        if mentioned_cols:
            m_time = [c for c in mentioned_cols if c in time_cols]
            m_num = [c for c in mentioned_cols if c in num_cols]
            m_dim = [c for c in mentioned_cols if c in dim_cols and c not in num_cols]

            if c_type == "LINE":
                if not resolved_x and m_time:
                    resolved_x = m_time[0]
                if not resolved_y and m_num:
                    resolved_y = m_num[0]
            elif c_type == "SCATTER":
                if len(m_num) >= 2:
                    if not resolved_x:
                        resolved_x = m_num[0]
                    if not resolved_y:
                        resolved_y = m_num[1]
                elif m_num:
                    if not resolved_x:
                        resolved_x = m_num[0]
            else:
                if not resolved_x and m_dim:
                    resolved_x = m_dim[0]
                elif not resolved_x and m_time:
                    resolved_x = m_time[0]

                if not resolved_y and m_num:
                    resolved_y = m_num[0]

        # Fallbacks if still unresolved
        if not resolved_x:
            if c_type == "LINE" and time_cols:
                resolved_x = time_cols[0]
            elif c_type == "SCATTER" and len(num_cols) >= 2:
                resolved_x = num_cols[0]
                resolved_y = num_cols[1]
            elif dim_cols:
                resolved_x = dim_cols[0]
            elif time_cols:
                resolved_x = time_cols[0]
            elif num_cols:
                resolved_x = num_cols[0]

        if not resolved_y:
            if num_cols:
                candidates = [c for c in num_cols if c != resolved_x]
                resolved_y = candidates[0] if candidates else num_cols[0]
            elif resolved_x and resolved_x in df.columns:
                sub_df = df
                filter_label = ""
                for col in df.columns:
                    for val in df[col].dropna().astype(str).unique()[:50]:
                        if len(val) >= 3 and val.lower() in q_low:
                            sub_df = df[df[col].astype(str).str.lower() == val.lower()]
                            filter_label = f"{val} "
                            break
                    if filter_label:
                        break

                val_counts = sub_df[resolved_x].dropna().astype(str).value_counts()
                c_data = [{"x": str(k), "y": int(v), "value": int(v), resolved_x: str(k), "count": int(v)} for k, v in val_counts.items()]
                chart_title = f"{filter_label}Cases by {resolved_x}".title()
                return {
                    "success": True,
                    "tool": "ChartGenerationTool",
                    "chart_spec": {
                        "chart_type": "BAR" if len(c_data) <= 6 else "HORIZONTAL_BAR",
                        "title": chart_title,
                        "subtitle": f"Exact count breakdown across {len(c_data)} categories",
                        "x_axis": resolved_x,
                        "y_axis": "count",
                        "data": c_data
                    },
                    "message": f"Generated interactive chart for '{chart_title}'."
                }

        agg_method = "mean" if any(w in q_low for w in ["average", "avg", "mean", "rate"]) else "sum"

        params = {
            "x_axis": resolved_x,
            "y_axis": resolved_y,
            "dim_col": resolved_x,
            "m_col": resolved_y,
            "measure": resolved_y,
            "agg": agg_method
        }

        spec = validate_chart_spec(c_type, df, params)
        if spec.get("is_valid"):
            return {
                "success": True,
                "tool": "ChartGenerationTool",
                "chart_spec": {
                    "chart_type": c_type,
                    "title": f"{resolved_y} by {resolved_x}" if resolved_y and resolved_x else f"{c_type} Analysis",
                    "x_axis": resolved_x,
                    "y_axis": resolved_y,
                    "data": spec.get("data", [])
                },
                "message": f"Generated interactive {c_type} chart for '{resolved_x}' and '{resolved_y}'."
            }
        else:
            return {
                "success": False,
                "tool": "ChartGenerationTool",
                "error": spec.get("fallback_message", "Could not generate valid chart for selected parameters.")
            }


class InsightTool:
    """
    Insight Tool:
    Retrieves evidence-backed analytical insights and strategic implications.
    """

    @staticmethod
    def get_summary_findings(insights: list, eda_results: dict, query: str = "") -> Dict[str, Any]:
        overview = eda_results.get("overview", {}) if isinstance(eda_results, dict) else {}
        top_insights = insights[:5] if insights else []

        findings = []
        for ins in top_insights:
            findings.append({
                "title": ins.get("title"),
                "what_happened": ins.get("insight") or ins.get("what_happened") or ins.get("observed_fact", ""),
                "why_it_matters": ins.get("why_it_matters") or ins.get("business_interpretation", ""),
                "priority": ins.get("priority", "HIGH"),
                "evidence": ins.get("numerical_evidence", [])
            })

        q_low = query.lower() if query else ""
        op_eda = eda_results.get("operational_eda", {}) if isinstance(eda_results, dict) else {}

        if any(k in q_low for k in ["prioritize", "priority", "priorities", "what should management", "next step", "what to do"]):
            next_steps = op_eda.get("what_should_we_do_next", {})
            if isinstance(next_steps, dict):
                priorities = next_steps.get("recommended_operational_actions", [])
            elif isinstance(next_steps, list):
                priorities = next_steps
            else:
                priorities = []
            return {
                "tool": "InsightTool",
                "mode": "PRIORITIES",
                "dataset_rows": overview.get("num_rows", 0),
                "dataset_columns": overview.get("num_columns", 0),
                "health_score": overview.get("health_score", 100),
                "top_findings": findings,
                "priorities": priorities
            }

        if any(k in q_low for k in ["problem", "problems", "gap", "gaps", "lacking", "deviating", "issue", "issues", "anomaly"]):
            lacking = op_eda.get("what_is_lacking_or_deviating", {})
            return {
                "tool": "InsightTool",
                "mode": "GAPS",
                "dataset_rows": overview.get("num_rows", 0),
                "dataset_columns": overview.get("num_columns", 0),
                "health_score": overview.get("health_score", 100),
                "top_findings": findings,
                "lacking": lacking
            }

        return {
            "tool": "InsightTool",
            "mode": "GENERAL",
            "dataset_rows": overview.get("num_rows", 0),
            "dataset_columns": overview.get("num_columns", 0),
            "health_score": overview.get("health_score", 100),
            "top_findings": findings,
            "operational_eda": op_eda
        }


class _PDFToolRunDispatcher:
    def __get__(self, instance, owner):
        if instance is None:
            return owner._run_internal
        def bound_run(query: str, **kwargs):
            if "document_corpus" not in kwargs and instance.engine_or_corpus is not None:
                kwargs["document_corpus"] = instance.engine_or_corpus
            return owner._run_internal(query=query, **kwargs)
        return bound_run


class PDFIntelligenceTool:
    """
    PDF and Document Intelligence Tool:
    Answers natural language questions about uploaded PDFs, reports, or text documentation.
    Extracts facts, summarizes documents, and retrieves grounded page-specific answers.
    """
    run = _PDFToolRunDispatcher()

    def __init__(self, engine_or_corpus: Optional[Any] = None):
        self.engine_or_corpus = engine_or_corpus

    def __call__(self, query: str, **kwargs) -> Dict[str, Any]:
        return self.run(query=query, **kwargs)

    @classmethod
    def _run_internal(
        cls,
        query: str,
        document_corpus: Optional[Dict[str, Any]] = None,
        pdf_path: Optional[str] = None,
        project_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        from engine.pdf_engine import DocumentIntelligenceEngine
        if not document_corpus and pdf_path:
            try:
                document_corpus = DocumentIntelligenceEngine.extract_document(pdf_path)
            except Exception as e:
                logger.warning(f"Could not extract PDF at {pdf_path}: {e}")

        if hasattr(document_corpus, "document_corpus"):
            document_corpus = document_corpus.document_corpus

        # If document_corpus is a list of document dicts or engines
        if isinstance(document_corpus, list) and document_corpus:
            matching = next((doc for doc in document_corpus if isinstance(doc, dict) and doc.get("file_name", "").lower() in query.lower()), None)
            document_corpus = matching if matching is not None else document_corpus[0]
            if hasattr(document_corpus, "document_corpus"):
                document_corpus = document_corpus.document_corpus

        # If document_corpus is a dictionary mapping filename -> corpus or engine
        if isinstance(document_corpus, dict) and "chunks" not in document_corpus and "passages" not in document_corpus:
            matching = next((doc for name, doc in document_corpus.items() if name.lower() in query.lower()), None)
            if matching is None and document_corpus:
                matching = next(iter(document_corpus.values()))
            document_corpus = matching
            if hasattr(document_corpus, "document_corpus"):
                document_corpus = document_corpus.document_corpus

        if not document_corpus:
            return {
                "status": "FAIL",
                "success": False,
                "tool": "DocumentIntelligenceTool",
                "answer": "No PDF document or text corpus is currently available for this project. Please upload a PDF file or generate an executive report to enable document intelligence.",
                "calculation_formula": "NO_PDF_DOCUMENT_LOADED",
                "evidence": [],
                "page_references": [],
                "source_file": "",
                "passages_retrieved": 0
            }

        res = DocumentIntelligenceEngine.answer_query(
            query=query,
            document_corpus=document_corpus,
            project_context=project_context
        )
        return {
            "status": "PASS",
            "success": True,
            "tool": "DocumentIntelligenceTool",
            "answer": res.get("answer", ""),
            "calculation_formula": f"DOCUMENT_RAG(file='{res.get('source_file')}', pages={res.get('page_references', [])})",
            "evidence": res.get("evidence", []),
            "page_references": res.get("page_references", []),
            "source_file": res.get("source_file", ""),
            "passages_retrieved": res.get("passages_retrieved", len(res.get("evidence", [])))
        }

