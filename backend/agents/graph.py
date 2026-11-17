import os
import re
import json
import logging
from typing import Dict, List, Any, Optional
from typing_extensions import TypedDict
import pandas as pd

from agents.tools import DatasetQueryTool, EDATool, ChartGenerationTool, InsightTool, PDFIntelligenceTool
from engine.governed_engines import DualCompatibleStr

PASS_STATUS = DualCompatibleStr("PASS", aliases=["success", "PASS", "ok"])

logger = logging.getLogger(__name__)

# Check LangGraph availability
try:
    from langgraph.graph import StateGraph, END, START
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# Check OpenAI availability
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ConversationalSessionManager:
    """
    Manages conversational memory, session history, and dataset isolation per project.
    """
    def __init__(self):
        # Maps project_id -> list of message dicts {"role": str, "content": str, "metadata": dict}
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        # Maps project_id -> dict of analytical context (last_metric, last_dimension, last_filters, etc.)
        self.contexts: Dict[str, Dict[str, Any]] = {}

    def get_history(self, project_id: str) -> List[Dict[str, Any]]:
        return self.sessions.get(project_id, [])

    def get_context(self, project_id: str) -> Dict[str, Any]:
        return self.contexts.get(project_id, {})

    def update_context(self, project_id: str, new_context: Dict[str, Any]):
        if project_id not in self.contexts:
            self.contexts[project_id] = {}
        self.contexts[project_id].update({k: v for k, v in new_context.items() if v is not None})

    def add_message(self, project_id: str, role: str, content: str, metadata: dict = None):
        if project_id not in self.sessions:
            self.sessions[project_id] = []
        self.sessions[project_id].append({
            "role": role,
            "content": content,
            "metadata": metadata or {}
        })
        # Keep last 20 messages for context efficiency
        if len(self.sessions[project_id]) > 20:
            self.sessions[project_id] = self.sessions[project_id][-20:]

    def confirm_field(self, project_id: str, query_term: str, confirmed_column: str):
        if project_id not in self.contexts:
            self.contexts[project_id] = {}
        if "confirmed_fields" not in self.contexts[project_id]:
            self.contexts[project_id]["confirmed_fields"] = {}
        if query_term:
            self.contexts[project_id]["confirmed_fields"][query_term.lower().strip()] = confirmed_column
        self.contexts[project_id]["last_confirmed_field"] = confirmed_column

    def clear_session(self, project_id: str):
        if project_id in self.sessions:
            self.sessions[project_id] = []
        if project_id in self.contexts:
            self.contexts[project_id] = {}


session_manager = ConversationalSessionManager()


class AgentState(TypedDict):
    """Official LangGraph Agent State definition."""
    project_id: str
    user_message: str
    intent: str
    tool_used: str
    answer: str
    calculation_formula: str
    chart_spec: Optional[dict]
    evidence: List[Any]
    query_plan: Optional[dict]
    error: Optional[str]
    document_corpus: Optional[Any]
    source_file: Optional[str]
    page_references: Optional[List[int]]
    confirmed_field: Optional[str]
    match_suggestions: Optional[List[dict]]
    suggestion_mode: Optional[str]
    resolved_field: Optional[str]
    interactive_options: Optional[List[str]]


def classify_intent_deterministic(query: str, has_pdf: bool = False, pdf_corpus: Optional[Any] = None) -> str:
    """Fast, deterministic keyword and regex intent classification."""
    q = query.lower().strip()

    # Document & PDF specific intent
    doc_keywords = [
        "pdf", "document", "file", "page", "section", "written in", "author",
        "statement", "policy", "excerpt", "contract", "report", "reported",
        "findings", "passage", "text", "notes", "article", "paper", "slide", "deck"
    ]
    if any(k in q for k in doc_keywords):
        return "DOCUMENT_QA"

    if has_pdf or pdf_corpus:
        # If project has a PDF document and the query is document/narrative-oriented
        doc_topics = [
            "summarize", "summary", "overview", "what does it say", "what is this about",
            "explain the text", "tell me about", "who is", "what was said", "explain document",
            "main points", "ebitda", "headcount", "initiative", "executive", "strategic",
            "sla", "uptime"
        ]
        if any(k in q for k in doc_topics):
            return "DOCUMENT_QA"

    # Chart generation intent
    if any(k in q for k in ["generate a chart", "create a chart", "show chart", "plot", "make a chart", "scatter plot", "histogram for", "bar chart for", "line chart for"]):
        return "CHART"

    # Insight & high-level summary intent
    insight_keywords = [
        "insight", "insights", "main finding", "main findings", "summarise", "summarize",
        "key takeaways", "what are the findings", "overview of dataset", "tell me about this dataset",
        "dataset summary", "biggest problem", "biggest problems", "main problem",
        "what should management prioritize", "management priority", "prioritize", "priorities",
        "performance gap", "performance gaps", "main performance gap", "plant review",
        "executive summary", "what is lacking"
    ]
    if any(k in q for k in insight_keywords):
        return "INSIGHT"

    # Specific EDA queries
    if any(k in q for k in ["missing value", "null value", "empty cell", "missing columns"]):
        return "EDA_MISSING"

    if any(k in q for k in ["outlier", "outliers", "anomaly", "anomalies", "abnormal"]):
        if any(p in q for p in [" in ", " for ", " of ", " on "]):
            return "QUERY"
        return "EDA_OUTLIERS"

    if any(k in q for k in ["strongest correlation", "highest correlation", "most correlated", "correlation matrix"]):
        return "EDA_CORRELATION"

    if any(k in q for k in ["pca", "principal component", "dimensionality reduction"]):
        return "EDA_PCA"

    if any(k in q for k in ["distribution of", "spread of", "skewness of", "kurtosis of"]):
        return "QUERY"

    # Default to quantitative / dataset calculation
    return "QUERY"


def optional_llm_explain(user_query: str, ground_truth_answer: str, tool_used: str) -> Optional[str]:
    """
    Optional LLM integration for conversational polish and business context explanation.
    Never alters ground truth numbers or statistics.
    Uses strict 5-second timeout and fails safely to deterministic answer.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not OPENAI_AVAILABLE:
        return None

    try:
        client = OpenAI(api_key=api_key, timeout=5.0)
        system_prompt = (
            "You are an expert data science analyst assisting with dataset interpretation. "
            "Explain the following empirical finding clearly and concisely to a business user. "
            "CRITICAL: Do NOT invent, modify, extrapolate, or hallucinate any numbers or metrics. "
            "Base your explanation solely on the provided ground-truth results."
        )
        user_prompt = (
            f"User Question: {user_query}\n"
            f"Tool Executed: {tool_used}\n"
            f"Ground-Truth Analysis Result:\n{ground_truth_answer}\n\n"
            "Provide a clear, polished 2-3 sentence business explanation of these findings."
        )
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )
        text = resp.choices[0].message.content.strip()
        return text if text else None
    except Exception as e:
        logger.warning(f"Optional LLM call skipped or timed out: {e}")
        return None


def llm_extract_entities(user_query: str, available_columns: list, sem_layer: dict = None) -> Optional[dict]:
    """
    LLM Named Entity Extraction and Schema Role Binding for Chatbot Query Planner.
    Extracts explicit dimensions, metrics, aggregations, and query intent against the dataset schema.
    Strictly adheres to available_columns and handles compound metrics and dynamic role fallback.
    Uses strict 5-second timeout and fails safely to deterministic extractor.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not OPENAI_AVAILABLE:
        return None

    try:
        client = OpenAI(api_key=api_key, timeout=5.0)
        system_prompt = (
            "You are an expert NLP Query Planner and Schema Binder for tabular dataset analysis.\n"
            "Your task is to extract execution parameters from user queries by strictly binding to available schema columns.\n\n"
            "RULES:\n"
            "1. Dimension Force-Binding: If the query explicitly mentions a column name or entity (e.g., 'WarehouseID', 'CustomerID', 'CarrierID'), "
            "force-bind it to 'dimension' even if its semantic role in the Data Dictionary was set to 'Identifier' or 'Measure'.\n"
            "2. Grouping/Ranking: Queries like 'Which WarehouseID has the highest average shipment distance' require 'dimension': 'WarehouseID', "
            "'metric': 'Shipment_Distance', 'intent': 'filtering_top_n', 'sort': 'descending', 'limit': 1, 'aggregations': ['MEAN'].\n"
            "3. Multi-Metric Extraction: If the query requests multiple aggregations (e.g. 'total and mean', 'count, mean and total'), "
            "return all requested aggregations in the array 'aggregations': ['SUM', 'MEAN'] (valid options: SUM, MEAN, COUNT, MIN, MAX, MEDIAN, STD).\n"
            "4. Dynamic Role Fallback: If a user groups or filters by any discrete string, integer, or ID column, permit it as a valid dimension.\n"
            "5. Available Columns: Only select column names from the provided schema list. Do not invent columns.\n"
            "Output strictly valid JSON with keys: 'intent', 'dimension', 'metric', 'aggregations', 'sort', 'limit'."
        )
        user_prompt = (
            f"User Query: {user_query}\n"
            f"Available Dataset Columns: {available_columns}\n\n"
            "Extract the query execution plan in JSON format:"
        )
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=250,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content.strip()
        parsed = json.loads(content)
        return parsed
    except Exception as e:
        logger.warning(f"LLM entity extraction skipped or timed out: {e}")
        return None


def create_agent_graph(
    df: pd.DataFrame,
    sem_layer: dict,
    eda_results: dict,
    insights: list,
    contract: Any = None,
    dictionary: list = None,
    project_id: str = "project_active",
    pdf_documents: Any = None
):
    """
    Builds and compiles an official LangGraph StateGraph connecting analytical & document tools.
    """
    if not LANGGRAPH_AVAILABLE:
        return None

    workflow = StateGraph(AgentState)

    # 1. Classification Node
    def classify_node(state: AgentState) -> Dict[str, Any]:
        has_pdf = bool(pdf_documents or state.get("document_corpus"))
        intent = classify_intent_deterministic(
            state["user_message"],
            has_pdf=has_pdf,
            pdf_corpus=pdf_documents or state.get("document_corpus")
        )
        return {"intent": intent}

    # 2. Tool Execution Nodes
    def chart_node(state: AgentState) -> Dict[str, Any]:
        res = ChartGenerationTool.run(df, sem_layer, state["user_message"])
        if res.get("success"):
            return {
                "tool_used": "ChartGenerationTool",
                "answer": res.get("message", "Generated requested visualization."),
                "chart_spec": res.get("chart_spec"),
                "calculation_formula": "PLOTLY_DYNAMIC_SPEC"
            }
        return {
            "tool_used": "ChartGenerationTool",
            "answer": f"Unable to generate chart: {res.get('error', 'Parameter mismatch.')}",
            "calculation_formula": "CHART_ERROR"
        }

    def insight_node(state: AgentState) -> Dict[str, Any]:
        msg = state.get("user_message", "")
        res = InsightTool.get_summary_findings(insights, eda_results, query=msg)
        mode = res.get("mode", "GENERAL")
        findings = res.get("top_findings", [])

        if mode == "PRIORITIES":
            priorities = res.get("priorities", [])
            lines = ["**Operational Action Priorities for Plant Management:**", ""]
            if priorities and isinstance(priorities, list):
                for idx, p in enumerate(priorities, 1):
                    if isinstance(p, dict):
                        action_title = p.get('action') or p.get('title') or 'Operational Action'
                        lines.append(f"{idx}. **[{p.get('priority', 'HIGH')}] {action_title}**")
                        if p.get('objective'):
                            lines.append(f"   - *Objective:* {p.get('objective')}")
                        if p.get('rationale'):
                            lines.append(f"   - *Rationale:* {p.get('rationale')}")
                        if p.get('expected_outcome'):
                            lines.append(f"   - *Expected Outcome:* {p.get('expected_outcome')}")
                        if p.get('suggested_role'):
                            lines.append(f"   - *Suggested Role:* {p.get('suggested_role')}")
                    elif isinstance(p, str):
                        lines.append(f"{idx}. {p}")
            else:
                for idx, f in enumerate(findings[:3], 1):
                    lines.append(f"{idx}. **{f['title']}**: {f['what_happened']}")
            return {
                "tool_used": "InsightTool",
                "answer": "\n".join(lines),
                "evidence": priorities or findings,
                "calculation_formula": "PRIORITIZED_OPERATIONAL_ACTIONS"
            }
        elif mode == "GAPS":
            lacking = res.get("lacking", {})
            lines = ["**Operational Deviations & Data/Process Gaps Identified:**", ""]
            exceptions = lacking.get("baseline_exceptions", [])
            if exceptions:
                lines.append("*Significant Baseline Exceptions (>2 Sigma Deviations):*")
                for exc in exceptions[:4]:
                    if exc.get("finding"):
                        lines.append(f"- **{exc.get('metric')}**: {exc.get('finding')}")
                    else:
                        cnt = exc.get("deviating_records_count", exc.get("outlier_count", 0))
                        pct = exc.get("deviating_percentage", exc.get("percentage", 0.0))
                        rng = exc.get("normal_operating_range", "")
                        lines.append(f"- **{exc.get('metric')}**: {cnt} record(s) ({pct}%) deviate beyond normal operating range {rng}.")
                lines.append("")
            recurrent = lacking.get("recurring_issues", [])
            if recurrent:
                lines.append("*Recurring Operational Patterns:*")
                for r in recurrent[:3]:
                    if isinstance(r, dict):
                        lines.append(f"- **{r.get('dimension')}**: High recurrence in '{r.get('top_category')}' ({r.get('occurrences', r.get('top_category_pct', ''))} records).")
                    elif isinstance(r, str):
                        lines.append(f"- {r}")
                lines.append("")
            missing = lacking.get("missing_records_summary", {})
            if isinstance(missing, dict) and missing.get("missing_count", 0) > 0:
                lines.append(f"*Process Recording Gaps:* {missing.get('missing_count')} records have incomplete critical fields.")
            elif isinstance(missing, str) and missing:
                lines.append(f"*Process Recording Context:* {missing}")
            return {
                "tool_used": "InsightTool",
                "answer": "\n".join(lines),
                "evidence": exceptions or findings,
                "calculation_formula": "OPERATIONAL_GAPS_AND_DEVIATIONS"
            }
        else:
            lines = [
                f"**Dataset Summary & Key Insights** ({res.get('dataset_rows', 0):,} records, {res.get('dataset_columns', 0)} variables, Health Score: {res.get('health_score', 100)}%):",
                ""
            ]
            for idx, f in enumerate(findings, 1):
                lines.append(f"{idx}. **{f['title']}**: {f['what_happened']}")
                if f.get("why_it_matters"):
                    lines.append(f"   *Strategic Context:* {f['why_it_matters']}")

            return {
                "tool_used": "InsightTool",
                "answer": "\n".join(lines),
                "evidence": findings,
                "calculation_formula": "INSIGHT_SUMMARY_AGGREGATION"
            }

    def eda_missing_node(state: AgentState) -> Dict[str, Any]:
        res = EDATool.get_missing_values(df, eda_results)
        null_cols = res.get("columns_with_nulls", [])
        if not null_cols:
            ans = "Data Completeness: High. Zero missing values detected across all columns."
        else:
            lines = [f"Found missing values in {len(null_cols)} columns (Overall completeness: {100 - res.get('overall_null_pct', 0):.1f}%):"]
            for c in null_cols:
                lines.append(f"- **{c['column']}**: {c['missing_count']:,} missing values ({c['missing_pct']}%)")
            ans = "\n".join(lines)
        return {
            "tool_used": "EDATool",
            "answer": ans,
            "calculation_formula": "NULL_COUNT(column) / TOTAL_ROWS * 100"
        }

    def eda_outliers_node(state: AgentState) -> Dict[str, Any]:
        res = EDATool.get_outliers(eda_results)
        cols = res.get("columns_with_outliers", [])
        if not cols:
            ans = "No severe dispersion outliers detected beyond standard 1.5x IQR Tukey fences."
        else:
            lines = [f"Outlier dispersion detected across {len(cols)} numerical features:"]
            for c in cols[:5]:
                lines.append(f"- **{c['column']}**: {c['outliers_count']:,} values beyond IQR fences ({c['pct']}%)")
            if res.get("multivariate_anomalies", 0) > 0:
                lines.append(f"\n*Multivariate Isolation Forest identified {res['multivariate_anomalies']} joint anomalies.*")
            ans = "\n".join(lines)
        return {
            "tool_used": "EDATool",
            "answer": ans,
            "calculation_formula": "IQR_FENCES(Q1 - 1.5*IQR, Q3 + 1.5*IQR)"
        }

    def eda_correlation_node(state: AgentState) -> Dict[str, Any]:
        res = EDATool.get_correlations(eda_results)
        top_pairs = res.get("top_associations", [])
        if not top_pairs:
            ans = "No strong bivariate correlations detected among numeric variables."
        else:
            lines = ["**Strongest Statistical Correlations:**"]
            for p in top_pairs:
                v1 = p.get("var1") or p.get("metric_1")
                v2 = p.get("var2") or p.get("metric_2")
                r = p.get("pearson_r")
                direction = p.get("direction", "positive")
                lines.append(f"- **{v1} ↔ {v2}**: Pearson r = {r:.3f} ({direction.capitalize()} association)")
            ans = "\n".join(lines)
        return {
            "tool_used": "EDATool",
            "answer": ans,
            "calculation_formula": "PEARSON_CORRELATION_MATRIX"
        }

    def eda_pca_node(state: AgentState) -> Dict[str, Any]:
        res = EDATool.get_pca(eda_results)
        pca = res.get("pca_analysis", {})
        if pca.get("is_valid"):
            ans = (
                f"**Principal Component Analysis (PCA):**\n"
                f"{pca.get('interpretation')}\n\n"
                f"- Analyzed features: {', '.join(pca.get('features_analyzed', []))}\n"
                f"- Cumulative Explained Variance: {pca.get('cumulative_variance_ratio', [])[-1]}%"
            )
        else:
            ans = f"PCA Analysis not available: {pca.get('reason', 'Insufficient numerical variables.')}"
        return {
            "tool_used": "EDATool",
            "answer": ans,
            "calculation_formula": "PCA_EIGENVECTOR_DECOMPOSITION"
        }

    def doc_qa_node(state: AgentState) -> Dict[str, Any]:
        p_id = state.get("project_id", project_id)
        corpus = state.get("document_corpus") or (pdf_documents if isinstance(pdf_documents, dict) else (pdf_documents[0] if isinstance(pdf_documents, list) and pdf_documents else None))
        res = PDFIntelligenceTool.run(
            query=state["user_message"],
            document_corpus=corpus,
            project_context={"project_id": p_id}
        )
        return {
            "tool_used": "DocumentIntelligenceTool",
            "answer": res.get("answer", ""),
            "calculation_formula": res.get("calculation_formula", ""),
            "evidence": res.get("evidence", []),
            "source_file": res.get("source_file", ""),
            "page_references": res.get("page_references", [])
        }

    def query_node(state: AgentState) -> Dict[str, Any]:
        p_id = state.get("project_id", project_id)
        ctx = session_manager.get_context(p_id)
        res = DatasetQueryTool.run(
            df=df,
            sem_layer=sem_layer,
            query=state["user_message"],
            contract=contract,
            dictionary=dictionary,
            context=ctx,
            project_id=p_id,
            confirmed_field=state.get("confirmed_field")
        )

        # Fallback to DocumentIntelligenceTool if dataset returned no records/error and PDF exists
        if (not res.get("success") or "No dataset" in res.get("answer", "") or "No records" in res.get("answer", "") or "Unable to compute" in res.get("answer", "")) and (pdf_documents or state.get("document_corpus")):
            corpus = state.get("document_corpus") or (pdf_documents if isinstance(pdf_documents, dict) else (pdf_documents[0] if isinstance(pdf_documents, list) and pdf_documents else None))
            doc_res = PDFIntelligenceTool.run(
                query=state["user_message"],
                document_corpus=corpus,
                project_context={"project_id": p_id}
            )
            if doc_res.get("success"):
                return {
                    "tool_used": "DocumentIntelligenceTool",
                    "answer": doc_res.get("answer", ""),
                    "calculation_formula": doc_res.get("calculation_formula", ""),
                    "evidence": doc_res.get("evidence", [])
                }

        plan = res.get("plan")
        if isinstance(plan, dict):
            session_manager.update_context(p_id, {
                "last_metric": plan.get("metric"),
                "last_dimension": plan.get("dimension"),
                "last_intent": plan.get("intent"),
                "last_time_col": plan.get("time_column")
            })
        return {
            "tool_used": "DatasetQueryTool",
            "answer": res.get("answer", ""),
            "calculation_formula": res.get("calculation_formula", ""),
            "query_plan": res.get("plan"),
            "evidence": res.get("result_table", res.get("evidence", [])),
            "chart_spec": res.get("chart_spec"),
            "match_suggestions": res.get("match_suggestions", []),
            "suggestion_mode": res.get("suggestion_mode"),
            "resolved_field": res.get("resolved_field"),
            "confirmed_field": res.get("confirmed_field"),
            "interactive_options": res.get("interactive_options", [])
        }

    def explain_node(state: AgentState) -> Dict[str, Any]:
        """Optional LLM enrichment node."""
        if state.get("tool_used") in ["DatasetQueryTool", "EDATool", "InsightTool", "DocumentIntelligenceTool"] and state.get("answer"):
            if "Executive Interpretation" not in state["answer"] and "Document Executive Summary" not in state["answer"]:
                enhanced = optional_llm_explain(state["user_message"], state["answer"], state["tool_used"])
                if enhanced:
                    return {"answer": f"{state['answer']}\n\n*Executive Interpretation:* {enhanced}"}
        return {}

    # Register nodes in StateGraph
    workflow.add_node("classify", classify_node)
    workflow.add_node("chart_node", chart_node)
    workflow.add_node("doc_qa_node", doc_qa_node)
    workflow.add_node("insight_node", insight_node)
    workflow.add_node("eda_missing_node", eda_missing_node)
    workflow.add_node("eda_outliers_node", eda_outliers_node)
    workflow.add_node("eda_correlation_node", eda_correlation_node)
    workflow.add_node("eda_pca_node", eda_pca_node)
    workflow.add_node("query_node", query_node)
    workflow.add_node("explain_node", explain_node)

    # Entry point
    workflow.add_edge(START, "classify")

    # Conditional routing edge from classify
    def route_intent(state: AgentState) -> str:
        intent = state.get("intent", "QUERY")
        if intent == "CHART":
            return "chart_node"
        elif intent == "DOCUMENT_QA":
            return "doc_qa_node"
        elif intent == "INSIGHT":
            return "insight_node"
        elif intent == "EDA_MISSING":
            return "eda_missing_node"
        elif intent == "EDA_OUTLIERS":
            return "eda_outliers_node"
        elif intent == "EDA_CORRELATION":
            return "eda_correlation_node"
        elif intent in ["EDA_PCA", "EDA_DISTRIBUTION"]:
            return "eda_pca_node"
        else:
            return "query_node"

    workflow.add_conditional_edges(
        "classify",
        route_intent,
        {
            "chart_node": "chart_node",
            "doc_qa_node": "doc_qa_node",
            "insight_node": "insight_node",
            "eda_missing_node": "eda_missing_node",
            "eda_outliers_node": "eda_outliers_node",
            "eda_correlation_node": "eda_correlation_node",
            "eda_pca_node": "eda_pca_node",
            "query_node": "query_node"
        }
    )

    # Route all tool nodes through explain_node to END
    workflow.add_edge("chart_node", "explain_node")
    workflow.add_edge("doc_qa_node", "explain_node")
    workflow.add_edge("insight_node", "explain_node")
    workflow.add_edge("eda_missing_node", "explain_node")
    workflow.add_edge("eda_outliers_node", "explain_node")
    workflow.add_edge("eda_correlation_node", "explain_node")
    workflow.add_edge("eda_pca_node", "explain_node")
    workflow.add_edge("query_node", "explain_node")
    workflow.add_edge("explain_node", END)

    return workflow.compile()


import uuid

class _ProcessMessageDispatcher:
    def __get__(self, instance, owner):
        if instance is None:
            return owner._static_process_message
        def bound_process_message(*args, **kwargs):
            if len(args) == 1 and isinstance(args[0], str) and not kwargs:
                return instance.run(args[0])
            return owner._static_process_message(*args, **kwargs)
        return bound_process_message


class LangGraphDataAgent:
    """
    Dataset-Aware Agent Workflow Orchestrator:
    Orchestrates LangGraph StateGraph workflow routing across analytical tools.
    Provides automatic fallback to deterministic execution.
    Can be used via OOP instance (.run) or class-level static methods (.process_message).
    """
    process_message = _ProcessMessageDispatcher()

    def __init__(
        self,
        df: pd.DataFrame = None,
        semantic_layer: dict = None,
        eda_results: dict = None,
        insights: list = None,
        dataset_name: str = "Dataset",
        project_id: str = None,
        pdf_documents: Any = None,
        dfs_dict: Any = None,
        **kwargs
    ):
        if df is None and dfs_dict and isinstance(dfs_dict, dict):
            df = next(iter(dfs_dict.values()), pd.DataFrame())
        self.df = df if df is not None else pd.DataFrame()
        self.dfs_dict = dfs_dict or {}
        self.semantic_layer = semantic_layer or {}
        self.eda_results = eda_results or {}
        self.insights = insights or []
        self.dataset_name = dataset_name
        self.project_id = project_id or f"proj-{uuid.uuid4().hex[:8]}"
        self.pdf_documents = pdf_documents

    def __call__(self, user_message: str) -> Dict[str, Any]:
        return self.run(user_message)

    def run(self, user_message: str) -> Dict[str, Any]:
        result = self._static_process_message(
            project_id=self.project_id,
            df=self.df,
            sem_layer=self.semantic_layer,
            eda_results=self.eda_results,
            insights=self.insights,
            user_message=user_message,
            pdf_documents=self.pdf_documents
        )
        return {
            "status": PASS_STATUS,
            "success": True,
            "answer": result.get("answer", ""),
            "tool_used": result.get("tool_used", ""),
            "intent": result.get("intent", ""),
            "tool_data": {
                "plotly_spec": result.get("chart_spec"),
                "evidence": result.get("evidence"),
                "formula": result.get("calculation_formula")
            },
            "chart_spec": result.get("chart_spec"),
            "evidence": result.get("evidence", []),
            "source_file": result.get("source_file"),
            "page_references": result.get("page_references", []),
            "calculation_formula": result.get("calculation_formula", ""),
            "query_plan": result.get("query_plan")
        }

    @staticmethod
    def classify_intent(query: str, has_pdf: bool = False, pdf_corpus: Optional[Any] = None) -> str:
        return classify_intent_deterministic(query, has_pdf=has_pdf, pdf_corpus=pdf_corpus)

    @classmethod
    def _static_process_message(
        cls,
        project_id: str,
        df: pd.DataFrame,
        sem_layer: dict,
        eda_results: dict,
        insights: list,
        user_message: str,
        contract: Optional[Any] = None,
        dictionary: Optional[List[Dict[str, Any]]] = None,
        pdf_documents: Optional[Any] = None,
        confirmed_field: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main execution node: routes user query through LangGraph StateGraph.
        Preserves dataset isolation, document intelligence, and multi-turn session history.
        """
        # Record user query in isolated project session
        session_manager.add_message(project_id, "user", user_message)

        if confirmed_field:
            session_manager.confirm_field(project_id, user_message, confirmed_field)

        initial_state: AgentState = {
            "project_id": project_id,
            "user_message": user_message,
            "intent": "",
            "tool_used": "",
            "answer": "",
            "calculation_formula": "",
            "chart_spec": None,
            "evidence": [],
            "query_plan": None,
            "error": None,
            "document_corpus": pdf_documents,
            "source_file": None,
            "page_references": [],
            "confirmed_field": confirmed_field,
            "match_suggestions": [],
            "suggestion_mode": None,
            "resolved_field": confirmed_field,
            "interactive_options": []
        }

        # Try official LangGraph execution
        if LANGGRAPH_AVAILABLE:
            try:
                graph = create_agent_graph(
                    df,
                    sem_layer,
                    eda_results,
                    insights,
                    contract=contract,
                    dictionary=dictionary,
                    project_id=project_id,
                    pdf_documents=pdf_documents
                )
                if graph is not None:
                    final_state = graph.invoke(initial_state)
                    response_payload = {
                        "status": PASS_STATUS,
                        "intent": final_state.get("intent", "QUERY"),
                        "answer": final_state.get("answer", ""),
                        "calculation_formula": final_state.get("calculation_formula", ""),
                        "chart_spec": final_state.get("chart_spec"),
                        "evidence": final_state.get("evidence", []),
                        "tool_used": final_state.get("tool_used", "DatasetQueryTool"),
                        "query_plan": final_state.get("query_plan"),
                        "source_file": final_state.get("source_file"),
                        "page_references": final_state.get("page_references", []),
                        "match_suggestions": final_state.get("match_suggestions", []),
                        "suggestion_mode": final_state.get("suggestion_mode"),
                        "resolved_field": final_state.get("resolved_field"),
                        "confirmed_field": final_state.get("confirmed_field"),
                        "interactive_options": final_state.get("interactive_options", [])
                    }

                    # Save assistant response in session
                    session_manager.add_message(
                        project_id,
                        "assistant",
                        response_payload["answer"],
                        metadata={"tool_used": response_payload["tool_used"], "intent": response_payload["intent"]}
                    )
                    return response_payload
            except Exception as e:
                logger.warning(f"LangGraph execution exception, falling back to deterministic: {e}")

        # Deterministic Fallback Execution
        has_pdf = bool(pdf_documents)
        intent = classify_intent_deterministic(user_message, has_pdf=has_pdf, pdf_corpus=pdf_documents)
        response_payload = {
            "status": PASS_STATUS,
            "intent": intent,
            "answer": "",
            "calculation_formula": "",
            "chart_spec": None,
            "evidence": [],
            "tool_used": "",
            "source_file": None,
            "page_references": []
        }

        if intent == "CHART":
            res = ChartGenerationTool.run(df, sem_layer, user_message)
            response_payload["tool_used"] = "ChartGenerationTool"
            if res.get("success"):
                response_payload["answer"] = res.get("message", "Generated requested visualization.")
                response_payload["chart_spec"] = res.get("chart_spec")
            else:
                response_payload["answer"] = f"Unable to generate chart: {res.get('error', 'Parameter mismatch.')}"
        elif intent == "DOCUMENT_QA":
            corpus = pdf_documents if isinstance(pdf_documents, dict) else (pdf_documents[0] if isinstance(pdf_documents, list) and pdf_documents else None)
            res = PDFIntelligenceTool.run(user_message, document_corpus=corpus, project_context={"project_id": project_id})
            response_payload["tool_used"] = "DocumentIntelligenceTool"
            response_payload["answer"] = res.get("answer", "")
            response_payload["calculation_formula"] = res.get("calculation_formula", "")
            response_payload["evidence"] = res.get("evidence", [])
            response_payload["source_file"] = res.get("source_file")
            response_payload["page_references"] = res.get("page_references", [])
            response_payload["status"] = PASS_STATUS
        elif intent == "INSIGHT":
            res = InsightTool.get_summary_findings(insights, eda_results, query=user_message)
            response_payload["tool_used"] = "InsightTool"
            mode = res.get("mode", "GENERAL")
            findings = res.get("top_findings", [])
            if mode == "PRIORITIES":
                priorities = res.get("priorities", [])
                lines = ["**Operational Action Priorities for Plant Management:**", ""]
                if priorities and isinstance(priorities, list):
                    for idx, p in enumerate(priorities, 1):
                        if isinstance(p, dict):
                            action_title = p.get('action') or p.get('title') or 'Operational Action'
                            lines.append(f"{idx}. **[{p.get('priority', 'HIGH')}] {action_title}**")
                            if p.get('objective'):
                                lines.append(f"   - *Objective:* {p.get('objective')}")
                            if p.get('rationale'):
                                lines.append(f"   - *Rationale:* {p.get('rationale')}")
                            if p.get('expected_outcome'):
                                lines.append(f"   - *Expected Outcome:* {p.get('expected_outcome')}")
                            if p.get('suggested_role'):
                                lines.append(f"   - *Suggested Role:* {p.get('suggested_role')}")
                        elif isinstance(p, str):
                            lines.append(f"{idx}. {p}")
                else:
                    for idx, f in enumerate(findings[:3], 1):
                        lines.append(f"{idx}. **{f['title']}**: {f['what_happened']}")
                response_payload["answer"] = "\n".join(lines)
                response_payload["evidence"] = priorities or findings
                response_payload["calculation_formula"] = "PRIORITIZED_OPERATIONAL_ACTIONS"
            elif mode == "GAPS":
                lacking = res.get("lacking", {})
                lines = ["**Operational Deviations & Data/Process Gaps Identified:**", ""]
                exceptions = lacking.get("baseline_exceptions", [])
                if exceptions:
                    lines.append("*Significant Baseline Exceptions (>2 Sigma Deviations):*")
                    for exc in exceptions[:4]:
                        if exc.get("finding"):
                            lines.append(f"- **{exc.get('metric')}**: {exc.get('finding')}")
                        else:
                            cnt = exc.get("deviating_records_count", exc.get("outlier_count", 0))
                            pct = exc.get("deviating_percentage", exc.get("percentage", 0.0))
                            rng = exc.get("normal_operating_range", "")
                            lines.append(f"- **{exc.get('metric')}**: {cnt} record(s) ({pct}%) deviate beyond normal operating range {rng}.")
                    lines.append("")
                recurrent = lacking.get("recurring_issues", [])
                if recurrent:
                    lines.append("*Recurring Operational Patterns:*")
                    for r in recurrent[:3]:
                        if isinstance(r, dict):
                            lines.append(f"- **{r.get('dimension')}**: High recurrence in '{r.get('top_category')}' ({r.get('occurrences', r.get('top_category_pct', ''))} records).")
                        elif isinstance(r, str):
                            lines.append(f"- {r}")
                    lines.append("")
                missing = lacking.get("missing_records_summary", {})
                if isinstance(missing, dict) and missing.get("missing_count", 0) > 0:
                    lines.append(f"*Process Recording Gaps:* {missing.get('missing_count')} records have incomplete critical fields.")
                elif isinstance(missing, str) and missing:
                    lines.append(f"*Process Recording Context:* {missing}")
                response_payload["answer"] = "\n".join(lines)
                response_payload["evidence"] = exceptions or findings
                response_payload["calculation_formula"] = "OPERATIONAL_GAPS_AND_DEVIATIONS"
            else:
                lines = [f"**Dataset Summary & Key Insights** ({res.get('dataset_rows', 0):,} records, {res.get('dataset_columns', 0)} variables, Health Score: {res.get('health_score', 100)}%):", ""]
                for idx, f in enumerate(findings, 1):
                    lines.append(f"{idx}. **{f['title']}**: {f['what_happened']}")
                    if f.get("why_it_matters"):
                        lines.append(f"   *Strategic Context:* {f['why_it_matters']}")
                response_payload["answer"] = "\n".join(lines)
                response_payload["evidence"] = findings
        elif intent == "EDA_MISSING":
            res = EDATool.get_missing_values(df, eda_results)
            response_payload["tool_used"] = "EDATool"
            null_cols = res.get("columns_with_nulls", [])
            if not null_cols:
                response_payload["answer"] = "Data Completeness: High. Zero missing values detected across all columns."
            else:
                lines = [f"Found missing values in {len(null_cols)} columns (Overall completeness: {100 - res.get('overall_null_pct', 0):.1f}%):"]
                for c in null_cols:
                    lines.append(f"- **{c['column']}**: {c['missing_count']:,} missing values ({c['missing_pct']}%)")
                response_payload["answer"] = "\n".join(lines)
        elif intent == "EDA_OUTLIERS":
            res = EDATool.get_outliers(eda_results)
            response_payload["tool_used"] = "EDATool"
            cols = res.get("columns_with_outliers", [])
            if not cols:
                response_payload["answer"] = "No severe dispersion outliers detected beyond standard 1.5x IQR Tukey fences."
            else:
                lines = [f"Outlier dispersion detected across {len(cols)} numerical features:"]
                for c in cols[:5]:
                    lines.append(f"- **{c['column']}**: {c['outliers_count']:,} values beyond IQR fences ({c['pct']}%)")
                if res.get("multivariate_anomalies", 0) > 0:
                    lines.append(f"\n*Multivariate Isolation Forest identified {res['multivariate_anomalies']} joint anomalies.*")
                response_payload["answer"] = "\n".join(lines)
        elif intent == "EDA_CORRELATION":
            res = EDATool.get_correlations(eda_results)
            response_payload["tool_used"] = "EDATool"
            top_pairs = res.get("top_associations", [])
            if not top_pairs:
                response_payload["answer"] = "No strong bivariate correlations detected among numeric variables."
            else:
                lines = ["**Strongest Statistical Correlations:**"]
                for p in top_pairs:
                    v1 = p.get("var1") or p.get("metric_1")
                    v2 = p.get("var2") or p.get("metric_2")
                    r = p.get("pearson_r")
                    direction = p.get("direction", "positive")
                    lines.append(f"- **{v1} ↔ {v2}**: Pearson r = {r:.3f} ({direction.capitalize()} association)")
                response_payload["answer"] = "\n".join(lines)
        elif intent in ["EDA_PCA", "EDA_DISTRIBUTION"]:
            res = EDATool.get_pca(eda_results)
            response_payload["tool_used"] = "EDATool"
            pca = res.get("pca_analysis", {})
            if pca.get("is_valid"):
                response_payload["answer"] = (
                    f"**Principal Component Analysis (PCA):**\n"
                    f"{pca.get('interpretation')}\n\n"
                    f"- Analyzed features: {', '.join(pca.get('features_analyzed', []))}\n"
                    f"- Cumulative Explained Variance: {pca.get('cumulative_variance_ratio', [])[-1]}%"
                )
            else:
                response_payload["answer"] = f"PCA Analysis not available: {pca.get('reason', 'Insufficient numerical variables.')}"
        else:
            ctx = session_manager.get_context(project_id)
            res = DatasetQueryTool.run(
                df=df,
                sem_layer=sem_layer,
                query=user_message,
                contract=contract,
                dictionary=dictionary,
                context=ctx,
                project_id=project_id,
                confirmed_field=confirmed_field
            )

            # Fallback to DocumentIntelligenceTool if dataset returned no records or error and PDF exists
            if (not res.get("success") or "No dataset" in res.get("answer", "") or "No records" in res.get("answer", "") or "Unable to compute" in res.get("answer", "")) and pdf_documents:
                corpus = pdf_documents if isinstance(pdf_documents, dict) else (pdf_documents[0] if isinstance(pdf_documents, list) and pdf_documents else None)
                doc_res = PDFIntelligenceTool.run(user_message, document_corpus=corpus, project_context={"project_id": project_id})
                if doc_res.get("success"):
                    res = doc_res

            response_payload["tool_used"] = res.get("tool", "DatasetQueryTool")
            response_payload["answer"] = res.get("answer", "")
            response_payload["calculation_formula"] = res.get("calculation_formula", "")
            response_payload["query_plan"] = res.get("plan")
            response_payload["evidence"] = res.get("result_table", res.get("evidence", []))
            response_payload["chart_spec"] = res.get("chart_spec")
            response_payload["match_suggestions"] = res.get("match_suggestions", [])
            response_payload["suggestion_mode"] = res.get("suggestion_mode")
            response_payload["resolved_field"] = res.get("resolved_field")
            response_payload["confirmed_field"] = res.get("confirmed_field")
            response_payload["interactive_options"] = res.get("interactive_options", [])
            if res.get("numerical_value") is not None:
                response_payload["numerical_value"] = res.get("numerical_value")
            if res.get("status"):
                response_payload["status"] = res.get("status")

            plan = res.get("plan")
            if isinstance(plan, dict):
                session_manager.update_context(project_id, {
                    "last_metric": plan.get("metric"),
                    "last_dimension": plan.get("dimension"),
                    "last_intent": plan.get("intent"),
                    "last_time_col": plan.get("time_column")
                })

        session_manager.add_message(
            project_id,
            "assistant",
            response_payload["answer"],
            metadata={"tool_used": response_payload["tool_used"], "intent": intent}
        )

        return response_payload
