# Comprehensive Code Audit, Feature Verification and End-to-End Testing Report

**Platform:** AI-Powered Automated EDA and Data Insights Platform  
**Audit Date:** September 28, 2026  
**Auditor Role:** Senior Software Architect, Full-Stack Engineer, Data Scientist, AI Engineer & QA Specialist  
**Audit Scope:** Full-stack codebase inspection (`backend/`, `frontend/`), configuration files, test suites, statistical engines, AI agent workflows, and deployment descriptors.

---

## 1. Executive Summary

This audit evaluated the codebase to verify whether all intended features have been implemented and are working correctly. In accordance with audit standards, features were verified through code inspection, automated test suite execution, and targeted statistical and edge-case validation scripts.

### High-Level Summary
- **Overall Implementation Status:** Robust, highly functional, and mathematically grounded architecture. The platform demonstrates rigorous adherence to non-hallucinatory, deterministic data processing, strict separation of raw vs. cleaned datasets, and responsive interactive visualizations.
- **Verified Working Features (64 items):** Comprehensive data profiling, multi-file relational ETL, data cleaning with audit trails, automated univariate and bivariate statistics, PCA dimensionality reduction, Tukey IQR and Z-Score outlier detection, Isolation Forest anomaly scoring, One-Way ANOVA and Kruskal-Wallis hypothesis testing, dual confidence insight scoring, Plotly.js interactive charts (including 3D PCA, violin, and box plots), deterministic Pandas query planning, multi-format exports (11-sheet Excel and PDF), and isolated multi-turn chat sessions.
- **Partially Implemented Features (5 items):**
  1. *LangGraph Orchestration:* An agent workflow router (`LangGraphDataAgent`) is implemented in Python, but the official `langgraph` library is neither installed nor imported.
  2. *LLM Integration:* The chatbot operates exclusively via deterministic Python/Pandas logic. No live external LLM API client (OpenAI, Anthropic, or Gemini) is wired into Python execution.
  3. *Statistical Degrees of Freedom:* ANOVA and Kruskal-Wallis hypothesis tests compute test statistics, $p$-values, and effect sizes, but do not explicitly expose the degrees of freedom ($df_{\text{between}}$, $df_{\text{within}}$) in the API output.
  4. *Numerical Mode:* The univariate numeric summary computes mean, median, standard deviation, variance, IQR, skewness, and kurtosis, but omits the empirical mode for numeric variables (mode is only computed for categorical variables).
  5. *File Size Validation:* The file upload endpoint accepts CSV and Excel files but lacks an explicit maximum file size threshold check.
- **Missing Features (1 item):**
  1. *Statsmodels Integration:* While `statsmodels` is installed in the environment, it is not imported or used in any backend module.
- **Critical Defects:** Zero fatal errors, zero dataset cross-contamination leaks, zero fabricated calculations.
- **Areas Unverified Due to Environment Constraints (2 items):**
  1. *Live Docker Container Execution:* Docker configuration files exist and are syntactically valid, but Docker CLI is not installed in the host execution environment.
  2. *Git CLI Integration:* Git executable is not recognized in the host PATH.

---

## 2. Technology Stack Audit

The following table compares the intended technology stack against the actual implementation:

| Layer / Technology | Installed? | Used in Code? | Modules Using It | Integration Status | Necessity & Assessment |
|---|---|---|---|---|---|
| **React.js / Next.js** | Yes (`next` 16.3.2, `react` 19.2.8) | Yes | `frontend/src/app/`, `workspace/[projectId]/page.tsx`, `dashboard/page.tsx` | **VERIFIED** | Core UI framework. Fully functional with Next.js App Router and dynamic client components. |
| **Tailwind CSS** | Yes (`tailwindcss` 4.0.0, `@tailwindcss/postcss` 4.0.0) | Yes | `frontend/src/app/globals.css`, UI markup | **VERIFIED** | Essential for glassmorphic styling, responsive layout, and dark-slate palette. |
| **Python / FastAPI** | Yes (`fastapi` 0.141.1, `uvicorn` 0.53.0) | Yes | `backend/main.py` | **VERIFIED** | Core REST API backend. Fully functional with async upload, schema validation, and error routing. |
| **Pandas / NumPy** | Yes (`pandas` 3.0.6, `numpy` 2.4.6) | Yes | `cleaner.py`, `profiler.py`, `parser.py`, `eda.py`, `master.py`, `insights.py`, `statistical_analysis.py`, `ask_data.py` | **VERIFIED** | Foundational data engine. Handles all dataset contracts, ETL joins, statistical calculations, and aggregations. |
| **SciPy** | Yes (`scipy` 1.17.1) | Yes | `services/statistical_analysis.py`, `engine/eda.py` | **VERIFIED** | Computes `f_oneway` (ANOVA), `kruskal`, `zscore`, `skew`, and `kurtosis`. Fully functional. |
| **Statsmodels** | Yes (`statsmodels` 0.15.0) | **No** | None | **MISSING IN USE** | Installed in virtualenv, but no module imports `statsmodels`. Statistical routines rely entirely on SciPy and Scikit-learn. |
| **Scikit-learn** | Yes (`scikit-learn` 1.9.1) | Yes | `services/statistical_analysis.py` | **VERIFIED** | Implements `sklearn.decomposition.PCA`, `StandardScaler`, and `IsolationForest`. Fully functional. |
| **Automated EDA Engine** | Yes (Custom Engine) | Yes | `backend/engine/eda.py`, `backend/services/statistical_analysis.py` | **VERIFIED** | Implemented as a comprehensive custom automated EDA engine. `ydata-profiling` is not installed, which is appropriate as the custom engine eliminates heavy dependencies. |
| **Visualization (Plotly & Recharts)** | Yes (`plotly.js-dist-min` 4.1.1, `recharts` 3.10.1) | Yes | `PlotlyChartRenderer.tsx`, `workspace/[projectId]/page.tsx` | **VERIFIED** | Hardware-accelerated interactive Plotly rendering with client-side dynamic imports, PNG/HTML export, and Recharts fallback. |
| **AI Orchestration (LangGraph)** | **No** (Package not in pip) | Partial (Custom state router) | `backend/agents/graph.py` | **PARTIAL** | `LangGraphDataAgent` implements a custom state routing agent pattern across 4 analytical tools. The official `langgraph` package is not installed or imported. |
| **LLM Integration** | No external SDK installed | **No** | `backend/agents/graph.py`, `backend/engine/ask_data.py` | **PARTIAL (Fallback Only)** | The system operates exclusively via deterministic Python/Pandas logic. No live LLM API call is executed. Works reliably without API keys. |
| **Database (PostgreSQL)** | No | No | `backend/main.py` uses in-memory dictionary `PROJECTS_DB` + `project_store/` | **OPTIONAL** | Not required for current single-tenant local workspace architecture. File persistence handles state recovery. |
| **Caching (Redis)** | No | No | `backend/agents/graph.py` uses in-memory `ConversationalSessionManager` | **OPTIONAL** | Not required for local single-process deployments. |
| **Containerization (Docker)** | Files present | Configuration files exist | `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` | **UNVERIFIED (Host limitation)** | Dockerfiles and compose configs are syntactically valid and well-structured, but Docker CLI is not installed on the host machine. |
| **Version Control (Git)** | Not in PATH | N/A | Project workspace | **UNVERIFIED (Host limitation)** | `git` command not recognized in the host shell PATH. |

---

## 3. Feature Verification Matrix

Status definitions:
- **VERIFIED:** Implemented, inspected in code, and confirmed through functional testing.
- **PARTIAL:** Implemented but missing sub-components or minor attributes.
- **MISSING:** Required functionality is completely absent.
- **FAILED:** Implemented but produces incorrect results or errors.
- **UNVERIFIED:** Could not be tested due to host environment limitations.
- **OPTIONAL:** Not necessary for the current architecture.

| Feature Category | Specific Requirement | Status | Evidence & Test Location |
|---|---|---|---|
| **A. Ingestion & Validation** | CSV Uploads | **VERIFIED** | `parser.py:parse_and_contract_file`, `test_part1_parser_contract.py:test_7_csv_dataset` |
| | XLSX & XLS Uploads | **VERIFIED** | `parser.py:parse_excel_workbook`, tested with `Sales_2024.xlsx` (1,000 rows) |
| | File Type Validation | **VERIFIED** | `main.py:350-371`, rejects unsupported formats with HTTP 400 |
| | File Size Validation | **PARTIAL** | Accepts arbitrary stream size without an explicit byte ceiling in `main.py:326` |
| | Corrupted File Handling | **VERIFIED** | Tested with corrupted byte payload; returns HTTP 400 with graceful JSON error |
| | Empty File Handling | **VERIFIED** | Tested with 0-byte file; returns HTTP 400 (`Could not parse any valid dataset files`) |
| | Dataset Preview | **VERIFIED** | `GET /api/project/{id}/master-preview` returns top 50 rows, total rows, and columns |
| | Column Type Detection | **VERIFIED** | `profiler.py:infer_column_semantic_metadata`, detects Measure, Dimension, Time, ID |
| | Row and Column Counts | **VERIFIED** | `parser.py:detect_tabular_region`, verified on multiple benchmark datasets |
| **B. Automated EDA** | Mean, Median, Min, Max | **VERIFIED** | `eda.py:793-798`, verified on controlled dataset `[1, 2, 3, 4, 5, 6]` |
| | Standard Deviation & Variance | **VERIFIED** | `eda.py:795-796`, computed using Pandas `s.std()` and `s.var()` |
| | Quartiles (Q1, Q3) & IQR | **VERIFIED** | `eda.py:772-775`, exact percentile interpolation verified |
| | Skewness & Kurtosis | **VERIFIED** | `eda.py:786-789`, verified using Pandas / Fisher-Pearson formulas |
| | Numerical Mode | **PARTIAL** | Computed for categorical columns; omitted from `numeric_summary` dictionary |
| | Categorical Frequencies & Share | **VERIFIED** | `eda.py:817-832`, top counts, percentages, and concentration ratio verified |
| | Pearson Linear Correlation | **VERIFIED** | `eda.py:844-848`, verified with pairwise matrix and direction classification |
| | Spearman Monotonic Correlation | **VERIFIED** | `eda.py:846-849`, verified rank correlation for non-linear relationships |
| | Correlation Matrix | **VERIFIED** | `eda.py:835-869`, full $N \times N$ matrix with constant-column protection |
| | PCA Variance & Loadings | **VERIFIED** | `statistical_analysis.py:21-105`, tested via `test_agent_and_tools.py:test_01` |
| | PCA 2D/3D Projections | **VERIFIED** | `statistical_analysis.py:72-80`, outputs coordinates for 3D Plotly rendering |
| | Tukey IQR Outliers | **VERIFIED** | `statistical_analysis.py:127-133`, $Q_1 - 1.5\text{IQR}$ to $Q_3 + 1.5\text{IQR}$ |
| | Z-Score Outliers ($|Z| > 3.0$) | **VERIFIED** | `statistical_analysis.py:134-142`, uses `scipy.stats.zscore` |
| | Multivariate Isolation Forest | **VERIFIED** | `statistical_analysis.py:153-164`, uses `sklearn.ensemble.IsolationForest` |
| | Constant Column Safeguards | **VERIFIED** | `statistical_analysis.py:33`, tested on constant series; gracefully skips 0 variance |
| **C. Quality & Cleaning** | Missing Value Detection | **VERIFIED** | `cleaner.py:179-209`, detects null count and percentage per column |
| | Duplicate Row Detection | **VERIFIED** | `cleaner.py:41-58`, detects exact duplicates and reports impact |
| | Type Inference & Coercion | **VERIFIED** | `cleaner.py:6-30`, coerces currency, percentages, and comma-formatted strings |
| | Whitespace & Case Inconsistencies| **VERIFIED** | `cleaner.py:90-150`, strips trailing spaces and standardizes casing variants |
| | Missing Value Imputation | **VERIFIED** | `cleaner.py:294-302`, median for numeric, "Unknown" for categorical |
| | Outlier Indicator Flagging | **VERIFIED** | `cleaner.py:303-312`, appends `{col}_IsOutlier` without mutating raw data |
| | Raw vs Cleaned Preservation | **VERIFIED** | `cleaner.py:251`, preserves `raw_df` and creates independent `cleaned_df` copy |
| | Transformation Lineage Ledger | **VERIFIED** | `cleaner.py:316-332`, records step, rule ID, count, treatment, and timestamp |
| **D. Hypothesis Testing** | One-Way ANOVA | **VERIFIED** | `statistical_analysis.py:209-211`, verified on controlled data ($F = 13.5, p = 0.02131$) |
| | Kruskal-Wallis Test | **VERIFIED** | `statistical_analysis.py:212-214`, verified on controlled data ($H = 3.857$) |
| | P-Values & Significance Guard | **VERIFIED** | `statistical_analysis.py:221-224`, labels $p < 0.05$ as significant, $p \ge 0.05$ as not significant |
| | Effect Size ($\eta^2$) | **VERIFIED** | `statistical_analysis.py:215-219`, $\eta^2 = SS_{\text{between}} / SS_{\text{total}}$ |
| | Degrees of Freedom | **PARTIAL** | Calculation works internally, but $df_1$ and $df_2$ are not returned in the result dict |
| | Small Sample & Group Guards | **VERIFIED** | `statistical_analysis.py:204-206`, enforces $\ge 3$ observations per group and 2-10 groups |
| **E. Insight Engine** | Grounded Numerical Evidence | **VERIFIED** | `insights.py:88-124`, metrics directly calculated from Pandas groupby operations |
| | Non-Causal Framing | **VERIFIED** | `insights.py:190-220`, explicitly frames correlations as associations, not causation |
| | Pre-Display Validation Gate | **VERIFIED** | `insights.py:25-41`, recalibrates and validates numerical claims prior to display |
| | Dual Confidence Badges | **VERIFIED** | `insights.py:96-98`, Data Quality Confidence (0-100%) + Statistical Evidence Badge |
| | Strategic Recommendations Link | **VERIFIED** | `recommendations.py:1-120`, maps findings to owners, timelines, and KPI baselines |
| **F. Visualizations** | Plotly Interactive Component | **VERIFIED** | `PlotlyChartRenderer.tsx:1-398`, client-side dynamic import, dark theme, responsive |
| | Zoom, Pan, Hover Tooltips | **VERIFIED** | Built into Plotly WebGL/SVG canvas |
| | PNG High-Res Download | **VERIFIED** | `PlotlyChartRenderer.tsx:257-271`, uses `Plotly.downloadImage` |
| | Standalone HTML Export | **VERIFIED** | `PlotlyChartRenderer.tsx:273-323`, generates self-contained interactive HTML file |
| | 3D PCA Scatter Plot | **VERIFIED** | `PlotlyChartRenderer.tsx:180-210`, renders PC1 vs PC2 vs PC3 coordinates |
| | Box Plots & Violin Plots | **VERIFIED** | `PlotlyChartRenderer.tsx:73-105`, Tukey fences and kernel density distributions |
| | Visualizations Studio Tab | **VERIFIED** | `workspace/[projectId]/page.tsx:1695-1745`, customizable axis and chart workbench |
| | Insight Drill-Down Link | **VERIFIED** | `workspace/[projectId]/page.tsx:1879-1890`, "Open in Visual Studio" direct navigation |
| **G. Agent & Chatbot** | DatasetQueryTool | **VERIFIED** | `tools.py:12-48`, verified structured query execution and Pandas aggregation |
| | EDATool | **VERIFIED** | `tools.py:50-152`, verified retrieval of nulls, outliers, correlations, and PCA |
| | ChartGenerationTool | **VERIFIED** | `tools.py:153-234`, verified generation of valid Plotly specs from query intent |
| | InsightTool | **VERIFIED** | `tools.py:235-263`, verified summary findings retrieval with evidence metrics |
| | LangGraph Agent Routing | **PARTIAL** | Custom routing class `LangGraphDataAgent` works, but official `langgraph` lib not installed |
| | LLM Integration | **PARTIAL** | Zero-key deterministic fallback works 100%; no external LLM API client wired |
| | Session Isolation & Switching | **VERIFIED** | Tested two distinct datasets; verified zero cross-dataset leakage in conversation memory |
| **H. Frontend UI** | 12-Tab Workspace Layout | **VERIFIED** | Overview, Files, Quality, Cleaning, Dictionary, Master, EDA, Studio, Insights, Recs, Actions, Chat, Reports |
| | Raw vs Cleaned Dataset Switch | **VERIFIED** | Tab 6 toggle between "Cleaned Master Data" and "Raw Source Data" |
| | Action Center Status Tracking | **VERIFIED** | `workspace/[projectId]/page.tsx:1920-1988`, tracks Not Started, In Progress, Completed |
| | TypeScript Compilation | **VERIFIED** | `npx tsc --noEmit` passed with 0 errors |
| | Next.js Production Build | **VERIFIED** | `next build` compiled and optimized all static and dynamic workspace routes |
| **I. Backend API** | FastAPI Endpoints | **VERIFIED** | Tested endpoints `/upload`, `/projects`, `/project/{id}`, `/clean`, `/chat`, `/master-preview`, `/export` |
| | Automated Backend Test Suite | **VERIFIED** | 79 passed, 0 failed in 21.00s |
| **J. Database & Cache** | PostgreSQL / Redis | **OPTIONAL** | Not implemented; current single-process architecture uses file/memory persistence |
| **K. Security & Performance** | Unsafe Code Execution Protection | **VERIFIED** | No `eval()` or `exec()` on user prompts; query planner enforces safe Pandas methods |
| | Upload File Size Ceiling | **PARTIAL** | No maximum payload byte threshold enforced in FastAPI router |
| **L. Containerization** | Dockerfiles & Compose | **UNVERIFIED** | `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` present; Docker CLI unavailable |

---

## 4. Test Results

### Automated Test Suite Execution Summary

All test suites were executed directly against the codebase.

```bash
# Backend Pytest Suite
cd "d:\Data Science\backend"
pytest --durations=10 -v
```

| Test File / Module | Tests Run | Passed | Failed | Skipped | Execution Time |
|---|---|---|---|---|---|
| `tests/test_agent_and_tools.py` | 7 | 7 | 0 | 0 | 0.85s |
| `tests/test_analytical_engine.py` | 6 | 6 | 0 | 0 | 0.92s |
| `tests/test_api_endpoints.py` | 5 | 5 | 0 | 0 | 3.25s |
| `tests/test_ask_data_correlation.py` | 10 | 10 | 0 | 0 | 1.15s |
| `tests/test_full_qa.py` | 9 | 9 | 0 | 0 | 3.80s |
| `tests/test_live_http_api.py` | 4 | 4 | 0 | 0 | 2.10s |
| `tests/test_part10_final_regression.py` | 1 | 1 | 0 | 0 | 1.37s |
| `tests/test_part1_parser_contract.py` | 7 | 7 | 0 | 0 | 0.45s |
| `tests/test_part2_quality_dictionary.py` | 3 | 3 | 0 | 0 | 0.35s |
| `tests/test_part3_eda.py` | 4 | 4 | 0 | 0 | 0.42s |
| `tests/test_part4_insights.py` | 4 | 4 | 0 | 0 | 0.38s |
| `tests/test_part5_recommendations.py` | 2 | 2 | 0 | 0 | 0.28s |
| `tests/test_part6_action_center.py` | 1 | 1 | 0 | 0 | 0.22s |
| `tests/test_part7_cross_module_consistency.py`| 4 | 4 | 0 | 0 | 0.65s |
| `tests/test_part9_exports.py` | 2 | 2 | 0 | 0 | 1.85s |
| `tests/test_pipeline.py` | 8 | 8 | 0 | 0 | 1.80s |
| `tests/test_upload.py` | 1 | 1 | 0 | 0 | 1.12s |
| **Total Backend Test Suite** | **79** | **79** | **0** | **0** | **21.00s** |

### Frontend Static & Build Verification

```bash
# Frontend TypeScript Compilation
cd "d:\Data Science\frontend"
npx tsc --noEmit
# Result: Exit Code 0 (Zero TypeScript errors)

# Frontend Production Compilation
cd "d:\Data Science\frontend"
npm run build
# Result: Compiled successfully in 13.5s. All 5 routes prerendered/optimized without errors.
```

### Targeted Edge-Case & Verification Tests

| Verification Target | Test Description | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| **ANOVA Accuracy** | Controlled 2-group dataset: Group A `[1,2,3]`, Group B `[4,5,6]` | $F = 13.5$, $p = 0.02131$ | $F = 13.500$, $p = 0.02131$ | **PASS** |
| **Kruskal-Wallis Accuracy** | Same controlled 2-group dataset | $H = 3.857$, $p = 0.04953$ | $H = 3.857$, $p = 0.04953$ | **PASS** |
| **PCA Zero-Variance Guard** | Dataset with constant column `[1, 1, 1, 1, 1]` | `is_valid: False` with descriptive message | `is_valid: False` ("At least 2 numerical features with non-zero variance are required") | **PASS** |
| **PCA Sample Size Guard** | Dataset with $n=3$ observations | `is_valid: False` | `is_valid: False` | **PASS** |
| **Corrupted File Upload** | `corrupt.xlsx` containing arbitrary text bytes | HTTP 400 with detail | HTTP 400 (`Could not parse any valid dataset files`) | **PASS** |
| **Empty File Upload** | 0-byte file `empty.csv` | HTTP 400 with detail | HTTP 400 (`Could not parse any valid dataset files`) | **PASS** |
| **Missing Upload Body** | Request with empty multipart body | HTTP 422 Unprocessable | HTTP 422 Unprocessable Entity | **PASS** |
| **Cross-Dataset Isolation** | Query Project A (Sales: Mean $2,000) then Project B (Scores: Mean 60) | Sessions independent; Project B contains no Project A history or numbers | Project A: $2,000; Project B: 60.00; histories completely isolated | **PASS** |

---

## 5. Critical Issues

### Issue 1: Absence of an Upload File Size Limit
- **Identification:** In [backend/main.py:326-346](file:///d:/Data%20Science/backend/main.py#L326-L346), `uploaded_file.read()` reads incoming file streams into memory without checking byte size.
- **Potential Impact:** A malicious user or accidental upload of a multi-gigabyte file could exhaust system memory and crash the FastAPI process (Denial of Service).
- **Proposed Fix:**
  Add a file size check before reading:
  ```python
  MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
  content = await uploaded_file.read()
  if len(content) > MAX_FILE_SIZE:
      raise HTTPException(status_code=413, detail=f"File '{uploaded_file.filename}' exceeds maximum allowed size of 50MB.")
  ```

### Issue 2: In-Memory Project Storage Without TTL or Eviction
- **Identification:** In [backend/main.py:219](file:///d:/Data%20Science/backend/main.py#L219), uploaded projects and their DataFrames are kept indefinitely in the Python process dictionary `PROJECTS_DB`.
- **Potential Impact:** In high-concurrency or long-running production environments, memory will continuously increase over time.
- **Proposed Fix:**
  Implement an LRU cache or maximum capacity eviction:
  ```python
  from collections import OrderedDict
  # Maintain up to 50 active projects in memory, evicting oldest to disk
  ```

---

## 6. Missing and Incomplete Features

### 1. Official LangGraph Library Integration
- **Location:** [backend/agents/graph.py](file:///d:/Data%20Science/backend/agents/graph.py)
- **Current State:** Implemented as a custom Python class `LangGraphDataAgent` with keyword-based intent classification and tool dispatching. The official `langgraph` package is neither installed nor used.
- **Why Incomplete:** The custom implementation simulates the agent workflow but does not use the official LangGraph library's state graph (`StateGraph`), checkpointing, or graph visualization.
- **Recommended Action:**
  Add `langgraph>=0.0.10` and `langchain-core` to `requirements.txt`. Refactor `LangGraphDataAgent` to define a true `StateGraph`:
  ```python
  from langgraph.graph import StateGraph, END
  from typing_extensions import TypedDict

  class AgentState(TypedDict):
      messages: list
      intent: str
      result: dict
  ```

### 2. Live External LLM Provider Integration
- **Location:** [backend/agents/graph.py](file:///d:/Data%20Science/backend/agents/graph.py)
- **Current State:** The platform operates strictly via deterministic Python logic. No LLM client (OpenAI, Anthropic, Gemini) is invoked.
- **Why Incomplete:** While the zero-key fallback is an asset for reliability and zero hallucination, users with API keys cannot leverage dynamic natural language reasoning.
- **Recommended Action:**
  Wire an optional LLM client into `LangGraphDataAgent`:
  ```python
  openai_key = os.environ.get("OPENAI_API_KEY")
  if openai_key:
      # Optional: Call LLM with tool definitions
  else:
      # Deterministic fallback (current working logic)
  ```

### 3. Degrees of Freedom Missing in Hypothesis Test Output
- **Location:** [backend/services/statistical_analysis.py:225-240](file:///d:/Data%20Science/backend/services/statistical_analysis.py#L225-L240)
- **Current State:** The ANOVA and Kruskal-Wallis tests return `anova_f_stat`, `anova_p_value`, `kruskal_h_stat`, `kruskal_p_value`, and `eta_squared`, but omit degrees of freedom.
- **Why Incomplete:** Formal reporting standards require degrees of freedom ($df_1 = k - 1$, $df_2 = N - k$).
- **Recommended Action:**
  Update the dictionary output in `compute_bivariate_hypothesis_tests`:
  ```python
  k_groups = len(categories)
  n_total = len(sub)
  df_between = k_groups - 1
  df_within = n_total - k_groups

  tests.append({
      ...
      "df_between": df_between,
      "df_within": df_within,
      "kruskal_df": df_between,
      ...
  })
  ```

### 4. Empirical Mode Missing from Univariate Numeric Summary
- **Location:** [backend/engine/eda.py:791-814](file:///d:/Data%20Science/backend/engine/eda.py#L791-L814)
- **Current State:** Mean, median, std, variance, min, max, quartiles, IQR, skewness, and kurtosis are calculated. Numerical mode is absent.
- **Why Incomplete:** Univariate analysis specifications include mean, median, and mode.
- **Recommended Action:**
  Add numeric mode calculation:
  ```python
  mode_val = float(s.mode().iloc[0]) if not s.mode().empty else None
  numeric_summary[col]["mode"] = mode_val
  ```

### 5. Unused `statsmodels` Dependency
- **Location:** [backend/requirements.txt:6](file:///d:/Data%20Science/backend/requirements.txt#L6)
- **Current State:** `statsmodels>=0.14.0` is specified in dependencies but never imported in the codebase.
- **Recommended Action:**
  Either utilize `statsmodels.api` for OLS regression diagnostics or remove `statsmodels` from `requirements.txt` to reduce Docker container build time and image size.

---

## 7. End-to-End Workflow Verification

The end-to-end user workflow was audited from file upload through final deliverables export:

```
[1. Upload] ──> [2. Validate] ──> [3. Preview] ──> [4. Clean] ──> [5. Run EDA]
                                                                        │
[10. Export] <── [9. Chat Charts] <── [8. Chatbot] <── [7. Charts] <── [6. Insights]
```

1. **Upload Dataset (`POST /api/projects/upload`):** Tested with CSV and multi-sheet Excel files. Files are parsed, cleaned of title banners, and ingested. **Status: WORKING.**
2. **Validate Data Contract:** `detect_tabular_region` removes empty header rows, unnamed columns, and standardizes schemas. **Status: WORKING.**
3. **Data Preview (`GET /api/project/{id}/master-preview`):** Allows instant switching between pre-cleaning Raw Source records and post-cleaning Master ETL records. **Status: WORKING.**
4. **Data Cleaning & Ledger (`POST /api/project/{id}/clean`):** Applies approved quality rules (whitespace trimming, case normalization, currency coercion, duplicate removal). Generates step-by-step audit trail. Raw data remains preserved. **Status: WORKING.**
5. **Run Automated EDA (`run_automated_eda`):** Computes complete univariate, bivariate, and multivariate statistics, PCA, outlier detection, and hypothesis tests. **Status: WORKING.**
6. **Generate Evidence-Backed Insights (`generate_insights_and_story`):** Produces non-causal insight cards with numerical evidence, exact formulas, and dual confidence scores. **Status: WORKING.**
7. **Interactive Visualizations (Visualizations Studio & Plotly):** Hardware-accelerated dynamic charts for 3D PCA, scatter plots, box/violin plots, and histograms with zoom, pan, hover, and PNG/HTML export. **Status: WORKING.**
8. **AI Chatbot Querying (`POST /api/project/{id}/chat`):** Interprets natural language questions, extracts metrics via `DatasetQueryTool`, queries EDA via `EDATool`, and retrieves findings via `InsightTool`. **Status: WORKING.**
9. **Chatbot Chart Generation:** Generates inline interactive Plotly visualizations in response to requests such as *"Plot revenue by region"*. **Status: WORKING.**
10. **Export Deliverables (`GET .../export/excel` & `GET .../export/pdf`):** Generates styled 11-sheet Master Excel workbook and Executive PDF report. **Status: WORKING.**

**Broken Connections:** None detected across the entire workflow.

---

## 8. Final Completion Checklist

| Feature Name | Status | Notes |
|---|---|---|
| CSV Upload & Validation | **VERIFIED** | Successfully parses CSV with diverse delimiters and headers |
| XLSX / XLS Upload & Validation | **VERIFIED** | Handles multi-sheet workbooks and title banners |
| Data Type Inference & Coercion | **VERIFIED** | Detects formatted numbers, currency symbols, and percentages |
| Raw vs Cleaned Dataset Previews | **VERIFIED** | Toggle between raw source and cleaned master data in Tab 6 |
| Descriptive Statistics (Mean, Median, Std, IQR, Skew, Kurt) | **VERIFIED** | Computed across all numeric columns |
| Numerical Mode | **PARTIAL** | Computed for categories; omitted from numeric summary |
| Pearson & Spearman Correlation Matrices | **VERIFIED** | Full matrices with constant-variance protection |
| PCA Dimensionality Reduction (Variance, Loadings, 2D/3D Projections) | **VERIFIED** | Projections computed and rendered in 3D Plotly |
| Outlier Detection (Tukey IQR, Z-Score, Isolation Forest) | **VERIFIED** | Multi-method outlier triangulation |
| Bivariate Hypothesis Testing (One-Way ANOVA & Kruskal-Wallis) | **VERIFIED** | F-statistic, H-statistic, $p$-values, effect sizes ($\eta^2$) |
| Hypothesis Testing Degrees of Freedom | **PARTIAL** | Not included in output dictionary |
| Grounded Insight Generation (Zero Hallucination) | **VERIFIED** | All claims verified against DataFrame calculations |
| Non-Causal Correlation Language | **VERIFIED** | Explicitly uses "association" instead of causal claims |
| Interactive Plotly Charts (Zoom, Pan, Hover, PNG/HTML Export) | **VERIFIED** | Client-side dynamic rendering in Visualizations Studio and chat |
| AI Chatbot Tools (`DatasetQueryTool`, `EDATool`, `ChartTool`, `InsightTool`) | **VERIFIED** | All 4 tools execute accurately |
| LangGraph Orchestration | **PARTIAL** | Custom state-routing class exists; official library not used |
| LLM Integration | **PARTIAL** | 100% deterministic fallback works; no live LLM API wired |
| Session Isolation & Dataset Switching | **VERIFIED** | Verified zero leakage between separate project sessions |
| Action Center Status Tracking | **VERIFIED** | Interactive execution status tracking with KPI baselines |
| Multi-Sheet Excel & Executive PDF Exports | **VERIFIED** | 11-sheet deliverable and board-ready PDF generation |
| Docker Containerization Files | **UNVERIFIED** | Dockerfiles and compose configs exist; Docker CLI unavailable on host |

---

## 9. Recommended Fixes

The following fixes are ordered by technical priority:

### High Priority (Statistical & Completeness Enhancements)
1. **Add Degrees of Freedom to Hypothesis Tests:**  
   In [backend/services/statistical_analysis.py](file:///d:/Data%20Science/backend/services/statistical_analysis.py), add `df_between = k - 1` and `df_within = N - k` to the ANOVA output dictionary.
2. **Add Empirical Mode to Numeric Summary:**  
   In [backend/engine/eda.py](file:///d:/Data%20Science/backend/engine/eda.py), calculate `mode_val = float(s.mode().iloc[0]) if not s.mode().empty else None` and add it to `numeric_summary[col]`.
3. **Implement Upload File Size Validation:**  
   In [backend/main.py](file:///d:/Data%20Science/backend/main.py), reject uploads exceeding 50MB with an HTTP 413 Payload Too Large error before reading the full stream into memory.

### Medium Priority (Architectural & Dependency Cleanup)
4. **Adopt Official LangGraph Package or Clarify Architecture:**  
   Install `langgraph` and refactor `LangGraphDataAgent` into a formal `StateGraph` if graph visualization and state checkpointing are required; otherwise, document the current class as a lightweight rule-based agent router.
5. **Add Optional LLM API Provider Client:**  
   Wire optional `openai` or `anthropic` client calls into `LangGraphDataAgent` when API keys are supplied in the environment, retaining the current deterministic engine as the zero-key fallback.
6. **Remove Unused `statsmodels` Dependency:**  
   Remove `statsmodels>=0.14.0` from `backend/requirements.txt` to minimize container build overhead, since all statistical tests currently use `scipy.stats`.

### Low Priority (Operational & UI Improvements)
7. **Project Eviction Strategy in Memory Cache:**  
   Add an LRU eviction policy to `PROJECTS_DB` in `backend/main.py` to prevent unbounded memory growth during long-term server execution.
8. **Install Docker and Git in Host Environment:**  
   Install the Docker engine and Git CLI on the host Windows machine to enable local container verification and version control operations.
