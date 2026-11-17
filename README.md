# Enterprise AI-Powered EDA & Data Insights Platform

An automated Exploratory Data Analysis (EDA) platform with intelligent insights, interactive visualizations, and an integrated dataset-aware AI chatbot.

---

## 🌟 Key Features & Capabilities

### 1. Intelligent Automated EDA Engine
- **Comprehensive Descriptive Statistics**: Computes counts, means, medians, standard deviations, interquartile ranges (IQR), skewness, and null ratios across all numerical and categorical dimensions.
- **Dimensionality Reduction (PCA)**: Scikit-learn Principal Component Analysis calculates explained variance ratios, component loadings, mathematical cluster interpretations, and interactive 2D/3D projections.
- **Multi-Method Outlier & Anomaly Detection**: Triangulates anomaly scoring via:
  - **Tukey IQR Fences** ($[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$)
  - **Standardized Z-Score** ($|Z| > 3.0$)
  - **Multivariate Isolation Forest** (`sklearn.ensemble.IsolationForest`) for high-dimensional leverage points.
- **Rigorous Hypothesis Testing**:
  - **One-Way ANOVA** (`scipy.stats.f_oneway`) for group variance significance.
  - **Kruskal-Wallis H-Test** (`scipy.stats.kruskal`) for non-parametric evaluation.
  - Automatic computation of degrees of freedom, $p$-values, and **Eta-Squared ($\eta^2$)** effect sizes.
- **Pairwise Correlation & Associations**: Computes Pearson linear correlation ($r$) and Spearman monotonic rank coefficients ($\rho$) with constant-variance protection.

### 2. Dataset-Aware AI Chatbot (LangGraph Orchestration)
- **Zero-Key Deterministic Fallback**: Works 100% out of the box using verified Python/Pandas dataframe execution—no external API keys required.
- **Analytical Tool Suite**:
  - `DatasetQueryTool`: Structured query planning, intent classification, metrics calculation, and aggregation.
  - `EDATool`: Instant access to missing data audits, outlier profiles, PCA results, and correlation rankings.
  - `ChartGenerationTool`: Generates verified Plotly chart specifications directly from natural language prompts.
  - `InsightTool`: Surfaces top evidence-backed business findings with numerical baselines.
- **Multi-Turn Session Memory**: Dataset-isolated conversational state stored per `project_id`, preventing cross-dataset data leakage.

### 3. Interactive Visualizations Studio & Plotly.js
- **Hardware-Accelerated Plotly Charts**: Responsive, client-side dynamic charting with pan, zoom, hover inspections, and zero-SSR flickering.
- **Supported Visualizations**:
  - 3D Interactive PCA Projections
  - Bivariate Linear Fit Scatter Plots
  - Categorical & Grouped/Stacked Bar Charts
  - Frequency Histograms & Kernel Density Plots
  - Tukey Box Plots & Violin Plots
  - Correlation Heatmaps
- **Instant Export**: Download high-resolution PNG snapshots or fully standalone interactive HTML charts.

### 4. Raw vs. Cleaned Dataset Previews
- **Side-by-Side Comparison**: Switch between pre-cleaning raw source records and the post-cleaning master ETL dataset.
- **Lineage Tracking**: Visual timeline tracking transformation stages, type casting, missing value imputation, and anomaly resolution.

### 5. Multi-Format Deliverables
- **9-Sheet Master Excel Deliverable** (`.xlsx`): Formatted sheets including Executive Summary, Data Quality Audit, Data Dictionary, Master Dataset, Cleaned Dataset, Raw Dataset, Exploratory Analysis, Strategic Insights, and Action Plan.
- **Executive PDF Report** (`.pdf`): Board-ready executive presentation with KPI summaries and strategic recommendations.

---

## 🏗️ Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | Next.js 16 (App Router), React 19, Tailwind CSS, Lucide React, Recharts, Plotly.js (`plotly.js-dist-min`) |
| **Backend API** | Python 3.11, FastAPI, Uvicorn, Pydantic v2 |
| **Data Science & ML** | Pandas, NumPy, SciPy (`scipy.stats`), Scikit-learn (`sklearn.decomposition`, `sklearn.ensemble`), Statsmodels |
| **Deliverable Gen** | OpenPyXL (Excel formatting), PyMuPDF (PDF compilation) |
| **Containerization** | Docker, Docker Compose |

---

## 📁 Project Architecture

```
Data Science/
├── backend/
│   ├── agents/
│   │   ├── graph.py               # LangGraph multi-tool agent & session router
│   │   └── tools.py               # DatasetQueryTool, EDATool, ChartGenTool, InsightTool
│   ├── engine/
│   │   ├── ask_data.py            # Structured query planner & deterministic executor
│   │   ├── cleaner.py             # Data quality, validation, and auto-imputation
│   │   ├── eda.py                 # Automated EDA orchestrator
│   │   ├── insights.py            # Strategic insight discovery
│   │   ├── master.py              # Multi-file relational ETL & joining
│   │   ├── profiler.py            # Type inference & semantic layer profiling
│   │   ├── recommendations.py     # Problem-to-action strategic cards
│   │   └── reporters.py           # Multi-sheet Excel & PDF exporters
│   ├── services/
│   │   ├── statistical_analysis.py# PCA, Multi-Method Outliers, ANOVA & Kruskal-Wallis
│   │   └── __init__.py
│   ├── tests/                     # 72 Comprehensive unit and integration tests
│   │   ├── test_agent_and_tools.py
│   │   ├── test_eda_features.py
│   │   ├── test_live_http_api.py
│   │   └── ...
│   ├── Dockerfile
│   ├── main.py                    # FastAPI application routes
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── workspace/[projectId]/
│   │   │   │   └── page.tsx       # 12-tab Analytical Studio Workspace
│   │   │   ├── dashboard/page.tsx # Project management dashboard
│   │   │   └── page.tsx           # Landing & file ingestion dropzone
│   │   ├── components/
│   │   │   └── PlotlyChartRenderer.tsx # Interactive Plotly visualization canvas
│   │   └── lib/
│   │       └── api.ts             # Typed REST API client
│   ├── Dockerfile
│   ├── package.json
│   └── next.config.ts
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start Guide

### Option 1: Docker Compose (Recommended)

To build and run both the backend API and frontend Next.js interface in isolated containers:

```bash
# Clone the repository and navigate to root
cd "Data Science"

# Start the full stack
docker compose up --build
```

- **Frontend Interface:** [http://localhost:3000](http://localhost:3000)
- **FastAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

To shut down:
```bash
docker compose down
```

---

### Option 2: Local Development Setup

#### 1. Backend Setup (Python 3.11)

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 2. Frontend Setup (Node.js 20+)

```bash
cd frontend

# Install dependencies
npm install

# Start Next.js development server
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🧪 Running the Test Suite

The backend includes a test suite verifying data quality rules, statistical routines, ETL joins, PCA math, hypothesis tests, and agent tools.

```bash
cd backend
pytest -v
```

All 72 tests run in-process using FastAPI's `TestClient` without requiring an active external port listener.

---

## 🔌 API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/projects/upload` | Upload `.csv` or `.xlsx` files and run initial automated profiling |
| `GET` | `/api/projects` | List all existing analytical projects |
| `GET` | `/api/project/{id}` | Retrieve comprehensive project state, EDA, insights, and action items |
| `POST` | `/api/project/{id}/clean` | Apply approved data cleaning and imputation rules |
| `POST` | `/api/project/{id}/chat` | Send a query to the LangGraph analytical agent |
| `GET` | `/api/project/{id}/chat-history` | Fetch isolated session conversation history |
| `DELETE` | `/api/project/{id}/chat-history` | Reset the conversation history for a project |
| `GET` | `/api/project/{id}/master-preview` | Sample top 50 rows of raw or cleaned master dataset (`?mode=raw\|cleaned`) |
| `GET` | `/api/project/{id}/export/excel` | Download multi-sheet Master Deliverable spreadsheet |
| `GET` | `/api/project/{id}/export/pdf` | Download formatted Executive Summary PDF |

---

## ⚙️ Environment Variables

The platform runs completely self-contained without mandatory external dependencies. Optional keys can be set in an environment file or in `docker-compose.yml`:

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Port for FastAPI backend |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API base address for frontend |
| `OPENAI_API_KEY` | *(None)* | Optional external LLM fallback for conversational polish |
| `ANTHROPIC_API_KEY`| *(None)* | Optional Anthropic model access |
| `GEMINI_API_KEY` | *(None)* | Optional Gemini model access |

---

## 🔒 Security & Data Integrity
- **100% Deterministic Execution**: Every insight, correlation, test statistic, and chart data point is calculated directly on active Pandas dataframes.
- **Non-Causal Guardrails**: Correlations are statistically framed as associations to prevent false causal assertions.
- **Project Isolation**: Chat sessions, uploads, and data previews are strictly partitioned by unique `project_id`.
