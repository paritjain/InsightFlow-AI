import os
import pandas as pd
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz  # PyMuPDF fallback
    except ImportError:
        fitz = None

def export_master_excel(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    master_df: pd.DataFrame,
    dictionary_list: list,
    quality_issues: list,
    eda_results: dict,
    insights_list: list,
    recommendations_list: list,
    action_items_list: list,
    output_path: str = "InsightFlow_Master_Deliverable.xlsx"
) -> str:
    """
    Exports a styled multi-sheet Excel workbook deliverable.
    Sheets:
    1. Executive Summary
    2. Data Quality Audit
    3. Data Dictionary
    4. Master Dataset
    5. Cleaned Dataset
    6. Raw Dataset
    7. Exploratory Analysis
    8. Strategic Insights
    9. Action Plan
    """
    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # Remove default sheet

    # Color Palette & Styles
    header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")  # Dark Slate
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Calibri", size=16, bold=True, color="0F172A")
    subtitle_font = Font(name="Calibri", size=11, italic=True, color="475569")
    bold_font = Font(name="Calibri", size=11, bold=True)
    regular_font = Font(name="Calibri", size=11)
    
    thin_border = Border(
        left=Side(style='thin', color='D1D5DB'),
        right=Side(style='thin', color='D1D5DB'),
        top=Side(style='thin', color='D1D5DB'),
        bottom=Side(style='thin', color='D1D5DB')
    )

    def apply_header_style(ws, row_idx=1):
        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    def auto_fit_columns(ws):
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val_str = str(cell.value or '')
                if cell.row == 1 and len(val_str) > 35:
                    val_str = val_str[:35]
                max_len = max(max_len, len(val_str))
            ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 45)

    # 1. Sheet: Executive Summary
    ws_sum = wb.create_sheet(title="Executive Summary")
    ws_sum.cell(row=1, column=1, value="INSIGHTFLOW AI — EXECUTIVE DELIVERABLE REPORT").font = title_font
    ws_sum.cell(row=2, column=1, value=f"Generated Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}").font = subtitle_font
    
    ws_sum.cell(row=4, column=1, value="Core KPI Highlights").font = Font(name="Calibri", size=13, bold=True, color="0F172A")
    ws_sum.cell(row=5, column=1, value="Metric").font = header_font
    ws_sum.cell(row=5, column=2, value="Value").font = header_font
    apply_header_style(ws_sum, 5)

    r = 6
    domain_kpis = eda_results.get("domain_kpis", {})
    for k, v in domain_kpis.items():
        ws_sum.cell(row=r, column=1, value=k).font = bold_font
        ws_sum.cell(row=r, column=2, value=str(v)).font = regular_font
        r += 1

    r += 2
    ws_sum.cell(row=r, column=1, value="Top Strategic Insights").font = Font(name="Calibri", size=13, bold=True, color="0F172A")
    r += 1
    ws_sum.cell(row=r, column=1, value="ID").font = header_font
    ws_sum.cell(row=r, column=2, value="Priority").font = header_font
    ws_sum.cell(row=r, column=3, value="Insight Title").font = header_font
    ws_sum.cell(row=r, column=4, value="Business Impact").font = header_font
    apply_header_style(ws_sum, r)

    r += 1
    for ins in insights_list[:5]:
        ws_sum.cell(row=r, column=1, value=ins.get("id")).font = regular_font
        ws_sum.cell(row=r, column=2, value=ins.get("priority")).font = bold_font
        ws_sum.cell(row=r, column=3, value=ins.get("title")).font = regular_font
        ws_sum.cell(row=r, column=4, value=ins.get("business_impact")).font = regular_font
        r += 1

    auto_fit_columns(ws_sum)

    # 2. Sheet: Data Quality Audit
    ws_qual = wb.create_sheet(title="Data Quality Audit")
    ws_qual.cell(row=1, column=1, value="Rule ID").font = header_font
    ws_qual.cell(row=1, column=2, value="Type").font = header_font
    ws_qual.cell(row=1, column=3, value="Column").font = header_font
    ws_qual.cell(row=1, column=4, value="Detected Quality Issue").font = header_font
    ws_qual.cell(row=1, column=5, value="Proposed Fix Action").font = header_font
    ws_qual.cell(row=1, column=6, value="Affected Records").font = header_font
    apply_header_style(ws_qual, 1)

    for i, issue in enumerate(quality_issues, start=2):
        ws_qual.cell(row=i, column=1, value=issue.get("id")).font = regular_font
        ws_qual.cell(row=i, column=2, value=issue.get("type")).font = regular_font
        ws_qual.cell(row=i, column=3, value=issue.get("column")).font = regular_font
        ws_qual.cell(row=i, column=4, value=issue.get("issue")).font = regular_font
        ws_qual.cell(row=i, column=5, value=issue.get("proposed_fix")).font = regular_font
        ws_qual.cell(row=i, column=6, value=issue.get("affected_records")).font = regular_font

    auto_fit_columns(ws_qual)

    # 3. Sheet: Data Dictionary
    ws_dict = wb.create_sheet(title="Data Dictionary")
    dict_headers = [
        "Column Name", "Physical Dtype", "Semantic Role", "Semantic Subtype", 
        "Allowed Aggregations", "Grouping Eligibility", "Time-Series Eligible", 
        "Confidence", "Classification Reason", "Impact Tier", "Impact Score", "Override Role"
    ]
    for c_idx, h in enumerate(dict_headers, start=1):
        ws_dict.cell(row=1, column=c_idx, value=h).font = header_font
    apply_header_style(ws_dict, 1)

    for r_idx, d in enumerate(dictionary_list, start=2):
        ws_dict.cell(row=r_idx, column=1, value=d.get("column_name")).font = bold_font
        ws_dict.cell(row=r_idx, column=2, value=d.get("data_type") or d.get("physical_dtype")).font = regular_font
        ws_dict.cell(row=r_idx, column=3, value=d.get("role") or d.get("semantic_role")).font = regular_font
        ws_dict.cell(row=r_idx, column=4, value=d.get("semantic_subtype")).font = regular_font
        ws_dict.cell(row=r_idx, column=5, value=d.get("aggregation_operations") or d.get("allowed_aggregations") or ("SUM, AVG, MIN, MAX, COUNT" if d.get("is_measure") else "COUNT ONLY")).font = regular_font
        ws_dict.cell(row=r_idx, column=6, value=d.get("grouping_status") or ("Allowed" if d.get("grouping_eligibility") else "Disabled")).font = regular_font
        ws_dict.cell(row=r_idx, column=7, value=d.get("time_series_status") or ("Yes" if d.get("time_series_eligibility") else "No")).font = regular_font
        ws_dict.cell(row=r_idx, column=8, value=d.get("classification_confidence") or f"{int(d.get('confidence_score', 95))}%").font = regular_font
        ws_dict.cell(row=r_idx, column=9, value=d.get("classification_reason") or d.get("reason_for_classification") or d.get("business_meaning")).font = regular_font
        ws_dict.cell(row=r_idx, column=10, value=d.get("impact_tier", "MEDIUM IMPACT")).font = regular_font
        ws_dict.cell(row=r_idx, column=11, value=d.get("impact_score", 50)).font = regular_font
        ws_dict.cell(row=r_idx, column=12, value=d.get("override_role", "Automatic/No Override")).font = regular_font

    auto_fit_columns(ws_dict)

    # Helper function to write dataframe to worksheet
    def write_df_sheet(ws_name, df_to_write):
        ws = wb.create_sheet(title=ws_name)
        cols = list(df_to_write.columns)
        for c_i, c_n in enumerate(cols, start=1):
            ws.cell(row=1, column=c_i, value=str(c_n)).font = header_font
        apply_header_style(ws, 1)

        for r_i, row in enumerate(df_to_write.itertuples(index=False), start=2):
            for c_i, val in enumerate(row, start=1):
                cell_val = None if pd.isna(val) else val
                ws.cell(row=r_i, column=c_i, value=cell_val).font = regular_font
        auto_fit_columns(ws)

    # 4. Sheet: Master Dataset
    write_df_sheet("Master Dataset", master_df)
    # 5. Sheet: Cleaned Dataset
    write_df_sheet("Cleaned Dataset", cleaned_df)
    # 6. Sheet: Raw Dataset
    write_df_sheet("Raw Dataset", raw_df)

    # 7. Sheet: Exploratory Analysis (Numeric & Categorical Summaries)
    ws_eda = wb.create_sheet(title="Exploratory Analysis")
    ws_eda.cell(row=1, column=1, value="Variable / Metric").font = header_font
    ws_eda.cell(row=1, column=2, value="Type").font = header_font
    ws_eda.cell(row=1, column=3, value="Count").font = header_font
    ws_eda.cell(row=1, column=4, value="Mean / Mode").font = header_font
    ws_eda.cell(row=1, column=5, value="Median / Unique").font = header_font
    ws_eda.cell(row=1, column=6, value="Std Dev / Conc %").font = header_font
    ws_eda.cell(row=1, column=7, value="Min / Q1").font = header_font
    ws_eda.cell(row=1, column=8, value="Max / Q3").font = header_font
    ws_eda.cell(row=1, column=9, value="IQR / Outliers").font = header_font
    ws_eda.cell(row=1, column=10, value="Skewness / Zero %").font = header_font
    apply_header_style(ws_eda, 1)

    r_idx = 2
    num_sum = eda_results.get("numeric_summary", {})
    for col, s in num_sum.items():
        ws_eda.cell(row=r_idx, column=1, value=col).font = bold_font
        ws_eda.cell(row=r_idx, column=2, value="Numeric").font = regular_font
        ws_eda.cell(row=r_idx, column=3, value=s.get("count")).font = regular_font
        ws_eda.cell(row=r_idx, column=4, value=s.get("mean")).font = regular_font
        ws_eda.cell(row=r_idx, column=5, value=s.get("median")).font = regular_font
        ws_eda.cell(row=r_idx, column=6, value=s.get("std")).font = regular_font
        ws_eda.cell(row=r_idx, column=7, value=s.get("min")).font = regular_font
        ws_eda.cell(row=r_idx, column=8, value=s.get("max")).font = regular_font
        iqr_val = s.get('iqr')
        out_cnt = s.get('outliers_count', 0)
        out_pct = s.get('outliers_percentage')
        if out_pct is None and s.get('count'):
            out_pct = round((out_cnt / s.get('count')) * 100, 2)
        out_pct_str = f" ({out_pct}%)" if out_pct is not None else ""
        ws_eda.cell(row=r_idx, column=9, value=f"IQR: {iqr_val} | Outliers: {out_cnt}{out_pct_str}").font = regular_font
        zero_info = f"Zero: {s.get('zero_percentage')}%"
        if s.get("is_zero_heavy") and s.get("non_zero_mean") is not None:
            zero_info += f" (Non-Zero Mean: {s.get('non_zero_mean')})"
        ws_eda.cell(row=r_idx, column=10, value=f"Skew: {s.get('skewness')} | {zero_info}").font = regular_font
        r_idx += 1

    cat_sum = eda_results.get("categorical_summary", {})
    for col, s in cat_sum.items():
        ws_eda.cell(row=r_idx, column=1, value=col).font = bold_font
        ws_eda.cell(row=r_idx, column=2, value="Categorical").font = regular_font
        ws_eda.cell(row=r_idx, column=3, value="-").font = regular_font
        ws_eda.cell(row=r_idx, column=4, value=str(list(s.get("top_counts", {}).keys())[0] if s.get("top_counts") else "N/A")).font = regular_font
        ws_eda.cell(row=r_idx, column=5, value=s.get("unique_categories")).font = regular_font
        ws_eda.cell(row=r_idx, column=6, value=f"{s.get('concentration_ratio')}%").font = regular_font
        ws_eda.cell(row=r_idx, column=7, value="-").font = regular_font
        ws_eda.cell(row=r_idx, column=8, value="-").font = regular_font
        ws_eda.cell(row=r_idx, column=9, value="-").font = regular_font
        ws_eda.cell(row=r_idx, column=10, value=f"Nulls: {s.get('missing_values')}").font = regular_font
        r_idx += 1

    auto_fit_columns(ws_eda)

    # 8. Sheet: Strategic Insights & Lineage
    ws_ins = wb.create_sheet(title="Strategic Insights")
    ins_headers = ["Insight ID", "Priority", "Confidence Badge", "Validation Status", "Insight Title", "Insight Narrative", "Calculation Method", "Numerical Evidence", "Business Impact", "Lineage Audit"]
    for c_i, h in enumerate(ins_headers, start=1):
        ws_ins.cell(row=1, column=c_i, value=h).font = header_font
    apply_header_style(ws_ins, 1)

    for r_i, ins in enumerate(insights_list, start=2):
        ev_list = ins.get("numerical_evidence", ins.get("evidence", []))
        ev_str = "; ".join([f"{e.get('metric')}: {e.get('value')}" for e in ev_list if isinstance(e, dict)])
        ws_ins.cell(row=r_i, column=1, value=ins.get("id")).font = bold_font
        ws_ins.cell(row=r_i, column=2, value=ins.get("priority")).font = bold_font
        ws_ins.cell(row=r_i, column=3, value=ins.get("statistical_evidence")).font = regular_font
        ws_ins.cell(row=r_i, column=4, value=ins.get("validation_status", "PASSED")).font = regular_font
        ws_ins.cell(row=r_i, column=5, value=ins.get("title")).font = bold_font
        ws_ins.cell(row=r_i, column=6, value=ins.get("insight")).font = regular_font
        ws_ins.cell(row=r_i, column=7, value=ins.get("calculation_method")).font = regular_font
        ws_ins.cell(row=r_i, column=8, value=ev_str).font = regular_font
        ws_ins.cell(row=r_i, column=9, value=ins.get("business_impact")).font = regular_font
        ws_ins.cell(row=r_i, column=10, value=str(ins.get("lineage", {}))).font = regular_font

    auto_fit_columns(ws_ins)

    # 9. Sheet: Action Plan & Recommendations (9 Required Fields)
    ws_act = wb.create_sheet(title="Action Plan")
    act_headers = ["Rec ID", "Source Insight", "Priority", "Title", "Owner", "Timeline", "KPI Monitored", "KPI Baseline", "Problem Addressed", "Recommended Action", "Evidence Base", "Expected Impact", "Status"]
    for c_i, h in enumerate(act_headers, start=1):
        ws_act.cell(row=1, column=c_i, value=h).font = header_font
    apply_header_style(ws_act, 1)

    for r_i, rec in enumerate(recommendations_list, start=2):
        ws_act.cell(row=r_i, column=1, value=rec.get("id")).font = bold_font
        ws_act.cell(row=r_i, column=2, value=rec.get("source_insight_id")).font = regular_font
        ws_act.cell(row=r_i, column=3, value=rec.get("priority")).font = bold_font
        ws_act.cell(row=r_i, column=4, value=rec.get("title")).font = bold_font
        ws_act.cell(row=r_i, column=5, value=rec.get("owner")).font = regular_font
        ws_act.cell(row=r_i, column=6, value=rec.get("timeline")).font = regular_font
        ws_act.cell(row=r_i, column=7, value=rec.get("kpi")).font = regular_font
        ws_act.cell(row=r_i, column=8, value=rec.get("kpi_baseline")).font = regular_font
        ws_act.cell(row=r_i, column=9, value=rec.get("problem")).font = regular_font
        ws_act.cell(row=r_i, column=10, value=rec.get("recommended_action")).font = regular_font
        ws_act.cell(row=r_i, column=11, value=str(rec.get("evidence"))).font = regular_font
        ws_act.cell(row=r_i, column=12, value=rec.get("expected_impact")).font = regular_font
        ws_act.cell(row=r_i, column=13, value=rec.get("status", "Approved")).font = regular_font

    auto_fit_columns(ws_act)

    # Save Workbook
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    wb.save(output_path)
    return output_path

def export_executive_pdf(
    eda_results: dict,
    insights_list: list,
    recommendations_list: list,
    dataset_name: str = "Master Dataset",
    output_path: str = "InsightFlow_Executive_Report.pdf",
    dictionary_list: list = None
) -> str:
    """
    Generates an Executive PDF Report containing KPIs, Executive Narrative, Statistical Findings, and Action Plan using PyMuPDF.
    """
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) module is not installed.")

    doc = fitz.open()
    page = doc.new_page(width=595, height=842) # A4 Size

    # Layout constants
    x_margin = 40
    y_curr = 40
    page_width = 595 - 2 * x_margin

    # Draw Header Banner
    page.draw_rect(fitz.Rect(0, 0, 595, 80), color=(0.12, 0.16, 0.23), fill=(0.12, 0.16, 0.23))
    page.insert_text(fitz.Point(x_margin, 45), "INSIGHTFLOW AI — EXECUTIVE CERTIFICATION REPORT", fontsize=16, color=(1, 1, 1), fontname="hebo")
    page.insert_text(fitz.Point(x_margin, 65), f"Dataset: {dataset_name} | Target Domain: {eda_results.get('domain', 'General')}", fontsize=10, color=(0.8, 0.85, 0.9), fontname="helv")

    y_curr = 110

    # Executive Summary Section
    page.insert_text(fitz.Point(x_margin, y_curr), "1. Executive Summary", fontsize=14, color=(0.06, 0.09, 0.16), fontname="hebo")
    y_curr += 20
    exec_summary = f"Deterministic pipeline analysis executed across {dataset_name}. The platform processed numerical distributions, category relationships, and temporal trends to construct evidence-backed business intelligence."
    
    rc = page.insert_textbox(fitz.Rect(x_margin, y_curr, x_margin + page_width, y_curr + 45), exec_summary, fontsize=10, fontname="helv")
    y_curr += 50

    # Core Domain KPIs Table
    page.insert_text(fitz.Point(x_margin, y_curr), "2. Core Domain Metrics & Health", fontsize=14, color=(0.06, 0.09, 0.16), fontname="hebo")
    y_curr += 20

    kpis = eda_results.get("domain_kpis", {})
    r_height = 20
    for k, v in list(kpis.items())[:6]:
        page.draw_rect(fitz.Rect(x_margin, y_curr, x_margin + page_width, y_curr + r_height), color=(0.9, 0.92, 0.95), fill=(0.95, 0.97, 0.99))
        page.insert_text(fitz.Point(x_margin + 10, y_curr + 14), str(k), fontsize=10, fontname="hebo")
        page.insert_text(fitz.Point(x_margin + 250, y_curr + 14), str(v), fontsize=10, fontname="helv")
        y_curr += r_height + 2

    y_curr += 15

    # Strategic Insights Section
    page.insert_text(fitz.Point(x_margin, y_curr), "3. Strategic Insights & Findings", fontsize=14, color=(0.06, 0.09, 0.16), fontname="hebo")
    y_curr += 20

    for ins in insights_list[:3]:
        title = ins.get("title", "")
        prio = ins.get("priority", "Medium")
        what = ins.get("what_happened", "")
        
        page.draw_rect(fitz.Rect(x_margin, y_curr, x_margin + page_width, y_curr + 55), color=(0.85, 0.88, 0.92), fill=(1, 1, 1))
        page.insert_text(fitz.Point(x_margin + 10, y_curr + 16), f"[{prio.upper()}] {title}", fontsize=11, fontname="hebo", color=(0.1, 0.2, 0.5))
        page.insert_textbox(fitz.Rect(x_margin + 10, y_curr + 22, x_margin + page_width - 10, y_curr + 50), what, fontsize=9, fontname="helv")
        y_curr += 65

    # Page 2: Recommendations and Semantic Governance
    page2 = doc.new_page(width=595, height=842)
    y_curr2 = 40
    page2.insert_text(fitz.Point(x_margin, y_curr2), "4. Recommended Action Plan", fontsize=14, color=(0.06, 0.09, 0.16), fontname="hebo")
    y_curr2 += 25

    for rec in recommendations_list[:3]:
        act_title = rec.get("title", "")
        owner = rec.get("owner", "")
        timeline = rec.get("timeline", "")
        impact = rec.get("expected_impact", "")
        
        page2.draw_rect(fitz.Rect(x_margin, y_curr2, x_margin + page_width, y_curr2 + 55), color=(0.85, 0.88, 0.92), fill=(0.98, 0.99, 1))
        page2.insert_text(fitz.Point(x_margin + 10, y_curr2 + 16), act_title, fontsize=10, fontname="hebo")
        page2.insert_text(fitz.Point(x_margin + 10, y_curr2 + 30), f"Owner: {owner} | Timeline: {timeline}", fontsize=9, fontname="heit")
        page2.insert_text(fitz.Point(x_margin + 10, y_curr2 + 45), f"Expected Impact: {impact}", fontsize=9, fontname="helv", color=(0, 0.4, 0.1))
        y_curr2 += 65

    # Section 5: Semantic Data Governance Schema
    if dictionary_list:
        y_curr2 += 15
        page2.insert_text(fitz.Point(x_margin, y_curr2), "5. Semantic Governance & Column Classification", fontsize=14, color=(0.06, 0.09, 0.16), fontname="hebo")
        y_curr2 += 20

        # Header row
        page2.draw_rect(fitz.Rect(x_margin, y_curr2, x_margin + page_width, y_curr2 + 20), color=(0.12, 0.16, 0.23), fill=(0.12, 0.16, 0.23))
        page2.insert_text(fitz.Point(x_margin + 6, y_curr2 + 14), "Column", fontsize=9, color=(1, 1, 1), fontname="hebo")
        page2.insert_text(fitz.Point(x_margin + 130, y_curr2 + 14), "Role", fontsize=9, color=(1, 1, 1), fontname="hebo")
        page2.insert_text(fitz.Point(x_margin + 200, y_curr2 + 14), "Subtype", fontsize=9, color=(1, 1, 1), fontname="hebo")
        page2.insert_text(fitz.Point(x_margin + 280, y_curr2 + 14), "Allowed Aggregations", fontsize=9, color=(1, 1, 1), fontname="hebo")
        page2.insert_text(fitz.Point(x_margin + 440, y_curr2 + 14), "Grouping", fontsize=9, color=(1, 1, 1), fontname="hebo")
        y_curr2 += 22

        for item in dictionary_list[:12]:
            col_name = str(item.get("column_name", ""))[:20]
            col_role = str(item.get("semantic_role", item.get("role", "")))
            col_sub = str(item.get("semantic_subtype", ""))
            col_agg = str(item.get("aggregation_operations", item.get("aggregation_display", "COUNT")))[:26]
            col_grp = "Allowed" if item.get("grouping_eligibility") else "Disabled"

            page2.draw_rect(fitz.Rect(x_margin, y_curr2, x_margin + page_width, y_curr2 + 18), color=(0.9, 0.92, 0.95), fill=(0.97, 0.98, 1.0))
            page2.insert_text(fitz.Point(x_margin + 6, y_curr2 + 13), col_name, fontsize=8, fontname="hebo")
            page2.insert_text(fitz.Point(x_margin + 130, y_curr2 + 13), col_role, fontsize=8, fontname="helv")
            page2.insert_text(fitz.Point(x_margin + 200, y_curr2 + 13), col_sub, fontsize=8, fontname="helv")
            page2.insert_text(fitz.Point(x_margin + 280, y_curr2 + 13), col_agg, fontsize=7.5, fontname="helv")
            page2.insert_text(fitz.Point(x_margin + 440, y_curr2 + 13), col_grp, fontsize=8, fontname="helv")
            y_curr2 += 20

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc.save(output_path)
    return output_path

