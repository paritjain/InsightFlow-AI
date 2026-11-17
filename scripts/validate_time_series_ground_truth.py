import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import math
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from engine.profiler import profile_file, infer_column_semantic_metadata
from engine.dictionary import generate_data_dictionary
from engine.eda import run_automated_eda
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan
from engine.exporter import export_master_excel, export_executive_pdf

excel_path = os.path.join(os.path.dirname(__file__), "..", "InsightFlow_Multi_Dataset_EDA_Test.xlsx")
df = pd.read_excel(excel_path, sheet_name="Time Series")

print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
assert len(df) == 731, f"Expected 731 rows, got {len(df)}"
assert list(df.columns) == ["date", "sales", "orders", "marketing_spend", "returns"]

# 1. Semantic Data Dictionary Validation
print("\n--- 1. SEMANTIC CLASSIFICATION ---")
dict_list = generate_data_dictionary(df, filename="Time Series")
for item in dict_list:
    col = item["column_name"]
    print(f"{col:16s} -> Role: {item['semantic_role']} | Subtype: {item['semantic_subtype']} | Grouping: {item['grouping_status']} | TimeSeries: {item['time_series_status']} | Agg: {item['aggregation_operations']}")

date_item = next(i for i in dict_list if i["column_name"] == "date")
assert date_item["semantic_role"] == "Date/Time"
assert date_item["semantic_subtype"] == "Date"
assert date_item["grouping_status"] == "Allowed"
assert date_item["time_series_status"] == "Yes"

sales_item = next(i for i in dict_list if i["column_name"] == "sales")
assert sales_item["semantic_role"] == "Measure"
assert sales_item["semantic_subtype"] == "Currency"

orders_item = next(i for i in dict_list if i["column_name"] == "orders")
assert orders_item["semantic_role"] == "Measure"
assert orders_item["semantic_subtype"] == "Count"

mktg_item = next(i for i in dict_list if i["column_name"] == "marketing_spend")
assert mktg_item["semantic_role"] == "Measure"
assert mktg_item["semantic_subtype"] == "Currency"

ret_item = next(i for i in dict_list if i["column_name"] == "returns")
assert ret_item["semantic_role"] == "Measure"
assert ret_item["semantic_subtype"] == "Count"
print("[PASS] Semantic classification rules verified 100%.")

# 2. Automated EDA Numerical Statistics & Outlier Validation
print("\n--- 2. AUTOMATED EDA & DESCRIPTIVE STATISTICS ---")
eda_res = run_automated_eda(df)
num_summary = eda_res["numeric_summary"]
adv_outliers = eda_res["advanced_outliers"]["columns"]

# Sales verification
s_stat = num_summary["sales"]
assert math.isclose(s_stat["mean"], 700.0514, abs_tol=0.01), f"Sales mean: {s_stat['mean']}"
assert math.isclose(s_stat["median"], 702.73, abs_tol=0.01)
assert math.isclose(s_stat["std"], 117.6916, abs_tol=0.01)
assert math.isclose(s_stat["min"], 424.95, abs_tol=0.01)
assert math.isclose(s_stat["max"], 958.32, abs_tol=0.01)
assert math.isclose(s_stat["q1"], 595.63, abs_tol=0.01)
assert math.isclose(s_stat["q3"], 793.57, abs_tol=0.01)
assert math.isclose(s_stat["iqr"], 197.94, abs_tol=0.01)
assert s_stat["outliers_count"] == 0
assert adv_outliers["sales"]["iqr_outliers_count"] == 0
print(f"Sales: mean={s_stat['mean']}, median={s_stat['median']}, std={s_stat['std']}, IQR={s_stat['iqr']}, outliers={s_stat['outliers_count']} [PASS]")

# Orders verification
o_stat = num_summary["orders"]
assert math.isclose(o_stat["mean"], 9.4364, abs_tol=0.01)
assert math.isclose(o_stat["median"], 9.0, abs_tol=0.01)
assert math.isclose(o_stat["std"], 2.4405, abs_tol=0.01)
assert math.isclose(o_stat["min"], 5.0, abs_tol=0.01)
assert math.isclose(o_stat["max"], 18.0, abs_tol=0.01)
assert math.isclose(o_stat["q1"], 8.0, abs_tol=0.01)
assert math.isclose(o_stat["q3"], 11.0, abs_tol=0.01)
assert math.isclose(o_stat["iqr"], 3.0, abs_tol=0.01)
assert math.isclose(o_stat["iqr_lower_bound"], 3.5, abs_tol=0.01)
assert math.isclose(o_stat["iqr_upper_bound"], 15.5, abs_tol=0.01)
assert o_stat["outliers_count"] == 4
assert adv_outliers["orders"]["iqr_outliers_count"] == 4
assert adv_outliers["orders"]["zscore_outliers_count"] == 1
assert o_stat["outliers_percentage"] == 0.55
print(f"Orders: mean={o_stat['mean']}, median={o_stat['median']}, std={o_stat['std']}, bounds=[{o_stat['iqr_lower_bound']}, {o_stat['iqr_upper_bound']}], IQR outliers={o_stat['outliers_count']} (0.55%), Z-score={adv_outliers['orders']['zscore_outliers_count']} [PASS]")

# Marketing Spend verification
m_stat = num_summary["marketing_spend"]
assert math.isclose(m_stat["mean"], 205.9113, abs_tol=0.01)
assert math.isclose(m_stat["median"], 206.28, abs_tol=0.01)
assert math.isclose(m_stat["std"], 25.5712, abs_tol=0.01)
assert math.isclose(m_stat["min"], 124.92, abs_tol=0.01)
assert math.isclose(m_stat["max"], 278.56, abs_tol=0.01)
assert math.isclose(m_stat["q1"], 188.265, abs_tol=0.01)
assert math.isclose(m_stat["q3"], 223.510, abs_tol=0.01)
assert math.isclose(m_stat["iqr"], 35.245, abs_tol=0.01)
assert m_stat["outliers_count"] == 6
assert adv_outliers["marketing_spend"]["iqr_outliers_count"] == 6
assert adv_outliers["marketing_spend"]["zscore_outliers_count"] == 3
assert m_stat["outliers_percentage"] == 0.82
print(f"Marketing Spend: mean={m_stat['mean']}, median={m_stat['median']}, std={m_stat['std']}, IQR={m_stat['iqr']}, IQR outliers={m_stat['outliers_count']} (0.82%), Z-score={adv_outliers['marketing_spend']['zscore_outliers_count']} [PASS]")

# Returns verification
r_stat = num_summary["returns"]
assert math.isclose(r_stat["mean"], 1.0903, abs_tol=0.01)
assert math.isclose(r_stat["median"], 1.0, abs_tol=0.01)
assert math.isclose(r_stat["std"], 1.3365, abs_tol=0.01)
assert math.isclose(r_stat["min"], 0.0, abs_tol=0.01)
assert math.isclose(r_stat["max"], 7.0, abs_tol=0.01)
assert math.isclose(r_stat["q1"], 0.0, abs_tol=0.01)
assert math.isclose(r_stat["q3"], 2.0, abs_tol=0.01)
assert math.isclose(r_stat["iqr"], 2.0, abs_tol=0.01)
assert math.isclose(r_stat["iqr_lower_bound"], -3.0, abs_tol=0.01)
assert math.isclose(r_stat["iqr_upper_bound"], 5.0, abs_tol=0.01)
assert r_stat["outliers_count"] == 5, f"Expected 5 IQR outliers in descriptive table, got {r_stat['outliers_count']}"
assert adv_outliers["returns"]["iqr_outliers_count"] == 5, f"Expected 5 IQR outliers in multi-method table, got {adv_outliers['returns']['iqr_outliers_count']}"
assert adv_outliers["returns"]["zscore_outliers_count"] == 5
assert r_stat["outliers_percentage"] == 0.68
assert math.isclose(r_stat["zero_percentage"], 48.29, abs_tol=0.1)
assert math.isclose(r_stat["nonzero_percentage"], 51.71, abs_tol=0.1)
assert math.isclose(r_stat["non_zero_mean"], 2.1085, abs_tol=0.01), f"Expected non_zero_mean 2.1085, got {r_stat['non_zero_mean']}"
print(f"Returns: mean={r_stat['mean']}, median={r_stat['median']}, std={r_stat['std']}, bounds=[-3, 5], IQR outliers={r_stat['outliers_count']} (0.68%), Z-score={adv_outliers['returns']['zscore_outliers_count']}, zero={r_stat['zero_percentage']}%, non_zero_mean={r_stat['non_zero_mean']} [PASS]")

# 3. Correlations & Regression Slope
print("\n--- 3. CORRELATIONS & REGRESSION SLOPE ---")
cov = np.cov(df["sales"], df["marketing_spend"])
slope = cov[0,1] / cov[0,0]
assert math.isclose(slope, 0.137604, abs_tol=0.0001), f"Expected regression slope 0.137604, got {slope}"
assert math.isclose(df["sales"].sum(), 511737.58, abs_tol=0.01)
assert math.isclose(df["sales"].mean(), 700.05, abs_tol=0.01)

p_matrix = eda_res["correlation_matrix"]
assert math.isclose(p_matrix["sales"]["marketing_spend"], 0.633, abs_tol=0.01)
assert math.isclose(p_matrix["sales"]["orders"], 0.583, abs_tol=0.01)
assert math.isclose(p_matrix["orders"]["marketing_spend"], 0.349, abs_tol=0.01)
assert math.isclose(p_matrix["sales"]["returns"], 0.075, abs_tol=0.01)
assert math.isclose(p_matrix["marketing_spend"]["returns"], 0.052, abs_tol=0.01)
print(f"sales total = {df['sales'].sum():,.2f}, sales mean = {df['sales'].mean():.2f}")
print(f"regression slope = {slope:.6f}")
print("Pairwise Pearson correlations verified [PASS]")

# 4. PCA Analysis
print("\n--- 4. PRINCIPAL COMPONENT ANALYSIS ---")
pca = eda_res["pca_analysis"]
assert pca["is_valid"] is True
ev = pca["explained_variance_ratio"]
cv = pca["cumulative_variance_ratio"]
assert math.isclose(ev[0], 51.80, abs_tol=0.05), f"PC1: {ev[0]}"
assert math.isclose(ev[1], 24.76, abs_tol=0.05), f"PC2: {ev[1]}"
assert math.isclose(ev[2], 16.08, abs_tol=0.05), f"PC3: {ev[2]}"
assert math.isclose(cv[2], 92.64, abs_tol=0.05), f"Cumulative: {cv[2]}"
print(f"PC1 = {ev[0]}%, PC2 = {ev[1]}%, PC3 = {ev[2]}%, Cumulative = {cv[2]}% [PASS]")
print(f"PCA Interpretation: {pca['interpretation']}")
assert "loading" in pca["interpretation"], "Loadings must be included in interpretation"

# 5. Strategic Insights & Returns Verification
print("\n--- 5. STRATEGIC INSIGHTS ---")
insights_res = generate_insights_and_story(df, eda_res)
insights = insights_res["insights"]

ret_insight = next((ins for ins in insights if "returns" in ins.get("metric_name", "").lower() or "returns" in ins.get("title", "").lower() or ins.get("category") == "Variance & Anomaly"), None)
assert ret_insight is not None, "Returns / Variance insight not found"
print("Returns Insight Title:", ret_insight["title"])
print("Returns Insight Text:", ret_insight["insight"])
assert "2.11" in ret_insight["insight"] or "2.108" in ret_insight["insight"], f"Non-zero mean 2.11 not found in insight: {ret_insight['insight']}"
assert "structural zero inflation" not in ret_insight["insight"].lower(), f"Forbidden phrase 'structural zero inflation' found: {ret_insight['insight']}"
assert "zero-heavy" in ret_insight["insight"].lower() or "high proportion" in ret_insight["insight"].lower() or "zero observations" in ret_insight["insight"].lower()

# Check Correlation insight
corr_insight = next((ins for ins in insights if ins.get("category") == "Correlation"), None)
if corr_insight:
    print("\nCorrelation Insight Title:", corr_insight["title"])
    print("Correlation Insight Action:", corr_insight.get("recommended_action"))
    assert "statistically significant" not in corr_insight.get("narrative_summary", "").lower(), "Found statistically significant in correlation without hypothesis test"
    assert "optimal threshold" not in corr_insight.get("recommended_action", "").lower(), "Found unsupported optimal threshold claim"
    assert "causation" in corr_insight.get("insight", "").lower() or "causation" in corr_insight.get("recommended_action", "").lower() or "causation" in corr_insight.get("business_interpretation", "").lower()
print("[PASS] Returns and Correlation insights verified 100%.")

# 6. Strategic Recommendations & Action Plan
print("\n--- 6. RECOMMENDATIONS & ACTION PLAN ---")
recs_res = generate_recommendations_and_action_plan(insights)
recs = recs_res["recommendations"]
for r in recs:
    print(f"Rec [{r['id']}] {r['title']} -> Owner: {r['owner']} | ActionType: {r['action_type']}")
    for s in r.get("implementation_steps", []):
        assert "optimal threshold" not in s.lower() or "before" in s.lower() or "controlled" in s.lower(), f"Unsupported threshold step: {s}"
print("[PASS] Recommendations and action plan verified 100%.")

# 7. Deliverable Exports
print("\n--- 7. MASTER EXCEL & EXECUTIVE PDF EXPORTS ---")
os.makedirs("scratch", exist_ok=True)
excel_out = "scratch/Master_Deliverable_TimeSeries_Verified.xlsx"
pdf_out = "scratch/Executive_Report_TimeSeries_Verified.pdf"

export_master_excel(
    raw_df=df,
    cleaned_df=df,
    master_df=df,
    dictionary_list=dict_list,
    quality_issues=[],
    eda_results=eda_res,
    insights_list=insights,
    recommendations_list=recs,
    action_items_list=recs_res["action_items"],
    output_path=excel_out
)
assert os.path.exists(excel_out) and os.path.getsize(excel_out) > 50000

export_executive_pdf(
    eda_results=eda_res,
    insights_list=insights,
    recommendations_list=recs,
    dataset_name="Time Series Test Dataset",
    output_path=pdf_out,
    dictionary_list=dict_list
)
assert os.path.exists(pdf_out) and os.path.getsize(pdf_out) > 10000
print(f"Excel export: {os.path.getsize(excel_out):,} bytes [PASS]")
print(f"PDF export: {os.path.getsize(pdf_out):,} bytes [PASS]")

print("\n==================================================================")
print("ALL GROUND TRUTH REQUIREMENTS & CONSISTENCY RULES FULLY VERIFIED!")
print("==================================================================")
