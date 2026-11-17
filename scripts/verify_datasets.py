import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from engine.dictionary import generate_data_dictionary
from engine.eda import run_automated_eda

excel_path = os.path.join(os.path.dirname(__file__), "..", "InsightFlow_Multi_Dataset_EDA_Test.xlsx")
xl = pd.ExcelFile(excel_path)
print("Sheets in workbook:", xl.sheet_names)

for sheet in xl.sheet_names:
    df = xl.parse(sheet)
    print(f"\n================ SHEET: {sheet} (rows: {len(df)}, cols: {len(df.columns)}) ================")
    dict_items = generate_data_dictionary(df, filename=sheet)
    for item in dict_items:
        print(f"  {item['column_name']}: Role={item['semantic_role']} | Subtype={item['semantic_subtype']} | Grouping={item['grouping_status']} | TimeSeries={item['time_series_status']} | Agg={item['aggregation_operations']} | Conf={item['classification_confidence']}")
        print(f"    Evidence: {item['classification_reason']}")

    # Also test EDA propagation
    eda = run_automated_eda(df)
    ts = eda.get("time_series_analysis") or {}
    print(f"  --> EDA Time Series Detection: date_col={ts.get('date_column')}, measure_col={ts.get('measure_column')}, trend={ts.get('overall_trend')}, periods={len(ts.get('monthly_trend', []))}")
    vis_types = [v.get("chart_type") for v in eda.get('recommended_visualizations', [])]
    print(f"  --> Recommended Visualizations ({len(vis_types)}): {vis_types}")
