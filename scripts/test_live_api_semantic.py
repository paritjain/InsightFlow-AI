import os
import requests
import pandas as pd

excel_path = os.path.abspath("InsightFlow_Multi_Dataset_EDA_Test.xlsx")
# Extract Time Series sheet as standalone file
ts_df = pd.read_excel(excel_path, sheet_name="Time Series")
temp_file = "temp_time_series.xlsx"
ts_df.to_excel(temp_file, index=False)

BASE_URL = "http://127.0.0.1:8000"

try:
    with open(temp_file, "rb") as f:
        resp = requests.post(f"{BASE_URL}/api/projects/upload", files={"files": ("TimeSeries_Test.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    print("Upload status:", resp.status_code)
    data = resp.json()
    proj_id = data.get("project_id")
    print("Project ID:", proj_id)

    # 1. Fetch details
    det_resp = requests.get(f"{BASE_URL}/api/project/{proj_id}")
    assert det_resp.status_code == 200
    proj = det_resp.json()
    print("\n--- Project Data Dictionary ---")
    for item in proj["data_dictionary"]:
        print(f"Col: {item['column_name']} | Role: {item['semantic_role']} | Subtype: {item['semantic_subtype']} | Grouping: {item['grouping_status']} | TimeSeries: {item['time_series_status']} | Agg: {item['aggregation_operations']}")

    # Check date column specifically
    date_item = next(i for i in proj["data_dictionary"] if i["column_name"] == "date")
    assert date_item["semantic_role"] == "Date/Time"
    assert date_item["semantic_subtype"] == "Date"
    assert date_item["grouping_status"] == "Allowed"
    assert date_item["time_series_status"] == "Yes"

    # Check sales
    sales_item = next(i for i in proj["data_dictionary"] if i["column_name"] == "sales")
    assert sales_item["semantic_role"] == "Measure"
    assert sales_item["semantic_subtype"] == "Currency"

    # Check orders
    orders_item = next(i for i in proj["data_dictionary"] if i["column_name"] == "orders")
    assert orders_item["semantic_role"] == "Measure"
    assert orders_item["semantic_subtype"] == "Count"

    # Check marketing_spend
    mktg_item = next(i for i in proj["data_dictionary"] if i["column_name"] == "marketing_spend")
    assert mktg_item["semantic_role"] == "Measure"
    assert mktg_item["semantic_subtype"] == "Currency"

    # Check returns
    ret_item = next(i for i in proj["data_dictionary"] if i["column_name"] == "returns")
    assert ret_item["semantic_role"] == "Measure"
    assert ret_item["semantic_subtype"] == "Count"

    print("\n[SUCCESS] All column classifications match requirements exactly!")

    # Check EDA time series analysis
    ts_eda = proj["eda_results"].get("time_series_analysis") or {}
    print(f"\nEDA Time Series: date_column={ts_eda.get('date_column')}, measure={ts_eda.get('measure_column')}, trend={ts_eda.get('overall_trend')}")
    assert ts_eda.get("date_column") == "date"
    assert ts_eda.get("measure_column") == "sales"
    print("[SUCCESS] EDA Time Series pipeline propagated correctly!")

    # Test Manual Override: Override 'orders' to Dimension
    ov_resp = requests.post(
        f"{BASE_URL}/api/project/{proj_id}/semantic-override",
        json={"column_name": "orders", "semantic_role": "Dimension"}
    )
    print("\nOverride to Dimension status:", ov_resp.status_code)
    ov_data = ov_resp.json()
    ov_orders = next(i for i in ov_data["data_dictionary"] if i["column_name"] == "orders")
    assert ov_orders["semantic_role"] == "Dimension"
    assert ov_orders["override_applied"] is True
    print(f"Overridden orders: Role={ov_orders['semantic_role']}, OverrideRole={ov_orders['override_role']}")

    # Clear override: Set back to Automatic
    clear_resp = requests.post(
        f"{BASE_URL}/api/project/{proj_id}/semantic-override",
        json={"column_name": "orders", "semantic_role": ""}
    )
    print("Clear override status:", clear_resp.status_code)
    clear_data = clear_resp.json()
    clear_orders = next(i for i in clear_data["data_dictionary"] if i["column_name"] == "orders")
    assert clear_orders["semantic_role"] == "Measure"
    assert clear_orders["semantic_subtype"] == "Count"
    assert clear_orders["override_applied"] is False
    print(f"Reverted orders: Role={clear_orders['semantic_role']}, Subtype={clear_orders['semantic_subtype']}, OverrideApplied={clear_orders['override_applied']}")

    # Test Excel and PDF export downloads
    excel_resp = requests.get(f"{BASE_URL}/api/project/{proj_id}/export/excel")
    assert excel_resp.status_code == 200
    print("Excel download status:", excel_resp.status_code, "Length:", len(excel_resp.content))

    pdf_resp = requests.get(f"{BASE_URL}/api/project/{proj_id}/export/pdf")
    assert pdf_resp.status_code == 200
    print("PDF download status:", pdf_resp.status_code, "Length:", len(pdf_resp.content))

    print("\nALL END-TO-END TESTS PASSED!")

finally:
    if os.path.exists(temp_file):
        os.remove(temp_file)
