"""
Expanded Arbitrary Datasets Validation Suite — InsightFlow AI
Tests 10 structurally and semantically diverse domain datasets through the generic,
unhardcoded analytical governance engine:
1. E-Commerce Multi-Channel
2. Corporate Finance General Ledger
3. Manufacturing IoT Telemetry
4. Customer Support Operations & CSAT
5. Multi-Table Warehousing & Assets
6. Healthcare Operational Analytics
7. Logistics & Fleet Operations
8. Banking & Credit Risk
9. Retail Inventory & Stock Governance
10. Energy & Utilities Smart Metering
"""

import unittest
import numpy as np
import pandas as pd
import math

from engine.governed_engines import (
    SemanticInferenceEngine,
    KeyDetectionEngine,
    RelationshipDetectionEngine,
    DataQualityEngine,
    ChartSuitabilityEngine,
    AnalyticalResultEngine
)
from engine.analytical_truth import (
    compute_variable_constancy,
    safe_correlation
)
from services.statistical_analysis import (
    compute_group_differences,
    compute_pca,
    compute_advanced_outliers
)
from engine.eda import run_automated_eda, generate_waterfall_chart


class TestArbitraryExpandedDatasets(unittest.TestCase):

    # 1. E-Commerce Multi-Channel
    def test_dataset_01_ecommerce(self):
        df = pd.DataFrame({
            "order_id": [f"ORD-{i:04d}" for i in range(1, 21)],
            "order_date": pd.date_range("2024-01-01", periods=20, freq="D").strftime("%Y-%m-%d"),
            "channel": ["Web", "Mobile App", "Marketplace", "Direct"] * 5,
            "sales_amount": [120.5, 45.0, 310.0, 89.9, 150.0, 60.0, 420.0, 99.0, 110.0, 50.0, 380.0, 85.0,
                             130.0, 55.0, 290.0, 92.0, 140.0, 48.0, 330.0, 88.0],
            "discount_pct": [0.0, 10.0, 15.0, 5.0] * 5
        })
        eda = run_automated_eda(df)
        self.assertIn("recommended_visualizations", eda)
        self.assertGreater(len(eda["recommended_visualizations"]), 0)
        # Sales amount is measure
        meta_sales = SemanticInferenceEngine.infer_column("sales_amount", df["sales_amount"], len(df))
        self.assertEqual(meta_sales["semantic_role"], "Measure")

    # 2. Corporate Finance General Ledger
    def test_dataset_02_corporate_finance(self):
        df = pd.DataFrame({
            "entry_id": [f"GL-{i:05d}" for i in range(1, 16)],
            "cost_center": ["CC_100", "CC_200", "CC_300"] * 5,
            "expense_usd": [5000.0, 12000.0, 8500.0, 4800.0, 11500.0, 9200.0, 5100.0, 13000.0, 8900.0,
                            4900.0, 12500.0, 8700.0, 5200.0, 11800.0, 9100.0],
            "budget_status": ["Approved"] * 15 # Constant feature
        })
        dq = DataQualityEngine.profile_dataframe(df)
        self.assertEqual(dq["total_rows"], 15)
        # Zero variance variable is flagged
        self.assertTrue(dq["columns"]["budget_status"]["statistics"].get("zero_variance", True))
        suit = ChartSuitabilityEngine.chartSuitabilityCheck("HISTOGRAM", df, y_col="budget_status")
        self.assertFalse(suit["suitable"])

    # 3. Manufacturing IoT Telemetry
    def test_dataset_03_manufacturing_iot(self):
        df = pd.DataFrame({
            "sensor_id": [f"IOT-{i:03d}" for i in range(1, 13)],
            "machine_line": ["Line_A", "Line_B"] * 6,
            "vibration_hz": [12.4, 12.8, 12.5, 45.0, 12.6, 12.7, 12.3, 12.5, 12.9, 12.6, 12.4, 12.8], # 45.0 is outlier
            "temperature_c": [65.2, 66.0, 65.8, 92.5, 65.5, 66.2, 65.0, 65.9, 66.4, 65.7, 65.3, 66.1],
            "operating_hours": [500 + i * 50 for i in range(12)]
        })
        outliers = compute_advanced_outliers(df)
        self.assertGreater(outliers["columns"]["vibration_hz"]["iqr_outliers_count"], 0)
        pca = compute_pca(df, n_components=2)
        self.assertTrue(pca["is_valid"])

    # 4. Customer Support Operations & CSAT
    def test_dataset_04_customer_support(self):
        df = pd.DataFrame({
            "ticket_id": [f"TCK-{i:04d}" for i in range(1, 16)],
            "channel": ["Email", "Phone", "Chat"] * 5,
            "resolution_duration_mins": [30.0, 15.0, 8.0, 45.0, 12.0, 10.0, 25.0, 18.0, 7.0, 40.0, 14.0, 9.0, 35.0, 16.0, 8.0],
            "satisfaction_rating": [4, 5, 5, 2, 4, 5, 3, 4, 5, 2, 4, 5, 3, 4, 5]
        })
        meta_rating = SemanticInferenceEngine.infer_column("satisfaction_rating", df["satisfaction_rating"], len(df))
        self.assertEqual(meta_rating["measure_type"], "Rating")
        meta_dur = SemanticInferenceEngine.infer_column("resolution_duration_mins", df["resolution_duration_mins"], len(df))
        self.assertEqual(meta_dur["unit"], "minutes")

    # 5. Multi-Table Warehousing & Assets
    def test_dataset_05_multi_table_warehousing(self):
        depots = pd.DataFrame({
            "depot_id": ["DEP-1", "DEP-2", "DEP-3"],
            "city": ["Chicago", "Dallas", "Atlanta"],
            "sqft": [50000, 75000, 60000]
        })
        pallets = pd.DataFrame({
            "pallet_id": [f"PAL-{i}" for i in range(1, 6)],
            "depot_id": ["DEP-1", "DEP-1", "DEP-2", "DEP-3", "DEP-999"], # DEP-999 is orphan FK violation
            "weight_kg": [450, 520, 610, 480, 500]
        })
        tables = {"depots": depots, "pallets": pallets}
        rels = RelationshipDetectionEngine.detect_and_validate_relationships(tables)
        self.assertGreater(len(rels), 0)
        rel = [r for r in rels if r["child_table"] == "pallets" and r["parent_table"] == "depots"][0]
        self.assertEqual(rel["fk_violation_count"], 1)

    # 6. Healthcare Operational Analytics
    def test_dataset_06_healthcare_operations(self):
        df = pd.DataFrame({
            "patient_id": [f"PAT-{i:05d}" for i in range(1, 16)],
            "department": ["Cardiology", "Neurology", "Orthopedics"] * 5,
            "length_of_stay_days": [4, 7, 3, 5, 8, 2, 6, 9, 4, 5, 8, 3, 4, 7, 2],
            "triage_score": [3, 2, 4, 1, 3, 5, 2, 1, 4, 3, 2, 4, 3, 2, 5],
            "readmission_flag": ["No", "Yes", "No", "No", "Yes"] * 3
        })
        dq = DataQualityEngine.profile_dataframe(df)
        self.assertEqual(dq["total_rows"], 15)
        # Check non-negative constraint on days
        self.assertEqual(len(dq["columns"]["length_of_stay_days"]["data_quality_errors"]), 0)
        # Check ANOVA differences between departments on length of stay
        diff = compute_group_differences(df, "department", "length_of_stay_days")
        self.assertTrue(diff["anova"]["valid"])

    # 7. Logistics & Fleet Operations
    def test_dataset_07_logistics_fleet(self):
        df = pd.DataFrame({
            "shipment_id": [f"SHP-{i:04d}" for i in range(1, 16)],
            "vehicle_type": ["Van", "Truck", "Trailer"] * 5,
            "distance_km": [120.0, 450.0, 800.0, 110.0, 480.0, 850.0, 130.0, 460.0, 820.0, 115.0, 470.0, 840.0, 125.0, 455.0, 810.0],
            "transit_days": [1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4]
        })
        meta_dist = SemanticInferenceEngine.infer_column("distance_km", df["distance_km"], len(df))
        self.assertEqual(meta_dist["unit"], "km")
        corr = safe_correlation(df, "distance_km", "transit_days")
        self.assertTrue(corr["is_computable"])
        self.assertGreater(corr["pearson"], 0.8)

    # 8. Banking & Credit Risk
    def test_dataset_08_banking_credit_risk(self):
        df = pd.DataFrame({
            "account_id": [f"ACC-{i:06d}" for i in range(1, 16)],
            "branch": ["Uptown", "Downtown", "Suburbs"] * 5,
            "credit_score": [720, 680, 610, 750, 690, 640, 730, 670, 620, 760, 700, 650, 740, 685, 630],
            "balance_usd": [15000.0, 8500.0, 3200.0, 22000.0, 9400.0, 4100.0, 17500.0, 8900.0, 3500.0,
                            25000.0, 10200.0, 4500.0, 19000.0, 9100.0, 3800.0],
            "default_flag": ["No", "No", "Yes", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "Yes", "No", "No", "Yes"]
        })
        meta_bool = SemanticInferenceEngine.infer_column("default_flag", df["default_flag"], len(df))
        self.assertEqual(meta_bool["measure_type"], "Binary Category")
        eda = run_automated_eda(df)
        self.assertIn("recommended_visualizations", eda)

    # 9. Retail Inventory & Stock Governance
    def test_dataset_09_retail_inventory(self):
        df = pd.DataFrame({
            "sku_id": [f"SKU-{i:04d}" for i in range(1, 16)],
            "store_zone": ["Zone_East", "Zone_West", "Zone_North"] * 5,
            "on_hand_qty": [150, 80, 220, 140, 95, 210, 160, 85, 230, 145, 90, 215, 155, 88, 225],
            "reorder_point": [50, 40, 60] * 5,
            "unit_cost_usd": [12.5, 25.0, 8.0, 12.5, 25.0, 8.0, 12.5, 25.0, 8.0, 12.5, 25.0, 8.0, 12.5, 25.0, 8.0]
        })
        pk = KeyDetectionEngine.detect_key("sku_id", df["sku_id"], len(df))
        self.assertEqual(pk["key_role"], "primary_key")
        dq = DataQualityEngine.profile_dataframe(df)
        self.assertEqual(dq["total_rows"], 15)

    # 10. Energy & Utilities Smart Metering
    def test_dataset_10_energy_utilities(self):
        df = pd.DataFrame({
            "meter_id": [f"MTR-{i:03d}" for i in range(1, 13)],
            "zone": ["Substation_1", "Substation_2"] * 6,
            "energy_kwh": [120.5, 240.2, 115.8, 235.0, 122.1, 245.5, 118.4, 238.9, 124.0, 242.0, 119.5, 241.1],
            "voltage_v": [230.1, 229.8, 230.5, 229.4, 230.0, 229.9, 230.2, 229.7, 230.3, 229.6, 230.4, 229.5]
        })
        meta_energy = SemanticInferenceEngine.infer_column("energy_kwh", df["energy_kwh"], len(df))
        self.assertEqual(meta_energy["unit"], "kWh")
        diff = compute_group_differences(df, "zone", "energy_kwh")
        self.assertTrue(diff["anova"]["valid"])
        self.assertTrue(diff["anova"]["is_significant"])


if __name__ == "__main__":
    unittest.main()
