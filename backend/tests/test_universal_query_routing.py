import unittest
import pandas as pd
import numpy as np
from engine.ask_data import ask_your_data, parse_query_plan
from engine.cleaner import coerce_numeric_columns
from engine.profiler import infer_column_semantic_metadata
from engine.semantic import SemanticLayer

class TestUniversalQueryRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n = 80

        # Schema 1: Logistics
        cls.logistics_df = pd.DataFrame({
            "ShipmentID": [f"SHP_{1000+i}" for i in range(n)],
            "Carrier": np.random.choice(["FedEx", "UPS", "DHL", "Maersk"], n),
            "Origin_City": np.random.choice(["Chicago", "Dallas", "Atlanta", "Seattle"], n),
            "Shipping_Cost_USD": np.random.uniform(50, 800, n),
            "Delivery_Days": np.random.uniform(1, 14, n),
            "Shipment_Date": pd.date_range("2026-01-01", periods=n, freq="D")
        })

        # Schema 2: Finance
        cls.finance_df = pd.DataFrame({
            "Transaction_ID": [f"TXN_{2000+i}" for i in range(n)],
            "Department": np.random.choice(["Engineering", "Sales", "Marketing", "G&A"], n),
            "Expense_Category": np.random.choice(["SaaS", "Travel", "Equipment", "Consulting"], n),
            "Budget_Amount": np.random.uniform(10000, 50000, n),
            "Actual_Spend": np.random.uniform(8000, 55000, n),
            "Fiscal_Date": pd.date_range("2026-01-01", periods=n, freq="D")
        })

        # Schema 3: E-commerce
        cls.ecommerce_df = pd.DataFrame({
            "OrderID": [f"ORD_{3000+i}" for i in range(n)],
            "Product_Category": np.random.choice(["Electronics", "Apparel", "Home", "Beauty"], n),
            "Sales_Channel": np.random.choice(["Web", "Mobile_App", "Retail_Store"], n),
            "Order_Value": np.random.uniform(20, 500, n),
            "Units_Sold": np.random.randint(1, 10, n).astype(float),
            "Order_Timestamp": pd.date_range("2026-01-01", periods=n, freq="D")
        })

        # Schema 4: HR
        cls.hr_df = pd.DataFrame({
            "Employee_ID": [f"EMP_{4000+i}" for i in range(n)],
            "Department": np.random.choice(["Product", "Engineering", "Design", "HR"], n),
            "Job_Role": np.random.choice(["Senior", "Mid", "Junior", "Lead"], n),
            "Annual_Salary": np.random.uniform(60000, 180000, n),
            "Tenure_Months": np.random.uniform(6, 72, n),
            "Hire_Date": pd.date_range("2024-01-01", periods=n, freq="D")
        })

    # -------------------------------------------------------------------
    # TEST CASE 1: Global Aggregation without Grouping
    # Query: "What is the total baseline volume and mean value for [Metric] across the dataset?"
    # Expected Plan: Intent: global_aggregation | Metrics: [Metric (SUM), Metric (MEAN)] | Dimension: None
    # -------------------------------------------------------------------
    def test_01_global_aggregation_logistics(self):
        query = "What is the total baseline volume and mean value for Shipping_Cost_USD across the dataset?"
        res = ask_your_data(self.logistics_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "global_aggregation")
        self.assertEqual(plan["Metrics"], ["Shipping_Cost_USD (SUM)", "Shipping_Cost_USD (MEAN)"])
        self.assertIsNone(plan["Dimension"])
        self.assertIn("Shipping_Cost_USD", res["answer"])
        self.assertGreater(len(res["evidence"]), 0)

    def test_01_global_aggregation_finance(self):
        query = "What is the total baseline volume and mean value for Actual_Spend across the dataset?"
        res = ask_your_data(self.finance_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "global_aggregation")
        self.assertEqual(plan["Metrics"], ["Actual_Spend (SUM)", "Actual_Spend (MEAN)"])
        self.assertIsNone(plan["Dimension"])
        self.assertIn("Actual_Spend", res["answer"])

    def test_01_global_aggregation_ecommerce(self):
        query = "What is the total baseline volume and mean value for Order_Value across the dataset?"
        res = ask_your_data(self.ecommerce_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "global_aggregation")
        self.assertEqual(plan["Metrics"], ["Order_Value (SUM)", "Order_Value (MEAN)"])
        self.assertIsNone(plan["Dimension"])

    def test_01_global_aggregation_hr(self):
        query = "What is the total baseline volume and mean value for Annual_Salary across the dataset?"
        res = ask_your_data(self.hr_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "global_aggregation")
        self.assertEqual(plan["Metrics"], ["Annual_Salary (SUM)", "Annual_Salary (MEAN)"])
        self.assertIsNone(plan["Dimension"])

    # -------------------------------------------------------------------
    # TEST CASE 2: Valid Group Comparison
    # Query: "Show me total [Metric] by [Category Column]"
    # Expected Plan: Intent: group_comparison | Metric: [Metric (SUM)] | Dimension: [Category Column]
    # -------------------------------------------------------------------
    def test_02_group_comparison_logistics(self):
        query = "Show me total Shipping_Cost_USD by Carrier"
        res = ask_your_data(self.logistics_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "group_comparison")
        self.assertEqual(plan["Metric"], ["Shipping_Cost_USD (SUM)"])
        self.assertEqual(plan["Dimension"], "Carrier")
        self.assertIn("Carrier", res["answer"])
        self.assertIn("rank #1", res["answer"].lower())

    def test_02_group_comparison_finance(self):
        query = "Show me total Actual_Spend by Department"
        res = ask_your_data(self.finance_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "group_comparison")
        self.assertEqual(plan["Metric"], ["Actual_Spend (SUM)"])
        self.assertEqual(plan["Dimension"], "Department")
        self.assertIn("Department", res["answer"])

    def test_02_group_comparison_ecommerce(self):
        query = "Show me total Order_Value by Product_Category"
        res = ask_your_data(self.ecommerce_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "group_comparison")
        self.assertEqual(plan["Metric"], ["Order_Value (SUM)"])
        self.assertEqual(plan["Dimension"], "Product_Category")

    def test_02_group_comparison_hr(self):
        query = "Show me total Annual_Salary by Department"
        res = ask_your_data(self.hr_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "group_comparison")
        self.assertEqual(plan["Metric"], ["Annual_Salary (SUM)"])
        self.assertEqual(plan["Dimension"], "Department")

    # -------------------------------------------------------------------
    # TEST CASE 3: Fallback Execution (Scenario A: group_comparison missing dimension)
    # Query: "Compare overall [Metric] across all records"
    # Expected Plan: Detects no categorical dimension -> Falls back seamlessly to global_aggregation.
    # -------------------------------------------------------------------
    def test_03_fallback_execution_logistics(self):
        query = "Compare overall Shipping_Cost_USD across all records"
        res = ask_your_data(self.logistics_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "global_aggregation")
        self.assertIsNone(plan["Dimension"])
        self.assertIsNotNone(plan.get("fallback_applied"))
        self.assertNotIn("UNABLE_TO_VALIDATE_QUERY_PLAN", res.get("calculation_formula", ""))
        self.assertIn("Shipping_Cost_USD", res["answer"])

    def test_03_fallback_execution_finance(self):
        query = "Compare overall Budget_Amount across all records"
        res = ask_your_data(self.finance_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "global_aggregation")
        self.assertIsNone(plan["Dimension"])
        self.assertIsNotNone(plan.get("fallback_applied"))
        self.assertIn("Budget_Amount", res["answer"])

    def test_03_fallback_execution_ecommerce(self):
        query = "Compare overall Order_Value across all records"
        res = ask_your_data(self.ecommerce_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "global_aggregation")
        self.assertIsNone(plan["Dimension"])
        self.assertIsNotNone(plan.get("fallback_applied"))

    def test_03_fallback_execution_hr(self):
        query = "Compare overall Annual_Salary across all records"
        res = ask_your_data(self.hr_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "global_aggregation")
        self.assertIsNone(plan["Dimension"])
        self.assertIsNotNone(plan.get("fallback_applied"))

    # -------------------------------------------------------------------
    # TEST CASE 4: Multi-Metric Extraction
    # Query: "Calculate count, mean, and total for [Metric]"
    # Expected Plan: Intent: global_aggregation | Aggregations: [COUNT, MEAN, SUM]
    # -------------------------------------------------------------------
    def test_04_multi_metric_logistics(self):
        query = "Calculate count, mean, and total for Shipping_Cost_USD"
        res = ask_your_data(self.logistics_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "global_aggregation")
        self.assertEqual(plan["Aggregations"], ["COUNT", "MEAN", "SUM"])
        self.assertIn("Count (COUNT)", res["answer"])
        self.assertIn("Average (MEAN)", res["answer"])
        self.assertIn("Total (SUM)", res["answer"])
        self.assertIn("COUNT(Shipping_Cost_USD)", res["calculation_formula"])

    def test_04_multi_metric_finance(self):
        query = "Calculate count, mean, and total for Actual_Spend"
        res = ask_your_data(self.finance_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "global_aggregation")
        self.assertEqual(plan["Aggregations"], ["COUNT", "MEAN", "SUM"])
        self.assertIn("Actual_Spend", res["answer"])

    def test_04_multi_metric_ecommerce(self):
        query = "Calculate count, mean, and total for Order_Value"
        res = ask_your_data(self.ecommerce_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "global_aggregation")
        self.assertEqual(plan["Aggregations"], ["COUNT", "MEAN", "SUM"])

    def test_04_multi_metric_hr(self):
        query = "Calculate count, mean, and total for Annual_Salary"
        res = ask_your_data(self.hr_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "global_aggregation")
        self.assertEqual(plan["Aggregations"], ["COUNT", "MEAN", "SUM"])

    # -------------------------------------------------------------------
    # FALLBACK SCENARIOS B & C:
    # -------------------------------------------------------------------
    def test_05_fallback_scenario_b_missing_date_column(self):
        # Create a dataset without date column
        no_date_df = self.logistics_df.drop(columns=["Shipment_Date"])
        query = "Show monthly trend for Shipping_Cost_USD over time"
        res = ask_your_data(no_date_df, query)
        plan = res["query_plan"]

        # Should fall back gracefully to group_comparison by highest-cardinality dimension
        self.assertEqual(plan["intent"], "group_comparison")
        self.assertIsNotNone(plan["Dimension"])
        self.assertIsNotNone(plan.get("fallback_applied"))
        self.assertNotIn("UNABLE_TO_VALIDATE_QUERY_PLAN", res.get("calculation_formula", ""))

    def test_06_fallback_scenario_c_single_metric_correlation(self):
        query = "What is the correlation for Annual_Salary?"
        res = ask_your_data(self.hr_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "correlation_analysis")
        self.assertIsNotNone(plan.get("fallback_applied"))
        self.assertIn("Tenure_Months", res["answer"])
        self.assertIn("Pairwise correlation analysis", res["answer"])

    # -------------------------------------------------------------------
    # DYNAMIC FILE-AGNOSTIC SCHEMA RESOLUTION & FUZZY MATCHING
    # -------------------------------------------------------------------
    def test_07_fuzzy_and_alias_resolution(self):
        # Query uses "cost" -> should map to Shipping_Cost_USD
        q_cost = "What is the total cost by carrier?"
        res1 = ask_your_data(self.logistics_df, q_cost)
        self.assertEqual(str(res1["query_plan"]["metric"]), "Shipping_Cost_USD")

        # Query uses "salary" -> should map to Annual_Salary
        q_salary = "Show average salary by department"
        res2 = ask_your_data(self.hr_df, q_salary)
        self.assertEqual(str(res2["query_plan"]["metric"]), "Annual_Salary")

    # -------------------------------------------------------------------
    # TEST CASE 8: Explicit Named Entity Extraction & Schema Role Binding
    # Query: "Which WarehouseID has the highest average shipment distance, and what is that average value?"
    # -------------------------------------------------------------------
    def test_08_explicit_entity_extraction_and_ranking_with_id_column(self):
        logistics_warehouse_df = pd.DataFrame({
            "ShipmentID": ["SHP_1", "SHP_2", "SHP_3", "SHP_4"],
            "CustomerID": ["CUST_1", "CUST_2", "CUST_3", "CUST_4"],
            "WarehouseID": ["WH_1", "WH_2", "WH_1", "WH_2"],
            "CarrierID": ["CAR_1", "CAR_2", "CAR_1", "CAR_2"],
            "OrderDate": pd.date_range("2026-01-01", periods=4, freq="D"),
            "ShippingMode": ["Air", "Ground", "Air", "Ground"],
            "Delivery_Status": ["Delivered", "Delivered", "In_Transit", "Delivered"],
            "Shipment_Distance": [150.0, 320.0, 180.0, 320.0]
        })

        query = "Which WarehouseID has the highest average shipment distance, and what is that average value?"
        res = ask_your_data(logistics_warehouse_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "filtering_top_n")
        self.assertEqual(plan["Dimension"], "WarehouseID")
        self.assertEqual(str(plan["Metric"]), "Shipment_Distance")
        self.assertEqual(plan["Aggregation"], "mean")
        self.assertEqual(plan["limit"], 1)
        self.assertEqual(plan["sort"], "descending")
        self.assertNotIn("UNABLE_TO_VALIDATE_QUERY_PLAN", res.get("calculation_formula", ""))
        self.assertIn("WH_2", res["answer"])
        self.assertEqual(res.get("numerical_value"), 320.0)

    # -------------------------------------------------------------------
    # TEST CASE 9: Dynamic Role Fallback in Guardrail Validator
    # -------------------------------------------------------------------
    def test_09_dynamic_role_fallback_grouping_by_id_column(self):
        query = "Compare average Shipping_Cost_USD by ShipmentID"
        res = ask_your_data(self.logistics_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "group_comparison")
        self.assertEqual(plan["Dimension"], "ShipmentID")
        self.assertNotIn("UNABLE_TO_VALIDATE_QUERY_PLAN", res.get("calculation_formula", ""))
        self.assertNotIn("cannot be used for business grouping", res["answer"])

    # -------------------------------------------------------------------
    # TEST CASE 10: Multi-Metric Extraction (Total and Mean Together)
    # -------------------------------------------------------------------
    def test_10_multi_metric_total_and_mean_together(self):
        query = "What is the total and mean for Actual_Spend?"
        res = ask_your_data(self.finance_df, query)
        plan = res["query_plan"]

        self.assertEqual(plan["intent"], "global_aggregation")
        aggs = [a.upper() for a in plan["Aggregations"]]
        self.assertIn("SUM", aggs)
        self.assertIn("MEAN", aggs)
        self.assertNotIn("UNRESOLVED_QUERY_PLAN", res.get("calculation_formula", ""))
        self.assertNotIn("Validation Guardrail Notice", res["answer"])

if __name__ == '__main__':
    unittest.main()

