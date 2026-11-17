import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from engine.contracts import DatasetContract
from engine.contracts.chat_contracts import QueryIntent, ResultValidationStatus
from engine.analytical_engine import UniversalAnalyticalChatEngine
from agents.graph import LangGraphDataAgent, session_manager


class TestChatbotUniversalMatrix(unittest.TestCase):
    """
    Phase 9: Universal Evaluation Matrix across 8 distinct unseen domain archetypes.
    Verifies that the engine never hardcodes or relies on dataset-specific rules,
    maintaining 100% mathematical fidelity and grounded reasoning.
    """

    def setUp(self):
        np.random.seed(42)

    # ------------------------------------------------------------------------
    # 1. Archetype 1: Academic & Education
    # ------------------------------------------------------------------------
    def test_01_academic_education_archetype(self):
        n = 100
        df = pd.DataFrame({
            "Student_ID": [f"STU_{1000+i}" for i in range(n)],
            "Academic_Program": np.random.choice(["Computer Science", "Economics", "Mechanical Engineering", "Biochemistry"], n),
            "GPA": np.round(np.random.uniform(2.0, 4.0, n), 2),
            "Attendance_Pct": np.round(np.random.uniform(60.0, 100.0, n), 1),
            "Credits_Completed": np.random.randint(15, 120, n),
            "Graduation_Year": np.random.choice([2024, 2025, 2026], n)
        })
        dictionary = [
            {"column_name": "Student_ID", "role": "Identifier", "is_identifier": True},
            {"column_name": "Academic_Program", "role": "Dimension", "is_dimension": True, "semantic_subtype": "Categorical"},
            {"column_name": "GPA", "role": "Measure", "is_measure": True, "semantic_subtype": "Rating"},
            {"column_name": "Attendance_Pct", "role": "Measure", "is_measure": True, "semantic_subtype": "Percentage"},
            {"column_name": "Credits_Completed", "role": "Measure", "is_measure": True, "semantic_subtype": "Quantity"},
            {"column_name": "Graduation_Year", "role": "Dimension", "is_dimension": True, "semantic_subtype": "Temporal"}
        ]
        contract = DatasetContract.create_from_pipeline("proj-edu", df, df, "students.csv", dictionary)
        engine = UniversalAnalyticalChatEngine(df=df, contract=contract, dictionary=dictionary, project_id="proj-edu")

        # Global Aggregation
        res1 = engine.process_query("What is the average GPA across all students?")
        self.assertEqual(res1.status, ResultValidationStatus.PASS)
        self.assertEqual(res1.intent, QueryIntent.GLOBAL_AGGREGATION)
        self.assertIn("GPA", res1.answer)
        self.assertAlmostEqual(res1.numerical_value, df["GPA"].mean(), delta=0.05)

        # Group Comparison
        res2 = engine.process_query("Compare average attendance percentage by academic program")
        self.assertEqual(res2.status, ResultValidationStatus.PASS)
        self.assertEqual(res2.intent, QueryIntent.GROUP_COMPARISON)
        self.assertIn("Academic_Program", res2.answer)
        self.assertTrue(len(res2.evidence) >= 4)

    # ------------------------------------------------------------------------
    # 2. Archetype 2: E-Commerce & Retail
    # ------------------------------------------------------------------------
    def test_02_ecommerce_retail_archetype(self):
        n = 120
        df = pd.DataFrame({
            "Order_Number": [f"ORD-{50000+i}" for i in range(n)],
            "Fulfillment_Hub": np.random.choice(["Dallas-North", "Chicago-Central", "Seattle-West", "Atlanta-South"], n),
            "Gross_Merchandise_Value": np.round(np.random.uniform(15.0, 1500.0, n), 2),
            "Discount_Amount": np.round(np.random.uniform(0.0, 150.0, n), 2),
            "Shipping_Cost": np.round(np.random.uniform(4.5, 45.0, n), 2),
            "Item_Quantity": np.random.randint(1, 12, n)
        })
        dictionary = [
            {"column_name": "Order_Number", "role": "Identifier", "is_identifier": True},
            {"column_name": "Fulfillment_Hub", "role": "Dimension", "is_dimension": True},
            {"column_name": "Gross_Merchandise_Value", "role": "Measure", "is_measure": True, "semantic_subtype": "Currency"},
            {"column_name": "Discount_Amount", "role": "Measure", "is_measure": True, "semantic_subtype": "Currency"},
            {"column_name": "Shipping_Cost", "role": "Measure", "is_measure": True, "semantic_subtype": "Currency"},
            {"column_name": "Item_Quantity", "role": "Measure", "is_measure": True, "semantic_subtype": "Quantity"}
        ]
        contract = DatasetContract.create_from_pipeline("proj-ecom", df, df, "orders.csv", dictionary)
        engine = UniversalAnalyticalChatEngine(df=df, contract=contract, dictionary=dictionary, project_id="proj-ecom")

        # Sum Aggregation
        res = engine.process_query("What is the total gross merchandise value?")
        self.assertEqual(res.status, ResultValidationStatus.PASS)
        self.assertEqual(res.intent, QueryIntent.GLOBAL_AGGREGATION)
        self.assertAlmostEqual(res.numerical_value, df["Gross_Merchandise_Value"].sum(), delta=1.0)
        self.assertIn("Gross_Merchandise_Value", res.answer)

    # ------------------------------------------------------------------------
    # 3. Archetype 3: Industrial Safety & Maintenance
    # ------------------------------------------------------------------------
    def test_03_industrial_safety_archetype(self):
        n = 80
        df = pd.DataFrame({
            "Incident_Ref": [f"INC-{200+i}" for i in range(n)],
            "Plant_Section": np.random.choice(["Boiler Room", "Refinery Unit 4", "Loading Dock", "Maintenance Bay"], n),
            "Lost_Workday_Hours": np.round(np.random.exponential(scale=10.0, size=n), 1),
            "Near_Miss_Reports": np.random.randint(0, 15, n),
            "Safety_Audit_Score": np.random.randint(40, 100, n)
        })
        # Inject deliberate outlier
        df.loc[0, "Lost_Workday_Hours"] = 250.0

        dictionary = [
            {"column_name": "Incident_Ref", "role": "Identifier", "is_identifier": True},
            {"column_name": "Plant_Section", "role": "Dimension", "is_dimension": True},
            {"column_name": "Lost_Workday_Hours", "role": "Measure", "is_measure": True, "semantic_subtype": "Duration"},
            {"column_name": "Near_Miss_Reports", "role": "Measure", "is_measure": True, "semantic_subtype": "Quantity"},
            {"column_name": "Safety_Audit_Score", "role": "Measure", "is_measure": True, "semantic_subtype": "Rating"}
        ]
        contract = DatasetContract.create_from_pipeline("proj-safety", df, df, "incidents.xlsx", dictionary)
        engine = UniversalAnalyticalChatEngine(df=df, contract=contract, dictionary=dictionary, project_id="proj-safety")

        # Outlier Detection
        res = engine.process_query("Identify outliers in lost workday hours")
        self.assertEqual(res.status, ResultValidationStatus.PASS)
        self.assertEqual(res.intent, QueryIntent.OUTLIER_ANALYSIS)
        self.assertIn("Lost_Workday_Hours", res.answer)
        self.assertTrue("outlier" in res.answer.lower())

    # ------------------------------------------------------------------------
    # 4. Archetype 4: Healthcare & Clinical Trials
    # ------------------------------------------------------------------------
    def test_04_healthcare_clinical_archetype(self):
        n = 150
        df = pd.DataFrame({
            "Patient_Code": [f"PAT-{800+i}" for i in range(n)],
            "Treatment_Arm": np.random.choice(["Placebo", "Low Dose (10mg)", "High Dose (50mg)"], n),
            "Recovery_Days": np.random.randint(3, 45, n),
            "Systolic_BP": np.random.normal(125, 12, n).round(1),
            "Adverse_Events": np.random.poisson(0.5, n)
        })
        dictionary = [
            {"column_name": "Patient_Code", "role": "Identifier", "is_identifier": True},
            {"column_name": "Treatment_Arm", "role": "Dimension", "is_dimension": True},
            {"column_name": "Recovery_Days", "role": "Measure", "is_measure": True, "semantic_subtype": "Duration"},
            {"column_name": "Systolic_BP", "role": "Measure", "is_measure": True, "semantic_subtype": "Rating"},
            {"column_name": "Adverse_Events", "role": "Measure", "is_measure": True, "semantic_subtype": "Quantity"}
        ]
        contract = DatasetContract.create_from_pipeline("proj-clinical", df, df, "trial.csv", dictionary)
        engine = UniversalAnalyticalChatEngine(df=df, contract=contract, dictionary=dictionary, project_id="proj-clinical")

        # Group comparison
        res = engine.process_query("What is the average recovery days by treatment arm?")
        self.assertEqual(res.status, ResultValidationStatus.PASS)
        self.assertEqual(res.intent, QueryIntent.GROUP_COMPARISON)
        self.assertIn("Treatment_Arm", res.answer)
        self.assertIn("Recovery_Days", res.answer)

    # ------------------------------------------------------------------------
    # 5. Archetype 5: Corporate HR & Talent Analytics
    # ------------------------------------------------------------------------
    def test_05_corporate_hr_archetype(self):
        n = 90
        df = pd.DataFrame({
            "Employee_ID": [f"EMP-{300+i}" for i in range(n)],
            "Business_Unit": np.random.choice(["Engineering", "Product", "Sales", "Legal", "Marketing"], n),
            "Base_Salary": np.random.uniform(55000, 185000, n).round(2),
            "Tenure_Years": np.random.uniform(0.5, 12.0, n).round(1),
            "Performance_Rating": np.random.choice([1, 2, 3, 4, 5], n)
        })
        dictionary = [
            {"column_name": "Employee_ID", "role": "Identifier", "is_identifier": True},
            {"column_name": "Business_Unit", "role": "Dimension", "is_dimension": True},
            {"column_name": "Base_Salary", "role": "Measure", "is_measure": True, "semantic_subtype": "Currency"},
            {"column_name": "Tenure_Years", "role": "Measure", "is_measure": True, "semantic_subtype": "Duration"},
            {"column_name": "Performance_Rating", "role": "Measure", "is_measure": True, "semantic_subtype": "Rating"}
        ]
        contract = DatasetContract.create_from_pipeline("proj-hr", df, df, "workforce.csv", dictionary)
        engine = UniversalAnalyticalChatEngine(df=df, contract=contract, dictionary=dictionary, project_id="proj-hr")

        # Max aggregation with filter
        res = engine.process_query("What is the maximum base salary?")
        self.assertEqual(res.status, ResultValidationStatus.PASS)
        self.assertEqual(res.intent, QueryIntent.GLOBAL_AGGREGATION)
        self.assertAlmostEqual(res.numerical_value, df["Base_Salary"].max(), delta=1.0)

    # ------------------------------------------------------------------------
    # 6. Archetype 6: Logistics & Supply Chain Fleet
    # ------------------------------------------------------------------------
    def test_06_logistics_fleet_archetype(self):
        n = 110
        df = pd.DataFrame({
            "Consignment_ID": [f"CN-{7000+i}" for i in range(n)],
            "Carrier_Partner": np.random.choice(["DHL Express", "FedEx Freight", "Maersk Intermodal"], n),
            "Freight_Charges": np.random.uniform(120.0, 4800.0, n).round(2),
            "Transit_Time_Hours": np.random.uniform(12.0, 144.0, n).round(1),
            "Payload_Weight_Kg": np.random.uniform(50.0, 8500.0, n).round(1)
        })
        dictionary = [
            {"column_name": "Consignment_ID", "role": "Identifier", "is_identifier": True},
            {"column_name": "Carrier_Partner", "role": "Dimension", "is_dimension": True},
            {"column_name": "Freight_Charges", "role": "Measure", "is_measure": True, "semantic_subtype": "Currency"},
            {"column_name": "Transit_Time_Hours", "role": "Measure", "is_measure": True, "semantic_subtype": "Duration"},
            {"column_name": "Payload_Weight_Kg", "role": "Measure", "is_measure": True, "semantic_subtype": "Quantity"}
        ]
        contract = DatasetContract.create_from_pipeline("proj-logistics", df, df, "shipments.csv", dictionary)
        engine = UniversalAnalyticalChatEngine(df=df, contract=contract, dictionary=dictionary, project_id="proj-logistics")

        # Correlation Analysis
        res = engine.process_query("What is the correlation between freight charges and payload weight?")
        self.assertEqual(res.status, ResultValidationStatus.PASS)
        self.assertEqual(res.intent, QueryIntent.CORRELATION_ANALYSIS)
        self.assertIn("Freight_Charges", res.answer)
        self.assertIn("Payload_Weight_Kg", res.answer)

    # ------------------------------------------------------------------------
    # 7. Archetype 7: IoT & Sensor Telemetry
    # ------------------------------------------------------------------------
    def test_07_iot_telemetry_archetype(self):
        n = 140
        df = pd.DataFrame({
            "Sensor_Node_ID": [f"SN-{100+i}" for i in range(n)],
            "Equipment_Subsystem": np.random.choice(["Compressor-1", "Turbine-3", "Exhaust-Chamber"], n),
            "Vibration_Amplitude": np.random.uniform(0.01, 1.45, n).round(3),
            "Operating_Temperature_C": np.random.uniform(35.0, 115.0, n).round(1),
            "Line_Voltage": np.random.uniform(210.0, 245.0, n).round(1)
        })
        dictionary = [
            {"column_name": "Sensor_Node_ID", "role": "Identifier", "is_identifier": True},
            {"column_name": "Equipment_Subsystem", "role": "Dimension", "is_dimension": True},
            {"column_name": "Vibration_Amplitude", "role": "Measure", "is_measure": True, "semantic_subtype": "Ratio"},
            {"column_name": "Operating_Temperature_C", "role": "Measure", "is_measure": True, "semantic_subtype": "Measurement"},
            {"column_name": "Line_Voltage", "role": "Measure", "is_measure": True, "semantic_subtype": "Measurement"}
        ]
        contract = DatasetContract.create_from_pipeline("proj-iot", df, df, "telemetry.csv", dictionary)
        engine = UniversalAnalyticalChatEngine(df=df, contract=contract, dictionary=dictionary, project_id="proj-iot")

        # Distribution Analysis
        res = engine.process_query("What is the distribution of operating temperature?")
        self.assertEqual(res.status, ResultValidationStatus.PASS)
        self.assertEqual(res.intent, QueryIntent.DISTRIBUTION_ANALYSIS)
        self.assertIn("Operating_Temperature_C", res.answer)
        self.assertIn("median", res.answer.lower())

    # ------------------------------------------------------------------------
    # 8. Archetype 8: Customer Survey & NPS Feedback
    # ------------------------------------------------------------------------
    def test_08_customer_survey_nps_archetype(self):
        n = 100
        df = pd.DataFrame({
            "Response_Token": [f"TOK-{9000+i}" for i in range(n)],
            "Subscription_Plan": np.random.choice(["Starter", "Professional", "Enterprise"], n),
            "NPS_Score": np.random.randint(0, 11, n),
            "Customer_Effort_Score": np.random.randint(1, 8, n),
            "Resolution_Time_Minutes": np.random.uniform(5.0, 180.0, n).round(1)
        })
        dictionary = [
            {"column_name": "Response_Token", "role": "Identifier", "is_identifier": True},
            {"column_name": "Subscription_Plan", "role": "Dimension", "is_dimension": True},
            {"column_name": "NPS_Score", "role": "Measure", "is_measure": True, "semantic_subtype": "Rating"},
            {"column_name": "Customer_Effort_Score", "role": "Measure", "is_measure": True, "semantic_subtype": "Rating"},
            {"column_name": "Resolution_Time_Minutes", "role": "Measure", "is_measure": True, "semantic_subtype": "Duration"}
        ]
        contract = DatasetContract.create_from_pipeline("proj-nps", df, df, "survey.csv", dictionary)
        engine = UniversalAnalyticalChatEngine(df=df, contract=contract, dictionary=dictionary, project_id="proj-nps")

        # Distinct Count
        res = engine.process_query("How many distinct subscription plans are there?")
        self.assertEqual(res.status, ResultValidationStatus.PASS)
        self.assertEqual(res.intent, QueryIntent.DISTINCT_COUNT)
        self.assertIn("3", res.answer)


if __name__ == "__main__":
    unittest.main()
