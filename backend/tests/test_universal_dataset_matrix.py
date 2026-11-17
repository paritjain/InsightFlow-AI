import os
import sys
import uuid
import pytest
import numpy as np
import pandas as pd

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.cleaner import coerce_numeric_columns, detect_quality_issues, apply_cleaning_rules
from engine.dictionary import generate_data_dictionary
from engine.eda import run_automated_eda
from engine.insights import generate_insights_and_story
from engine.action_plan import generate_recommendations_and_action_plan
from engine.contracts import DatasetContract, ColumnSemanticContract, AnalyticalResult
from engine.lineage_derivation import DerivationAndLineageEngine
from engine.canonical_eligibility import CanonicalEligibilityEngine
from engine.governed_engines import calculate_canonical_dataset_health, DerivedMetricEngine
from services.statistical_analysis import compute_pca
from engine.analytical_truth import safe_correlation


class TestUniversalDatasetMatrix:
    """
    Universal Dataset Matrix Regression Suite (Datasets A through P).
    Validates that the governed analytical engine operates universally across
    arbitrary, domain-agnostic datasets without dataset-specific hardcoded branches.
    """

    # =========================================================================
    # DATASET A: Safety Observation Dataset (Real Multi-Column Operational Data)
    # =========================================================================
    def test_dataset_a_safety_observation(self):
        path = "backend/demo_data/store/proj-upload-1791368466-b7fa0d/InsightFlow_Safety_Observation_EDA_Validation_v3.xlsx"
        assert os.path.exists(path), f"Safety Observation dataset file missing at {path}"

        df = pd.read_excel(path)
        df = coerce_numeric_columns(df)
        n_rows = len(df)
        assert n_rows > 1000

        # 1. Eligibility Governance on Special Columns
        el_obs_id = CanonicalEligibilityEngine.evaluate_column("Observation_ID", df["Observation_ID"], n_rows)
        assert el_obs_id["analytical_eligible"] is False
        assert el_obs_id["exclusion_reason"] == "PRIMARY_KEY"

        el_obs_cnt = CanonicalEligibilityEngine.evaluate_column("Observation_Count", df["Observation_Count"], n_rows)
        assert el_obs_cnt["analytical_eligible"] is False
        assert el_obs_cnt["exclusion_reason"] == "CONSTANT"

        el_fixture = CanonicalEligibilityEngine.evaluate_column("Relationship_Test_Severity_Rating", df["Relationship_Test_Severity_Rating"], n_rows)
        assert el_fixture["analytical_eligible"] is False
        assert el_fixture["exclusion_reason"] == "TEST_FIXTURE"

        # 2. Lineage Derivation & Deterministic Transformation
        det_risk = DerivationAndLineageEngine.check_deterministic_relationship(df, "Risk_Score", "Risk_Index_Pct")
        assert det_risk["is_deterministic"] is True
        assert det_risk["relationship_type"] == "DETERMINISTIC_TRANSFORMATION"
        assert "Risk_Index_Pct = Risk_Score * 4" in det_risk["formula"]

        # Transitive relationship detection
        trans_rel = DerivedMetricEngine.check_relationship_between_variables(df, "Severity_Rating", "Risk_Index_Pct")
        assert trans_rel.get("is_derived") is True
        assert trans_rel.get("relationship_type") == "DERIVED"

        # 3. Automated EDA Execution
        eda = run_automated_eda(df)
        struct_rels = eda.get("relationships", {}).get("structural_relationships", [])
        assert len(struct_rels) >= 1
        top_struct = struct_rels[0]
        assert top_struct["is_deterministic"] is True
        assert top_struct["metric_1"] in ["Risk_Score", "Risk_Index_Pct"]

        # 4. Strategic Insights & Governance
        ins = generate_insights_and_story(df, eda)
        insights_list = ins.get("insights", [])
        corr_insights = [i for i in insights_list if i.get("category") in ["Correlation", "Derivation Validation"]]
        assert len(corr_insights) >= 1
        dv_insight = corr_insights[0]
        assert dv_insight["category"] == "Derivation Validation"
        assert dv_insight["priority"] == "LOW"
        assert dv_insight["is_deterministic"] is True

        # 5. Strategic Recommendations & Action Plan
        acts = generate_recommendations_and_action_plan(insights_list)
        recs = acts.get("recommendations", [])
        rec_types = [str(r.get("recommendation_type")) for r in recs]
        assert "ANALYTICAL_GOVERNANCE" in rec_types
        # Ensure Bivariate Regression is NOT suggested for the deterministic pair
        biv_recs = [r for r in recs if "Conduct Bivariate Regression" in str(r.get("title"))]
        assert len(biv_recs) == 0

        # 6. Canonical Dataset Contract Generation
        contract = DatasetContract.create_from_pipeline(
            dataset_id="safety-test",
            raw_df=df,
            cleaned_df=df,
            source_file_path=path
        )
        assert contract.raw_row_count == n_rows
        assert contract.dataset_health_result is not None
        assert contract.dataset_health_result.health_score > 80.0

    # =========================================================================
    # DATASET B: E-Commerce / Retail Transactions
    # =========================================================================
    def test_dataset_b_ecommerce_retail(self):
        np.random.seed(42)
        n = 200
        quantities = np.random.randint(1, 10, size=n)
        unit_prices = np.random.uniform(10.0, 100.0, size=n).round(2)
        discount_pcts = np.random.choice([0.0, 0.05, 0.10, 0.20], size=n)
        revenue = (quantities * unit_prices * (1.0 - discount_pcts)).round(2)

        df = pd.DataFrame({
            "Order_ID": [f"ORD-{i:05d}" for i in range(1, n + 1)],
            "Customer_ID": [f"CUST-{np.random.randint(1, 50):03d}" for _ in range(n)],
            "Quantity": quantities,
            "Unit_Price": unit_prices,
            "Discount_Pct": discount_pcts,
            "Total_Revenue": revenue,
            "Category": np.random.choice(["Electronics", "Apparel", "Home", "Sports"], size=n)
        })

        eda = run_automated_eda(df)
        assert eda["domain"].startswith("Structured Dataset Profile")
        assert "Total_Revenue" in eda["numeric_summary"]

        ins = generate_insights_and_story(df, eda)
        assert len(ins["insights"]) >= 1

        contract = DatasetContract.create_from_pipeline("ecommerce-test", df, df)
        assert contract.get_column("Order_ID").semantic_role == "IDENTIFIER"
        assert contract.get_column("Order_ID").analytical_eligible is False

    # =========================================================================
    # DATASET C: SaaS / Subscription ARR and MRR
    # =========================================================================
    def test_dataset_c_saas_arr_mrr(self):
        n = 150
        mrr = np.random.uniform(100.0, 5000.0, size=n).round(2)
        arr = (mrr * 12.0).round(2)

        df = pd.DataFrame({
            "Account_ID": [f"ACC-{i:04d}" for i in range(n)],
            "Monthly_Recurring_Revenue": mrr,
            "Annual_Recurring_Revenue": arr,
            "Plan_Tier": np.random.choice(["Starter", "Professional", "Enterprise"], size=n)
        })

        det = DerivationAndLineageEngine.check_deterministic_relationship(
            df, "Monthly_Recurring_Revenue", "Annual_Recurring_Revenue"
        )
        assert det["is_deterministic"] is True
        assert det["relationship_type"] == "DETERMINISTIC_TRANSFORMATION"

        eda = run_automated_eda(df)
        struct_rels = eda["relationships"]["structural_relationships"]
        assert len(struct_rels) == 1
        assert struct_rels[0]["is_deterministic"] is True

    # =========================================================================
    # DATASET D: Manufacturing / Quality Control
    # =========================================================================
    def test_dataset_d_manufacturing_qc(self):
        n = 120
        df = pd.DataFrame({
            "Batch_ID": [f"BATCH-{i:04d}" for i in range(n)],
            "Chamber_Temp": np.random.normal(250.0, 15.0, size=n).round(1),
            "Pressure_PSI": np.random.normal(45.0, 3.0, size=n).round(2),
            "Defect_Count": np.random.poisson(2, size=n),
            "Operator_Shift": np.random.choice(["Morning", "Evening", "Night"], size=n)
        })

        eda = run_automated_eda(df)
        assert eda["overview"]["num_rows"] == n
        assert "Chamber_Temp" in eda["numeric_summary"]
        assert "Pressure_PSI" in eda["numeric_summary"]

    # =========================================================================
    # DATASET E: Financial / Accounting Ledger
    # =========================================================================
    def test_dataset_e_financial_ledger(self):
        n = 100
        debit = np.random.uniform(500.0, 10000.0, size=n).round(2)
        credit = np.random.uniform(100.0, 5000.0, size=n).round(2)
        net_balance = (debit - credit).round(2)

        df = pd.DataFrame({
            "Transaction_ID": [f"TX-{i:06d}" for i in range(n)],
            "Debit_Amount": debit,
            "Credit_Amount": credit,
            "Net_Balance": net_balance,
            "Account_Code": np.random.choice(["1010-Cash", "2010-AP", "4010-Revenue"], size=n)
        })

        derived_rel = DerivedMetricEngine.check_relationship_between_variables(df, "Debit_Amount", "Net_Balance")
        assert derived_rel["is_derived"] is True

    # =========================================================================
    # DATASET F: HR / People Analytics
    # =========================================================================
    def test_dataset_f_hr_people_analytics(self):
        n = 80
        salary = np.random.uniform(50000.0, 180000.0, size=n).round(2)
        bonus = (salary * 0.15).round(2)
        comp = (salary + bonus).round(2)

        df = pd.DataFrame({
            "Employee_ID": [f"EMP-{i:04d}" for i in range(n)],
            "Base_Salary": salary,
            "Bonus_Amount": bonus,
            "Total_Comp": comp,
            "Department": np.random.choice(["Engineering", "Sales", "HR", "Finance"], size=n)
        })

        eda = run_automated_eda(df)
        ins = generate_insights_and_story(df, eda)
        assert len(ins["insights"]) >= 1

    # =========================================================================
    # DATASET G: Logistics / Fleet Delivery
    # =========================================================================
    def test_dataset_g_logistics_fleet(self):
        n = 90
        miles = np.random.uniform(50.0, 800.0, size=n).round(1)
        gallons = (miles / np.random.uniform(6.0, 10.0, size=n)).round(2)

        df = pd.DataFrame({
            "VIN_Number": [f"1HGCR2F83HA{i:06d}" for i in range(n)],
            "Distance_Miles": miles,
            "Fuel_Gallons": gallons,
            "Route_Type": np.random.choice(["Urban", "Highway", "Mixed"], size=n)
        })

        el_vin = CanonicalEligibilityEngine.evaluate_column("VIN_Number", df["VIN_Number"], n)
        assert el_vin["analytical_eligible"] is False
        assert el_vin["exclusion_reason"] == "PRIMARY_KEY"

    # =========================================================================
    # DATASET H: Clinical / Healthcare Data
    # =========================================================================
    def test_dataset_h_clinical_trials(self):
        n = 70
        df = pd.DataFrame({
            "Patient_ID": [f"PT-{i:04d}" for i in range(n)],
            "Age": np.random.randint(21, 80, size=n),
            "Systolic_BP": np.random.randint(100, 160, size=n),
            "Diastolic_BP": np.random.randint(65, 100, size=n),
            "Treatment_Arm": np.random.choice(["Placebo", "Dose_10mg", "Dose_50mg"], size=n)
        })

        eda = run_automated_eda(df)
        assert eda["overview"]["num_rows"] == n

    # =========================================================================
    # DATASET I: Minimal / Small Sample (N = 3)
    # =========================================================================
    def test_dataset_i_minimal_small_sample(self):
        df = pd.DataFrame({
            "ID": [1, 2, 3],
            "Score_A": [10.0, 20.0, 30.0],
            "Score_B": [15.0, 25.0, 35.0]
        })

        eda = run_automated_eda(df)
        assert eda["overview"]["num_rows"] == 3
        # Should execute cleanly without ZeroDivisionError
        ins = generate_insights_and_story(df, eda)
        assert isinstance(ins["insights"], list)

    # =========================================================================
    # DATASET J: Extreme Missingness (>80% missing)
    # =========================================================================
    def test_dataset_j_extreme_missingness(self):
        n = 100
        sparse_vals = [np.nan] * 90 + [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        all_null_vals = [np.nan] * 100

        df = pd.DataFrame({
            "Record_ID": list(range(n)),
            "Sparse_Metric": sparse_vals,
            "Empty_Metric": all_null_vals,
            "Full_Metric": np.random.normal(50.0, 5.0, size=n)
        })

        pca_res = compute_pca(df)
        # Extreme missingness should safely mark PCA as not applicable or exclude sparse columns
        assert pca_res["is_valid"] is False
        assert pca_res["status"] == "NOT_APPLICABLE"

    # =========================================================================
    # DATASET K: Zero Variance / Constant Column
    # =========================================================================
    def test_dataset_k_constant_zero_variance(self):
        n = 50
        df = pd.DataFrame({
            "ID": list(range(n)),
            "Const_Val": [42.0] * n,
            "Dynamic_Val": np.random.uniform(1.0, 10.0, size=n)
        })

        corr = safe_correlation(df, "Const_Val", "Dynamic_Val")
        assert corr["is_computable"] is False
        assert corr["correlation_status"] == "not_applicable"
        assert "zero variance" in corr["reason"].lower()

    # =========================================================================
    # DATASET L: High-Cardinality String IDs & Hashes
    # =========================================================================
    def test_dataset_l_high_cardinality_identifiers(self):
        n = 50
        df = pd.DataFrame({
            "UUID_Token": [str(uuid.uuid4()) for _ in range(n)],
            "Hash_Key": [f"sha256_{i}_{uuid.uuid4().hex[:8]}" for i in range(n)],
            "Value": np.random.uniform(10.0, 50.0, size=n)
        })

        contract = DatasetContract.create_from_pipeline("hash-test", df, df)
        assert contract.get_column("UUID_Token").semantic_role == "IDENTIFIER"
        assert contract.get_column("Hash_Key").semantic_role == "IDENTIFIER"

    # =========================================================================
    # DATASET M: Rich Temporal Timestamps
    # =========================================================================
    def test_dataset_m_rich_temporal_data(self):
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        df = pd.DataFrame({
            "Timestamp": dates,
            "Daily_Throughput": np.random.poisson(100, size=len(dates)),
            "Station": ["Station_A"] * 30 + ["Station_B"] * 30
        })

        eda = run_automated_eda(df)
        assert eda["time_series_analysis"]["status"] == "sufficient_data"

    # =========================================================================
    # DATASET N: Formatted Currency Strings
    # =========================================================================
    def test_dataset_n_currency_strings(self):
        df_raw = pd.DataFrame({
            "Raw_Currency": ["$1,234.50", "€ 500.00", "$99.99", "( $50.00 )"],
            "Label": ["A", "B", "C", "D"]
        })

        df_clean = coerce_numeric_columns(df_raw.copy())
        assert pd.api.types.is_numeric_dtype(df_clean["Raw_Currency"])
        assert df_clean["Raw_Currency"].iloc[0] == 1234.50
        assert df_clean["Raw_Currency"].iloc[3] == -50.00

    # =========================================================================
    # DATASET O: Formatted Percentage Strings
    # =========================================================================
    def test_dataset_o_percentage_strings(self):
        df_raw = pd.DataFrame({
            "Raw_Pct": ["15.5%", "99.0%", "-4.2%", "(12.0%)"],
            "Label": ["A", "B", "C", "D"]
        })

        df_clean = coerce_numeric_columns(df_raw.copy())
        assert pd.api.types.is_numeric_dtype(df_clean["Raw_Pct"])
        assert df_clean["Raw_Pct"].iloc[0] == 15.5
        assert df_clean["Raw_Pct"].iloc[3] == -12.0

    # =========================================================================
    # DATASET P: Synthetic Test Fixtures & Adversarial Columns
    # =========================================================================
    def test_dataset_p_synthetic_fixtures(self):
        n = 40
        df = pd.DataFrame({
            "real_metric": np.random.uniform(1.0, 10.0, size=n),
            "test_fixture_val": np.random.uniform(1.0, 10.0, size=n),
            "relationship_test_col": np.random.uniform(1.0, 10.0, size=n),
            "col_dummy": [1] * n
        })

        el_real = CanonicalEligibilityEngine.evaluate_column("real_metric", df["real_metric"], n)
        assert el_real["analytical_eligible"] is True

        el_test = CanonicalEligibilityEngine.evaluate_column("test_fixture_val", df["test_fixture_val"], n)
        assert el_test["analytical_eligible"] is False
        assert el_test["exclusion_reason"] == "TEST_FIXTURE"

        el_rel = CanonicalEligibilityEngine.evaluate_column("relationship_test_col", df["relationship_test_col"], n)
        assert el_rel["analytical_eligible"] is False
        assert el_rel["exclusion_reason"] == "TEST_FIXTURE"
