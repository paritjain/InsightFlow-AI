import os
import random
import pandas as pd
import numpy as np

def generate_all_test_datasets(target_dir: str = "backend/test_datasets"):
    os.makedirs(target_dir, exist_ok=True)
    np.random.seed(42)
    random.seed(42)

    # -------------------------------------------------------------
    # DATASET A: test_sales.xlsx
    # -------------------------------------------------------------
    print("Generating Dataset A: test_sales.xlsx...")
    n_sales = 1200
    regions = ["West", "East", "North", "South", "Central"]
    region_variants = ["West", "west", "WEST", "West ", "East", "east", "North", "South", "Central"]
    categories = ["Enterprise Software", "Cloud Infrastructure", "Security", "Hardware", "Consulting"]
    products = [f"PROD_{i:03d}" for i in range(1, 21)]
    product_names = [f"Product {c} {i}" for c in ["Alpha", "Beta", "Gamma", "Delta"] for i in range(1, 6)]
    salespersons = ["Alice Smith", "Bob Jones", "Carol White", "David Green", "Eva Black"]

    dates = pd.date_range(start="2024-01-01", end="2024-08-15", freq="D")
    
    sales_data = []
    for i in range(1, n_sales + 1):
        order_id = f"ORD-{1000 + i}"
        order_date = random.choice(dates)
        cust_id = f"CUST-{(i % 250) + 1:04d}"
        prod_idx = random.randint(0, len(products) - 1)
        prod_id = products[prod_idx]
        prod_name = product_names[prod_idx]
        category = categories[prod_idx % len(categories)]
        region = random.choice(region_variants) if random.random() < 0.15 else random.choice(regions)
        sp = random.choice(salespersons)
        qty = random.randint(1, 50)
        unit_price = round(random.uniform(50.0, 1500.0), 2)
        discount = round(random.choice([0.0, 0.05, 0.1, 0.15, 0.2]), 2)
        
        # Inject anomalies / suspicious values
        if i == 50:
            qty = -10  # Negative quantity anomaly
        if i == 100:
            unit_price = -200.0  # Negative price anomaly
        if i == 150:
            qty = 500  # Outlier
            unit_price = 5000.0

        revenue = round(qty * unit_price * (1.0 - discount), 2)
        unit_cost = round(unit_price * random.uniform(0.4, 0.7), 2)
        cost = round(qty * unit_cost, 2)
        profit = round(revenue - cost, 2)

        sales_data.append({
            "Order_ID": order_id,
            "Order_Date": order_date.strftime("%Y-%m-%d"),
            "Customer_ID": cust_id,
            "Product_ID": prod_id,
            "Product": prod_name,
            "Category": category,
            "Region": region,
            "Salesperson": sp,
            "Quantity": qty,
            "Unit_Price": unit_price,
            "Revenue": revenue,
            "Cost": cost,
            "Discount": discount,
            "Profit": profit
        })

    df_sales = pd.DataFrame(sales_data)

    # Inject missing values
    for col in ["Salesperson", "Discount", "Region"]:
        mask = np.random.rand(len(df_sales)) < 0.04
        df_sales.loc[mask, col] = np.nan

    # Inject duplicate records
    dup_rows = df_sales.iloc[10:35].copy()
    df_sales = pd.concat([df_sales, dup_rows], ignore_index=True)

    sales_path = os.path.join(target_dir, "test_sales.xlsx")
    df_sales.to_excel(sales_path, index=False)
    print(f"Saved {sales_path} with {len(df_sales)} rows.")

    # -------------------------------------------------------------
    # DATASET B: test_customers.xlsx
    # -------------------------------------------------------------
    print("Generating Dataset B: test_customers.xlsx...")
    n_cust = 300
    cust_data = []
    industries = ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing", "Education"]
    segments = ["Enterprise", "Mid-Market", "SMB"]
    statuses = ["Active", "Active", "Active", "Churned", "Suspended"]

    for i in range(1, n_cust + 1):
        cid = f"CUST-{i:04d}"
        cname = f"Acme Enterprise {i}" if i % 2 == 0 else f"Global Corp {i}"
        reg = random.choice(regions)
        ind = random.choice(industries)
        acq_date = pd.Timestamp("2022-01-01") + pd.Timedelta(days=random.randint(0, 700))
        status = random.choice(statuses)
        segment = random.choice(segments)
        ann_rev = round(random.uniform(50000, 5000000), 2)
        ltv = round(ann_rev * random.uniform(1.5, 4.0), 2)
        last_date = acq_date + pd.Timedelta(days=random.randint(30, 500))

        cust_data.append({
            "Customer_ID": cid,
            "Customer_Name": cname,
            "Region": reg,
            "Industry": ind,
            "Acquisition_Date": acq_date.strftime("%Y-%m-%d"),
            "Customer_Status": status,
            "Customer_Segment": segment,
            "Annual_Revenue": ann_rev,
            "Lifetime_Value": ltv,
            "Last_Purchase_Date": last_date.strftime("%Y-%m-%d")
        })

    df_cust = pd.DataFrame(cust_data)
    cust_path = os.path.join(target_dir, "test_customers.xlsx")
    df_cust.to_excel(cust_path, index=False)
    print(f"Saved {cust_path} with {len(df_cust)} rows.")

    # -------------------------------------------------------------
    # DATASET C: test_products.xlsx
    # -------------------------------------------------------------
    print("Generating Dataset C: test_products.xlsx...")
    prod_data = []
    suppliers = ["Apex Suppliers", "Nexus Tech", "Vanguard Global", "Titan Distribution"]
    for i in range(len(products)):
        pid = products[i]
        pname = product_names[i]
        cat = categories[i % len(categories)]
        std_price = round(random.uniform(100.0, 2000.0), 2)
        cost = round(std_price * 0.55, 2)
        supplier = random.choice(suppliers)
        launch_date = pd.Timestamp("2021-06-01") + pd.Timedelta(days=random.randint(0, 600))

        prod_data.append({
            "Product_ID": pid,
            "Product": pname,
            "Category": cat,
            "Cost": cost,
            "Standard_Price": std_price,
            "Supplier": supplier,
            "Launch_Date": launch_date.strftime("%Y-%m-%d")
        })

    df_prod = pd.DataFrame(prod_data)
    prod_path = os.path.join(target_dir, "test_products.xlsx")
    df_prod.to_excel(prod_path, index=False)
    print(f"Saved {prod_path} with {len(df_prod)} rows.")

    # -------------------------------------------------------------
    # DATASET D: test_marketing.csv
    # -------------------------------------------------------------
    print("Generating Dataset D: test_marketing.csv...")
    mkt_dates = pd.date_range(start="2024-01-01", end="2024-07-31", freq="D")
    channels = ["Google Ads", "Meta Ads", "LinkedIn Ads", "Email Marketing", "Organic Search"]
    campaigns = ["Q1 Brand Awareness", "Q2 Lead Gen", "Summer Sale", "Retargeting"]

    mkt_data = []
    for d in mkt_dates:
        for ch in channels:
            spend = round(random.uniform(200.0, 3500.0), 2) if ch != "Organic Search" else 0.0
            impressions = int(spend * random.uniform(20, 50)) + random.randint(1000, 5000)
            clicks = int(impressions * random.uniform(0.02, 0.08))
            leads = int(clicks * random.uniform(0.05, 0.20))
            conversions = int(leads * random.uniform(0.10, 0.40))
            revenue = round(conversions * random.uniform(150.0, 800.0), 2)

            mkt_data.append({
                "Date": d.strftime("%Y-%m-%d"),
                "Campaign": random.choice(campaigns),
                "Channel": ch,
                "Spend": spend,
                "Impressions": impressions,
                "Clicks": clicks,
                "Leads": leads,
                "Conversions": conversions,
                "Revenue": revenue
            })

    df_mkt = pd.DataFrame(mkt_data)
    mkt_path = os.path.join(target_dir, "test_marketing.csv")
    df_mkt.to_csv(mkt_path, index=False)
    print(f"Saved {mkt_path} with {len(df_mkt)} rows.")

    # -------------------------------------------------------------
    # DATASET E: test_hr.xlsx
    # -------------------------------------------------------------
    print("Generating Dataset E: test_hr.xlsx...")
    n_hr = 500
    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]
    locations = ["New York", "San Francisco", "London", "Austin", "Remote"]
    job_levels = ["Junior", "Mid", "Senior", "Lead", "Director"]

    hr_data = []
    for i in range(1, n_hr + 1):
        emp_id = f"EMP-{i:04d}"
        dept = random.choice(departments)
        level = random.choice(job_levels)
        loc = random.choice(locations)
        gender = random.choice(["Male", "Female", "Non-Binary"])
        age = random.randint(22, 62)
        tenure = round(random.uniform(0.5, 12.0), 1)
        
        base_sal = 60000 if level == "Junior" else (90000 if level == "Mid" else 140000)
        salary = round(base_sal + random.uniform(-10000, 30000), 2)
        perf_score = round(random.uniform(2.5, 5.0), 1)
        sat_score = round(random.uniform(1.0, 5.0), 1)
        attrition = "Yes" if (sat_score < 2.5 and random.random() < 0.6) else "No"

        # Outlier
        if i == 45:
            salary = 850000.0  # CEO Outlier salary

        hr_data.append({
            "Employee_ID": emp_id,
            "Department": dept,
            "Job_Level": level,
            "Location": loc,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Salary": salary,
            "Performance_Score": perf_score,
            "Satisfaction_Score": sat_score,
            "Attrition": attrition
        })

    df_hr = pd.DataFrame(hr_data)
    hr_path = os.path.join(target_dir, "test_hr.xlsx")
    df_hr.to_excel(hr_path, index=False)
    print(f"Saved {hr_path} with {len(df_hr)} rows.")

    # -------------------------------------------------------------
    # DATASET F: test_messy_sales.xlsx
    # -------------------------------------------------------------
    print("Generating Dataset F: test_messy_sales.xlsx...")
    n_messy = 350
    messy_data = []
    date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y", "%Y.%m.%d"]
    
    for i in range(1, n_messy + 1):
        oid = f"ORD-{2000 + i}"
        
        # Mixed date formats & invalid dates
        if i % 15 == 0:
            odate = "INVALID_DATE_999"
        else:
            raw_dt = pd.Timestamp("2024-01-01") + pd.Timedelta(days=random.randint(0, 180))
            fmt = random.choice(date_formats)
            odate = raw_dt.strftime(fmt)

        cid = f"CUST-{(i % 50) + 1:04d}"
        pid = f"PROD-{(i % 10) + 1:03d}"
        prod_name = f"  Messy Product {i % 5}  "
        
        # Inconsistent region names with spaces & capitalization
        region = random.choice([" Mumbai ", "mumbai", "MUMBAI", "Delhi", "DELHI", " delhi ", "Bangalore", "bangalore"])

        # Numeric values stored as text with symbols/commas
        qty_str = f"{random.randint(1, 20)}"
        if i % 10 == 0:
            qty_str = f" {qty_str} "
        
        price_num = round(random.uniform(100, 2500), 2)
        price_str = f"${price_num:,.2f}" if i % 2 == 0 else f"₹{price_num:,.2f}"

        rev_num = round(price_num * int(qty_str.strip()), 2)
        rev_str = f"{rev_num:,.2f}"

        disc_str = f"{random.choice([5, 10, 15, 20])}%"

        messy_data.append({
            "Order_ID": oid,
            "Order_Date": odate,
            "Customer_ID": cid,
            "Product_ID": pid,
            "Product": prod_name,
            "Region": region,
            "Quantity": qty_str,
            "Unit_Price": price_str,
            "Revenue": rev_str,
            "Discount": disc_str,
            "Notes": "  Extra spacing  " if i % 3 == 0 else None
        })

    df_messy = pd.DataFrame(messy_data)
    
    # Add blank rows & exact duplicate rows
    df_messy = pd.concat([df_messy, df_messy.iloc[5:20]], ignore_index=True)

    messy_path = os.path.join(target_dir, "test_messy_sales.xlsx")
    df_messy.to_excel(messy_path, index=False)
    print(f"Saved {messy_path} with {len(df_messy)} rows.")
    print("All 6 test datasets successfully created in:", target_dir)

if __name__ == "__main__":
    generate_all_test_datasets()
