import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_demo_datasets(output_dir="backend/demo_data/store"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # 1. Customers Dataset
    num_customers = 150
    industries = ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing", "Education"]
    regions = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"]
    statuses = ["Active", "Active", "Active", "Churned", "Pending"]
    
    customer_ids = [f"CUST-{1000 + i}" for i in range(num_customers)]
    companies = [f"Acme Corp {i}" if i % 10 == 0 else f"Enterprise Inc {i}" for i in range(num_customers)]
    
    start_date = datetime(2023, 1, 1)
    acq_dates = [start_date + timedelta(days=int(np.random.randint(0, 500))) for _ in range(num_customers)]
    
    customers_df = pd.DataFrame({
        "Customer_ID": customer_ids,
        "Company_Name": companies,
        "Industry": [np.random.choice(industries) for _ in range(num_customers)],
        "Region": [np.random.choice(regions) for _ in range(num_customers)],
        "Acquisition_Date": [d.strftime("%Y-%m-%d") for d in acq_dates],
        "Customer_Status": [np.random.choice(statuses) for _ in range(num_customers)],
        "Account_Tier": [np.random.choice(["Enterprise", "Mid-Market", "SMB"], p=[0.2, 0.5, 0.3]) for _ in range(num_customers)]
    })
    
    # Introduce subtle dirty data issues for Data Quality engine to detect
    # Issue 1: Mixed whitespace & casing in Region
    dirty_regions = customers_df["Region"].tolist()
    dirty_regions[5] = "North America "
    dirty_regions[12] = "north america"
    dirty_regions[25] = "EUROPE"
    customers_df["Region"] = dirty_regions

    # Issue 2: Duplicate customer record
    customers_df = pd.concat([customers_df, customers_df.iloc[[10]]], ignore_index=True)

    # 2. Products Dataset
    products_data = [
        {"Product_ID": "PROD-101", "Product_Name": "InsightFlow Pro License", "Category": "Software", "Cost": 120.00, "Price": 499.00},
        {"Product_ID": "PROD-102", "Product_Name": "InsightFlow Enterprise", "Category": "Software", "Cost": 500.00, "Price": 1999.00},
        {"Product_ID": "PROD-103", "Product_Name": "Data Connector Plugin", "Category": "Add-on", "Cost": 25.00, "Price": 99.00},
        {"Product_ID": "PROD-104", "Product_Name": "Custom BI Dashboard", "Category": "Services", "Cost": 800.00, "Price": 2500.00},
        {"Product_ID": "PROD-105", "Product_Name": "Dedicated Support SLA", "Category": "Services", "Cost": 300.00, "Price": 1200.00},
        {"Product_ID": "PROD-106", "Product_Name": "API Throughput Booster", "Category": "Add-on", "Cost": 50.00, "Price": 299.00},
        {"Product_ID": "PROD-107", "Product_Name": "AI Story Generator", "Category": "Software", "Cost": 150.00, "Price": 699.00},
    ]
    products_df = pd.DataFrame(products_data)

    # 3. Sales Dataset
    num_orders = 1200
    salespersons = ["Sarah Connor", "Alex Mercer", "Diana Prince", "Bruce Wayne", "Clark Kent", "Peter Parker"]
    
    order_ids = [f"ORD-{20000 + i}" for i in range(num_orders)]
    order_custs = [np.random.choice(customer_ids) for _ in range(num_orders)]
    order_prods = [np.random.choice([p["Product_ID"] for p in products_data]) for _ in range(num_orders)]
    
    order_start = datetime(2024, 1, 1)
    order_dates = [order_start + timedelta(days=int(np.random.randint(0, 240))) for _ in range(num_orders)]
    
    prod_lookup = {p["Product_ID"]: p for p in products_data}
    
    rows = []
    for i in range(num_orders):
        p_id = order_prods[i]
        p_info = prod_lookup[p_id]
        qty = int(np.random.randint(1, 10))
        unit_price = p_info["Price"]
        
        # Simulate Q2 decline in Enterprise accounts for realistic story insight
        o_date = order_dates[i]
        is_q2 = 4 <= o_date.month <= 6
        cust_tier = customers_df.loc[customers_df["Customer_ID"] == order_custs[i], "Account_Tier"].values
        tier_val = cust_tier[0] if len(cust_tier) > 0 else "SMB"
        
        discount = float(np.random.choice([0.0, 0.05, 0.10, 0.15], p=[0.5, 0.3, 0.15, 0.05]))
        if is_q2 and tier_val == "Enterprise" and np.random.rand() > 0.4:
            qty = max(1, qty - 2) # Enterprise drop in Q2
        
        gross_rev = qty * unit_price
        net_rev = round(gross_rev * (1 - discount), 2)
        total_cost = round(qty * p_info["Cost"], 2)
        
        rows.append({
            "Order_ID": order_ids[i],
            "Customer_ID": order_custs[i],
            "Product_ID": p_id,
            "Order_Date": o_date.strftime("%Y-%m-%d"),
            "Salesperson": np.random.choice(salespersons),
            "Quantity": qty,
            "Discount": discount,
            "Revenue": net_rev,
            "Cost": total_cost,
            "Margin": round(net_rev - total_cost, 2),
            "Region": np.random.choice(regions)
        })
        
    sales_df = pd.DataFrame(rows)
    
    # Introduce dirty issues in Sales dataset for detection
    # Issue 3: Missing dates and null values
    sales_df.loc[15, "Revenue"] = np.nan
    sales_df.loc[42, "Order_Date"] = np.nan
    sales_df.loc[88, "Salesperson"] = "  Sarah Connor "

    # Save to Excel & CSV files
    sales_path = os.path.join(output_dir, "Sales_2024.xlsx")
    cust_path = os.path.join(output_dir, "Customers.xlsx")
    prod_path = os.path.join(output_dir, "Products.xlsx")
    
    sales_df.to_excel(sales_path, index=False)
    customers_df.to_excel(cust_path, index=False)
    products_df.to_excel(prod_path, index=False)
    
    print(f"Demo datasets generated successfully in {output_dir}")
    return sales_path, cust_path, prod_path

if __name__ == "__main__":
    generate_demo_datasets()
