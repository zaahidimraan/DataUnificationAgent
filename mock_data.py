import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

def generate_complex_data():
    print("🚀 Generating complex mock data...")
    
    # Ensure output directory exists
    output_dir = "mock_data_batch"
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 1. MASTER FILE (Excel - Multi-sheet)
    # ==========================================
    # Contains the "Truth" IDs for Flats and Villas
    
    # Generate IDs
    flat_ids = [f'FL-{i:03d}' for i in range(1, 21)]  # FL-001 to FL-020
    villa_ids = [f'VL-{i:03d}' for i in range(1, 11)] # VL-001 to VL-010
    all_property_ids = flat_ids + villa_ids

    df_flats = pd.DataFrame({
        'Property_ID': flat_ids,
        'Type': 'Flat',
        'Area_SqFt': np.random.randint(600, 1500, len(flat_ids)),
        'Bedrooms': np.random.choice([1, 2, 3], len(flat_ids)),
        'Base_Price': np.random.randint(150000, 400000, len(flat_ids))
    })

    df_villas = pd.DataFrame({
        'Ref_Code': villa_ids, # Note: Different header name for ID
        'Type': 'Villa',
        'Plot_Area': np.random.randint(2000, 5000, len(villa_ids)),
        'Has_Pool': np.random.choice(['Yes', 'No'], len(villa_ids)),
        'Base_Price': np.random.randint(800000, 2000000, len(villa_ids))
    })

    master_path = os.path.join(output_dir, "01_Master_Property_List.xlsx")
    with pd.ExcelWriter(master_path) as writer:
        df_flats.to_excel(writer, sheet_name='Flats_Registry', index=False)
        df_villas.to_excel(writer, sheet_name='Villas_Registry', index=False)
    print(f"✅ Created Master File: {master_path}")


    # ==========================================
    # 2. REPAIR HISTORY (CSV)
    # ==========================================
    # Foreign Key: 'Prop_ID' maps to Master IDs
    # Primary Key: 'Ticket_ID'
    
    num_repairs = 50
    repair_data = {
        'Ticket_ID': [f'REP-{random.randint(1000, 9999)}' for _ in range(num_repairs)],
        'Prop_ID': [random.choice(all_property_ids) for _ in range(num_repairs)],
        'Repair_Date': [datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365)) for _ in range(num_repairs)],
        'Issue_Type': np.random.choice(['Plumbing', 'Electrical', 'HVAC', 'Paint', 'Roof'], num_repairs),
        'Cost_USD': np.random.randint(100, 2500, num_repairs),
        'Status': np.random.choice(['Completed', 'Pending', 'In Progress'], num_repairs, p=[0.7, 0.2, 0.1])
    }
    
    df_repairs = pd.DataFrame(repair_data)
    repair_path = os.path.join(output_dir, "02_Repair_History_Log.csv")
    df_repairs.to_csv(repair_path, index=False)
    print(f"✅ Created Repair Log: {repair_path}")


    # ==========================================
    # 3. PRICE INCREASE HISTORY (Excel)
    # ==========================================
    # Tracks market value changes over years.
    # Composite Key logic: ID + Year
    
    price_records = []
    years = [2021, 2022, 2023]
    
    for pid in all_property_ids:
        base = 200000 if 'FL' in pid else 900000
        for year in years:
            # Random fluctuation
            market_val = base * (1 + (random.uniform(-0.02, 0.08) * (year - 2020)))
            price_records.append({
                'Asset_Ref': pid,  # Foreign Key
                'Valuation_Year': year,
                'Market_Value': round(market_val, 2),
                'Assessor': random.choice(['Agency A', 'Agency B', 'Govt'])
            })

    df_prices = pd.DataFrame(price_records)
    price_path = os.path.join(output_dir, "03_Market_Value_Trends.xlsx")
    df_prices.to_excel(price_path, sheet_name='Yearly_Valuations', index=False)
    print(f"✅ Created Price Trends: {price_path}")


    # ==========================================
    # 4. TENANT LEASING HISTORY (CSV)
    # ==========================================
    # Foreign Key: 'Unit_Number'
    
    tenant_records = []
    for _ in range(30): # 30 random leases
        start_date = datetime(2022, 1, 1) + timedelta(days=random.randint(0, 500))
        tenant_records.append({
            'Lease_ID': f'L-{random.randint(100, 999)}',
            'Unit_Number': random.choice(all_property_ids),
            'Tenant_Name': f"Tenant_{random.randint(1, 100)}",
            'Start_Date': start_date.strftime('%Y-%m-%d'),
            'Lease_Term_Months': random.choice([6, 12, 24]),
            'Monthly_Rent': random.randint(1200, 5000)
        })
        
    df_tenants = pd.DataFrame(tenant_records)
    tenant_path = os.path.join(output_dir, "04_Lease_Records.csv")
    df_tenants.to_csv(tenant_path, index=False)
    print(f"✅ Created Lease Records: {tenant_path}")


    # ==========================================
    # 5. TAX & UTILITY EXPENSES (Excel)
    # ==========================================
    # Foreign Key: 'Property_Code'
    
    expense_records = []
    for pid in all_property_ids:
        # Create records for Q1 and Q2 2024
        for q in ['Q1', 'Q2']:
            expense_records.append({
                'Property_Code': pid,
                'Fiscal_Year': 2024,
                'Quarter': q,
                'Municipal_Tax': random.randint(500, 1500),
                'Water_Fee': random.randint(100, 300),
                'Electricity_Fee': random.randint(200, 800)
            })

    df_expenses = pd.DataFrame(expense_records)
    expense_path = os.path.join(output_dir, "05_Tax_And_Utilities.xlsx")
    df_expenses.to_excel(expense_path, sheet_name='2024_Expenses', index=False)
    print(f"✅ Created Expenses File: {expense_path}")

    print("\n🎉 All files generated in folder: /mock_data_batch")

if __name__ == "__main__":
    generate_complex_data()