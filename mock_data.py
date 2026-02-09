import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# ==========================================
# OPTION 1: RELATIONAL DATA (Primary/Foreign Keys)
# ==========================================
def generate_relational_data():
    output_dir = "mock_data_relational"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🚀 Generating OPTION 1: Relational Data in '{output_dir}/'...")

    # 1. MASTER FILE
    flat_ids = [f'FL-{i:03d}' for i in range(1, 21)]
    villa_ids = [f'VL-{i:03d}' for i in range(1, 11)]
    all_property_ids = flat_ids + villa_ids

    df_flats = pd.DataFrame({
        'Property_ID': flat_ids,
        'Type': 'Flat',
        'Area_SqFt': np.random.randint(600, 1500, len(flat_ids)),
        'Base_Price': np.random.randint(150000, 400000, len(flat_ids))
    })
    df_villas = pd.DataFrame({
        'Ref_Code': villa_ids,
        'Type': 'Villa',
        'Has_Pool': np.random.choice(['Yes', 'No'], len(villa_ids)),
        'Base_Price': np.random.randint(800000, 2000000, len(villa_ids))
    })

    with pd.ExcelWriter(os.path.join(output_dir, "01_Master_Property_List.xlsx")) as writer:
        df_flats.to_excel(writer, sheet_name='Flats_Registry', index=False)
        df_villas.to_excel(writer, sheet_name='Villas_Registry', index=False)

    # 2. REPAIR HISTORY
    num_repairs = 50
    df_repairs = pd.DataFrame({
        'Ticket_ID': [f'REP-{random.randint(1000, 9999)}' for _ in range(num_repairs)],
        'Prop_ID': [random.choice(all_property_ids) for _ in range(num_repairs)],
        'Repair_Date': [datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365)) for _ in range(num_repairs)],
        'Cost_USD': np.random.randint(100, 2500, num_repairs)
    })
    df_repairs.to_csv(os.path.join(output_dir, "02_Repair_History_Log.csv"), index=False)

    # 3. PRICE TRENDS
    price_records = []
    for pid in all_property_ids:
        for year in [2021, 2022, 2023]:
            price_records.append({
                'Asset_Ref': pid,
                'Valuation_Year': year,
                'Market_Value': random.randint(200000, 900000)
            })
    pd.DataFrame(price_records).to_excel(os.path.join(output_dir, "03_Market_Value_Trends.xlsx"), index=False)

    # 4. LEASE RECORDS
    tenant_records = []
    for _ in range(30):
        tenant_records.append({
            'Lease_ID': f'L-{random.randint(100, 999)}',
            'Unit_Number': random.choice(all_property_ids),
            'Tenant': f"Tenant_{random.randint(1, 100)}",
            'Rent': random.randint(1200, 5000)
        })
    pd.DataFrame(tenant_records).to_csv(os.path.join(output_dir, "04_Lease_Records.csv"), index=False)

    # 5. EXPENSES
    expense_records = []
    for pid in all_property_ids:
        expense_records.append({
            'Property_Code': pid,
            'Tax': random.randint(500, 1500),
            'Utilities': random.randint(200, 800)
        })
    pd.DataFrame(expense_records).to_excel(os.path.join(output_dir, "05_Tax_And_Utilities.xlsx"), index=False)

    print("✅ Option 1 Complete.")


# ==========================================
# OPTION 2: COMPOSITE DATA (Hard/Messy Keys)
# ==========================================
def generate_composite_data():
    output_dir = "mock_data_composite"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🚀 Generating OPTION 2: Composite/Hard Data in '{output_dir}/'...")

    cities = ['NY', 'LDN', 'DXB']
    buildings = [f'B-{i}' for i in range(1, 6)]

    # BASE DATA
    flat_records = []
    for city in cities:
        for bldg in buildings:
            for f_num in range(101, 106):
                flat_records.append({'City': city, 'Building': bldg, 'Flat_ID': f_num, 'Type': 'Flat'})

    villa_records = []
    for city in cities:
        for v_num in range(1, 6):
            villa_records.append({'City': city, 'Villa_ID': f'V-{v_num}', 'Type': 'Villa'})

    house_records = [{'House_Code': f'H-{h:03d}', 'Type': 'House'} for h in range(1, 11)]

    # FILE 1: MASTER REGISTRY
    df_f = pd.DataFrame(flat_records).rename(columns={'City': 'City_Code', 'Building': 'Bldg_Ref', 'Flat_ID': 'Unit_No'})
    df_v = pd.DataFrame(villa_records).rename(columns={'City': 'City_Code', 'Villa_ID': 'Villa_No'})
    df_h = pd.DataFrame(house_records)

    with pd.ExcelWriter(os.path.join(output_dir, "01_Property_Registry.xlsx")) as writer:
        df_f.to_excel(writer, sheet_name='Flats_Master', index=False)
        df_v.to_excel(writer, sheet_name='Villas_Master', index=False)
        df_h.to_excel(writer, sheet_name='Houses_Master', index=False)

    # FILE 2: RENTALS (Composite Key Mapping)
    rent_data = []
    for rec in flat_records:
        if random.random() > 0.3:
            rent_data.append({
                'C_ID': rec['City'], 
                'B_ID': rec['Building'], 
                'F_ID': rec['Flat_ID'], 
                'Rent': random.randint(1500, 4000)
            })
    pd.DataFrame(rent_data).to_csv(os.path.join(output_dir, "02_Rental_Agreements.csv"), index=False)

    # FILE 3: SALES (Villas & Houses)
    # FIX: Replaced 5e5, 1e6 with explicit integers
    sale_villas = [{'Region': r['City'], 'V_Code': r['Villa_ID'], 'Price': random.randint(500000, 1000000)} for r in villa_records]
    # FIX: Replaced 3e5, 8e5 with explicit integers
    sale_houses = [{'H_Ref': h['House_Code'], 'Price': random.randint(300000, 800000)} for h in house_records]
    
    with pd.ExcelWriter(os.path.join(output_dir, "03_Market_Prices.xlsx")) as writer:
        pd.DataFrame(sale_villas).to_excel(writer, sheet_name='Villa_Prices', index=False)
        pd.DataFrame(sale_houses).to_excel(writer, sheet_name='House_Prices', index=False)

    # FILE 4: MAINTENANCE (Mixed Keys)
    maint_data = []
    for r in flat_records[:5]:
        maint_data.append({'Key_1': r['City'], 'Key_2': r['Building'], 'Key_3': r['Flat_ID'], 'Type': 'Flat', 'Cost': 200})
    for r in villa_records[:5]:
        maint_data.append({'Key_1': r['City'], 'Key_2': r['Villa_ID'], 'Key_3': None, 'Type': 'Villa', 'Cost': 800})
    for r in house_records[:5]:
        maint_data.append({'Key_1': r['House_Code'], 'Key_2': None, 'Key_3': None, 'Type': 'House', 'Cost': 400})
    
    pd.DataFrame(maint_data).to_csv(os.path.join(output_dir, "04_Maintenance_Log.csv"), index=False)

    # FILE 5: TAXES (Generic Keys)
    tax_data = []
    for r in flat_records:
        tax_data.append({'Ref_A': r['City'], 'Ref_B': r['Building'], 'Ref_C': r['Flat_ID'], 'Tax': 150})
    for r in villa_records:
        tax_data.append({'Ref_A': r['City'], 'Ref_B': r['Villa_ID'], 'Ref_C': 'N/A', 'Tax': 400})

    with pd.ExcelWriter(os.path.join(output_dir, "05_Government_Taxes.xlsx")) as writer:
        pd.DataFrame(tax_data).to_excel(writer, sheet_name='2024_Taxes', index=False)

    print("✅ Option 2 Complete.")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--------------------------------------------------")
    print("   MOCK DATA GENERATOR FOR DATA AGENT TESTING")
    print("--------------------------------------------------")
    print("1. Generate Relational Data (Standard Primary/Foreign Keys)")
    print("2. Generate Composite Data (Hard, Mixed Column Names)")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == '1':
        generate_relational_data()
    elif choice == '2':
        generate_composite_data()
    else:
        print("Invalid choice. Please run the script again and type 1 or 2.")