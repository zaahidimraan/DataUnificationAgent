import pandas as pd
import numpy as np
import random
import os

def generate_hard_data():
    print("🚀 Generating Challenge Data (Composite Keys & schema variations)...")
    
    output_dir = "complex_data"
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 0. BASE DATA GENERATION (The "Truth")
    # ==========================================
    # We generate a "Truth" list first so that the files actually link together,
    # even if the column names are different.
    
    cities = ['NY', 'LDN', 'DXB']
    buildings = [f'B-{i}' for i in range(1, 6)] # B-1 to B-5
    
    # 1. FLATS (Composite Key: City + Building + Flat_Num)
    flat_records = []
    for city in cities:
        for bldg in buildings:
            for f_num in range(101, 106): # Flats 101-105
                flat_records.append({
                    'City': city,
                    'Building': bldg,
                    'Flat_ID': f_num,
                    'Type': 'Flat',
                    'Area': random.randint(500, 1200)
                })

    # 2. VILLAS (Composite Key: City + Villa_No)
    villa_records = []
    for city in cities:
        for v_num in range(1, 6): # Villas 1-5 per city
            villa_records.append({
                'City': city,
                'Villa_ID': f'V-{v_num}',
                'Type': 'Villa',
                'Garden': random.choice(['Yes', 'No'])
            })

    # 3. HOUSES (Simple Key: House_Code)
    house_records = []
    for h in range(1, 11):
        house_records.append({
            'House_Code': f'H-{h:03d}',
            'Type': 'House',
            'Floors': random.randint(1, 3)
        })

    # ==========================================
    # FILE 1: MASTER REGISTRY (Excel - Multi Sheet)
    # ==========================================
    # Scenario: Clean-ish headers, but different naming conventions per sheet.
    
    df_f = pd.DataFrame(flat_records)
    df_f.rename(columns={'City': 'City_Code', 'Building': 'Bldg_Ref', 'Flat_ID': 'Unit_No'}, inplace=True)
    
    df_v = pd.DataFrame(villa_records)
    df_v.rename(columns={'City': 'City_Code', 'Villa_ID': 'Villa_No'}, inplace=True)
    
    df_h = pd.DataFrame(house_records)
    # House keeps 'House_Code'
    
    path1 = os.path.join(output_dir, "01_Property_Registry.xlsx")
    with pd.ExcelWriter(path1) as writer:
        df_f.to_excel(writer, sheet_name='Flats_Master', index=False)
        df_v.to_excel(writer, sheet_name='Villas_Master', index=False)
        df_h.to_excel(writer, sheet_name='Houses_Master', index=False)
    
    print(f"✅ Created File 1: {path1} (Master Registry)")

    # ==========================================
    # FILE 2: RENTAL CONTRACTS (CSV) - Flats Only
    # ==========================================
    # Scenario: COMPOSITE KEY MAPPING REQUIRED
    # Headers are completely different: 'C_ID', 'B_ID', 'F_ID'
    
    rent_data = []
    for rec in flat_records:
        if random.random() > 0.3: # 70% have rent info
            rent_data.append({
                'C_ID': rec['City'],         # Maps to City_Code
                'B_ID': rec['Building'],     # Maps to Bldg_Ref
                'F_ID': rec['Flat_ID'],      # Maps to Unit_No
                'Monthly_Rent': random.randint(1500, 4000),
                'Tenant': f"Tenant_{random.randint(100,999)}"
            })
            
    df_rent = pd.DataFrame(rent_data)
    path2 = os.path.join(output_dir, "02_Rental_Agreements.csv")
    df_rent.to_csv(path2, index=False)
    print(f"✅ Created File 2: {path2} (Rentals - Flats)")

    # ==========================================
    # FILE 3: SALES MARKET PRICES (Excel) - Villas & Houses
    # ==========================================
    # Scenario: Different Property Types, Different Keys
    
    # Villas (Composite: Region + V_Code)
    sale_villas = []
    for rec in villa_records:
        sale_villas.append({
            'Region': rec['City'],        # Maps to City_Code
            'V_Code': rec['Villa_ID'],    # Maps to Villa_No
            'Sale_Price': random.randint(500000, 1500000)
        })

    # Houses (Simple: H_Ref)
    sale_houses = []
    for rec in house_records:
        sale_houses.append({
            'H_Ref': rec['House_Code'],   # Maps to House_Code
            'Listing_Price': random.randint(300000, 800000)
        })

    path3 = os.path.join(output_dir, "03_Market_Prices.xlsx")
    with pd.ExcelWriter(path3) as writer:
        pd.DataFrame(sale_villas).to_excel(writer, sheet_name='Villa_Prices', index=False)
        pd.DataFrame(sale_houses).to_excel(writer, sheet_name='House_Prices', index=False)
    print(f"✅ Created File 3: {path3} (Sales - Villas/Houses)")

    # ==========================================
    # FILE 4: MAINTENANCE LOGS (CSV) - All Types Mixed
    # ==========================================
    # Scenario: The "Messy" File.
    # It tries to use one set of columns for everything.
    # Key columns: 'Key_1', 'Key_2', 'Key_3'
    # For Flats: Key_1=City, Key_2=Bldg, Key_3=Flat
    # For Villas: Key_1=City, Key_2=Villa, Key_3=NaN
    # For Houses: Key_1=House, Key_2=NaN, Key_3=NaN
    
    maint_data = []
    
    # Add some flats
    for rec in flat_records[:5]:
        maint_data.append({
            'Key_1': rec['City'],
            'Key_2': rec['Building'],
            'Key_3': rec['Flat_ID'],
            'Property_Type': 'Flat',
            'Cost': random.randint(100, 500)
        })
        
    # Add some villas
    for rec in villa_records[:5]:
        maint_data.append({
            'Key_1': rec['City'],
            'Key_2': rec['Villa_ID'],
            'Key_3': None, # Empty
            'Property_Type': 'Villa',
            'Cost': random.randint(500, 1000)
        })

    # Add some houses
    for rec in house_records[:5]:
        maint_data.append({
            'Key_1': rec['House_Code'],
            'Key_2': None,
            'Key_3': None,
            'Property_Type': 'House',
            'Cost': random.randint(200, 600)
        })

    path4 = os.path.join(output_dir, "04_Maintenance_Log.csv")
    pd.DataFrame(maint_data).to_csv(path4, index=False)
    print(f"✅ Created File 4: {path4} (Maintenance - Mixed Keys)")

    # ==========================================
    # FILE 5: GOVT TAXES (Excel)
    # ==========================================
    # Scenario: Generic column names 'Ref_A', 'Ref_B', 'Ref_C'
    
    tax_data = []
    # Mix of all data again
    for rec in flat_records:
        tax_data.append({
            'Ref_A': rec['City'],
            'Ref_B': rec['Building'],
            'Ref_C': rec['Flat_ID'],
            'Tax_Amt': 150
        })
    for rec in villa_records:
        tax_data.append({
            'Ref_A': rec['City'],
            'Ref_B': rec['Villa_ID'],
            'Ref_C': 'N/A',
            'Tax_Amt': 400
        })
        
    path5 = os.path.join(output_dir, "05_Government_Taxes.xlsx")
    with pd.ExcelWriter(path5) as writer:
        pd.DataFrame(tax_data).to_excel(writer, sheet_name='2024_Taxes', index=False)
    print(f"✅ Created File 5: {path5} (Taxes - Generic Keys)")

    print("\n🎉 DONE! 5 Complex files generated in 'complex_data/' folder.")

if __name__ == "__main__":
    generate_hard_data()