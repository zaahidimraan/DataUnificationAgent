import pandas as pd
import numpy as np
import os

def create_messy_data():
    print("Generating messy data...")

    # --- Sheet 1: Flats ---
    # Scenario: Standard columns, explicit ID name
    df_flats = pd.DataFrame({
        'Flat_ID': ['1001', '1002', '1003', '1004'],
        'Price': [120000, 135000, 95000, 110000],
        'Area': [650, 700, 500, 600],
        'Bedrooms': [2, 2, 1, 2]
    })

    # --- Sheet 2: Houses ---
    # Scenario: 
    # 1. Different ID Header ('House_Ref')
    # 2. Semantic overlap ('Cost' instead of 'Price', 'SqFt' instead of 'Area')
    # 3. CONFLICT: ID '1001' exists here too! (Should trigger conflict resolution)
    df_houses = pd.DataFrame({
        'House_Ref': ['1001', '2002', '2003'],  # <--- '1001' duplicates a Flat ID
        'Cost': [250000, 320000, 280000],       # <--- Synonymous with Price
        'SqFt': [1200, 1500, 1350],             # <--- Synonymous with Area
        'Garage': ['Yes', 'Yes', 'No']          # <--- Unique column to Houses
    })

    # --- Sheet 3: Villas ---
    # Scenario: 
    # 1. Ambiguous ID Header ('Code')
    # 2. 'Amount' instead of 'Price'
    df_villas = pd.DataFrame({
        'Code': ['V-301', 'V-302', 'V-303'],
        'Amount': [500000, 750000, 600000],
        'Garden_Size': [500, 800, 600],
        'Pool': ['Yes', 'Yes', 'No']
    })

    # --- Save to Excel ---
    output_file = 'messy_properties.xlsx'
    
    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_flats.to_excel(writer, sheet_name='Flats', index=False)
        df_houses.to_excel(writer, sheet_name='Houses', index=False)
        df_villas.to_excel(writer, sheet_name='Villas', index=False)

    print(f"Success! Created '{output_file}' with 3 sheets.")
    print("1. Flats (ID: Flat_ID)")
    print("2. Houses (ID: House_Ref) -> Contains ID conflict '1001'")
    print("3. Villas (ID: Code)")

if __name__ == "__main__":
    create_messy_data()