import pandas as pd
import random

def create_complex_data():
    # File 1: Master Records (Static Info)
    # Sheet A: Flats
    df_flats = pd.DataFrame({
        'Flat_ID': [f'F-{i}' for i in range(1, 11)],
        'Area': [random.randint(500, 1000) for _ in range(10)],
        'Purchase_Price': [random.randint(100000, 200000) for _ in range(10)]
    })
    
    # Sheet B: Villas
    df_villas = pd.DataFrame({
        'Villa_Ref': [f'V-{i}' for i in range(1, 6)],
        'Garden_Size': [random.randint(200, 500) for _ in range(5)],
        'Purchase_Price': [random.randint(500000, 900000) for _ in range(5)]
    })

    # File 2: History (Transactions)
    # Sheet: Repair_Logs (Links to both Flats and Villas)
    
    # Create mixed IDs (Flats and Villas having repairs)
    repair_ids = [f'F-{random.randint(1,10)}' for _ in range(15)] + \
                 [f'V-{random.randint(1,5)}' for _ in range(5)]
                 
    df_repairs = pd.DataFrame({
        'Property_Code': repair_ids, # Note: Different header name!
        'Repair_Date': pd.date_range(start='1/1/2023', periods=20),
        'Cost': [random.randint(100, 2000) for _ in range(20)],
        'Description': ['Plumbing', 'Electrical', 'Paint', 'Roof', 'Window'] * 4
    })

    # Save File 1
    with pd.ExcelWriter('data_master_properties.xlsx') as writer:
        df_flats.to_excel(writer, sheet_name='Flats_Master', index=False)
        df_villas.to_excel(writer, sheet_name='Villas_Master', index=False)

    # Save File 2
    with pd.ExcelWriter('data_property_history.xlsx') as writer:
        df_repairs.to_excel(writer, sheet_name='Repairs_History', index=False)

    print("Created 'data_master_properties.xlsx' and 'data_property_history.xlsx'")

if __name__ == "__main__":
    create_complex_data()