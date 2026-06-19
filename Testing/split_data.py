import pandas as pd
import os

input_path  = 'Testing\\PCA Result\\Outliered Data'
output_path = 'Testing\\Test Data Dirty'
run_count   = 7

categories = {
    'PC1_Combustion': 'C',
    'PC1_Systemic'  : 'S',
    'PC1_Global'    : 'G',
}

os.makedirs(output_path, exist_ok=True)

for col_name, prefix in categories.items():
    for i in range(1, run_count + 1):
        file = os.path.join(input_path, f'PCA_Results_run_to_failure{i}.csv')
        if not os.path.exists(file):
            print(f"File tidak ditemukan: {file}")
            continue

        values = pd.read_csv(file, usecols=[col_name])[col_name].values

        out_file = os.path.join(output_path, f'PC1_RTF_{prefix}{i}.csv')
        pd.DataFrame({col_name: values}).to_csv(out_file, index=False)
        print(f"Selesai: {prefix}{i}")