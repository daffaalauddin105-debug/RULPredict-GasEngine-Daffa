import pandas as pd
import joblib
import os
import sys

def batch_load_scalers(file_list):
    all_results = []
    
    # Mendapatkan path folder tempat script ini berada
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Mencari file di folder: {script_dir}")

    for file_name in file_list:
        # Menggabungkan path folder script dengan nama file
        full_path = os.path.join(script_dir, file_name)
        
        if not os.path.exists(full_path):
            print(f"Peringatan: File {file_name} tidak ditemukan di {script_dir}. Melewati...")
            continue
            
        # Memuat objek scaler
        scaler = joblib.load(full_path)
        
        # Ekstrak data berdasarkan tipe scaler
        data_dict = {'File_Source': file_name}
        params = {}
        
        # Cek jika StandardScaler[cite: 1, 5, 6]
        if hasattr(scaler, 'mean_'):
            data_dict['Type'] = 'StandardScaler'
            params = {
                'Mean': scaler.mean_,
                'Variance': scaler.var_,
                'Scale': scaler.scale_
            }
        # Cek jika MinMaxScaler[cite: 2, 3, 4]
        elif hasattr(scaler, 'data_min_'):
            data_dict['Type'] = 'MinMaxScaler'
            params = {
                'Min': scaler.data_min_,
                'Max': scaler.data_max_,
                'Range': scaler.data_range_,
                'Scale': scaler.scale_
            }
            
        if params:
            df_temp = pd.DataFrame(params)
            df_temp['File_Source'] = file_name
            df_temp['Type'] = data_dict['Type']
            all_results.append(df_temp)

    # Menggabungkan semua data menjadi satu tabel
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        output_name = os.path.join(script_dir, 'all_scalers_report.csv')
        final_df.to_csv(output_name, index=False)
        print(f"\nSelesai! Data berhasil disimpan ke: {output_name}")
    else:
        print("\nError: Tidak ada data yang berhasil dimuat. Pastikan file .pkl ada di folder yang sama dengan script.")

# Daftar file sesuai permintaan
files_to_read = [
    "scaler_minmax_Combustion.pkl",
    "scaler_std_Combustion.pkl",
]

if __name__ == "__main__":
    batch_load_scalers(files_to_read)