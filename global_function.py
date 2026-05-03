import os
import pandas as pd
import numpy as np

# =================================================================
# 1. OPERATIONAL FUNCTIONS
# =================================================================

import pandas as pd
import numpy as np

def pchip_imputer(df, columns=None, log_file="imputation_log.txt"):
    """
    Refined PCHIP Imputer:
    1. Mendeteksi titik akhir data valid tiap sensor secara independen.
    2. Hanya mengisi lubang (NaN) yang berada di antara titik data valid.
    3. Membiarkan baris kosong di akhir (trailing NaNs) tetap kosong.
    """
    df_result = df.copy()
    target_cols = columns if columns else df_result.select_dtypes(include=[np.number]).columns
    
    log_entries = []
    
    for col in target_cols:
        # Menemukan index pertama dan terakhir yang memiliki nilai (bukan NaN)
        first_valid_idx = df_result[col].first_valid_index()
        last_valid_idx = df_result[col].last_valid_index()
        
        # Jika kolom benar-benar kosong, lewati
        if last_valid_idx is None:
            msg = f"Sensor {col:25} | Status: EMPTY COLUMN          | Method: Skip"
            print(msg)
            log_entries.append(msg)
            continue
            
        # Tentukan rentang operasional (Internal Range)
        # Segala sesuatu setelah last_valid_idx dianggap 'Post-Failure'
        internal_range = df_result.loc[first_valid_idx:last_valid_idx, col]
        nan_count_internal = internal_range.isna().sum()
        
        if nan_count_internal > 0:
            # PCHIP memerlukan minimal 3 titik data untuk bekerja
            if internal_range.notna().sum() > 2:
                # Lakukan interpolasi HANYA pada rentang internal
                df_result.loc[first_valid_idx:last_valid_idx, col] = internal_range.interpolate(method='pchip')
                
                # Fallback ffill/bfill HANYA di dalam rentang internal tersebut
                # (untuk menangani NaN yang mungkin ada di baris pertama atau titik ekstrem lokal)
                df_result.loc[first_valid_idx:last_valid_idx, col] = df_result.loc[first_valid_idx:last_valid_idx, col].ffill().bfill()
                
                msg = f"Sensor {col:25} | Status: Imputed (NaN: {nan_count_internal:>4}) | Method: PCHIP Internal"
            else:
                # Jika data terlalu sedikit, gunakan ffill sederhana (tetap terbatas pada rentang internal)
                df_result.loc[first_valid_idx:last_valid_idx, col] = internal_range.ffill().bfill()
                msg = f"Sensor {col:25} | Status: Fallback (Too few)     | Method: FFILL Internal"
        else:
            msg = f"Sensor {col:25} | Status: OK (No internal NaN)   | Method: -"
        
        print(msg)
        log_entries.append(msg)

    # Logging ke file
    with open(log_file, "a") as f:
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write("\n" + "="*70 + "\n")
        f.write(f"[{timestamp}] Refined Imputation: {getattr(df, 'filename', 'Unknown')}\n")
        f.write("\n".join(log_entries) + "\n")
            
    return df_result

def outlier_cleaner_pchip(df, columns=None, protection_percent=0.20, log_file="outlier_cleaning_log.txt"):
    """
    Membersihkan outlier menggunakan metode IQR pada data awal, 
    melindungi data akhir (fase kegagalan), dan mengisi gap dengan PCHIP.
    """
    df_result = df.copy()
    target_cols = columns if columns else df_result.select_dtypes(include=[np.number]).columns
    
    # Hitung batas indeks perlindungan
    split_index = int(len(df_result) * (1 - protection_percent))
    
    log_entries = []
    
    for col in target_cols:
        # Pisahkan data untuk dibersihkan dan data yang dilindungi
        to_clean = df_result.loc[:split_index, col].copy()
        
        # Hitung IQR hanya pada bagian data yang akan dibersihkan
        Q1 = to_clean.quantile(0.25)
        Q3 = to_clean.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Tandai outlier sebagai NaN
        outliers_mask = (to_clean < lower_bound) | (to_clean > upper_bound)
        outlier_count = outliers_mask.sum()
        
        if outlier_count > 0:
            # Terapkan penghapusan outlier
            df_result.loc[outliers_mask[outliers_mask].index, col] = np.nan
            
            # Tambal menggunakan PCHIP agar transisi halus
            df_result[col] = df_result[col].interpolate(method='pchip')
            
            # Pastikan tidak ada NaN tersisa di ujung data
            df_result[col] = df_result[col].ffill().bfill()
            
            msg = f"Sensor {col:25} | Status: Outlier Removed ({outlier_count:>4}) | Method: IQR + PCHIP"
        else:
            msg = f"Sensor {col:25} | Status: Clean (No Outliers)    | Method: -"
            
        print(msg)
        log_entries.append(msg)

    # Logging ke file
    with open(log_file, "a") as f:
        f.write("\n" + "="*70 + "\n")
        f.write(f"[{pd.Timestamp.now()}] Outlier Cleaning: {getattr(df, 'filename', 'Unknown')}\n")
        f.write("\n".join(log_entries) + "\n")
            
    return df_result

# =================================================================
# 2. MAIN ITERATIVE PROCESSOR
# =================================================================

def iterative_processor(file_list, operation_func, input_dir=None, output_dir=None, **kwargs):
    """
    Fungsi universal untuk mengiterasi operasi pada list file CSV.
    """
    processed_files = []
    
    # Ambil daftar kolom dari kwargs untuk digunakan di dropna.
    # Menggunakan .get() agar tidak error jika 'columns' tidak dikirim.
    target_cols = kwargs.get('columns')
    
    for file_name in file_list:
        input_path = os.path.join(input_dir, file_name) if input_dir else file_name
        
        if not os.path.exists(input_path):
            print(f"File tidak ditemukan: {input_path}")
            continue
            
        print(f"\n--- Processing: {file_name} ---")
        df = pd.read_csv(input_path)
        
        # Tempelkan atribut nama file agar fungsi imputer bisa membaca untuk logging
        df.filename = file_name 
        
        # 1. Jalankan operasi utama (misal: pchip_imputer atau outlier_cleaner)
        df_result = operation_func(df, **kwargs)
        
        # 2. Perbaikan NameError: Gunakan variabel target_cols yang sudah diambil
        # how='all' memastikan kita hanya menghapus baris yang benar-benar kosong di semua sensor
        if target_cols is not None:
            df_result = df_result.dropna(how='all', subset=target_cols)
        
        # 3. Proses penyimpanan
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name)
            df_result.to_csv(output_path, index=False)
            processed_files.append(output_path)
        else:
            processed_files.append(df_result)
            
    return processed_files