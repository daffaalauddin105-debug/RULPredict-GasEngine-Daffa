import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from model import BiLSTM
import sys

# =============================================================
# MODEL LOADER
# =============================================================
# def load_model(data_type, variation,
#                base_dir='Testing'):
    
#     config_path = os.path.join(base_dir, 'Training Results', 'best_configs.json')
#     model_dir   = os.path.join(base_dir, 'Training Results', data_type)

#     with open(config_path, 'r') as f:
#         all_configs = json.load(f)

#     config = all_configs[data_type][variation]

#     model_filename = os.path.basename(config['model_file'])
#     model_path     = os.path.join(model_dir, model_filename)

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

#     model = BiLSTM(hidden_dim=config['best_params']['hidden_dim'])
#     state_dict = torch.load(model_path, map_location='cpu')
#     model.load_state_dict(state_dict)
#     model.eval()

#     return model, config.get('target_min', 0.0), config.get('target_max', 1.0), config

def load_model(data_type, variation,
               base_dir='Testing',
               custom_model_path=None):

    config_path = os.path.join(base_dir, 'Training Results', 'best_configs.json')

    with open(config_path, 'r') as f:
        all_configs = json.load(f)

    config = all_configs[data_type][variation]

    # =========================================================
    # PILIH MODEL PATH
    # =========================================================
    if custom_model_path is not None:
        model_path = custom_model_path
        print(f"[INFO] Using custom model: {model_path}")
    else:
        model_dir = os.path.join(base_dir, 'Training Results', data_type)
        model_filename = os.path.basename(config['model_file'])
        model_path = os.path.join(model_dir, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    # =========================================================
    # LOAD MODEL
    # =========================================================
    model = BiLSTM(hidden_dim=config['best_params']['hidden_dim'])
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    return model, config.get('target_min', 0.0), config.get('target_max', 1.0), config

# =============================================================
# DATA LOADER
# =============================================================
def load_data(file_path, col_name=None):
    df = pd.read_csv(file_path)
    col = col_name if col_name else df.columns[0]
    return df[col].values.astype(np.float32)


# =============================================================
# KONFIGURASI
# =============================================================
BASE_DIR   = 'Testing'
DATA_TYPE  = 'Dirty'
VARIATION  = 'Combustion'  # Pilihan: 'Combustion', 'Systemic', 'Global'
RTF = '6'  # Run To Failure
TESTING = {'Combustion':'C', 'Systemic':'S', 'Global':'G'}[VARIATION]
if TESTING is None:
    raise ValueError(f"VARIATION tidak dikenali: {VARIATION}")
DATA_FILE  = os.path.join(BASE_DIR, f'Test Data {DATA_TYPE}', f'PC1_RTF_{TESTING}{RTF}.csv')

custom_path = os.path.join(
    BASE_DIR,
    'Training Results',
    DATA_TYPE,
    f'BiLSTM_{VARIATION}_finetuned.pt'
)

START_AT = 0.0
SPLIT_AT = 0.5

# =============================================================
# LOAD MODEL & DATA
# =============================================================
model, target_min, target_max, config = load_model(
    DATA_TYPE,
    VARIATION,
    custom_model_path=custom_path
)
print(f"Model     : {config['model_file']}")
print(f"Params    : {config['best_params']}")
print(f"Scaling   : min={target_min}, max={target_max}")

window_size     = config['best_params']['window_size']
data            = load_data(DATA_FILE)
normalized_data = data  # data sudah dalam bentuk HI (0-1)

# Ambil nilai max dari file testing sebagai threshold shutdown
SHUTDOWN_VAL = np.max(normalized_data) 
print(f"Shutdown Threshold (Max Testing): {SHUTDOWN_VAL}")

# =============================================================
# PREDIKSI REKURSIF
# =============================================================
start_idx = int(len(normalized_data) * START_AT)
split_idx = int(len(normalized_data) * SPLIT_AT)

input_seq = normalized_data[split_idx - window_size : split_idx]
curr_seq  = torch.tensor(input_seq, dtype=torch.float32).view(1, window_size, 1)

recursive_preds  = []
max_steps        = (len(normalized_data) - split_idx) + 500

with torch.no_grad():
    for _ in range(max_steps):
        pred = model(curr_seq)
        val  = pred.item()
        recursive_preds.append(val)

        if val >= SHUTDOWN_VAL:
            break

        new_val  = pred.view(1, 1, 1)
        curr_seq = torch.cat((curr_seq[:, 1:, :], new_val), dim=1)

preds_actual = np.array(recursive_preds) * (target_max - target_min) + target_min

# =============================================================
# VISUALISASI
# =============================================================
plt.figure(figsize=(14, 7))

plt.plot(range(len(normalized_data)), normalized_data,
         label='Actual Data (Reference)', color='gray', alpha=0.3)

plt.plot(range(start_idx, split_idx), normalized_data[start_idx:split_idx],
         label=f'Input Data ({int(START_AT*100)}%-{int(SPLIT_AT*100)}%)',
         color='blue', linewidth=2)

plt.plot(range(split_idx, split_idx + len(preds_actual)), preds_actual,
         label='Recursive Prediction', color='blue', linestyle='--')

plt.axhline(y=SHUTDOWN_VAL, color='red', linestyle='--', label=f'Shutdown ({SHUTDOWN_VAL:.4f})')
plt.axvline(x=start_idx, color='black', linestyle=':', linewidth=2, label='Input Start')
plt.axvline(x=split_idx, color='black', linestyle=':', linewidth=2, label='Prediction Start')

# Titik potong prediksi vs threshold
idx_shutdown = np.where(preds_actual >= SHUTDOWN_VAL)[0]
point_shutdown = (idx_shutdown[0] + split_idx, SHUTDOWN_VAL) if len(idx_shutdown) > 0 else None

# Titik potong aktual vs threshold
idx_shutdown_act = np.where(normalized_data >= SHUTDOWN_VAL)[0]
time_shutdown_actual = idx_shutdown_act[0] if len(idx_shutdown_act) > 0 else None

if point_shutdown:
    plt.scatter(point_shutdown[0], point_shutdown[1], color='red', s=100, edgecolors='black', zorder=5)
    y_offset = 0.1 * (plt.ylim()[1] - plt.ylim()[0])

    plt.annotate(f'Time: {point_shutdown[0]}', xy=point_shutdown,
                xytext=(point_shutdown[0] - 60, point_shutdown[1] - y_offset),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                fontweight='bold')

# Hitung error
def calc_error(actual, pred):
    if actual is None or pred[0] is None:
        return 'N/A', 'N/A'
    err = abs(actual - pred[0])
    pct = (err / actual) * 100.0 if actual != 0 else 0
    return err, pct

time_shutdown_pred = point_shutdown[0] if point_shutdown else None
sh_err, sh_pct  = calc_error(time_shutdown_actual, [time_shutdown_pred])

info_text = (
    f"ACTUAL TIMES:\n"
    f"  Shutdown: {time_shutdown_actual} h\n\n"
    f"PREDICTED TIMES:\n"
    f"  Shutdown: {time_shutdown_pred} h\n\n"
    f"ERROR RATE:\n"
    f"  Shutdown: {sh_err} h / {f'{sh_pct:.2f}%' if sh_pct != 'N/A' else 'N/A'}"
)

plt.text(0.98, 0.05, info_text, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

plt.xlabel("Timestep (Hours)")
plt.ylabel("Health Index")
plt.title(f"Recursive Forecasting — {DATA_TYPE} / {VARIATION} "
          f"(Input: {int(START_AT*100)}%–{int(SPLIT_AT*100)}%)")
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()