import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from model import BiLSTM
# =============================================================
# MODEL LOADER
# =============================================================
def load_model(data_type, variation,
               config_path  = 'From Project\\Training Results\\best_configs.json',
               model_dir    = 'From Project\\Training Results'):
    """
    Load BiLSTM model dan konfigurasinya dari Training Results.

    Args:
        data_type  : 'Clean' atau 'Dirty'
        variation  : 'Combustion', 'Systemic', atau 'Global'
        config_path: path ke best_configs.json
        model_dir  : folder tempat file .pt disimpan

    Returns:
        model       : BiLSTM siap pakai (eval mode)
        target_min  : float, min scaling dari training
        target_max  : float, max scaling dari training
        config      : dict konfigurasi lengkap
    """
    with open(config_path, 'r') as f:
        all_configs = json.load(f)

    config      = all_configs[data_type][variation]
    best_params = config['best_params']
    hidden_dim  = best_params['hidden_dim']

    # Bangun path model dari model_dir (abaikan path di JSON yang pakai backslash Windows)
    model_filename = os.path.basename(config['model_file'])
    model_path     = os.path.join(model_dir, model_filename)

    model = BiLSTM(hidden_dim=hidden_dim)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Target scaling — ambil dari config jika ada, fallback ke 0/1
    target_min = config.get('target_min', 0.0)
    target_max = config.get('target_max', 1.0)

    return model, target_min, target_max, config


# =============================================================
# DATA LOADER
# =============================================================
def load_data(file_path, col_name=None):
    """
    Load data CSV dan kembalikan sebagai numpy array 1D.

    Args:
        file_path: path ke file CSV
        col_name : nama kolom yang dibaca (None = kolom pertama)

    Returns:
        numpy array 1D
    """
    df = pd.read_csv(file_path)
    col = col_name if col_name else df.columns[0]
    return df[col].values.astype(np.float32)


# =============================================================
# KONFIGURASI
# =============================================================
DATA_TYPE  = 'Dirty'
VARIATION  = 'Global'
DATA_FILE  = 'From Project\\Test Data Dirty\\PC1_RTF_G4.csv'

ALARM_VAL    = 1
SHUTDOWN_VAL = 0.75

START_AT = 0.0
SPLIT_AT = 0.5

# =============================================================
# LOAD MODEL & DATA
# =============================================================
model, target_min, target_max, config = load_model(DATA_TYPE, VARIATION)
print(f"Model     : {config['model_file']}")
print(f"Params    : {config['best_params']}")
print(f"Scaling   : min={target_min}, max={target_max}")

window_size    = config['best_params']['window_size']
data           = load_data(DATA_FILE)
normalized_data = data  # data sudah dalam bentuk HI (0-1)

# =============================================================
# PREDIKSI REKURSIF
# =============================================================
start_idx = int(len(normalized_data) * START_AT)
split_idx = int(len(normalized_data) * SPLIT_AT)

input_seq = normalized_data[split_idx - window_size : split_idx]
curr_seq  = torch.tensor(input_seq, dtype=torch.float32).view(1, window_size, 1)

recursive_preds  = []
max_steps        = (len(normalized_data) - split_idx) + 250  # max 100 step setelah data aktual habis

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

plt.axhline(y=ALARM_VAL,    color='orange', linestyle='--', label=f'Alarm ({ALARM_VAL})')
plt.axhline(y=SHUTDOWN_VAL, color='red',    linestyle='--', label=f'Shutdown ({SHUTDOWN_VAL})')
plt.axvline(x=start_idx, color='black', linestyle=':', linewidth=2, label='Input Start')
plt.axvline(x=split_idx, color='black', linestyle=':', linewidth=2, label='Prediction Start')

# Titik potong prediksi vs threshold
idx_alarm    = np.where(preds_actual >= ALARM_VAL)[0]
idx_shutdown = np.where(preds_actual >= SHUTDOWN_VAL)[0]
point_alarm    = (idx_alarm[0]    + split_idx, ALARM_VAL)    if len(idx_alarm)    > 0 else None
point_shutdown = (idx_shutdown[0] + split_idx, SHUTDOWN_VAL) if len(idx_shutdown) > 0 else None

# Titik potong aktual vs threshold
idx_alarm_act    = np.where(normalized_data >= ALARM_VAL)[0]
idx_shutdown_act = np.where(normalized_data >= SHUTDOWN_VAL)[0]
time_alarm_actual    = idx_alarm_act[0]    if len(idx_alarm_act)    > 0 else None
time_shutdown_actual = idx_shutdown_act[0] if len(idx_shutdown_act) > 0 else None

for pt, col in zip([point_alarm, point_shutdown], ['orange', 'red']):
    if pt:
        plt.scatter(pt[0], pt[1], color=col, s=100, edgecolors='black', zorder=5)
        plt.annotate(f'Time: {pt[0]}', xy=pt,
                     xytext=(pt[0] - 60, pt[1] + 0.05),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     fontweight='bold')

# Hitung error
def calc_error(actual, pred):
    if actual is None or pred[0] is None:
        return 'N/A', 'N/A'
    err = abs(actual - pred[0])
    pct = (err / actual) * 100.0
    return err, pct

time_alarm_pred    = point_alarm[0]    if point_alarm    else None
time_shutdown_pred = point_shutdown[0] if point_shutdown else None

al_err,  al_pct  = calc_error(time_alarm_actual,    [time_alarm_pred])
sh_err,  sh_pct  = calc_error(time_shutdown_actual, [time_shutdown_pred])

info_text = (
    f"ACTUAL TIMES:\n"
    f"  Alarm:    {time_alarm_actual} h\n"
    f"  Shutdown: {time_shutdown_actual} h\n\n"
    f"PREDICTED TIMES:\n"
    f"  Alarm:    {time_alarm_pred} h\n"
    f"  Shutdown: {time_shutdown_pred} h\n\n"
    f"ERROR RATE:\n"
    f"  Alarm:    {al_err} h / {f'{al_pct:.2f}%' if al_pct != 'N/A' else 'N/A'}\n"
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