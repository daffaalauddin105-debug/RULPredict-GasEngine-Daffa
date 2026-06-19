import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dataset import RULDataset
from torch.utils.data import DataLoader
from model_train import BiLSTM

# --- KONFIGURASI (sama dengan main.py) ---

DIRS = {
    'Dirty': 'Training\\PCA Result\\Outliered Data',
    'Clean': 'Training\\PCA Result\\No-Outlier Data',
}
VARIATIONS = ['Combustion', 'Systemic', 'Global']
TRAIN_FILES = [
    'PCA_Results_run_to_failure1.csv',
    'PCA_Results_run_to_failure3.csv',
    'PCA_Results_run_to_failure5.csv',
]
VAL_FILES = ['PCA_Results_run_to_failure2.csv']

OUTPUT_DIR   = 'Training\\Training Results'
CONFIG_PATH  = os.path.join(OUTPUT_DIR, 'best_configs.json')

print("Imports OK")
print("Config path:", CONFIG_PATH)
print("Config exists:", os.path.exists(CONFIG_PATH))

device = torch.device('cpu')
criterion = nn.MSELoss()

# --- LOAD CONFIG ---
with open(CONFIG_PATH) as f:
    configs = json.load(f)

# --- HELPER: hitung MSE dari DataLoader ---
def compute_loss(model, files, variation, window_size):
    dataset    = RULDataset(files, variation, window_size)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            losses.append(criterion(pred, y).item())
    return float(np.mean(losses)) if losses else float('nan')

# --- MAIN ---
results = []

for data_type, data_dir in DIRS.items():
    train_files = [os.path.join(data_dir, f) for f in TRAIN_FILES]
    val_files   = [os.path.join(data_dir, f) for f in VAL_FILES]

    for variation in VARIATIONS:
        info        = configs[data_type][variation]
        params      = info['best_params']
        hidden_dim  = params['hidden_dim']
        window_size = params['window_size']
        model_path = os.path.join('Training', info['model_file'])

        # Load model
        model = BiLSTM(1, hidden_dim, 1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        train_loss = compute_loss(model, train_files, variation, window_size)
        val_loss   = compute_loss(model, val_files,   variation, window_size)

        print(f"[{data_type}][{variation}] Train MSE={train_loss:.6f} | Val MSE={val_loss:.6f}")
        results.append({
            'data_type'  : data_type,
            'variation'  : variation,
            'train_mse'  : train_loss,
            'val_mse'    : val_loss,
        })

# --- SIMPAN ---
df = pd.DataFrame(results)
out_path = os.path.join(OUTPUT_DIR, 'loss_summary.csv')
df.to_csv(out_path, index=False, float_format='%.8f')
print(f"\nDisimpan: {out_path}")
print(df.to_string(index=False))