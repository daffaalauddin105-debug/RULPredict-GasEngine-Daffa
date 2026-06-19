import os
import torch
import numpy as np
import pandas as pd
import json
from model import BiLSTM

# =============================================================
# KONFIGURASI
# =============================================================
BASE_DIR   = 'Testing'
DATA_TYPE  = 'Dirty'
VARIATION  = 'Combustion'
RTF        = '6'
MODEL_TYPE = 'best'  # 'finetuned' atau 'best'

TESTING   = {'Combustion': 'C', 'Systemic': 'S', 'Global': 'G'}[VARIATION]
DATA_FILE = os.path.join(BASE_DIR, f'Test Data {DATA_TYPE}', f'PC1_RTF_{TESTING}{RTF}.csv')

config_path = os.path.join(BASE_DIR, 'Training Results', 'best_configs.json')
with open(config_path) as f:
    all_configs = json.load(f)
config      = all_configs[DATA_TYPE][VARIATION]
window_size = config['best_params']['window_size']

model_path = os.path.join(BASE_DIR, 'Training Results', DATA_TYPE, f'BiLSTM_{VARIATION}_{MODEL_TYPE}.pt')

model = BiLSTM(hidden_dim=config['best_params']['hidden_dim'])
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

data         = pd.read_csv(DATA_FILE).iloc[:, 0].values.astype(np.float32)
SHUTDOWN_VAL = float(np.max(data))
T            = len(data)

idx_shutdown_actual = np.where(data >= SHUTDOWN_VAL)[0]
t_shutdown_actual   = idx_shutdown_actual[0] if len(idx_shutdown_actual) > 0 else T - 1

# =============================================================
# HITUNG PREDICTED RUL TIAP TIMESTEP
# =============================================================
MAX_STEPS = 2000
results   = []

for t in range(window_size, T):
    seed     = data[t - window_size: t]
    curr_seq = torch.tensor(seed, dtype=torch.float32).view(1, window_size, 1)
    rul_pred = MAX_STEPS

    with torch.no_grad():
        for step in range(MAX_STEPS):
            pred = model(curr_seq)
            val  = pred.item()

            if val >= SHUTDOWN_VAL:
                rul_pred = step
                break

            new_val  = pred.view(1, 1, 1)
            curr_seq = torch.cat((curr_seq[:, 1:, :], new_val), dim=1)

    rul_actual = max(t_shutdown_actual - t, 0)
    results.append({'timestep': t, 'rul_actual': rul_actual, 'rul_pred': rul_pred})
    print(f"t={t:4d} | RUL actual={rul_actual:5d} | RUL pred={rul_pred:5d}")

# =============================================================
# HITUNG RMSE & MAE
# =============================================================
df            = pd.DataFrame(results)
errors        = df['rul_actual'] - df['rul_pred']
df['error']   = errors
df['mae_contribution']  = np.abs(errors)
df['rmse_contribution'] = errors ** 2

mae  = float(np.mean(df['mae_contribution']))
rmse = float(np.sqrt(np.mean(df['rmse_contribution'])))

print(f"\nMAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

summary_row = pd.DataFrame([{
    'timestep'          : 'SUMMARY',
    'rul_actual'        : '',
    'rul_pred'          : '',
    'error'             : '',
    'mae_contribution'  : mae,
    'rmse_contribution' : rmse,
}])

df_out   = pd.concat([df, summary_row], ignore_index=True)
out_path = os.path.join(BASE_DIR, f'rul_predictions_{DATA_TYPE}_{VARIATION}_RTF{RTF}.csv')
df_out.to_csv(out_path, index=False, float_format='%.6f')
print(f"Disimpan: {out_path}")