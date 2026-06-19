import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
from model import BiLSTM

# =============================================================
# CONFIG
# =============================================================
BASE_DIR   = 'Testing'
DATA_TYPE  = 'Dirty'
VARIATION  = 'Combustion'

DEVICE = torch.device('cpu')

# =============================================================
# LOAD CONFIG
# =============================================================
config_path = os.path.join(BASE_DIR, 'Training Results', 'best_configs.json')

with open(config_path, 'r') as f:
    all_configs = json.load(f)

config = all_configs[DATA_TYPE][VARIATION]

hidden_dim  = config['best_params']['hidden_dim']
window_size = config['best_params']['window_size']
base_lr     = config['best_params']['lr']

# gunakan LR lebih kecil untuk fine-tuning
lr = base_lr * 0.1

# =============================================================
# LOAD MODEL
# =============================================================
model_path = os.path.join(BASE_DIR, config['model_file'])

model = BiLSTM(hidden_dim=hidden_dim).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.train()

print(f"[INFO] Loaded model from: {model_path}")
print(f"[INFO] Fine-tune LR: {lr}")

# =============================================================
# OPTIONAL: FREEZE LSTM (fine-tune hanya FC layer)
# =============================================================
FREEZE_LSTM = True

if FREEZE_LSTM:
    for name, param in model.named_parameters():
        if "lstm" in name:
            param.requires_grad = False

# =============================================================
# DATA LOADER
# =============================================================
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

def load_data(file_path):
    df = pd.read_csv(file_path)
    data = df[df.columns[0]].values.astype(np.float32)
    return data

# GANTI dengan dataset fine-tuning kamu
DATA_FILE = os.path.join(BASE_DIR, 'Test Data Dirty', 'PC1_RTF_C7.csv')

data = load_data(DATA_FILE)

X, y = create_sequences(data, window_size)

X = torch.tensor(X).unsqueeze(-1).to(DEVICE)
y = torch.tensor(y).unsqueeze(-1).to(DEVICE)

# =============================================================
# TRAIN SETUP
# =============================================================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

EPOCHS = 20
BATCH_SIZE = 16

# =============================================================
# TRAIN LOOP
# =============================================================
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(X.size(0))

    total_loss = 0

    for i in range(0, X.size(0), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]

        batch_X = X[idx]
        batch_y = y[idx]

        optimizer.zero_grad()

        output = model(batch_X)
        loss = criterion(output, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {total_loss:.6f}")

# =============================================================
# SAVE MODEL BARU
# =============================================================
save_path = os.path.join(BASE_DIR, 'Training Results', DATA_TYPE,
                         f'BiLSTM_{VARIATION}_finetuned.pt')

torch.save(model.state_dict(), save_path)

print(f"[INFO] Fine-tuned model saved to: {save_path}")

# =============================================================
# UPDATE CONFIG JSON
# =============================================================
new_entry = {
    "best_params": {
        "hidden_dim": hidden_dim,
        "lr": lr,
        "window_size": window_size,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS
    },
    "finetuned_from": config['model_file'],
    "model_file": f"Training Results\\{DATA_TYPE}\\BiLSTM_{VARIATION}_finetuned.pt"
}

# Simpan sebagai versi baru (tidak overwrite lama)
all_configs[DATA_TYPE][VARIATION + "_finetuned"] = new_entry

with open(config_path, 'w') as f:
    json.dump(all_configs, f, indent=4)

print("[INFO] Config JSON updated (finetuned version added)")