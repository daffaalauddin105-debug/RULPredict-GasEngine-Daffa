# -*- coding: utf-8 -*-
"""
Test snippet: preprocessing pipeline
StandardScaler → PCA → CKF → MinMax

Ganti INPUT_CSV dan REFERENCE_CSV sesuai path lokal.
"""

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
INPUT_CSV     = "Testing\\run_to_failure4.csv"   # raw sensor data
REFERENCE_CSV = "Testing\\PCA_Results_run_to_failure4.csv"                      # hasil preprocessing asli (opsional, bisa kosong)

EXH_COLS = [f'Exh Cyl #{i} Temp' for i in range(1, 6)]  # Cyl #1–5

# ── Tabel 4.4: StandardScaler (Combustion +Outlier) ─────────
SS_MEANS = np.array([1159.8593, 1170.6201, 1147.2312, 1177.0129, 1188.4866])
SS_STDS  = np.sqrt([722.5864, 818.3638, 799.4549, 940.6627, 854.7322])

# ── Tabel 4.6: PCA PC1 loadings (Combustion +Outlier) ───────
PCA_LOADINGS = np.array([0.455067, 0.448755, 0.447310, 0.436433, 0.448300])

# ── Tabel 4.7: MinMax (Combustion +Outlier) ─────────────────
MM_MIN, MM_MAX = -4.8861, 2.1413

# ── CKF params (sama dengan dataset.py) ─────────────────────
CKF_MEAS_VAR    = 15.0
CKF_PROC_VAR    = 0.001
CKF_GROWTH_RATE = 1.01


# ============================================================
# CKF
# ============================================================
class StrictMonotonicCKF:
    def __init__(self, initial_value, growth_rate=1.01, proc_var=0.001, meas_var=15.0):
        self.x           = initial_value
        self.P           = 1.0
        self.Q           = proc_var
        self.R           = meas_var
        self.growth_rate = growth_rate
        self.prev_x      = initial_value

    def process(self, measurement):
        x_pred = self.growth_rate * self.x
        P_pred = (self.growth_rate ** 2) * self.P + self.Q
        K      = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        if self.x < self.prev_x:
            self.x = self.prev_x
        self.prev_x = self.x
        return self.x


# ============================================================
# PIPELINE
# ============================================================
df  = pd.read_csv(INPUT_CSV)
X   = df[EXH_COLS].values.astype(np.float64)

# Step 1: StandardScaler
X_scaled = (X - SS_MEANS) / SS_STDS

# Step 2: PCA → PC1 raw
pc1_raw = X_scaled @ PCA_LOADINGS

# Step 3: CKF
ckf          = StrictMonotonicCKF(pc1_raw[0], CKF_GROWTH_RATE, CKF_PROC_VAR, CKF_MEAS_VAR)
pc1_filtered = np.array([ckf.process(v) for v in pc1_raw])

# Step 4: MinMax
hi = (pc1_filtered - MM_MIN) / (MM_MAX - MM_MIN)

# ============================================================
# OUTPUT
# ============================================================
print("=== Hasil Pipeline ===")
print(f"PC1 raw  — min: {pc1_raw.min():.4f}, max: {pc1_raw.max():.4f}")
print(f"PC1 CKF  — min: {pc1_filtered.min():.4f}, max: {pc1_filtered.max():.4f}")
print(f"HI       — min: {hi.min():.4f}, max: {hi.max():.4f}")
print(f"\nHI[:5]  = {hi[:5].round(4)}")
print(f"HI[-5:] = {hi[-5:].round(4)}")
print(f"Monotonic: {all(hi[i] >= hi[i-1] for i in range(1, len(hi)))}")

# ============================================================
# PERBANDINGAN (opsional)
# ============================================================
if REFERENCE_CSV:
    ref = pd.read_csv(REFERENCE_CSV)

    # Sesuaikan nama kolom referensi jika perlu
    REF_COL_RAW = "PC1_Raw_Combustion"
    REF_COL_CKF = "PC1_Combustion"

    if REF_COL_RAW in ref.columns:
        diff_raw = np.abs(pc1_raw - ref[REF_COL_RAW].values)
        print(f"\n=== Diff vs Referensi: PC1 Raw ===")
        print(f"  Max diff : {diff_raw.max():.6f}")
        print(f"  Mean diff: {diff_raw.mean():.6f}")

    # PC1_Combustion referensi = CKF + MinMax (HI final)
    # Bandingkan langsung dengan hi kita
    if REF_COL_CKF in ref.columns:
        hi_ref  = ref[REF_COL_CKF].values
        diff_hi = np.abs(hi - hi_ref)
        print(f"\n=== Diff vs Referensi: HI (CKF + MinMax) ===")
        print(f"  Ref HI   — min: {hi_ref.min():.4f}, max: {hi_ref.max():.4f}")
        print(f"  Our HI   — min: {hi.min():.4f},     max: {hi.max():.4f}")
        print(f"  Max diff : {diff_hi.max():.6f}")
        print(f"  Mean diff: {diff_hi.mean():.6f}")
        print(f"\n  Diff per-step (10 pertama):")
        for i in range(min(10, len(diff_hi))):
            print(f"    [{i:3d}] ref={hi_ref[i]:.4f}  ours={hi[i]:.4f}  diff={diff_hi[i]:.6f}")