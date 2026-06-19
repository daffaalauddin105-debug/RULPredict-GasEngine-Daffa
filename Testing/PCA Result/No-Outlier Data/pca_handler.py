import pandas as pd
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
INPUT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(INPUT_DIR, "extracted")
NUM_RUNS   = 5

# Mapping: output filename (without .csv) → column name in each run file
VARIATIONS = {
    "Combustion_Raw": "PC1_Raw_Combustion",
    "Combustion":     "PC1_Combustion",
    "Systemic_Raw":   "PC1_Raw_Systemic",
    "Systemic":       "PC1_Systemic",
    "Global_Raw":     "PC1_Raw_Global",
    "Global":         "PC1_Global",
}
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load all 5 run files
runs = {}
for i in range(1, NUM_RUNS + 1):
    filename = f"PCA_Results_run_to_failure{i}.csv"
    filepath = os.path.join(INPUT_DIR, filename)
    runs[i] = pd.read_csv(filepath)
    print(f"Loaded {filename}  ({len(runs[i])} rows)")

# Extract and save each variation
for out_name, col in VARIATIONS.items():
    # Build list of Series (each may have different length)
    series_list = [runs[i][col].rename(f"RTF{i}") for i in range(1, NUM_RUNS + 1)]

    # Check for unequal lengths and warn
    lengths = [len(s) for s in series_list]
    if len(set(lengths)) > 1:
        print(f"  ⚠  Unequal row counts for '{out_name}': {dict(zip([f'RTF{i}' for i in range(1,NUM_RUNS+1)], lengths))}")
        print(f"     Shorter runs will be padded with NaN up to max length ({max(lengths)} rows)")

    # Concat along axis=1 — pandas auto-pads shorter columns with NaN
    combined = pd.concat(series_list, axis=1)

    out_path = os.path.join(OUTPUT_DIR, f"{out_name}.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved  → {out_path}  (shape: {combined.shape})")

print("\nDone! 6 files created in:", os.path.abspath(OUTPUT_DIR))