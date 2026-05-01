import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class RULDataset(Dataset):
    """
    Dataset standar: x = window, y = nilai berikutnya (next-step).
    y shape: (N, 1) — sesuai training standar MSE.
    """
    def __init__(self, file_list, variation='Global', window_size=15, horizon=1):
        target_column = f"PC1_{variation}"
        xs = []
        ys = []

        for f in file_list:
            df = pd.read_csv(f)

            if target_column not in df.columns:
                raise ValueError(f"{target_column} not found in {f}")

            series = df[target_column].values.astype(np.float32)
            x, y   = self._make_windows(series, window_size)
            xs.append(x)
            ys.append(y)

        # x: (N, window_size, 1) | y: (N, 1)
        self.x = torch.from_numpy(np.concatenate(xs)).float().unsqueeze(-1)
        self.y = torch.from_numpy(np.concatenate(ys)).float().unsqueeze(-1)

    def _make_windows(self, series, window_size):
        n = len(series) - window_size
        x = np.array([series[i:         i + window_size] for i in range(n)])
        y = np.array([series[i + window_size]            for i in range(n)])
        return x, y

    def __len__(self):      return self.x.shape[0]
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


# --- CLASS CKF (Strictly Monotonic) ---
class StrictMonotonicCKF:
    def __init__(self, initial_value, growth_rate=1.01, proc_var=0.001, meas_var=10.0):
        self.x          = initial_value
        self.P          = 1.0
        self.Q          = proc_var
        self.R          = meas_var
        self.growth_rate = growth_rate
        self.prev_x     = initial_value

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
