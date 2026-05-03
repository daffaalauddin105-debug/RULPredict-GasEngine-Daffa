import optuna
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from dataset import RULDataset
from torch.utils.data import DataLoader
from model_train import BiLSTM


class OptunaTuner:
    """
    Mencari 1 set hyperparameter terbaik untuk semua variasi & data type sekaligus.
    Data Dirty dan Clean tetap terpisah — tidak dicampur.
    Validasi: MSE biasa + penalti flatline dari rollout rekursif.
    """

    def __init__(self, data_splits, variations, device, hyper_space):
        """
        data_splits: {
            'Dirty': (train_files_dirty, val_files_dirty),
            'Clean': (train_files_clean, val_files_clean),
        }
        """
        self.data_splits = data_splits
        self.variations  = variations
        self.device      = device
        self.space       = hyper_space

    # ---------------------------------------------------------------
    # INTERNAL: Standard training loop (MSE)
    # ---------------------------------------------------------------
    def _train_loop(self, model, train_loader, optimizer, epochs):
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"    Epoch {epoch:3d} | MSE = {total_loss / len(train_loader):.6f}")

    # ---------------------------------------------------------------
    # INTERNAL: Rollout rekursif — cek flatline
    # ---------------------------------------------------------------
    def _rollout(self, model, seed_seq, threshold,
                 max_steps=2000, flatline_tol=1e-5, flatline_window=50):
        seq         = seed_seq.clone().detach().to(self.device)
        t_pred      = 0
        history     = []
        is_flatline = False

        with torch.no_grad():
            while t_pred < max_steps:
                out = model(seq.unsqueeze(0)).item()
                history.append(out)

                if out >= threshold:
                    break

                if len(history) >= flatline_window:
                    recent = history[-flatline_window:]
                    if (max(recent) - min(recent)) < flatline_tol:
                        is_flatline = True
                        break

                new_val = torch.tensor([[out]], device=self.device)
                seq     = torch.cat([seq[1:], new_val], dim=0)
                t_pred += 1

        timed_out = (t_pred >= max_steps)
        t_result  = 9999.0 if (timed_out or is_flatline) else t_pred
        return t_result, is_flatline

    # ---------------------------------------------------------------
    # INTERNAL: Evaluasi MSE + flatline untuk satu variasi
    # ---------------------------------------------------------------
    def _evaluate_variation(self, model, window_size, variation, val_files):
        criterion  = nn.MSELoss()
        target_col = f"PC1_{variation}"
        model.eval()
        all_losses    = []
        flatline_hits = 0
        total_checks  = 0

        with torch.no_grad():
            for f in val_files:
                df = pd.read_csv(f)
                if target_col not in df.columns:
                    continue
                series = df[target_col].values.astype(np.float32)
                T      = len(series)

                if T <= window_size + 1:
                    continue

                xs = np.array([series[i: i + window_size] for i in range(T - window_size - 1)])
                ys = np.array([series[i + window_size]    for i in range(T - window_size - 1)])

                if len(xs) == 0:
                    continue

                x_t = torch.tensor(xs, dtype=torch.float32).unsqueeze(-1).to(self.device)
                y_t = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1).to(self.device)

                pred = model(x_t)
                all_losses.append(criterion(pred, y_t).item())

                # Cek flatline di 50% dan 75%
                threshold = series[-1]
                for pct in [0.50, 0.75]:
                    split_idx   = max(int(T * pct), window_size)
                    seed_data   = series[split_idx - window_size: split_idx]
                    seed_tensor = torch.from_numpy(seed_data).unsqueeze(-1)
                    _, is_flat  = self._rollout(model, seed_tensor, threshold)
                    total_checks += 1
                    if is_flat:
                        flatline_hits += 1

        mse = float(np.mean(all_losses)) if all_losses else 9999.0

        if total_checks > 0:
            flat_ratio  = flatline_hits / total_checks
            is_flatline = flat_ratio >= 0.5
            status      = "FLATLINE ✗" if is_flatline else "OK ✓"
            print(f"      [{variation}] flatline {flatline_hits}/{total_checks} → {status} | MSE={mse:.6f}")

            if is_flatline:
                FLATLINE_PENALTY = 1.0
                return FLATLINE_PENALTY + mse

        return mse

    # ---------------------------------------------------------------
    # INTERNAL: Aggregate score semua data type & variasi
    # ---------------------------------------------------------------
    def _evaluate_all(self, hidden_dim, window_size, lr, epochs=15):
        """
        Train & evaluasi tiap kombinasi (data_type x variation) secara terpisah.
        Data Dirty dan Clean tidak dicampur.
        Return rata-rata score keseluruhan.
        """
        scores = []

        for data_type, (train_files, val_files) in self.data_splits.items():
            print(f"\n  >> Data type: [{data_type}]")

            for variation in self.variations:
                print(f"     Variation : [{variation}]")

                train_set    = RULDataset(train_files, variation, window_size)
                train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

                model     = BiLSTM(1, hidden_dim, 1).to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                self._train_loop(model, train_loader, optimizer, epochs)
                score = self._evaluate_variation(model, window_size, variation, val_files)
                scores.append(score)

                del model, train_loader, train_set
                torch.cuda.empty_cache()

        avg_score = float(np.mean(scores))
        print(f"\n  >> Aggregate score (mean all): {avg_score:.6f}")
        return avg_score

    # ---------------------------------------------------------------
    # objective
    # ---------------------------------------------------------------
    def objective(self, trial):
        h_low, h_high, h_step = self.space["hidden_dim"]
        hidden_dim = trial.suggest_int("hidden_dim", h_low, h_high, step=h_step)

        lr_low, lr_high = self.space["lr"]
        lr = trial.suggest_float("lr", lr_low, lr_high, log=True)

        w_low, w_high, w_step = self.space["window_size"]
        window_size = trial.suggest_int("window_size", w_low, w_high, step=w_step)

        score = self._evaluate_all(hidden_dim, window_size, lr, epochs=15)

        print(f"\nTrial {trial.number} | Window={window_size} | Hidden={hidden_dim} | "
              f"lr={lr:.2e} | Score={score:.6f}")

        return score

    # ---------------------------------------------------------------
    # train_best_model — train final model per variasi
    # ---------------------------------------------------------------
    def train_best_model(self, best_params, variation, train_files, save_path=None):
        hidden_dim  = best_params["hidden_dim"]
        lr          = best_params["lr"]
        window_size = best_params["window_size"]

        save_path = save_path or f"BiLSTM_{variation}_best.pt"

        train_set    = RULDataset(train_files, variation, window_size)
        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

        model     = BiLSTM(1, hidden_dim, 1).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self._train_loop(model, train_loader, optimizer, epochs=50)

        torch.save(model.state_dict(), save_path)
        print(f"  Model disimpan: {save_path}")
        return model

    # ---------------------------------------------------------------
    # run
    # ---------------------------------------------------------------
    def run(self, n_trials=30):
        print(f"\n=== Optuna Tuning | variations={self.variations} | "
              f"data_types={list(self.data_splits.keys())} ===")
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        print("\nBest Params:", study.best_params)
        return study