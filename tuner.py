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
    val_mode:
        'standard' — validasi hanya pakai MSE loss biasa (cepat).
        'rul'      — validasi pakai MSE + RUL error dari rollout rekursif
                     di 3 titik (25%, 50%, 75%).
    """

    def __init__(self, train_files, val_files, variation, device, hyper_space,
                 val_mode: str = 'standard'):
        assert val_mode in ('standard', 'rul'), "val_mode harus 'standard' atau 'rul'"
        self.train_files = train_files
        self.val_files   = val_files
        self.variation   = variation
        self.device      = device
        self.space       = hyper_space
        self.val_mode    = val_mode

    # ---------------------------------------------------------------
    # INTERNAL: Standard training loop (MSE, next-step)
    # ---------------------------------------------------------------
    def _train_loop(self, model, train_loader, optimizer, epochs):
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                pred = model(x)              # (batch, 1)
                loss = criterion(pred, y)    # y: (batch, 1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d} | MSE = {total_loss / len(train_loader):.6f}")

    # ---------------------------------------------------------------
    # INTERNAL: Validasi mode 'standard' + deteksi flatline
    # ---------------------------------------------------------------
    def _evaluate_standard(self, model, window_size):
        criterion     = nn.MSELoss()
        target_col    = f"PC1_{self.variation}"
        model.eval()
        all_losses    = []
        flatline_hits = 0
        total_checks  = 0

        with torch.no_grad():
            for f in self.val_files:
                df     = pd.read_csv(f)
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

                # Cek flatline dari titik tengah file
                mid_idx     = max(T // 2, window_size)
                seed_data   = series[mid_idx - window_size: mid_idx]
                seed_tensor = torch.from_numpy(seed_data).unsqueeze(-1)
                threshold   = series[-1]
                _, is_flat  = self._rollout(model, seed_tensor, threshold)
                total_checks += 1
                if is_flat:
                    flatline_hits += 1

        mse = float(np.mean(all_losses)) if all_losses else 9999.0

        if total_checks > 0:
            flat_ratio = flatline_hits / total_checks
            is_flatline = flat_ratio >= 0.5
            status      = "FLATLINE ✗" if is_flatline else "OK ✓"
            print(f"    [flatline] {flatline_hits}/{total_checks} files flat → {status}")

            if is_flatline:
                # Penalti: kembalikan score besar agar Optuna menjauhi params ini.
                # Tetap sertakan mse agar gradien Optuna tidak sepenuhnya buta —
                # model yang "hampir tidak flat" masih lebih baik dari yang total flat.
                FLATLINE_PENALTY = 1.0
                penalized = FLATLINE_PENALTY + mse
                print(f"    [flatline] Penalized score: {penalized:.6f}")
                return penalized

        return mse

    # ---------------------------------------------------------------
    # INTERNAL: Rollout autoregressive sampai threshold
    # Mengembalikan (t_pred, is_flatline)
    #   is_flatline = True jika output model berubah < flatline_tol
    #                 selama flatline_window langkah terakhir
    # ---------------------------------------------------------------
    def _rollout(self, model, seed_seq, threshold,
                 max_steps=2000, flatline_tol=1e-5, flatline_window=50):
        seq       = seed_seq.clone().detach().to(self.device)
        t_pred    = 0
        history   = []
        is_flatline = False

        with torch.no_grad():
            while t_pred < max_steps:
                out = model(seq.unsqueeze(0)).item()
                history.append(out)

                if out >= threshold:
                    break

                # Cek flatline: range output dalam window terakhir < tol
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
    # INTERNAL: Validasi mode 'rul'
    # ---------------------------------------------------------------
    def _evaluate_rul(self, model, window_size, rul_weight: float = 0.5):
        mse_score  = self._evaluate_standard(model, window_size)
        target_col = f"PC1_{self.variation}"
        file_errors = []
        model.eval()

        for f in self.val_files:
            df     = pd.read_csv(f)
            series = df[target_col].values.astype(np.float32)
            T_total   = len(series)
            threshold = series[-1]

            if T_total <= window_size + 10:
                continue

            errors = []
            for pct in [0.25, 0.50, 0.75]:
                split_idx   = max(int(T_total * pct), window_size)
                seed_data   = series[split_idx - window_size: split_idx]
                seed_tensor = torch.from_numpy(seed_data).unsqueeze(-1)

                T_true         = T_total - split_idx
                T_pred, is_flat = self._rollout(model, seed_tensor, threshold)
                errors.append(abs(T_pred - T_true))

            if len(errors) == 3:
                weighted_err = 0.2 * errors[0] + 0.3 * errors[1] + 0.5 * errors[2]
                file_errors.append(weighted_err)

        rul_error      = float(np.mean(file_errors)) if file_errors else 9999.0
        rul_normalized = rul_error / 1000.0
        score          = (1 - rul_weight) * mse_score + rul_weight * rul_normalized

        print(f"    [eval] MSE={mse_score:.6f} | RUL_err={rul_error:.1f} | Score={score:.6f}")
        return score

    # ---------------------------------------------------------------
    # INTERNAL: Dispatch evaluasi
    # ---------------------------------------------------------------
    def _evaluate(self, model, window_size):
        if self.val_mode == 'rul':
            return self._evaluate_rul(model, window_size)
        return self._evaluate_standard(model, window_size)

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

        train_set    = RULDataset(self.train_files, self.variation, window_size)
        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

        model     = BiLSTM(1, hidden_dim, 1).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self._train_loop(model, train_loader, optimizer, epochs=15)

        score = self._evaluate(model, window_size)

        print(f"Trial {trial.number} | Window={window_size} | Hidden={hidden_dim} | "
              f"lr={lr:.2e} | Score={score:.6f} | mode={self.val_mode}")

        del model, train_loader, train_set
        torch.cuda.empty_cache()

        return score

    # ---------------------------------------------------------------
    # train_best_model
    # ---------------------------------------------------------------
    def train_best_model(self, best_params, save_path=None):
        hidden_dim  = best_params["hidden_dim"]
        lr          = best_params["lr"]
        window_size = best_params["window_size"]

        save_path = save_path or f"BiLSTM_{self.variation}_best.pt"

        train_set    = RULDataset(self.train_files, self.variation, window_size)
        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

        model     = BiLSTM(1, hidden_dim, 1).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self._train_loop(model, train_loader, optimizer, epochs=50)

        torch.save(model.state_dict(), save_path)
        print(f"\nModel disimpan: {save_path}")
        return model

    # ---------------------------------------------------------------
    # run
    # ---------------------------------------------------------------
    def run(self, n_trials=30):
        print(f"\n=== Optuna Tuning | variation={self.variation} | val_mode={self.val_mode} ===")
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        print("\nBest Params:", study.best_params)
        self.train_best_model(study.best_params)
        return study