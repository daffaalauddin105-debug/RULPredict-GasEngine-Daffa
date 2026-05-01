# -*- coding: utf-8 -*-
"""
Predictive Maintenance GUI - RUL Monitor
Model: BiLSTM (Clean/Combustion)
"""

import tkinter as tk
from tkinter import filedialog, ttk
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import json
import os
from model import BiLSTM
# ============================================================
# MODEL LOADER (inline, baca dari best_configs.json)
# ============================================================
def load_bilstm_model(data_type, variation,
                      config_path='Training Results/best_configs.json',
                      model_dir='Training Results'):
    with open(config_path, 'r') as f:
        all_configs = json.load(f)

    config      = all_configs[data_type][variation]
    hidden_dim  = config['best_params']['hidden_dim']
    window_size = config['best_params']['window_size']

    model_filename = os.path.basename(config['model_file'])
    model_path     = os.path.join(model_dir, model_filename)

    model = BiLSTM(hidden_dim=hidden_dim)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded: {model_path}")
    print(f"Hidden dim: {hidden_dim}, Window size: {window_size}")

    return model, window_size, config


# ============================================================
# PREPROCESSING & ML PIPELINE
# ============================================================
class MLPipeline:
    def __init__(self, config):
        self.model       = None
        self.window_size = 10
        self.update_config(config)
        self.load_model()

    def update_config(self, config):
        self.alarm_val    = float(config.get('alarm_val',    0.80))
        self.shutdown_val = float(config.get('shutdown_val', 0.85))
        self.data_type    = config.get('data_type',  'Clean')
        self.variation    = config.get('variation',  'Combustion')
        self.config_path  = config.get('config_path', 'Training Results/best_configs.json')
        self.model_dir    = config.get('model_dir',   'Training Results')

    def load_model(self):
        try:
            self.model, self.window_size, _ = load_bilstm_model(
                self.data_type, self.variation,
                self.config_path, self.model_dir
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def run_inference(self, hi_data):
        """
        Input  : hi_data — list/array HI (0-1), sudah dinormalisasi dari sumber.
                 Langsung masuk model tanpa preprocessing tambahan.
        Output : rul, forecast_path, point_alarm, point_shutdown
        """
        if self.model is None or len(hi_data) < self.window_size:
            return 0, [], None, None

        window    = np.array(hi_data[-self.window_size:], dtype=np.float32)
        curr_seq  = torch.tensor(window).view(1, self.window_size, 1)
        split_idx = len(hi_data)

        forecast_path  = []
        point_alarm    = None
        point_shutdown = None
        step = 0

        with torch.no_grad():
            while True:
                pred     = self.model(curr_seq)
                pred_val = pred.item()
                forecast_path.append(pred_val)

                # Deteksi alarm
                if point_alarm is None and pred_val >= self.alarm_val:
                    point_alarm = (split_idx + step, self.alarm_val)

                # Deteksi shutdown → stop
                if pred_val >= self.shutdown_val:
                    point_shutdown = (split_idx + step, self.shutdown_val)
                    break

                # Update sliding window
                new_val  = pred.view(1, 1, 1)
                curr_seq = torch.cat((curr_seq[:, 1:, :], new_val), dim=1)
                step += 1

                # Safety break
                if step > 5000:
                    break

        rul = len(forecast_path)
        return rul, forecast_path, point_alarm, point_shutdown


# ============================================================
# SETTINGS DIALOG
# ============================================================
class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, settings, on_save):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("360x420")
        self.resizable(False, False)
        self.settings = settings
        self.on_save  = on_save

        fields = [
            ("Alarm Threshold (HI):",    "alarm_val"),
            ("Shutdown Threshold (HI):", "shutdown_val"),
            ("Simulation Delay (s):",    "sim_delay"),
            ("Data Type (Clean/Dirty):", "data_type"),
            ("Variation:",               "variation"),
            ("Config Path:",             "config_path"),
            ("Model Dir:",               "model_dir"),
        ]
        self.entries = {}
        for label, key in fields:
            tk.Label(self, text=label, anchor='w').pack(fill='x', padx=20, pady=(8, 0))
            e = tk.Entry(self)
            e.insert(0, str(settings.get(key, '')))
            e.pack(fill='x', padx=20)
            self.entries[key] = e

        tk.Button(self, text="Save & Reload Model", bg="#28a745", fg="white",
                  command=self._save, width=20).pack(pady=16)

    def _save(self):
        try:
            self.settings['alarm_val']    = float(self.entries['alarm_val'].get())
            self.settings['shutdown_val'] = float(self.entries['shutdown_val'].get())
            self.settings['sim_delay']    = float(self.entries['sim_delay'].get())
            self.settings['data_type']    = self.entries['data_type'].get().strip()
            self.settings['variation']    = self.entries['variation'].get().strip()
            self.settings['config_path']  = self.entries['config_path'].get().strip()
            self.settings['model_dir']    = self.entries['model_dir'].get().strip()
            self.on_save()
        except ValueError as e:
            tk.messagebox.showerror("Invalid input", str(e))
        self.destroy()


# ============================================================
# MAIN GUI
# ============================================================
class RULPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictive Maintenance - BiLSTM RUL Monitor")
        self.root.minsize(1200, 750)
        self.root.configure(bg="#f0f0f0")

        self.config_file = "app_config.json"
        self.load_settings()
        self.pipeline = MLPipeline(self.settings)

        self.is_running      = False
        self.last_csv_data   = None
        self.all_rul_history = []

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------
    def load_settings(self):
        default = {
            "alarm_val":    0.80,
            "shutdown_val": 0.85,
            "sim_delay":    0.1,
            "data_type":    "Clean",
            "variation":    "Combustion",
            "config_path":  "Training Results/best_configs.json",
            "model_dir":    "Training Results",
        }
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.settings = {**default, **json.load(f)}
                return
            except Exception:
                pass
        self.settings = default
        self._save_settings()

    def _save_settings(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.settings, f, indent=4)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ── TOP TOOLBAR ────────────────────────────────────────────────
        toolbar = tk.Frame(self.root, bg="#e8e8e8", height=36)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        tk.Button(toolbar, text="⚙ Settings", command=self._open_settings,
                  relief=tk.GROOVE, padx=8).pack(side=tk.LEFT, padx=6, pady=4)

        # Model info label
        self.lbl_model_info = tk.Label(
            toolbar,
            text=f"Model: {self.settings['data_type']} / {self.settings['variation']}",
            bg="#e8e8e8", font=('Helvetica', 9)
        )
        self.lbl_model_info.pack(side=tk.LEFT, padx=12)

        # ── BODY ───────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg="#f0f0f0")
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # ── LEFT: matplotlib canvas ────────────────────────────────────
        left = tk.Frame(body, bg="white", relief=tk.FLAT)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=2)

        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(8, 9),
            gridspec_kw={'height_ratios': [2, 1]}
        )
        self.fig.patch.set_facecolor('white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── RIGHT: info panel ──────────────────────────────────────────
        right = tk.Frame(body, bg="#f0f0f0")
        right.grid(row=0, column=1, sticky="nsew", pady=2)

        # TABLE 1: RUL info
        tbl1_frame = tk.Frame(right, bg="white", relief=tk.GROOVE, bd=1)
        tbl1_frame.pack(fill=tk.X, padx=4, pady=(4, 6))

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Info.Treeview",
                        background="white", foreground="black",
                        rowheight=30, font=('Helvetica', 11))
        style.configure("Info.Treeview.Heading",
                        font=('Helvetica', 11, 'bold'))
        style.map("Info.Treeview", background=[('selected', '#d0e8ff')])

        self.tbl1 = ttk.Treeview(tbl1_frame, style="Info.Treeview",
                                  columns=("metric", "value"),
                                  show="headings", height=3)
        self.tbl1.heading("metric", text="Metric")
        self.tbl1.heading("value",  text="Value")
        self.tbl1.column("metric", width=140, anchor='w')
        self.tbl1.column("value",  width=140, anchor='w')
        self.tbl1.pack(fill=tk.X)
        self.tbl1.insert("", "end", iid="rul",         values=("RUL",         "-- H"))
        self.tbl1.insert("", "end", iid="alarm_at",    values=("Alarm At",    "--"))
        self.tbl1.insert("", "end", iid="shutdown_at", values=("Shutdown At", "--"))

        # TABLE 2: Threshold info
        tbl2_outer = tk.Frame(right, bg="white", relief=tk.GROOVE, bd=1)
        tbl2_outer.pack(fill=tk.X, padx=4, pady=(0, 6))

        hdr = tk.Frame(tbl2_outer, bg="#1a6fa8")
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Threshold Information",
                 bg="#1a6fa8", fg="white",
                 font=('Helvetica', 11, 'bold'), pady=5).pack()

        self.tbl2 = ttk.Treeview(tbl2_outer, style="Info.Treeview",
                                  columns=("label", "val"),
                                  show="headings", height=2)
        self.tbl2.heading("label", text="Type")
        self.tbl2.heading("val",   text="Value (HI)")
        self.tbl2.column("label", width=140, anchor='w')
        self.tbl2.column("val",   width=140, anchor='w')
        self.tbl2.pack(fill=tk.X)
        self._refresh_threshold_table()

        # STATS BOX
        stats_box = tk.Frame(right, bg="white", relief=tk.GROOVE, bd=1)
        stats_box.pack(fill=tk.X, padx=4, pady=(0, 6))
        stats_box.columnconfigure(0, weight=1)
        stats_box.columnconfigure(1, weight=1)

        rul_col = tk.Frame(stats_box, bg="white")
        rul_col.grid(row=0, column=0, padx=10, pady=12)
        self.lbl_rul = tk.Label(rul_col, text="-- H",
                                font=('Helvetica', 26, 'bold'),
                                fg="#FF8C00", bg="white")
        self.lbl_rul.pack()
        tk.Label(rul_col, text="REMAINING USEFUL LIFE",
                 font=('Helvetica', 7, 'bold'), bg="white", fg="#555").pack()

        mean_col = tk.Frame(stats_box, bg="white")
        mean_col.grid(row=0, column=1, padx=10, pady=12)
        self.lbl_mean_rul = tk.Label(mean_col, text="-- H",
                                     font=('Helvetica', 26, 'bold'),
                                     fg="#1a6fa8", bg="white")
        self.lbl_mean_rul.pack()
        tk.Label(mean_col, text="MEAN RUL (10 STEP)",
                 font=('Helvetica', 7, 'bold'), bg="white", fg="#555").pack()

        # BUTTONS
        btn_frame = tk.Frame(right, bg="#f0f0f0")
        btn_frame.pack(fill=tk.X, padx=4, pady=4)

        self.btn_load = tk.Button(
            btn_frame, text="📁  Load CSV Data",
            command=self._mode_load_csv,
            bg="#1a6fa8", fg="white",
            font=('Helvetica', 10, 'bold'),
            relief=tk.FLAT, padx=6, pady=8, cursor="hand2"
        )
        self.btn_load.pack(fill=tk.X, pady=(0, 4))

        self.btn_sim = tk.Button(
            btn_frame, text="▶  Start Simulation",
            command=self._toggle_sim,
            bg="#c0392b", fg="white",
            font=('Helvetica', 10, 'bold'),
            relief=tk.FLAT, padx=6, pady=8, cursor="hand2"
        )
        self.btn_sim.pack(fill=tk.X)

        self._draw_empty_plots()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _refresh_threshold_table(self):
        for row in self.tbl2.get_children():
            self.tbl2.delete(row)
        self.tbl2.insert("", "end", values=("Shutdown", f"{self.pipeline.shutdown_val:.2f}"))
        self.tbl2.insert("", "end", values=("Alarm",    f"{self.pipeline.alarm_val:.2f}"))

    def _draw_empty_plots(self):
        for ax in (self.ax1, self.ax2):
            ax.clear()
            ax.grid(True, linestyle=':', alpha=0.5)
        self.ax1.set_title("Recursive Forecasting", fontweight='bold')
        self.ax1.set_ylabel("Health Index")
        self.ax2.set_title("RUL over Time", fontweight='bold', fontsize=10)
        self.ax2.set_xlabel("Timestep (Hours)")
        self.ax2.set_ylabel("RUL (Hours)")
        self.fig.tight_layout()
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Core plot update
    # ------------------------------------------------------------------
    def update_plot(self, hi_data):
        rul, forecast, point_alarm, point_shutdown = self.pipeline.run_inference(hi_data)

        self.all_rul_history.append(rul)
        # Mean RUL = rata-rata 10 timestep terakhir
        last_10  = self.all_rul_history[-10:]
        mean_rul = int(np.mean(last_10))
        split_idx = len(hi_data)

        # ── AXES 1 ──────────────────────────────────────────────────
        self.ax1.clear()

        if self.last_csv_data is not None:
            self.ax1.plot(self.last_csv_data, color='gray', alpha=0.3,
                          label='Actual Data (Reference)')

        self.ax1.plot(range(split_idx), hi_data, color='blue',
                      linewidth=2, label='Input Data')

        if forecast:
            x_fc = range(split_idx, split_idx + len(forecast))
            self.ax1.plot(x_fc, forecast, color='blue', linestyle='--',
                          label='Recursive Prediction')

            for pt, col in zip([point_alarm, point_shutdown], ['orange', 'red']):
                if pt:
                    self.ax1.scatter(pt[0], pt[1], color=col,
                                     s=80, edgecolors='black', zorder=5)
                    self.ax1.annotate(
                        f'Time: {pt[0]}',
                        xy=pt,
                        xytext=(pt[0] - 80, pt[1] + 0.02),
                        arrowprops=dict(facecolor='black', arrowstyle='->'),
                        fontweight='bold', fontsize=8
                    )

        self.ax1.axhline(y=self.pipeline.alarm_val, color='orange',
                         linestyle='--', alpha=0.8,
                         label=f'Alarm ({self.pipeline.alarm_val:.2f})')
        self.ax1.axhline(y=self.pipeline.shutdown_val, color='red',
                         linestyle='--', alpha=0.8,
                         label=f'Shutdown ({self.pipeline.shutdown_val:.2f})')
        self.ax1.axvline(x=split_idx, color='black', linestyle=':',
                         linewidth=1.5, label='Prediction Start')

        self.ax1.set_title(f"Recursive Forecasting — {self.settings['data_type']} / "
                           f"{self.settings['variation']} (Step: {split_idx})",
                           fontweight='bold')
        self.ax1.set_ylabel("Health Index")
        self.ax1.set_ylim(0, 1.05)
        self.ax1.legend(loc='upper left', fontsize='x-small')
        self.ax1.grid(True, linestyle=':', alpha=0.6)

        alarm_t    = point_alarm[0]    if point_alarm    else "N/A"
        shutdown_t = point_shutdown[0] if point_shutdown else "N/A"
        info_text  = (
            f"PREDICTED TIMES:\n"
            f"  Alarm:    {alarm_t} h\n"
            f"  Shutdown: {shutdown_t} h\n\n"
            f"REMAINING LIFE:\n"
            f"  RUL: {rul} Hours"
        )
        self.ax1.text(0.98, 0.05, info_text,
                      transform=self.ax1.transAxes, fontsize=8,
                      verticalalignment='bottom', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white',
                                alpha=0.85, edgecolor='black'))

        # ── AXES 2 ──────────────────────────────────────────────────
        self.ax2.clear()
        self.ax2.plot(range(len(self.all_rul_history)), self.all_rul_history,
                      color='#1b4332', linewidth=2)
        self.ax2.set_title("RUL over Time", fontweight='bold', fontsize=10)
        self.ax2.set_xlabel("Timestep (Hours)")
        self.ax2.set_ylabel("RUL (Hours)")
        self.ax2.grid(True, linestyle=':', alpha=0.6)

        self.fig.tight_layout()
        self.canvas.draw()

        # ── RIGHT PANEL ──────────────────────────────────────────────
        self.lbl_rul.config(text=f"{rul} H")
        self.lbl_mean_rul.config(text=f"{mean_rul} H")

        self.tbl1.item("rul",         values=("RUL",         f"{rul} H"))
        self.tbl1.item("alarm_at",    values=("Alarm At",    f"Hour {alarm_t}"    if alarm_t    != "N/A" else "--"))
        self.tbl1.item("shutdown_at", values=("Shutdown At", f"Hour {shutdown_t}" if shutdown_t != "N/A" else "--"))

    # ------------------------------------------------------------------
    # Control logic
    # ------------------------------------------------------------------
    def _mode_load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return

        df = pd.read_csv(path)
        # Ambil kolom numerik pertama — handle header ada atau tidak
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            self.last_csv_data = df[num_cols[0]].values.tolist()
        else:
            # Fallback: baca ulang tanpa header, ambil kolom pertama
            df = pd.read_csv(path, header=None)
            self.last_csv_data = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().tolist()
        self.all_rul_history = []

        self.is_running = True
        self.btn_sim.config(text="⏹  Stop Simulation", bg="#c0392b")
        threading.Thread(target=self._run_csv_simulation, daemon=True).start()

    def _run_csv_simulation(self):
        ws    = self.pipeline.window_size
        delay = float(self.settings.get('sim_delay', 0.1))
        for i in range(ws, len(self.last_csv_data) + 1):
            if not self.is_running:
                break
            self.root.after(0, self.update_plot, self.last_csv_data[:i])
            time.sleep(delay)
        self.root.after(0, self._stop_all)

    def _toggle_sim(self):
        if self.is_running:
            self._stop_all()
        else:
            self._mode_load_csv()

    def _stop_all(self):
        self.is_running = False
        try:
            if self.btn_sim.winfo_exists():
                self.btn_sim.config(text="▶  Start Simulation", bg="#28a745")
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _open_settings(self):
        def on_save():
            self._save_settings()
            self.pipeline.update_config(self.settings)
            self.pipeline.load_model()
            self._refresh_threshold_table()
            self.lbl_model_info.config(
                text=f"Model: {self.settings['data_type']} / {self.settings['variation']}"
            )

        SettingsDialog(self.root, self.settings, on_save)

    def _on_close(self):
        self.is_running = False
        self._save_settings()
        self.root.destroy()


# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = RULPredictorGUI(root)
    root.mainloop()