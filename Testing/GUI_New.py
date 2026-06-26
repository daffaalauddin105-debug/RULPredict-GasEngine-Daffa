# -*- coding: utf-8 -*-
"""RUL Monitor — a small predictive-maintenance demo.

It turns raw exhaust-temperature readings into a Health Index (PCA + Cubature
Kalman Filter), then lets a trained BiLSTM forecast forward until the engine
crosses the shutdown threshold. Everything runs on offline CSV data and is
meant for the thesis defense demo, not production monitoring.
"""

import os
import json
import time
import threading
from collections import namedtuple

import numpy as np
import pandas as pd
import torch

import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from model import BiLSTM


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The five exhaust-cylinder temperature channels we read from the CSV.
EXH_COLS = [
    'Exh Cyl #1 Temp', 'Exh Cyl #2 Temp', 'Exh Cyl #3 Temp',
    'Exh Cyl #4 Temp', 'Exh Cyl #5 Temp',
]

# Fixed preprocessing parameters taken straight from the thesis
# (Combustion, +Outlier condition, RTF 1–5). These are intentionally frozen so
# the GUI reproduces the exact same Health Index as the training pipeline.
SS_MEANS = np.array([1159.8593, 1170.6201, 1147.2312, 1177.0129, 1188.4866])   # Tabel 4.4
SS_STDS  = np.sqrt([722.5864, 818.3638, 799.4549, 940.6627, 854.7322])          # Tabel 4.4
PCA_LOADINGS = np.array([0.455067, 0.448755, 0.447310, 0.436433, 0.448300])     # Tabel 4.6 (PC1)
MM_MIN, MM_MAX = -4.8861, 2.1413                                                # Tabel 4.7

CONFIG_FILE = "app_config.json"

DEFAULT_SETTINGS = {
    "shutdown_val":       0.85,
    "sim_delay":          0.1,
    "data_type":          "Clean",
    "variation":          "Combustion",
    "model_path":         os.path.join(BASE_DIR, "Training Results", "model.pt"),
    "hidden_dim":         64,
    "window_size":        10,
    "max_forecast_steps": 2000,
    "ckf_meas_var":       15.0,
    "ckf_proc_var":       0.001,
    "ckf_growth_rate":    1.01,
    "app_name":           "RUL Monitor",
    "app_subtitle":       "Predictive Maintenance — BiLSTM",
    "author_name":        "Name",
    "author_nrp":         "NRP",
    "author_prodi":       "Prodi x",
    "author_pembimbing":  "Dr.",
    "author_institusi":   "Institusi",
}

# What run_inference hands back. reached_shutdown tells the UI whether `rul` is a
# real time-to-failure or just the length of the horizon we managed to forecast.
Forecast = namedtuple("Forecast", "rul path shutdown_point reached_shutdown")


class StrictMonotonicCKF:
    """Cubature Kalman filter with a hard monotonic constraint.

    Same implementation as dataset.py — the filtered value is never allowed to
    drop below its previous value, which keeps the Health Index monotonically
    increasing toward end-of-life.
    """

    def __init__(self, initial_value, growth_rate=1.01, proc_var=0.001, meas_var=15.0):
        self.x = initial_value
        self.P = 1.0
        self.Q = proc_var
        self.R = meas_var
        self.growth_rate = growth_rate
        self.prev_x = initial_value

    def process(self, measurement):
        x_pred = self.growth_rate * self.x
        P_pred = (self.growth_rate ** 2) * self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        if self.x < self.prev_x:
            self.x = self.prev_x
        self.prev_x = self.x
        return self.x


class Preprocessor:
    """Raw exhaust temperatures -> Health Index.

    Steps: standardize -> project onto PC1 -> CKF smoothing -> min-max scale.
    Only the CKF parameters are configurable (via Settings); everything else is
    frozen to the thesis values above.
    """

    def __init__(self, ckf_meas_var=15.0, ckf_proc_var=0.001, ckf_growth_rate=1.01):
        self.ckf_meas_var = ckf_meas_var
        self.ckf_proc_var = ckf_proc_var
        self.ckf_growth_rate = ckf_growth_rate

    def run(self, df):
        """Return a list of HI values, one per row of `df`."""
        missing = [c for c in EXH_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Kolom tidak ditemukan: {missing}")

        X = df[EXH_COLS].values.astype(np.float64)

        X_scaled = (X - SS_MEANS) / SS_STDS
        pc1_raw = X_scaled @ PCA_LOADINGS

        ckf = StrictMonotonicCKF(
            initial_value=pc1_raw[0],
            growth_rate=self.ckf_growth_rate,
            proc_var=self.ckf_proc_var,
            meas_var=self.ckf_meas_var,
        )
        pc1_filtered = np.array([ckf.process(v) for v in pc1_raw])

        hi = (pc1_filtered - MM_MIN) / (MM_MAX - MM_MIN)
        return hi.tolist()


def load_bilstm_model(model_path, hidden_dim, window_size):
    """Load a BiLSTM state dict from a .pt file. Hidden dim is supplied manually
    because we no longer keep a side-car config JSON."""
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")

    model = BiLSTM(hidden_dim=int(hidden_dim))
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded: {model_path}")
    print(f"Hidden dim: {hidden_dim}, Window size: {window_size}")
    return model, int(window_size)


class MLPipeline:
    """Wraps the loaded model and the recursive forecasting loop."""

    # A forecast that stops climbing is treated as a plateau: if the prediction
    # barely moves for this many steps, we give up rather than spin to the cap.
    PLATEAU_EPS = 1e-5
    PLATEAU_PATIENCE = 15

    def __init__(self, config):
        self.model = None
        self.window_size = 10
        self.update_config(config)
        self.load_model()

    def update_config(self, config):
        self.shutdown_val = float(config.get('shutdown_val', 0.85))
        self.data_type = config.get('data_type', 'Clean')
        self.variation = config.get('variation', 'Combustion')
        self.model_path = config.get('model_path', '')
        self.hidden_dim = int(config.get('hidden_dim', 64))
        self.window_size = int(config.get('window_size', 10))
        self.max_forecast_steps = int(config.get('max_forecast_steps', 2000))

    def load_model(self):
        try:
            self.model, self.window_size = load_bilstm_model(
                self.model_path, self.hidden_dim, self.window_size
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def run_inference(self, hi_data):
        """Forecast forward from the last window until shutdown, plateau, or cap.

        Returns a Forecast. If shutdown is reached, `rul` is the steps until it;
        otherwise `rul` is just how far we forecast before stopping.
        """
        if self.model is None or len(hi_data) < self.window_size:
            return Forecast(0, [], None, False)

        window = np.array(hi_data[-self.window_size:], dtype=np.float32)
        curr_seq = torch.tensor(window).view(1, self.window_size, 1)
        split_idx = len(hi_data)

        forecast_path = []
        point_shutdown = None
        step = 0
        prev_val = None
        plateau_count = 0

        with torch.no_grad():
            while True:
                pred = self.model(curr_seq)
                pred_val = pred.item()

                # A NaN/inf would never satisfy the shutdown test and would spin
                # the loop to the cap, so bail out immediately.
                if not np.isfinite(pred_val):
                    break

                forecast_path.append(pred_val)

                if pred_val >= self.shutdown_val:
                    point_shutdown = (split_idx + step, self.shutdown_val)
                    break

                # Prediction stopped rising -> it will never reach shutdown.
                if prev_val is not None and abs(pred_val - prev_val) < self.PLATEAU_EPS:
                    plateau_count += 1
                    if plateau_count >= self.PLATEAU_PATIENCE:
                        break
                else:
                    plateau_count = 0
                prev_val = pred_val

                new_val = pred.view(1, 1, 1)
                curr_seq = torch.cat((curr_seq[:, 1:, :], new_val), dim=1)
                step += 1
                if step >= self.max_forecast_steps:
                    break

        return Forecast(len(forecast_path), forecast_path,
                        point_shutdown, point_shutdown is not None)


class SettingsDialog(tk.Toplevel):
    """Scrollable settings window. Fields are declared once and reused for both
    building the form and saving it back."""

    # (label, settings key, kind). kind drives both the widget and the parser.
    FIELDS = [
        ("Model File (.pt):",        "model_path",         "file"),
        ("Hidden Dim:",              "hidden_dim",         "int"),
        ("Window Size:",             "window_size",        "int"),
        ("Max Forecast Steps:",      "max_forecast_steps", "int"),
        ("Shutdown Threshold (HI):", "shutdown_val",       "float"),
        ("Simulation Delay (s):",    "sim_delay",          "float"),
        ("Data Type (label):",       "data_type",          "str"),
        ("Variation (label):",       "variation",          "str"),
        ("CKF Meas Var (R):",        "ckf_meas_var",       "float"),
        ("CKF Proc Var (Q):",        "ckf_proc_var",       "float"),
        ("CKF Growth Rate:",         "ckf_growth_rate",    "float"),
        ("App Name:",                "app_name",           "str"),
        ("App Subtitle:",            "app_subtitle",       "str"),
        ("Nama:",                    "author_name",        "str"),
        ("NRP:",                     "author_nrp",         "str"),
        ("Prodi:",                   "author_prodi",       "str"),
        ("Pembimbing:",              "author_pembimbing",  "str"),
        ("Institusi:",               "author_institusi",   "str"),
    ]
    PARSERS = {"int": int, "float": float, "str": str.strip, "file": str.strip}

    def __init__(self, parent, settings, on_save):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("440x520")
        self.minsize(400, 300)
        self.settings = settings
        self.on_save = on_save
        self.entries = {}

        body = self._build_scroll_area()

        for i, (label, key, kind) in enumerate(self.FIELDS):
            tk.Label(body, text=label, anchor='w').pack(
                fill='x', padx=20, pady=(12 if i == 0 else 8, 0))

            if kind == "file":
                row = tk.Frame(body)
                row.pack(fill='x', padx=20)
                entry = tk.Entry(row)
                entry.pack(side='left', fill='x', expand=True)
                tk.Button(row, text="Browse…",
                          command=self._browse_model).pack(side='left', padx=(6, 0))
                self.model_entry = entry
            else:
                entry = tk.Entry(body)
                entry.pack(fill='x', padx=20)

            entry.insert(0, str(settings.get(key, '')))
            self.entries[key] = entry

        tk.Button(body, text="Save & Reload Model", bg="#28a745", fg="white",
                  command=self._save, width=20).pack(pady=16)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_scroll_area(self):
        """Set up the canvas + scrollbar and return the inner frame to fill."""
        outer = tk.Frame(self)
        outer.pack(fill='both', expand=True)

        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        inner = tk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
                    lambda e: canvas.itemconfig(inner_id, width=e.width))

        canvas.bind_all('<MouseWheel>',
                        lambda e: canvas.yview_scroll(int(-e.delta / 120), 'units'))
        canvas.bind_all('<Button-4>', lambda e: canvas.yview_scroll(-1, 'units'))
        canvas.bind_all('<Button-5>', lambda e: canvas.yview_scroll(1, 'units'))

        self._canvas = canvas
        return inner

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select model (.pt)",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, path)

    def _save(self):
        try:
            for _, key, kind in self.FIELDS:
                self.settings[key] = self.PARSERS[kind](self.entries[key].get())
            self.on_save()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        self._on_close()

    def _on_close(self):
        # The mousewheel bindings are global (bind_all), so drop them or they'd
        # keep scrolling the main window after this dialog is gone.
        for seq in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            self._canvas.unbind_all(seq)
        self.destroy()


class RULPredictorGUI:
    SENSOR_COLORS = ['#e63946', '#f4a261', '#2a9d8f', '#264653', '#8d5524']

    def __init__(self, root):
        self.root = root
        self.root.title("Predictive Maintenance - BiLSTM RUL Monitor")
        self.root.minsize(1200, 750)
        self.root.configure(bg="#f0f0f0")

        self.settings = self._load_settings()
        self.pipeline = MLPipeline(self.settings)
        self.preprocessor = self._make_preprocessor()

        self.is_running = False
        self.last_csv_data = None      # HI series after preprocessing
        self.last_exh_data = None      # raw exhaust temps (N x 5) for the sensor strip
        self.all_rul_history = []
        self._sim_thread = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # --- settings ---------------------------------------------------------

    def _load_settings(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return {**DEFAULT_SETTINGS, **json.load(f)}
            except Exception:
                pass
        settings = dict(DEFAULT_SETTINGS)
        self._save_settings(settings)
        return settings

    def _save_settings(self, settings=None):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(settings or self.settings, f, indent=4)

    def _make_preprocessor(self):
        return Preprocessor(
            ckf_meas_var=self.settings.get('ckf_meas_var', 15.0),
            ckf_proc_var=self.settings.get('ckf_proc_var', 0.001),
            ckf_growth_rate=self.settings.get('ckf_growth_rate', 1.01),
        )

    # --- UI construction --------------------------------------------------

    def _build_ui(self):
        self._build_toolbar()
        self._build_sensor_strip()

        body = tk.Frame(self.root, bg="#f0f0f0")
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=1, minsize=360)
        body.rowconfigure(0, weight=1)

        self._build_main_plots(body)
        self._build_side_panel(body)

        self._draw_empty_plots()

    def _build_toolbar(self):
        toolbar = tk.Frame(self.root, bg="#e8e8e8", height=36)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        tk.Button(toolbar, text="⚙ Settings", command=self._open_settings,
                  relief=tk.GROOVE, padx=8).pack(side=tk.LEFT, padx=6, pady=4)
        self.lbl_model_info = tk.Label(
            toolbar, bg="#e8e8e8", font=('Helvetica', 9),
            text=f"Model: {self.settings['data_type']} / {self.settings['variation']}",
        )
        self.lbl_model_info.pack(side=tk.LEFT, padx=12)

    def _build_sensor_strip(self):
        """The row of five small per-cylinder plots along the bottom."""
        strip = tk.Frame(self.root, bg="white", relief=tk.GROOVE, bd=1)
        strip.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 6))

        self.fig_sensor = Figure(figsize=(12, 1.8))
        self.sensor_axes = self.fig_sensor.subplots(1, 5)
        self.fig_sensor.patch.set_facecolor('white')
        self.canvas_sensor = FigureCanvasTkAgg(self.fig_sensor, master=strip)
        self.canvas_sensor.get_tk_widget().pack(fill=tk.X, expand=False)

    def _build_main_plots(self, parent):
        left = tk.Frame(parent, bg="white", relief=tk.FLAT)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=2)

        self.fig = Figure(figsize=(8, 9))
        self.ax1, self.ax2 = self.fig.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
        self.fig.patch.set_facecolor('white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_side_panel(self, parent):
        right = tk.Frame(parent, bg="#f0f0f0")
        right.grid(row=0, column=1, sticky="nsew", pady=2)

        # Buttons are packed first and pinned to the bottom so they keep their
        # space even when the panel above gets crowded.
        self._build_buttons(right)

        tbl1_frame = tk.Frame(right, bg="white", relief=tk.GROOVE, bd=1)
        tbl1_frame.pack(fill=tk.X, padx=4, pady=(4, 6))

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Info.Treeview", background="white", foreground="black",
                        rowheight=30, font=('Helvetica', 11))
        style.configure("Info.Treeview.Heading", font=('Helvetica', 11, 'bold'))
        style.map("Info.Treeview", background=[('selected', '#d0e8ff')])

        self.tbl1 = ttk.Treeview(tbl1_frame, style="Info.Treeview",
                                 columns=("metric", "value"), show="headings", height=2)
        self.tbl1.heading("metric", text="Metric")
        self.tbl1.heading("value", text="Value")
        self.tbl1.column("metric", width=160, anchor='w')
        self.tbl1.column("value", width=180, anchor='w')
        self.tbl1.pack(fill=tk.X)
        self.tbl1.insert("", "end", iid="rul", values=("RUL", "-- H"))
        self.tbl1.insert("", "end", iid="shutdown_at", values=("Shutdown At", "--"))

        self._build_threshold_table(right)
        self._build_stats_box(right)
        self._build_info_card(right, before=tbl1_frame)

    def _build_buttons(self, parent):
        frame = tk.Frame(parent, bg="#f0f0f0")
        frame.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=4)

        self.btn_load = tk.Button(
            frame, text="📁  Load CSV Data", command=self._load_csv,
            bg="#1a6fa8", fg="white", font=('Helvetica', 10, 'bold'),
            relief=tk.FLAT, padx=6, pady=8, cursor="hand2",
        )
        self.btn_load.pack(fill=tk.X, pady=(0, 4))

        self.btn_sim = tk.Button(
            frame, text="▶  Start Simulation", command=self._toggle_sim,
            bg="#c0392b", fg="white", font=('Helvetica', 10, 'bold'),
            relief=tk.FLAT, padx=6, pady=8, cursor="hand2",
        )
        self.btn_sim.pack(fill=tk.X)

    def _build_threshold_table(self, parent):
        outer = tk.Frame(parent, bg="white", relief=tk.GROOVE, bd=1)
        outer.pack(fill=tk.X, padx=4, pady=(0, 6))

        hdr = tk.Frame(outer, bg="#1a6fa8")
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Threshold Information", bg="#1a6fa8", fg="white",
                 font=('Helvetica', 11, 'bold'), pady=5).pack()

        self.tbl2 = ttk.Treeview(outer, style="Info.Treeview",
                                 columns=("label", "val"), show="headings", height=1)
        self.tbl2.heading("label", text="Type")
        self.tbl2.heading("val", text="Value (HI)")
        self.tbl2.column("label", width=160, anchor='w')
        self.tbl2.column("val", width=180, anchor='w')
        self.tbl2.pack(fill=tk.X)
        self._refresh_threshold_table()

    def _build_stats_box(self, parent):
        box = tk.Frame(parent, bg="white", relief=tk.GROOVE, bd=1)
        box.pack(fill=tk.X, padx=4, pady=(0, 6))
        box.columnconfigure(0, weight=1)
        box.columnconfigure(1, weight=1)

        self.lbl_rul = self._stat_column(box, 0, "#FF8C00", "REMAINING USEFUL LIFE")
        self.lbl_mean_rul = self._stat_column(box, 1, "#1a6fa8", "MEAN RUL (10 STEP)")

    def _stat_column(self, parent, col, color, caption):
        wrap = tk.Frame(parent, bg="white")
        wrap.grid(row=0, column=col, padx=10, pady=12)
        value = tk.Label(wrap, text="-- H", font=('Helvetica', 26, 'bold'),
                         fg=color, bg="white")
        value.pack()
        tk.Label(wrap, text=caption, font=('Helvetica', 7, 'bold'),
                 bg="white", fg="#555").pack()
        return value

    def _build_info_card(self, parent, before):
        """Application / author card, pinned to the very top of the panel."""
        outer = tk.Frame(parent, bg="white", relief=tk.GROOVE, bd=1)
        outer.pack(fill=tk.X, padx=4, pady=(4, 6), before=before)

        hdr = tk.Frame(outer, bg="#1a6fa8")
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="Application Info", bg="#1a6fa8", fg="white",
                 font=('Helvetica', 11, 'bold'), pady=5).pack()

        body = tk.Frame(outer, bg="white")
        body.pack(fill=tk.X, padx=12, pady=10)

        tk.Label(body, text=self.settings.get('app_name', 'RUL Monitor'),
                 bg="white", fg="#1a6fa8", font=('Helvetica', 14, 'bold')).pack(anchor='w')
        tk.Label(body, text=self.settings.get('app_subtitle', ''),
                 bg="white", fg="#555", font=('Helvetica', 8)).pack(anchor='w', pady=(0, 8))
        ttk.Separator(body, orient='horizontal').pack(fill='x', pady=(0, 8))

        self.lbl_credit_widgets = {}
        rows = [
            ("Nama", 'author_name'), ("NRP", 'author_nrp'),
            ("Prodi", 'author_prodi'), ("Pembimbing", 'author_pembimbing'),
            ("Institusi", 'author_institusi'),
        ]
        for label, key in rows:
            row = tk.Frame(body, bg="white")
            row.pack(fill='x', pady=1)
            tk.Label(row, text=label, bg="white", fg="#888", font=('Helvetica', 8),
                     width=11, anchor='nw').pack(side='left', anchor='n')
            val = tk.Label(row, text=self.settings.get(key, '-'), bg="white", fg="#222",
                           font=('Helvetica', 9, 'bold'), anchor='w', justify='left',
                           wraplength=240)
            val.pack(side='left', fill='x', expand=True)
            self.lbl_credit_widgets[key] = val

        # Let the values wrap to whatever width the panel currently has.
        def fit_wrap(event):
            width = max(120, event.width - 100)
            for lbl in self.lbl_credit_widgets.values():
                lbl.config(wraplength=width)
        body.bind('<Configure>', fit_wrap)

    # --- plotting helpers -------------------------------------------------

    def _refresh_threshold_table(self):
        for row in self.tbl2.get_children():
            self.tbl2.delete(row)
        self.tbl2.insert("", "end", values=("Shutdown", f"{self.pipeline.shutdown_val:.2f}"))

    def _draw_empty_plots(self):
        for ax in (self.ax1, self.ax2):
            ax.clear()
            ax.grid(True, linestyle=':', alpha=0.5)
        self.ax1.set_title("Recursive Forecasting", fontweight='bold')
        self.ax1.set_ylabel("Health Index (PCA–CKF)")
        self.ax2.set_title("RUL over Time", fontweight='bold', fontsize=10)
        self.ax2.set_xlabel("Timestep (Hours)")
        self.ax2.set_ylabel("RUL (Hours)")
        self.fig.tight_layout()
        self.canvas.draw()
        self._draw_empty_sensor_plots()

    def _draw_empty_sensor_plots(self):
        for ax, name in zip(self.sensor_axes, EXH_COLS):
            ax.clear()
            ax.set_title(name, fontsize=7, fontweight='bold')
            ax.tick_params(axis='both', labelsize=6)
            ax.grid(True, linestyle=':', alpha=0.5)
        self.fig_sensor.tight_layout()
        self.canvas_sensor.draw()

    def _update_sensor_plots(self, idx, window=50):
        """Show the last `window` timesteps of each cylinder up to `idx`."""
        if self.last_exh_data is None:
            return
        start = max(0, idx - window)
        x = range(start, idx)
        for j, (ax, name) in enumerate(zip(self.sensor_axes, EXH_COLS)):
            ax.clear()
            series = self.last_exh_data[start:idx, j]
            ax.plot(x, series, color=self.SENSOR_COLORS[j], linewidth=1.2)
            ax.set_title(name, fontsize=7, fontweight='bold')
            ax.tick_params(axis='both', labelsize=6)
            ax.grid(True, linestyle=':', alpha=0.5)
            if len(series) > 0:
                ax.scatter(idx - 1, series[-1], color=self.SENSOR_COLORS[j],
                           s=18, edgecolors='black', zorder=5)
        self.fig_sensor.tight_layout()
        self.canvas_sensor.draw()

    def update_plot(self, hi_data, result=None):
        # `result` is normally computed by the worker thread; recompute if a
        # caller didn't supply one.
        if result is None:
            result = self.pipeline.run_inference(hi_data)
        rul = result.rul
        forecast = result.path
        point_shutdown = result.shutdown_point
        failure_predicted = result.reached_shutdown

        self.all_rul_history.append(rul)
        mean_rul = int(np.mean(self.all_rul_history[-10:]))
        split_idx = len(hi_data)

        # Top axis: actual HI so far + recursive forecast + thresholds.
        self.ax1.clear()
        self.ax1.plot(range(split_idx), hi_data, color='blue',
                      linewidth=2, label='Input HI (PCA–CKF)')

        if forecast:
            x_fc = range(split_idx, split_idx + len(forecast))
            self.ax1.plot(x_fc, forecast, color='blue', linestyle='--',
                          label='Recursive Prediction')
            if point_shutdown:
                self.ax1.scatter(point_shutdown[0], point_shutdown[1], color='red',
                                 s=80, edgecolors='black', zorder=5)
                self.ax1.annotate(
                    f'Time: {point_shutdown[0]}', xy=point_shutdown,
                    xytext=(point_shutdown[0] - 80, point_shutdown[1] + 0.02),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontweight='bold', fontsize=8,
                )

        self.ax1.axhline(y=self.pipeline.shutdown_val, color='red', linestyle='--',
                         alpha=0.8, label=f'Shutdown ({self.pipeline.shutdown_val:.2f})')
        self.ax1.axvline(x=split_idx, color='black', linestyle=':',
                         linewidth=1.5, label='Prediction Start')
        self.ax1.set_title(
            f"Recursive Forecasting — {self.settings['data_type']} / "
            f"{self.settings['variation']} (Step: {split_idx})", fontweight='bold')
        self.ax1.set_ylabel("Health Index (PCA–CKF)")
        self.ax1.set_ylim(0, 1.05)
        self.ax1.legend(loc='upper left', fontsize='x-small')
        self.ax1.grid(True, linestyle=':', alpha=0.6)

        shutdown_t = point_shutdown[0] if point_shutdown else "N/A"
        self.ax1.text(
            0.98, 0.05,
            f"PREDICTED TIMES:\n  Shutdown: {shutdown_t} h\n\n"
            f"REMAINING LIFE:\n  RUL: {rul} Hours",
            transform=self.ax1.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='black'))

        # Bottom axis: RUL history over the run.
        self.ax2.clear()
        self.ax2.plot(range(len(self.all_rul_history)), self.all_rul_history,
                      color='#1b4332', linewidth=2)
        self.ax2.set_title("RUL over Time", fontweight='bold', fontsize=10)
        self.ax2.set_xlabel("Timestep (Hours)")
        self.ax2.set_ylabel("RUL (Hours)")
        self.ax2.grid(True, linestyle=':', alpha=0.6)

        self.fig.tight_layout()
        self.canvas.draw()
        self._update_sensor_plots(split_idx)

        # ">" prefix means we never actually hit shutdown within the horizon.
        rul_text = f"{rul} H" if failure_predicted else f">{rul} H"
        self.lbl_rul.config(text=rul_text)
        self.lbl_mean_rul.config(text=f"{mean_rul} H")
        self.tbl1.item("rul", values=("RUL", rul_text))
        self.tbl1.item("shutdown_at",
                       values=("Shutdown At",
                               f"Hour {shutdown_t}" if shutdown_t != "N/A" else "--"))

    # --- run control ------------------------------------------------------

    def _load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return

        try:
            df = pd.read_csv(path)
            if all(c in df.columns for c in EXH_COLS):
                # Full pipeline: raw exhaust temps -> HI, and keep the raw temps
                # around for the sensor strip.
                hi_full = self.preprocessor.run(df)
                exh_full = df[EXH_COLS].values.astype(np.float64)
            else:
                # CSV already holds a single HI column; no per-cylinder data.
                exh_full = None
                num_cols = df.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    hi_full = df[num_cols[0]].values.tolist()
                else:
                    df2 = pd.read_csv(path, header=None)
                    hi_full = pd.to_numeric(df2.iloc[:, 0], errors='coerce').dropna().tolist()
        except Exception as e:
            messagebox.showerror("Error loading CSV", str(e))
            return

        self.last_csv_data = hi_full
        self.last_exh_data = exh_full
        self.all_rul_history = []
        self._draw_empty_sensor_plots()

        self.is_running = True
        self.btn_sim.config(text="⏹  Stop Simulation", bg="#c0392b")
        self._sim_thread = threading.Thread(target=self._run_csv_simulation, daemon=True)
        self._sim_thread.start()

    def _run_csv_simulation(self):
        ws = self.pipeline.window_size
        delay = float(self.settings.get('sim_delay', 0.1))
        for i in range(ws, len(self.last_csv_data) + 1):
            if not self.is_running:
                break
            hi_slice = self.last_csv_data[:i]
            # Do the heavy inference here on the worker thread; only hand the
            # finished result back to the main thread for drawing.
            result = self.pipeline.run_inference(hi_slice)
            if not self.is_running:
                break
            try:
                self.root.after(0, self.update_plot, hi_slice, result)
            except (RuntimeError, tk.TclError):
                break   # main window was closed mid-run
            time.sleep(delay)
        try:
            self.root.after(0, self._stop_all)
        except (RuntimeError, tk.TclError):
            pass

    def _toggle_sim(self):
        if self.is_running:
            self._stop_all()
        else:
            self._load_csv()

    def _stop_all(self):
        self.is_running = False
        try:
            if self.btn_sim.winfo_exists():
                self.btn_sim.config(text="▶  Start Simulation", bg="#28a745")
        except tk.TclError:
            pass

    # --- settings dialog + shutdown --------------------------------------

    def _open_settings(self):
        def on_save():
            self._save_settings()
            self.pipeline.update_config(self.settings)
            self.pipeline.load_model()
            self.preprocessor = self._make_preprocessor()
            self._refresh_threshold_table()
            self.lbl_model_info.config(
                text=f"Model: {self.settings['data_type']} / {self.settings['variation']}")
            for key, widget in self.lbl_credit_widgets.items():
                widget.config(text=self.settings.get(key, '-'))

        SettingsDialog(self.root, self.settings, on_save)

    def _on_close(self):
        # Stop the worker, then quit the loop before destroying widgets — this
        # order is what lets the process exit cleanly instead of needing Ctrl+C.
        self.is_running = False
        if self._sim_thread is not None and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=1.0)
        try:
            self._save_settings()
        except Exception:
            pass
        try:
            self.root.quit()
        finally:
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = RULPredictorGUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app._on_close()