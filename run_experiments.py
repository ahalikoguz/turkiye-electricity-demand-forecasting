"""Stage 3: train and evaluate every model on identical forecast targets.

Usage (from the project root):
  python run_experiments.py list           --experiment grid        show the runs of an experiment
  python run_experiments.py run            --experiment grid [--demo] [--force] [--devices auto]
  python run_experiments.py run            --model LightTCN_medium --regime cal --H 24 --w 168 --seeds 42
  python run_experiments.py collect                                 rebuild results/runs.csv (validation only)
  python run_experiments.py select-lr                               learning rate per model (validation year)
  python run_experiments.py select-windows                          window per model/regime/horizon (validation)
  python run_experiments.py freeze                                  seal every design decision (hashes)
  python run_experiments.py evaluate       --experiment panel       test-year errors, only after `freeze`
  python run_experiments.py profile        --experiment panel       parameters, memory and latency (laptop only)

Protocol (why it is built this way):
  * A forecast origin t is the index of the FIRST target hour. The model sees hours [t-w, t) and
    predicts hours [t, t+H). This is nominal data availability: the load of the hour ending at t is used
    at t, although EPİAŞ publishes it about two hours later. The experiment `publication_delay` repeats key
    past-only models with inputs ending at t-3 (spec key `gap`: 2) to measure what that delay costs.
  * Training: 2016-2023. Validation: every origin whose H targets lie in 2024 (inputs may reach into
    2023). Test: every origin whose H targets lie in 2025 (inputs may reach into 2024). Target sets are
    identical for every window length and every model.
  * Scaling statistics come from the training period only.
  * Test forecasts are written sealed (test_sealed.npz) and never scored while models, learning rates
    and windows are chosen. `freeze` requires every campaign run to be present, current and verified,
    and records hashes of the configuration, the selections, the code, the data and every run's files;
    `evaluate` scores the test year only if nothing changed since.
  * Every run executes in its own process, in a fresh attempt folder, and is published to
    results/runs/<run_id>/ only after its files are verified. The run_id hashes the run's full contract:
    specification, resolved training settings, data hashes, protocol version, a fingerprint of the code
    that determines the family's numbers, package versions and the pinned foundation-model snapshot. A
    failed re-run never leaves an older result looking current.
  * Forecasts are stored for every (origin, lead) pair; all metrics can be recomputed from them.

Information regimes
  P: inputs are observed values of the input hours only (main comparison)
    uni              consumption history only
    cal              consumption + calendar of the input hours
    cal_no_hour, cal_no_week, cal_no_holiday      single-feature-group ablations of `cal`
  F: inputs known for the target hours are added (information experiments)
    F0    LightTCN-F with its future inputs masked (zeros): the same network as F1 without the information
    F1    + calendar of the target hours
    F1A0  F1 + the holiday-analog availability flag only (analog value masked): separates the event flag
          (which also marks bayram eves) from the information in last year's load
    F1A   F1 + holiday analog: last year's load on the same holiday day and hour, relative to a normal
          day, rescaled to the current level (known at t, see HolidayAnalog)
    F2    F1 + past temperature. ERA5 appears ~5 days late, so the last 6 days of the input window come
          from day-ahead forecasts (see Weather)
    F34   F2 + temperature of the target hours. One training run (ERA5 is the only temperature that
          exists for 2016-2023). Scheduler and early stopping follow the F3 validation loss; the best
          checkpoint of every stream is kept along that one trajectory:
            F3       archived ECMWF forecasts whose model vintage precedes t by >= 12 h, running bias
                     correction per province
            F3raw    the same forecasts without bias correction
            F4       ERA5 reanalysis in past and future: the oracle weather scenario
          and two inference-only variants that reuse the F3 checkpoint:
            F3_m24   vintage margin 24 h instead of 12 h (sensitivity)
            F3_diag  F3 past inputs, ERA5 future temperature (isolates the effect of forecast errors)
    NeuralForecast (N-HiTS) takes known-future inputs from one static table, so origin-dependent
    (realistic) weather is not implemented in its contiguous-split batch adapter. Its arms share one
    architecture: NF F0 = past calendar + zero future channels, NF F1 = past + future calendar. NF F4
    (F1 + ERA5) is implemented but not part of the campaign: NeuralForecast standardises every input
    within its window; with the cooling degree-hours, exactly zero in about half of the hours, every run
    stopped with a non-finite loss (numerical instability under per-window scaling). F1A0 exists at H = 24 and 168 only, so the
    analog value alone is separated there; at H = 1 and 720 F1 -> F1A is a combined intervention.
  "Realistic" is pseudo-real-time: the final ERA5 reanalysis stands in for the ERA5T data that would
  have been available at t (historical ERA5T snapshots are not archived), and forecasts are chosen by
  model vintage (Open-Meteo previous_dayN); their public release time is not in the data.

Training contract across families: our networks and NeuralForecast share the Huber form, the validation
scale, the optimizer, clipping, precision, learning-rate rule, stopping rule, the maximum number of updates
and the training windows (complete inputs and targets only); NeuralForecast keeps its per-window
normalisation and random window sampling. LightGBM (L2), Vanilla (OLS) and Chronos-2 LoRA (its own loss)
are trained by their own rules.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "processed" / "tr_load_hourly.csv"
DATA_REPORT = ROOT / "data" / "processed" / "data_report.json"
PROVINCE_FILE = ROOT / "data" / "processed" / "tr_temperature_provinces.npz"
CONFIG_FILE = ROOT / "configs" / "experiments.yaml"
SELECTED_FILE = ROOT / "configs" / "selected.yaml"
FROZEN_FILE = ROOT / "configs" / "frozen.json"

CALENDAR_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday"]
TEMP_FEATURES = ["t2m_popw_c", "hdh18_popw", "cdh22_popw"]
HEATING_BASE_C, COOLING_BASE_C = 18.0, 22.0          # as in prepare_data.py
ABLATION_GROUPS = {"cal_no_hour": ["hour_sin", "hour_cos"],
                   "cal_no_week": ["dow_sin", "dow_cos", "is_weekend"],
                   "cal_no_holiday": ["is_holiday"]}
P_REGIMES = ("uni", "cal", *ABLATION_GROUPS)
F_REGIMES = ("F0", "F1", "F1A0", "F1A", "F2", "F34")
NF_REGIMES = ("uni", "cal", "F0", "F1", "F4")
SEEDS_PANEL = [42, 123, 777, 2024, 3407]

# Realistic temperature inputs (see Weather)
ERA5_LATENCY_H = 144      # ERA5/ERA5T is treated as known only for hours <= t - 6 days
ISSUE_MARGIN_H = 12       # a forecast is usable only if its model vintage precedes the origin by >= 12 h
MAX_LEAD_DAY = 7          # the archive holds forecast vintages up to 7 days before the valid time
MOS_DAYS = 30             # running bias correction: mean error of the last 30 days at the same UTC hour

# Checkpoint-selection streams and evaluation variants of the F regimes. The first stream controls the
# learning-rate schedule and early stopping.
F_STREAMS = {"F0": ["main"], "F1": ["main"], "F1A0": ["main"], "F1A": ["main"], "F2": ["main"],
             "F34": ["F3", "F3raw", "F4"]}
F_TEST_ONLY = {"F34": {"F3_m24": "F3", "F3_diag": "F3", "F3raw_diag": "F3"}}   # variant -> checkpoint it reuses
TEMP_SOURCE = {            # variant -> (past temperature, future temperature, vintage margin in hours)
    "train": ("era5", "era5", ISSUE_MARGIN_H), "main": ("fc_mos", "fc_mos", ISSUE_MARGIN_H),
    "F4": ("era5", "era5", ISSUE_MARGIN_H), "F3": ("fc_mos", "fc_mos", ISSUE_MARGIN_H),
    "F3raw": ("fc_raw", "fc_raw", ISSUE_MARGIN_H), "F3_m24": ("fc_mos", "fc_mos", 24),
    "F3_diag": ("fc_mos", "era5", ISSUE_MARGIN_H), "F3raw_diag": ("fc_raw", "fc_raw", ISSUE_MARGIN_H)}
CHECKPOINT_FREE_SKIP = {"F3raw_diag"}   # identical to F3raw for models without checkpoints (foundation, Vanilla)

BASELINE_MODELS = ("Naive", "SeasonalNaive", "SeasonalNaiveWeekly")
TSFM_MODELS = ("Chronos2", "TimesFM3", "TiRex2")
RNN_MODELS = ("LSTM", "BiLSTM", "CNN_LSTM")
CPU_FAMILIES = {"baseline", "lightgbm", "vanilla"}
TSFM_MAX_HORIZON = {"TiRex2": 320}      # TiRex-2 0.2.1 truncates longer horizons; no autoregressive roll-outs

# Foundation-model weights are loaded from these local snapshots only (no silent upgrades).
HF_REPOS = {"Chronos2": "amazon/chronos-2", "Chronos2_LoRA": "amazon/chronos-2",
            "TimesFM3": "google/timesfm-3.0-pytorch", "TiRex2": "NX-AI/TiRex-2"}
HF_REVISIONS = {"amazon/chronos-2": "29ec3766d36d6f73f0696f85560a422f50e8498c",
                "google/timesfm-3.0-pytorch": "43046b85ec22d584a13f8098c2ed39c889e129c2",
                "NX-AI/TiRex-2": "05e5b26db52bfb256f1ae1bdf785589850482de3"}

# Chronos-2 LoRA fine-tuning. Chronos counts a batch in series rows (target + each covariate): with the
# 6 calendar covariates of `cal` one forecast window is 7 rows, so 252 rows = 36 windows per update.
CHRONOS_FT_LR = 1e-5      # the chronos-forecasting documentation recommends ~1e-5 for LoRA
CHRONOS_FT_STEPS = 2000   # fixed budget; validation every 100 steps, best checkpoint kept (library behaviour)
CHRONOS_FT_WINDOWS = 36
CHRONOS_LORA = {"r": 8, "lora_alpha": 16, "target_modules": ["self_attention.q", "self_attention.v",
                "self_attention.k", "self_attention.o", "output_patch_embedding.output_layer"]}



def available_cpus() -> int:
    """CPUs this process may use: its affinity mask and, inside a container, the cgroup CPU quota
    (os.cpu_count() reports the whole machine)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
    for quota_file, period_file in (("/sys/fs/cgroup/cpu.max", None),                        # cgroup v2
                                    ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        try:
            parts = Path(quota_file).read_text().split()
            quota = parts[0]
            period = parts[1] if period_file is None else Path(period_file).read_text().strip()
            if quota not in ("max", "-1"):
                n = min(n, max(1, int(quota) // int(period)))
            break
        except (OSError, ValueError, IndexError):
            continue
    return n


# LightGBM threads; the launcher sets LGB_THREADS so that the processes feeding the GPUs keep their cores
LGB_THREADS = int(os.environ.get("LGB_THREADS") or max(1, available_cpus() - 2))

os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_report() -> dict:
    report = json.loads(DATA_REPORT.read_text(encoding="utf-8"))
    if not report.get("approved"):
        sys.exit("data_report.json is not approved; review the warnings and rebuild with --accept-warnings")
    return report


# ==============================================================================================
# 1. Data and protocol
# ==============================================================================================
class Dataset:
    """Arrays shared by all models: consumption, calendar, temperature, split boundaries, scaling."""

    def __init__(self, path: Path = DATA_FILE):
        report = data_report()
        self.data_sha = report["outputs_sha256"]["tr_load_hourly.csv"]
        if sha256_file(path) != self.data_sha:
            sys.exit(f"{path.name} differs from the file described in data_report.json; rebuild the data")
        df = pd.read_csv(path)
        self.labels = df["datetime_local"].to_numpy(str)                  # 'YYYY-mm-dd HH:MM'
        self.local = pd.to_datetime(df["datetime_local"])
        self.utc_ns = pd.to_datetime(df["datetime_utc"], utc=True).astype("int64").to_numpy()
        self.y = df["consumption_mwh"].to_numpy(np.float64)
        self.calendar = df[CALENDAR_FEATURES].to_numpy(np.float32)
        self.hour, self.month = df["hour"].to_numpy(), df["month"].to_numpy()
        self.dow, self.is_holiday = df["day_of_week"].to_numpy(), df["is_holiday"].to_numpy()
        # dates with a full-day holiday or a half-day eve: excluded from "normal day" reference levels
        day = df["datetime_local"].str[:10]
        special = df["is_holiday"].eq(1) | df["holiday_type"].eq("half_day")
        self.special_day = day.isin(set(day[special])).to_numpy()
        self.n = len(df)
        year = self.local.dt.year.to_numpy()
        self.train_end = int(np.searchsorted(year, 2024))        # first index of 2024
        self.val_start, self.test_start = self.train_end, int(np.searchsorted(year, 2025))
        # consumption is standardised with training statistics only; calendar columns are already bounded
        self.y_mean = float(self.y[:self.train_end].mean())
        self.y_std = float(self.y[:self.train_end].std())
        self.y_scaled = ((self.y - self.y_mean) / self.y_std).astype(np.float32)
        self.temp = df[TEMP_FEATURES].to_numpy(np.float64) if set(TEMP_FEATURES) <= set(df.columns) else None
        if self.temp is not None:
            self.temp_mean = self.temp[:self.train_end].mean(axis=0)
            self.temp_std = self.temp[:self.train_end].std(axis=0)
        self._weather = self._analog = None

    @property
    def weather(self) -> "Weather":
        if self._weather is None:
            self._weather = Weather(self)
        return self._weather

    @property
    def analog(self) -> "HolidayAnalog":
        if self._analog is None:
            self._analog = HolidayAnalog(self)
        return self._analog

    def past_matrix(self, regime: str) -> np.ndarray:
        """Input channels of the input hours; consumption is always the LAST column (see models.py).
        Temperature channels (F2, F34) hold ERA5 here; realistic inputs replace them per origin."""
        if regime == "uni":
            return self.y_scaled[:, None]
        if regime not in P_REGIMES + F_REGIMES:
            raise ValueError(f"unknown regime {regime}")
        drop = ABLATION_GROUPS.get(regime, [])
        cols = [self.calendar[:, [i for i, c in enumerate(CALENDAR_FEATURES) if c not in drop]]]
        if regime in ("F2", "F34"):
            cols.append(((self.temp - self.temp_mean) / self.temp_std).astype(np.float32))
        return np.concatenate(cols + [self.y_scaled[:, None]], axis=1)

    def origins(self, part: str, H: int) -> np.ndarray:
        """Origins whose H targets lie entirely inside the given part of the data."""
        if part == "train_all":
            return np.arange(0, self.train_end - H + 1)
        if part == "val":
            return np.arange(self.val_start, self.test_start - H + 1)
        if part == "test":
            return np.arange(self.test_start, self.n - H + 1)
        raise ValueError(part)

    def f_origins(self, part: str, H: int) -> np.ndarray:
        """Origins of the known-future regimes: validation starts when realistic temperature inputs
        exist (the same start for every F regime and horizon, so the arms are selected on identical
        targets); the test year is complete."""
        o = self.origins(part, H)
        return o[o >= self.weather.f_val_start] if part == "val" else o

    def aggregation_origin(self) -> int:
        """2025-12-31 00:00: only used to complete the monthly aggregation of H = 168 (not part of P2)."""
        return int(np.searchsorted(self.local.to_numpy(), np.datetime64("2025-12-31T00:00")))

    def targets(self, origins: np.ndarray, H: int) -> np.ndarray:
        return self.y[np.asarray(origins)[:, None] + np.arange(H)[None, :]]


class Weather:
    """Temperature inputs of the F regimes, aligned to the load index.

    era5      (n, 3)        population-weighted ERA5 features T, HDH18, CDH22 (prepare_data.py)
    era5_prov (n, 81)       ERA5 2 m temperature of every province capital
    fc_prov[N] (m, 81)      archived ECMWF IFS forecast of model vintage ~24*N h before the valid time
                            (N = 1..7) for the m hours of the archive (from Dec 2023), NaN where missing
    mos_prov[N] (m, 81)     mean of (fc - ERA5) over the 30 previous days at the same UTC hour
    clim      (n, 3)        2016-2023 ERA5 climatology of that calendar day and hour (+-3 days smoothing)

    A realistic input for origin t never reads information published after t:
      past hours tau <= t - 6 days  -> ERA5 (ERA5T appears about 5 days behind real time)
      past hours tau >  t - 6 days  -> day-ahead forecast fc[1]
      future hours tau = t + L      -> fc[N] with N = ceil((L + margin) / 24), i.e. the freshest vintage
                                       at least `margin` hours before t; beyond N = 7 climatology (the last
                                       11 leads of H = 168 with the 12 h margin, 23 with 24 h)
      bias correction (fc_mos)      -> per province, minus mos[N] at the latest hour j <= t - 6 days with
                                       the same UTC hour as tau, so every error in the mean is known at t.
    Degree-hours are computed from the corrected province temperatures and only then weighted, exactly
    like the ERA5 features: correcting the national degree-hours directly would break HDH >= 18 - T.
    The correction removes the systematic difference between the operational model and the reanalysis
    used for training (about -0.9 C in the population-weighted temperature)."""

    def __init__(self, D: Dataset):
        from prepare_data import content_sha256
        if D.temp is None:
            sys.exit("weather columns are missing: run `prepare_data.py weather` and `prepare_data.py build`")
        z = {k: v for k, v in np.load(PROVINCE_FILE).items()}
        expected = data_report().get("outputs_content_sha256", {}).get(PROVINCE_FILE.name)
        if content_sha256(z) != expected:
            sys.exit(f"{PROVINCE_FILE.name} differs from the file described in data_report.json; rebuild the data")
        self.D, n = D, D.n
        self.w = z["weights"].astype(np.float64)
        self.era5 = D.temp
        pos, ok = self._align(z["era5_utc_ns"])
        self.era5_prov = np.full((n, len(self.w)), np.nan, np.float32)
        self.era5_prov[pos[ok]] = z["era5_t2m"][ok]
        if not np.allclose(self.era5_prov @ self.w, D.temp[:, 0], atol=1e-3):
            raise RuntimeError("province ERA5 temperatures do not reproduce the weighted feature")
        pos, ok = self._align(z["fc_utc_ns"])
        rows = pos[ok]
        if not np.array_equal(rows, np.arange(rows[0], rows[0] + len(rows))):
            raise RuntimeError("the forecast archive is not a contiguous block of hours")
        self.f0, self.m = int(rows[0]), len(rows)
        self.fc_prov = np.full((MAX_LEAD_DAY + 1, self.m, len(self.w)), np.nan, np.float32)
        self.fc_prov[1:] = z["fc_t2m"][:, ok]
        self.clim = self._climatology()
        self._build_mos()
        self.f_val_start = self._first_complete_origin()

    def _align(self, utc_ns):
        pos = np.searchsorted(self.D.utc_ns, utc_ns)
        ok = (pos < self.D.n) & (self.D.utc_ns[np.minimum(pos, self.D.n - 1)] == utc_ns)
        return pos, ok

    def _build_mos(self) -> None:
        """Running bias per province and lead day; also the availability masks used for f_val_start."""
        self.mos_prov = np.full_like(self.fc_prov, np.nan)
        era = self.era5_prov[self.f0:self.f0 + self.m]
        for N in range(1, MAX_LEAD_DAY + 1):
            err = self.fc_prov[N] - era
            # consecutive rows are consecutive UTC hours, so rows r::24 share one UTC hour
            for r in range(24):
                self.mos_prov[N, r::24] = pd.DataFrame(err[r::24]).rolling(MOS_DAYS, min_periods=MOS_DAYS).mean().to_numpy()
        self.fc_ok = np.isfinite(self.fc_prov).all(axis=2)
        self.mos_ok = np.isfinite(self.mos_prov).all(axis=2)

    def _climatology(self) -> np.ndarray:
        D = self.D
        md = pd.Series(D.labels).str[5:10].replace("02-29", "02-28")
        doy = pd.to_datetime("2019-" + md, format="%Y-%m-%d").dt.dayofyear.to_numpy() - 1
        key = doy * 24 + D.hour
        tr = slice(0, D.train_end)
        total = np.zeros((365 * 24, 3))
        count = np.zeros(365 * 24)
        np.add.at(total, key[tr], self.era5[tr])
        np.add.at(count, key[tr], 1)
        table = (total / count[:, None]).reshape(365, 24, 3)
        table = sum(np.roll(table, s, axis=0) for s in range(-3, 4)) / 7          # circular +-3 days
        return table.reshape(-1, 3)[key]

    def _gather(self, arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """arr[idx - f0] for absolute load indices; NaN outside the archive."""
        r = idx - self.f0
        inside = (r >= 0) & (r < self.m)
        out = arr[np.clip(r, 0, self.m - 1)]
        out[~inside] = np.nan
        return out

    def _features(self, T: np.ndarray) -> np.ndarray:
        """Province temperatures (..., 81) -> weighted (T, HDH18, CDH22), as in prepare_data.py."""
        return np.stack([T @ self.w, np.maximum(HEATING_BASE_C - T, 0) @ self.w,
                         np.maximum(T - COOLING_BASE_C, 0) @ self.w], axis=-1)

    @staticmethod
    def _chunked(fn, origins: np.ndarray, size: int = 512) -> np.ndarray:
        origins = np.asarray(origins, dtype=np.int64)
        if not len(origins):
            return fn(origins)
        return np.concatenate([fn(origins[i:i + size]) for i in range(0, len(origins), size)])

    def past_temperature(self, origins: np.ndarray, w: int, source: str) -> np.ndarray:
        """(B, w, 3) weighted temperature features of the input hours [t-w, t) as known at t."""
        return self._chunked(lambda o: self._past(o, w, source), origins)

    def _past(self, o: np.ndarray, w: int, source: str) -> np.ndarray:
        idx = o[:, None] + np.arange(-w, 0)[None, :]
        out = self.era5[idx].copy()
        if source == "era5":
            return out
        if source not in ("fc_mos", "fc_raw"):
            raise ValueError(source)
        k = min(w, ERA5_LATENCY_H - 1)                   # hours not yet covered by ERA5 at t
        ridx = idx[:, w - k:]
        T = self._gather(self.fc_prov[1], ridx)
        if source == "fc_mos":
            lag = o[:, None] - ridx
            j = ridx - 24 * np.ceil((ERA5_LATENCY_H - lag) / 24).astype(np.int64)
            T = T - self._gather(self.mos_prov[1], j)
        out[:, w - k:] = self._features(T)
        return out

    def future_temperature(self, origins: np.ndarray, leads: np.ndarray, source: str,
                           margin_h: int = ISSUE_MARGIN_H) -> np.ndarray:
        """(B, len(leads), 3) weighted temperature features of the target hours t+L as known at t."""
        leads = np.asarray(leads, dtype=np.int64)
        return self._chunked(lambda o: self._future(o, leads, source, margin_h), origins)

    def _future(self, o: np.ndarray, leads: np.ndarray, source: str, margin_h: int) -> np.ndarray:
        idx = o[:, None] + leads[None, :]
        if source == "era5":
            return self.era5[idx].copy()
        if source not in ("fc_mos", "fc_raw"):
            raise ValueError(source)
        N = np.ceil((leads + margin_h) / 24).astype(np.int64)
        out = self.clim[idx].copy()
        for n_day in range(1, MAX_LEAD_DAY + 1):
            c = N == n_day
            if not c.any():
                continue
            T = self._gather(self.fc_prov[n_day], idx[:, c])
            if source == "fc_mos":
                j = idx[:, c] - 24 * np.ceil((leads[c] + ERA5_LATENCY_H) / 24).astype(np.int64)[None, :]
                T = T - self._gather(self.mos_prov[n_day], j)
            out[:, c] = self._features(T)
        return out

    def scale(self, raw: np.ndarray) -> np.ndarray:
        return ((raw - self.D.temp_mean) / self.D.temp_std).astype(np.float32)

    def _ok(self, mask: np.ndarray, idx: np.ndarray) -> np.ndarray:
        r = idx - self.f0
        return ((r >= 0) & (r < self.m)) & mask[np.clip(r, 0, self.m - 1)]

    def _complete(self, o: np.ndarray) -> np.ndarray:
        """True where every realistic variant (bias-corrected or raw, 12 or 24 h margin) is complete for
        H = 168 and any window: the 143 most recent input hours and every forecast lead."""
        lag = np.arange(ERA5_LATENCY_H - 1, 0, -1)
        ridx = o[:, None] - lag[None, :]
        j = ridx - 24 * np.ceil((ERA5_LATENCY_H - lag) / 24).astype(np.int64)[None, :]
        ok = (self._ok(self.fc_ok[1], ridx) & self._ok(self.mos_ok[1], j)).all(axis=1)
        leads = np.arange(168)
        for margin in (ISSUE_MARGIN_H, 24):
            N = np.ceil((leads + margin) / 24).astype(np.int64)
            for n_day in range(1, MAX_LEAD_DAY + 1):
                c = N == n_day
                idx = o[:, None] + leads[c][None, :]
                j = idx - 24 * np.ceil((leads[c] + ERA5_LATENCY_H) / 24).astype(np.int64)[None, :]
                ok &= (self._ok(self.fc_ok[n_day], idx) & self._ok(self.mos_ok[n_day], j)).all(axis=1)
        return ok

    def _first_complete_origin(self) -> int:
        o = self.D.origins("val", 168)
        bad = np.flatnonzero(~self._complete(o))
        if len(bad) == len(o):
            raise RuntimeError("no validation origin has complete realistic temperature inputs")
        if not self._complete(self.D.origins("test", 168)).all():
            raise RuntimeError("realistic temperature inputs are incomplete in the test year")
        return int(o[bad[-1] + 1]) if len(bad) else int(o[0])


class HolidayAnalog:
    """Known-future holiday input (Regimes F1A0, F1A): what happened on the same holiday day last year.

    For a target hour on a religious holiday (Ramazan: eve + 3 days, Kurban: eve + 4 days) the analog is
    the same day of the same bayram one year earlier (it moves ~11 days per year) at the same hour; for a
    fixed-date national holiday (and the 28 October eve) it is the same date one year earlier, if that
    date was the same holiday. The analog is expressed relative to a normal day at the same hour:
        ratio = load(analog hour) / level(analog hour - 24 h) - 1
    where level = mean load at that hour of day over the 28 most recent normal days (no holiday, no eve).
    This hour-of-day level (not the same-weekday level) was chosen on 2017-2024 holiday hours: used as a
    stand-alone day-ahead forecast it had 5.0 % MAPE against 7.3 % for the same-weekday level and 22.8 %
    without any holiday adjustment; the test year played no part in the choice.
    For origin t and target hour t+L the input is
        ratio * level(most recent same hour before t) / y_std        (0 where no analog exists)
    plus a 0/1 availability channel. Every load it reads was observed before the origin (tested)."""

    def __init__(self, D: Dataset):
        from prepare_data import FIXED_HOLIDAYS, KURBAN_FIRST_DAY, RAMAZAN_FIRST_DAY
        self.y_std = D.y_std
        pos = {lab: i for i, lab in enumerate(D.labels)}
        analog = np.full(D.n, -1, dtype=np.int64)

        def link(day: date, prev: date) -> None:
            for h in range(24):
                a, b = pos.get(f"{day:%Y-%m-%d} {h:02d}:00"), pos.get(f"{prev:%Y-%m-%d} {h:02d}:00")
                if a is not None and b is not None:
                    analog[a] = b

        religious = set()
        for first, n_days in ((RAMAZAN_FIRST_DAY, 3), (KURBAN_FIRST_DAY, 4)):
            for year in sorted(first):
                for k in range(-1, n_days):                              # k = -1 is the eve (arife)
                    religious.add(first[year] + timedelta(days=k))
                    if year - 1 in first:
                        link(first[year] + timedelta(days=k), first[year - 1] + timedelta(days=k))
        years = sorted(set(RAMAZAN_FIRST_DAY))
        national = list(FIXED_HOLIDAYS) + [(10, 28, "Republic Day eve", years[0])]
        for m, d, _, since in national:
            for year in years:
                day, prev = date(year, m, d), date(year - 1, m, d)
                if year - 1 >= since and day not in religious and prev not in religious:
                    link(day, prev)

        # level: mean of the 28 most recent normal values at the same hour of day (stride 24 hours)
        normal = pd.Series(np.where(D.special_day, np.nan, D.y))
        self.level = np.full(D.n, np.nan)
        for r in range(24):
            v = normal.iloc[r::24]
            self.level[r::24] = v.dropna().rolling(28, min_periods=28).mean().reindex(v.index).ffill().to_numpy()
        has = analog >= 0
        ref = np.where(has, analog - 24, 0)
        ratio = np.full(D.n, np.nan)
        ok = has & (ref >= 0)
        ratio[ok] = D.y[analog[ok]] / self.level[ref[ok]] - 1
        self.ratio = ratio
        self.available = np.isfinite(ratio)
        self.analog_index = analog

    def channels(self, origins: np.ndarray, start: int, stop: int) -> np.ndarray:
        """(B, stop-start, 2): [scaled expected holiday deviation, availability] for leads start..stop-1
        relative to each origin (negative leads = input hours)."""
        L = np.arange(start, stop)
        tau = origins[:, None] + L[None, :]
        k = np.where(L < 0, 1, L // 24 + 1)                            # most recent same hour before t
        ref = tau - 24 * k[None, :]
        level = self.level[np.maximum(ref, 0)]
        ok = self.available[tau] & (ref >= 0) & np.isfinite(level)
        dev = np.where(ok, np.nan_to_num(self.ratio[tau]) * np.nan_to_num(level), 0.0) / self.y_std
        return np.stack([dev, ok.astype(np.float64)], axis=-1)


class FInputs:
    """Inputs of LightTCN-F for any set of origins, built per batch (vectorised numpy):
    past (B, w, P): calendar [+ temperature] + consumption (last column), as known at t;
    future (B, H, n_future): calendar [+ holiday analog] [+ temperature] of the target hours.
    Realistic temperature of fixed origin sets (validation, test) is computed once and cached."""

    def __init__(self, D: Dataset, regime: str, w: int, H: int):
        if regime not in F_REGIMES:
            raise ValueError(regime)
        self.D, self.regime, self.w, self.H = D, regime, w, H
        self.past = D.past_matrix(regime)
        n_cal = len(CALENDAR_FEATURES)
        self.temp_cols = slice(n_cal, n_cal + len(TEMP_FEATURES))
        self.n_future = n_cal + 2 * (regime in ("F1A0", "F1A")) + len(TEMP_FEATURES) * (regime == "F34")
        self._cache = {}

    def _temperatures(self, origins: np.ndarray, variant: str):
        """Scaled (recent past, future) temperature of a variant, from the cache if possible. The recent
        past is None when ERA5 is used (it is already in the past matrix); the future is None outside F34."""
        past_src, future_src, margin = TEMP_SOURCE[variant]
        hit = self._cache.get(variant)
        if hit is not None:
            rows = hit[0][origins]
            if (rows >= 0).all():
                return tuple(a[rows] if a is not None else None for a in hit[1:])
        W, k = self.D.weather, min(self.w, ERA5_LATENCY_H - 1)
        past = W.scale(W.past_temperature(origins, self.w, past_src)[:, -k:]) if past_src != "era5" else None
        fut = W.scale(W.future_temperature(origins, np.arange(self.H), future_src, margin)) \
            if self.regime == "F34" else None
        return past, fut

    def cache(self, origins: np.ndarray, variant: str) -> None:
        if self.regime not in ("F2", "F34"):
            return
        origins = np.asarray(origins)
        past, fut = self._temperatures(origins, variant)
        rows = np.full(self.D.n, -1, dtype=np.int64)
        rows[origins] = np.arange(len(origins))
        self._cache[variant] = (rows, past, fut)

    def arrays(self, origins: np.ndarray, variant: str) -> tuple[np.ndarray, np.ndarray]:
        D, w, H = self.D, self.w, self.H
        past = self.past[origins[:, None] + np.arange(-w, 0)[None, :]]
        parts = [D.calendar[origins[:, None] + np.arange(H)[None, :]]]
        if self.regime in ("F1A0", "F1A"):
            a = D.analog.channels(origins, 0, H)
            if self.regime == "F1A0":
                a[..., 0] = 0.0                                  # analog value masked, flag kept
            parts.append(a)
        if self.regime in ("F2", "F34"):
            recent, fut = self._temperatures(origins, variant)
            if recent is not None:
                past[:, -recent.shape[1]:, self.temp_cols] = recent
            if self.regime == "F34":
                parts.append(fut)
        future = np.concatenate(parts, axis=-1).astype(np.float32)
        if self.regime == "F0":
            future = np.zeros_like(future)
        return past.astype(np.float32), future

    def collate(self, variant: str, with_target: bool = True):
        import torch
        H = self.H

        def build(batch):
            o = np.asarray(batch, dtype=np.int64)
            past, future = self.arrays(o, variant)
            y = self.D.y_scaled[o[:, None] + np.arange(H)[None, :]] if with_target else np.zeros((len(o), 0), np.float32)
            return torch.from_numpy(past), torch.from_numpy(future), torch.from_numpy(np.ascontiguousarray(y))
        return build


def seasonal_naive(y: np.ndarray, origins: np.ndarray, H: int, m: int, gap: int = 0) -> np.ndarray:
    """Repeat the last observed season cyclically: the forecast for t+h is the most recent value of the
    same phase that is available, i.e. at an index <= t-1-gap (gap = 0: y[t - m + (h mod m)])."""
    h = np.arange(H)[None, :]
    return y[np.asarray(origins)[:, None] + h - m * np.ceil((h + 1 + gap) / m).astype(np.int64)]


def seasonal_period(H: int) -> int:
    return 24 if H <= 24 else 168


def metrics(y_true: np.ndarray, y_pred: np.ndarray, y: np.ndarray, train_end: int, H: int) -> dict:
    """MAE/RMSE in MWh, MAPE and MASE dimensionless. MASE uses the in-sample seasonal-naive MAE of the
    training period with the fixed seasonal period of the horizon. Averages run over all (origin, lead)."""
    err = y_pred - y_true
    m = seasonal_period(H)
    scale = float(np.mean(np.abs(y[m:train_end] - y[:train_end - m])))
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAPE": float(np.mean(np.abs(err) / y_true) * 100), "MASE": float(np.mean(np.abs(err)) / scale),
            "mean_signed_pct_error": float(np.mean(err / y_true) * 100),
            "under_forecast_share": float(np.mean(err < 0)),
            "MAE_by_lead": np.mean(np.abs(err), axis=0).round(3).tolist()}


def variant_key(variant: str) -> str:
    """Suffix of prediction/metric keys: '' for the single variant of a run, '@F3' etc. otherwise."""
    return "" if variant == "main" else f"@{variant}"


# ==============================================================================================
# 2. Neural network training (models.py architectures)
# ==============================================================================================
def set_seed(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_window_dataset(past: np.ndarray, target: np.ndarray, origins: np.ndarray, w: int, H: int, gap: int = 0):
    """Item i = (past[t-gap-w:t-gap], target[t:t+H]) for t = origins[i]; gap > 0 leaves the most recent hours
    out (publication-delay sensitivity). Windows are cut lazily, so long windows (w = 2160) do not need a
    materialised (N, w, F) array."""
    import torch

    class _Windows(torch.utils.data.Dataset):
        def __init__(self):
            self.p = torch.from_numpy(np.ascontiguousarray(past, dtype=np.float32))
            self.y = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))
            self.o = np.asarray(origins, dtype=np.int64)

        def __len__(self):
            return len(self.o)

        def __getitem__(self, i):
            t = int(self.o[i])
            return self.p[t - gap - w:t - gap], self.y[t:t + H]

    return _Windows()


def micro_batches(spec: dict) -> int:
    """Training cells whose activations at batch 256 exceed the 16 GB laptop GPU (measured with --demo:
    DualTCN(_Attn)_large needs 20 GB at w = 2160; its two stacks of 128 channels hold 12 dilated blocks
    each). They accumulate the size-weighted gradients of the same 256 windows over two half-batches:
    the same nominal update (no batch statistics in these models), not a bit-identical one, because
    dropout draws and the summation order change. Without it Windows silently spills to shared system
    memory and the run becomes ~50x slower instead of failing."""
    if spec["model"] in ("DualTCN_large", "DualTCN_Attn_large") and spec["w"] >= 1440:
        return 2
    return 1


def prediction_batch(w: int) -> int:
    """Inference batch (fp32, no gradients). Long windows use smaller batches: BiLSTM at w = 2160 needs
    17.5 GB at 1024 windows but 4.4 GB at 256."""
    return 1024 if w <= 720 else 256


def train_torch(model, loader, val_loaders: dict, device, tc: dict, log, micro: int = 1, control: str | None = None) -> dict:
    """Adam + Huber loss + ReduceLROnPlateau + early stopping on the validation loss (same rule for every
    model); mixed precision on GPU. A batch is (inputs..., target) and the model is called as model(*inputs).
    Several validation loaders keep separate best checkpoints along ONE training trajectory (F34); the
    `control` loader alone drives the scheduler and early stopping, so the trajectory is the one a single
    loader would produce. Returns the history and the best state of every loader."""
    import torch
    loss_fn = torch.nn.HuberLoss(delta=tc["huber_delta"])
    opt = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=tc["scheduler_factor"],
                                                       patience=tc["scheduler_patience"], min_lr=tc["scheduler_min_lr"])
    use_amp = device.type == "cuda" and tc["use_amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    names = list(val_loaders)
    control = control or names[0]
    best = dict.fromkeys(names, float("inf"))
    best_state, best_epoch, bad = dict.fromkeys(names), dict.fromkeys(names, 0), dict.fromkeys(names, 0)
    hist = {"train": [], "lr": [], **{f"val_{k}": [] for k in names}}
    t0 = time.time()
    for epoch in range(1, tc["max_epochs"] + 1):
        model.train()
        tot = 0.0
        for batch in loader:
            *inputs, yb = [b.to(device, non_blocking=True) for b in batch]
            opt.zero_grad(set_to_none=True)
            for part in np.array_split(np.arange(len(yb)), micro):      # micro = 1: the whole batch
                sl = slice(int(part[0]), int(part[-1]) + 1)
                with torch.autocast("cuda", enabled=use_amp):
                    loss = loss_fn(model(*[x[sl] for x in inputs]), yb[sl]) * (len(part) / len(yb))
                scaler.scale(loss).backward()
                tot += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["gradient_clip_norm"])
            scaler.step(opt)
            scaler.update()
        vl = {k: evaluate_loss(model, l, loss_fn, device, use_amp) for k, l in val_loaders.items()}
        hist["train"].append(tot / max(1, len(loader)))
        hist["lr"].append(opt.param_groups[0]["lr"])
        for k, v in vl.items():
            hist[f"val_{k}"].append(v)
        sched.step(vl[control])
        for k, v in vl.items():
            if v < best[k]:
                best[k], best_epoch[k], bad[k] = v, epoch, 0
                best_state[k] = {p: q.detach().cpu().clone() for p, q in model.state_dict().items()}
            else:
                bad[k] += 1
        if epoch == 1 or epoch % 5 == 0:
            log(f"epoch {epoch:3d} train {hist['train'][-1]:.5f} "
                + " ".join(f"val_{k} {v:.5f}" for k, v in vl.items()) + f" lr {hist['lr'][-1]:.1e}")
        if bad[control] >= tc["early_stopping_patience"]:
            break
    single = len(names) == 1
    return {"best_val_loss": best[names[0]] if single else best,
            "best_epoch": best_epoch[names[0]] if single else best_epoch, "control_stream": control,
            "epochs_run": len(hist["train"]), "train_time_sec": round(time.time() - t0, 2),
            "history": hist, "best_states": best_state}


def evaluate_loss(model, loader, loss_fn, device, use_amp) -> float:
    import torch
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", enabled=use_amp):
        for batch in loader:
            *inputs, yb = [b.to(device) for b in batch]
            tot += loss_fn(model(*inputs), yb).item() * len(yb)
            n += len(yb)
    return tot / max(1, n)


def predict_loader(model, loader, device) -> tuple[np.ndarray, float]:
    """Forecasts (scaled) for every batch of the loader (the last element of a batch is ignored);
    also returns the pure inference time in seconds (fp32, as used for all reported forecasts)."""
    import torch
    model.eval()
    out = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            out.append(model(*[b.to(device) for b in batch[:-1]]).float().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(out), time.time() - t0


def predict_torch(model, past, origins, w, H, device, batch=None, gap: int = 0) -> tuple[np.ndarray, float]:
    """Forecasts (scaled) of a past-only model for the given origins, plus the inference time."""
    import torch
    batch = batch or prediction_batch(w)
    ds = make_window_dataset(past, np.zeros(len(past), np.float32), origins, w, 0, gap)
    return predict_loader(model, torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0), device)


def _torch_precision(model, device, tc: dict) -> dict:
    """Effective numeric settings of a run. PyTorch lets cuDNN convolutions use TF32 in float32 by default
    (fp32_conv = 'tf32'); this is recorded rather than changed."""
    import torch
    amp = device.type == "cuda" and tc["use_amp"]
    return {"device": device.type, "amp_training": amp, "autocast_dtype": "float16" if amp else None,
            "param_dtype": str(next(model.parameters()).dtype), "inference": "float32",
            "fp32_matmul": torch.backends.cuda.matmul.fp32_precision, "fp32_conv": torch.backends.cudnn.conv.fp32_precision}


def _feature_order(regime: str) -> list[str]:
    if regime == "uni":
        return ["consumption_scaled"]
    drop = ABLATION_GROUPS.get(regime, [])
    temp = [f"{c}_scaled" for c in TEMP_FEATURES] if regime in ("F2", "F34") else []
    return [c for c in CALENDAR_FEATURES if c not in drop] + temp + ["consumption_scaled"]


def run_torch(spec: dict, D: Dataset, tc: dict, out: Path, log) -> dict:
    if spec["regime"] in F_REGIMES:
        return run_torch_f(spec, D, tc, out, log)
    import torch
    from torch.utils.data import DataLoader
    from models import count_parameters, create_model
    H, w, gap = spec["H"], spec["w"], spec.get("gap", 0)
    past = D.past_matrix(spec["regime"])
    train_o = D.origins("train_all", H)
    train_o = train_o[train_o >= w + gap]               # the input window must lie inside the data
    val_o = D.origins("val", H)
    if spec.get("demo"):
        train_o, val_o = train_o[-3000:], val_o[::24]
    set_seed(spec["seed"])                              # seed -> model init -> training (fixed order)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(spec["model"], past.shape[1], H, w).to(device)
    loader = DataLoader(make_window_dataset(past, D.y_scaled, train_o, w, H, gap), batch_size=tc["batch_size"],
                        shuffle=True, num_workers=0, pin_memory=True)
    vloader = DataLoader(make_window_dataset(past, D.y_scaled, val_o, w, H, gap), batch_size=2 * tc["batch_size"],
                         shuffle=False, num_workers=0, pin_memory=True)
    res = train_torch(model, loader, {"val": vloader}, device, tc, log, micro=micro_batches(spec))
    model.load_state_dict(res.pop("best_states")["val"])
    torch.save(model.state_dict(), out / "checkpoint.pt")
    info = {"params": count_parameters(model), "train_windows": int(len(train_o)),
            "micro_batches": micro_batches(spec), "precision": _torch_precision(model, device, tc), **res}
    preds = {}
    for part in ("val", "test"):
        o = D.origins(part, H) if not spec.get("demo") or part == "test" else val_o
        p, secs = predict_torch(model, past, o, w, H, device, gap=gap)
        preds[part] = (o, p * D.y_std + D.y_mean)
        info[f"{part}_inference_sec"] = round(secs, 3)
    if H == 168 and spec["regime"] in ("uni", "cal") and not gap:
        o = np.array([D.aggregation_origin()])
        preds["aggregation_extra"] = (o, predict_torch(model, past, o, w, H, device)[0] * D.y_std + D.y_mean)
    artefacts = {"files": ["checkpoint.pt"], "feature_order": _feature_order(spec["regime"]),
                 "scaler": {"y_mean": D.y_mean, "y_std": D.y_std}}
    return {"info": info, "preds": preds, "artefacts": artefacts}


def run_torch_f(spec: dict, D: Dataset, tc: dict, out: Path, log) -> dict:
    """LightTCN-F in the known-future regimes. Training always uses ERA5 temperature (the only one that
    exists for 2016-2023); checkpoints are selected with the inputs of each evaluation variant."""
    import torch
    from torch.utils.data import DataLoader
    from models import count_parameters, create_model
    H, w, regime = spec["H"], spec["w"], spec["regime"]
    if spec.get("gap"):
        raise ValueError("the publication-gap sensitivity is defined for past-only regimes")
    FI = FInputs(D, regime, w, H)
    train_o = D.origins("train_all", H)
    train_o = train_o[train_o >= w]
    val_o, test_o = D.f_origins("val", H), D.origins("test", H)
    if spec.get("demo"):
        train_o, val_o = train_o[-3000:], val_o[::24]
    streams = F_STREAMS[regime]
    variants = {s: s for s in streams} | F_TEST_ONLY.get(regime, {})
    for v in variants:                                   # realistic temperature of fixed origins: once
        FI.cache(np.concatenate([val_o, test_o]), v)
    set_seed(spec["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(spec["model"], FI.past.shape[1], H, w, n_future=FI.n_future).to(device)
    bs = tc["batch_size"]
    loader = DataLoader(train_o, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True,
                        collate_fn=FI.collate("train"))
    vloaders = {s: DataLoader(val_o, batch_size=2 * bs, shuffle=False, num_workers=0, collate_fn=FI.collate(s))
                for s in streams}
    res = train_torch(model, loader, vloaders, device, tc, log, micro=micro_batches(spec), control=streams[0])
    states = res.pop("best_states")
    for s, st in states.items():
        torch.save(st, out / f"checkpoint_{s}.pt")
    info = {"params": count_parameters(model), "train_windows": int(len(train_o)), "n_future": FI.n_future,
            "val_first_origin_local": str(D.local[val_o[0]]), "precision": _torch_precision(model, device, tc), **res}
    preds = {}
    for variant, ckpt in variants.items():
        model.load_state_dict(states[ckpt])
        for part, o in (("val", val_o), ("test", test_o)):
            dl = DataLoader(o, batch_size=prediction_batch(w), shuffle=False, num_workers=0,
                            collate_fn=FI.collate(variant, False))
            p, secs = predict_loader(model, dl, device)
            preds[part + variant_key(variant)] = (o, p * D.y_std + D.y_mean)
            info[f"{part}{variant_key(variant)}_inference_sec"] = round(secs, 3)
    artefacts = {"files": [f"checkpoint_{s}.pt" for s in states], "feature_order": _feature_order(regime),
                 "future_order": CALENDAR_FEATURES + (["analog_dev", "analog_flag"] if regime in ("F1A0", "F1A") else [])
                 + ([f"{c}_scaled" for c in TEMP_FEATURES] if regime == "F34" else []),
                 "scaler": {"y_mean": D.y_mean, "y_std": D.y_std, "temp_mean": D.temp_mean.tolist(),
                            "temp_std": D.temp_std.tolist()}, "checkpoint_of_variant": variants}
    return {"info": info, "preds": preds, "artefacts": artefacts}


# ==============================================================================================
# 3. Other model families: baselines, NeuralForecast, foundation models, LightGBM, Vanilla
# ==============================================================================================
def run_baseline(spec: dict, D: Dataset, tc: dict, out: Path, log) -> dict:
    H, gap = spec["H"], spec.get("gap", 0)
    m = {"Naive": None, "SeasonalNaive": seasonal_period(H), "SeasonalNaiveWeekly": 168}[spec["model"]]

    def forecast(o):
        return np.repeat(D.y[o - 1 - gap][:, None], H, axis=1) if m is None else seasonal_naive(D.y, o, H, m, gap)

    preds = {part: (D.origins(part, H), forecast(D.origins(part, H))) for part in ("val", "test")}
    if H == 168 and not gap:
        o = np.array([D.aggregation_origin()])
        preds["aggregation_extra"] = (o, forecast(o))
    return {"info": {"params": 0, "seasonal_period": m, "precision": "float64 CPU"}, "preds": preds}


def _nf_frame(cv: pd.DataFrame, H: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cv = (cv.reset_index() if "unique_id" not in cv.columns else cv).sort_values(["cutoff", "ds"])
    col = [c for c in cv.columns if c not in ("unique_id", "ds", "cutoff", "y")][0]
    o = cv["cutoff"].to_numpy().reshape(-1, H)[:, 0] + 1
    return o, cv[col].to_numpy(np.float64).reshape(-1, H), cv["y"].to_numpy(np.float64).reshape(-1, H)


def _nf_callback_class():
    """Lightning callback that gives NeuralForecast the rules of our own loop: ReduceLROnPlateau on the
    validation loss at every validation check (NF's own scheduler is set to a constant factor of 1), and,
    at the end of fit, the weights of the best check (NF's early stopping alone keeps the last weights)."""
    import torch
    from pytorch_lightning import Callback

    class KeepBestAndPlateau(Callback):
        def __init__(self, factor, patience, min_lr):
            self.factor, self.patience, self.min_lr = factor, patience, min_lr
            self.best, self.best_step, self.state, self.plateau, self.checks = float("inf"), -1, None, None, 0

        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            v = trainer.callback_metrics.get("ptl/val_loss")
            if v is None:
                return
            v = float(v)
            self.checks += 1
            if self.plateau is None:
                self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    trainer.optimizers[0], factor=self.factor, patience=self.patience, min_lr=self.min_lr)
            self.plateau.step(v)
            if v < self.best:
                self.best, self.best_step = v, int(trainer.global_step)
                self.state = {k: t.detach().cpu().clone() for k, t in pl_module.state_dict().items()}

        def on_fit_end(self, trainer, pl_module):
            if self.state is not None:
                pl_module.load_state_dict(self.state)

    return KeepBestAndPlateau


def nf_dataframe(D: Dataset, regime: str) -> tuple[pd.DataFrame, dict]:
    """Long table of NeuralForecast (training-standardised load and exogenous columns) and the exogenous
    lists of the regime. NF F0 has the future columns of F1 set to zero: the same network without them."""
    df = pd.DataFrame({"unique_id": "TR", "ds": np.arange(D.n, dtype=np.int64), "y": D.y_scaled.astype(np.float64)})
    kw = {}
    if regime != "uni":
        for c, col in zip(CALENDAR_FEATURES, D.calendar.T):
            df[f"h_{c}"] = col
        kw["hist_exog_list"] = [f"h_{c}" for c in CALENDAR_FEATURES]
    if regime in ("F0", "F1", "F4"):
        for c, col in zip(CALENDAR_FEATURES, D.calendar.T):
            df[f"f_{c}"] = 0.0 if regime == "F0" else col
        kw["futr_exog_list"] = [f"f_{c}" for c in CALENDAR_FEATURES]
        if regime == "F4":                                  # oracle: ERA5 temperature of every hour
            for c, col in zip(TEMP_FEATURES, ((D.temp - D.temp_mean) / D.temp_std).T):
                df[f"f_{c}"] = col
            kw["futr_exog_list"] += [f"f_{c}" for c in TEMP_FEATURES]
    return df, kw


def run_neuralforecast(spec: dict, D: Dataset, tc: dict, out: Path, log) -> dict:
    """NeuralForecast models on exactly our origins and under our training contract: the load is the
    training-standardised series (so Huber delta = 1 and the validation loss use the same scale as our
    models; NF keeps its own per-window normalisation, part of these architectures), Adam with weight
    decay, gradient clipping 1.0, fp16 mixed precision, ReduceLROnPlateau and early stopping on the 2024
    validation loss checked once per epoch-equivalent of updates, best validation weights restored.
    One fit on 2016-2023; cross_validation(refit=False, step_size=1) forecasts every origin of 2025 and
    the fitted model (use_fitted=True, no retraining) every origin of 2024. The validation split of NF is
    the whole of 2024 in every regime: a contiguous split cannot drop January-March 2024 without either
    training on it or masking inputs, so NF comparisons across F arms are made within NF only."""
    import torch
    import neuralforecast
    import neuralforecast.models as nfm
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import HuberLoss
    H, w, regime = spec["H"], spec["w"], spec["regime"]
    if spec.get("gap"):
        raise ValueError("the publication-gap sensitivity is not implemented for this family")
    if regime not in NF_REGIMES:
        raise ValueError(f"regime {regime} is not available for NeuralForecast models")
    name = spec["model"].removeprefix("NF_")
    cls = getattr(nfm, name)
    df, kw = nf_dataframe(D, regime)
    if kw.get("hist_exog_list") and not getattr(cls, "EXOGENOUS_HIST", False):
        raise ValueError(f"{name} does not accept historical exogenous inputs in this NeuralForecast version")
    if kw.get("futr_exog_list") and not getattr(cls, "EXOGENOUS_FUTR", False):
        raise ValueError(f"{name} does not accept known-future inputs in this NeuralForecast version")
    if name in ("TimeXer", "iTransformer"):
        kw["n_series"] = 1
    # train only on windows with complete inputs and all H targets inside 2016-2023, as our networks do
    # (the library default also accepts windows padded beyond the end of the training part)
    kw["training_data_availability_threshold"] = 1.0
    n_train_windows = D.train_end - w - H + 1
    steps_per_epoch = -(-n_train_windows // tc["batch_size"])        # ceil, as our DataLoader
    use_amp = tc["use_amp"] and torch.cuda.is_available()
    callback = _nf_callback_class()(tc["scheduler_factor"], tc["scheduler_patience"], tc["scheduler_min_lr"])
    budget = dict(max_steps=tc["max_epochs"] * steps_per_epoch, val_check_steps=steps_per_epoch,
                  early_stop_patience_steps=tc["early_stopping_patience"], learning_rate=tc["learning_rate"],
                  windows_batch_size=tc["batch_size"], loss=HuberLoss(delta=tc["huber_delta"]),
                  valid_loss=HuberLoss(delta=tc["huber_delta"]), scaler_type="standard", random_seed=spec["seed"])
    if spec.get("demo"):
        budget.update(max_steps=5, early_stop_patience_steps=-1, val_check_steps=5)
    kw["inference_windows_batch_size"] = 256 if w <= 720 else 64       # inference only
    model = cls(h=H, input_size=w, **budget, **kw, optimizer=torch.optim.Adam,
                optimizer_kwargs={"weight_decay": tc["weight_decay"]},
                lr_scheduler=torch.optim.lr_scheduler.ConstantLR, lr_scheduler_kwargs={"factor": 1.0, "total_iters": 0},
                gradient_clip_val=tc["gradient_clip_norm"], precision="16-mixed" if use_amp else "32-true",
                callbacks=[callback], enable_checkpointing=False, enable_progress_bar=False, logger=False,
                enable_model_summary=False, default_root_dir=str(out))
    nf = NeuralForecast(models=[model], freq=1)
    n_val, n_test = D.test_start - D.val_start, D.n - D.test_start
    if D.n - n_val - n_test != D.train_end:
        raise AssertionError("NeuralForecast training targets would not end on 2023-12-31")

    def unscale(a):
        return a * D.y_std + D.y_mean

    t0 = time.time()
    cv = nf.cross_validation(df=df, n_windows=None, step_size=1, val_size=n_val, test_size=n_test, refit=False)
    fit_sec = time.time() - t0
    o, p, y = _nf_frame(cv, H)
    assert np.array_equal(o, D.origins("test", H)), "NeuralForecast test origins differ from the protocol"
    assert np.allclose(unscale(y), D.targets(o, H), atol=0.5), "NeuralForecast test targets misaligned"
    trained = nf.models[0]                              # NeuralForecast trains its own copy of `model`
    cb = next(c for c in trained.trainer_kwargs.get("callbacks", []) if type(c).__name__ == "KeepBestAndPlateau")
    t0 = time.time()
    cvv = nf.cross_validation(df=df.iloc[:D.test_start], n_windows=None, step_size=1, test_size=n_val,
                              refit=False, use_fitted=True)
    val_sec = time.time() - t0
    ov, pv, yv = _nf_frame(cvv, H)
    assert np.array_equal(ov, D.origins("val", H)), "NeuralForecast validation origins differ from the protocol"
    assert np.allclose(unscale(yv), D.targets(ov, H), atol=0.5), "NeuralForecast validation targets misaligned"
    preds = {"test": (o, unscale(p)), "val": (ov, unscale(pv))}
    if H == 168 and regime in ("uni", "cal"):
        agg = D.aggregation_origin()
        fc = nf.predict(df=df.iloc[:agg])
        fc = (fc.reset_index() if "unique_id" not in fc.columns else fc).sort_values("ds")
        assert np.array_equal(fc["ds"].to_numpy(), np.arange(agg, agg + H)), "aggregation origin misaligned"
        col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
        preds["aggregation_extra"] = (np.array([agg]), unscale(fc[col].to_numpy(np.float64))[None, :])
    nf.save(path=str(out / "nf_model"), save_dataset=False, overwrite=True)
    info = {"params": int(sum(q.numel() for q in trained.parameters() if q.requires_grad)),
            "fit_and_predict_sec": round(fit_sec, 2), "val_predict_sec": round(val_sec, 2),
            "neuralforecast": neuralforecast.__version__,
            "nf_config": {k: (str(v) if k in ("loss", "valid_loss") else v) for k, v in budget.items()},
            # the trainer is replaced when predicting, so the step count comes from the recorded trajectories
            "steps_trained": int(trained.train_trajectories[-1][0]) if trained.train_trajectories else None,
            "best_step": cb.best_step, "best_val_loss": cb.best, "validation_checks": cb.checks,
            "valid_trajectory": [[int(a), float(b)] for a, b in trained.valid_trajectories],
            "training_windows": "complete inputs and targets only (training_data_availability_threshold = 1.0)",
            "precision": {"device": "cuda" if torch.cuda.is_available() else "cpu",
                          "trainer": "16-mixed" if use_amp else "32-true",
                          "param_dtype": str(next(trained.parameters()).dtype)}}
    artefacts = {"files": ["nf_model"], "feature_order": kw.get("hist_exog_list", []) + kw.get("futr_exog_list", []),
                 "scaler": {"y_mean": D.y_mean, "y_std": D.y_std, "nf_window_scaler": "standard"}}
    return {"info": info, "preds": preds, "artefacts": artefacts}


def hf_snapshot(model: str) -> Path:
    """Local folder of the pinned weights of a foundation model (never a download or a newer revision)."""
    repo = HF_REPOS[model]
    d = Path(os.environ["HF_HOME"]) / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / HF_REVISIONS[repo]
    if not d.is_dir():
        sys.exit(f"pinned snapshot of {repo} ({HF_REVISIONS[repo]}) not found in {d.parent}")
    return d


def _chronos_predictor(pipe):
    qs = [float(q) for q in pipe.quantiles]
    k = qs.index(0.5)

    def predict(target, po, kn, H):
        w = target.shape[1]
        items = []
        for b in range(len(target)):
            item = {"target": target[b]}
            past = {f"p{i}": po[b, i] for i in range(po.shape[1])} if po is not None else {}
            if kn is not None:
                past |= {f"k{i}": kn[b, i, :w] for i in range(kn.shape[1])}
                item["future_covariates"] = {f"k{i}": kn[b, i, w:] for i in range(kn.shape[1])}
            if past:
                item["past_covariates"] = past
            items.append(item)
        # cross_learning=False (also the library default): every origin is its own group, so no origin can
        # attend to another one's context, which would contain its targets
        res = pipe.predict(items, prediction_length=H, cross_learning=False)
        return np.stack([r[0, k].float().cpu().numpy() for r in res])
    return predict


def _tsfm_load(name: str):
    """Returns (predict, n_params). predict(target (B, w), past_only (B, n, w) | None,
    known (B, n, w+H) | None, H) -> (B, H) point forecast = median of the predicted quantiles."""
    import torch

    def n_params(obj):
        for cand in (obj, getattr(obj, "model", None)):
            if isinstance(cand, torch.nn.Module):
                return int(sum(p.numel() for p in cand.parameters()))
        return None

    def with_dtype(predict, obj):                           # the parameter dtype actually loaded
        for cand in (obj, getattr(obj, "model", None)):
            if isinstance(cand, torch.nn.Module):
                predict.param_dtype = str(next(cand.parameters()).dtype)
        return predict

    path = str(hf_snapshot(name))
    if name == "Chronos2":
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained(path, device_map="cuda")
        return with_dtype(_chronos_predictor(pipe), pipe), n_params(pipe)
    if name == "TimesFM3":
        import timesfm
        m = timesfm.TimesFM3Forecaster.from_pretrained(path, device="cuda")

        def predict(target, po, kn, H):
            res = list(m.predict_batch(list(target), horizon=H,
                                       past_only_covariates=list(po) if po is not None else None,
                                       past_future_covariates=list(kn) if kn is not None else None))
            return np.stack([np.asarray(r.forecast, dtype=np.float32).reshape(-1)[:H] for r in res])
        return with_dtype(predict, m), n_params(m)
    if name == "TiRex2":
        import tirex2
        import tirex2.model.component.flashrnn_slstm as fs
        # The fused sLSTM CUDA kernels need a C++/CUDA compiler toolchain; the library's own pure-PyTorch
        # kernels give the same model output and run on the GPU without compilation.
        fs._flashrnn_backend = lambda device: "vanilla"
        m = tirex2.load_model(path, device="cuda")
        q = m.quantiles.detach().float().cpu().numpy()
        k = int(np.argmin(np.abs(q - 0.5)))
        if abs(q[k] - 0.5) > 1e-6:
            raise RuntimeError("TiRex-2 has no 0.5 quantile")

        def predict(target, po, kn, H):
            items = [tirex2.TimeseriesType(
                target=torch.from_numpy(np.ascontiguousarray(target[b][None, :])),          # (1 target, w)
                past_covariates=torch.from_numpy(np.ascontiguousarray(po[b])) if po is not None else None,
                future_covariates=torch.from_numpy(np.ascontiguousarray(kn[b])) if kn is not None else None)
                for b in range(len(target))]
            res = m.forecast(items, prediction_length=H, batch_size=len(items))
            res = res if isinstance(res, (list, tuple)) else [res]
            for r in res:
                if tuple(r.shape[-2:]) != (len(q), H):
                    raise RuntimeError(f"unexpected TiRex-2 output shape {tuple(r.shape)}")
            return np.concatenate([r[..., k, :].float().cpu().numpy().reshape(-1, H) for r in res])
        return with_dtype(predict, m), n_params(m)
    raise ValueError(name)


def _tsfm_arrays(D: Dataset, regime: str, variant: str, o: np.ndarray, w: int, H: int):
    """Context and covariates of the foundation models (raw units; the models normalise internally)."""
    ctx = o[:, None] + np.arange(-w, 0)[None, :]
    target = D.y.astype(np.float32)[ctx]
    po = kn = None
    if regime == "cal" or regime in ABLATION_GROUPS:
        drop = ABLATION_GROUPS.get(regime, [])
        po = D.calendar[ctx][..., [i for i, c in enumerate(CALENDAR_FEATURES) if c not in drop]]
    elif regime in ("F1", "F1A", "F2", "F34"):
        past_src, future_src, margin = TEMP_SOURCE[variant]
        known = [D.calendar[o[:, None] + np.arange(-w, H)[None, :]]]
        if regime == "F1A":
            known.append(D.analog.channels(o, -w, H))
        if regime == "F2":
            po = D.weather.past_temperature(o, w, past_src)
        if regime == "F34":
            known.append(np.concatenate([D.weather.past_temperature(o, w, past_src),
                                         D.weather.future_temperature(o, np.arange(H), future_src, margin)], axis=1))
        kn = np.concatenate(known, axis=-1)
    elif regime != "uni":
        raise ValueError(f"regime {regime} is not available for foundation models")

    def tr(a):
        return None if a is None else np.ascontiguousarray(a.transpose(0, 2, 1), dtype=np.float32)
    return target, tr(po), tr(kn)


def _tsfm_forecast(predict, D, regime, variant, o, w, H) -> np.ndarray:
    batch = 256 if w <= 720 else 64
    return np.concatenate([predict(*_tsfm_arrays(D, regime, variant, o[i:i + batch], w, H), H)
                           for i in range(0, len(o), batch)])


def run_foundation(spec: dict, D: Dataset, tc: dict, out: Path, log) -> dict:
    """Pretrained models without fine-tuning: a forecast from each context window (the same windows as
    the trained models), on the validation year (window selection, ensemble) and the test year (sealed).
    F regimes use the common F validation origins."""
    H, w, regime = spec["H"], spec["w"], spec["regime"]
    if spec.get("gap"):
        raise ValueError("the publication-gap sensitivity is not implemented for this family")
    if H > TSFM_MAX_HORIZON.get(spec["model"], H):
        raise ValueError(f"{spec['model']} forecasts at most {TSFM_MAX_HORIZON[spec['model']]} steps")
    predict, n_params = _tsfm_load(spec["model"])
    val = D.f_origins("val", H) if regime in F_REGIMES else D.origins("val", H)
    parts = {"val": val, "test": D.origins("test", H)}
    if spec.get("demo"):
        parts = {k: v[::240] for k, v in parts.items()}
    variants = [v for v in F_STREAMS[regime] + list(F_TEST_ONLY.get(regime, {})) if v not in CHECKPOINT_FREE_SKIP] \
        if regime in F_REGIMES else ["main"]
    preds, t_all = {}, time.time()
    info = {"params": n_params, "precision": {"param_dtype": getattr(predict, "param_dtype", None), "autocast": None}}
    for variant in variants:
        for part, o in parts.items():
            t0 = time.time()
            preds[part + variant_key(variant)] = (o, _tsfm_forecast(predict, D, regime, variant, o, w, H))
            info[f"{part}{variant_key(variant)}_inference_sec"] = round(time.time() - t0, 2)
    if H == 168 and regime in ("uni", "cal"):
        o = np.array([D.aggregation_origin()])
        preds["aggregation_extra"] = (o, _tsfm_forecast(predict, D, regime, "main", o, w, H))
    info["inference_sec"] = round(time.time() - t_all, 2)
    repo = HF_REPOS[spec["model"]]
    return {"info": info, "preds": preds, "artefacts": {"files": [], "weights": {repo: HF_REVISIONS[repo]}}}


def run_chronos_lora(spec: dict, D: Dataset, tc: dict, out: Path, log) -> dict:
    """Chronos-2 fine-tuned with LoRA on the training years (past calendar as covariates, like `cal`):
    random windows of 2016-2023, a fixed budget of 2000 updates of 36 windows, validation loss every 100
    steps on 2024 origins spread over every hour of the day (stride 25 h), best checkpoint kept
    (chronos-forecasting's fit()). Forecasts as in the zero-shot runs."""
    from chronos import Chronos2Pipeline
    H, w, regime = spec["H"], spec["w"], spec["regime"]
    if spec.get("gap"):
        raise ValueError("the publication-gap sensitivity is not implemented for this family")
    if regime != "cal":
        raise ValueError("Chronos2_LoRA is defined for the `cal` regime")
    set_seed(spec["seed"])
    pipe = Chronos2Pipeline.from_pretrained(str(hf_snapshot("Chronos2_LoRA")), device_map="cuda")
    y, cal = D.y.astype(np.float32), D.calendar

    def covariates(a, b):
        return {f"p{j}": cal[a:b, j] for j in range(cal.shape[1])}

    train = [{"target": y[:D.train_end], "past_covariates": covariates(0, D.train_end)}]
    val_fit = D.origins("val", H)[::25]                  # 25 h stride: the origin hour rotates through the day
    valid = [{"target": y[t - w:t + H], "past_covariates": covariates(t - w, t + H)} for t in val_fit]
    rows_per_window = 1 + cal.shape[1]                   # Chronos counts target and covariate rows
    steps = 10 if spec.get("demo") else CHRONOS_FT_STEPS
    lr = float(spec.get("lr", CHRONOS_FT_LR))
    accum = 1 if w + H <= 1024 else 4                    # long contexts: the same 36 windows in 4 parts
    rows = CHRONOS_FT_WINDOWS // accum * rows_per_window
    import torch
    saved_precision = (torch.backends.cuda.matmul.fp32_precision, torch.backends.cudnn.conv.fp32_precision)
    t0 = time.time()
    ft = pipe.fit(train, prediction_length=H, validation_inputs=valid, finetune_mode="lora", lora_config=dict(CHRONOS_LORA),
                  context_length=w, learning_rate=lr, num_steps=steps, batch_size=rows,
                  gradient_accumulation_steps=accum, output_dir=out / "chronos_lora", min_past=w,
                  remove_printer_callback=True, seed=spec["seed"], data_seed=spec["seed"],
                  report_to="none", disable_tqdm=True)
    fit_sec = time.time() - t0
    lora_params = int(sum(p.numel() for n, p in ft.model.named_parameters() if "lora_" in n))
    del ft, pipe
    torch.cuda.empty_cache()
    # fit() leaves TF32 matrix products switched on. Forecast with the settings the zero-shot runs use and
    # with the model reloaded from its saved checkpoint, so the stored forecasts are exactly those of the
    # artefact (under TF32 the in-memory unmerged adapter and the saved merged one differed by up to 0.4 %).
    torch.backends.cuda.matmul.fp32_precision, torch.backends.cudnn.conv.fp32_precision = saved_precision
    ft = Chronos2Pipeline.from_pretrained(str(out / "chronos_lora" / "finetuned-ckpt"), device_map="cuda")
    predict = _chronos_predictor(ft)
    parts = {"val": D.origins("val", H), "test": D.origins("test", H)}
    if spec.get("demo"):
        parts = {k: v[::240] for k, v in parts.items()}
    preds = {part: (o, _tsfm_forecast(predict, D, regime, "main", o, w, H)) for part, o in parts.items()}
    if H == 168:
        o = np.array([D.aggregation_origin()])
        preds["aggregation_extra"] = (o, _tsfm_forecast(predict, D, regime, "main", o, w, H))
    info = {"params": int(sum(p.numel() for p in ft.model.parameters())), "lora_params": lora_params or None,
            "fit_sec": round(fit_sec, 2), "lora_lr": lr, "steps": steps, "windows_per_update": CHRONOS_FT_WINDOWS,
            "rows_per_micro_batch": rows, "gradient_accumulation": accum, "lora_config": CHRONOS_LORA,
            "validation_origins": int(len(val_fit)),
            "precision": {"fine_tuning": "bf16 autocast" if torch.cuda.get_device_capability()[0] >= 8 else "fp32",
                          "inference_param_dtype": str(next(ft.model.parameters()).dtype),
                          "inference_fp32_matmul": torch.backends.cuda.matmul.fp32_precision,
                          "inference_fp32_conv": torch.backends.cudnn.conv.fp32_precision}}
    repo = HF_REPOS["Chronos2_LoRA"]
    return {"info": info, "preds": preds, "artefacts": {"files": ["chronos_lora"], "weights": {repo: HF_REVISIONS[repo]}}}


LGB_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]
LGB_PARAMS = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, objective="l2", metric="l2", verbose=-1)


def _lgb_base(D: Dataset, o: np.ndarray) -> np.ndarray:
    """Features known at the origin: recent and weekly lags, rolling means, calendar of the last input hour."""
    y = D.y
    cols = [y[o - l] for l in LGB_LAGS]
    cs = np.concatenate([[0.0], np.cumsum(y)])
    cols += [(cs[o] - cs[o - r]) / r for r in (24, 168)]
    cols += list(D.calendar[o - 1].T)
    return np.column_stack(cols).astype(np.float32)


def _lgb_feature_names(regime: str) -> list[str]:
    names = [f"lag_{l}" for l in LGB_LAGS] + ["mean_24h", "mean_168h"] + [f"{c}_last_input" for c in CALENDAR_FEATURES]
    if regime in ("F1", "F1A0", "F1A", "F2", "F34"):
        names += [f"{c}_target" for c in CALENDAR_FEATURES]
    if regime == "F1A0":
        names += ["analog_flag_target"]
    if regime == "F1A":
        names += ["analog_dev_target", "analog_flag_target"]
    if regime in ("F2", "F34"):
        names += [f"{c}_last_input" for c in TEMP_FEATURES] + [f"{c}_mean_24h" for c in TEMP_FEATURES]
    if regime == "F34":
        names += [f"{c}_target" for c in TEMP_FEATURES]
    return names


def _lgb_lead(D: Dataset, o: np.ndarray, k: int, regime: str, variant: str, past_t) -> np.ndarray:
    """Extra features of the direct model for lead k in the F regimes (empty in `cal` and F0)."""
    cols = []
    if regime in ("F1", "F1A0", "F1A", "F2", "F34"):
        cols.append(D.calendar[o + k])
    if regime in ("F1A0", "F1A"):
        a = D.analog.channels(o, k, k + 1)[:, 0, :]
        cols.append(a[:, 1:] if regime == "F1A0" else a)
    if regime in ("F2", "F34"):
        cols += [past_t[:, -1, :], past_t.mean(axis=1)]           # last input hour, mean of the last day
    if regime == "F34":
        _, future_src, margin = TEMP_SOURCE[variant]
        cols.append(D.weather.future_temperature(o, np.array([k]), future_src, margin)[:, 0, :])
    return np.column_stack(cols).astype(np.float32) if cols else np.zeros((len(o), 0), np.float32)


def _early_stop_all(patience: int, metric: str = "l2"):
    """LightGBM callback: remember each validation set's best round of `metric` (one fit, separate model
    selection per evaluation variant, like F34) and stop when the first set, the control stream as for the
    neural networks, has not improved for `patience` rounds."""
    import lightgbm as lgb
    best, first = {}, []

    def callback(env):
        seen = set()
        for data_name, eval_name, value, higher_better in env.evaluation_result_list:
            if eval_name != metric or higher_better:
                raise RuntimeError(f"unexpected evaluation metric {eval_name}")
            if data_name in seen:
                raise RuntimeError(f"more than one metric for {data_name}")
            seen.add(data_name)
            if not first:
                first.append(data_name)
            if data_name not in best or value < best[data_name][0]:
                best[data_name] = (value, env.iteration + 1)
        if env.iteration + 1 - best[first[0]][1] >= patience:
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
    callback.order = 30
    return callback, best


def run_lightgbm(spec: dict, D: Dataset, tc: dict, out: Path | None, log) -> dict:
    """Direct strategy: one LightGBM model per lead k, fed with features known at the origin (recent lags,
    weekly lags, rolling means, calendar of the last input hour) and, in the F regimes, the target-hour
    calendar, holiday analog and temperature. L2 objective; early stopping on the validation year
    (50 rounds). Every booster is saved (gzip text) with the best round of every stream."""
    import lightgbm as lgb
    H, regime = spec["H"], spec["regime"]
    if regime != "cal" and regime not in F_REGIMES:
        raise ValueError(f"regime {regime} is not available for LightGBM")
    features = "cal" if regime == "F0" else regime      # F0: no future input, F validation origins
    is_f = regime in F_REGIMES
    gap = spec.get("gap", 0)                            # publication delay: origin features from t - gap
    if gap and is_f:
        raise ValueError("the publication-gap sensitivity is defined for past-only regimes")
    tr = np.arange(336 + gap, D.train_end - H + 1)
    val = D.f_origins("val", H) if is_f else D.origins("val", H)
    te = D.origins("test", H)
    H_fit = H
    if spec.get("demo"):
        tr, H_fit = tr[-4000:], min(H, 3)
    streams = F_STREAMS[regime] if is_f else ["main"]
    ckpt_of = {s: s for s in streams} | F_TEST_ONLY.get(regime, {})
    base = {"train": _lgb_base(D, tr - gap), "val": _lgb_base(D, val - gap), "test": _lgb_base(D, te - gap)}
    temp = features in ("F2", "F34")
    past_t = {("train", "train"): D.weather.past_temperature(tr, 24, "era5")} if temp else {}
    for v in ckpt_of if temp else []:
        for part, o in (("val", val), ("test", te)):
            past_t[(part, v)] = D.weather.past_temperature(o, 24, TEMP_SOURCE[v][0])
    origins = {"train": tr, "val": val, "test": te}

    def X(part, k, v):
        return np.hstack([base[part], _lgb_lead(D, origins[part], k, features, v, past_t.get((part, v)))])

    preds = {f"{part}{variant_key(v)}": np.zeros((len(origins[part]), H)) for v in ckpt_of for part in ("val", "test")}
    best_rounds = {v: [] for v in streams}
    agg = np.array([D.aggregation_origin()]) if H == 168 and regime == "cal" and not gap else None
    pa = np.zeros((1, H))
    boosters = []
    t0 = time.time()
    for k in range(H_fit):
        Xva = {s: X("val", k, s) for s in streams}
        mdl = lgb.LGBMRegressor(**LGB_PARAMS, random_state=spec["seed"], n_jobs=LGB_THREADS)
        cb, best = _early_stop_all(50)
        mdl.fit(X("train", k, "train"), D.y[tr + k], eval_X=tuple(Xva[s] for s in streams),
                eval_y=tuple(D.y[val + k] for _ in streams), eval_names=streams, callbacks=[cb])
        for s in streams:
            best_rounds[s].append(best[s][1])
        for v, ck in ckpt_of.items():
            for part in ("val", "test"):
                Xp = Xva[v] if part == "val" and v in Xva else X(part, k, v)
                preds[f"{part}{variant_key(v)}"][:, k] = mdl.booster_.predict(Xp, num_iteration=best[ck][1])
        if agg is not None:
            pa[0, k] = mdl.booster_.predict(_lgb_base(D, agg), num_iteration=best["main"][1])[0]
        boosters.append(mdl.booster_.model_to_string(num_iteration=max(best[s][1] for s in streams)))
    if H_fit < H:  # demo only: fill the remaining leads to keep array shapes
        for p in list(preds.values()) + [pa]:
            p[:, H_fit:] = p[:, [H_fit - 1]]
    out_preds = {key: (origins[key.partition("@")[0]], p) for key, p in preds.items()}
    if agg is not None:
        out_preds["aggregation_extra"] = (agg, pa)
    files = []
    if out is not None:
        with gzip.open(out / "lightgbm_models.json.gz", "wt", encoding="utf-8") as f:
            json.dump({"lead_models": boosters, "best_rounds": best_rounds, "feature_names": _lgb_feature_names(features)}, f)
        files = ["lightgbm_models.json.gz"]
    return {"info": {"params": None, "fit_sec": round(time.time() - t0, 2), "lags": "1-336 h",
                     "best_rounds": best_rounds, "lgb_params": LGB_PARAMS, "threads": LGB_THREADS,
                     "precision": "float64 CPU"},
            "preds": out_preds, "artefacts": {"files": files, "feature_order": _lgb_feature_names(features)}}


def _vanilla_design(D: Dataset, T: np.ndarray | None, holiday: bool = False) -> np.ndarray:
    """Hong's Vanilla benchmark (GEFCom2012): trend + month + weekday x hour [+ T, T^2, T^3, each also
    interacted with month and with hour]; `holiday` adds a holiday dummy (Vanilla_holiday, our addition).
    T is the population-weighted temperature, centred and scaled for numerical conditioning only."""
    cols = [np.ones(D.n), np.arange(D.n) / 8766.0]
    cols += [(D.month == m).astype(float) for m in range(2, 13)]
    wh = D.dow * 24 + D.hour
    cols += [(wh == c).astype(float) for c in range(1, 168)]
    if holiday:
        cols.append(D.is_holiday.astype(float))
    Xc = np.column_stack(cols)
    return Xc if T is None else np.hstack([Xc, _vanilla_temp_terms(D, T)])


def _vanilla_temp_terms(D: Dataset, T: np.ndarray) -> np.ndarray:
    z = (T - 15.0) / 10.0
    month = [(D.month == m).astype(float) for m in range(2, 13)]
    hour = [(D.hour == h).astype(float) for h in range(1, 24)]
    cols = []
    for p in (1, 2, 3):
        zp = z ** p
        cols += [zp] + [zp * d for d in month] + [zp * d for d in hour]
    return np.column_stack(cols)


def vanilla_forecast(D: Dataset, beta: np.ndarray, regime: str, o: np.ndarray, H: int, variant: str,
                     base: np.ndarray | None = None, holiday: bool = False) -> np.ndarray:
    """Forecasts of fitted coefficients for origins o (also used to check the saved coefficients)."""
    Xc = _vanilla_design(D, None, holiday) if base is None else None
    n_cal = Xc.shape[1] if Xc is not None else len(beta) - (105 if regime == "F34" else 0)
    base = Xc @ beta[:n_cal] if base is None else base
    tau = o[:, None] + np.arange(H)[None, :]
    p = base[tau]
    if regime == "F34":
        bt = beta[n_cal:].reshape(3, 1 + 11 + 23)
        _, future_src, margin = TEMP_SOURCE[variant]
        z = (D.weather.future_temperature(o, np.arange(H), future_src, margin)[..., 0] - 15.0) / 10.0
        for q in range(3):
            c = np.full(D.n, bt[q, 0])
            c += np.where(D.month >= 2, bt[q, 1:12][np.clip(D.month - 2, 0, 10)], 0.0)
            c += np.where(D.hour >= 1, bt[q, 12:][np.clip(D.hour - 1, 0, 22)], 0.0)
            p = p + z ** (q + 1) * c[tau]
    return p


def run_vanilla(spec: dict, D: Dataset, tc: dict, out: Path | None, log) -> dict:
    """Ordinary least squares on 2016-2023. No load lags: the forecast of an hour depends only on its
    calendar (F1, the temperature-free ablation) and temperature (F34), so it shows what calendar and
    weather alone explain."""
    H, regime = spec["H"], spec["regime"]
    if regime not in ("F1", "F34"):
        raise ValueError("Vanilla is defined for F1 (calendar) and F34 (calendar + temperature)")
    if spec.get("gap"):
        raise ValueError("the publication-gap sensitivity is not implemented for this family")
    holiday = spec["model"] == "Vanilla_holiday"
    Xc = _vanilla_design(D, None, holiday)
    X = Xc if regime == "F1" else np.hstack([Xc, _vanilla_temp_terms(D, D.temp[:, 0])])
    tr = np.arange(D.train_end)
    beta, *_ = np.linalg.lstsq(X[tr], D.y[tr], rcond=None)
    base = Xc @ beta[:Xc.shape[1]]
    val, te = D.f_origins("val", H), D.origins("test", H)
    variants = ["F3", "F3raw", "F4", "F3_m24"] if regime == "F34" else ["main"]
    preds = {part + variant_key(v): (o, vanilla_forecast(D, beta, regime, o, H, v, base, holiday))
             for v in variants for part, o in (("val", val), ("test", te))}
    files = []
    if out is not None:
        np.save(out / "vanilla_beta.npy", beta)
        files = ["vanilla_beta.npy"]
    return {"info": {"params": int(len(beta)), "fit": "OLS 2016-2023", "precision": "float64 CPU"}, "preds": preds,
            "artefacts": {"files": files, "design": "_vanilla_design in run_experiments.py"}}


RUNNERS = {"torch": run_torch, "baseline": run_baseline, "nf": run_neuralforecast, "foundation": run_foundation,
           "foundation_ft": run_chronos_lora, "lightgbm": run_lightgbm, "vanilla": run_vanilla}
# nominal precision of each family, used only to label profile rows (runs record their effective settings)
PRECISION = {"torch": "fp16 autocast training, fp32 inference", "nf": "16-mixed training (Lightning)",
             "foundation": "model default (fp32)", "foundation_ft": "bf16 fine-tuning on sm80+, fp32 inference",
             "lightgbm": "float64 CPU", "vanilla": "float64 CPU", "baseline": "float64 CPU"}


def family(model: str) -> str:
    if model in BASELINE_MODELS:
        return "baseline"
    if model.startswith("NF_"):
        return "nf"
    if model in TSFM_MODELS:
        return "foundation"
    if model == "Chronos2_LoRA":
        return "foundation_ft"
    if model == "LightGBM":
        return "lightgbm"
    if model in ("Vanilla", "Vanilla_holiday"):
        return "vanilla"
    return "torch"


# ==============================================================================================
# 4. Run bookkeeping: contract, identity, sealed outputs, one-process-per-run, selection, freeze
# ==============================================================================================
# Code changes are detected automatically: a run's identity contains a fingerprint of exactly the code
# that determines its numbers (code_fingerprint), so an edit of a comment, a docstring or of bookkeeping
# code keeps finished runs valid, while a numerical change (say in LGB_PARAMS or Weather) invalidates the
# affected families only. Bump PROTOCOL_VERSION to invalidate every run on purpose.
PROTOCOL_VERSION = "2026-09-28.3"
FAMILY_PACKAGES = {         # libraries whose versions enter a family's identity (local build tags dropped)
    "torch": ["torch", "numpy", "pandas"],
    "nf": ["torch", "numpy", "pandas", "neuralforecast", "pytorch-lightning"],
    "foundation": ["torch", "numpy", "pandas", "chronos-forecasting", "timesfm", "tirex-2", "transformers"],
    "foundation_ft": ["torch", "numpy", "pandas", "chronos-forecasting", "transformers", "peft", "accelerate"],
    "lightgbm": ["numpy", "pandas", "lightgbm"],
    "vanilla": ["numpy", "pandas"],
    "baseline": ["numpy", "pandas"],
}
CORE_FILES = ("contract.json", "metrics.json", "val_predictions.npz", "test_sealed.npz")


def code_hash() -> str:
    """Raw hash of the three scripts, stored in every run for auditing (not part of the identity)."""
    h = hashlib.sha256()
    for f in ("run_experiments.py", "models.py", "prepare_data.py"):
        h.update((ROOT / f).read_bytes())
    return h.hexdigest()[:12]


def _normalized_definitions(path: Path) -> dict:
    """Module-level definitions of a source file -> (normalised AST dump without docstrings, comments or
    layout; module-level names it refers to; names it imports from prepare_data/models)."""
    import ast
    import copy

    def strip(node):
        for n in ast.walk(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and n.body \
                    and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) \
                    and isinstance(n.body[0].value.value, str):
                n.body = n.body[1:] or [ast.Pass()]
        return node

    def bound(target) -> list[str]:
        """Module-level names an assignment target binds or modifies (A; A, B = ...; A.x = ...; A[k] = ...)."""
        if isinstance(target, (ast.Tuple, ast.List)):
            return [n for t in target.elts for n in bound(t)]
        if isinstance(target, ast.Starred):
            return bound(target.value)
        while isinstance(target, (ast.Attribute, ast.Subscript)):
            target = target.value
        return [target.id] if isinstance(target, ast.Name) else []

    defs = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            names = [n for t in (node.targets if isinstance(node, ast.Assign) else [node.target]) for n in bound(t)]
        else:
            continue                    # imports, the docstring, environment settings, the __main__ guard
        dump = ast.dump(strip(copy.deepcopy(node)), include_attributes=False)
        refs = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        imports = {(i.module, a.name) for i in ast.walk(node) if isinstance(i, ast.ImportFrom)
                   and i.module in ("prepare_data", "models") for a in i.names}
        for name in names:              # a later statement on the same name adds to its definition
            old = defs.get(name, ("", set(), set()))
            defs[name] = (old[0] + dump, old[1] | refs, old[2] | imports)
    return defs


def code_fingerprint(fam: str, root: Path = ROOT) -> str:
    """Hash of the code that determines a family's numbers: starting from its runner, Dataset and the
    metrics, every module-level definition they use (transitively, across run_experiments.py and
    prepare_data.py; models.py as a whole when it is imported), in normalised form."""
    key = (fam, str(root))
    if key not in _FINGERPRINTS:
        src = {"run_experiments": _normalized_definitions(root / "run_experiments.py"),
               "prepare_data": _normalized_definitions(root / "prepare_data.py"),
               "models": _normalized_definitions(root / "models.py")}
        stack = [("run_experiments", n) for n in (RUNNERS[fam].__name__, "Dataset", "metrics", "variant_key")]
        seen = set()
        while stack:
            mod, name = stack.pop()
            if (mod, name) in seen or name not in src[mod]:
                continue                                   # builtins, locals and library names
            seen.add((mod, name))
            _, refs, imports = src[mod][name]
            stack += [(mod, r) for r in refs] + [(m, n) for m, n in imports if m in src]
        if any(m == "models" for m, _ in seen):
            seen |= {("models", n) for n in src["models"]}
        h = hashlib.sha256()
        for mod, name in sorted(seen):
            h.update(f"{mod}.{name}\n{src[mod][name][0]}\n".encode())
        _FINGERPRINTS[key] = h.hexdigest()[:16]
    return _FINGERPRINTS[key]


_FINGERPRINTS: dict = {}


def package_versions(fam: str) -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for p in FAMILY_PACKAGES[fam]:
        try:
            out[p] = version(p).split("+")[0]              # "2.11.0+cu128" -> "2.11.0": the build tag is platform detail
        except PackageNotFoundError:
            out[p] = None
    return out


def data_hashes() -> dict:
    """Hashes of every input a run may consume, verified against the build report."""
    report = data_report()
    load = report["outputs_sha256"]["tr_load_hourly.csv"]
    if sha256_file(DATA_FILE) != load:
        sys.exit(f"{DATA_FILE.name} differs from data_report.json; rebuild the data")
    return {"load": load, "weather": report.get("outputs_content_sha256", {}).get(PROVINCE_FILE.name)}


def make_contract(spec: dict, cfg: dict, hashes: dict) -> dict:
    """Everything that determines a run's numbers. Its hash is the run identity; the child process
    receives it verbatim instead of re-reading files that could change meanwhile."""
    fam = family(spec["model"])
    s = {k: v for k, v in spec.items() if k != "demo"}
    training = None
    if fam in ("torch", "nf"):
        training = dict(cfg["training"])
        if "lr" in spec:
            training["learning_rate"] = float(spec["lr"])
        if spec.get("demo"):
            training.update(max_epochs=2, early_stopping_patience=2)
    data = {"load": hashes["load"]}
    if spec["regime"] in F_REGIMES + ("F4",):
        data["weather"] = hashes["weather"]
    code = {"protocol": PROTOCOL_VERSION, "fingerprint": code_fingerprint(fam), "packages": package_versions(fam)}
    weights = {HF_REPOS[spec["model"]]: HF_REVISIONS[HF_REPOS[spec["model"]]]} if spec["model"] in HF_REPOS else None
    return {"spec": s, "demo": bool(spec.get("demo")), "training": training, "data": data, "code": code,
            "weights": weights}


def run_id(contract: dict) -> str:
    s = contract["spec"]
    body = {k: v for k, v in contract.items() if k != "run_id"}
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:10]
    gap = f"_gap{s['gap']}" if s.get("gap") else ""
    return f"{s['model']}_{s['regime']}_H{s['H']}_w{s['w']}{gap}_s{s['seed']}_{h}"


def environment() -> dict:
    import torch
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for p in ("numpy", "pandas", "neuralforecast", "pytorch-lightning", "chronos-forecasting", "timesfm", "tirex-2",
              "lightgbm", "peft", "transformers"):
        try:
            packages[p] = version(p)
        except PackageNotFoundError:
            pass
    try:        # torch >= 2.9 API; libraries that use it (Chronos-2 fine-tuning) make the legacy flag unreadable
        tf32 = str(torch.backends.cuda.matmul.fp32_precision)
    except Exception:
        tf32 = str(torch.backends.cuda.matmul.allow_tf32)
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "cpu",
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "matmul_fp32_precision": tf32,
            "conv_fp32_precision": str(torch.backends.cudnn.conv.fp32_precision), "packages": packages}


def _hash_path(p: Path) -> str:
    """sha256 of a file, or of every file below a folder (relative paths and contents). A missing or empty
    path is an error, never an 'empty' hash. Files are ordered by their case-sensitive path components, so
    a folder hashes the same on Linux and Windows (Path objects compare case-insensitively on Windows)."""
    if p.is_file():
        return sha256_file(p)
    if not p.is_dir():
        raise FileNotFoundError(p)
    files = sorted((q for q in p.rglob("*") if q.is_file()), key=lambda q: q.relative_to(p).parts)
    if not files:
        raise FileNotFoundError(f"{p} is empty")
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(p)).replace("\\", "/").encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def _path_bytes(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    if not p.is_dir():
        raise FileNotFoundError(p)
    return sum(q.stat().st_size for q in p.rglob("*") if q.is_file())


def verify_run(folder: Path, full: bool = False, expected_id: str | None = None) -> bool:
    """The single acceptance test of a run folder (resume, selection, collection, evaluation, freeze).
    Valid only if DONE exists, the manifest lists the core files, the contract names this run, every
    listed file exists with its recorded size, and the core files (with full=True every file, including
    checkpoints and saved models) match their sha256."""
    try:
        if not (folder / "DONE").exists():
            return False
        manifest = json.loads((folder / "manifest.json").read_text())
        if not manifest or any(f not in manifest for f in CORE_FILES):
            return False
        if json.loads((folder / "contract.json").read_text()).get("run_id") != (expected_id or folder.name):
            return False
        for f, rec in manifest.items():
            if _path_bytes(folder / f) != rec["bytes"]:
                return False
            if (full or f in CORE_FILES) and _hash_path(folder / f) != rec["sha256"]:
                return False
        return True
    except Exception:
        return False


def execute_one(contract: dict, out: Path) -> None:
    """Runs inside the child process, in an attempt folder. Writes validation forecasts and metrics, the
    sealed test forecasts (no test metric is computed here), artefacts and a checksum manifest; the DONE
    marker comes last. The parent publishes the folder only after verifying it."""
    t_start, started = time.time(), datetime.now(timezone.utc).isoformat(timespec="seconds")
    spec = dict(contract["spec"], demo=contract["demo"])
    fam = family(spec["model"])
    now = {"protocol": PROTOCOL_VERSION, "fingerprint": code_fingerprint(fam), "packages": package_versions(fam)}
    if contract["code"] != now:
        raise RuntimeError("the code or the packages changed since this run was launched; launch it again")
    tc = contract["training"]
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(msg):
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    D = Dataset()
    hashes = data_hashes()
    for k, v in contract["data"].items():
        if hashes.get(k) != v:
            raise RuntimeError(f"data '{k}' changed since the run was launched")
    log(f"start {spec}")
    res = RUNNERS[fam](spec, D, contract["training"], out, log)
    H = spec["H"]
    record = {"spec": spec, "run_id": contract.get("run_id"), "contract": contract, "code_hash": code_hash(),
              "environment": environment(), "started_utc": started,
              "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              **{k: v for k, v in res["info"].items() if k != "history"}}
    try:
        import torch
        if torch.cuda.device_count() > 0:        # allocated by tensors / held by the caching allocator
            record["peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
            record["peak_gpu_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)
    except Exception:
        pass
    val_arrays, test_arrays = {}, {}
    for key, (o, p) in res["preds"].items():
        p = np.asarray(p, dtype=np.float64)
        if p.shape != (len(o), H) or not np.isfinite(p).all():
            raise ValueError(f"{key}: forecasts have shape {p.shape} or non-finite values")
        part = key.partition("@")[0]
        target = val_arrays if part == "val" else test_arrays          # test and the aggregation origin: sealed
        target[f"{key}_origin_utc_ns"] = D.utc_ns[o]
        target[f"{key}_forecast_mwh"] = p32 = p.astype(np.float32)
        if part == "val":        # scored on the stored float32 values, so metrics are exactly reproducible
            record[f"{key}_metrics"] = metrics(D.targets(o, H), p32.astype(np.float64), D.y, D.train_end, H)
    record["sealed_test_keys"] = sorted(k.removesuffix("_forecast_mwh") for k in test_arrays if k.endswith("_forecast_mwh"))
    np.savez_compressed(out / "val_predictions.npz", **val_arrays)
    np.savez_compressed(out / "test_sealed.npz", **test_arrays)
    if "history" in res["info"]:
        (out / "history.json").write_text(json.dumps(res["info"]["history"]))
    art = res.get("artefacts", {})
    record["artefacts"] = {k: v for k, v in art.items() if k != "files"}
    record["wall_sec"] = round(time.time() - t_start, 1)              # data, training, forecasting, saving
    (out / "metrics.json").write_text(json.dumps(record, indent=1, default=float))
    files = list(CORE_FILES) + (["history.json"] if "history" in res["info"] else []) + list(art.get("files", []))
    (out / "manifest.json").write_text(json.dumps({f: {"sha256": _hash_path(out / f), "bytes": _path_bytes(out / f)}
                                                   for f in files}, indent=1))
    (out / "DONE").write_text(record["finished_utc"])
    vm = record.get("val_metrics") or next((v for k, v in record.items() if k.startswith("val") and k.endswith("_metrics")), {})
    log(f"done: validation MAPE {vm.get('MAPE', float('nan')):.3f}%  (test forecasts sealed)")
    logf.close()


def load_config() -> dict:
    import yaml
    return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))


def load_selected() -> dict:
    import yaml
    return (yaml.safe_load(SELECTED_FILE.read_text(encoding="utf-8")) or {}) if SELECTED_FILE.exists() else {}


def save_selected(sel: dict) -> None:
    import yaml
    SELECTED_FILE.write_text(yaml.safe_dump(sel, sort_keys=True), encoding="utf-8")


def base_model(model: str) -> str:
    """The P-regime model whose validated choices (window, learning rate) a model inherits."""
    if model.startswith("LightTCN_F_"):
        return "LightTCN_" + model.rsplit("_", 1)[1]
    return {"Chronos2_LoRA": "Chronos2"}.get(model, model)


def base_regime(regime: str) -> str:
    return "uni" if regime == "uni" else "cal"


def expand_experiment(cfg: dict, name: str, demo: bool = False) -> list[dict]:
    """Expand one or more comma-separated experiments. A block may list learning rates (lr: [...]);
    otherwise the learning rate selected on the validation year (configs/selected.yaml) is used. The
    default learning rate is never written into the spec, so identical runs keep identical ids.
    `windows: [selected]` uses the window chosen on the validation year for the model's P counterpart;
    `models: selected_h720_rivals` uses the best recurrent and NeuralForecast models at H = 720."""
    default_lr = float(cfg["training"]["learning_rate"])
    sel = load_selected()
    specs = []
    for exp in name.split(","):
        for block in cfg["experiments"][exp.strip()]["blocks"]:
            lrs = block.get("lr", [None])
            lrs = lrs if isinstance(lrs, list) else [lrs]
            if block["models"] == "selected_h720_rivals":
                pairs = sel.get("h720_rivals") or ([["LSTM", "cal"], ["NF_NHITS", "cal"]] if demo else None)
                if not pairs:
                    sys.exit("no H = 720 rivals selected yet: run `select-windows` after long_horizon and nf_grid")
            else:
                pairs = list(itertools.product(block["models"], block.get("regimes", ["cal"])))
            for (m, reg), H, w, s, lr in itertools.product(pairs, block["horizons"], block["windows"],
                                                           block.get("seeds", [42]), lrs):
                if w == "selected":
                    key = f"{base_model(m)}|{base_regime(reg)}|{int(H)}"
                    w = (sel.get("window") or {}).get(key)
                    if w is None and demo:                  # demo runs before any selection exists
                        w = 168 if int(H) < 720 else 1440
                    if w is None:
                        sys.exit(f"no selected window for {key}: run `select-windows` after the grid experiments")
                spec = {"model": m, "regime": reg, "H": int(H), "w": int(w), "seed": int(s)}
                if block.get("gap"):
                    spec["gap"] = int(block["gap"])
                lr_sel = sel.get("learning_rate") or {}
                lr = lr if lr is not None else lr_sel.get(m, lr_sel.get(base_model(m)))
                if lr is not None and float(lr) != default_lr:
                    spec["lr"] = float(lr)
                specs.append(spec)
    return list({json.dumps(s, sort_keys=True): s for s in specs}.values())


def _candidates(cfg: dict, experiments: str, results: Path) -> pd.DataFrame:
    """Every run an experiment declares, with its current run_id and its validation MAPE (NaN when the
    run is missing, fails verification or belongs to an older contract). Selections refuse to work on an
    incomplete table."""
    hashes = data_hashes()
    rows = []
    for s in expand_experiment(cfg, experiments):
        rid = run_id(make_contract(s, cfg, hashes))
        f = results / "runs" / rid
        mape = np.nan
        if verify_run(f):
            r = json.loads((f / "metrics.json").read_text())
            mape = (r.get("val_metrics") or {}).get("MAPE", np.nan)
        rows.append({**s, "lr": s.get("lr", float(cfg["training"]["learning_rate"])), "run_id": rid, "val_MAPE": mape})
    return pd.DataFrame(rows)


def _require_complete(c: pd.DataFrame, what: str) -> None:
    missing = c[c.val_MAPE.isna()]
    if len(missing):
        sys.exit(f"{what}: {len(missing)} of {len(c)} declared runs are not finished or fail verification, e.g. "
                 + ", ".join(missing.run_id.head(5)))


def select_learning_rates(cfg: dict, results: Path) -> None:
    """For every model of lr_search keep the declared learning rate with the lower validation MAPE (2024,
    reference cell H = 24, w = 168, seed 42). Every declared candidate must be finished and verified; the
    test year is never looked at."""
    c = _candidates(cfg, "lr_search", results)
    _require_complete(c, "select-lr")
    chosen = {m: float(g.loc[g.val_MAPE.idxmin(), "lr"]) for m, g in c.groupby("model")}
    sel = load_selected()
    sel["learning_rate"] = chosen
    sel.setdefault("manifest", {})["learning_rate"] = {"candidates": c.run_id.tolist(), "protocol": PROTOCOL_VERSION,
                                                       "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    save_selected(sel)
    print(f"selected learning rates -> {SELECTED_FILE}:", chosen)


SELECTION_EXPERIMENTS = "grid,long_horizon,nf_grid,foundation"


def select_windows(cfg: dict, results: Path) -> None:
    """For every (model, regime, horizon) keep the declared window with the lowest validation MAPE (seed 42,
    the model's selected learning rate), and pick the H = 720 panel rivals of LightTCN: the best recurrent
    and the best NeuralForecast model on the validation year. Every declared candidate must be finished."""
    if not load_selected().get("learning_rate"):
        sys.exit("run select-lr first")
    c = _candidates(cfg, SELECTION_EXPERIMENTS, results)
    _require_complete(c, "select-windows")
    windows = {f"{m}|{reg}|{H}": int(g.loc[g.val_MAPE.idxmin(), "w"]) for (m, reg, H), g in c.groupby(["model", "regime", "H"])}
    h = c[c.H == 720]
    h = h[[windows.get(f"{m}|{r}|720") == w for m, r, w in zip(h.model, h.regime, h.w)]]
    rivals = []
    for label, group in (("recurrent", h[h.model.isin(RNN_MODELS)]), ("NeuralForecast", h[h.model.str.startswith("NF_")])):
        if not len(group):
            sys.exit(f"no {label} model at H = 720 among the declared candidates")
        b = group.loc[group.val_MAPE.idxmin()]
        rivals.append([str(b.model), str(b.regime)])
    sel = load_selected()
    sel["window"], sel["h720_rivals"] = windows, rivals
    sel.setdefault("manifest", {})["window"] = {"experiments": SELECTION_EXPERIMENTS, "candidates": c.run_id.tolist(),
                                                "protocol": PROTOCOL_VERSION,
                                                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    save_selected(sel)
    print(f"selected {len(windows)} windows and H=720 rivals {rivals} -> {SELECTED_FILE}")


def freeze_state() -> dict:
    """What must not change between `freeze` and `evaluate`."""
    files = [CONFIG_FILE, SELECTED_FILE, ROOT / "run_experiments.py", ROOT / "models.py", ROOT / "prepare_data.py"]
    return {f.name: sha256_file(f) for f in files if f.exists()} | {"data": data_hashes()}


def _frozen() -> dict | None:
    return json.loads(FROZEN_FILE.read_text()) if FROZEN_FILE.exists() else None


def freeze(cfg: dict, results: Path) -> None:
    """Seal the design before any test error exists: every selection is made and every run of the campaign
    (configs/experiments.yaml, key `campaign`) is present, current and fully verified. The freeze records
    the design hashes and the manifest hash of every run; `evaluate` refuses anything else."""
    sel = load_selected()
    missing = [k for k in ("learning_rate", "window", "h720_rivals") if not sel.get(k)]
    if missing:
        sys.exit(f"cannot freeze: selections missing {missing}")
    hashes, runs, bad = data_hashes(), {}, []
    specs = expand_experiment(cfg, ",".join(cfg["campaign"]))
    for s in specs:
        rid = run_id(make_contract(s, cfg, hashes))
        f = results / "runs" / rid
        if verify_run(f, full=True):
            runs[rid] = sha256_file(f / "manifest.json")
        else:
            bad.append(rid)
    if bad:
        sys.exit(f"cannot freeze: {len(bad)} of {len(specs)} campaign runs are missing, stale or fail verification, "
                 f"e.g. {bad[:5]}")
    state = freeze_state()
    frozen = {"state": state, "runs": runs, "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "freeze_id": hashlib.sha256(json.dumps({"state": state, "runs": runs}, sort_keys=True).encode()).hexdigest()[:16]}
    old = _frozen()
    history = (old.get("history", []) + [{k: old[k] for k in ("freeze_id", "frozen_utc")}]) if old else []
    FROZEN_FILE.write_text(json.dumps({**frozen, "history": history}, indent=1))
    print(f"design frozen ({len(runs)} runs, freeze_id {frozen['freeze_id']}) -> {FROZEN_FILE}")


def evaluate(specs: list[dict], cfg: dict, results: Path, demo: bool) -> int:
    """Score the sealed test forecasts. Outside --demo this requires `freeze`, an unchanged design and runs
    whose manifests are the frozen ones; every test metric carries the freeze_id it was computed under."""
    frozen = None
    if not demo:
        frozen = _frozen()
        if frozen is None:
            sys.exit("run `freeze` first: test errors are computed only after every selection is final")
        if frozen["state"] != freeze_state():
            sys.exit("configuration, selections, code or data changed after `freeze`; re-freeze deliberately")
    D, hashes, missing = Dataset(), data_hashes(), []
    for s in specs:
        s = dict(s, demo=demo)
        f = results / "runs" / run_id(make_contract(s, cfg, hashes))
        if not verify_run(f, full=True) or (frozen and frozen["runs"].get(f.name) != sha256_file(f / "manifest.json")):
            missing.append(f.name)
            continue
        z = np.load(f / "test_sealed.npz")
        out = {}
        for key in json.loads((f / "metrics.json").read_text())["sealed_test_keys"]:
            if key.startswith("test"):
                o = np.searchsorted(D.utc_ns, z[f"{key}_origin_utc_ns"])
                full = len(o) == len(D.origins("test", s["H"]))
                out[f"{key}_metrics" + ("" if full else "_subset")] = metrics(
                    D.targets(o, s["H"]), z[f"{key}_forecast_mwh"].astype(np.float64), D.y, D.train_end, s["H"])
        out["freeze_id"] = frozen["freeze_id"] if frozen else "demo"
        out["evaluated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tmp = f / "test_metrics.json.tmp"
        tmp.write_text(json.dumps(out, indent=1))
        os.replace(tmp, f / "test_metrics.json")
    print(f"evaluated {len(specs) - len(missing)} runs" + (f"; missing, unverified or not frozen: {missing[:5]}" if missing else ""))
    collect(results, cfg)
    return 1 if missing else 0


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))    # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259                                               # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _write_lock(f: Path, child: int | None = None) -> None:
    f.write_text(json.dumps({"pid": os.getpid(), "child": child, "host": platform.node(),
                             "utc": datetime.now(timezone.utc).isoformat()}))


def _lock(lock_dir: Path, rid: str) -> bool:
    """One launcher per run. A lock is stale only when it was written on this machine and neither the
    launcher nor its child process is alive; exactly one launcher can take a stale lock over (atomic
    rename), then the new lock is created exclusively."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    f = lock_dir / f"{rid}.lock"
    if f.exists():
        try:
            info = json.loads(f.read_text())
        except Exception:
            info = {}
        if info.get("host") != platform.node() or any(_pid_alive(info.get(k)) for k in ("pid", "child")):
            return False
        stale = lock_dir / f"{rid}.stale.{os.getpid()}.{time.time_ns()}"
        try:
            os.replace(f, stale)
        except OSError:
            return False                                   # another launcher took it over first
        stale.unlink(missing_ok=True)
    try:
        fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.close(fd)
    _write_lock(f)
    return True


def finish_attempt(attempt: Path, final: Path, results: Path) -> bool:
    """Publish a fully verified attempt as the run's folder; an older result moves to _replaced/ and is put
    back if the publication fails. A failed attempt goes to _failed/ and leaves any older result untouched."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        if verify_run(attempt, full=True, expected_id=final.name):
            final.parent.mkdir(parents=True, exist_ok=True)
            moved = None
            if final.exists():
                (results / "_replaced").mkdir(parents=True, exist_ok=True)
                moved = results / "_replaced" / f"{final.name}.{stamp}"
                os.replace(final, moved)
            try:
                os.replace(attempt, final)
            except OSError:
                if moved is not None:
                    os.replace(moved, final)
                raise
            return True
        (results / "_failed").mkdir(parents=True, exist_ok=True)
        os.replace(attempt, results / "_failed" / f"{final.name}.{stamp}")
    except OSError as e:                                  # e.g. a file of the run is open in another program
        print(f"  could not publish {attempt.name}: {e}; the attempt stays in {attempt.parent}")
    return False


def launch(specs: list[dict], args, cfg: dict) -> int:
    """Parent process: one child process per run (a crashed CUDA context cannot affect later runs).
    Devices: GPUs or GPU partitions take GPU runs; a 'cpu' worker takes the CPU-only families (baselines,
    LightGBM, Vanilla) so they run in parallel with GPU training. Returns 1 if any run failed and 130 if
    interrupted (running children are stopped and their attempts kept under _failed/)."""
    results = ROOT / ("results_demo" if args.demo else "results")
    hashes = data_hashes()
    pending = []
    for s in specs:
        s = dict(s, demo=bool(args.demo))
        c = make_contract(s, cfg, hashes)
        rid = run_id(c)
        final = results / "runs" / rid
        if not args.force and verify_run(final):
            continue
        if (final / "DONE").exists() and not args.force:
            print(f"  {rid} exists but fails verification: it will be run again")
        pending.append((s, dict(c, run_id=rid)))
    print(f"{len(specs)} runs, {len(specs) - len(pending)} already done, {len(pending)} to run -> {results / 'runs'}")
    devices = _devices(args.devices)
    has_cpu = "cpu" in devices
    n_gpu = sum(d != "cpu" for d in devices)

    def fits(spec, dev):
        cpu_run = family(spec["model"]) in CPU_FAMILIES
        return cpu_run if dev == "cpu" else (not cpu_run or not has_cpu)

    orphans = [c["run_id"] for s, c in pending if not any(fits(s, d) for d in devices)]
    if orphans:
        sys.exit(f"no worker in {devices} can run {orphans[:3]} ...: add a GPU device or drop 'cpu'")
    running, failed, skipped = [], [], []
    try:
        while pending or running:
            still = []
            for proc, c, attempt, dev in running:
                code = proc.poll()
                if code is None:
                    still.append((proc, c, attempt, dev))
                    continue
                ok = finish_attempt(attempt, results / "runs" / c["run_id"], results)
                (results / "locks" / f"{c['run_id']}.lock").unlink(missing_ok=True)
                if ok and code != 0:
                    print(f"  finished {c['run_id']}, but the process exited with code {code} after writing its results")
                elif not ok:
                    failed.append(c["run_id"])
                    print(f"  FAILED {c['run_id']} (exit {code}); see {results / '_failed'}")
            running = still
            busy = {r[3] for r in running}
            for dev in devices:
                if dev in busy:
                    continue
                i = next((k for k, (s, _) in enumerate(pending) if fits(s, dev)), None)
                if i is None:
                    continue
                s, c = pending.pop(i)
                if not _lock(results / "locks", c["run_id"]):
                    skipped.append(c["run_id"])
                    try:
                        owner = json.loads((results / "locks" / f"{c['run_id']}.lock").read_text())
                    except Exception:
                        owner = {}
                    print(f"  skip {c['run_id']}: locked by pid {owner.get('pid')} on {owner.get('host')}. If that "
                          f"launcher is gone (e.g. the machine restarted), delete {results / 'locks'} and launch again")
                    continue
                attempt = results / "attempts" / f"{c['run_id']}.{datetime.now():%Y%m%d-%H%M%S}"
                attempt.mkdir(parents=True, exist_ok=True)
                (attempt / "contract.json").write_text(json.dumps(c, indent=1, sort_keys=True))
                env = dict(os.environ)
                if dev == "cpu":
                    # "-1" hides every GPU from both the CUDA runtime and PyTorch; an empty string does not
                    # on Windows (the runtime then still reports a device that PyTorch rejects)
                    env["CUDA_VISIBLE_DEVICES"] = "-1"
                    # LightGBM leaves two cores to each process that feeds a GPU
                    env.setdefault("LGB_THREADS", str(max(1, available_cpus() - 2 * n_gpu - 1)))
                elif dev != "default":
                    env["CUDA_VISIBLE_DEVICES"] = dev
                cmd = [sys.executable, str(ROOT / "run_experiments.py"), "_child", "--contract",
                       str(attempt / "contract.json"), "--out", str(attempt)]
                print(f"  start {c['run_id']} on {dev}")
                proc = subprocess.Popen(cmd, env=env)
                _write_lock(results / "locks" / f"{c['run_id']}.lock", child=proc.pid)
                running.append((proc, c, attempt, dev))
                busy.add(dev)
            time.sleep(2)
    except KeyboardInterrupt:
        print("interrupted: stopping the running runs")
        for proc, c, attempt, dev in running:
            proc.terminate()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for proc, c, attempt, dev in running:
            try:
                proc.wait(timeout=60)
            except Exception:
                proc.kill()
            (results / "_failed").mkdir(parents=True, exist_ok=True)
            try:
                os.replace(attempt, results / "_failed" / f"{c['run_id']}.interrupted.{stamp}")
            except OSError:
                pass
            (results / "locks" / f"{c['run_id']}.lock").unlink(missing_ok=True)
        collect(results, cfg)
        return 130
    collect(results, cfg)
    if failed:
        print(f"{len(failed)} runs FAILED: {failed}")
    if skipped:
        print(f"{len(skipped)} runs skipped (locked by another launcher); they still count as missing")
    return 1 if failed or skipped else 0


def _devices(arg: str) -> list[str]:
    if arg in (None, "", "default"):
        return ["default"]
    if arg == "auto":
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis is not None and vis.strip() in ("", "-1"):    # GPUs deliberately hidden: CPU work only
            return ["cpu"]
        if vis:                                            # respect an allocation made by a scheduler or the user
            return [d.strip() for d in vis.split(",") if d.strip()] + ["cpu"]
        try:
            lines = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()
            # `nvidia-smi -L` lists the partitions of a GPU on indented lines below it
            parts = [l.split("UUID: ")[1].rstrip(")") for l in lines if l.startswith(" ") and "UUID: " in l]
            return (parts or [str(i) for i, l in enumerate(lines) if l.startswith("GPU")]) + ["cpu"]
        except Exception:
            return ["default"]
    return arg.split(",")


def collect(results: Path, cfg: dict) -> None:
    """Single aggregator: runs.csv from every published run folder that verifies and belongs to the current
    contract (workers never share a file). One row per evaluation variant; test columns appear only after
    `evaluate`, and only when computed under the current freeze."""
    rows, stale, broken = [], 0, 0
    hashes = data_hashes()
    frozen = _frozen()
    freeze_id = "demo" if results.name == "results_demo" else (frozen or {}).get("freeze_id")
    for f in sorted((results / "runs").glob("*/metrics.json")):
        if not (f.parent / "DONE").exists():
            continue
        r = json.loads(f.read_text())
        spec = r["spec"]
        if run_id(make_contract(spec, cfg, hashes)) != f.parent.name:   # older contract: never mixed in
            stale += 1
            continue
        if not verify_run(f.parent):
            broken += 1
            continue
        manifest = json.loads((f.parent / "manifest.json").read_text())
        saved_mb = round(sum(m["bytes"] for k, m in manifest.items()
                             if k not in CORE_FILES + ("history.json",)) / 2 ** 20, 2)     # checkpoints, models
        tf = f.parent / "test_metrics.json"
        test = json.loads(tf.read_text()) if tf.exists() else {}
        if test.get("freeze_id") != freeze_id:
            test = {}
        variants = sorted({k.split("_metrics")[0].partition("@")[2] for k in r if "_metrics" in k})
        for v in variants:
            ckpt = F_TEST_ONLY.get(spec["regime"], {}).get(v, v) or "main"
            be = r.get("best_epoch")
            row = {**{k: v2 for k, v2 in spec.items() if k != "demo"}, "variant": v or None,
                   "regime_label": v or spec["regime"], "run_id": f.parent.name,
                   "params": r.get("params"), "best_epoch": be.get(ckpt) if isinstance(be, dict) else be,
                   "epochs_run": r.get("epochs_run"), "device": (r.get("environment") or {}).get("gpu"),
                   "wall_sec": r.get("wall_sec"), "train_time_sec": r.get("train_time_sec"),
                   "fit_sec": r.get("fit_sec"), "fit_and_predict_sec": r.get("fit_and_predict_sec"),
                   "inference_sec": r.get("inference_sec"), "val_predict_sec": r.get("val_predict_sec"),
                   "peak_gpu_mem_mb": r.get("peak_gpu_mem_mb"), "peak_gpu_reserved_mb": r.get("peak_gpu_reserved_mb"),
                   "saved_model_mb": saved_mb}
            for part, src in (("val", r), ("test", test)):
                key = f"{part}{'@' + v if v else ''}_metrics"
                m = src.get(key) or src.get(key + "_subset") or {}
                row[f"{part}_is_subset"] = key not in src and bool(m)
                for k, val in m.items():
                    if not isinstance(val, list):
                        row[f"{part}_{k}"] = val
            rows.append(row)
    if stale:
        print(f"ignored {stale} finished runs of an older contract (protocol, training settings, data or code)")
    if broken:
        print(f"ignored {broken} runs that fail verification (missing or altered files)")
    out = results / "runs.csv"
    if rows:
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"collected {len(rows)} rows -> {out}")
    elif out.exists():                                   # derived file: never leave an outdated table behind
        out.unlink()
        print(f"no current runs; removed the outdated {out}")


def load_torch_model(folder: Path, D: Dataset, checkpoint: str = "checkpoint.pt"):
    """Rebuild a trained network from its run folder (for re-forecasting and the analyses)."""
    import torch
    from models import create_model
    spec = json.loads((folder / "contract.json").read_text())["spec"]
    H, w, regime = spec["H"], spec["w"], spec["regime"]
    if regime in F_REGIMES:
        FI = FInputs(D, regime, w, H)
        n_feat, n_fut = FI.past.shape[1], FI.n_future
    else:
        n_feat, n_fut = D.past_matrix(regime).shape[1], 0
    model = create_model(spec["model"], n_feat, H, w, n_future=n_fut)
    model.load_state_dict(torch.load(folder / checkpoint, map_location="cpu", weights_only=True))
    return model.eval()


# ==============================================================================================
# 5. Efficiency profile (run on the laptop only, nothing else on the GPU)
# ==============================================================================================
def _percentiles_ms(fn, n_warm: int = 20, n_rep: int = 200) -> tuple[float, float]:
    import torch
    for _ in range(n_warm):
        fn()
    times = []
    for _ in range(n_rep):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))


def _profile_torch(spec: dict, D: Dataset, tc: dict) -> dict:
    """Training step (forward + backward + optimizer, batch 256, AMP as in training): peak memory and
    time; inference (fp32 as in forecasting): batch-1 latency p50/p95 and batch throughput. Model-only
    scope: inputs are prepared tensors on the GPU."""
    import torch
    from models import count_parameters, create_model
    H, w, regime = spec["H"], spec["w"], spec["regime"]
    dev = torch.device("cuda")
    if regime in F_REGIMES:
        FI = FInputs(D, regime, w, H)
        n_feat, n_fut = FI.past.shape[1], FI.n_future
    else:
        n_feat, n_fut = D.past_matrix(regime).shape[1], 0
    model = create_model(spec["model"], n_feat, H, w, n_future=n_fut).to(dev)

    def inputs(b):
        x = [torch.randn(b, w, n_feat, device=dev)]
        return x + ([torch.randn(b, H, n_fut, device=dev)] if n_fut else [])

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = torch.nn.HuberLoss()
    micro = micro_batches(spec)                         # as in training
    xb, yb = inputs(tc["batch_size"]), torch.randn(tc["batch_size"], H, device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        for part in np.array_split(np.arange(len(yb)), micro):
            sl = slice(int(part[0]), int(part[-1]) + 1)
            with torch.autocast("cuda"):
                loss = loss_fn(model(*[x[sl] for x in xb]), yb[sl]) * (len(part) / len(yb))
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

    model.train()
    step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step_p50, _ = _percentiles_ms(step, 3, 20)
    train_mem = torch.cuda.max_memory_allocated() / 2 ** 20
    model.eval()
    x1, xt = inputs(1), inputs(prediction_batch(w))
    with torch.no_grad():
        p50, p95 = _percentiles_ms(lambda: model(*x1))
        torch.cuda.reset_peak_memory_stats()
        t_big, _ = _percentiles_ms(lambda: model(*xt), 3, 10)
        infer_mem = torch.cuda.max_memory_allocated() / 2 ** 20
    return {"params": count_parameters(model), "micro_batches": micro, "train_step_ms": step_p50,
            "train_peak_mem_mb": train_mem, "latency_b1_p50_ms": p50, "latency_b1_p95_ms": p95,
            "throughput_windows_per_s": prediction_batch(w) / (t_big / 1000), "infer_peak_mem_mb": infer_mem,
            "scope": "model only (prepared GPU tensors)", "precision": PRECISION["torch"]}


def _profile_nf(spec: dict, D: Dataset, tc: dict) -> dict:
    """NeuralForecast: the same architecture configuration on a synthetic window batch; training step
    (forward + Huber + backward + Adam, fp16 autocast, batch 256) and inference, model-only scope like
    for our models (normalisation, window sampling and pandas handling excluded)."""
    import torch
    import neuralforecast.models as nfm
    H, w, regime = spec["H"], spec["w"], spec["regime"]
    name = spec["model"].removeprefix("NF_")
    kw = {}
    if regime != "uni":
        kw["hist_exog_list"] = [f"h_{c}" for c in CALENDAR_FEATURES]
    if regime in ("F0", "F1", "F4"):
        kw["futr_exog_list"] = [f"f_{c}" for c in CALENDAR_FEATURES] + ([f"f_{c}" for c in TEMP_FEATURES] if regime == "F4" else [])
    multi = name in ("TimeXer", "iTransformer")
    if multi:
        kw["n_series"] = 1
    model = getattr(nfm, name)(h=H, input_size=w, **kw).to("cuda")
    X, F = len(kw.get("hist_exog_list", [])), len(kw.get("futr_exog_list", []))

    def batch(b):
        y = torch.randn(b, w, 1, device="cuda")
        hist = (torch.randn(b, X, w, 1, device="cuda") if multi else torch.randn(b, w, X, device="cuda")) if X else None
        futr = (torch.randn(b, F, w + H, 1, device="cuda") if multi else torch.randn(b, w + H, F, device="cuda")) if F else None
        return dict(insample_y=y, insample_mask=torch.ones_like(y), futr_exog=futr, hist_exog=hist, stat_exog=None)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    bt = batch(tc["batch_size"])

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            out = model(bt)
            loss = torch.nn.functional.huber_loss(out, torch.zeros_like(out))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    model.train()
    step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step_p50, _ = _percentiles_ms(step, 3, 20)
    train_mem = torch.cuda.max_memory_allocated() / 2 ** 20
    model.eval()
    b1, bi = batch(1), batch(prediction_batch(w))
    with torch.no_grad():
        p50, p95 = _percentiles_ms(lambda: model(b1))
        torch.cuda.reset_peak_memory_stats()
        t_big, _ = _percentiles_ms(lambda: model(bi), 3, 10)
        infer_mem = torch.cuda.max_memory_allocated() / 2 ** 20
    return {"params": int(sum(p.numel() for p in model.parameters())), "train_step_ms": step_p50,
            "train_peak_mem_mb": train_mem, "latency_b1_p50_ms": p50, "latency_b1_p95_ms": p95,
            "throughput_windows_per_s": prediction_batch(w) / (t_big / 1000), "infer_peak_mem_mb": infer_mem,
            "scope": "model only (prepared GPU tensors)", "precision": "fp16 autocast training, fp32 inference"}


def _profile_foundation(spec: dict, D: Dataset, loaded: dict) -> dict:
    import torch
    H, w, regime = spec["H"], spec["w"], spec["regime"]
    if spec["model"] not in loaded:
        loaded.clear()
        torch.cuda.empty_cache()
        loaded[spec["model"]] = _tsfm_load(spec["model"])
    predict, n_params = loaded[spec["model"]]
    o = D.origins("test", H)
    one = _tsfm_arrays(D, regime, "main", o[:1], w, H)
    many = _tsfm_arrays(D, regime, "main", o[:256 if w <= 720 else 64], w, H)
    p50, p95 = _percentiles_ms(lambda: predict(*one, H), 3, 30)
    torch.cuda.reset_peak_memory_stats()
    t_big, _ = _percentiles_ms(lambda: predict(*many, H), 1, 3)
    return {"params": n_params, "latency_b1_p50_ms": p50, "latency_b1_p95_ms": p95,
            "throughput_windows_per_s": len(many[0]) / (t_big / 1000),
            "infer_peak_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
            "scope": "library predict call incl. its preprocessing (numpy in, numpy out)", "precision": PRECISION["foundation"]}


def profile(specs: list[dict], tc: dict) -> None:
    """One row per (model, regime, H, w) cell -> results/profile.csv (existing rows of the same cell are
    replaced). Pure-PyTorch sLSTM kernels are used for TiRex-2 (no compiler): its latency is marked."""
    import torch
    if not torch.cuda.is_available():
        sys.exit("profiling needs the GPU")
    D = Dataset()
    cells = {(s["model"], s["regime"], s["H"], s["w"]): s for s in specs if family(s["model"]) in ("torch", "nf", "foundation")}
    rows, loaded = [], {}
    for (m, reg, H, w), s in cells.items():
        torch.cuda.empty_cache()
        row = {"model": m, "regime": reg, "H": H, "w": w, "gpu": torch.cuda.get_device_name(0)}
        try:
            fam = family(m)
            row |= (_profile_torch(s, D, tc) if fam == "torch" else _profile_nf(s, D, tc) if fam == "nf"
                    else _profile_foundation(s, D, loaded))
            row["note"] = "pure-PyTorch sLSTM kernels" if m == "TiRex2" else ""
        except Exception as e:                                          # e.g. out of memory: recorded, not fatal
            row["note"] = f"failed: {type(e).__name__}: {str(e)[:200]}"
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        rows.append(row)
    f = ROOT / "results" / "profile.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if f.exists():
        old = pd.read_csv(f)
        keys = set(zip(new.model, new.regime, new.H, new.w))
        old = old[[k not in keys for k in zip(old.model, old.regime, old.H, old.w)]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(f, index=False)
    print(f"profile -> {f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "run", "collect", "select-lr", "select-windows", "freeze", "evaluate",
                                        "profile", "_child"])
    ap.add_argument("--experiment")
    ap.add_argument("--model")
    ap.add_argument("--regime", default="cal")
    ap.add_argument("--H", type=int)
    ap.add_argument("--w", type=int)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--demo", action="store_true", help="tiny end-to-end check of every step (results_demo/)")
    ap.add_argument("--force", action="store_true", help="re-run finished runs (the old result stays until the new one is verified)")
    ap.add_argument("--devices", default="default", help="'auto' = one worker per GPU or GPU partition + one CPU worker, "
                                                         "or a list such as '0,cpu'")
    ap.add_argument("--contract")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.command == "_child":
        try:
            execute_one(json.loads(Path(args.contract).read_text()), Path(args.out))
        except Exception:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            with open(Path(args.out) / "log.txt", "a", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            traceback.print_exc()
            return 1
        return 0
    cfg = load_config()
    results = ROOT / ("results_demo" if args.demo else "results")
    if args.command in ("select-lr", "select-windows", "freeze") and args.demo:
        sys.exit("selections and freezing are made on real runs only (they write configs/)")
    if args.command == "collect":
        collect(results, cfg)
        return 0
    if args.command == "select-lr":
        select_learning_rates(cfg, results)
        return 0
    if args.command == "select-windows":
        select_windows(cfg, results)
        return 0
    if args.command == "freeze":
        freeze(cfg, results)
        return 0
    if args.experiment:
        specs = expand_experiment(cfg, args.experiment, demo=args.demo)
    elif args.model:
        specs = [{"model": args.model, "regime": args.regime, "H": args.H, "w": args.w, "seed": s} for s in args.seeds]
    elif args.command == "evaluate":                        # default: the whole campaign
        specs = expand_experiment(cfg, ",".join(cfg["campaign"]), demo=args.demo)
    else:
        ap.error("give --experiment or --model/--H/--w")
    if args.command == "list":
        for s in specs:
            print(" ", s)
        counts = pd.Series([family(s["model"]) for s in specs]).value_counts().to_dict()
        print(f"{len(specs)} runs", counts)
        return 0
    if args.command == "evaluate":
        return evaluate(specs, cfg, results, args.demo)
    if args.command == "profile":
        profile(specs, dict(cfg["training"]))
        return 0
    return launch(specs, args, cfg)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip interpreter and DLL teardown. On Windows the cuDNN RNN libraries can crash while being unloaded
    # (exit code 0xC0000409) after a run has already written and closed all of its files.
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.TerminateProcess(ctypes.windll.kernel32.GetCurrentProcess(), code)
    os._exit(code)
