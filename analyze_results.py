"""Stage 4: analysis of the finished campaign.

Usage (from the project root):
  python analyze_results.py plan       fix, from the validation year only, what the test year will be asked:
                                       the primary comparisons, the information contrasts and the ensemble.
                                       Writes configs/analysis_plan.json; refuses to run once a test score exists.
  python analyze_results.py losses     score every stored test forecast once: per-origin loss series of runs and
                                       systems, lead curves and the system inventory (results/analysis/)
  python analyze_results.py inference  paired moving-block bootstrap, Holm and Diebold-Mariano for the plan's
                                       comparisons and information contrasts (results/analysis/inference.csv)
  python analyze_results.py seedboot   exploratory: the same comparisons with the seeds resampled as well
                                       (results/analysis/inference_seeds.csv)
  python analyze_results.py breakdowns errors by target-hour group, 2025 event, month, hour of day and daily peak;
                                       monthly energy totals of weekly updated H = 168 forecasts (descriptive)
  python analyze_results.py tables     tables T1-T6 and the supplementary tables as CSV and one Excel workbook
                                       (results/analysis/tables/)
  python analyze_results.py figures [data leads ...]  figures as vector PDF and 600 dpi PNG (results/analysis/figures/)
  python analyze_results.py profile [passN]  one efficiency pass, a fresh process per reported system (efficiency_passN.csv)
  python analyze_results.py profile-merge  median of the passes -> efficiency.csv
  python analyze_results.py diagnostics  gates, learning curves and attention weights of the dual-path TCNs
  python analyze_results.py manuscript manuscript/manuscript.md with its tables and numbered references -> Word,
                                       the claim table paper_numbers.json, then the supplementary material
  python analyze_results.py supplement manuscript/supplement_notes.md with tables S1-S21 and Figs. S1-S2 -> Word

Why a plan file: the test year is scored once. Which model counts as "the best TCN" or "the best rival",
and which models form the ensemble, must therefore be decided on the validation year (2024) and written
down before `freeze` and `evaluate`. The plan file is checked by hash before any analysis and is never
rewritten.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_experiments as rx

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "runs"
ANALYSIS = ROOT / "results" / "analysis"
PLAN_FILE = ROOT / "configs" / "analysis_plan.json"
# written before the test year was scored (plan) and before any inference result existed (note, addendum)
FIXED_INPUTS = {PLAN_FILE: "c1155605b5abdba0efa73b030527e1e2b7b8fd1a145cdca15f636688c1aad36d"}


def check_fixed_inputs() -> None:
    """Every analysis command runs on the analysis plan exactly as it was fixed."""
    for f, sha in FIXED_INPUTS.items():
        if rx.sha256_file(f) != sha:
            sys.exit(f"{f.name} differs from the fixed version")

# experiments whose runs are reported in the past-only (P) tables
P_EXPERIMENTS = "baselines,grid,long_horizon,nf_grid,foundation,lightgbm,panel,chronos_lora"
# information contrasts of the known-future regimes, fixed by the design (effect = second minus first)
INFORMATION_CONTRASTS = [
    ["F0", "F1", "calendar of the target hours"],
    ["F1", "F1A0", "holiday-analog flag alone"],
    ["F1A0", "F1A", "last year's holiday load, given the flag"],
    ["F1", "F1A", "holiday analog (flag and value)"],
    ["F1", "F2", "past temperature"],
    ["F2", "F3", "forecast temperature of the target hours (bias-corrected)"],
    ["F3raw_diag", "F3", "bias correction of the forecasts (same checkpoint)"],
    ["F3_m24", "F3", "12 h instead of 24 h vintage margin (same checkpoint)"],
    ["F3", "F3_diag", "observed instead of forecast future temperature (same checkpoint)"],
    ["F3", "F4", "oracle weather scenario instead of the realistic one"],
]


def group(model: str) -> str:
    """Model family as used in the tables."""
    if model == "ensemble":
        return "ensemble"
    if model.startswith("LightTCN_F_"):
        return "TCN (known future)"
    if model.startswith(("LightTCN_", "ConcatTCN_", "DualTCN_")):
        return "TCN"
    return {"torch": "RNN", "nf": "NF", "foundation": "foundation", "foundation_ft": "foundation (LoRA)",
            "lightgbm": "LightGBM", "baseline": "naive", "vanilla": "Vanilla"}[rx.family(model)]


def reported_cells(cfg: dict, sel: dict, hashes: dict) -> pd.DataFrame:
    """One row per run of the P tables: every model at its validation-selected window and learning rate
    (w = 0 for models without a window), with its validation MAPE. No test record is read."""
    rows, seen = [], set()
    for s in rx.expand_experiment(cfg, P_EXPERIMENTS):
        if s["regime"] not in ("uni", "cal") or s.get("gap"):
            continue
        chosen = sel["window"].get(f"{rx.base_model(s['model'])}|{rx.base_regime(s['regime'])}|{s['H']}")
        if s["w"] and s["w"] != chosen:
            continue
        rid = rx.run_id(rx.make_contract(s, cfg, hashes))
        if rid in seen:
            continue
        seen.add(rid)
        rec = json.loads((RUNS / rid / "metrics.json").read_text())
        rows.append({"model": s["model"], "regime": s["regime"], "H": s["H"], "w": s["w"], "seed": s["seed"],
                     "group": group(s["model"]), "run_id": rid, "val_MAPE": rec["val_metrics"]["MAPE"]})
    return pd.DataFrame(rows)


def make_plan() -> None:
    if any(RUNS.glob("*/test_metrics.json")):
        sys.exit("test scores exist: the analysis plan can only be written before `evaluate`")
    cfg, sel, hashes = rx.load_config(), rx.load_selected(), rx.data_hashes()
    runs = reported_cells(cfg, sel, hashes)
    cells = (runs.groupby(["group", "model", "regime", "H", "w"], as_index=False)
             .agg(n_seeds=("seed", "size"), val_MAPE=("val_MAPE", "mean")))

    def best(H, groups, regimes, min_seeds=1):
        c = cells[(cells.H == H) & cells.group.isin(groups) & cells.regime.isin(regimes) & (cells.n_seeds >= min_seeds)]
        r = c.loc[c.val_MAPE.idxmin()]
        return [str(r.model), str(r.regime)]

    rivals = ["RNN", "NF", "foundation", "foundation (LoRA)", "LightGBM"]
    primary, secondary, ensemble = [], [], {}
    for H in (1, 24, 168):
        primary.append({"H": H, "question": "best TCN vs best rival, both with the past calendar",
                        "a": best(H, ["TCN"], ["cal"]), "b": best(H, rivals, ["cal"])})
        for size in ("medium", "large"):
            primary.append({"H": H, "question": "parameter-free attention read-out vs the same network without it",
                            "a": [f"DualTCN_Attn_{size}", "cal"], "b": [f"DualTCN_{size}", "cal"]})
        secondary.append({"H": H, "question": "best LightTCN vs best rival, both univariate",
                          "a": best(H, ["TCN"], ["uni"]), "b": best(H, rivals, ["uni"])})
    # H = 720: only LightTCN and the two validation-selected rivals have five seeds
    tcn720 = best(720, ["TCN"], ["cal"], min_seeds=5)
    for m, reg in sel["h720_rivals"]:
        primary.append({"H": 720, "question": "best LightTCN vs a validation-selected rival (five seeds each)",
                        "a": tcn720, "b": [m, reg]})
    for H in (1, 24, 168, 720):
        ensemble[str(H)] = [best(H, [g], ["uni", "cal"]) for g in ("TCN", "RNN", "NF", "foundation", "LightGBM")]

    plan = {
        "basis": "validation year 2024 only; written before `freeze` and `evaluate`",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reported_cell": "every model at the window and learning rate selected on the validation year",
        "loss": {"primary": "absolute percentage error, averaged over the H leads of an origin and over the seeds "
                            "(one value per hourly origin)",
                 "secondary": "squared error in MWh^2, averaged the same way"},
        "inference": {"primary": "paired moving-block bootstrap of the mean loss difference, 95 % percentile interval",
                      "block_hours": "max(2 H, 168)", "block_sensitivity": [0.5, 2.0], "resamples": 10000,
                      "rng_seed": 20260930,
                      "secondary": "Diebold-Mariano test with a HAC variance (lag H - 1) and the HLN correction",
                      "multiplicity": "Holm over the primary comparisons; effect sizes are always reported"},
        "primary_comparisons": primary,
        "secondary_comparisons": secondary,
        "information_contrasts": {"models": ["LightTCN_F_medium", "LightGBM", "NF_NHITS", "Chronos2", "TimesFM3", "TiRex2"],
                                  "contrasts": INFORMATION_CONTRASTS,
                                  "note": "a contrast is reported where both arms were run; the zero-shot models' "
                                          "F0 is their `cal` run on the same origins"},
        "ensemble": {"rule": "unweighted mean of the seed-averaged forecasts of the best model of each family "
                             "(TCN, RNN, NF, zero-shot foundation, LightGBM) on the validation year",
                     "members": ensemble},
        "validation_table": cells.sort_values(["H", "val_MAPE"]).round({"val_MAPE": 4}).to_dict("records"),
    }
    PLAN_FILE.write_text(json.dumps(plan, indent=1))
    pd.set_option("display.width", 200)
    print(f"analysis plan -> {PLAN_FILE}")
    for c in primary:
        print(f"  primary   H={c['H']:<3d} {c['a'][0]} ({c['a'][1]})  vs  {c['b'][0]} ({c['b'][1]})")
    for c in secondary:
        print(f"  secondary H={c['H']:<3d} {c['a'][0]} ({c['a'][1]})  vs  {c['b'][0]} ({c['b'][1]})")
    for H, members in ensemble.items():
        print(f"  ensemble  H={H:<3s} " + ", ".join(f"{m} ({r})" for m, r in members))


# ==============================================================================================
# Loss series (implementation note §3, addendum 1 §1-2)
# ==============================================================================================
HORIZONS = (1, 24, 168, 720)
CHECKPOINT_FREE = (*rx.TSFM_MODELS, "Vanilla", "Vanilla_holiday")    # their F3raw_diag would equal F3raw
CSV_COLUMNS = {"MAPE": "test_MAPE", "MAE": "test_MAE", "RMSE": "test_RMSE", "MASE": "test_MASE",
               "bias_pct": "test_mean_signed_pct_error", "under_share": "test_under_forecast_share"}
SCALARS = list(CSV_COLUMNS)


def campaign_runs(cfg: dict, hashes: dict) -> pd.DataFrame:
    """Every run of the campaign once, with the experiments that declare it (a run can serve several)."""
    rows = {}
    for exp in cfg["campaign"]:
        for s in rx.expand_experiment(cfg, exp):
            rid = rx.run_id(rx.make_contract(s, cfg, hashes))
            if rid in rows:
                rows[rid]["experiments"] += f",{exp}"
            else:
                rows[rid] = {"run_id": rid, "model": s["model"], "regime": s["regime"], "H": s["H"], "w": s["w"],
                             "seed": s["seed"], "gap": s.get("gap", 0), "experiments": exp}
    return pd.DataFrame(rows.values())


def score(forecast: np.ndarray, y: np.ndarray, mase_scale: float) -> tuple[dict, dict]:
    """Per-origin losses (means over the leads) and lead curves of one forecast matrix (origins x H), and
    its scalar metrics as defined by run_experiments.metrics."""
    e = forecast.astype(np.float64) - y
    ape, pe = 100.0 * np.abs(e) / y, 100.0 * e / y
    series = {"ape": ape.mean(axis=1), "se": (e ** 2).mean(axis=1), "lead_MAE": np.abs(e).mean(axis=0),
              "lead_MAPE": ape.mean(axis=0), "lead_bias_pct": pe.mean(axis=0)}
    mae = float(np.abs(e).mean())
    scalars = {"MAPE": float(ape.mean()), "MAE": mae, "RMSE": float(np.sqrt((e ** 2).mean())),
               "MASE": mae / mase_scale, "bias_pct": float(pe.mean()), "under_share": float((e < 0).mean())}
    return series, scalars


def seed_summary(values: list[float]) -> dict:
    """Mean, SD and t-based 95 % half-width over seeds; no spread for a single seed (note §8)."""
    from scipy import stats
    n = len(values)
    if n == 1:
        return {"mean": values[0], "sd": np.nan, "ci95": np.nan}
    sd = float(np.std(values, ddof=1))
    return {"mean": float(np.mean(values)), "sd": sd, "ci95": float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))}


def build_losses() -> None:
    """Score every stored test forecast of the campaign once (float64 from the float32 files), check each run
    against runs.csv, and cache the per-origin series of runs and of the reported systems."""
    cfg, sel, hashes = rx.load_config(), rx.load_selected(), rx.data_hashes()
    plan = json.loads(PLAN_FILE.read_text())
    D = rx.Dataset()
    csv = pd.read_csv(ROOT / "results" / "runs.csv")
    csv = csv.assign(variant=csv.variant.fillna("main")).set_index(["run_id", "variant"])
    runs = campaign_runs(cfg, hashes)
    if set(runs.run_id) != set(csv.index.get_level_values(0)):
        sys.exit("the campaign's runs and runs.csv differ; rebuild runs.csv with `run_experiments.py collect`")
    cells = reported_cells(cfg, sel, hashes)
    members = {int(H): [tuple(m) for m in ms] for H, ms in plan["ensemble"]["members"].items()}
    meta = runs.set_index("run_id")
    run_rows, lead_rows, system_rows, mismatches = [], [], [], []
    series_by_h = {}
    for H in HORIZONS:
        o = D.origins("test", H)
        keys, y = D.utc_ns[o], D.targets(o, H)
        m = rx.seasonal_period(H)
        scale = float(np.mean(np.abs(D.y[m:D.train_end] - D.y[:D.train_end - m])))
        run_cols, sys_cols, stored, member_fc = {}, {}, {}, {}

        def add_system(system, kind, model, arm, parts, note="", combined=None):
            """parts: (run_id, variant) pairs, one per seed. `combined` is the forecast of a seed-mean or
            ensemble system; otherwise the system's losses are the means of its runs' losses."""
            if combined is None:
                sys_cols[f"{system}|L"] = np.mean([run_cols[f"{r}|{v}|ape"] for r, v in parts], axis=0)
                sys_cols[f"{system}|S"] = np.mean([run_cols[f"{r}|{v}|se"] for r, v in parts], axis=0)
                stats_ = {k: seed_summary([stored[p][k] for p in parts]) for k in SCALARS}
                values = {k: s["mean"] for k, s in stats_.items()}
                values.update(MAPE_sd=stats_["MAPE"]["sd"], MAPE_ci95=stats_["MAPE"]["ci95"],
                              RMSE_sd=stats_["RMSE"]["sd"])
            else:
                s, values = score(combined, y, scale)
                sys_cols[f"{system}|L"], sys_cols[f"{system}|S"] = s["ape"], s["se"]
            w = sorted({int(meta.w[r]) for r, _ in parts})
            system_rows.append({"system": system, "kind": kind, "group": group(model), "model": model, "arm": arm,
                                "H": H, "w": ";".join(map(str, w)), "n_seeds": len(parts),
                                "seeds": ";".join(str(meta.seed[r]) for r, _ in parts),
                                "runs": ";".join(f"{r}|{v}" for r, v in parts), "note": note, **values})

        # P cells first: their seed forecasts also form the seed-mean systems and the ensembles
        units = [(g.run_id.tolist(), (mdl, reg)) for (mdl, reg), g in cells[cells.H == H].groupby(["model", "regime"])]
        in_cells = {r for ids, _ in units for r in ids}
        units += [([r], None) for r in runs[(runs.H == H) & ~runs.run_id.isin(in_cells)].run_id]
        for ids, cell in units:
            total = None
            for rid in ids:
                sealed = json.loads((RUNS / rid / "metrics.json").read_text())["sealed_test_keys"]
                with np.load(RUNS / rid / "test_sealed.npz") as z:
                    for key in (k for k in sealed if k.startswith("test")):
                        variant = key.partition("@")[2] or "main"
                        if not np.array_equal(z[f"{key}_origin_utc_ns"], keys):
                            sys.exit(f"{rid} {key}: the test origins differ from the full test year (note §4)")
                        fc = z[f"{key}_forecast_mwh"].astype(np.float64)
                        s, scalars = score(fc, y, scale)
                        ref = csv.loc[(rid, variant)]
                        bad = [k for k, c in CSV_COLUMNS.items() if not np.isclose(scalars[k], ref[c], rtol=1e-9, atol=1e-12)]
                        if bad:
                            mismatches.append(f"{rid}|{variant}: {bad}")
                        run_cols[f"{rid}|{variant}|ape"], run_cols[f"{rid}|{variant}|se"] = s["ape"], s["se"]
                        stored[(rid, variant)] = scalars
                        r = meta.loc[rid]
                        run_rows.append({"run_id": rid, "variant": variant, "model": r.model, "regime": r.regime, "H": H,
                                         "w": r.w, "seed": r.seed, "gap": r.gap, "experiments": r.experiments, **scalars})
                        lead_rows.append(pd.DataFrame({"run_id": rid, "variant": variant, "H": H, "lead": np.arange(1, H + 1),
                                                       "MAE": s["lead_MAE"], "MAPE": s["lead_MAPE"],
                                                       "bias_pct": s["lead_bias_pct"]}))
                        if cell is not None:
                            total = fc if total is None else total + fc
            if cell is None:
                continue
            mdl, reg = cell
            parts = [(r, "main") for r in ids]
            add_system(f"{mdl}|{reg}|H{H}", "P", mdl, reg, parts)
            if len(ids) > 1:
                add_system(f"{mdl}|{reg}|H{H}|seedmean", "P seed mean", mdl, reg, parts,
                           "forecasts averaged over seeds, then scored", combined=total / len(ids))
            if (mdl, reg) in members[H]:
                member_fc[(mdl, reg)] = total / len(ids)

        if set(member_fc) != set(members[H]):
            sys.exit(f"H={H}: ensemble members missing from the reported cells: {set(members[H]) - set(member_fc)}")
        ensemble = [p for mdl, reg in members[H] for p in [(r, "main") for r in cells[(cells.H == H) & (cells.model == mdl)
                                                                                     & (cells.regime == reg)].run_id]]
        add_system(f"ensemble|H{H}", "ensemble", "ensemble", "P", ensemble,
                   "unweighted mean of the seed-mean forecasts of " + ", ".join(f"{a} ({b})" for a, b in members[H]),
                   combined=np.mean([member_fc[k] for k in members[H]], axis=0))

        # F systems: one per (model, arm) with all seeds of the arm
        rh = pd.DataFrame(r for r in run_rows if r["H"] == H)
        f_runs = rh[rh.regime.isin(rx.F_REGIMES)].assign(arm=lambda t: np.where(t.regime == "F34", t.variant, t.regime))
        arms = {k: list(zip(g.run_id, g.variant)) for k, g in f_runs.groupby(["model", "arm"])}
        for (mdl, arm), parts in sorted(arms.items()):
            add_system(f"{mdl}|{arm}|H{H}", "F", mdl, arm, parts)
        for mdl in sorted({mdl for mdl, _ in arms}):
            if mdl in rx.TSFM_MODELS:
                cal = cells[(cells.H == H) & (cells.model == mdl) & (cells.regime == "cal")].run_id
                add_system(f"{mdl}|F0|H{H}", "F", mdl, "F0", [(r, "main") for r in cal],
                           "zero-shot F0 = the model's cal run (plan)")
            if mdl in CHECKPOINT_FREE and (mdl, "F3raw") in arms and (mdl, "F3raw_diag") not in arms:
                add_system(f"{mdl}|F3raw_diag|H{H}", "F", mdl, "F3raw_diag", arms[(mdl, "F3raw")],
                           "identical to F3raw: no checkpoint (addendum 1 §2)")
        # ablations, publication delay and short windows (descriptive, matched references in the tables)
        extra = rh[(rh.variant == "main") & ~rh.run_id.isin(in_cells) & ~rh.regime.isin(rx.F_REGIMES)]
        for (mdl, reg), g in extra[extra.regime.isin(rx.ABLATION_GROUPS)].groupby(["model", "regime"]):
            add_system(f"{mdl}|{reg}|H{H}", "ablation", mdl, reg, list(zip(g.run_id, g.variant)))
        for (mdl, reg, gap), g in extra[extra.gap > 0].groupby(["model", "regime", "gap"]):
            add_system(f"{mdl}|{reg}|H{H}|gap{gap}", "publication delay", mdl, reg, list(zip(g.run_id, g.variant)),
                       f"inputs end at t-{gap + 1}")
        short = extra[extra.experiments.str.contains("short_windows") & (extra.gap == 0)]
        for (mdl, reg, w), g in short.groupby(["model", "regime", "w"]):
            add_system(f"{mdl}|{reg}|H{H}|w{w}", "short window", mdl, reg, list(zip(g.run_id, g.variant)))
        series_by_h[H] = (keys, run_cols, sys_cols)
        print(f"H={H}: {len(run_cols) // 2} run series, {len(sys_cols) // 2} systems")

    if mismatches:
        sys.exit(f"{len(mismatches)} runs do not reproduce runs.csv to 1e-9 (note §3), e.g. {mismatches[:5]}")
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    for H, (keys, run_cols, sys_cols) in series_by_h.items():
        index = pd.Index(keys, name="origin_utc_ns")
        pd.DataFrame(run_cols, index=index).to_parquet(ANALYSIS / f"runs_H{H}.parquet")
        pd.DataFrame(sys_cols, index=index).to_parquet(ANALYSIS / f"systems_H{H}.parquet")
    pd.concat(lead_rows, ignore_index=True).to_parquet(ANALYSIS / "lead_curves.parquet", index=False)
    pd.DataFrame(run_rows).to_csv(ANALYSIS / "run_metrics.csv", index=False)
    systems = pd.DataFrame(system_rows)
    systems.to_csv(ANALYSIS / "systems.csv", index=False)
    print(f"{len(run_rows)} run variants reproduce runs.csv to 1e-9; {len(systems)} systems -> {ANALYSIS}")
    print(systems.groupby(["kind", "H"]).size().unstack(fill_value=0).to_string())


# ==============================================================================================
# Inference (implementation note §4-§8, addendum 1 §3-§5)
# ==============================================================================================
BOOT_B, BOOT_SEED = 10_000, 20260930
_STARTS: dict = {}


def block_length(H: int) -> int:
    return max(2 * H, 168)


def block_starts(N: int, b: int) -> np.ndarray:
    """B x k starts of the non-circular moving-block bootstrap; one fixed set per (N, b) (note §5)."""
    if (N, b) not in _STARTS:
        k = -(-N // b)
        _STARTS[N, b] = np.random.default_rng(BOOT_SEED).integers(0, N - b + 1, size=(BOOT_B, k))
    return _STARTS[N, b]


def bootstrap_means(d: np.ndarray, b: int) -> np.ndarray:
    """Means of the B resamples of d: k - 1 whole blocks and the first N - (k - 1) b values of the last,
    i.e. the concatenated blocks cut to N, from prefix sums of the centred series (addendum 1 §3)."""
    N = len(d)
    starts = block_starts(N, b)
    tail = N - (starts.shape[1] - 1) * b
    centre = float(d.mean())
    cs = np.concatenate([[0.0], np.cumsum(d - centre)])
    whole = (cs[starts[:, :-1] + b] - cs[starts[:, :-1]]).sum(axis=1)
    return centre + (whole + cs[starts[:, -1] + tail] - cs[starts[:, -1]]) / N


def bootstrap_p(d: np.ndarray, means: np.ndarray) -> float:
    """Two-sided p of H0: E[d] = 0, centred at the mean of the resampled means (note §5)."""
    return float((1 + np.sum(np.abs(means - means.mean()) >= abs(d.mean()))) / (len(means) + 1))


def dm_hln(d: np.ndarray, H: int) -> tuple[float, float]:
    """Diebold-Mariano statistic with the rectangular HAC variance of lag H - 1 and the HLN correction, and
    its two-sided p (Student t, N - 1 df); (nan, nan) when the variance estimate is not positive (note §7)."""
    from scipy import stats
    N, dc = len(d), d - d.mean()
    v = dc @ dc / N + 2 * sum(dc[k:] @ dc[:N - k] / N for k in range(1, H))
    if v <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(v / N) * np.sqrt((N + 1 - 2 * H + H * (H - 1) / N) / N)
    return float(stat), float(2 * stats.t.sf(abs(stat), N - 1))


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values (note §6)."""
    p = np.asarray(p, dtype=float)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def contrast_type(model: str, first: str, second: str) -> str:
    if model in rx.TSFM_MODELS:
        return "same pretrained model, zero-shot"
    if {first, second} <= {"F3", "F3raw_diag", "F3_m24", "F3_diag"}:
        return "same checkpoint"
    if {first, second} == {"F3", "F4"}:
        return "scenario-specific checkpoint of one training run"
    return "separately trained arms"


def comparisons(plan: dict, have: set) -> list[dict]:
    """The plan's primary and secondary comparisons (d = A - B, A = the plan's `a`) and its information
    contrasts where both arms exist (d = second - first), each with its Holm family (note §4, §6)."""
    out = []
    for family, key in (("primary", "primary_comparisons"), ("secondary", "secondary_comparisons")):
        for c in plan[key]:
            attention = c["a"][0].startswith("DualTCN_Attn")
            out.append({"family": family, "H": c["H"], "A": f"{c['a'][0]}|{c['a'][1]}|H{c['H']}",
                        "B": f"{c['b'][0]}|{c['b'][1]}|H{c['H']}", "question": c["question"],
                        "type": "separately tuned systems" if attention else "validation-selected systems",
                        "exploratory": False})
    for model in plan["information_contrasts"]["models"]:
        for H in HORIZONS:
            for first, second, label in plan["information_contrasts"]["contrasts"]:
                A, B = f"{model}|{second}|H{H}", f"{model}|{first}|H{H}"
                if A in have and B in have:
                    out.append({"family": f"information {model} H={H}", "H": H, "A": A, "B": B, "question": label,
                                "type": contrast_type(model, first, second), "exploratory": "F1A" in first + second})
    return out


def run_inference() -> None:
    """Bootstrap intervals and p-values at b, 0.5 b and 2 b, Holm within each family, DM/HLN beside them;
    primary loss APE, secondary loss squared error (addendum 1 §5)."""
    plan = json.loads(PLAN_FILE.read_text())
    systems = pd.read_csv(ANALYSIS / "systems.csv").set_index("system")
    series = {H: pd.read_parquet(ANALYSIS / f"systems_H{H}.parquet") for H in HORIZONS}
    run_series = {H: pd.read_parquet(ANALYSIS / f"runs_H{H}.parquet") for H in HORIZONS}
    rows, draws = [], {}
    for c in comparisons(plan, set(systems.index)):
        H, b = c["H"], block_length(c["H"])
        A, B = systems.loc[c["A"]], systems.loc[c["B"]]
        for loss, col in (("APE", "L"), ("SE", "S")):
            d = (series[H][f"{c['A']}|{col}"] - series[H][f"{c['B']}|{col}"]).to_numpy()
            if np.isnan(d).any():
                sys.exit(f"{c['A']} vs {c['B']}: the origin sets differ (note §4)")
            means = {f: bootstrap_means(d, int(f * b)) for f in (0.5, 1, 2)}
            ci = {f: np.percentile(m, [2.5, 97.5]) for f, m in means.items()}
            dm, p_dm = dm_hln(d, H)
            ref = float(series[H][f"{c['B']}|{col}"].mean())
            row = {**c, "loss": loss, "n_A": int(A.n_seeds), "n_B": int(B.n_seeds), "N": len(d), "block": b,
                   "blocks_in_N": len(d) / b, "mean_A": float(series[H][f"{c['A']}|{col}"].mean()), "mean_B": ref,
                   "diff": float(d.mean()), "rel_diff_pct": 100 * float(d.mean()) / ref,
                   "ci_low": ci[1][0], "ci_high": ci[1][1], "ci_low_half_b": ci[0.5][0], "ci_high_half_b": ci[0.5][1],
                   "ci_low_double_b": ci[2][0], "ci_high_double_b": ci[2][1], "p_boot": bootstrap_p(d, means[1]),
                   "dm_hln": dm, "p_dm": p_dm, "note": "; ".join(n for n in (A.note, B.note) if isinstance(n, str) and n)}
            if loss == "APE" and A.n_seeds != B.n_seeds:       # seed-42 matched diagnostic, no p-value (note §10)
                one = [[p for p, s in zip(x.runs.split(";"), x.seeds.split(";")) if s == "42"] for x in (A, B)]
                if all(one):
                    rs = run_series[H]
                    row["diff_seed42"] = float((rs[f"{one[0][0]}|ape"] - rs[f"{one[1][0]}|ape"]).mean())
            row["draws_key"] = f"c{len(draws):03d}"
            draws[row["draws_key"]] = means[1]
            rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby(["family", "loss"]).p_boot.transform(lambda p: holm(p.to_numpy()))
    out["significant"] = out.p_holm < 0.05
    out.to_csv(ANALYSIS / "inference.csv", index=False)
    np.savez_compressed(ANALYSIS / "bootstrap_means.npz", **draws)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    show = ["H", "A", "B", "diff", "ci_low", "ci_high", "ci_low_double_b", "ci_high_double_b", "p_boot", "p_holm", "p_dm"]
    for family in ("primary", "secondary"):
        print(f"\n{family} comparisons, APE (d = A - B, negative favours A):")
        print(out[(out.family == family) & (out.loss == "APE")][show].round(4).to_string(index=False))
    info = out[out.family.str.startswith("information") & (out.loss == "APE")]
    print(f"\ninformation contrasts: {len(info)} in {info.family.nunique()} families; "
          f"{int(info.significant.sum())} significant after Holm -> {ANALYSIS / 'inference.csv'}")


def run_seed_bootstrap() -> None:
    """Exploratory, not in the plan: the 14 pre-specified comparisons with a two-stage bootstrap. The planned
    test averages the seeds before resampling the origins, so it conditions on the fitted networks; here each
    replicate also draws every system's seeds with replacement (independently for A and B) before the blocks,
    so the 95 % percentile interval also reflects training randomness. Same blocks as the planned bootstrap.
    -> results/analysis/inference_seeds.csv"""
    inf = pd.read_csv(ANALYSIS / "inference.csv")
    comp = inf[inf.family.isin(["primary", "secondary"])]
    runs_of = pd.read_csv(ANALYSIS / "systems.csv").drop_duplicates("system").set_index("system").runs
    rng = np.random.default_rng(BOOT_SEED + 1)
    rows = []
    for H in HORIZONS:
        per_run = pd.read_parquet(ANALYSIS / f"runs_H{H}.parquet")
        b = block_length(H)
        for r in comp[comp.H == H].itertuples():
            loss = "ape" if r.loss == "APE" else "se"
            side = {}
            for k, system in (("A", r.A), ("B", r.B)):
                X = per_run[[f"{p}|{loss}" for p in runs_of[system].split(";")]].to_numpy()       # origins x seeds
                means = np.stack([bootstrap_means(X[:, j], b) for j in range(X.shape[1])], axis=1)  # replicates x seeds
                draw = rng.integers(0, X.shape[1], size=means.shape)
                side[k] = (np.take_along_axis(means, draw, axis=1).mean(axis=1), X.mean())
            d = side["A"][0] - side["B"][0]
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append({"family": r.family, "loss": r.loss, "H": H, "A": r.A, "B": r.B, "n_A": r.n_A, "n_B": r.n_B,
                         "diff": side["A"][1] - side["B"][1], "ci_low_seeds": lo, "ci_high_seeds": hi,
                         "excludes_zero": bool(lo > 0 or hi < 0)})
    out = pd.DataFrame(rows)
    if not np.allclose(out["diff"], comp.set_index(["family", "loss", "H", "A", "B"]).loc[
            list(zip(out.family, out.loss, out.H, out.A, out.B)), "diff"].to_numpy()):
        sys.exit("per-run series do not reproduce the planned differences")
    out.to_csv(ANALYSIS / "inference_seeds.csv", index=False)
    print(out[out.loss == "APE"][["family", "H", "A", "B", "diff", "ci_low_seeds", "ci_high_seeds"]].round(4).to_string(index=False))


# ==============================================================================================
# Breakdowns and aggregation (implementation note §9, §12; addendum 1 §6; descriptive, no p-values)
# ==============================================================================================
DAY_TYPES = ["religious holiday", "eve", "national holiday", "normal"]


def hour_categories(D: rx.Dataset) -> dict:
    """Category codes (-1 = none) for every hour of the data, used as masks on the TARGET hour (note §9):
    day type, 2025 event, month, hour of day and the daily-peak hour of each day of 2025. Holidays and
    eves are marked for all 24 hours of their day in the data."""
    cal = pd.read_csv(rx.DATA_FILE, usecols=["holiday_name", "holiday_type"])
    kind, name = cal.holiday_type.to_numpy(object), cal.holiday_name.fillna("").to_numpy(str)
    day_type = np.select([kind == "religious", kind == "half_day", kind == "national"], [0, 1, 2], 3)
    in2025 = np.arange(D.n) >= D.test_start
    events = ["Ramazan Bayramı 2025 (eve and 3 days)", "Kurban Bayramı 2025 (eve and 4 days)"]
    event = np.select([in2025 & np.char.startswith(name, "Ramazan"), in2025 & np.char.startswith(name, "Kurban")], [0, 1], -1)
    for n in dict.fromkeys(name[in2025 & (kind == "national")]):
        event[in2025 & (name == n)] = len(events)
        events.append(n)
    peak = np.full(D.n, -1)
    days = pd.Series(D.y[D.test_start:]).groupby(pd.Series(D.labels[D.test_start:]).str[:10].to_numpy())
    peak[D.test_start + days.idxmax().to_numpy()] = 0
    codes = {"day type": (day_type, DAY_TYPES), "event": (event, events),
             "month": (D.month - 1, [f"{m:02d}" for m in range(1, 13)]),
             "hour of day": (D.hour, [f"{h:02d}" for h in range(24)]), "daily peak": (peak, ["daily peak hour"])}
    return {f: (np.asarray(c).astype(np.int16), labels) for f, (c, labels) in codes.items()}


def weekly_path(D: rx.Dataset, z) -> np.ndarray:
    """Hourly forecasts of every 2025 hour from weekly updated H = 168 forecasts: origins 2025-01-01 00:00
    + 168 k (k = 0..51) and the first 24 hours of the extra origin 2025-12-31 00:00 (note §12)."""
    o = D.origins("test", 168)
    weekly, extra = np.arange(0, 52 * 168, 168), D.aggregation_origin()
    keys = z["test_origin_utc_ns"]
    if not (D.labels[o[0]] == "2025-01-01 00:00" and o[weekly[-1]] + 168 == extra
            and z["aggregation_extra_origin_utc_ns"][0] == D.utc_ns[extra] and np.array_equal(keys, D.utc_ns[o])):
        sys.exit("the weekly aggregation schedule does not match the stored origins (note §12)")
    path = np.concatenate([z["test_forecast_mwh"][weekly].astype(np.float64).ravel(),
                           z["aggregation_extra_forecast_mwh"][0, :24].astype(np.float64)])
    targets = np.concatenate([(o[weekly][:, None] + np.arange(168)).ravel(), extra + np.arange(24)])
    if not np.array_equal(targets, np.arange(D.test_start, D.n)):
        sys.exit("the weekly aggregation does not cover every 2025 hour exactly once (note §12)")
    return path


def energy_measures(f: np.ndarray, y: np.ndarray, month: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Monthly energy of an hourly path: energy APE, cancellation |sum e| / sum |e| ('n/a' when sum |e| = 0)."""
    e = f - y
    rows = []
    for m in range(1, 13):
        k = month == m
        a = np.abs(e[k]).sum()
        rows.append({"month": f"{m:02d}", "actual_GWh": y[k].sum() / 1e3, "forecast_GWh": f[k].sum() / 1e3,
                     "energy_error_GWh": e[k].sum() / 1e3, "energy_APE": 100 * abs(e[k].sum()) / y[k].sum(),
                     "cancellation": abs(e[k].sum()) / a if a > 0 else np.nan,
                     "hourly_MAPE": 100 * np.mean(np.abs(e[k]) / y[k])})
    monthly = pd.DataFrame(rows)
    a = np.abs(e).sum()
    return monthly, {"hourly_MAPE": 100 * np.mean(np.abs(e) / y), "mean_monthly_energy_APE": monthly.energy_APE.mean(),
                     "max_monthly_energy_APE": monthly.energy_APE.max(),
                     "month_of_max": monthly.month[monthly.energy_APE.idxmax()],
                     "annual_energy_bias_pct": 100 * e.sum() / y.sum(),
                     "annual_cancellation": abs(e.sum()) / a if a > 0 else np.nan}


def build_breakdowns() -> None:
    """Target-hour breakdowns of every run (system values = means over seeds, like the primary loss), the
    F3_m24 - F3 check on leads 1-145, and the monthly aggregation of the reported H = 168 systems."""
    D = rx.Dataset()
    families = hour_categories(D)
    runs = pd.read_csv(ANALYSIS / "run_metrics.csv")
    systems = pd.read_csv(ANALYSIS / "systems.csv")
    run_rows, count_rows = [], []
    for H in HORIZONS:
        o = D.origins("test", H)
        t = o[:, None] + np.arange(H)[None, :]
        y = D.y[t]
        codes = {f: c[t] for f, (c, _) in families.items()}
        for f, (_, labels) in families.items():
            c = codes[f]
            for k, label in enumerate(labels):
                hit = c == k
                count_rows.append({"H": H, "family": f, "category": label, "pairs": int(hit.sum()),
                                   "origins": int(hit.any(axis=1).sum()), "target_hours": int(np.unique(t[hit]).size)})
        for r in runs[runs.H == H].itertuples():
            key = "test" if r.variant == "main" else f"test@{r.variant}"
            with np.load(RUNS / r.run_id / "test_sealed.npz") as z:
                e = z[f"{key}_forecast_mwh"].astype(np.float64) - y
            ape, pe, under = 100 * np.abs(e) / y, 100 * e / y, (e < 0).astype(float)
            for f, (_, labels) in families.items():
                c = codes[f].ravel()
                v = c >= 0
                n = np.bincount(c[v], minlength=len(labels))
                sums = [np.bincount(c[v], weights=x.ravel()[v], minlength=len(labels)) for x in (ape, pe, under)]
                run_rows += [{"run_id": r.run_id, "variant": r.variant, "H": H, "family": f, "category": labels[k],
                              "MAPE": sums[0][k] / n[k], "bias_pct": sums[1][k] / n[k], "under_share": sums[2][k] / n[k]}
                             for k in range(len(labels)) if n[k]]
            if H == 168:                  # common lead range of the 12 h and 24 h vintage margins (addendum 1 §6)
                run_rows.append({"run_id": r.run_id, "variant": r.variant, "H": H, "family": "leads", "category": "1-145",
                                 "MAPE": ape[:, :145].mean(), "bias_pct": pe[:, :145].mean(), "under_share": under[:, :145].mean()})
        print(f"H={H}: breakdowns of {int((runs.H == H).sum())} run variants")
    per_run = pd.DataFrame(run_rows)
    counts = pd.DataFrame(count_rows)

    # system values: the mean over the seeds of the runs' values (systems scored from a combined forecast are skipped)
    parts = systems[~systems.kind.isin(["P seed mean", "ensemble"])].assign(part=lambda s: s.runs.str.split(";")).explode("part")
    parts[["run_id", "variant"]] = parts.part.str.split("|", expand=True)
    merged = parts[["system", "kind", "group", "model", "arm", "H", "n_seeds", "run_id", "variant"]].merge(
        per_run, on=["run_id", "variant", "H"])
    by_system = (merged.groupby(["system", "kind", "group", "model", "arm", "H", "n_seeds", "family", "category"], sort=False)
                 [["MAPE", "bias_pct", "under_share"]].mean().reset_index()
                 .merge(counts, on=["H", "family", "category"], how="left"))

    # monthly energy of weekly updated H = 168 forecasts: seed-mean path (primary) and per-seed paths (note §12)
    hours = np.arange(D.test_start, D.n)
    y, month = D.y[hours], D.month[hours]
    monthly_rows, summary_rows = [], []
    for s in systems[(systems.kind == "P") & (systems.H == 168)].itertuples():
        paths = []
        for part in s.runs.split(";"):
            with np.load(RUNS / part.split("|")[0] / "test_sealed.npz") as z:
                paths.append(weekly_path(D, z))
        monthly, summary = energy_measures(np.mean(paths, axis=0), y, month)
        per_seed = [energy_measures(p, y, month)[1] for p in paths]
        monthly_rows.append(monthly.assign(system=s.system, model=s.model, regime=s.arm, n_seeds=s.n_seeds))
        summary_rows.append({"system": s.system, "group": s.group, "model": s.model, "regime": s.arm,
                             "n_seeds": s.n_seeds, **summary,
                             "per_seed_mean_monthly_energy_APE": np.mean([p["mean_monthly_energy_APE"] for p in per_seed]),
                             "per_seed_annual_energy_bias_pct": np.mean([p["annual_energy_bias_pct"] for p in per_seed])})
    per_run.to_csv(ANALYSIS / "breakdowns_runs.csv", index=False)
    by_system.to_csv(ANALYSIS / "breakdowns_systems.csv", index=False)
    pd.concat(monthly_rows).to_csv(ANALYSIS / "aggregation_monthly.csv", index=False)
    pd.DataFrame(summary_rows).sort_values("mean_monthly_energy_APE").to_csv(ANALYSIS / "aggregation_summary.csv", index=False)
    print(f"breakdowns of {by_system.system.nunique()} systems and the monthly aggregation of {len(summary_rows)} "
          f"H=168 systems -> {ANALYSIS}")


# ==============================================================================================
# Tables (T1-T6 of the main text and the supplementary tables; values only, rounding in the paper)
# ==============================================================================================
TABLES = ANALYSIS / "tables"
METHOD_CLASS = {"TCN": "trained from scratch", "TCN (known future)": "trained from scratch",
                "RNN": "trained from scratch", "NF": "trained from scratch", "LightGBM": "trained from scratch",
                "foundation": "zero-shot (pretrained)", "foundation (LoRA)": "fine-tuned (LoRA)",
                "Vanilla": "linear regression (OLS)", "naive": "no training"}
F_BRANCH = {"F0": "calendar", "F1": "calendar", "F1A0": "holiday analog", "F1A": "holiday analog",
            "F2": "weather", "F3": "weather", "F3raw": "weather", "F4": "weather",
            "F3_m24": "weather (F3 checkpoint)", "F3_diag": "weather (F3 checkpoint)",
            "F3raw_diag": "weather (F3 checkpoint)"}
F_ORDER = list(F_BRANCH)


def run_table() -> pd.DataFrame:
    """runs.csv rows of the main variant with the learning rate actually used and the validation MAPE."""
    cfg = rx.load_config()
    runs = pd.read_csv(ROOT / "results" / "runs.csv")
    runs = runs[runs.variant.isna()].copy()
    trained = runs.model.map(rx.family).isin(["torch", "nf"])
    runs["lr_used"] = np.where(runs.lr.notna(), runs.lr, np.where(trained, float(cfg["training"]["learning_rate"]), np.nan))
    return runs


def table_p_systems(systems: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """T2 and supplement: every reported P system with seed statistics, validation MAPE (seed 42, the value
    that selected the window, and the panel mean) and size; seed-mean systems and ensembles below."""
    p = systems[systems.kind.isin(["P", "P seed mean", "ensemble"])].copy()
    first = p.runs.str.split(";").str[0].str.split("|").str[0]
    info = runs.set_index("run_id")
    p["params"] = first.map(info.params)
    p["lr"] = first.map(info.lr_used)
    val = {s: runs[runs.run_id.isin([r.split("|")[0] for r in rs.split(";")])] for s, rs in zip(p.system, p.runs)}
    p["val_MAPE_seed42"] = [v[v.seed == 42].val_MAPE.mean() for v in val.values()]
    p["val_MAPE_mean"] = [v.val_MAPE.mean() for v in val.values()]
    p.loc[p.kind == "ensemble", ["params", "lr", "val_MAPE_seed42", "val_MAPE_mean"]] = np.nan
    p["method_class"] = p.group.map(METHOD_CLASS).fillna("combination")
    cols = ["kind", "H", "group", "method_class", "model", "arm", "w", "lr", "params", "n_seeds", "seeds",
            "val_MAPE_seed42", "val_MAPE_mean", "MAPE", "MAPE_sd", "MAPE_ci95", "RMSE", "RMSE_sd", "MAE", "MASE",
            "bias_pct", "under_share", "note"]
    order = {"P": 0, "P seed mean": 1, "ensemble": 2}
    return (p.assign(o=p.kind.map(order)).sort_values(["o", "H", "arm", "MAPE"])[cols]
            .rename(columns={"arm": "regime"}))


def table_design(D: rx.Dataset, systems: pd.DataFrame) -> pd.DataFrame:
    """T1: data, splits, origins, information sets and the systems run per family (long format)."""
    rows = [("data", "source", "EPİAŞ Transparency Platform, real-time consumption (national, hourly, MWh)"),
            ("data", "period", f"{D.labels[0]} to {D.labels[-1]} local time (UTC+2 until 2016-03-27, then UTC+3), "
                               f"{D.n:,} hours"),
            ("split", "training", f"2016–2023, {D.train_end:,} hours"),
            ("split", "validation", f"2024, {D.test_start - D.val_start:,} hours; F arms from {D.labels[D.weather.f_val_start][:10]}"),
            ("split", "test", f"2025, {D.n - D.test_start:,} hours")]
    for H in HORIZONS:
        rows.append(("origins", f"H = {H}", f"validation {len(D.origins('val', H)):,} (F arms {len(D.f_origins('val', H)):,}); "
                                            f"test {len(D.origins('test', H)):,}"))
    info_sets = {"P uni": "load history only", "P cal": "load history and the calendar of the input hours",
                 "F0": "F1 architecture trained with the future channels set to zero (zero-shot models: their cal system)",
                 "F1": "+ calendar of the target hours",
                 "F1A0": "F1 + holiday-analog availability flag (value masked; exploratory)",
                 "F1A": "F1 + holiday analog (relative deviation of last year's holiday or eve load from its normal-day level, rescaled and standardised; exploratory)",
                 "F2": "F1 + past temperature", "F3": "F2 + forecast temperature of the target hours (bias-corrected)",
                 "F3raw": "F3 without bias correction (own checkpoint)", "F4": "ERA5 in past and future (oracle weather scenario)",
                 "F3_m24": "F3 checkpoint, 24 h vintage margin", "F3_diag": "F3 checkpoint, ERA5 future temperature",
                 "F3raw_diag": "F3 checkpoint, raw forecasts"}
    rows += [("information set", k, v) for k, v in info_sets.items()]
    for (group, model), g in systems[systems.kind.isin(["P", "F"])].groupby(["group", "model"]):
        p, f = g[g.kind == "P"], g[g.kind == "F"]
        seeds = (f"P {'/'.join(map(str, sorted(set(p.n_seeds))))}" if len(p) else "") + (
            f"{'; ' if len(p) and len(f) else ''}F {'/'.join(map(str, sorted(set(f.n_seeds))))}" if len(f) else "")
        rows.append(("systems", model, f"{group}; {METHOD_CLASS.get(group, '')}; "
                                       f"P: {', '.join(sorted(set(p.arm))) or '-'}; "
                                       f"F: {', '.join(a for a in F_ORDER if a in set(f.arm)) or '-'}; "
                                       f"horizons {', '.join(map(str, sorted(set(g.H))))}; seeds {seeds}"))
    return pd.DataFrame(rows, columns=["section", "item", "value"])


def table_information(systems: pd.DataFrame) -> pd.DataFrame:
    """T3 (systems): every F arm with its branch and seed statistics, main model first."""
    f = systems[systems.kind == "F"].copy()
    f["branch"] = f.arm.map(F_BRANCH)
    order = {m: i for i, m in enumerate(["LightTCN_F_medium", "LightGBM", "NF_NHITS", "Chronos2", "TimesFM3", "TiRex2",
                                         "Vanilla", "Vanilla_holiday"])}
    f = f.assign(o=f.model.map(order), a=f.arm.map({a: i for i, a in enumerate(F_ORDER)})).sort_values(["o", "H", "a"])
    return f[["model", "H", "arm", "branch", "n_seeds", "MAPE", "MAPE_sd", "MAPE_ci95", "RMSE", "MAE", "bias_pct", "note"]]


def table_contrasts(inf: pd.DataFrame) -> pd.DataFrame:
    """T3 (contrasts) and T4: comparisons with intervals and adjusted p-values, both losses."""
    out = inf.copy()
    out["first"] = np.where(out.family.str.startswith("information"), out.B.str.split("|").str[1], "")
    out["second"] = np.where(out.family.str.startswith("information"), out.A.str.split("|").str[1], "")
    cols = ["family", "loss", "H", "A", "B", "question", "type", "exploratory", "n_A", "n_B", "mean_A", "mean_B",
            "diff", "rel_diff_pct", "ci_low", "ci_high", "ci_low_half_b", "ci_high_half_b", "ci_low_double_b",
            "ci_high_double_b", "N", "block", "blocks_in_N", "p_boot", "p_holm", "significant", "dm_hln", "p_dm",
            "diff_seed42", "note"]
    return out[cols]


def table_matched(rm: pd.DataFrame, systems: pd.DataFrame) -> pd.DataFrame:
    """T5: interventions against the full system on the same seeds, horizon, window and learning rate:
    single calendar-group ablations, the publication delay and the short windows (note §10)."""
    main = rm[rm.variant == "main"]
    cells = systems[systems.kind == "P"]
    rows = []
    arms = main[main.regime.isin(rx.ABLATION_GROUPS) | (main.gap > 0) | main.experiments.str.contains("short_windows")]
    for (model, regime, H, w, gap), g in arms.groupby(["model", "regime", "H", "w", "gap"]):
        base = "cal" if regime in rx.ABLATION_GROUPS else regime
        cell = cells[(cells.model == model) & (cells.arm == base) & (cells.H == H)].iloc[0]
        ids = [p.split("|")[0] for p in cell.runs.split(";")]
        ref = main[main.run_id.isin(ids) & main.seed.isin(g.seed)].set_index("seed")
        arm = g.set_index("seed")
        if set(ref.index) != set(arm.index):
            sys.exit(f"{model} {regime} H={H}: no matched reference for seeds {sorted(arm.index)} (note §10)")
        kind = "ablation" if regime in rx.ABLATION_GROUPS else ("publication delay" if gap else "short window")
        label = {"cal_no_hour": "without the hour encodings", "cal_no_week": "without the weekday group",
                 "cal_no_holiday": "without the holiday flag"}.get(regime, f"inputs end at t-{gap + 1}" if gap else f"w = {w}")
        paired = (arm.MAPE - ref.MAPE).to_numpy()
        rows.append({"kind": kind, "H": H, "model": model, "intervention": label, "reference": f"{base}, w = {cell.w}",
                     "seeds": "/".join(map(str, sorted(arm.index))), "n": len(arm),
                     "MAPE_reference": ref.MAPE.mean(), "MAPE_arm": arm.MAPE.mean(), "diff": paired.mean(),
                     "rel_diff_pct": 100 * paired.mean() / ref.MAPE.mean(),
                     "paired_sd": paired.std(ddof=1) if len(paired) > 1 else np.nan,
                     "RMSE_reference": ref.RMSE.mean(), "RMSE_arm": arm.RMSE.mean()})
    order = {"ablation": 0, "publication delay": 1, "short window": 2}
    return pd.DataFrame(rows).assign(o=lambda d: d.kind.map(order)).sort_values(["o", "H", "model", "intervention"]).drop(columns="o")


def table_grid(runs: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Supplement: the single-seed window grid (seed 42, selected learning rate) with validation and test
    MAPE; the selected window is marked. Test values are shown for completeness, never used to select."""
    cfg = rx.load_config()
    hashes = rx.data_hashes()
    ids = {rx.run_id(rx.make_contract(s, cfg, hashes)) for s in rx.expand_experiment(cfg, rx.SELECTION_EXPERIMENTS)}
    g = runs[runs.run_id.isin(ids)].copy()
    chosen = set(zip(cells.model, cells.regime, cells.H, cells.w))
    g["selected"] = [(m, r, h, w) in chosen for m, r, h, w in zip(g.model, g.regime, g.H, g.w)]
    return g.sort_values(["H", "model", "regime", "w"])[["H", "model", "regime", "w", "lr_used", "params",
                                                           "val_MAPE", "test_MAPE", "selected"]]


HOLIDAYS_EN = {"Ramazan Bayramı": "Eid al-Fitr", "Kurban Bayramı": "Eid al-Adha"}


def british(text):
    """'analogue' as the paper spells it (British English); the code and the analysis files keep 'analog'."""
    import re
    return re.sub(r"\b([Aa])nalog\b", r"\1nalogue", text) if isinstance(text, str) else text


def english_holidays(text):
    """Holiday names as the paper writes them: 'Eid al-Fitr (Ramazan Bayramı), day 2' where the data give both
    names, the English name alone otherwise. The data and the analysis files keep the Turkish names."""
    if not isinstance(text, str):
        return text
    for tr, en in HOLIDAYS_EN.items():
        if f"{tr} ({en})" in text:
            text = text.replace(f"{tr} ({en}) ", f"{en} ({tr}), ")
        elif f"{en} ({tr})" not in text:                    # already converted: leave as it is
            text = text.replace(tr, en)
    return text


def table_dataset(D: rx.Dataset) -> pd.DataFrame:
    """TD: the load series and the splits (values unrounded; MWh, TWh and hours)."""
    df = pd.read_csv(rx.DATA_FILE, usecols=["datetime_local", "consumption_mwh", "is_imputed", "holiday_name"])
    y = df.consumption_mwh
    annual = y.groupby(df.datetime_local.str[:4]).sum() / 1e6

    def when(i):
        name = df.holiday_name[i]
        return df.datetime_local[i] + (f", {english_holidays(name)}" if isinstance(name, str) and name else "")
    rows = [("Data source", "EPİAŞ Transparency Platform", "real-time consumption, national total"),
            ("Period", f"{df.datetime_local.iloc[0][:10]} to {df.datetime_local.iloc[-1][:10]}",
             "ten calendar years, local time (UTC+2 until 2016-03-27, then UTC+3)"),
            ("Resolution", "hourly", "MWh per hour"),
            ("Observations", len(df), f"consecutive hours; {int(df.is_imputed.sum())} imputed at the clock change of 2016-03-27"),
            ("Minimum", y.min(), when(int(y.idxmin()))), ("Maximum", y.max(), when(int(y.idxmax()))),
            ("Mean", y.mean(), "all hours"), ("Standard deviation", y.std(), "all hours"),
            (f"Annual consumption {annual.index[0]}", annual.iloc[0], "TWh"),
            (f"Annual consumption {annual.index[-1]}", annual.iloc[-1], "TWh"),
            ("Training set", D.train_end, "2016–2023, hours"), ("Validation set", D.test_start - D.val_start, "2024, hours"),
            ("Test set", D.n - D.test_start, "2025, hours")]
    return pd.DataFrame(rows, columns=["item", "value", "description"])


def table_features(D: rx.Dataset) -> pd.DataFrame:
    """TF: the input features with definition, observed range and the information sets that use them."""
    def span(a, unit):
        return f"{np.nanmin(a):.1f} to {np.nanmax(a):.1f} {unit}".replace("-", "−")
    t = D.temp                                            # ERA5: temperature, HDH (18 °C), CDH (22 °C)
    rows = [("hour_sin, hour_cos", "sin(2π·hour/24), cos(2π·hour/24)", "−1 to 1", "hour of day, local time", "cal, F"),
            ("dow_sin, dow_cos", "sin(2π·dow/7), cos(2π·dow/7); Monday = 0", "−1 to 1", "day of week", "cal, F"),
            ("is_weekend", "1 on Saturday and Sunday", "0 or 1", "weekend flag", "cal, F"),
            ("is_holiday", "1 on official full-day holidays", "0 or 1", "holiday flag (Law No. 2429; Diyanet dates)", "cal, F"),
            ("Load", "MWh; z-score with training-year statistics in the TCN and recurrent networks",
             f"{D.y.min():,.0f} to {D.y.max():,.0f} MWh", "target for all models; historical input where used", "all"),
            ("Temperature", "Σ wₚTₚ over the 81 provinces", span(t[:, 0], "°C"), "population-weighted 2 m temperature", "F2–F4"),
            ("HDH", "Σ wₚ max(0, 18 − Tₚ)", span(t[:, 1], "°C"), "heating degree-hours", "F2–F4"),
            ("CDH", "Σ wₚ max(0, Tₚ − 22)", span(t[:, 2], "°C"), "cooling degree-hours", "F2–F4"),
            ("Holiday analog", "(holiday load / prior normal-day level − 1) × current normal-day level / training-"
             "load SD; availability flag", "dimensionless; 0 or 1",
             "same holiday or eve one year earlier (exploratory)", "F1A0 (flag), F1A")]
    return pd.DataFrame(rows, columns=["feature", "definition", "range", "description", "used in"])


def sci(x: float) -> str:
    """1e-05 -> '1 × 10⁻⁵' (Word-ready)."""
    m, e = f"{x:.0e}".split("e")
    return f"{m} × 10" + str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def table_setup(runs: pd.DataFrame, D: rx.Dataset) -> pd.DataFrame:
    """TS: the experimental setup and the training hyperparameters, read from configs/experiments.yaml."""
    cfg = rx.load_config()
    tr, ex = cfg["training"], cfg["experiments"]
    short = sorted({w for b in ex["grid"]["blocks"] for w in b["windows"]})
    long_ = sorted({w for b in ex["long_horizon"]["blocks"] for w in b["windows"]})
    lrs = [sorted({x for b in ex["lr_search"]["blocks"] if (b["models"] == ["Chronos2_LoRA"]) == lora for x in b["lr"]},
                  reverse=True) for lora in (False, True)]
    eff_gpu = pd.read_csv(EFFICIENCY).device.dropna().iloc[0] if EFFICIENCY.exists() else "laptop GPU"
    rows = [("Forecast horizons", "1, 24, 168 and 720 h", "direct multi-output forecasts"),
            ("Input windows", f"{', '.join(map(str, short[:-1]))} and {short[-1]} h (H ≤ 168 h); "
                              f"{long_[0]:,} and {long_[1]:,} h (H = 720 h)",
             "one week to three months of history, selected per model, input setting and horizon on the validation "
             "MAPE (seed 42)"),
            ("Data split", "training 2016–2023, validation 2024, test 2025", "by calendar year; no target crosses a boundary"),
            ("Normalisation", "z-score of load and temperature, training-year statistics",
             "NeuralForecast: per window; foundation models: internal"),
            ("Loss", f"Huber, δ = {tr['huber_delta']}", "on the standardised load: errors above one training-year "
             f"standard deviation (about {round(D.y[:D.train_end].std(), -2):,.0f} MWh) count linearly, which limits the "
             "weight of rare large errors; NeuralForecast: on the per-window standardised targets"),
            ("Optimiser", f"Adam, weight decay {sci(tr['weight_decay'])}", ""),
            ("Learning rate", f"{' or '.join(sci(x) for x in lrs[0])} (LoRA: {' or '.join(sci(x) for x in lrs[1])})",
             "chosen per model on the validation MAPE at H = 24 h, w = 168 h"),
            ("Learning-rate schedule", f"reduce on plateau: factor {tr['scheduler_factor']}, patience {tr['scheduler_patience']}, "
                                       f"minimum {sci(tr['scheduler_min_lr'])}", "validation loss"),
            ("Early stopping", f"patience {tr['early_stopping_patience']} epochs", "validation loss"),
            ("Maximum epochs", str(tr["max_epochs"]), ""),
            ("Batch size", f"{tr['batch_size']} windows", "accumulated over two half-batches for DualTCN-L and "
             "DualTCN-Attn-L at w ≥ 1,440 h (activation memory)"),
            ("Precision", "mixed precision in training, full precision in inference (NeuralForecast: mixed precision in "
                          "both)" if tr["use_amp"] else "full precision",
             "convolutions in PyTorch's default TF32 mode"),
            ("Gradient clipping", f"L2 norm {tr['gradient_clip_norm']}", "limits rare large updates"),
            ("Seeds", ", ".join(map(str, cfg["seeds_panel"])),
             f"three ({', '.join(map(str, cfg['seeds_small']))}) for LoRA and the F arms of the networks; one for most "
             "H = 720 h systems; each seed fixes the Python, NumPy and PyTorch generators, and cuDNN "
             "runs in deterministic mode"),
            # the training device as the paper describes it: GPU partitions and their memory
            ("Training hardware", "NVIDIA data-centre GPU partitions", "20 GB of memory each"),
            ("Profile hardware", f"{eff_gpu} (16 GB), Intel Core i9-13900HX", "efficiency profile (Section 6.6 of the paper)")]
    return pd.DataFrame(rows, columns=["parameter", "value", "note"])


def table_efficiency(systems: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """TE: the laptop profile of every reported system with its test MAPE, and the campaign's training cost of
    the same run (wall time on a GPU partition, a different device; LightGBM and Vanilla on the CPU)."""
    e = pd.read_csv(EFFICIENCY)
    e = e[e.e2e_b1_p50_ms.notna()].copy()
    s = systems.drop_duplicates("system").set_index("system")
    e["MAPE"] = s.loc[e.system, "MAPE"].to_numpy()
    e["n_seeds"] = s.loc[e.system, "n_seeds"].to_numpy()
    info = runs.drop_duplicates("run_id").set_index("run_id")
    e["train_wall_min"] = e.run_id.map(info.wall_sec) / 60
    e["epochs_run"] = e.run_id.map(info.epochs_run)
    e["train_peak_gpu_gib"] = e.run_id.map(info.peak_gpu_mem_mb) / 1024
    first =["system", "group", "model", "regime", "H", "w", "run_id", "n_seeds", "MAPE"]
    return e[first + [c for c in e.columns if c not in first]].sort_values(["H", "group", "model", "regime"])


def table_lr_search(runs: pd.DataFrame) -> pd.DataFrame:
    """Supplement: the learning-rate search (H = 24 h, w = 168 h, seed 42) with the validation MAPE of each
    candidate and the selected rate."""
    sel = rx.load_selected()
    ids = sel["manifest"]["learning_rate"]["candidates"]
    g = runs[runs.run_id.isin(ids)][["model", "regime", "lr_used", "val_MAPE", "epochs_run", "wall_sec"]].copy()
    g["selected"] = [abs(lr - sel["learning_rate"][m]) < 1e-12 for m, lr in zip(g.model, g.lr_used)]
    return g.sort_values(["model", "lr_used"], ascending=[True, False])


def table_literature(D: rx.Dataset, systems: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """T12 and S1: the studies of manuscript/literature.csv and this study's row, whose numbers are computed (the
    1 h error with the publication-delay result next to it, as everywhere in the paper)."""
    lit = pd.read_csv(ROOT / "manuscript" / "literature.csv")
    p = systems[systems.kind == "P"]
    best = [p[p.H == H].sort_values("MAPE").iloc[0] for H in HORIZONS]
    f3 = systems[(systems.arm == "F3") & (systems.H == 168)].sort_values("MAPE").iloc[:2]
    delay = matched[(matched.kind == "publication delay") & (matched.model == "LightTCN_medium") & (matched.H == 1)].iloc[0]
    lowest = [f"{b.H} h: {b.MAPE:.3f}% ({NAMES[b.model]}"
              + (f"; previous hour's load known, with a 2 h publication delay the MAPE of LightTCN-M rose from "
                 f"{delay.MAPE_reference:.2f}% to {delay.MAPE_arm:.2f}%)" if b.H == 1 else ")") for b in best]
    this = {"key": "This work", "main": 1, "region": "Türkiye",
            "data": "EPİAŞ national hourly consumption with calendar, holiday analog and population-weighted temperature",
            "period": f"2016–2025 ({D.n:,} h)", "resolution": "Hourly", "horizon": "1, 24, 168 and 720 h",
            "window": "168–2,160 h", "model": "26 models: TCNs, recurrent, NeuralForecast, LightGBM, linear benchmark, "
                                             "foundation models (zero-shot and LoRA)",
            "results": "Lowest past-only MAPE " + "; ".join(lowest) + "; with forecast temperature, 168 h: "
                       + " and ".join(f"{r.MAPE:.3f}% ({NAMES[r.model]})" for r in f3.itertuples())}
    return pd.concat([lit, pd.DataFrame([this])], ignore_index=True)


def table_training(systems: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """S16: the campaign's training cost of every past-only system (means over its runs; peak memory: maximum)."""
    info = runs.drop_duplicates("run_id").set_index("run_id")
    rows = []
    for s in systems[systems.kind == "P"].sort_values(["H", "model", "arm"]).itertuples():
        r = info.loc[info.index.intersection([q.split("|")[0] for q in s.runs.split(";")])]
        rows.append({"H": s.H, "model": s.model, "regime": s.arm, "w": s.w, "n_seeds": s.n_seeds,
                     "epochs_mean": r.epochs_run.mean(), "wall_min_mean": r.wall_sec.mean() / 60,
                     "peak_gpu_gib_max": r.peak_gpu_mem_mb.max() / 1024,
                     "peak_reserved_gib_max": r.peak_gpu_reserved_mb.max() / 1024})
    return pd.DataFrame(rows)


def table_models(systems: pd.DataFrame, runs: pd.DataFrame, p2: pd.DataFrame) -> pd.DataFrame:
    """S4: every model with its class, the parameter range over its validation-selected past-only systems and
    known-future arms (pretrained weights included), its past-only inputs and its known-future arms."""
    classes = {"Foundation, zero-shot": "pretrained, zero-shot"}
    allruns = pd.read_csv(ROOT / "results" / "runs.csv")              # every variant: Vanilla F34 has no main row
    rows = []
    for family, members in P_ROWS + [("Known-future variant", [("LightTCN_F_medium", "F")]),
                                     ("Linear benchmark", [("Vanilla", "F"), ("Vanilla_holiday", "F")])]:
        for m in dict.fromkeys(mm for mm, _ in members):
            s = systems[systems.model == m]
            f_runs = allruns[(allruns.model == m) & allruns.regime.astype(str).str.startswith("F")]
            k = pd.concat([p2[(p2.model == m) & (p2.kind == "P")].params, f_runs.params]).dropna()
            k = k[k > 0]
            cls = ("fine-tuned (LoRA)" if m == "Chronos2_LoRA" else "gradient boosting" if m == "LightGBM"
                   else "linear regression" if m.startswith("Vanilla") else classes.get(family, "trained from scratch"))
            rows.append({"model": m, "name": NAMES.get(m, m), "model_class": cls,
                         "params_min": k.min() if len(k) else np.nan, "params_max": k.max() if len(k) else np.nan,
                         "P_inputs": ", ".join(sorted(set(s[s.kind == "P"].arm))) or "–",
                         "F_arms": ", ".join(a for a in ("F0", "F1", "F1A0", "F1A", "F2", "F3", "F4")
                                             if a in set(s[s.kind == "F"].arm)) or "–"})
    rows += [{"model": m, "name": NAMES[m], "model_class": "naive reference", "params_min": np.nan, "params_max": np.nan,
              "P_inputs": "uni", "F_arms": "–"} for m in rx.BASELINE_MODELS if m in NAMES]
    return pd.DataFrame(rows)


def table_runs() -> pd.DataFrame:
    """S18: the runs of the campaign by experiment block and horizon, with their run-hours (unrounded)."""
    rr = pd.read_csv(ROOT / "results" / "runs.csv").drop_duplicates("run_id")

    def block(x):                                        # experiment blocks of the campaign, in the order of the paper
        if x.gap == x.gap and x.gap > 0:
            return 6, "Publication delay (inputs end at t − 3)"
        if str(x.regime).startswith("cal_no"):
            return 5, "Calendar ablations of LightTCN-M"
        if x.w in (24, 48):
            return 7, "Short windows (24 and 48 h)"
        if str(x.regime).startswith("F"):
            return 4, "Known-future arms (F0–F4), incl. Vanilla"
        if x.model == "Chronos2_LoRA":
            return 3, "Chronos-2 LoRA, past-only (incl. its learning-rate runs)"
        if x.seed == 42:
            return 1, "Past-only, seed 42 (learning rates, windows, zero-shot, LightGBM, naive)"
        return 2, "Past-only, further seeds at the selected windows"
    rr[["order", "block"]] = rr.apply(lambda x: pd.Series(block(x)), axis=1)
    rr["gpu"] = rr.device.str.contains("NVIDIA", na=False)
    rows = []
    for (o, b), g in rr.groupby(["order", "block"]):
        rows.append({"block": b, **{f"runs_H{H}": int((g.H == H).sum()) for H in HORIZONS}, "runs": len(g),
                     "run_hours": g.wall_sec.sum() / 3600, "run_hours_gpu": g.wall_sec[g.gpu].sum() / 3600})
    return pd.DataFrame(rows)


WORKBOOK_NAMES = [  # code names in the Supplementary Data and what the paper calls them
    ("uni", "past-only input setting: load history only"),
    ("cal", "past-only input setting: load history and the calendar of the input hours"),
    ("F0, F1, F1A0, F1A, F2, F3, F3raw, F4", "known-future arms (Supplementary Table S3)"),
    ("F3_m24, F3_diag, F3raw_diag", "diagnostic arms that reuse the F3 checkpoint (Supplementary Table S3)"),
    ("cal_no_hour, cal_no_week, cal_no_holiday", "calendar ablations: without the hour encodings, the weekday group "
                                                 "or the holiday flag"),
    ("model|setting|H<h>", "a system: model, input setting or arm, and horizon"),
    ("…|gap2", "publication-delay arm: the inputs end at t − 3 instead of t − 1"),
    ("…|w24, …|w48", "short-window arms (w = 24 or 48 h)"),
    ("APE, SE", "absolute percentage error (primary loss), squared error (secondary loss)")]


def write_workbook(sheets: dict, notes: list[tuple[str, str]]) -> Path:
    """One Excel workbook of plain values (no formulas): a notes sheet, then one sheet per table."""
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter
    path = TABLES / "tables.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        xw.book.properties.creator = ""                 # no author name in the file properties
        pd.DataFrame(notes, columns=["Sheet", "Table or figure", "Content"]).map(british).to_excel(
            xw, sheet_name="Notes", index=False)
        pd.DataFrame([(k, v) for k, v in NAMES.items()] + WORKBOOK_NAMES,
                     columns=["Code in the sheets", "Name in the paper"]).to_excel(xw, sheet_name="Names", index=False)
        for name, df in sheets.items():
            df.apply(lambda c: c.map(lambda v: british(english_holidays(v))) if c.dtype == object else c).to_excel(
                xw, sheet_name=name, index=False)
        for ws in xw.book.worksheets:
            for row in ws.iter_rows():
                for c in row:
                    c.font = Font(name="Arial", size=10, bold=c.row == 1)
                    if isinstance(c.value, float):
                        c.number_format = "0.0000"
            ws.freeze_panes = "A2"
            for i, col in enumerate(ws.columns, start=1):
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[get_column_letter(i)].width = min(max(10, width + 2), 60 if ws.title != "Notes" else 110)
            if ws.title == "Names":
                ws.column_dimensions["B"].width = 90
            if ws.title == "Notes":
                ws.column_dimensions["B"].width = 34
                for c in [*ws["B"], *ws["C"]]:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
    return path


def build_tables() -> None:
    """Every table of the paper and the supplement from the cached analysis files (run `losses`, `inference`
    and `breakdowns` first). Values are unrounded; the paper rounds them."""
    D = rx.Dataset()
    systems = pd.read_csv(ANALYSIS / "systems.csv")
    rm = pd.read_csv(ANALYSIS / "run_metrics.csv")
    inf = pd.read_csv(ANALYSIS / "inference.csv")
    runs = run_table()
    cells = reported_cells(rx.load_config(), rx.load_selected(), rx.data_hashes())
    TABLES.mkdir(parents=True, exist_ok=True)
    t = {"TD_dataset": table_dataset(D), "TF_features": table_features(D), "TS_setup": table_setup(runs, D),
         "T1_design": table_design(D, systems),
         "T2_p_systems": table_p_systems(systems, runs),
         "T3_information_systems": table_information(systems)}
    contrasts = table_contrasts(inf)
    t["T3_information_contrasts"] = contrasts[contrasts.family.str.startswith("information")]
    t["T4_comparisons"] = contrasts[~contrasts.family.str.startswith("information")]
    if (ANALYSIS / "inference_seeds.csv").exists():                 # exploratory seed-and-block intervals
        seeds = pd.read_csv(ANALYSIS / "inference_seeds.csv")[["family", "loss", "H", "A", "B", "ci_low_seeds", "ci_high_seeds"]]
        t["T4_comparisons"] = t["T4_comparisons"].merge(seeds, on=["family", "loss", "H", "A", "B"], how="left")
    t["T5_matched"] = table_matched(rm, systems)
    t["T6_aggregation"] = pd.read_csv(ANALYSIS / "aggregation_summary.csv")
    t["S_grid"] = table_grid(runs, cells)
    t["S_aggregation_monthly"] = pd.read_csv(ANALYSIS / "aggregation_monthly.csv")
    b = pd.read_csv(ANALYSIS / "breakdowns_systems.csv")
    t["S_day_type_events"] = b[b.family.isin(["day type", "event", "daily peak"])]
    t["S_months_hours"] = b[b.family.isin(["month", "hour of day"])]
    t["S_leads_1_145"] = b[b.family == "leads"]
    if EFFICIENCY.exists():
        t["TE_efficiency"] = table_efficiency(systems, runs)
    t["T12_literature"] = table_literature(D, systems, t["T5_matched"])
    t["S_lr_search"] = table_lr_search(runs)
    t["S_training"] = table_training(systems, runs)
    t["S_runs"] = table_runs()
    t["S_models"] = table_models(systems, runs, t["T2_p_systems"])
    for name, f in (("S_attention_matched", "diagnostics_attention_matched.csv"), ("S_gates", "diagnostics_gates.csv"),
                    ("S_attention_summary", "diagnostics_attention_summary.csv"),
                    ("S_attention_weights", "diagnostics_attention.csv")):
        if (ANALYSIS / f).exists():
            t[name] = pd.read_csv(ANALYSIS / f)
    for name, df in t.items():
        df.to_csv(TABLES / f"{name}.csv", index=False)

    def supp(key):
        return f"Supplementary Table {table_label(key)}"
    notes = [  # the Supplementary Data workbook: which sheet holds which table of the paper and the supplement
        ("Notes", "",
         "Supplementary Data. Every value is computed by analyze_results.py from the stored forecasts and is unrounded. "
         "MAPE, bias_pct and rel_diff_pct are in %; diff and the confidence limits of MAPE differences in percentage "
         "points; squared-error differences in MWh²; RMSE and MAE in MWh; n_seeds is the number of training seeds of a "
         "system. Bias = mean of 100·(forecast − actual)/actual; negative values mean under-forecasting. MAPE_ci95 is "
         "the half-width of the t-based 95% interval over seeds; under_share is the fraction (0-1) of forecasts below "
         "the actual load; n_A and n_B are the seeds of the compared systems; ci_low_seeds and ci_high_seeds are the "
         "exploratory intervals that also resample the seeds. Errors at H = 1 h assume that the previous hour's load "
         "is known (nominal availability; see the sheet of Supplementary Table S12 for the two-hour publication "
         "delay). The sheet Names maps the code names of models, settings and arms to the names in the paper; the "
         "comparison questions quote the frozen analysis plan, with 'validation-selected' for its 'best'."),
        ("TD_dataset", "Table 1", "The load series and the calendar-year split (MWh, TWh, hours)."),
        ("T4_comparisons", f"Table {table_label('T4')} and {supp('comparisons')}",
         "The 11 primary and 3 secondary comparisons under the absolute percentage error (APE, primary) and the "
         "squared error (SE, secondary): diff = A − B, negative values favour A; moving-block bootstrap intervals "
         "with b = max(2H, 168) h, 0.5b and 2b; Holm p within each family and loss; Diebold–Mariano (HLN) statistic."),
        ("T3_information_systems", f"Table {table_label('T3')} and {supp('f_systems')}",
         "Known-future (F) arms of every model, with seed statistics; holiday-analog arms (F1A0, F1A) are exploratory."),
        ("T12_literature", supp("literature"), "Literature comparison (manuscript/literature.csv) and the computed "
                                               "row of this study."),
        ("TF_features", supp("features"), "Input features: definition, observed range and information sets."),
        ("T1_design", supp("design"), "Data, splits, origins, information sets and the systems run per model."),
        ("S_models", supp("models"), "Models: class, parameter range (params_min, params_max) over the "
                                         "validation-selected past-only systems and known-future arms, past-only "
                                         "inputs and known-future arms."),
        ("TS_setup", supp("setup"), "Experimental setup and training hyperparameters (configs/experiments.yaml)."),
        ("S_lr_search", supp("lr"), "Learning-rate search at H = 24 h, w = 168 h, seed 42, and the selected rate."),
        ("T2_p_systems", supp("p_systems"),
         "Validation-selected past-only (P) systems, then the seed-mean forecasts and the family ensembles. MAPE_sd and "
         "MAPE_ci95 over seeds (empty for one seed); val_MAPE_seed42 selected the window."),
        ("S_grid", supp("grid"), "Window grid (seed 42); the window was selected on the validation MAPE only."),
        ("T3_information_contrasts", f"Fig. {FIGURE_NUMBERS['information']} and {supp('contrasts')}",
         "Information contrasts: diff = second arm − first arm, negative values mean the added information helped; "
         "rel_diff_pct relative to the first arm; Holm within each model and horizon; analog contrasts exploratory."),
        ("T5_matched", supp("matched"), "Ablations, publication delay and short windows against the full system on the "
                                        "same seeds: diff = arm − reference."),
        ("S_day_type_events", f"Fig. {FIGURE_NUMBERS['profiles']} and {supp('day_types')}",
         "Errors by type of target hour (religious holiday, eve, national holiday, normal), 2025 event and daily peak hour."),
        ("S_months_hours", f"Fig. {FIGURE_NUMBERS['profiles']}", "Errors by target month and hour of day."),
        ("T6_aggregation", f"Fig. {FIGURE_NUMBERS['energy']} and {supp('energy_summary')}",
         "Monthly energy of the weekly updated H = 168 h forecasts (seed-mean forecast of each system); "
         "annual_cancellation = |Σe| / Σ|e|."),
        ("S_aggregation_monthly", supp("energy_months"), "Monthly energy errors per system and month."),
        ("S_leads_1_145", "Supplementary Note S4", "H = 168 h errors on leads 1–145, which the 12 h and 24 h vintage "
                                                   "margins both cover with forecasts."),
        ("TE_efficiency", f"Fig. {FIGURE_NUMBERS['efficiency']} and {supp('efficiency')}",
         "Laptop efficiency profile: cold start (s), end-to-end and model-only latency (ms, p50 and p95), throughput "
         "(forecasts/s), GPU memory (MiB) and sizes; train_wall_min is the campaign wall time of the same run on a "
         "data-centre GPU partition (zero-shot models: inference only)."),
        ("S_training", supp("training"), "Campaign wall time per past-only system (training plus validation and test "
                                         "forecasts, or inference only for zero-shot models), mean epochs and maximum "
                                         "peak allocated and reserved GPU memory (GiB); the pretraining cost of the "
                                         "foundation models is unknown, not zero."),
        ("S_runs", supp("runs"), "Runs of the campaign by experiment block and horizon, with run-hours in total and "
                                 "on the GPU partitions."),
        ("S_attention_matched", supp("attention_matched"), "DualTCN-Attn-M against DualTCN-M with equal window, "
                                                           "learning rate and seed (selection grid)."),
        ("S_gates", supp("gates"), "Gate statistics and learning curves of the selected seed-42 networks."),
        ("S_attention_summary", supp("attention_summary"), "Attention mass per group of input hours at four origins."),
        ("S_attention_weights", f"Supplementary Fig. S{SUPP_FIGURES.index('attention') + 1}", "Attention weight of every input hour at the four origins."),
    ]
    def lab(key):                                       # 'Tab2' for Table 2, 'S9' for Supplementary Table S9
        label = table_label(key)
        return label if label.startswith("S") else f"Tab{label}"
    sheet = {"TD_dataset": f"{lab('TD')}_dataset", "T4_comparisons": f"{lab('T4')}_{lab('comparisons')}_comparisons",
             "T3_information_systems": f"{lab('T3')}_{lab('f_systems')}_F_systems",
             "T12_literature": f"{lab('literature')}_literature", "TF_features": f"{lab('features')}_features",
             "T1_design": f"{lab('design')}_design", "TS_setup": f"{lab('setup')}_setup",
             "S_models": f"{lab('models')}_models", "S_lr_search": f"{lab('lr')}_learning_rates",
             "T2_p_systems": f"{lab('p_systems')}_P_systems", "S_grid": f"{lab('grid')}_window_grid",
             "T3_information_contrasts": f"{lab('contrasts')}_F_contrasts", "T5_matched": f"{lab('matched')}_matched",
             "S_day_type_events": f"{lab('day_types')}_day_types_events",
             "S_months_hours": f"Fig{FIGURE_NUMBERS['profiles']}_months_hours",
             "T6_aggregation": f"{lab('energy_summary')}_energy_summary",
             "S_aggregation_monthly": f"{lab('energy_months')}_energy_months", "S_leads_1_145": "NoteS4_leads_1_145",
             "TE_efficiency": f"{lab('efficiency')}_efficiency", "S_training": f"{lab('training')}_training",
             "S_runs": f"{lab('runs')}_runs", "S_attention_matched": f"{lab('attention_matched')}_attention_matched",
             "S_gates": f"{lab('gates')}_gates", "S_attention_summary": f"{lab('attention_summary')}_attention_summary",
             "S_attention_weights": f"FigS{SUPP_FIGURES.index('attention') + 1}_attention_weights"}
    assert all(len(v) <= 31 for v in sheet.values()) and len(set(sheet.values())) == len(sheet)
    shown = {name: t[name].assign(question=t[name].question.str.replace("best ", "validation-selected ", regex=False))
             if "question" in t[name].columns else t[name] for name, _, _ in notes[1:] if name in t}
    notes = [notes[0]] + [(sheet[name], what, content) for name, what, content in notes[1:] if name in t]
    path = write_workbook({sheet[name]: df for name, df in shown.items()}, notes)
    print(f"{len(t)} tables -> {TABLES} and {path.name}")


# ==============================================================================================
# Figures (matplotlib, one style: journal widths, 7 pt Arial, vector PDF and 600 dpi PNG, no plot titles)
# ==============================================================================================
FIGURES = ANALYSIS / "figures"
MM = 1 / 25.4
INK, INK2, GRID, SHADE, SHADE2 = "#0b0b0b", "#52514e", "#e1e0d9", "#f0efec", "#e1e0d9"
# the first three slots of the dataviz reference palette; together they pass the colour-vision checks for
# any pair (worst CVD dE 9.2, normal vision 24.0), so at most three hues appear in a figure
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, GRAY = "#cde2fb", "#a8a79f"                 # blue ramp step 100; context marks
NAMES = {"LightTCN_medium": "LightTCN-M", "LightTCN_large": "LightTCN-L", "ConcatTCN_medium": "ConcatTCN-M",
         "ConcatTCN_large": "ConcatTCN-L", "DualTCN_medium": "DualTCN-M", "DualTCN_large": "DualTCN-L",
         "DualTCN_Attn_medium": "DualTCN-Attn-M", "DualTCN_Attn_large": "DualTCN-Attn-L", "LSTM": "LSTM",
         "BiLSTM": "BiLSTM", "CNN_LSTM": "CNN-LSTM", "NF_NHITS": "N-HiTS", "NF_PatchTST": "PatchTST",
         "NF_TimeXer": "TimeXer", "NF_iTransformer": "iTransformer", "Chronos2": "Chronos-2", "TimesFM3": "TimesFM-3",
         "TiRex2": "TiRex-2", "Chronos2_LoRA": "Chronos-2 LoRA", "LightGBM": "LightGBM",
         "LightTCN_F_medium": "LightTCN-F", "Vanilla": "Vanilla", "Vanilla_holiday": "Vanilla + holiday",
         "Naive": "Persistence", "SeasonalNaive": "Seasonal naive", "SeasonalNaiveWeekly": "Seasonal naive, weekly"}
P_ROWS = [("TCN family", [("LightTCN_medium", "cal"), ("LightTCN_medium", "uni"), ("LightTCN_large", "cal"),
                              ("LightTCN_large", "uni"), ("ConcatTCN_medium", "cal"), ("ConcatTCN_large", "cal"),
                              ("DualTCN_medium", "cal"), ("DualTCN_large", "cal"), ("DualTCN_Attn_medium", "cal"),
                              ("DualTCN_Attn_large", "cal")]),
          ("Recurrent", [("LSTM", "cal"), ("BiLSTM", "cal"), ("CNN_LSTM", "cal")]),
          ("NeuralForecast", [("NF_NHITS", "cal"), ("NF_NHITS", "uni"), ("NF_TimeXer", "cal"), ("NF_PatchTST", "uni"),
                              ("NF_iTransformer", "uni")]),
          ("Foundation, zero-shot", [("Chronos2", "cal"), ("Chronos2", "uni"), ("TimesFM3", "cal"), ("TimesFM3", "uni"),
                                     ("TiRex2", "cal"), ("TiRex2", "uni")]),
          ("Fine-tuned / boosting", [("Chronos2_LoRA", "cal"), ("LightGBM", "cal")])]


def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2,
        "ytick.major.pad": 2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "legend.frameon": False, "lines.linewidth": 1.0, "lines.solid_capstyle": "round", "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.dpi": 600, "mathtext.default": "regular", "axes.unicode_minus": True})
    return plt


def save_figure(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=600)
    fig.clf()


def panel_letter(ax, letter: str, x: float = -0.02, y: float = 1.02) -> None:
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold")


def hgrid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.4)
    ax.set_axisbelow(True)


def row_layout(groups: list[tuple[str, list]]) -> tuple[dict, list, list, float]:
    """y positions for a dot plot with one header row per group: (positions, row labels, headers, bottom)."""
    ypos, labels, headers, y = {}, [], [], 0.0
    for header, members in groups:
        headers.append((y, header))
        y += 1
        for key, label in members:
            ypos[key] = y
            labels.append((y, label))
            y += 1
        y += 0.3
    return ypos, labels, headers, y - 0.3


def label_rows(ax, labels: list, headers: list, bottom: float) -> None:
    ax.set_yticks([t for t, _ in labels], [s for _, s in labels])
    for y, header in headers:
        ax.text(-0.035, y, header, transform=ax.get_yaxis_transform(), ha="right", va="center", fontweight="bold")
    ax.set_ylim(bottom, -0.7)


def fig_data(plt, D: rx.Dataset) -> None:
    """F1: the hourly load 2016-2025 with the splits, the mean daily profile by day type in the training
    years, and daily mean load against population-weighted temperature."""
    frame = pd.DataFrame({"y": D.y / 1000, "t": D.temp[:, 0], "day": pd.Series(D.labels).str[:10],
                          "hour": D.hour, "dow": D.local.dt.dayofweek.to_numpy()})
    kind = pd.read_csv(rx.DATA_FILE, usecols=["holiday_type"]).holiday_type
    daily = frame.groupby("day").agg(mean=("y", "mean"), lo=("y", "min"), hi=("y", "max"), t=("t", "mean"))
    dates = pd.to_datetime(daily.index)
    fig = plt.figure(figsize=(190 * MM, 112 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.42, wspace=0.26, left=0.06, right=0.99,
                          top=0.95, bottom=0.09)
    ax = fig.add_subplot(gs[0, :])
    for year, shade, label in ((2024, SHADE, "Validation 2024"), (2025, SHADE2, "Test 2025")):
        ax.axvspan(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01"), color=shade, lw=0, zorder=0)
        ax.text(pd.Timestamp(f"{year}-07-02"), 61.5, label, ha="center", va="bottom", color=INK2)
    ax.text(pd.Timestamp("2020-01-01"), 61.5, "Training 2016–2023", ha="center", va="bottom", color=INK2)
    ax.fill_between(dates, daily.lo, daily.hi, color=BLUE_LIGHT, lw=0, zorder=1, label="Daily range")
    ax.plot(dates, daily["mean"], color=BLUE, lw=0.6, zorder=2, label="Daily mean")
    ax.set_xlim(dates[0], pd.Timestamp("2026-01-01"))
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(2016, 2026)], [str(y) for y in range(2016, 2026)])
    ax.set_ylim(12, 61)
    ax.set_ylabel("Load (GW)")
    ax.legend(loc="lower left", ncol=2, handlelength=1.4, borderaxespad=0.2)
    hgrid(ax)
    panel_letter(ax, "a", x=-0.035)

    train = frame.iloc[:D.train_end]
    special = D.special_day[:D.train_end]
    groups = [("Weekdays", BLUE, (train.dow < 5) & ~special), ("Sundays", ORANGE, (train.dow == 6) & ~special),
              ("Religious holidays", AQUA, kind.iloc[:D.train_end].eq("religious").to_numpy())]
    ax = fig.add_subplot(gs[1, 0])
    for label, color, mask in groups:
        profile = train[mask].groupby("hour").y.mean()
        ax.plot(profile.index, profile.to_numpy(), color=color, lw=1.2, label=label)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("Hour of day (local time)")
    ax.set_ylabel("Mean load, 2016–2023 (GW)")
    ax.legend(loc="lower right", handlelength=1.4)
    hgrid(ax)
    panel_letter(ax, "b", x=-0.07)

    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(daily.t, daily["mean"], s=2.5, color=BLUE, alpha=0.35, lw=0, rasterized=True)
    bins = np.arange(np.floor(daily.t.min()), np.ceil(daily.t.max()) + 1, 1.0)
    binned = daily.groupby(pd.cut(daily.t, bins), observed=True)["mean"].agg(["mean", "size"])
    binned = binned[binned["size"] >= 10]
    ax.plot([i.mid for i in binned.index], binned["mean"], color=INK, lw=1.0)
    ax.set_xlabel("Daily mean temperature, population-weighted (°C)")
    ax.set_ylabel("Daily mean load (GW)")
    hgrid(ax)
    panel_letter(ax, "c", x=-0.07)
    save_figure(fig, "data")


def fig_p_errors(plt, systems: pd.DataFrame, plan: dict) -> None:
    """F3: test MAPE of every validation-selected P system per horizon (mean over seeds with the t-based 95 %
    seed interval; hollow = one seed). Blue = TCN family, orange = foundation models, grey = others; a dark ring
    marks the pre-specified primary rival(s)."""
    from matplotlib.ticker import MaxNLocator
    p = systems[systems.kind == "P"].set_index(["model", "arm", "H"])
    rows = [r for _, members in P_ROWS for r in members]
    ypos, labels, headers, bottom = row_layout([(family, [(m, f"{NAMES[m[0]]} ({m[1]})") for m in members])
                                                for family, members in P_ROWS])
    rivals = {H: {tuple(c["b"]) for c in plan["primary_comparisons"] if c["H"] == H and not c["a"][0].startswith("DualTCN")}
              for H in HORIZONS}
    fig, axes = plt.subplots(1, 4, figsize=(190 * MM, 124 * MM), sharey=True,
                             gridspec_kw={"wspace": 0.12, "left": 0.15, "right": 0.99, "top": 0.95, "bottom": 0.075})
    for ax, H, letter in zip(axes, HORIZONS, "abcd"):
        xs = []
        for m in rows:
            if (m[0], m[1], H) not in p.index:
                continue
            s = p.loc[(m[0], m[1], H)]
            cls = group_colour(m[0])
            color = BLUE if cls == "tcn" else ORANGE if cls == "foundation" else GRAY
            n = int(s.n_seeds)
            if n > 1:
                ax.errorbar(s.MAPE, ypos[m], xerr=s.MAPE_ci95, fmt="none", ecolor=color, elinewidth=0.8, capsize=0)
            ax.plot(s.MAPE, ypos[m], "o", ms=3.6, mfc=color if n > 1 else "white", mec=color, mew=0.9)
            if m in rivals[H]:
                ax.plot(s.MAPE, ypos[m], "o", ms=6.6, mfc="none", mec=INK, mew=0.7)
            xs += [s.MAPE - (s.MAPE_ci95 if n > 1 else 0), s.MAPE + (s.MAPE_ci95 if n > 1 else 0)]
        lo, hi = min(xs), max(xs)
        ax.set_xlim(lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.set_xlabel(f"MAPE (%), H = {H} h")
        hgrid(ax, "x")
        ax.tick_params(axis="y", length=0)
        panel_letter(ax, letter, x=0.0)
    label_rows(axes[0], labels, headers, bottom)
    save_figure(fig, "p_errors")


def fig_information(plt, inf: pd.DataFrame) -> None:
    """F4: relative effect of the main information contrasts (second arm minus first, % of the first arm's
    MAPE) with the 95 % block-bootstrap interval; filled = significant after Holm within (model, H).
    Orange = calendar, aqua = holiday analog (exploratory), blue = weather."""
    info = inf[inf.family.str.startswith("information") & (inf.loss == "APE")].copy()
    info["first"], info["second"] = info.B.str.split("|").str[1], info.A.str.split("|").str[1]
    info["model"] = info.A.str.split("|").str[0]
    panels = [("F0", "F1", "Future calendar\n(F0 → F1)", ORANGE), ("F1", "F1A", "Holiday analogue, exploratory\n(F1 → F1A)", AQUA),
              ("F1", "F2", "Past temperature\n(F1 → F2)", BLUE), ("F2", "F3", "Forecast temperature\n(F2 → F3)", BLUE),
              ("F3", "F4", "Oracle weather scenario\n(F3 → F4)", BLUE)]
    models = ["LightTCN_F_medium", "LightGBM", "NF_NHITS", "Chronos2", "TimesFM3", "TiRex2"]
    ypos, labels, headers, bottom = row_layout(
        [(NAMES[m], [((m, H), f"H = {H} h") for H in HORIZONS if ((info.model == m) & (info.H == H)).any()])
         for m in models])
    fig, axes = plt.subplots(1, len(panels), figsize=(190 * MM, 118 * MM), sharey=True,
                             gridspec_kw={"wspace": 0.1, "left": 0.1, "right": 0.99, "top": 0.9, "bottom": 0.075})
    for ax, (first, second, title, color), letter in zip(axes, panels, "abcde"):
        sel = info[(info["first"] == first) & (info["second"] == second)]
        for r in sel.itertuples():
            yy = ypos[(r.model, r.H)]
            lo, hi = 100 * r.ci_low / r.mean_B, 100 * r.ci_high / r.mean_B
            ax.plot([lo, hi], [yy, yy], color=color, lw=0.9)
            ax.plot(r.rel_diff_pct, yy, "o", ms=3.6, mfc=color if r.significant else "white", mec=color, mew=0.9)
        ax.axvline(0, color=INK2, lw=0.5, zorder=0)
        ax.set_xlabel("Change in MAPE (%)")
        ax.text(0.5, 1.015, title, transform=ax.transAxes, ha="center", va="bottom", color=INK)
        hgrid(ax, "x")
        ax.tick_params(axis="y", length=0)
        panel_letter(ax, letter, x=0.02, y=1.015)
    label_rows(axes[0], labels, headers, bottom)
    save_figure(fig, "information")


def system_leads(systems: pd.DataFrame, lead: pd.DataFrame, system: str) -> pd.DataFrame:
    """Per-lead MAPE and signed error of a system: the mean over its runs (seeds) of the runs' lead curves."""
    s = systems.set_index("system").loc[system]
    parts = pd.DataFrame([p.split("|") for p in s.runs.split(";")], columns=["run_id", "variant"])
    return lead.merge(parts, on=["run_id", "variant"]).groupby("lead")[["MAPE", "bias_pct"]].mean()


def fig_leads(plt, systems: pd.DataFrame) -> None:
    """Error by lead: (a) MAPE at H = 168 h with LightTCN-F F3, whose last 11 leads use climatology; (b) MAPE and
    (c) bias at H = 720 h. Colours follow the system in every panel."""
    lead = pd.read_parquet(ANALYSIS / "lead_curves.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(190 * MM, 64 * MM),
                             gridspec_kw={"wspace": 0.3, "left": 0.06, "right": 0.99, "top": 0.84, "bottom": 0.19})
    a168 = [("LightTCN_large|cal|H168", "LightTCN-L (cal)", BLUE), ("Chronos2_LoRA|cal|H168", "Chronos-2 LoRA (cal)", ORANGE),
            ("LightTCN_F_medium|F3|H168", "LightTCN-F (F3)", AQUA)]
    a720 = [("LightTCN_large|cal|H720", "LightTCN-L (cal)", BLUE), ("Chronos2_LoRA|cal|H720", "Chronos-2 LoRA (cal)", ORANGE),
            ("NF_PatchTST|uni|H720", "PatchTST (uni)", INK2)]
    handles = {}
    for ax, series, col, letter, ylabel in ((axes[0], a168, "MAPE", "a", "MAPE (%), H = 168 h"),
                                            (axes[1], a720, "MAPE", "b", "MAPE (%), H = 720 h"),
                                            (axes[2], a720, "bias_pct", "c", "Bias (%), H = 720 h")):
        for system, label, color in series:
            g = system_leads(systems, lead, system)
            handles[label], = ax.plot(g.index, g[col], color=color, lw=1.0, label=label)
        if col == "bias_pct":
            ax.axhline(0, color=INK2, lw=0.5, zorder=0)
        if letter == "a":
            ax.axvline(157.5, color=INK2, lw=0.5, ls=(0, (2, 2)), zorder=0)
            ax.text(155, ax.get_ylim()[0] + 0.04 * np.diff(ax.get_ylim())[0], "climatology\nfor F3 →", ha="right",
                    va="bottom", color=INK2, fontsize=6.5)
        ax.set_xlim(1, 168 if letter == "a" else 720)
        ax.set_xticks([1, 24, 48, 72, 96, 120, 144, 168] if letter == "a" else [1, 168, 336, 504, 720])
        ax.set_xlabel("Lead (h)")
        ax.set_ylabel(ylabel)
        hgrid(ax)
        panel_letter(ax, letter, x=-0.1)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=4, bbox_to_anchor=(0.52, 1.0),
               handlelength=1.8, columnspacing=1.6)
    save_figure(fig, "leads")


def fig_profiles(plt, systems: pd.DataFrame) -> None:
    """Errors by target-hour type at H = 24 h and 168 h and by month at H = 168 h, for selected past-only systems
    and LightTCN-F arms (filled = past-only; LightTCN-F arms by marker shape)."""
    b = pd.read_csv(ANALYSIS / "breakdowns_systems.csv")
    fig = plt.figure(figsize=(190 * MM, 72 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.3, left=0.135, right=0.99, top=0.84, bottom=0.2)
    series = [("LightTCN_large|cal", "LightTCN-L (cal)", BLUE, "o", True), ("Chronos2_LoRA|cal", "Chronos-2 LoRA", ORANGE, "o", True),
              ("NF_NHITS|cal", "N-HiTS (cal)", GRAY, "o", True), ("LightTCN_F_medium|F0", "LightTCN-F (F0)", AQUA, "o", False),
              ("LightTCN_F_medium|F1", "LightTCN-F (F1)", AQUA, "s", False), ("LightTCN_F_medium|F1A", "LightTCN-F (F1A)", AQUA, "D", True)]
    for k, (H, letter) in enumerate(((24, "a"), (168, "b"))):
        ax = fig.add_subplot(gs[0, k])
        d = b[(b.family == "day type") & (b.H == H)]
        for j, (key, label, color, marker, filled) in enumerate(series):
            s = d[d.system == f"{key}|H{H}"].set_index("category").reindex(DAY_TYPES)
            ys = np.arange(len(DAY_TYPES)) + (j - 2.5) * 0.12
            ax.plot(s.MAPE, ys, marker, ms=3.4, mfc=color if filled else "white", mec=color, mew=0.9, ls="none",
                    label=label)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
        ax.set_yticks(range(len(DAY_TYPES)), [t.capitalize() for t in DAY_TYPES] if k == 0 else [""] * len(DAY_TYPES))
        ax.set_ylim(len(DAY_TYPES) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(f"MAPE (%), H = {H} h, log scale")
        hgrid(ax, "x")
        panel_letter(ax, letter, x=-0.04 if k else -0.5)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.52, 1.0), handletextpad=0.2,
               columnspacing=1.0)
    ax = fig.add_subplot(gs[0, 2])
    d = b[(b.family == "month") & (b.H == 168)]
    for key, label, color, ls in (("LightTCN_large|cal", "LightTCN-L (cal)", BLUE, "-"),
                                  ("Chronos2_LoRA|cal", "Chronos-2 LoRA", ORANGE, "-"),
                                  ("LightTCN_F_medium|F2", "LightTCN-F (F2)", AQUA, (0, (3, 1.5))),
                                  ("LightTCN_F_medium|F3", "LightTCN-F (F3)", AQUA, "-")):
        s = d[d.system == f"{key}|H168"].sort_values("category")
        ax.plot(s.category.astype(int), s.MAPE, color=color, lw=1.0, ls=ls, marker="o", ms=2.2, label=label)
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_xlabel("Month of the target hour, 2025")
    ax.set_ylabel("MAPE (%), H = 168 h")
    ax.legend(loc="upper right", handlelength=1.8, fontsize=6.5)
    hgrid(ax)
    panel_letter(ax, "c", x=-0.12)
    save_figure(fig, "profiles")


def fig_energy(plt) -> None:
    """Monthly energy totals of weekly updated H = 168 h forecasts: (a) monthly energy APE by month for selected
    systems; (b) mean monthly energy APE against the hourly MAPE of the same forecasts (learned systems)."""
    monthly = pd.read_csv(ANALYSIS / "aggregation_monthly.csv")
    summary = pd.read_csv(ANALYSIS / "aggregation_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(190 * MM, 74 * MM), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.25,
                                                                               "left": 0.06, "right": 0.98, "top": 0.8, "bottom": 0.15})
    ax = axes[0]
    for system, label, color, ls in (("NF_TimeXer|cal|H168", "TimeXer", INK2, "-"), ("Chronos2_LoRA|cal|H168", "Chronos-2 LoRA", ORANGE, "-"),
                                     ("LightTCN_large|cal|H168", "LightTCN-L (cal)", BLUE, "-"),
                                     ("SeasonalNaiveWeekly|uni|H168", "Seasonal naive, weekly", GRAY, (0, (3, 1.5)))):
        s = monthly[monthly.system == system].sort_values("month")
        ax.plot(s.month.astype(int), s.energy_APE, color=color, lw=1.0, ls=ls, marker="o", ms=2.4, label=label)
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Monthly energy APE (%)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, handlelength=1.8, borderaxespad=0)
    hgrid(ax)
    panel_letter(ax, "a", x=-0.07, y=1.17)
    ax = axes[1]
    learned = summary[~summary.model.isin(rx.BASELINE_MODELS)]
    shown = {}
    for r in learned.itertuples():
        cls = group_colour(r.model)
        color = BLUE if cls == "tcn" else ORANGE if cls == "foundation" else GRAY
        shown[cls], = ax.plot(r.hourly_MAPE, r.mean_monthly_energy_APE, "o", ms=3.6, mfc=color, mec="white", mew=0.5)
    for system, label, dx, dy, ha in (("NF_TimeXer|cal|H168", "TimeXer", 4, -4, "left"),
                                      ("NF_NHITS|cal|H168", "N-HiTS (cal)", 4, 5, "left"),
                                      ("Chronos2_LoRA|cal|H168", "Chronos-2 LoRA", -4, 0, "right"),
                                      ("LightTCN_large|cal|H168", "LightTCN-L (cal)", -4, 0, "right")):
        r = summary[summary.system == system].iloc[0]
        ax.annotate(label, (r.hourly_MAPE, r.mean_monthly_energy_APE), xytext=(dx, dy), textcoords="offset points",
                    va="center", ha=ha, fontsize=6.5, color=INK)
    ax.legend([shown[k] for k in ("tcn", "foundation", "other")], ["TCN family", "Foundation models", "Other models"],
              loc="lower right", handletextpad=0.2)
    ax.set_xlim(left=3.0)
    ax.set_ylim(bottom=0.75)
    ax.set_xlabel("Hourly MAPE of the same forecasts (%)")
    ax.set_ylabel("Mean monthly energy APE (%)")
    hgrid(ax)
    hgrid(ax, "x")
    panel_letter(ax, "b", x=-0.1, y=1.17)
    save_figure(fig, "energy")


def group_colour(model: str) -> str:
    """Colour class of a model in the scatter figures: TCN family, foundation models (zero-shot and LoRA), other."""
    if model.startswith(("LightTCN", "ConcatTCN", "DualTCN")):
        return "tcn"
    return "foundation" if model in (*rx.TSFM_MODELS, "Chronos2_LoRA") else "other"


def fig_efficiency(plt, systems: pd.DataFrame) -> None:
    """Test MAPE against the end-to-end latency of one forecast (batch 1, p50, log scale) per horizon; one dot per
    validation-selected past-only system profiled on the laptop GPU (LightGBM on the CPU, hollow)."""
    from matplotlib.ticker import FuncFormatter
    eff = pd.read_csv(EFFICIENCY)
    eff = eff[eff.e2e_b1_p50_ms.notna()]
    p = systems[systems.kind == "P"].set_index("system")
    # labelled systems per horizon (most of them on the latency-accuracy front) with offsets in points
    labels = {1: [("LightGBM|cal", "LightGBM", -2, 7, "left"), ("BiLSTM|cal", "BiLSTM", 3, -7, "left"),
                  ("LightTCN_large|cal", "LightTCN-L", 0, -8, "center"), ("NF_NHITS|uni", "N-HiTS (uni)", 4, -5, "left")],
              24: [("LSTM|cal", "LSTM", 3, 5, "left"), ("BiLSTM|cal", "BiLSTM", -4, 0, "right"),
                   ("LightTCN_large|cal", "LightTCN-L", -4, 0, "right"), ("Chronos2_LoRA|cal", "Chronos-2 LoRA", 4, 0, "left")],
              168: [("LSTM|cal", "LSTM", 4, 2, "left"), ("BiLSTM|cal", "BiLSTM", 4, -7, "left"),
                    ("LightTCN_medium|cal", "LightTCN-M", -4, 0, "right"), ("NF_NHITS|cal", "N-HiTS", -4, 0, "right"),
                    ("Chronos2_LoRA|cal", "Chronos-2 LoRA", -4, 0, "right")],
              720: [("CNN_LSTM|cal", "CNN-LSTM", 4, -6, "left"), ("LightTCN_medium|uni", "LightTCN-M (uni)", 3, -7, "left"),
                    ("LightTCN_large|cal", "LightTCN-L", 4, 3, "left"), ("NF_PatchTST|uni", "PatchTST", -4, 0, "right")]}
    fig, axes = plt.subplots(2, 2, figsize=(190 * MM, 118 * MM),
                             gridspec_kw={"wspace": 0.16, "hspace": 0.42, "left": 0.07, "right": 0.99, "top": 0.95,
                                          "bottom": 0.09})
    for ax, H, letter in zip(axes.ravel(), HORIZONS, "abcd"):
        e = eff[(eff.H == H) & eff.system.isin(p.index)].copy()
        e["MAPE"] = p.loc[e.system, "MAPE"].to_numpy()
        for r in e.itertuples():
            cls = group_colour(r.model)
            color = BLUE if cls == "tcn" else ORANGE if cls == "foundation" else GRAY
            cpu = r.model == "LightGBM"
            ax.plot(r.e2e_b1_p50_ms, r.MAPE, "o", ms=3.4, mfc="white" if cpu else color, mec=color, mew=0.8 if cpu else 0.4)
        front = e[e.model != "LightGBM"].sort_values("e2e_b1_p50_ms")          # GPU systems only
        front = front[front.MAPE <= front.MAPE.cummin()]   # no other system is both faster and more accurate
        ax.plot(front.e2e_b1_p50_ms, front.MAPE, color=INK2, lw=0.5, ls=(0, (2, 2)), zorder=0, drawstyle="steps-post")
        at = e.set_index("system")
        for key, label, dx, dy, ha in labels[H]:
            r = at.loc[f"{key}|H{H}"]
            ax.annotate(label, (r.e2e_b1_p50_ms, r.MAPE), xytext=(dx, dy), textcoords="offset points", fontsize=6.5,
                        color=INK, ha=ha, va="center")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        lo, hi = e.MAPE.min(), e.MAPE.max()
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.05 * (hi - lo))   # room for the labels below the lowest points
        ax.set_xlabel("Latency per forecast (ms)")
        if letter in "ac":
            ax.set_ylabel("Test MAPE (%)")
        ax.set_title(f"H = {H} h", fontsize=7, pad=3)
        hgrid(ax)
        panel_letter(ax, letter, x=-0.06)
    save_figure(fig, "efficiency")


def fig_attention(plt) -> None:
    """Supplementary figure: attention weights of DualTCN-Attn-M (H = 24 h, seed 42) over its input window at the
    four origins fixed by calendar (implementation note §11); weight × w, so 1 is a uniform weight. Shading marks
    the type of each input hour. Descriptive only."""
    a = pd.read_csv(ANALYSIS / "diagnostics_attention.csv")
    shade = {"religious holiday": AQUA, "eve": AQUA, "national holiday": ORANGE, "weekend": GRAY}
    alpha = {"religious holiday": 0.35, "eve": 0.45, "national holiday": 0.25, "weekend": 0.22}
    origins = list(ATTENTION_ORIGINS)
    fig, axes = plt.subplots(len(origins), 1, figsize=(190 * MM, 128 * MM), sharex=True,
                             gridspec_kw={"hspace": 0.55, "left": 0.08, "right": 0.99, "top": 0.9, "bottom": 0.085})
    w = int(a.lag_h.max())
    for ax, origin, letter in zip(axes, origins, "abcd"):
        d = a[a.origin == origin].sort_values("lag_h", ascending=False)
        x = -d.lag_h.to_numpy()
        for kind in shade:
            m = (d.input_hour_type == kind).to_numpy()
            for xi in x[m]:
                ax.axvspan(xi - 0.5, xi + 0.5, color=shade[kind], alpha=alpha[kind], lw=0, zorder=0)
        ax.plot(x, d.weight_x_w, color=BLUE, lw=0.8, zorder=2)
        ax.axhline(1, color=INK2, lw=0.5, ls=(0, (2, 2)), zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(0.005, 100)
        ax.set_yticks([0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
        ax.set_title(f"({letter}) Origin {origin}, {english_holidays(ATTENTION_ORIGINS[origin])}", loc="left",
                     fontsize=7, color=INK, pad=2.5)
        hgrid(ax)
    axes[-1].set_xlim(-w - 0.5, -0.5)
    axes[-1].set_xticks(np.arange(-w, 1, 24)[::2])
    axes[-1].set_xlabel("Input hour relative to the origin (h)")
    fig.text(0.015, 0.5, "Attention weight × window length (log scale)", rotation=90, va="center", ha="center")
    from matplotlib.patches import Patch
    present = set(a.input_hour_type)
    handles = [Patch(color=shade[k], alpha=alpha[k], label=k[0].upper() + k[1:]) for k in shade if k in present]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.0))
    assert a.weight_x_w.min() >= 0.005 and a.weight_x_w.max() <= 100      # no value outside the axis
    save_figure(fig, "attention")


def fig_graphical_abstract(plt, D: rx.Dataset, systems: pd.DataFrame) -> None:
    """Graphical abstract (Elsevier: at least 1328 × 531 px, readable at 13 × 5 cm), drawn at 133 × 53 mm: the data
    and the two information sets, the models and horizons, and the main results. Every number is read from the
    analysis files."""
    from matplotlib.patches import FancyBboxPatch
    W, HT = 133.0, 53.0
    p = systems[systems.kind == "P"]
    lead = p.loc[p.groupby("H").MAPE.idxmin()].set_index("H")
    f = systems[(systems.model == "LightTCN_F_medium") & (systems.H == 168)].set_index("arm").MAPE
    temp = 100 * (f["F3"] / f["F2"] - 1)
    m = pd.read_csv(TABLES / "T5_matched.csv")
    delay = m[(m.kind == "publication delay") & (m.model == "LightTCN_medium") & (m.H == 1)].iloc[0]
    eff = pd.read_csv(EFFICIENCY)
    eff = eff[eff.e2e_b1_p50_ms.notna() & ~eff.system.str.endswith(("|w24", "|w48"))]
    tcn = eff[eff.model.str.contains("TCN")].e2e_b1_p50_ms
    fm = eff[eff.model.isin(["Chronos2", "Chronos2_LoRA", "TimesFM3", "TiRex2"])].e2e_b1_p50_ms

    fig = plt.figure(figsize=(W * MM, HT * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, HT)
    ax.axis("off")
    TITLE, BODY, SMALL, BIG = 9.0, 7.5, 7.2, 9.0       # pt; Elsevier: readable at 13 x 5 cm, so nothing below 7 pt

    def text(x, y, s, size=BODY, color=INK, **kw):
        ax.text(x, y, s, fontsize=size, color=color, va="center", **kw)

    def chip(x, y, w, h, s, face, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9", fc=face, ec=edge, lw=0.6))
        text(x + w / 2, y + h / 2, s, 7.0, ha="center")

    def arrow(x0, x1, y):
        ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18",
                                                         color=GRAY, lw=0.8))

    # (1) data and information sets (x 2-37 mm)
    text(2, HT - 3.3, "Data and information", TITLE, fontweight="bold")
    text(2, HT - 7.8, "Hourly load, 2016–2025", color=INK2)
    day = pd.Series(D.y / 1000).groupby(pd.Series(D.labels).str[:10].to_numpy()).mean()
    week = day.groupby(np.arange(len(day)) // 7).mean().to_numpy()
    at = {y: int(np.searchsorted(day.index.to_numpy(), f"{y}-01-01")) / 7 for y in (2024, 2025)}
    sp = fig.add_axes([2 / W, 33.4 / HT, 35 / W, 8.8 / HT])
    sp.axvspan(at[2024], at[2025], color=SHADE, lw=0)
    sp.axvspan(at[2025], len(week), color=SHADE2, lw=0)
    sp.plot(np.arange(len(week)), week, color=BLUE, lw=0.7)
    sp.set_xlim(0, len(week))
    sp.axis("off")
    text(37, 31.6, "validation · test", SMALL, INK2, ha="right")
    for y0, h, title, lines, edge in ((19.8, 9.8, "Past only", ["load of the input hours,", "with or without calendar"], GRAY),
                                      (0.8, 18.0, "Known future", ["target-hour calendar,", "holiday analogue",
                                                                   "(exploratory),", "temperature forecasts",
                                                                   "(pseudo-real-time)"], AQUA)):
        ax.add_patch(FancyBboxPatch((2, y0), 35, h, boxstyle="round,pad=0,rounding_size=0.9", fc="white", ec=edge, lw=0.8))
        text(3.4, y0 + h - 2.5, title, BODY, fontweight="bold")
        for k, line in enumerate(lines):
            text(3.4, y0 + h - 5.3 - 2.65 * k, line, SMALL, INK2)
    arrow(37.6, 40.6, 27.0)

    # (2) models and horizons (x 41.2-80.2 mm)
    text(41.2, HT - 3.3, "26 models, 4 horizons", TITLE, fontweight="bold")
    chips = [("TCN variants", BLUE_LIGHT, BLUE), ("Foundation", "#fbe1d6", ORANGE), ("Recurrent", SHADE, GRAY),
             ("NeuralForecast", SHADE, GRAY), ("LightGBM", SHADE, GRAY), ("Linear, naive", SHADE, GRAY)]
    for k, (label, face, edge) in enumerate(chips):
        chip(41.2 + 19.9 * (k % 2), 38.6 - 5.4 * (k // 2), 19.1, 4.4, label, face, edge)
    ax.plot([42.7, 78.7], [19.6, 19.6], color=GRAY, lw=0.8, solid_capstyle="butt")
    for k, (h, unit) in enumerate(zip(HORIZONS, ["hour", "day", "week", "month"])):
        x = 45.2 + 10.4 * k
        ax.plot(x, 19.6, "o", ms=3.2, mfc="white", mec=INK2, mew=0.7)
        text(x, 22.7, f"{h} h", BODY, ha="center")
        text(x, 16.5, unit, SMALL, INK2, ha="center")
    for k, line in enumerate(["all eligible 2025 origins", "up to five training seeds", "Holm-adjusted bootstrap"]):
        text(41.2, 10.9 - 2.8 * k, line, SMALL, INK2)
    arrow(80.8, 83.6, 27.0)

    # (3) results (x 84.5-131.5 mm); the delay result sits directly under the 1 h row
    text(84.5, HT - 3.3, "Results, 2025 test year", TITLE, fontweight="bold")
    text(84.5, HT - 7.8, "Most accurate past-only model", color=INK2)
    for k, (h, r) in enumerate(lead.iterrows()):
        y = 41.0 - 3.0 * k
        cls = group_colour(r.model)
        text(91.5, y, f"{h} h", ha="right")
        ax.plot(93.2, y, "o", ms=3.0, color=ORANGE if cls == "foundation" else BLUE if cls == "tcn" else GRAY,
                mec="white", mew=0.4)
        text(94.8, y, NAMES[r.model])
        text(131.0, y, f"{r.MAPE:.2f}%", ha="right")
    ax.plot([84.5, 131.5], [29.6, 29.6], color=GRID, lw=0.6)
    stats = [(f"{delay.MAPE_reference:.2f}→{delay.MAPE_arm:.2f}%", ["1 h TCN error with a", "2 h load-data delay"]),
             (f"{temp:.1f}%".replace("-", "−"), ["week-ahead TCN error", "with temperature", "forecasts"]),
             (f"{tcn.min():.1f}–{tcn.max():.1f} ms", ["TCN time per forecast", "on a laptop GPU;",
                                                    f"foundation {fm.min():.0f}–{fm.max():.0f} ms"])]
    for (big, lines), y in zip(stats, (25.6, 17.4, 7.8)):
        text(84.5, y, big, BIG, fontweight="bold")
        for n, line in enumerate(lines):
            text(105.3, y + 1.3 * (len(lines) - 1) - 2.6 * n, line, SMALL, INK2)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"graphical_abstract.{ext}", dpi=600)
    fig.clf()
    from PIL import Image                              # TIFF for the submission: RGB, LZW, 600 dpi
    Image.open(FIGURES / "graphical_abstract.png").convert("RGB").save(
        FIGURES / "graphical_abstract.tiff", compression="tiff_lzw", dpi=(600, 600))


def fig_architecture(plt) -> None:
    """Fig. 2: block diagrams of the four TCN architectures and the known-future decoder, drawn from models.py,
    in two rows of three panels; panel (f) is the legend. Boxes: blue = TCN stacks on the load (and calendar),
    orange = calendar path, aqua = known-future decoder."""
    from matplotlib.patches import FancyBboxPatch

    fig, grid = plt.subplots(2, 3, figsize=(190 * MM, 158 * MM),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.03, "left": 0.005, "right": 0.995, "top": 0.99,
                                          "bottom": 0.005})
    axes = list(grid.ravel())

    def box(ax, x, y, w, h, text, fc="white", ec=INK2, size=7.4):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.25",
                                    fc=fc, ec=ec, lw=0.7))
        ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.25)

    def arrow(ax, x1, y1, x2, y2, dashed=False):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK2, shrinkA=0,
                                                            shrinkB=0, mutation_scale=6, ls=(0, (2, 1.5)) if dashed else "-"))

    tcn, exo, fut, inp = dict(fc=BLUE_LIGHT, ec=BLUE), dict(fc="#fde3d7", ec=ORANGE), dict(fc="#d3f1e5", ec=AQUA), dict(fc=SHADE)
    for ax, title in zip(axes, ["(a) LightTCN", "(b) ConcatTCN", "(c) DualTCN", "(d) DualTCN-Attn", "(e) LightTCN-F"]):
        ax.set_xlim(0, 10)
        ax.set_ylim(2.6, 20.4)
        ax.axis("off")
        ax.text(5, 20.2, title, ha="center", va="top", fontsize=8.6, fontweight="bold", color=INK)
    out = r"$\hat{y}_1,\ \ldots,\ \hat{y}_H$"
    Y_IN, Y_STACK, Y_STATE, Y_FUSE, Y_READ, Y_OUT = 17.6, 14.4, 11.1, 8.3, 5.9, 3.7

    ax = axes[0]                                                        # LightTCN
    box(ax, 5, Y_IN, 9.0, 1.7, "Input window ($w$ h):\nload + 6 calendar channels", **inp)
    box(ax, 5, Y_STACK, 9.0, 3.3, "Causal TCN stack\n" r"$n=\max(3,\lceil\log_2 w\rceil)$ blocks" "\n"
        r"dilations $1, 2, \ldots, 2^{n-1}$" "\n$C$ channels", **tcn)
    box(ax, 5, Y_STATE, 7.4, 1.4, "State $u$ of the\nlast input hour")
    box(ax, 5, Y_READ, 7.4, 1.2, r"Linear ($C \rightarrow H$)")
    box(ax, 5, Y_OUT, 7.4, 1.2, out)
    for y1, y2 in ((Y_IN - 0.85, Y_STACK + 1.65), (Y_STACK - 1.65, Y_STATE + 0.7), (Y_STATE - 0.7, Y_READ + 0.6),
                   (Y_READ - 0.6, Y_OUT + 0.6)):
        arrow(ax, 5, y1, 5, y2)

    ax = axes[1]                                                        # ConcatTCN
    box(ax, 5, Y_IN, 9.0, 1.7, "Input window ($w$ h):\nload + 6 calendar channels", **inp)
    box(ax, 3.1, Y_STACK, 5.4, 3.3, "Causal TCN stack\n(as in a)", **tcn)
    box(ax, 8.15, Y_STACK, 3.5, 3.3, "Calendar of\nthe last\ninput hour\n→ MLP", **exo)
    box(ax, 3.1, Y_STATE, 4.6, 1.4, "State $u$")
    box(ax, 5, Y_FUSE, 9.0, 1.4, "Concatenate → LayerNorm")
    box(ax, 5, Y_READ, 7.4, 1.2, r"Linear ($1.5C \rightarrow H$)")
    box(ax, 5, Y_OUT, 7.4, 1.2, out)
    arrow(ax, 3.1, Y_IN - 0.85, 3.1, Y_STACK + 1.65)
    arrow(ax, 8.15, Y_IN - 0.85, 8.15, Y_STACK + 1.65)
    arrow(ax, 3.1, Y_STACK - 1.65, 3.1, Y_STATE + 0.7)
    arrow(ax, 3.1, Y_STATE - 0.7, 3.1, Y_FUSE + 0.7)
    arrow(ax, 8.15, Y_STACK - 1.65, 8.15, Y_FUSE + 0.7)
    arrow(ax, 5, Y_FUSE - 0.7, 5, Y_READ + 0.6)
    arrow(ax, 5, Y_READ - 0.6, 5, Y_OUT + 0.6)

    for ax, attn in ((axes[2], False), (axes[3], True)):                # DualTCN and DualTCN-Attn
        xl, xr = (1.95, 7.4) if attn else (2.6, 7.4)
        box(ax, 2.6, Y_IN, 4.5, 1.7, "Load\n(1 channel)", **inp)
        box(ax, 7.4, Y_IN, 4.5, 1.7, "Calendar\n(6 channels)", **inp)
        box(ax, 2.6, Y_STACK, 4.5, 3.3, "Load\nTCN stack", **tcn)
        box(ax, 7.4, Y_STACK, 4.5, 3.3, "Calendar\nTCN stack", **exo)
        box(ax, xl, Y_STATE, 3.4 if attn else 4.5, 1.4, "$T$: last state", size=7.0 if attn else 7.4)
        if attn:                                         # the dashed arrow: T queries all calendar states
            box(ax, 7.0, Y_STATE + 0.05, 5.8, 2.6, r"$E'=\Sigma_{\tau}\,\alpha_{\tau}e_{\tau}$" "\n"
                r"$\alpha=\mathrm{softmax}(T^{\top}e/\sqrt{C})$" "\nno learnable projections", size=6.8)
            arrow(ax, xl + 1.7, Y_STATE, 4.1, Y_STATE, dashed=True)
        else:
            box(ax, xr, Y_STATE, 4.5, 1.4, "$E$: last state")
        e = "E'" if attn else "E"
        box(ax, 5, Y_FUSE, 9.6, 1.9, rf"$g=\sigma(W_g[T;{e}]+b_g)$" "\n" rf"$z=g\odot T+(1-g)\odot {e}$")
        box(ax, 5, Y_READ, 9.6, 1.2, "Dropout → LayerNorm → Linear")
        box(ax, 5, Y_OUT, 7.4, 1.2, out)
        arrow(ax, 2.6, Y_IN - 0.85, 2.6, Y_STACK + 1.65)
        arrow(ax, 7.4, Y_IN - 0.85, 7.4, Y_STACK + 1.65)
        arrow(ax, 2.6, Y_STACK - 1.65, xl, Y_STATE + 0.7)
        arrow(ax, 7.4, Y_STACK - 1.65, 7.4, Y_STATE + (1.35 if attn else 0.7))
        arrow(ax, xl, Y_STATE - 0.7, xl, Y_FUSE + 0.95)
        arrow(ax, 7.4, Y_STATE - (1.25 if attn else 0.7), 7.4, Y_FUSE + 0.95)
        arrow(ax, 5, Y_FUSE - 0.95, 5, Y_READ + 0.6)
        arrow(ax, 5, Y_READ - 0.6, 5, Y_OUT + 0.6)

    ax = axes[4]                                                        # LightTCN-F
    box(ax, 5, Y_IN, 9.4, 1.7, "Input window ($w$ h): load,\ncalendar (+ past temperature)", **inp)
    box(ax, 5, Y_STACK, 9.0, 2.2, "Causal TCN stack\n(as in a) → $u$", **tcn)
    box(ax, 2.3, Y_STATE - 0.2, 4.0, 2.0, "Linear\n" r"$a_k^{\top}u+b_k$")
    box(ax, 7.3, Y_STATE - 0.2, 5.0, 2.6, "Decoder, lead $k$:\n" r"MLP$([u;\,x_k;\,s_k])$")
    ax.patches[-1].set(fc=fut["fc"], ec=fut["ec"])
    box(ax, 5, Y_FUSE - 0.6, 9.6, 1.4, r"$\hat{y}_k=a_k^{\top}u+b_k+f([u;\,x_k;\,s_k])$")
    box(ax, 5, Y_OUT + 0.6, 8.2, 1.6, "F0 control: $x_k$ = 0\n(same network)")
    arrow(ax, 5, Y_IN - 0.85, 5, Y_STACK + 1.1)
    arrow(ax, 3.4, Y_STACK - 1.1, 2.3, Y_STATE + 0.8)
    arrow(ax, 6.6, Y_STACK - 1.1, 7.3, Y_STATE + 1.1)
    arrow(ax, 2.3, Y_STATE - 1.2, 3.2, Y_FUSE + 0.1)
    arrow(ax, 7.3, Y_STATE - 1.5, 6.6, Y_FUSE + 0.1)

    ax = axes[5]                                                        # legend
    ax.set_xlim(0, 10)
    ax.set_ylim(2.6, 20.4)
    ax.axis("off")
    ax.text(5, 20.2, "(f) Legend", ha="center", va="top", fontsize=8.6, fontweight="bold", color=INK)
    for y, kw, label in ((17.8, inp, "inputs of the window"), (16.2, tcn, "TCN stack"), (14.6, exo, "calendar path"),
                         (13.0, fut, "decoder for inputs known\nfor the target hours"), (11.4, {}, "state, fusion or read-out")):
        box(ax, 1.4, y, 1.8, 1.0, "", **kw)
        ax.text(2.7, y, label, ha="left", va="center", fontsize=7.4, color=INK, linespacing=1.15)
    ax.text(0.4, 9.9, "Each TCN stack: residual blocks of a causal\n"
                      "dilated convolution (kernel 3, left padding),\n"
                      "layer normalisation, GELU and dropout, with a\n"
                      "1 × 1 convolution on the residual path where\n"
                      "the channels change. $C$ = 64 (medium) or\n"
                      "128 (large) channels; $w$: input window;\n"
                      "$H$: horizon; $u$: last encoder state;\n"
                      "$x_k$: inputs known for target hour $k$;\n"
                      "$s_k$: learned step embedding;\n"
                      "dashed arrow: query of the attention.",
            ha="left", va="top", fontsize=7.0, color=INK, linespacing=1.35)
    save_figure(fig, "architecture")


def build_figures(names: list[str]) -> None:
    """Figures of the paper from the analysis files; `names` limits the output (e.g. data leads)."""
    plt = plot_style()
    systems = pd.read_csv(ANALYSIS / "systems.csv")
    plan = json.loads(PLAN_FILE.read_text())
    inf = pd.read_csv(ANALYSIS / "inference.csv")
    todo = {"data": lambda: fig_data(plt, rx.Dataset()), "p_errors": lambda: fig_p_errors(plt, systems, plan),
            "leads": lambda: fig_leads(plt, systems), "information": lambda: fig_information(plt, inf),
            "profiles": lambda: fig_profiles(plt, systems), "energy": lambda: fig_energy(plt),
            "efficiency": lambda: fig_efficiency(plt, systems), "architecture": lambda: fig_architecture(plt),
            "attention": lambda: fig_attention(plt),
            "graphical_abstract": lambda: fig_graphical_abstract(plt, rx.Dataset(), systems)}
    for name in names or list(todo):
        if name == "efficiency" and not EFFICIENCY.exists():
            print("efficiency: no profile yet (run `profile`)")
            continue
        todo[name]()
        print(f"{name} -> {FIGURES}")


# ==============================================================================================
# Efficiency profile (implementation note §13): one fresh process per reported system on the idle laptop GPU
# ==============================================================================================
EFFICIENCY = ANALYSIS / "efficiency.csv"


def profile_targets() -> pd.DataFrame:
    """The seed-42 run of every reported P system (naive references excluded), of LightTCN-F and Vanilla in
    their calendar arm F1, and of the short-window arms (w = 24 and 48 h; what a shorter window saves)."""
    s = pd.read_csv(ANALYSIS / "systems.csv")
    keep = ((s.kind == "P") & ~s.model.isin(rx.BASELINE_MODELS)) | (
        (s.kind == "F") & (s.arm == "F1") & s.model.isin(["LightTCN_F_medium", "Vanilla"])) | (s.kind == "short window")
    rows = []
    for r in s[keep].itertuples():
        parts, seeds = [p.split("|") for p in r.runs.split(";")], r.seeds.split(";")
        rid = parts[seeds.index("42")][0] if "42" in seeds else parts[0][0]
        rows.append({"system": r.system, "group": r.group, "model": r.model, "regime": r.arm, "H": r.H, "run_id": rid})
    return pd.DataFrame(rows)


def _timed_ms(fn, budget_s: float = 8.0, warm: int = 5, min_rep: int = 10, max_rep: int = 200) -> tuple:
    """p50 and p95 wall time of fn() in ms, CUDA synchronised, after warm-up; the number of repetitions
    adapts to the cost of one call within [min_rep, max_rep]."""
    import torch

    def once():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t0
    for _ in range(warm):
        fn()
    n = int(min(max_rep, max(min_rep, budget_s / max(once(), 1e-6))))
    times = np.array([once() for _ in range(n)]) * 1e3
    return float(np.percentile(times, 50)), float(np.percentile(times, 95)), n


def _size_mib(path: Path) -> float:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2 ** 20
    return path.stat().st_size / 2 ** 20 if path.exists() else 0.0


def profile_one(rid: str) -> dict:
    """Load one saved system in this (fresh) process and measure it. End-to-end scope: raw load and calendar
    arrays of the origin in, forecast in MWh out, including the system's own preprocessing; model-only scope
    (torch and NeuralForecast): prepared tensors on the GPU."""
    import torch
    folder = RUNS / rid
    spec = json.loads((folder / "contract.json").read_text())["spec"]
    m, regime, H, w = spec["model"], spec["regime"], spec["H"], spec["w"]
    fam = rx.family(m)
    D = rx.Dataset()
    dev = torch.device("cuda")
    o_test = D.origins("test", H)
    t_data = time.time()
    torch.cuda.reset_peak_memory_stats()
    out = {"run_id": rid, "model": m, "regime": regime, "H": H, "w": w, "device": torch.cuda.get_device_name(0)}
    model_only, local, base = None, [], None

    if fam == "torch":
        ckpt = "checkpoint.pt" if (folder / "checkpoint.pt").exists() else "checkpoint_main.pt"   # F arms: per stream
        model = rx.load_torch_model(folder, D, ckpt).to(dev)
        local = [folder / ckpt]
        out["params_total"] = out["params_trained"] = int(sum(p.numel() for p in model.parameters()))
        if regime in rx.F_REGIMES:
            FI = rx.FInputs(D, regime, w, H)

            def e2e(o):
                past, fut = FI.arrays(o, "main")
                with torch.no_grad():
                    p = model(torch.from_numpy(past).to(dev), torch.from_numpy(fut).to(dev)).cpu().numpy()
                return p * D.y_std + D.y_mean
            n_feat, n_fut = FI.past.shape[1], FI.n_future
        else:
            def e2e(o):
                idx = o[:, None] + np.arange(-w, 0)[None, :]
                x = ((D.y[idx] - D.y_mean) / D.y_std).astype(np.float32)[..., None]
                if regime != "uni":
                    x = np.concatenate([D.calendar[idx], x], axis=-1)        # load is the last channel (models.py)
                with torch.no_grad():
                    p = model(torch.from_numpy(x).to(dev)).cpu().numpy()
                return p * D.y_std + D.y_mean
            n_feat, n_fut = D.past_matrix(regime).shape[1], 0

        def model_only(b):
            xs = [torch.randn(b, w, n_feat, device=dev)] + ([torch.randn(b, H, n_fut, device=dev)] if n_fut else [])
            return lambda: model(*xs)
        batch = rx.prediction_batch(w)
    elif fam == "nf":
        from neuralforecast import NeuralForecast
        import copy
        nf = NeuralForecast.load(str(folder / "nf_model"))
        local = [folder / "nf_model"]
        df, _ = rx.nf_dataframe(D, regime)
        out["params_total"] = out["params_trained"] = int(sum(p.numel() for p in nf.models[0].parameters()))
        copies = {}                     # the model-only copy is made after the end-to-end measurements, so that
                                        # their load time and peak memory see only the model nf.predict uses

        def e2e(o):
            res = []
            for t in o:
                fc = nf.predict(df=df.iloc[t - w:t])
                fc = (fc.reset_index() if "unique_id" not in fc.columns else fc).sort_values("ds")
                res.append(fc[[c for c in fc.columns if c not in ("unique_id", "ds")][0]].to_numpy(np.float64))
            return np.stack(res) * D.y_std + D.y_mean
        name = m.removeprefix("NF_")
        multi, X = name in ("TimeXer", "iTransformer"), (0 if regime == "uni" else len(rx.CALENDAR_FEATURES))

        def model_only(b):
            if "net" not in copies:
                copies["net"] = copy.deepcopy(nf.models[0]).to(dev).eval()
            net = copies["net"]
            y = torch.randn(b, w, 1, device=dev)
            hist = (torch.randn(b, X, w, 1, device=dev) if multi else torch.randn(b, w, X, device=dev)) if X else None
            bt = dict(insample_y=y, insample_mask=torch.ones_like(y), futr_exog=None, hist_exog=hist, stat_exog=None)
            return lambda: net(bt)
        batch = None                                            # one window per predict call
    elif fam in ("foundation", "foundation_ft"):
        if fam == "foundation":
            predict, n_params = rx._tsfm_load(m)
            out["params_total"], out["params_trained"] = n_params, 0
        else:
            import shutil
            import tempfile
            from chronos import Chronos2Pipeline
            # the adapter config names the base model by its hub name (the original records: by the absolute path
            # of the training machine); a temporary copy points it to the pinned snapshot on this machine (the run
            # folder is not touched)
            tmp = Path(tempfile.mkdtemp()) / "finetuned-ckpt"
            shutil.copytree(folder / "chronos_lora" / "finetuned-ckpt", tmp)
            cfg = json.loads((tmp / "adapter_config.json").read_text())
            cfg["base_model_name_or_path"] = str(rx.hf_snapshot(m))
            (tmp / "adapter_config.json").write_text(json.dumps(cfg))
            pipe = Chronos2Pipeline.from_pretrained(str(tmp), device_map="cuda")
            predict = rx._chronos_predictor(pipe)
            out["params_total"] = int(sum(p.numel() for p in pipe.model.parameters()))
            out["params_trained"] = json.loads((folder / "metrics.json").read_text()).get("lora_params")
            local = [folder / "chronos_lora"]
        base = rx.hf_snapshot(m)

        def e2e(o):
            return rx._tsfm_forecast(predict, D, regime, "main", o, w, H)
        batch = 256 if w <= 720 else 64
    elif fam == "lightgbm":
        import gzip
        import lightgbm as lgb
        with gzip.open(folder / "lightgbm_models.json.gz", "rt", encoding="utf-8") as f:
            blob = json.load(f)
        boosters = [lgb.Booster(model_str=s) for s in blob["lead_models"]]
        best = blob["best_rounds"]["main"]
        local = [folder / "lightgbm_models.json.gz"]
        out["params_total"] = out["params_trained"] = None
        out["trees"] = int(sum(best))

        def e2e(o):
            Xo = rx._lgb_base(D, o)
            return np.column_stack([b.predict(Xo, num_iteration=best[k]) for k, b in enumerate(boosters)])
        batch = rx.prediction_batch(w or 168)
    elif fam == "vanilla":
        beta = np.load(folder / "vanilla_beta.npy")
        local = [folder / "vanilla_beta.npy"]
        out["params_total"] = out["params_trained"] = int(len(beta))
        design = rx._vanilla_design(D, None, m == "Vanilla_holiday")

        def e2e(o):
            tau = o[:, None] + np.arange(H)[None, :]
            return design[tau] @ beta
        batch = 1024
    else:
        raise ValueError(f"no profile for family {fam}")
    torch.cuda.synchronize()
    out["t_loaded"] = time.time()
    out["model_load_s"] = out["t_loaded"] - t_data
    out["weights_gpu_mib"] = torch.cuda.memory_allocated() / 2 ** 20
    with np.load(folder / "test_sealed.npz") as z:
        stored = z["test_forecast_mwh"][0].astype(np.float64)
    out["max_abs_diff_vs_stored_mwh"] = float(np.max(np.abs(e2e(o_test[:1])[0] - stored)))
    torch.cuda.reset_peak_memory_stats()
    out["e2e_b1_p50_ms"], out["e2e_b1_p95_ms"], out["e2e_b1_reps"] = _timed_ms(lambda: e2e(o_test[:1]))
    out["peak_gpu_b1_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    if batch:
        torch.cuda.reset_peak_memory_stats()
        t50, _, n = _timed_ms(lambda: e2e(o_test[:batch]), budget_s=6.0, warm=1, min_rep=3, max_rep=20)
        out["e2e_batch"], out["e2e_throughput_per_s"] = batch, batch / (t50 / 1e3)
        out["peak_gpu_batch_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
    if model_only:
        with torch.no_grad():
            out["model_b1_p50_ms"], out["model_b1_p95_ms"], _ = _timed_ms(model_only(1))
            mb = rx.prediction_batch(w)
            t50, _, _ = _timed_ms(model_only(mb), budget_s=4.0, warm=2, min_rep=3, max_rep=20)
            out["model_batch"], out["model_throughput_per_s"] = mb, mb / (t50 / 1e3)
    out["local_files_mib"] = sum(_size_mib(p) for p in local)
    out["inference_weights_mib"] = out["local_files_mib"] + (_size_mib(base) if base else 0.0)
    out["note"] = "pure-PyTorch sLSTM kernels" if m == "TiRex2" else ""
    return out


PROFILE_TIMINGS = ["cold_start_s", "model_load_s", "e2e_b1_p50_ms", "e2e_b1_p95_ms", "e2e_throughput_per_s",
                   "model_b1_p50_ms", "model_b1_p95_ms", "model_throughput_per_s"]


def gpu_state() -> str:
    """GPU temperature (°C), graphics clock (MHz) and power draw (W) just before a measurement."""
    import subprocess
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.gr,power.draw",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


P_CORES = list(range(16))     # logical processors of the 8 performance cores of the i9-13900HX (hybrid CPU)


def run_profile(tag: str = "pin1") -> None:
    """One profiling pass: every reported system in its own fresh process (cold start = spawn to model loaded),
    written to efficiency_<tag>.csv with the GPU state and the time before each system. The process is pinned to
    the performance cores at high priority: left to Windows, it often ran on the efficiency cores, which halved
    the speed of CPU work and made the latencies bimodal. The systems are visited in a random order seeded by the
    tag. Resumes, retrying failed rows. The laptop GPU must be otherwise idle; `profile-merge` turns the passes into
    efficiency.csv."""
    import os
    import psutil
    out_file = ANALYSIS / f"efficiency_{tag}.csv"
    targets = profile_targets().sample(frac=1, random_state=sum(map(ord, tag))).reset_index(drop=True)
    rows = pd.read_csv(out_file).to_dict("records") if out_file.exists() else []
    rows = [r for r in rows if pd.notna(r.get("e2e_b1_p50_ms"))]   # failed rows are measured again
    done = {r["run_id"] for r in rows}
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for i, t in enumerate(targets.itertuples(), 1):
        if t.run_id in done:
            continue
        state, started = gpu_state(), datetime.now(timezone.utc).isoformat(timespec="seconds")
        t0 = time.time()
        proc = psutil.Popen([sys.executable, str(Path(__file__).resolve()), "profile-one", t.run_id], cwd=ROOT,
                            stdout=-1, stderr=-1, text=True, encoding="utf-8", errors="replace", env=env)
        proc.cpu_affinity(P_CORES)
        proc.nice(psutil.HIGH_PRIORITY_CLASS)
        stdout, stderr = proc.communicate(timeout=3600)
        line = next((l for l in reversed((stdout or "").splitlines()) if l.startswith("{")), None)
        if proc.returncode or line is None:
            err = (stderr or "").strip().splitlines()
            row = {"run_id": t.run_id, "model": t.model, "regime": t.regime, "H": t.H,
                   "note": f"failed: {err[-1][:200] if err else proc.returncode}"}
        else:
            row = json.loads(line)
            row["cold_start_s"] = row.pop("t_loaded") - t0
        rows.append({"system": t.system, "group": t.group, **row, "pass": tag, "order": i, "started_utc": started,
                     "gpu_temp_clock_power_before": state})
        pd.DataFrame(rows).to_csv(out_file, index=False)                # families report different fields
        print(f"[{tag} {i}/{len(targets)}] {t.system}: " + ", ".join(
            f"{k} {row[k]:.3g}" for k in ("cold_start_s", "e2e_b1_p50_ms", "peak_gpu_b1_mib") if isinstance(row.get(k), float))
            + f" [{state}]" + (f" {row['note']}" if row.get("note") else ""), flush=True)


def merge_profiles() -> None:
    """efficiency.csv: for every run the median of each timing over the profiling passes; memory, sizes and the
    reproduction check do not depend on the pass and are taken from the first pass (pinned passes only)."""
    passes = pd.concat([pd.read_csv(f) for f in sorted(ANALYSIS.glob("efficiency_pin*.csv"))], ignore_index=True)
    passes = passes[passes.e2e_b1_p50_ms.notna()]
    first = passes.sort_values("pass").drop_duplicates("run_id").set_index("run_id")
    first[PROFILE_TIMINGS] = passes.groupby("run_id")[PROFILE_TIMINGS].median()
    first["n_passes"] = passes.groupby("run_id").size()
    first = first.drop(columns=["pass", "order", "started_utc", "gpu_temp_clock_power_before"], errors="ignore")
    first.reset_index().to_csv(EFFICIENCY, index=False)
    print(f"{len(first)} systems, passes per system: {first.n_passes.value_counts().to_dict()} -> {EFFICIENCY}")


# ==============================================================================================
# Descriptive diagnostics of the dual-path TCNs (implementation note §11; no p-values, no model changed)
# ==============================================================================================
ATTENTION_ORIGINS = {"2025-03-30 00:00": "Ramazan Bayramı, day 1", "2025-06-06 00:00": "Kurban Bayramı, day 1",
                     "2025-03-19 00:00": "Wednesday", "2025-03-23 00:00": "Sunday"}   # fixed by calendar (note §11)
INPUT_HOUR_TYPES = ["religious holiday", "eve", "national holiday", "weekend", "weekday"]


def seed42_run(systems: pd.DataFrame, system: str) -> str:
    s = systems.set_index("system").loc[system]
    parts, seeds = [p.split("|") for p in s.runs.split(";")], s.seeds.split(";")
    return parts[seeds.index("42")][0] if "42" in seeds else parts[0][0]


def past_inputs(D: rx.Dataset, o: np.ndarray, w: int) -> np.ndarray:
    """The `cal` input of the networks: calendar of the input hours and the standardised load (last channel)."""
    idx = o[:, None] + np.arange(-w, 0)[None, :]
    y = ((D.y[idx] - D.y_mean) / D.y_std).astype(np.float32)[..., None]
    return np.concatenate([D.calendar[idx], y], axis=-1)


def run_diagnostics() -> None:
    """(i) matched grid pairs of DualTCN-Attn-M and DualTCN-M (same window, learning rate 1e-3 and seed 42);
    (ii) gate values, saturation and branch norms of the selected DualTCNs on the validation origins, with
    their learning curves; (iii) attention weights of the selected DualTCN-Attn-M at H = 24 (seed 42) at four
    origins fixed by calendar before any attention output was seen, normalised by the share of each input-hour
    type in the window. All descriptive."""
    import torch
    D = rx.Dataset()
    systems = pd.read_csv(ANALYSIS / "systems.csv")
    runs = run_table().drop_duplicates("run_id").set_index("run_id")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = pd.read_csv(TABLES / "S_grid.csv")
    g = g[g.model.isin(["DualTCN_Attn_medium", "DualTCN_medium"]) & (g.regime == "cal") & (g.H < 720)]
    pv = g.pivot_table(index=["H", "w"], columns="model", values=["val_MAPE", "test_MAPE", "lr_used"])
    matched = pd.DataFrame({"H": pv.index.get_level_values(0), "w": pv.index.get_level_values(1),
                            "lr_attn": pv[("lr_used", "DualTCN_Attn_medium")].to_numpy(),
                            "lr_dual": pv[("lr_used", "DualTCN_medium")].to_numpy(),
                            "val_MAPE_attn": pv[("val_MAPE", "DualTCN_Attn_medium")].to_numpy(),
                            "val_MAPE_dual": pv[("val_MAPE", "DualTCN_medium")].to_numpy(),
                            "test_MAPE_attn": pv[("test_MAPE", "DualTCN_Attn_medium")].to_numpy(),
                            "test_MAPE_dual": pv[("test_MAPE", "DualTCN_medium")].to_numpy()})
    if not np.allclose(matched.lr_attn, matched.lr_dual):
        sys.exit("the matched attention pairs must share the learning rate (note §11)")
    matched["test_diff_attn_minus_dual"] = matched.test_MAPE_attn - matched.test_MAPE_dual
    matched["val_diff_attn_minus_dual"] = matched.val_MAPE_attn - matched.val_MAPE_dual

    gate_rows = []
    for model in ("DualTCN_medium", "DualTCN_large", "DualTCN_Attn_medium", "DualTCN_Attn_large"):
        for H in (1, 24, 168):
            system = f"{model}|cal|H{H}"
            rid = seed42_run(systems, system)
            net = rx.load_torch_model(RUNS / rid, D).to(dev).eval()
            w = int(runs.loc[rid, "w"])
            stats = []
            o_all = D.origins("val", H)
            with torch.no_grad():
                for k in range(0, len(o_all), 1024):
                    x = torch.from_numpy(past_inputs(D, o_all[k:k + 1024], w)).to(dev)
                    cons = net.cons_tcn(x[:, :, -1:].transpose(1, 2)).transpose(1, 2)
                    exo = net.exo_tcn(x[:, :, :-1].transpose(1, 2)).transpose(1, 2)
                    t_last = cons[:, -1, :]
                    e = net.exo_readout(cons, exo)
                    gate = net.gate(torch.cat([t_last, e], dim=-1))
                    stats.append(torch.stack([gate.mean(1), ((gate > 0.9) | (gate < 0.1)).float().mean(1),
                                              t_last.norm(dim=1), e.norm(dim=1), (gate * t_last).norm(dim=1),
                                              ((1 - gate) * e).norm(dim=1)], dim=1).cpu())
            s = torch.cat(stats).numpy()
            info = runs.loc[rid]
            gate_rows.append({"system": system, "model": model, "H": H, "w": w, "run_id": rid,
                              "gate_mean_load_path": s[:, 0].mean(), "gate_saturated_share": s[:, 1].mean(),
                              "norm_load_state": s[:, 2].mean(), "norm_exogenous_state": s[:, 3].mean(),
                              "norm_load_contribution": s[:, 4].mean(), "norm_exogenous_contribution": s[:, 5].mean(),
                              "epochs_run": info.epochs_run, "best_epoch": info.best_epoch, "val_MAPE": info.val_MAPE,
                              "val_bias_pct": info.val_mean_signed_pct_error})
            del net
    single = sorted({seed42_run(systems, f"{m}|cal|H{H}") for m in
                     ("LightTCN_medium", "LightTCN_large", "ConcatTCN_medium", "ConcatTCN_large") for H in (1, 24, 168)})
    for rid, r in runs.loc[single].iterrows():          # single-path networks: learning curves only
        gate_rows.append({"system": f"{r.model}|cal|H{r.H}", "model": r.model, "H": r.H, "w": r.w, "run_id": rid,
                          "epochs_run": r.epochs_run, "best_epoch": r.best_epoch, "val_MAPE": r.val_MAPE,
                          "val_bias_pct": r.val_mean_signed_pct_error})
    gates = pd.DataFrame(gate_rows)

    system = "DualTCN_Attn_medium|cal|H24"
    rid = seed42_run(systems, system)
    net = rx.load_torch_model(RUNS / rid, D).to(dev).eval()
    net.keep_attention = True
    w = int(runs.loc[rid, "w"])
    labels = list(D.labels)
    o = np.array([labels.index(t) for t in ATTENTION_ORIGINS])
    with torch.no_grad():
        net(torch.from_numpy(past_inputs(D, o, w)).to(dev))
    alpha = net.last_attention.cpu().numpy().astype(np.float64)              # (4, w), rows sum to 1
    kind = pd.read_csv(rx.DATA_FILE, usecols=["holiday_type"]).holiday_type.fillna("").to_numpy(str)
    hour_type = np.select([kind == "religious", kind == "half_day", kind == "national", D.dow >= 5],
                          INPUT_HOUR_TYPES[:4], INPUT_HOUR_TYPES[4])
    att_rows, summary_rows = [], []
    for i, (t, name) in enumerate(ATTENTION_ORIGINS.items()):
        idx = o[i] + np.arange(-w, 0)
        types = hour_type[idx]
        for lag, j, a in zip(range(w, 0, -1), idx, alpha[i]):
            att_rows.append({"origin": t, "origin_type": name, "lag_h": lag, "input_hour": D.labels[j],
                             "input_hour_type": hour_type[j], "weight": a, "weight_x_w": a * w})
        groups = {f"type: {k}": types == k for k in INPUT_HOUR_TYPES}
        groups |= {"recency: last 24 h": np.arange(-w, 0) >= -24,
                   "recency: 25-168 h": (np.arange(-w, 0) < -24) & (np.arange(-w, 0) >= -168),
                   "recency: older than 168 h": np.arange(-w, 0) < -168,
                   "same hour of day as the origin": (np.arange(-w, 0) % 24) == 0}
        for gname, m in groups.items():
            if m.any():
                share = m.mean()
                summary_rows.append({"origin": t, "origin_type": name, "group": gname, "hours": int(m.sum()),
                                     "share_of_window": share, "attention_mass": alpha[i][m].sum(),
                                     "mass_over_share": alpha[i][m].sum() / share})
    pd.DataFrame(att_rows).to_csv(ANALYSIS / "diagnostics_attention.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(ANALYSIS / "diagnostics_attention_summary.csv", index=False)
    gates.to_csv(ANALYSIS / "diagnostics_gates.csv", index=False)
    matched.to_csv(ANALYSIS / "diagnostics_attention_matched.csv", index=False)
    print(f"diagnostics: {len(matched)} matched pairs, {len(gates)} gate/learning-curve rows, attention of {rid} "
          f"(w = {w}) at {len(o)} origins -> {ANALYSIS}")


# ==============================================================================================
# Manuscript (Word through pandoc). The text is manuscript/manuscript.md; `{{table:Tn}}` is replaced by a
# table built here from the analysis files and `[@key]` by a number in order of first citation.
# ==============================================================================================
MANUSCRIPT = ROOT / "manuscript"
DRAFT = "1.6"                     # version in the names of the Word files (earlier drafts stay untouched)
TNR = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman"/>'


def reference_docx(landscape: bool = False) -> Path:
    """Word template for pandoc: A4, 2.5 cm margins, Times New Roman 12 pt, 1.5 lines, black headings, 10 pt
    captions and tables, horizontal table rules only (no vertical rules, no shading). No line numbers (the
    authors add them for submission)."""
    import re
    import subprocess
    import zipfile
    MANUSCRIPT.mkdir(exist_ok=True)
    default = MANUSCRIPT / "_pandoc_default.docx"
    out = MANUSCRIPT / ("reference_landscape.docx" if landscape else "reference.docx")
    subprocess.run(["pandoc", "-o", str(default), "--print-default-data-file", "reference.docx"], check=True)

    def style(xml, sid, ppr=None, rpr=None):
        m = re.search(r'(<w:style [^>]*w:styleId="%s".*?)(</w:style>)' % sid, xml, re.S)
        block = re.sub(r"<w:pPr>.*?</w:pPr>|<w:pPr ?/>", "", m.group(1), flags=re.S)
        block = re.sub(r"<w:rPr>.*?</w:rPr>|<w:rPr ?/>", "", block, flags=re.S)
        new = block + (f"<w:pPr>{ppr}</w:pPr>" if ppr else "") + (f"<w:rPr>{rpr}</w:rPr>" if rpr else "") + m.group(2)
        return xml[:m.start()] + new + xml[m.end():]

    line = '<w:spacing w:before="0" w:after="120" w:line="360" w:lineRule="auto"/>'
    with zipfile.ZipFile(default) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    st = parts["word/styles.xml"].decode("utf8")
    st = re.sub(r"<w:rPrDefault>.*?</w:rPrDefault>", f'<w:rPrDefault><w:rPr>{TNR}<w:sz w:val="24"/><w:szCs w:val="24"/>'
                f'<w:lang w:val="en-GB" w:eastAsia="en-US" w:bidi="ar-SA"/></w:rPr></w:rPrDefault>', st, flags=re.S)
    st = re.sub(r"<w:pPrDefault>.*?</w:pPrDefault>", f"<w:pPrDefault><w:pPr>{line}</w:pPr></w:pPrDefault>", st, flags=re.S)
    st = style(st, "BodyText", ppr=line + '<w:jc w:val="both"/>')
    # table cells: in the paper every row keeps with the next, so a short table is not split across pages; the
    # long supplementary tables may break
    st = style(st, "Compact", ppr=("" if landscape else "<w:keepNext/>")
               + '<w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto"/>',
               rpr='<w:sz w:val="18"/><w:szCs w:val="18"/>')
    st = style(st, "CaptionedFigure", ppr="<w:keepNext/>")              # a figure stays with its caption
    st = style(st, "Title", ppr='<w:spacing w:after="240" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/>',
               rpr=f'{TNR}<w:b/><w:sz w:val="28"/><w:szCs w:val="28"/>')
    st = style(st, "Author", ppr='<w:spacing w:after="120" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/>',
               rpr=f'{TNR}<w:sz w:val="24"/>')
    for sid, lvl, extra in (("Heading1", 0, "<w:b/>"), ("Heading2", 1, "<w:b/>"), ("Heading3", 2, "<w:i/>")):
        st = style(st, sid, ppr=f'<w:keepNext/><w:keepLines/><w:spacing w:before="240" w:after="120" w:line="360" '
                                f'w:lineRule="auto"/><w:outlineLvl w:val="{lvl}"/>',
                   rpr=f'{TNR}{extra}<w:color w:val="000000"/><w:sz w:val="24"/><w:szCs w:val="24"/>')
    cap = '<w:spacing w:before="120" w:after="120" w:line="240" w:lineRule="auto"/><w:jc w:val="both"/>'
    st = style(st, "Caption", ppr=cap, rpr='<w:sz w:val="20"/><w:szCs w:val="20"/>')
    # a caption is never split across pages; a table caption stays with its table
    st = style(st, "TableCaption", ppr="<w:keepNext/><w:keepLines/>" + cap, rpr='<w:sz w:val="20"/><w:szCs w:val="20"/>')
    st = style(st, "ImageCaption", ppr="<w:keepLines/>" + cap, rpr='<w:sz w:val="20"/><w:szCs w:val="20"/>')
    st = style(st, "Abstract", ppr=line + '<w:jc w:val="both"/>', rpr='<w:sz w:val="24"/><w:szCs w:val="24"/>')
    st = style(st, "Bibliography", ppr='<w:spacing w:before="0" w:after="60" w:line="240" w:lineRule="auto"/>'
                                       '<w:ind w:left="454" w:hanging="454"/>', rpr='<w:sz w:val="20"/><w:szCs w:val="20"/>')
    rule = '<w:{0} w:val="single" w:sz="{1}" w:space="0" w:color="000000"/>'
    st = st.replace('<w:tblInd w:w="0" w:type="dxa" />\n      <w:tblCellMar>',
                    '<w:tblInd w:w="0" w:type="dxa" /><w:tblBorders>' + rule.format("top", 8) + rule.format("bottom", 8)
                    + '</w:tblBorders>\n      <w:tblCellMar>', 1)
    st = st.replace('<w:bottom w:val="single"/>', rule.format("bottom", 4), 1)
    parts["word/styles.xml"] = st.encode("utf8")
    doc = parts["word/document.xml"].decode("utf8")
    size = '<w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/>' if landscape else '<w:pgSz w:w="11906" w:h="16838"/>'
    doc = re.sub(r"<w:sectPr>.*?</w:sectPr>", '<w:sectPr><w:footnotePr><w:numRestart w:val="eachSect"/></w:footnotePr>'
                 f'{size}<w:pgMar w:top="1417" w:right="1417" w:bottom="1417" w:left="1417" '
                 'w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>', doc, flags=re.S)
    parts["word/document.xml"] = doc.encode("utf8")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n, data in parts.items():
            z.writestr(n, data)
    default.unlink()
    return out


def fmt(x, d: int = 3, keep_sign: bool = False) -> str:
    """Rounded for display with a true minus sign. A value that rounds to zero loses its sign, except an interval
    bound (keep_sign), whose sign tells on which side of zero it lies."""
    if pd.isna(x):
        return "–"
    s = f"{x:,.{d}f}"
    if not keep_sign and s.startswith("-") and float(s[1:].replace(",", "")) == 0:
        s = s[1:]
    return s.replace("-", "−")


DISPLAY_ORDER = [m for _, members in P_ROWS for m, _ in members] + ["LightTCN_F_medium", "Vanilla", "Vanilla_holiday",
                                                                   "Naive", "SeasonalNaive", "SeasonalNaiveWeekly"]


def display_key(model: str, arm: str = "") -> tuple:
    """Rows of the supplementary tables in the order of the paper: families as in Fig. 3, then LightTCN-F's arms."""
    m = DISPLAY_ORDER.index(model) if model in DISPLAY_ORDER else len(DISPLAY_ORDER)
    a = F_ORDER.index(arm) if arm in F_ORDER else ["cal", "uni"].index(arm) if arm in ("cal", "uni") else 99
    return m, a


def fmt_p(p) -> str:
    return "–" if pd.isna(p) else "<0.001" if p < 0.001 else f"{p:.3f}"


# Main-text tables and figures are numbered in the order of first mention in manuscript.md (build_manuscript checks
# it); the text refers to them as {{tab:KEY}} and {{fig:KEY}}. Supplementary tables follow the order in which
# supplement_notes.md places them, and {{stab:KEY}} prints their number (S1, S2, ...). Tables of MOVED are built
# with the main-text tables but printed in the supplement.
TABLE_NUMBERS = {"TD": 1, "T4": 2, "T3": 3}
FIGURE_NUMBERS = {"data": 1, "architecture": 2, "p_errors": 3, "information": 4, "profiles": 5, "energy": 6,
                  "efficiency": 7}
SUPP_TABLES = ["literature", "features", "design", "models", "setup", "lr", "p_systems", "grid", "comparisons", "contrasts",
               "f_systems", "matched", "day_types", "energy_summary", "energy_months", "efficiency", "training",
               "runs", "attention_matched", "gates", "attention_summary"]
SUPP_FIGURES = ["leads", "attention"]
MOVED = {"TF": "features", "T1": "design", "TM": "models", "TS": "setup", "T5": "matched", "T6": "energy_summary"}


def table_label(key: str) -> str:
    """The printed number of a table: '3' in the main text, 'S11' in the supplement."""
    return str(TABLE_NUMBERS[key]) if key in TABLE_NUMBERS else f"S{SUPP_TABLES.index(MOVED.get(key, key)) + 1}"


def fill_numbers(text: str) -> str:
    """Replace {{tab:K}}, {{fig:K}}, {{stab:K}} and {{sfig:K}} by the printed numbers."""
    import re
    text = re.sub(r"\{\{tab:(\w+)\}\}", lambda m: str(TABLE_NUMBERS[m.group(1)]), text)
    text = re.sub(r"\{\{fig:(\w+)\}\}", lambda m: str(FIGURE_NUMBERS[m.group(1)]), text)
    text = re.sub(r"\{\{stab:(\w+)\}\}", lambda m: f"S{SUPP_TABLES.index(m.group(1)) + 1}", text)
    return re.sub(r"\{\{sfig:(\w+)\}\}", lambda m: f"S{SUPP_FIGURES.index(m.group(1)) + 1}", text)


def md_table(df: pd.DataFrame, caption: str, align: str, widths: list[int], number: int | str) -> str:
    """A pandoc pipe table with its numbered caption; align is one character per column (l, r or c) and
    widths are relative column widths (pandoc reads them from the lengths of the separator dashes)."""
    def cell(v):
        return str(v).replace("|", "\\|")
    head = "| " + " | ".join(cell(c) for c in df.columns) + " |\n|"
    head += "|".join((":" if a in "lc" else "-") + "-" * (w - 2) + (":" if a in "rc" else "-")
                     for a, w in zip(align, widths)) + "|\n"
    body = "".join("| " + " | ".join(cell(v) for v in row) + " |\n" for row in df.itertuples(index=False))
    spacer = ('```{=openxml}\n<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="120" w:lineRule="exact"/>'
              '</w:pPr></w:p>\n```\n')                    # 6 pt of space between the table and the next paragraph
    return f"{head}{body}\nTable: **Table {number}.** {caption}\n\n{spacer}"


def model_name(model: str, regime: str | None = None) -> str:
    return NAMES.get(model, model) + (f" ({regime})" if regime else "")


def manuscript_tables() -> dict:
    """The main-text tables and the tables of MOVED as pandoc markdown, built from the analysis files (rounding
    happens only here)."""
    T = {name: pd.read_csv(TABLES / f"{name}.csv") for name in
         ("T1_design", "T2_p_systems", "T3_information_systems", "T4_comparisons", "T5_matched", "T6_aggregation")}
    out = {}
    runs = run_table()
    rows = []
    systems = pd.read_csv(ANALYSIS / "systems.csv")
    for r in table_models(systems, runs, T["T2_p_systems"]).itertuples():
        size = ("119.5 M frozen + 1.2 M trained" if r.model == "Chronos2_LoRA" else "–" if pd.isna(r.params_max) else
                f"{r.params_min:.0f}–{r.params_max:.0f} coefficients" if r.params_max < 1e4 else
                f"{r.params_min / 1e6:.2f}–{r.params_max / 1e6:.2f} M" if r.params_min != r.params_max
                else f"{r.params_max / 1e6:.1f} M")
        rows.append([r.name, r.model_class, size, r.P_inputs, r.F_arms])
    out["TM"] = md_table(pd.DataFrame(rows, columns=["Model", "Class", "Parameters", "P inputs", "F arms"]),
                         "Models. Parameters: range over the validation-selected past-only systems and known-future arms "
                         "(the TCN depth grows with the window), including the pretrained weights of the foundation "
                         "models; the LoRA adapter's trained parameters are merged into the frozen ones after training. "
                         "P inputs: uni, load history; "
                         "cal, load history and past calendar. F arms: known-future arms (Table "
                         f"{table_label('T1')}).",
                         "llrll", [18, 24, 22, 12, 24], table_label("TM"))
    d = T["T1_design"]
    rows = [(r.item[:1].upper() + r.item[1:] if r.section != "origins" else f"Origins, {r.item} h", r.value)
            for r in d.itertuples() if r.section in ("data", "split", "origins", "information set")]
    out["T1"] = md_table(pd.DataFrame(rows, columns=["Item", "Definition"]),
        "Data, splits, forecast origins and information sets. An origin is the first target hour; every model is "
        "scored on the same origins. F arms are validated from 17 March 2024, the first origin at which bias-corrected "
        "temperature forecasts are complete for every lead; the exceptions are N-HiTS F0 and F1, which use all of 2024 (the F-arm "
        "counts in parentheses do not apply to them); the holiday-analog arms (F1A0, F1A) are exploratory.",
        "ll", [22, 78], table_label("T1"))

    f = T["T3_information_systems"]
    f = f[f.model == "LightTCN_F_medium"]
    labels = {"F0": "F0: future inputs set to zero", "F1": "F1: F0 + calendar of the target hours",
              "F1A0": "F1A0: F1 + analog flag", "F1A": "F1A: F1 + holiday analog",
              "F2": "F2: F1 + past temperature", "F3": "F3: F2 + forecast temperature",
              "F3raw": "F3raw: raw forecasts (own checkpoint)", "F4": "F4: ERA5 oracle scenario (own checkpoint)",
              "F3_m24": "F3_m24: 24 h vintage margin (F3 checkpoint)", "F3_diag": "F3_diag: ERA5 future (F3 checkpoint)",
              "F3raw_diag": "F3raw_diag: raw forecasts (F3 checkpoint)"}
    rows = []
    for arm in [a for a in F_ORDER if a not in ("F3_m24", "F3_diag", "F3raw_diag")]:   # those: Supplementary Table
        cells = []
        for H in HORIZONS:
            s = f[(f.arm == arm) & (f.H == H)]
            cells.append("–" if s.empty else f"{fmt(s.iloc[0].MAPE)} ({fmt(s.iloc[0].MAPE_sd)})")
        rows.append([F_BRANCH[arm].replace("holiday analog", "analog"), labels[arm], *cells])
    out["T3"] = md_table(pd.DataFrame(rows, columns=["Branch", "Arm", "H = 1 h", "H = 24 h", "H = 168 h", "H = 720 h"]),
                         "Test-year MAPE (%) of LightTCN-F under the information arms (three seeds; standard deviation "
                         "in parentheses); the analog rows are exploratory (Section 5.3). F3, F3raw and F4 are checkpoints of "
                         "one training run, each selected on its own validation stream; –: arm not run at this horizon. At H = 1 h the previous "
                         "hour's load is assumed known (with a 2 h delay, LightTCN-M: 0.60% → 1.03%; Section 6.4).", "llrrrr",
                         [10, 34, 14, 14, 14, 14], table_label("T3"))

    c = T["T4_comparisons"]
    c = c[c.loss == "APE"]
    rows = []
    for family, header, usual in (("primary", "*Primary comparisons, past calendar unless marked (Holm over 11)*", "cal"),
                                  ("secondary", "*Secondary comparisons, univariate (Holm over 3)*", "uni")):
        rows.append([header] + [""] * 6)
        for r in c[c.family == family].itertuples():
            a, b = (x.split("|") for x in (r.A, r.B))
            label = [NAMES.get(m, m) + ("" if reg == usual else f" ({reg})") for m, reg in (a[:2], b[:2])]
            rows.append([f"{label[0]} − {label[1]}", r.H, f"{int(r.n_A)}/{int(r.n_B)}", fmt(r.diff),
                         f"{fmt(r.ci_low, keep_sign=True)} to {fmt(r.ci_high, keep_sign=True)}", fmt_p(r.p_boot),
                         fmt_p(r.p_holm)])
    out["T4"] = md_table(pd.DataFrame(rows, columns=["Comparison (A − B)", "H (h)", "Seeds", "ΔMAPE (pp)", "95% CI",
                                                     "p", "Holm p"]),
                         "Pre-specified comparisons on the test year. Seeds: A/B. ΔMAPE: mean over hourly origins of the per-origin "
                         "loss difference; negative values favour A. Intervals and p-values: paired moving-block bootstrap "
                         f"(Section 5.3; other block lengths and the squared-error loss in Supplementary Table "
                         f"{table_label('comparisons')}); Holm's adjustment within each family. DualTCN-Attn and DualTCN were tuned separately, so their pairs are not a "
                         "mechanism ablation. At H = 1 h the previous hour's load is assumed known (Section 6.4).", "lrcrrrr",
                         [36, 7, 8, 11, 18, 10, 10], table_label("T4"))

    m = T["T5_matched"]
    rows = [[r.kind, r.H, NAMES.get(r.model, r.model), r.intervention.replace("t-", "t − ") + (" h" if r.intervention.startswith("w = ") else ""),
             "–" if r.model in rx.BASELINE_MODELS else r.seeds,
             fmt(r.MAPE_reference), fmt(r.MAPE_arm),
             fmt(r.diff), fmt(r.rel_diff_pct, 1)] for r in m.itertuples()]
    out["T5"] = md_table(pd.DataFrame(rows, columns=["Intervention", "H (h)", "Model", "Change", "Seed IDs", "Reference MAPE",
                                                     "Arm MAPE", "Δ (pp)", "Δ (%)"]),
                         "Matched interventions: each arm against the full system with the same seeds, horizon and "
                         "learning rate and, except for the short-window arms, the same window (MAPE in %; Δ = arm − "
                         "reference). Ablations remove one calendar group from the input channels; the publication "
                         "delay ends the inputs at t − 3 instead of t − 1; short windows change the network depth "
                         "together with the window. The naive references have no seeds.", "lrllcrrrr", [15, 6, 13, 18, 11, 10, 9, 9, 9],
                         table_label("T5"))

    g = T["T6_aggregation"]
    g = g[g.model != "SeasonalNaive"]                  # at H = 168 h it is the weekly seasonal naive
    rows = [[NAMES.get(r.model, r.model), r.regime,
             int(r.n_seeds), fmt(r.hourly_MAPE, 2), fmt(r.mean_monthly_energy_APE, 2), fmt(r.max_monthly_energy_APE, 2),
             fmt(r.annual_energy_bias_pct, 2), fmt(r.annual_cancellation, 3)] for r in g.itertuples()]
    out["T6"] = md_table(pd.DataFrame(rows, columns=["Model", "Inputs", "Seeds", "Hourly MAPE (%)", "Monthly energy APE, mean (%)",
                                                     "Monthly energy APE, max (%)", "Annual energy bias (%)", "|Σe| / Σ|e|"]),
                         "Monthly energy totals of weekly updated H = 168 h forecasts (origins every 168 h from "
                         "1 January 2025; every hour of 2025 appears once). Seed-mean forecast of each system; APE: absolute "
                         "percentage error; the seasonal naive forecast coincides with its weekly variant at H = 168 h "
                         "and is listed once; annual "
                         "energy bias: 100·(forecast − actual)/actual of the annual total; |Σe| / Σ|e| is the annual share "
                         "of hourly errors that does not cancel.", "llcrrrrr",
                         [19, 9, 8, 12, 14, 14, 13, 11], table_label("T6"))

    d = pd.read_csv(TABLES / "TD_dataset.csv")
    unit = {"Minimum": "{:,.0f} MWh", "Maximum": "{:,.0f} MWh", "Mean": "{:,.0f} MWh", "Standard deviation": "{:,.0f} MWh",
            "Observations": "{:,.0f} h", "Training set": "{:,.0f} h", "Validation set": "{:,.0f} h", "Test set": "{:,.0f} h"}
    rows = []
    for r in d.itertuples():
        fmt_ = unit.get(r.item, "{:.1f} TWh" if r.item.startswith("Annual") else "{}")
        value = fmt_.format(float(r.value)) if fmt_ != "{}" else r.value
        rows.append([r.item, value, "" if r.description in ("TWh",) else r.description])
    out["TD"] = md_table(pd.DataFrame(rows, columns=["Item", "Value", "Description"]),
                         "The EPİAŞ hourly consumption series, 2016–2025, and the calendar-year split.", "lll", [26, 30, 44],
                         table_label("TD"))
    f = pd.read_csv(TABLES / "TF_features.csv")
    out["TF"] = md_table(f.set_axis(["Feature", "Definition", "Range", "Description", "Used in"], axis=1),
                         "Input features. cal: the calendar of the input hours in the past-only comparison; F: the "
                         "known-future arms, which also give the calendar of the target hours. Temperature ranges are "
                         "those of the hourly ERA5 values, 2016–2025; wₚ are the provinces' population shares and Tₚ their temperatures.",
                         "lllll", [16, 30, 18, 24, 12], table_label("TF"))
    s = pd.read_csv(TABLES / "TS_setup.csv").fillna("")
    out["TS"] = md_table(s.set_axis(["Parameter", "Value", "Note"], axis=1),
                         "Experimental setup and training hyperparameters of the networks trained from scratch "
                         "(LightGBM, the linear benchmark and the LoRA fine-tuning follow their own rules, Note S3).",
                         "lll", [20, 44, 36], table_label("TS"))
    return out


def cite_numbers(text: str) -> tuple[str, list]:
    """Replace [@a; @b] by [1,2] in order of first citation; ranges of three or more become 1–3."""
    import re
    order = []

    def one(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        nums = []
        for k in keys:
            if k not in order:
                order.append(k)
            nums.append(order.index(k) + 1)
        nums = sorted(set(nums))
        spans, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            spans.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(map(str, nums[i:j + 1])))
            i = j + 1
        return "[" + ",".join(spans) + "]"
    return re.sub(r"\[(@[^\]]+)\]", one, text), order


JOURNALS = {  # ISO 4 (LTWA) abbreviations of the cited journals, as Energy asks; other containers are printed in full
    "International Journal of Forecasting": "Int J Forecast", "Applied Energy": "Appl Energy", "Energies": "Energies", "Cognitive Computation": "Cogn Comput",
    "Energy": "Energy", "Energy Reports": "Energy Rep",
    "Engineering Applications of Artificial Intelligence": "Eng Appl Artif Intell",
    "Journal of Building Engineering": "J Build Eng", "Expert Systems with Applications": "Expert Syst Appl",
    "IEEE Access": "IEEE Access", "Energy and Buildings": "Energy Build",
    "Electric Power Systems Research": "Electr Power Syst Res",
    "Renewable and Sustainable Energy Reviews": "Renew Sustain Energy Rev",
    "Journal of Modern Power Systems and Clean Energy": "J Mod Power Syst Clean Energy",
    "Sustainable Energy Technologies and Assessments": "Sustain Energy Technol Assess",
    "Journal of Business & Economic Statistics": "J Bus Econ Stat",
    "Journal of Environmental Economics and Management": "J Environ Econ Manag", "Energy Policy": "Energy Policy",
    "Advances in Mechanical Engineering": "Adv Mech Eng", "Applied Soft Computing": "Appl Soft Comput",
    "Journal of the Operational Research Society": "J Oper Res Soc",
    "Sustainable Energy, Grids and Networks": "Sustain Energy Grids Netw", "Energy and AI": "Energy AI",
    "IEEE Transactions on Neural Networks and Learning Systems": "IEEE Trans Neural Netw Learn Syst",
    "Quarterly Journal of the Royal Meteorological Society": "Q J R Meteorol Soc",
    "Transactions on Machine Learning Research": "Trans Mach Learn Res",
    "Energy Conversion and Management: X": "Energy Convers Manag X",
    "International Journal of Energy Economics and Policy": "Int J Energy Econ Policy", "The Annals of Statistics": "Ann Stat",
    "Scandinavian Journal of Statistics": "Scand J Stat", "Renewable Energy": "Renew Energy",
    "International Journal of Electrical Power & Energy Systems": "Int J Electr Power Energy Syst",
    "Sustainable Cities and Society": "Sustain Cities Soc", "Advanced Engineering Informatics": "Adv Eng Inform",
    "Neural Computing and Applications": "Neural Comput Appl", "Sustainability": "Sustainability",
    "Electronics": "Electronics", "Applied Mathematical Modelling": "Appl Math Model",
    "International Journal of Hydrogen Energy": "Int J Hydrogen Energy", "Applied Sciences": "Appl Sci",
    "Artificial Intelligence Review": "Artif Intell Rev", "Procedia Computer Science": "Procedia Comput Sci",
    "Journal of Energy Systems": "J Energy Syst",
    "Niğde Ömer Halisdemir Üniversitesi Mühendislik Bilimleri Dergisi": "Niğde Ömer Halisdemir Üniv Mühendis Bilim Derg"}


def format_reference(r: dict) -> str:
    """Energy's numbered style: 'Surname AB, ... Title. J Abbrev Year;Volume:Pages. DOI', at most six authors then
    'et al.', last page shortened (925–38); conference papers 'In: Proceedings; Year, p. Pages'; reports, software and
    data sets 'Container; Year', data sets tagged '[dataset]' and web resources without a DOI ending with the access
    date. An entry with a `text` field is printed as given."""
    if r.get("text"):
        return r["text"]

    def name(a):
        sur, _, ini = a.partition(", ")
        return f"{sur} {''.join(c for c in ini if c.isalpha()).upper()}".strip()        # 'Nie, R.-x.' -> 'Nie RX'

    def pages(p):
        a, sep, b = str(p).replace("–", "-").partition("-")
        if sep and a.isdigit() and b.isdigit() and len(a) == len(b):
            k = next((i for i in range(len(a)) if a[i] != b[i]), len(a) - 1)
            b = b[k:]
        return f"{a}–{b}" if sep else a
    authors = [name(a) for a in r.get("authors", [])]
    who = ", ".join(authors[:6]) + (", et al" if len(authors) > 6 else "")
    title = r["title"].rstrip(".") + (f" [{r['title_note']}]" if r.get("title_note") else "")
    title += f". Version {r['version']}" if r.get("version") else ""
    if r.get("type") == "software" and "[" not in title:
        title += " [software]"
    year, container, volume = str(r.get("year", "")), r.get("container", ""), str(r.get("volume") or "")
    if container in JOURNALS or r.get("type") == "journal":        # journals, and proceedings published as one
        where = f"{JOURNALS.get(container, container)} {year}" + (f";{volume}" if volume else "")
        where += f":{pages(r['pages_or_article'])}" if r.get("pages_or_article") else ""
    elif r.get("type") == "conference":
        where = "In: " + container + (f", vol. {volume}" if volume and volume not in container else "")
        where += f"; {year}" if year else ""
        where += f", p. {pages(r['pages_or_article'])}" if r.get("pages_or_article") else ""
    else:
        where = container + (f"; {year}" if year and year not in container else "")
    link = f" https://doi.org/{r['doi']}" if r.get("doi") else (f" {r['url']}" if r.get("url") else "")
    accessed = (f" [accessed {r['accessed']}]" if r.get("accessed") and (not r.get("doi") or r.get("type") == "dataset")
                else "")                                    # growing data sets carry the date even with a DOI
    tag = "[dataset] " if r.get("type") == "dataset" else ""
    return f"{tag}{who}. {title}{'' if title[-1] in '?!' else '.'} {where}.{link}{accessed}"


DESIGN_NUMBERS = {  # constants of the design and the code, and external facts the text quotes (not analysis results)
    0.05, 0.15, 0.2, 0.25, 0.5, 0.7, 0.8, 0.9, 1.1, 5.0, 7.3, 22.8, 360.9, 1440, 2160, 2880, 2000, 10000, 336, 720, 168,
    145, 158, 320, 2016, 2017, 2019, 2020, 2022, 2023, 2024, 2025, 2026, 2015, 81, 128, 4090,
    2429, 6282, 10711, 13900}                        # Law No. 2429, EPDK decisions 6282-4 and 10711, i9-13900HX
ID_COLUMNS = ("system", "run_id", "variant", "family", "loss", "A", "B", "intervention", "category", "month", "H",
              "origin", "group")


def number_bank(D: rx.Dataset) -> dict:
    """Every analysis value the text may quote with where it comes from: {decimals: {rounded |value|: [source]}}
    (up to three sources per value, as file:column:row; shares and cancellation also in %)."""
    import models
    bank = {d: {} for d in range(5)}

    def add(values, sources):
        for v, src in zip(np.abs(np.asarray(values, float)), sources):
            if np.isfinite(v):
                for d in range(5):
                    hits = bank[d].setdefault(round(float(v), d), [])
                    if len(hits) < 8:
                        hits.append(src)
    files = ["systems.csv", "inference.csv", "inference_seeds.csv", "breakdowns_systems.csv", "aggregation_summary.csv",
             "aggregation_monthly.csv", "run_metrics.csv", "tables/T5_matched.csv", "tables/S_grid.csv",
             "tables/TE_efficiency.csv", "tables/S_training.csv", "diagnostics_gates.csv",
             "diagnostics_attention_matched.csv", "diagnostics_attention_summary.csv"]
    for f in files:
        if not (ANALYSIS / f).exists():
            continue
        t = pd.read_csv(ANALYSIS / f)
        ids = [c for c in ID_COLUMNS if c in t.columns]
        label = t[ids].astype(str).agg("|".join, axis=1) if ids else t.index.astype(str).to_series()
        for c in t.select_dtypes("number").columns:
            ok = t[c].notna()
            add(t[c][ok], [f"{f}:{c}:{x}" for x in label[ok]])
            if "share" in c or "cancellation" in c:
                add(100 * t[c][ok], [f"{f}:{c} x 100:{x}" for x in label[ok]])
                add(100 * (1 - t[c][ok]), [f"{f}:(1 - {c}) x 100:{x}" for x in label[ok]])
    td = pd.read_csv(TABLES / "TD_dataset.csv")
    v = pd.to_numeric(td.value, errors="coerce")
    add(v[v.notna()], [f"TD_dataset.csv:{i}" for i in td.item[v.notna()]])
    runs = pd.read_csv(ROOT / "results" / "runs.csv").drop_duplicates("run_id")
    for c, scale, name in (("params", 1e6, "params (M)"), ("wall_sec", 60, "wall time (min)"),
                           ("peak_gpu_mem_mb", 1024, "peak GPU (GiB)")):
        ok = runs[c].notna()
        add(runs[c][ok] / scale, [f"runs.csv:{name}:{r}" for r in runs.run_id[ok]])
    add([1206912 / 1e6], ["Chronos2_LoRA metrics.json: lora_params (M)"])
    gpu = runs.device.str.contains("NVIDIA", na=False)
    nets = runs[runs.epochs_run.notna()]                  # the networks trained with the shared loop
    cap = nets[nets.epochs_run >= nets.epochs_run.max()]
    add([len(nets), len(nets) - len(cap), len(cap), cap.model.str.startswith("LightTCN").sum()],
        ["runs.csv: network runs", "runs.csv: stopped early", "runs.csv: at the epoch limit",
         "runs.csv: LightTCN runs at the epoch limit"])
    add([runs.run_id.nunique(), runs.wall_sec.sum() / 3600, runs.wall_sec[gpu].sum() / 3600],
        ["runs.csv: runs", "runs.csv: total run-hours", "runs.csv: run-hours on the GPU partitions"])
    p2 = pd.read_csv(TABLES / "T2_p_systems.csv")
    gain = (p2[p2.kind == "P"].set_index(["model", "regime", "H"]).MAPE
            - p2[p2.kind == "P seed mean"].set_index(["model", "regime", "H"]).MAPE).dropna()
    add([gain.min(), gain.max()], ["T2_p_systems: smallest seed-mean gain (pp)", "T2_p_systems: largest seed-mean gain (pp)"])
    unpinned = sorted((ROOT / "archive" / "profile_unpinned_2026-10-01").glob("efficiency_pass*.csv"))
    if unpinned:                                   # the three earlier passes without pinning (Note S3)
        lat = pd.concat([pd.read_csv(f).set_index("system").e2e_b1_p50_ms for f in unpinned], axis=1)
        add([(lat.max(axis=1) / lat.min(axis=1)).median()], ["unpinned passes: median max/min latency ratio"])
        flat = lat.stack()
        add(flat.to_numpy(), [f"unpinned passes: {i}" for i in flat.index])
    pins = [ANALYSIS / f"efficiency_pin{k}.csv" for k in (1, 2) if (ANALYSIS / f"efficiency_pin{k}.csv").exists()]
    if len(pins) > 1:                              # agreement of the pinned passes
        lat = pd.concat([pd.read_csv(f).set_index("system").e2e_b1_p50_ms for f in pins], axis=1)
        ratio = lat.max(axis=1) / lat.min(axis=1)
        add([100 * (ratio.median() - 1), 100 * (ratio.quantile(0.9) - 1)],
            ["pinned passes: median relative difference (%)", "pinned passes: 90th percentile relative difference (%)"])
    lead = pd.read_parquet(ANALYSIS / "lead_curves.parquet")
    systems = pd.read_csv(ANALYSIS / "systems.csv")
    for s in systems[systems.kind.isin(["P", "F"])].itertuples():
        parts = pd.DataFrame([p.split("|") for p in s.runs.split(";")], columns=["run_id", "variant"])
        g = lead.merge(parts, on=["run_id", "variant"]).groupby("lead")[["MAE", "MAPE", "bias_pct"]].mean()
        add(list(g.iloc[0]) + list(g.iloc[-1]) + [g.bias_pct.mean()],
            [f"lead_curves:{s.system}:{x}" for x in ("lead 1 MAE", "lead 1 MAPE", "lead 1 bias", "last lead MAE",
                                                       "last lead MAPE", "last lead bias", "mean bias")])
    pp = systems[systems.kind == "P"].pivot_table(index=["model", "H"], columns="arm", values="MAPE")
    rel = (100 * (pp["cal"] / pp["uni"] - 1)).dropna()          # cal against uni, % of the univariate MAPE
    add(rel.to_numpy(), [f"systems.csv: cal vs uni MAPE (%) {m} H={h}" for m, h in rel.index])
    y = pd.Series(D.y)
    year = D.local.dt.year.to_numpy()
    annual = y.groupby(year).sum() / 1e6
    add([y.mean() / 1e3, y.std() / 1e3, y.min() / 1e3, y.max() / 1e3, y.min(), y.max(), D.n, D.train_end,
         D.test_start - D.val_start, D.n - D.test_start],
        ["data:mean (GW)", "data:SD (GW)", "data:min (GW)", "data:max (GW)", "data:min (MWh)", "data:max (MWh)",
         "data:hours", "data:training hours", "data:validation hours", "data:test hours"])
    add(annual, [f"data:annual TWh {k}" for k in annual.index])
    add([len(D.origins(p, H)) for p in ("val", "test") for H in HORIZONS],
        [f"origins:{p} H={H}" for p in ("val", "test") for H in HORIZONS])

    def walk(obj, path):                                  # every number of the data report (and shares in %)
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}/{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            add([obj] + ([100 * obj] if abs(obj) < 1 else []), [f"data_report.json:{path}"] * (2 if abs(obj) < 1 else 1))
    walk(json.loads((rx.DATA_FILE.parent / "data_report.json").read_text(encoding="utf-8").replace("NaN", "null")), "")
    manifest = ROOT / "data" / "external" / "openmeteo" / "MANIFEST.csv"
    if manifest.exists():
        add([len(pd.read_csv(manifest))], ["MANIFEST.csv: downloaded weather files"])
    windows = (24, 48, 168, 336, 720, 1440, 2160)        # receptive fields of the TCN depth rule (models.py)
    add([2 ** (models.n_blocks_for(w) + 1) - 1 for w in windows], [f"models.py:receptive field w={w}" for w in windows])
    add([2 * np.sin(np.pi / 24)], ["distance of consecutive hours on the unit circle"])
    add(sorted(DESIGN_NUMBERS), ["design constant or quoted fact"] * len(DESIGN_NUMBERS))
    return bank


def text_numbers(text: str, bank: dict) -> list[dict]:
    """Every number in the text outside the title page, tables, equations, superscripts, citations and file paths,
    with the analysis values it matches at its own precision. Integers below 100 (counts, horizons, list items) are
    skipped."""
    import re
    text = re.sub(r'::: \{custom-style="Front matter"\}.*?\n:::', " ", text, flags=re.S)   # addresses, postal codes
    body = re.sub(r"\{\{\w+:\w+\}\}|\$\$.*?\$\$|\^[^^]*\^|\[@[^\]]*\]|results/analysis/\S+|manuscript/\S+|`[^`]*`",
                  " ", text, flags=re.S)
    found = []
    for m in re.finditer(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?", body):
        whole, frac = m.group(1).replace(",", ""), m.group(2) or ""
        value, d = float(whole + frac), len(frac) - 1 if frac else 0
        if not frac and value < 100:
            continue
        found.append({"number": m.group(0), "context": body[max(0, m.start() - 60):m.end() + 25].replace("\n", " "),
                      "sources": bank[min(d, 4)].get(round(value, d), [])})
    return found


def check_order(src: str) -> None:
    """Tables and figures must be numbered in the order of their first mention in the text."""
    import re
    for what, pattern, numbers in (("table", r"\{\{(?:tab|table):(\w+)\}\}", TABLE_NUMBERS),
                                   ("figure", r"\{\{fig:(\w+)\}\}", FIGURE_NUMBERS)):
        seen = list(dict.fromkeys(re.findall(pattern, src)))
        if seen != sorted(seen, key=lambda k: numbers[k]):
            sys.exit(f"{what} numbers do not follow the order of first mention: {seen}")
        if set(numbers) - set(seen):
            print(f"warning: {what}s never mentioned: {sorted(set(numbers) - set(seen))}")


def references_md(order: list, refs: dict) -> str:
    missing = [k for k in order if k not in refs]
    if missing:
        sys.exit(f"cited but not in manuscript/references.json: {missing}")
    return "\n\n# References\n\n::: {custom-style=\"Bibliography\"}\n" + "\n\n".join(
        f"[{i}] {format_reference(refs[k])}" for i, k in enumerate(order, 1)) + "\n:::\n"


def build_manuscript() -> None:
    """manuscript/manuscript.md + tables + numbered references -> manuscript/manuscript_draft_v<DRAFT>.docx; the
    claim table results/analysis/paper_numbers.json (note §14), which also audits the supplementary notes; then the
    supplementary material."""
    import re
    raw = (MANUSCRIPT / "manuscript.md").read_text(encoding="utf-8")
    left = sorted(set(re.findall(r"\{\{(?!table:|tab:|fig:|stab:|sfig:)[^}]*\}\}", raw)))
    if left:
        sys.exit(f"placeholders still to be written in manuscript.md: {left}")
    check_order(raw)
    tables = manuscript_tables()
    highlights = re.search(r"(?ms)^# Highlights\s*\n(.*?)(?=^# )", raw)
    src = raw.replace(highlights.group(0), "")          # Energy: the highlights are a separate file
    for key in TABLE_NUMBERS:
        src = src.replace("{{table:%s}}" % key, tables[key])
    text, order = cite_numbers(fill_numbers(src))
    refs = {r["key"]: r for r in json.loads((MANUSCRIPT / "references.json").read_text(encoding="utf-8"))}
    out = pandoc_docx(text + references_md(order, refs), MANUSCRIPT / f"manuscript_draft_v{DRAFT}.docx", reference_docx())
    print(f"{len(order)} references cited -> {out}")
    out = pandoc_docx("# Highlights\n\n" + highlights.group(1).strip() + "\n", MANUSCRIPT / f"highlights_v{DRAFT}.docx",
                      reference_docx())
    print(f"highlights -> {out}")

    bank = number_bank(rx.Dataset())
    numbers = text_numbers(raw, bank)
    notes = text_numbers((MANUSCRIPT / "supplement_notes.md").read_text(encoding="utf-8"), bank)
    inf = pd.read_csv(ANALYSIS / "inference.csv")
    claims = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "note": "every number quoted in manuscript/manuscript.md and manuscript/supplement_notes.md with the "
                      "analysis cells it matches; every pre-specified comparison and information contrast, significant "
                      "or not",
              "comparisons": json.loads(inf.to_json(orient="records")), "numbers": numbers,
              "supplement_numbers": notes}
    (ANALYSIS / "paper_numbers.json").write_text(json.dumps(claims, ensure_ascii=False, indent=1), encoding="utf-8")
    for what, found in (("paper", numbers), ("supplementary notes", notes)):
        unmatched = [n for n in found if not n["sources"]]
        print(f"{len(found)} numbers in the {what}, {len(unmatched)} without a matching analysis value")
        for u in unmatched:
            print(f"   {u['number']}  …{u['context']}…")
    build_supplement(refs, tables)


def supplement_tables(main_tables: dict) -> dict:
    """The supplementary tables as pandoc markdown, keyed as in SUPP_TABLES."""
    T = {n: pd.read_csv(TABLES / f"{n}.csv") for n in
         ("T12_literature", "T2_p_systems", "S_grid", "T4_comparisons", "T3_information_contrasts",
          "T3_information_systems", "S_lr_search", "T6_aggregation", "S_aggregation_monthly")}
    systems = pd.read_csv(ANALYSIS / "systems.csv")
    out = {MOVED[k]: main_tables[k] for k in MOVED}
    name = model_name

    delay = (" At H = 1 h the previous hour's load is assumed known (with a 2 h publication delay, the MAPE of "
             f"LightTCN-M rose from 0.60% to 1.03%; Table {table_label('matched')}).")

    def add(key, df, caption, align, widths):
        if key in ("grid", "comparisons", "contrasts", "f_systems", "day_types", "attention_matched", "gates"):
            caption += delay
        out[key] = md_table(df, caption, align, widths, table_label(key))

    def window(w):                                       # 0 = no fixed input window (naive, LightGBM, Vanilla)
        return "–" if str(w) in ("0", "0.0", "nan") else f"{int(float(w)):,}"

    def size(params, trees=np.nan, unit=False):           # millions of parameters, trees or coefficients
        if pd.notna(trees):
            return f"{int(trees):,} trees"
        if pd.isna(params) or params == 0:
            return "–"
        return (f"{params / 1e6:.2f}" + (" M" if unit else "")) if params >= 1e4 else f"{int(params)} coef."

    lit = T["T12_literature"]
    years = {r["key"]: int(r.get("year") or 0)
             for r in json.loads((MANUSCRIPT / "references.json").read_text(encoding="utf-8"))}
    lit = (lit.assign(_last=lit.key.eq("This work"), _year=lit.key.map(years).fillna(0))
           .sort_values(["_last", "_year"], kind="stable"))

    def dash(v):                                         # ranges and name pairs with en dashes, empty item '–'
        import re
        s = str(v).strip()
        if s == "-":
            return "–"
        s = re.sub(r"(?<=\d)-(?=\d)", "–", s).replace(" - ", " – ")
        s = re.sub(r"\bR2\b", "R²", s)
        return s.replace("Holt-Winters", "Holt–Winters").replace("Diebold-Mariano", "Diebold–Mariano")
    add("literature", pd.DataFrame([["This work" if r.key == "This work" else f"[@{r.key}]",
                                     *map(dash, (r.region, r.data, r.period, r.resolution, r.horizon, r.window,
                                                 r.model, r.results))] for r in lit.itertuples()],
                                   columns=["Ref.", "Region", "Data", "Period", "Resolution", "Horizon", "Input window",
                                            "Model", "Reported results"]),
        "Comparison of selected electricity and energy consumption forecasting studies. Errors are as reported by each "
        "study, on different data, horizons and test periods, and are not comparable across rows; '–' where a study "
        "does not use or report the item; 'Not stated' where the item could not be determined from the published "
        "record. Studies in order of publication year. CVRMSE: coefficient of variation of the RMSE; EMOS: ensemble "
        "model output statistics; ML/DL: machine learning/deep learning; MSE: mean squared error; NMBE: normalised "
        "mean bias error; NRMSE: normalised RMSE; R: correlation coefficient; R²: coefficient of determination; SMAPE: "
        "symmetric MAPE; SSE: sum of squared errors; TSO: transmission system operator; normalisations and units "
        "follow the cited studies, and model and method names and their abbreviations are those of the cited studies.", "lllllllll", [6, 10, 17, 11, 8, 9, 9, 14, 16])

    lr = T["S_lr_search"]
    add("lr", pd.DataFrame([[name(r.model), r.regime, sci(r.lr_used), fmt(r.val_MAPE), fmt(r.epochs_run, 0)
                             if pd.notna(r.epochs_run) else "–", "selected" if r.selected else ""] for r in lr.itertuples()],
                           columns=["Model", "Inputs", "Learning rate", "Validation MAPE", "Epochs", "Selected"]),
        "Learning-rate search at H = 24 h, w = 168 h, seed 42: the rate with the lower validation MAPE was used at "
        "every horizon and window, and also for the known-future arms of LightTCN-F (that of LightTCN-M) and N-HiTS. "
        "Epochs: epochs trained (the best epoch plus ten epochs of patience, or the 90-epoch limit).", "llrrrl", [24, 8, 14, 14, 10, 12])

    p = T["T2_p_systems"]
    planned = {(v["model"], v["regime"], v["H"]): v["val_MAPE"]
               for v in json.loads(PLAN_FILE.read_text(encoding="utf-8"))["validation_table"]}
    rows = []
    for r in p.itertuples():
        label = ("Ensemble of the family winners" if r.kind == "ensemble" else
                 name(r.model) + (", seed-mean forecast" if r.kind == "P seed mean" else ""))
        rows.append([r.H, label, r.regime if r.kind != "ensemble" else "–", window(r.w) if r.kind != "ensemble" else "–",
                     sci(r.lr) if pd.notna(r.lr) else "–", size(r.params), int(r.n_seeds), fmt(r.val_MAPE_seed42),
                     fmt(planned[(r.model, r.regime, r.H)]) if r.kind == "P" and (r.model, r.regime, r.H) in planned else "–",
                     fmt(r.MAPE),
                     fmt(r.MAPE_sd) if r.n_seeds > 1 and r.kind == "P" else "–", fmt(r.RMSE, 0), fmt(r.MAE, 0),
                     fmt(r.MASE), fmt(r.bias_pct, 2), fmt(100 * r.under_share, 1)])
    add("p_systems", pd.DataFrame(rows, columns=["H (h)", "System", "Inputs", "w (h)", "Learning rate", "Parameters (M)",
                                                 "Seeds", "Validation MAPE, seed 42", "Validation MAPE, plan", "Test MAPE", "SD", "RMSE", "MAE",
                                                 "MASE", "Bias (%)", "Below actual (%)"]),
        "All validation-selected past-only systems on the test year, followed by the seed-mean forecasts (forecasts "
        "averaged over seeds, then scored) and the equally weighted ensembles of the validation-selected model of "
        "each family in the analysis plan (TCN, recurrent, NeuralForecast, zero-shot foundation model and LightGBM, "
        "chosen on the column 'Validation MAPE, plan'; at H = 720 h the recurrent member is therefore BiLSTM, whereas "
        "the recurrent rival of the primary comparisons, CNN-LSTM, was selected with seed 42; members in the "
        "Supplementary Data). Test metrics of multi-seed systems are means over seeds. SD: standard deviation of the "
        "test MAPE over seeds; Bias: mean of 100·(forecast − actual)/actual; Below actual: share of forecasts below "
        "the actual load; for the ensembles, Seeds counts the member runs. At H = 1 h the previous hour's load is assumed known (with a 2 h publication "
        f"delay, the MAPE of LightTCN-M rose from 0.60% to 1.03%; Table {table_label('matched')}). The seasonal naive "
        "forecast and its weekly variant coincide at H ≥ 168 h. Validation MAPE, plan: the seed-mean validation MAPE "
        "recorded in the frozen analysis plan (one, three or five seeds, depending on the model); the selection rules "
        "are given in Section 5.3 of the paper.",
        "rllrrrcrrrrrrrrr", [10, 33, 13, 13, 16, 21, 13, 17, 17, 13, 12, 12, 12, 13, 12, 14])   # mm

    g = T["S_grid"]
    g = g.assign(_k=[display_key(m, a) for m, a in zip(g.model, g.regime)]).sort_values(["H", "_k", "w"], kind="stable")
    add("grid", pd.DataFrame([[r.H, name(r.model), r.regime, f"{int(r.w):,}", sci(r.lr_used) if pd.notna(r.lr_used) else "–",
                               fmt(r.params / 1e6, 2) if pd.notna(r.params) else "–", fmt(r.val_MAPE), fmt(r.test_MAPE),
                               "selected" if r.selected else ""] for r in g.itertuples()],
                             columns=["H (h)", "Model", "Inputs", "w (h)", "Learning rate", "Parameters (M)",
                                      "Validation MAPE", "Test MAPE", "Window"]),
        "Window grid (seed 42, selected learning rate). The window was selected on the validation MAPE only; the test "
        "MAPE of the other windows is shown for completeness.", "rllrrrrrl", [6, 20, 7, 6, 10, 10, 12, 10, 10])

    c = T["T4_comparisons"]
    rows = []
    for r in c.itertuples():
        a, b = (x.split("|") for x in (r.A, r.B))
        unit, d = (1, 3) if r.loss == "APE" else (1e-3, 0)
        rows.append([r.family, "MAPE" if r.loss == "APE" else "SE", r.H, f"{name(a[0], a[1])} − {name(b[0], b[1])}",
                     fmt(r.diff * unit, d),
                     *[f"{fmt(lo * unit, d, keep_sign=True)} to {fmt(hi * unit, d, keep_sign=True)}"
                       for lo, hi in ((r.ci_low, r.ci_high), (r.ci_low_half_b, r.ci_high_half_b),
                                      (r.ci_low_double_b, r.ci_high_double_b), (r.ci_low_seeds, r.ci_high_seeds))],
                     fmt_p(r.p_boot),
                     fmt_p(r.p_holm), fmt(r.dm_hln, 2), fmt_p(r.p_dm)])
    add("comparisons", pd.DataFrame(rows, columns=["Family", "Loss", "H (h)", "A − B", "Δ", "95% CI (b)", "95% CI (0.5b)",
                                                   "95% CI (2b)", "95% CI, seeds and b", "p", "Holm p", "DM", "p (DM)"]),
        "Pre-specified comparisons under both losses: Δ is the mean per-origin loss difference A − B (MAPE in pp; "
        "SE, squared error, in 10³ MWh²); negative values favour A. Bootstrap intervals with block lengths b = max(2H, 168) h, "
        "0.5b and 2b; p and Holm p from the bootstrap with b, Holm's adjustment within each family and loss; "
        "DM: Diebold–Mariano statistic with the Harvey–Leybourne–Newbold correction, a secondary test. The planned test "
        "averages the seeds before resampling and so conditions on the fitted networks; the exploratory column 'seeds "
        "and b' also draws each system's seeds with replacement in every resample, so that it reflects training "
        "randomness as well (not part of the plan, no p-values).", "llrlrrrrrrrrr",
        [17, 13, 9, 39, 13, 26, 26, 26, 26, 13, 13, 13, 13])       # widths in mm: no number wraps

    ic = T["T3_information_contrasts"]
    ic = ic[ic.loss == "APE"]
    rows = []
    for r in ic.itertuples():
        a, b = r.A.split("|"), r.B.split("|")
        rows.append([name(a[0]), r.H, f"{b[1]} → {a[1]}", r.type, "yes" if r.exploratory else "", fmt(r.mean_B), fmt(r.mean_A),
                     fmt(r.diff), fmt(r.rel_diff_pct, 1), f"{fmt(r.ci_low, keep_sign=True)} to {fmt(r.ci_high, keep_sign=True)}",
                     fmt_p(r.p_boot),
                     fmt_p(r.p_holm)])
    add("contrasts", pd.DataFrame(rows, columns=["Model", "H (h)", "Contrast", "Comparison type", "Exploratory",
                                                 "MAPE, first arm", "MAPE, second arm", "Δ (pp)", "Δ (%)", "95% CI", "p",
                                                 "Holm p"]),
        "Information contrasts of the known-future experiments (second arm minus first; negative values mean that the "
        "added information lowered the error); Δ (%) is relative to the first arm. Intervals and p-values from the "
        "moving-block bootstrap with b = max(2H, 168) h; Holm's adjustment within each model and horizon. Analog "
        "contrasts are exploratory (Section 5.3 of the paper); squared-error results are in the Supplementary Data. "
        "Seeds as in Table S11: LightGBM has three seeds in F1, F1A0 and F1A at H = 24 and 168 h and one otherwise; in "
        "the four contrasts that mix them (F0 → F1 and F1 → F2), the seed-42 matched differences were −0.174 and "
        "−0.022 pp at H = 24 h and −0.525 and +0.077 pp at H = 168 h.",
        "lrlllrrrrrrr", [22, 10, 28, 38, 20, 16, 16, 14, 13, 30, 14, 15])     # widths in mm: no number wraps

    f = T["T3_information_systems"]
    add("f_systems", pd.DataFrame([[name(r.model), r.H, r.arm, r.branch, int(r.n_seeds), fmt(r.MAPE),
                                    fmt(r.MAPE_sd) if r.n_seeds > 1 else "–", fmt(r.RMSE, 0), fmt(r.MAE, 0),
                                    fmt(r.bias_pct, 2)] for r in f.itertuples()],
                                  columns=["Model", "H (h)", "Arm", "Branch", "Seeds", "Test MAPE", "SD", "RMSE", "MAE",
                                           "Bias (%)"]),
        "All known-future systems on the test year. For LightTCN-F and LightGBM, F3, F3raw and F4 are stream-specific "
        "selections (best epoch or boosting round) from one training run, and F3_m24, F3_diag and F3raw_diag reuse "
        "the F3 selection; the linear benchmark and the zero-shot models use the same coefficients or pretrained "
        f"weights for every weather stream. Holiday-analog arms (F1A0, F1A) are exploratory; bias as in Table {table_label('p_systems')}; "
        "Branch: as in Table 3 of the paper.",
        "lrllrrrrrr", [16, 6, 10, 24, 6, 8, 6, 8, 8, 8])

    b = pd.read_csv(ANALYSIS / "breakdowns_systems.csv")
    keep = systems[(systems.kind == "P") | ((systems.kind == "F") & (systems.model == "LightTCN_F_medium"))].system
    b = b[b.system.isin(keep) & b.family.isin(["day type", "daily peak"])]
    wide = b.pivot_table(index=["H", "system"], columns="category", values="MAPE", sort=False)
    wide = wide.iloc[sorted(range(len(wide)), key=lambda i: (wide.index[i][0], display_key(*wide.index[i][1].split("|")[:2])))]
    peak_bias = b[b.family == "daily peak"].set_index(["H", "system"]).bias_pct
    rows = []
    for (H, system), r in wide.iterrows():
        model, arm, _ = system.split("|")
        rows.append([H, name(model, arm), *[fmt(r.get(k), 2) for k in DAY_TYPES], fmt(r.get("daily peak hour"), 2),
                     fmt(peak_bias.get((H, system)), 2)])
    add("day_types", pd.DataFrame(rows, columns=["H (h)", "System", "Religious holiday", "Eve", "National holiday", "Normal",
                                                 "Daily peak hour", "Daily peak bias (%)"]),
        "Test MAPE by type of target hour and on the daily peak hour of each day of 2025, for every past-only system "
        "and the LightTCN-F arms (means over seeds). In 2025 there are 168 religious-holiday hours, 72 eve hours and "
        "168 national-holiday hours; normal: all other hours; the groups are small and the values descriptive. The "
        "analog arms (F1A0, F1A) are exploratory; daily peak bias: mean of 100·(forecast − actual)/actual on the daily "
        "peak hours.", "rlrrrrrr",
        [6, 30, 10, 8, 10, 8, 10, 10])

    m = T["S_aggregation_monthly"]
    wide = m.pivot_table(index="system", columns="month", values="energy_APE", sort=False)
    summary = T["T6_aggregation"]
    summary = summary[summary.model != "SeasonalNaive"].set_index("system")   # at H = 168 h it is the weekly one
    actual = m.groupby("month").actual_GWh.first()
    rows = [["Actual energy (TWh)", *[fmt(actual[k] / 1000, 2) for k in sorted(wide.columns)], "–", "–"]]
    for system in summary.index:
        model, arm, _ = system.split("|")
        r = wide.loc[system]
        rows.append([name(model, arm), *[fmt(r[k], 2) for k in sorted(wide.columns)],
                     fmt(summary.loc[system, "mean_monthly_energy_APE"], 2),
                     fmt(summary.loc[system, "annual_energy_bias_pct"], 2)])
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    add("energy_months", pd.DataFrame(rows, columns=["System", *months, "Mean", "Annual bias"]),
        "Absolute percentage error (%) of the monthly energy of weekly updated H = 168 h forecasts, 2025, with the "
        "mean over the months and the annual energy bias (%). Seed-mean forecast of each system; the seasonal naive "
        "forecast coincides with its weekly variant at H = 168 h and is listed once. First row: the "
        f"actual energy of each month ({actual.sum() / 1000:.2f} TWh in the year).",
        "l" + "r" * 14, [20] + [5] * 12 + [6, 7])

    e = pd.read_csv(TABLES / "TE_efficiency.csv")
    e = e.assign(_w=e.system.str.endswith("|w48").map({True: 1, False: 0}) + 2 * e.system.str.endswith("|w24"))
    e = e.sort_values(["H", "group", "model", "regime", "_w"], kind="stable")
    rows = []
    for r in e.itertuples():
        cpu = r.model in ("LightGBM", "Vanilla")
        label = name(r.model, r.regime) + (", short window" if r.system.endswith(("|w24", "|w48")) else "")
        rows.append([r.H, label, window(r.w), size(r.params_total, getattr(r, "trees", np.nan), unit=True),
                     fmt(r.inference_weights_mib, 1) if r.inference_weights_mib >= 0.05 else "<0.1", fmt(r.cold_start_s, 1),
                     fmt(r.e2e_b1_p50_ms, 2), fmt(r.e2e_b1_p95_ms, 2),
                     fmt(r.e2e_throughput_per_s, 0) if pd.notna(r.e2e_throughput_per_s) else "–",
                     fmt(r.model_b1_p50_ms, 2) if pd.notna(r.model_b1_p50_ms) else "–",
                     "CPU" if cpu else fmt(r.peak_gpu_b1_mib, 0),
                     "CPU" if cpu else fmt(r.peak_gpu_batch_mib, 0) if pd.notna(r.peak_gpu_batch_mib) else "–",
                     fmt(r.max_abs_diff_vs_stored_mwh, 3), fmt(r.train_wall_min, 1)])
    add("efficiency", pd.DataFrame(rows, columns=["H (h)", "System", "w (h)", "Parameters", "Weights (MiB)",
                                                  "Cold start (s)", "Latency p50 (ms)", "Latency p95 (ms)",
                                                  "Throughput (1/s)", "Model-only p50 (ms)", "GPU memory, batch 1 (MiB)",
                                                  "GPU memory, batch (MiB)", "Max |Δ| to stored (MWh)",
                                                  "Campaign wall time (min)"]),
        "Efficiency profile (seed-42 run) of every learned past-only system, of LightTCN-F and the linear benchmark in "
        "their calendar arm F1 and of the short-window arms, on an idle RTX 4090 laptop GPU on mains power (Windows "
        "balanced power plan); the naive references were not profiled. Median of two passes in random order (three "
        "for the two systems whose passes differed by more than 20%), each profiled process pinned to the "
        "performance cores at high priority (Note S3); one fresh process per system; LightGBM and the linear "
        "benchmark ran on the CPU. Cold start: from the start of the process to the loaded model. Latency from the raw "
        "inputs of one origin to the forecast in MWh (batch 1, CUDA-synchronised, after warm-up); model-only latency "
        "on prepared tensors (trained networks only), timed in a separate loop, which is why for five systems its "
        "median exceeds the end-to-end median. Throughput: origins forecast per second at the campaign's prediction "
        "batch, whose peak memory is 'GPU memory, batch'; throughput and batch memory were not measured for the "
        "NeuralForecast models; TiRex-2 ran with the library's pure-PyTorch kernels, as "
        "in the campaign. Weights: size of the model files on disk. Max |Δ| to stored: largest difference between the reloaded system's forecast of the "
        "first test origin and the stored forecast. Campaign wall time: the run's wall time in the campaign "
        "(NVIDIA data-centre GPU partition, or the CPU for LightGBM and the linear benchmark), including its "
        "validation and test forecasts; for the zero-shot models it is inference only, and "
        "their pretraining cost is unknown, not zero.", "rlrrrrrrrrrrrr",
        [10, 35, 13, 22, 17, 15, 16, 16, 19, 16, 16, 16, 16, 18])          # widths in mm: no header word breaks

    tr = pd.read_csv(TABLES / "S_training.csv")
    tr = tr.assign(_k=[display_key(m, a) for m, a in zip(tr.model, tr.regime)]).sort_values(["H", "_k"], kind="stable")
    rows = [[r.H, name(r.model, r.regime), window(r.w), int(r.n_seeds),
             fmt(r.epochs_mean, 0) if pd.notna(r.epochs_mean) else "–", fmt(r.wall_min_mean, 1),
             fmt(r.peak_gpu_gib_max, 2) if pd.notna(r.peak_gpu_gib_max) else "CPU",
             fmt(r.peak_reserved_gib_max, 2) if pd.notna(r.peak_reserved_gib_max) else "CPU"] for r in tr.itertuples()]
    add("training", pd.DataFrame(rows, columns=["H (h)", "System", "w (h)", "Seeds", "Epochs (mean)",
                                                "Wall time per run (min)", "Peak allocated GPU memory (GiB)",
                                                "Peak reserved GPU memory (GiB)"]),
        "Training cost of the past-only systems in the campaign (NVIDIA data-centre GPU partitions with 20 GB of "
        "memory; LightGBM, the linear benchmark and the naive references on the CPU). Wall time per run includes the validation and test forecasts; for the zero-shot foundation models it "
        "is inference only, and their pretraining cost is unknown, not zero. Epochs (mean): mean number of epochs "
        "trained, including the ten epochs of patience after the best one, given only for the TCN and recurrent "
        "networks; wall time: mean over the runs of a system; peak memory: maximum over its runs. Allocated: memory "
        "held by tensors; reserved: memory held by the caching allocator. The whole campaign "
        "of 759 runs took 165.7 run-hours, 162.3 of them on the GPU partitions.", "rlrrrrrr",
        [7, 28, 6, 6, 9, 12, 13, 13])

    rt = table_runs()
    rows = [[r.block, *[int(getattr(r, f"runs_H{H}")) for H in HORIZONS], int(r.runs), fmt(r.run_hours, 1)]
            for r in rt.itertuples()]
    rows.append(["Total", *[int(rt[f"runs_H{H}"].sum()) for H in HORIZONS], int(rt.runs.sum()), fmt(rt.run_hours.sum(), 1)])
    add("runs", pd.DataFrame(rows, columns=["Block", *[f"H = {H} h" for H in HORIZONS], "Runs", "Run-hours"]),
        f"Runs of the campaign by experiment block and horizon ({int(rt.runs.sum())} runs, {rt.run_hours.sum():.1f} "
        f"run-hours, {rt.run_hours_gpu.sum():.1f} of them on the GPU partitions). A run trains or applies one system "
        "with one seed; the F3, F3raw and F4 streams of a network share one run, and the selection runs of the "
        "learning rate and the window are in the seed-42 block, except the two learning-rate runs of the LoRA "
        "fine-tuning, which are in its row.", "lrrrrrr", [70, 14, 14, 14, 14, 12, 14])

    am = pd.read_csv(ANALYSIS / "diagnostics_attention_matched.csv")
    add("attention_matched", pd.DataFrame([[r.H, r.w, fmt(r.val_MAPE_attn), fmt(r.val_MAPE_dual), fmt(r.test_MAPE_attn),
                                            fmt(r.test_MAPE_dual), fmt(r.test_diff_attn_minus_dual)] for r in am.itertuples()],
                                          columns=["H (h)", "w (h)", "Validation MAPE, Attn", "Validation MAPE, Dual",
                                                   "Test MAPE, Attn", "Test MAPE, Dual", "Test Δ (pp)"]),
        "Matched diagnostic of the attention read-out: DualTCN-Attn-M against DualTCN-M with the same window, learning "
        "rate (10⁻³) and seed (42), from the selection grid at H = 1, 24 and 168 h, as fixed in the implementation "
        f"note (the two pairs at H = 720 h are in Table {table_label('grid')}); Δ = Attn − Dual, negative values favour "
        "the attention read-out. Descriptive.", "rrrrrrr", [8, 8, 16, 16, 16, 16, 14])

    gt = pd.read_csv(ANALYSIS / "diagnostics_gates.csv")
    rows = []
    for r in gt.sort_values(["H", "model"]).itertuples():
        gate = pd.notna(r.gate_mean_load_path)
        rows.append([name(r.model), r.H, r.w, fmt(r.gate_mean_load_path, 2) if gate else "–",
                     fmt(100 * r.gate_saturated_share, 1) if gate else "–",
                     fmt(r.norm_load_contribution, 1) if gate else "–", fmt(r.norm_exogenous_contribution, 1) if gate else "–",
                     f"{int(r.best_epoch)} / {int(r.epochs_run)}", fmt(r.val_MAPE), fmt(r.val_bias_pct, 2)])
    add("gates", pd.DataFrame(rows, columns=["Model", "H (h)", "w (h)", "Gate on load path", "Saturated gates (%)",
                                             "Load contribution", "Calendar contribution", "Best / last epoch",
                                             "Validation MAPE", "Validation bias (%)"]),
        "Gates and learning curves of the selected seed-42 networks at H = 1, 24 and 168 h on the validation origins. "
        "Gate on load path: mean "
        "of g in Eq. (S2); saturated: share of gate units below 0.1 or above 0.9; contributions: mean norms "
        "of g ⊙ T and (1 − g) ⊙ E (or E′). The single-path networks have no gate and are listed for their learning "
        "curves. Descriptive.", "lrrrrrrrrr", [16, 6, 6, 9, 9, 9, 9, 10, 9, 9])

    at = pd.read_csv(ANALYSIS / "diagnostics_attention_summary.csv")
    wide = at.pivot_table(index="group", columns="origin_type", values="mass_over_share", sort=False)
    cols = [ATTENTION_ORIGINS[o] for o in ATTENTION_ORIGINS]
    import re
    rows = [[re.sub(r"(?<=\d)-(?=\d)", "–", grp), *[fmt(wide.loc[grp].get(o), 2) for o in cols]] for grp in wide.index]
    add("attention_summary", pd.DataFrame(rows, columns=["Input hours", *map(english_holidays, cols)]),
        "Attention mass of DualTCN-Attn-M (H = 24 h, seed 42, w = 336 h) on each group of input hours divided by the "
        "group's share of the window, at the four origins fixed by calendar (00:00 on 30 March, 6 June, 19 March and "
        "23 March 2025); 1 means that the group receives the same attention mass as under uniform weighting, '–' that the window contains no such hours. Descriptive.",
        "lrrrr", [32, 17, 17, 17, 17])
    missing = set(SUPP_TABLES) - set(out)
    if missing:
        sys.exit(f"supplementary tables without a builder: {sorted(missing)}")
    return out


def build_supplement(refs: dict, main_tables: dict | None = None) -> None:
    """manuscript/supplementary_material_v<DRAFT>.docx on landscape A4: manuscript/supplement_notes.md with its
    tables ({{supptable:KEY}}) and figures ({{suppfig:KEY}}) filled in, and its own reference list."""
    import re
    notes = (MANUSCRIPT / "supplement_notes.md").read_text(encoding="utf-8")
    placed = re.findall(r"\{\{supptable:(\w+)\}\}", notes)
    if placed != SUPP_TABLES:
        sys.exit(f"supplement_notes.md must place the tables in the order of SUPP_TABLES: {placed}")
    tables = supplement_tables(main_tables or manuscript_tables())
    for key, table in tables.items():
        notes = notes.replace("{{supptable:%s}}" % key, table)
    captions = {"leads": "Error by lead time on the test year. (a) MAPE by lead at H = 168 h, including LightTCN-F with "
                         "forecast temperature (F3), whose last 11 leads use climatology; (b) MAPE and (c) bias, the mean "
                         "of 100·(forecast − actual)/actual, by lead at H = 720 h. Seed means of the validation-selected "
                         "LightTCN-L (cal) and Chronos-2 LoRA (cal), with LightTCN-F (F3) at H = 168 h and PatchTST (uni) "
                         "at H = 720 h.",
                "attention": "Attention weights of DualTCN-Attn-M (H = 24 h, seed 42, w = 336 h) over its input window "
                             "at four origins fixed by calendar before any attention output was seen, as weight × window "
                             "length (1 = uniform weight; logarithmic scale). Shading marks the eve and weekend hours; the four windows "
                             "contain no holiday hours. "
                             "Descriptive; the weights show how this model reads the calendar path, not a mechanism."}
    note = MANUSCRIPT / "figure_note.txt"                # e.g. the disclosure the journal asks for in captions
    if note.exists():
        captions = {k: f"{v} {note.read_text(encoding='utf-8').strip()}" for k, v in captions.items()}
    for key in SUPP_FIGURES:
        notes = notes.replace("{{suppfig:%s}}" % key, f"![**Fig. S{SUPP_FIGURES.index(key) + 1}.** {captions[key]}]"
                                                         f"(results/analysis/figures/{key}.png){{width={'20cm' if key == 'attention' else '24cm'}}}")
    title = re.search(r'(?m)^title: "(.*)"$', (MANUSCRIPT / "manuscript.md").read_text(encoding="utf-8")).group(1)
    header = (f'---\ntitle: "Supplementary Material"\nsubtitle: "{title}"\n---\n\n'
              "The notes and tables below complement the paper; every table is assembled by analyze_results.py from the cited "
              "sources, design records and stored analysis outputs, and the Supplementary Data workbook (tables.xlsx; its Notes sheet maps every sheet to "
              "its table) holds every value unrounded. MAPE, bias and relative changes are in %, MAPE differences in "
              "percentage points (pp), RMSE and MAE in MWh; bias is the mean of 100·(forecast − actual)/actual, so "
              "negative values mean under-forecasting. "
              "uni: load history only; cal: load history and the calendar of the input hours; F0–F4: the known-future "
              "arms of Section 5.1 of the paper. A model is an architecture or algorithm; a system is a model with its "
              "inputs, horizon, selected window, learning rate and seeds; an arm is one information set of the "
              "known-future experiments. Seeds: number of training seeds of a system. Except in the publication-delay arm, "
              "every error at H = 1 h assumes "
              "that the load of the previous hour is known at the origin (nominal availability); with the two-hour "
              "publication delay of the real-time series, the MAPE of LightTCN-M rose from 0.60% to 1.03% "
              f"(Table S{SUPP_TABLES.index('matched') + 1}).\n\n")
    text, order = cite_numbers(fill_numbers(header + notes))
    out = pandoc_docx(text + references_md(order, refs), MANUSCRIPT / f"supplementary_material_v{DRAFT}.docx",
                      reference_docx(landscape=True))
    print(f"supplementary notes S1-S5, tables S1-S{len(SUPP_TABLES)}, figures S1-S{len(SUPP_FIGURES)}, {len(order)} references -> {out}")


def pandoc_docx(text: str, out: Path, reference: Path) -> Path:
    """Markdown text -> .docx with the given Word template (the temporary .md is always removed); table rows are
    then marked so that Word never splits a row across two pages."""
    import re
    import subprocess
    import zipfile
    filled = out.with_name(f"_{out.stem}.md")
    filled.write_text(british(text), encoding="utf-8")
    try:
        subprocess.run(["pandoc", str(filled), "-o", str(out), "--reference-doc", str(reference),
                        "--resource-path", str(ROOT)], check=True)
    finally:
        filled.unlink()
    with zipfile.ZipFile(out) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    doc = parts["word/document.xml"].decode("utf8")
    doc = re.sub(r"<w:trPr>(?!<w:cantSplit/>)", "<w:trPr><w:cantSplit/>", doc)
    doc = re.sub(r"<w:tr>(?!<w:trPr>)", "<w:tr><w:trPr><w:cantSplit/></w:trPr>", doc)

    def keep_together(m):                    # a table of at most 15 rows is never split across pages
        tbl = m.group(0)
        rows = re.findall(r"<w:tr>.*?</w:tr>", tbl, flags=re.S)
        if len(rows) > 15:                   # a long table may break, but never right after its header row
            rows = rows[:1] + [rows[-1]]
        for r in rows[:-1]:
            kept = re.sub(r"(<w:pPr>(?:<w:pStyle [^>]*/>)?)(?!<w:keepNext/>)", r"\1<w:keepNext/>", r)
            kept = re.sub(r"<w:p>(?!<w:pPr>)", "<w:p><w:pPr><w:keepNext/></w:pPr>", kept)
            tbl = tbl.replace(r, kept, 1)
        return tbl
    doc = re.sub(r"<w:tbl>.*?</w:tbl>", keep_together, doc, flags=re.S)
    from xml.dom import minidom
    minidom.parseString(doc.encode("utf8"))           # the edited XML must still be well formed
    parts["word/document.xml"] = doc.encode("utf8")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n, data in parts.items():
            z.writestr(n, data)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "losses", "inference", "seedboot", "breakdowns", "tables", "figures",
                                        "profile", "profile-merge", "profile-one", "diagnostics", "manuscript",
                                        "supplement"])
    ap.add_argument("names", nargs="*", help="figures: which figures to draw (default: all); profile-one: a run id")
    args = ap.parse_args(argv)
    if args.command == "plan":
        make_plan()
        return 0
    if args.command == "profile-one":                  # internal: one fresh process per system (run_profile)
        print(json.dumps(profile_one(args.names[0])), flush=True)
        return 0
    check_fixed_inputs()
    if args.command == "figures":
        build_figures(args.names)
        return 0
    if args.command == "profile":
        run_profile(args.names[0] if args.names else "pin1")
        return 0
    if args.command == "supplement":
        build_supplement({r["key"]: r for r in json.loads((MANUSCRIPT / "references.json").read_text(encoding="utf-8"))})
        return 0
    {"losses": build_losses, "inference": run_inference, "seedboot": run_seed_bootstrap, "breakdowns": build_breakdowns,
     "tables": build_tables, "profile-merge": merge_profiles, "diagnostics": run_diagnostics,
     "manuscript": build_manuscript}[args.command]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
