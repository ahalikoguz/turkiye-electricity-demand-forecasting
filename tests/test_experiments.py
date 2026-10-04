"""Protocol tests for run_experiments.py (need the built dataset: `python prepare_data.py build`)."""
import numpy as np
import pytest

import run_experiments as rx

if not rx.DATA_FILE.exists():
    pytest.skip("run `python prepare_data.py build` first", allow_module_level=True)

D = rx.Dataset()


@pytest.mark.parametrize("H", [1, 24, 168, 720])
def test_target_sets_do_not_depend_on_the_window(H):
    # origins are defined by the targets only, so every window length and model share them
    test, val = D.origins("test", H), D.origins("val", H)
    assert len(test) == 8760 - H + 1 and len(val) == 8784 - H + 1
    assert D.local[test[0]].strftime("%Y-%m-%d %H:%M") == "2025-01-01 00:00"
    assert D.local[val[-1] + H - 1].strftime("%Y-%m-%d %H:%M") == "2024-12-31 23:00"


def test_oracle_forecast_scores_zero():
    o = D.origins("test", 24)
    m = rx.metrics(D.targets(o, 24), D.targets(o, 24), D.y, D.train_end, 24)
    assert m["MAE"] == 0 and m["MAPE"] == 0


def test_windows_contain_only_the_past():
    past = D.past_matrix("cal")
    ds = rx.make_window_dataset(past, D.y_scaled, np.array([D.test_start]), 168, 24)
    x, y = ds[0]
    assert np.allclose(x.numpy(), past[D.test_start - 168:D.test_start])        # inputs end at t-1
    assert np.allclose(y.numpy(), D.y_scaled[D.test_start:D.test_start + 24])   # targets start at t


def test_future_values_do_not_change_the_forecast():
    # changing every value from the origin onwards must leave the model input unchanged
    import torch
    from models import create_model
    past = D.past_matrix("cal").copy()
    t = D.test_start + 500
    model = create_model("LightTCN_medium", past.shape[1], 24, 168).eval()
    a = rx.predict_torch(model, past, np.array([t]), 168, 24, torch.device("cpu"))[0]
    past[t:] = 99.0
    b = rx.predict_torch(model, past, np.array([t]), 168, 24, torch.device("cpu"))[0]
    assert np.allclose(a, b)


def test_scaling_uses_training_years_only():
    assert np.isclose(D.y_mean, D.y[:D.train_end].mean())
    assert D.local[D.train_end - 1].year == 2023 and D.local[D.train_end].year == 2024


def test_seasonal_naive_uses_only_observed_values():
    o = D.origins("test", 168)
    p = rx.seasonal_naive(D.y, o, 168, 24)
    assert np.allclose(p[:, 0], D.y[o - 24]) and np.allclose(p[:, 30], D.y[o - 24 + 6])
    assert (o[:, None] - 24 + np.arange(168)[None, :] % 24 < o[:, None]).all()      # never reads t or later


def test_run_id_follows_the_whole_contract():
    # the identity changes with anything that changes the numbers, and only with that
    cfg = rx.load_config()
    h = {"load": "a", "weather": "w", "models.py": "m"}
    s = {"model": "LightTCN_medium", "regime": "cal", "H": 24, "w": 168, "seed": 42}

    def rid(spec, cf=cfg, hh=h):
        return rx.run_id(rx.make_contract(spec, cf, hh))

    base = rid(s)
    assert base != rid(dict(s, seed=123)) and base != rid(s, hh=dict(h, load="b"))
    assert base != rid(s, cf=dict(cfg, training=dict(cfg["training"], batch_size=128)))
    assert base == rid(s, hh=dict(h, weather="other"))                              # P regime: no weather
    f1 = dict(s, model="LightTCN_F_medium", regime="F1")
    assert rid(f1) != rid(f1, hh=dict(h, weather="other"))
    lgb = {"model": "LightGBM", "regime": "cal", "H": 24, "w": 0, "seed": 42}
    assert rid(lgb) == rid(lgb, cf=dict(cfg, training=dict(cfg["training"], batch_size=128)))


# ---------------------------------------------------------------------------- known-future regimes (F)
def _with_future_load_changed(t):
    """Copy of the dataset in which every load from the origin t on is replaced (weather untouched)."""
    import copy
    D2 = copy.copy(D)
    D2.y, D2.y_scaled = D.y.copy(), D.y_scaled.copy()
    D2.y[t:] *= 3.0
    D2.y_scaled[t:] = 99.0
    D2._analog = None
    return D2


def test_f_inputs_never_read_future_load():
    # the holiday analog and every other F input must be identical when the future load changes
    t = D.test_start + 2000
    D2 = _with_future_load_changed(t)
    for reg in rx.F_REGIMES:
        H = 720 if reg in ("F0", "F1", "F1A") else 168
        a, b = rx.FInputs(D, reg, 168, H), rx.FInputs(D2, reg, 168, H)
        for v in ["train"] + rx.F_STREAMS[reg] + list(rx.F_TEST_ONLY.get(reg, {})):
            (pa, fa), (pb, fb) = a.arrays(np.array([t]), v), b.arrays(np.array([t]), v)
            assert np.allclose(pa, pb) and np.allclose(fa, fb), (reg, v)


def test_realistic_temperature_uses_only_published_information():
    # ERA5 of the last 6 days and of the future is not published at t: changing it (weighted features and
    # every province, then recomputing the running bias correction) must leave the realistic inputs unchanged
    import copy
    W, t = D.weather, D.test_start + 3000
    W2 = copy.copy(W)
    W2.era5, W2.era5_prov = W.era5.copy(), W.era5_prov.copy()
    W2.era5[t - rx.ERA5_LATENCY_H + 1:] = 999.0
    W2.era5_prov[t - rx.ERA5_LATENCY_H + 1:] = 999.0
    W2._build_mos()
    for src in ("fc_mos", "fc_raw"):
        assert np.allclose(W.past_temperature(np.array([t]), 720, src), W2.past_temperature(np.array([t]), 720, src))
        assert np.allclose(W.future_temperature(np.array([t]), np.arange(168), src),
                           W2.future_temperature(np.array([t]), np.arange(168), src))


def test_corrected_degree_hours_stay_physically_consistent():
    # degree-hours come from corrected province temperatures: weighted HDH >= 18 - T and CDH >= T - 22
    o = D.origins("test", 168)[::7]
    for src in ("fc_mos", "fc_raw"):
        f = D.weather.future_temperature(o, np.arange(168), src)
        T, hdh, cdh = f[..., 0], f[..., 1], f[..., 2]
        assert (hdh >= np.maximum(18 - T, 0) - 1e-4).all() and (cdh >= np.maximum(T - 22, 0) - 1e-4).all()


def test_future_temperature_uses_forecasts_issued_before_the_origin():
    W, t, L = D.weather, D.test_start + 500, np.arange(168)
    N = np.ceil((L + rx.ISSUE_MARGIN_H) / 24).astype(int)
    assert (24 * N - L >= rx.ISSUE_MARGIN_H).all()          # issued >= 12 h before t (~24*N h before tau)
    f = W.future_temperature(np.array([t]), L, "fc_raw")[0]
    for lead in L:
        want = W._features(W.fc_prov[N[lead], t + lead - W.f0]) if N[lead] <= rx.MAX_LEAD_DAY else W.clim[t + lead]
        assert np.allclose(f[lead], want)


def test_f_validation_origins_are_common_and_after_the_archive_start():
    s = D.weather.f_val_start
    assert D.local[s].year == 2024 and D.local[s].month >= 2
    for H in (1, 24, 168, 720):
        assert D.f_origins("val", H)[0] == s
        assert np.array_equal(D.f_origins("test", H), D.origins("test", H))


def test_f0_is_f1_with_the_future_masked():
    t = np.array([D.test_start])
    f0 = rx.FInputs(D, "F0", 168, 24).arrays(t, "main")[1]
    f1 = rx.FInputs(D, "F1", 168, 24).arrays(t, "main")[1]
    assert f0.shape == f1.shape and not f0.any()
    assert np.allclose(f1[0], D.calendar[D.test_start:D.test_start + 24])


def test_holiday_analog_links_the_same_bayram_day():
    A, lab = D.analog, list(D.labels)
    assert D.labels[A.analog_index[lab.index("2025-03-30 14:00")]] == "2024-04-10 14:00"   # Ramazan day 1
    assert D.labels[A.analog_index[lab.index("2025-06-06 09:00")]] == "2024-06-16 09:00"   # Kurban day 1
    assert D.labels[A.analog_index[lab.index("2025-10-29 20:00")]] == "2024-10-29 20:00"   # Republic Day
    k = lab.index("2025-05-07 12:00")                                                       # ordinary day
    assert A.analog_index[k] == -1 and not A.available[k]


def test_lightgbm_features_use_only_the_past():
    t = D.test_start + 1234
    D2 = _with_future_load_changed(t)
    assert np.allclose(rx._lgb_base(D, np.array([t])), rx._lgb_base(D2, np.array([t])))


def test_vanilla_is_fitted_on_training_years_only():
    import copy
    D2 = copy.copy(D)
    D2.y = D.y.copy()
    D2.y[D.train_end:] = 0.0
    spec = {"model": "Vanilla", "regime": "F1", "H": 24, "w": 0, "seed": 42}
    a = rx.run_vanilla(spec, D, None, None, print)["preds"]["test"][1]
    b = rx.run_vanilla(spec, D2, None, None, print)["preds"]["test"][1]
    assert np.allclose(a, b)


def test_f1a0_masks_the_analog_value_and_keeps_the_flag():
    t = np.array([D.test_start + 2000])
    a = rx.FInputs(D, "F1A", 168, 720).arrays(t, "main")[1]
    b = rx.FInputs(D, "F1A0", 168, 720).arrays(t, "main")[1]
    assert a.shape == b.shape and a[..., 7].any()
    assert not b[..., 6].any() and np.array_equal(a[..., 7], b[..., 7]) and np.array_equal(a[..., :6], b[..., :6])


def test_gradient_accumulation_equals_the_full_batch_gradient():
    # two size-weighted half-batches give the full-batch gradient (dropout off; float tolerance)
    import torch
    from models import LightTCN
    torch.manual_seed(0)
    m = LightTCN(7, 24, 168, base_channels=16, kernel_size=3, dropout=0.0)
    x, y = torch.randn(10, 168, 7), torch.randn(10, 24)
    loss_fn = torch.nn.HuberLoss()
    m.zero_grad()
    loss_fn(m(x), y).backward()
    full = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    for part in np.array_split(np.arange(10), 2):
        sl = slice(int(part[0]), int(part[-1]) + 1)
        (loss_fn(m(x[sl]), y[sl]) * (len(part) / 10)).backward()
    assert all(torch.allclose(a, p.grad, atol=1e-6) for a, p in zip(full, m.parameters()))


def test_saved_models_reproduce_their_forecasts(tmp_path):
    # LightGBM boosters and Vanilla coefficients written by a run give back the stored forecasts
    import gzip
    import json
    import lightgbm as lgb
    spec = {"model": "LightGBM", "regime": "cal", "H": 24, "w": 0, "seed": 42, "demo": True}
    res = rx.run_lightgbm(spec, D, None, tmp_path, print)
    with gzip.open(tmp_path / "lightgbm_models.json.gz", "rt") as f:
        saved = json.load(f)
    o, p = res["preds"]["val"]
    booster = lgb.Booster(model_str=saved["lead_models"][0])
    assert np.allclose(booster.predict(rx._lgb_base(D, o), num_iteration=saved["best_rounds"]["main"][0]), p[:, 0])
    spec = {"model": "Vanilla", "regime": "F34", "H": 24, "w": 0, "seed": 42}
    res = rx.run_vanilla(spec, D, None, tmp_path, print)
    beta = np.load(tmp_path / "vanilla_beta.npy")
    o, p = res["preds"]["test@F3"]
    assert np.allclose(rx.vanilla_forecast(D, beta, "F34", o, 24, "F3"), p)


def test_selection_refuses_incomplete_candidates(tmp_path, monkeypatch):
    # a missing declared run must stop the selection instead of silently shrinking the candidate set
    cfg = rx.load_config()
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(rx, "SELECTED_FILE", tmp_path / "selected.yaml")
    with pytest.raises(SystemExit):
        rx.select_learning_rates(cfg, tmp_path)
    assert not (tmp_path / "selected.yaml").exists()


def _fake_run(folder, rid, text="old"):
    """A minimal run folder in the published format (core files, manifest with sha256 and sizes, DONE)."""
    import json
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "contract.json").write_text(json.dumps({"run_id": rid}))
    (folder / "metrics.json").write_text(json.dumps({"val_metrics": {"MAPE": 9.0}, "note": text}))
    for f in ("val_predictions.npz", "test_sealed.npz"):
        np.savez(folder / f, x=np.arange(3))
    (folder / "manifest.json").write_text(json.dumps(
        {f: {"sha256": rx._hash_path(folder / f), "bytes": rx._path_bytes(folder / f)} for f in rx.CORE_FILES}))
    (folder / "DONE").write_text("t")
    return folder


def test_failed_rerun_keeps_the_verified_result(tmp_path):
    final = _fake_run(tmp_path / "runs" / "X", "X", "old")
    bad = tmp_path / "attempts" / "X.1"
    bad.mkdir(parents=True)
    (bad / "metrics.json").write_text("new")                  # crashed before manifest and DONE
    assert not rx.finish_attempt(bad, final, tmp_path)
    assert '"old"' in (final / "metrics.json").read_text() and rx.verify_run(final)
    good = _fake_run(tmp_path / "attempts" / "X.2", "X", "new")
    assert rx.finish_attempt(good, final, tmp_path)
    assert '"new"' in (final / "metrics.json").read_text() and len(list((tmp_path / "_replaced").iterdir())) == 1


def test_first_run_publishes_into_a_clean_results_tree(tmp_path):
    # no results/runs folder exists yet on a new machine: publication must create it
    attempt = _fake_run(tmp_path / "attempts" / "Y.1", "Y")
    final = tmp_path / "runs" / "Y"
    assert not final.parent.exists()
    assert rx.finish_attempt(attempt, final, tmp_path) and rx.verify_run(final)


def test_verification_rejects_altered_missing_or_empty_records(tmp_path):
    import json
    run = _fake_run(tmp_path / "runs" / "Z", "Z")
    assert rx.verify_run(run) and rx.verify_run(run, full=True)
    (run / "metrics.json").write_text(json.dumps({"val_metrics": {"MAPE": 0.1}}))   # an edited result
    assert not rx.verify_run(run)
    run = _fake_run(tmp_path / "runs" / "Z2", "Z2")
    (run / "test_sealed.npz").unlink()                                           # a missing file
    assert not rx.verify_run(run)
    run = _fake_run(tmp_path / "runs" / "Z3", "Z3")
    (run / "manifest.json").write_text("{}")                                     # an empty manifest
    assert not rx.verify_run(run)
    run = _fake_run(tmp_path / "runs" / "Z4", "other-id")                         # contract of another run
    assert not rx.verify_run(run)
    with pytest.raises(FileNotFoundError):
        rx._hash_path(tmp_path / "does-not-exist")


def test_folder_hash_does_not_depend_on_the_platform(tmp_path):
    # run folders travel between Linux and Windows: "B" sorts before "a" on both (case-sensitive components)
    import hashlib
    (tmp_path / "sub").mkdir()
    for name in ("a.txt", "B.txt", "sub/C.txt"):
        (tmp_path / name).write_text(name)
    h = hashlib.sha256()
    for name in ("B.txt", "a.txt", "sub/C.txt"):
        h.update(name.encode())
        h.update(rx.sha256_file(tmp_path / name).encode())
    assert rx._hash_path(tmp_path) == h.hexdigest()


def test_code_fingerprint_ignores_comments_but_not_numbers(tmp_path):
    import shutil
    for f in ("run_experiments.py", "models.py", "prepare_data.py"):
        shutil.copy(rx.ROOT / f, tmp_path / f)
    fams = ("torch", "lightgbm", "nf", "vanilla")
    base = {fam: rx.code_fingerprint(fam, tmp_path) for fam in fams}
    src = (tmp_path / "run_experiments.py").read_text(encoding="utf-8")
    edited = src.replace("def run_lightgbm(", "# a comment\ndef run_lightgbm(", 1).replace(
        "Temperature inputs of the F regimes, aligned to the load index.", "Temperature inputs (edited docstring).", 1)
    (tmp_path / "run_experiments.py").write_text(edited, encoding="utf-8")
    rx._FINGERPRINTS.clear()
    assert {fam: rx.code_fingerprint(fam, tmp_path) for fam in fams} == base      # comment and docstring only
    (tmp_path / "run_experiments.py").write_text(src.replace("n_estimators=2000", "n_estimators=123", 1), encoding="utf-8")
    rx._FINGERPRINTS.clear()
    assert rx.code_fingerprint("lightgbm", tmp_path) != base["lightgbm"]          # a numerical change ...
    assert rx.code_fingerprint("torch", tmp_path) == base["torch"]                # ... affects only its family
    for old, new in (("MOS_DAYS = 30", "MOS_DAYS = 29"),                          # shared weather code: all
                     ("HEATING_BASE_C, COOLING_BASE_C = 18.0, 22.0", "HEATING_BASE_C, COOLING_BASE_C = 17.0, 22.0")):
        assert src.count(old) == 1
        (tmp_path / "run_experiments.py").write_text(src.replace(old, new), encoding="utf-8")
        rx._FINGERPRINTS.clear()
        assert all(rx.code_fingerprint(fam, tmp_path) != base[fam] for fam in fams), old
    rx._FINGERPRINTS.clear()


def test_hidden_gpus_stay_hidden(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert rx._devices("auto") == ["cpu"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert rx._devices("auto") == ["0", "1", "cpu"]


def test_available_cpus_never_exceeds_the_machine():
    import os
    n = rx.available_cpus()
    assert isinstance(n, int) and 1 <= n <= (os.cpu_count() or n)


def test_lock_outlives_its_launcher_while_the_child_runs(tmp_path):
    # a killed launcher leaves its child training: the run stays locked until the child is gone too
    import json
    import os
    import platform
    import subprocess
    import sys
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    lock = tmp_path / "R.lock"
    try:
        lock.write_text(json.dumps({"pid": dead.pid, "child": child.pid, "host": platform.node()}))
        assert not rx._lock(tmp_path, "R")
    finally:
        child.kill()
        child.wait()
    assert rx._lock(tmp_path, "R")                                    # both gone: stale, taken over once
    assert json.loads(lock.read_text())["pid"] == os.getpid()
    assert not rx._lock(tmp_path, "R")                                # held by a live launcher
    lock.write_text(json.dumps({"pid": dead.pid, "child": None, "host": "another-machine"}))
    assert not rx._lock(tmp_path, "R")                                # other hosts' locks are never taken over


def test_nf_callback_restores_the_best_check_and_lowers_the_rate():
    # best validation loss at the 2nd of 8 checks: the weights of that check come back at the end of fit,
    # and five checks without improvement (patience 4) lower the learning rate once by the factor 0.7
    import torch
    from types import SimpleNamespace
    cb = rx._nf_callback_class()(0.7, 4, 1e-6)
    module = torch.nn.Linear(1, 1, bias=False)
    opt = torch.optim.SGD(module.parameters(), lr=1.0)
    trainer = None
    for i, v in enumerate([3.0, 1.0, 2.0, 2.5, 2.6, 2.7, 2.8, 2.9]):
        with torch.no_grad():
            module.weight.fill_(float(i))
        trainer = SimpleNamespace(sanity_checking=False, callback_metrics={"ptl/val_loss": torch.tensor(v)},
                                  optimizers=[opt], global_step=10 * (i + 1))
        cb.on_validation_end(trainer, module)
    cb.on_fit_end(trainer, module)
    assert module.weight.item() == 1.0 and cb.best_step == 20 and cb.checks == 8
    assert abs(opt.param_groups[0]["lr"] - 0.7) < 1e-12


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="foundation models need the GPU")
@pytest.mark.parametrize("name", ["Chronos2", "TimesFM3", "TiRex2"])
def test_foundation_forecast_does_not_depend_on_batch_mates(name):
    # later origins contain the targets of earlier ones: forecasting an origin alone or inside a batch
    # (in any order) must give the same numbers, so no origin can learn from another's context
    predict, _ = rx._tsfm_load(name)
    o = D.origins("test", 24)[[500, 524, 548, 800]]
    alone = rx._tsfm_forecast(predict, D, "cal", "main", o[:1], 168, 24)
    together = rx._tsfm_forecast(predict, D, "cal", "main", o[::-1], 168, 24)
    assert np.allclose(alone[0], together[-1], rtol=1e-4, atol=1.0)


def test_publication_gap_hides_the_latest_hours():
    # with gap = 2 (EPIAS publishes an hour ~2 h after it ends) no input may come from t-2 or t-1
    t, gap = D.test_start + 100, 2
    past = D.past_matrix("cal")
    x, _ = rx.make_window_dataset(past, D.y_scaled, np.array([t]), 168, 24, gap)[0]
    assert np.allclose(x.numpy(), past[t - gap - 168:t - gap])
    o = D.origins("test", 168)
    for m in (24, 168):
        idx = o[:, None] + np.arange(168)[None, :] - m * np.ceil((np.arange(168)[None, :] + 1 + gap) / m).astype(int)
        assert (idx <= o[:, None] - 1 - gap).all()
        assert np.allclose(rx.seasonal_naive(D.y, o, 168, m, gap), D.y[idx])
    assert np.allclose(rx.seasonal_naive(D.y, o, 168, 24, 0), D.y[o[:, None] - 24 + np.arange(168)[None, :] % 24])


# ---------------------------------------------------------------- saved models reload (need the GPU)
GPU = __import__("torch").cuda.is_available()


@pytest.mark.skipif(not GPU, reason="needs the GPU")
def test_saved_torch_model_reproduces_its_forecasts(tmp_path):
    # save -> rebuild from the run folder -> the same forecasts
    import json
    import torch
    spec = {"model": "LightTCN_medium", "regime": "cal", "H": 24, "w": 168, "seed": 42, "demo": True}
    tc = dict(rx.load_config()["training"], max_epochs=1, early_stopping_patience=1)
    res = rx.run_torch(spec, D, tc, tmp_path, lambda m: None)
    (tmp_path / "contract.json").write_text(json.dumps({"spec": spec}))
    model = rx.load_torch_model(tmp_path, D).to("cuda")
    o, p = res["preds"]["val"]
    again = rx.predict_torch(model, D.past_matrix("cal"), o[:64], 168, 24, torch.device("cuda"))[0] * D.y_std + D.y_mean
    assert np.allclose(again, p[:64], atol=0.5)


@pytest.mark.skipif(not GPU, reason="needs the GPU")
def test_saved_neuralforecast_model_reproduces_its_forecast(tmp_path):
    from neuralforecast import NeuralForecast
    spec = {"model": "NF_NHITS", "regime": "cal", "H": 168, "w": 168, "seed": 42, "demo": True}
    res = rx.run_neuralforecast(spec, D, dict(rx.load_config()["training"]), tmp_path, lambda m: None)
    nf = NeuralForecast.load(str(tmp_path / "nf_model"))
    df, _ = rx.nf_dataframe(D, "cal")
    agg = D.aggregation_origin()
    fc = nf.predict(df=df.iloc[:agg])
    fc = (fc.reset_index() if "unique_id" not in fc.columns else fc).sort_values("ds")
    col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
    again = fc[col].to_numpy(np.float64) * D.y_std + D.y_mean
    assert np.allclose(again, res["preds"]["aggregation_extra"][1][0], rtol=1e-4, atol=1.0)


@pytest.mark.skipif(not GPU, reason="needs the GPU")
def test_saved_chronos_lora_reproduces_its_forecasts(tmp_path):
    from chronos import Chronos2Pipeline
    spec = {"model": "Chronos2_LoRA", "regime": "cal", "H": 24, "w": 168, "seed": 42, "demo": True}
    res = rx.run_chronos_lora(spec, D, None, tmp_path, lambda m: None)
    pipe = Chronos2Pipeline.from_pretrained(str(tmp_path / "chronos_lora" / "finetuned-ckpt"), device_map="cuda")
    o, p = res["preds"]["val"]
    again = rx._tsfm_forecast(rx._chronos_predictor(pipe), D, "cal", "main", o[:8], 168, 24)
    assert np.allclose(again, p[:8], rtol=1e-4, atol=1.0)
