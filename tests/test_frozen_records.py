"""The archived run records are those of the freeze.

configs/frozen.json holds the SHA-256 of every run's manifest at the freeze. In the released records the device name
in metrics.json was generalised after the freeze, and the saved NeuralForecast and Chronos-2 LoRA models store
paths relative to the project instead of the training machine's. Those runs keep their manifest at the freeze as
manifest_at_freeze.json, and their current manifest differs from it only in the entries of metrics.json and of the
saved model, so contracts, forecasts and all other files are those of the freeze. Needs the run folders (archived
on Zenodo); skipped without them.
"""
import json

import pytest

import run_experiments as rx

RUNS = rx.ROOT / "results" / "runs"
if not RUNS.is_dir() or not any(RUNS.iterdir()):
    pytest.skip("the run folders are not present (archived on Zenodo)", allow_module_level=True)

EDITED = {"metrics.json", "nf_model", "chronos_lora"}


def test_every_run_keeps_the_manifest_of_the_freeze():
    frozen = json.loads((rx.ROOT / "configs" / "frozen.json").read_text())["runs"]
    for rid, frozen_sha in frozen.items():
        f = RUNS / rid
        at_freeze = f / "manifest_at_freeze.json"
        if not at_freeze.exists():
            at_freeze = f / "manifest.json"
        assert rx.sha256_file(at_freeze) == frozen_sha, rid
        old, new = json.loads(at_freeze.read_text()), json.loads((f / "manifest.json").read_text())
        assert old.keys() == new.keys(), rid
        assert all(old[k] == new[k] for k in old if k not in EDITED), rid
        assert rx.verify_run(f), rid
