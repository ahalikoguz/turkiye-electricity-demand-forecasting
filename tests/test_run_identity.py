"""The code, configuration and data of this folder reproduce the identifier of every run of the frozen campaign.

A run identifier hashes the run's full contract: specification, training settings, data hashes, protocol
version, the fingerprint of the code that determines the family's numbers (a normalised syntax tree without
comments or docstrings) and the versions of the family's packages. Matching all identifiers recorded in
configs/frozen.json therefore shows that the computations are those of the campaign. The package versions must
be those of requirements-lock.txt.
"""
import json

import run_experiments as rx


def test_every_frozen_run_identifier_is_reproduced():
    cfg = rx.load_config()
    hashes = rx.data_hashes()
    specs = rx.expand_experiment(cfg, ",".join(cfg["campaign"]))
    computed = {rx.run_id(rx.make_contract(s, cfg, hashes)) for s in specs}
    frozen = set(json.loads((rx.ROOT / "configs" / "frozen.json").read_text())["runs"])
    assert computed == frozen, (sorted(computed - frozen)[:5], sorted(frozen - computed)[:5])
