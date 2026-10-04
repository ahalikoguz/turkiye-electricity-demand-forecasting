"""Tests of the inference code in analyze_results.py on synthetic series (no campaign results needed)."""
import numpy as np
import pytest

import analyze_results as ar
import run_experiments as rx


def test_block_starts_are_fixed_and_cover_every_admissible_start():
    N, b = 500, 40
    a, c = ar.block_starts(N, b), ar.block_starts(N, b)
    assert a is c and a.shape == (ar.BOOT_B, 13)                      # k = ceil(500 / 40)
    assert a.min() == 0 and a.max() == N - b                          # starts drawn from {0, ..., N - b}
    ar._STARTS.pop((N, b))
    assert np.array_equal(ar.block_starts(N, b), a)                   # the same indices on every call


@pytest.mark.parametrize("N, b", [(500, 40), (480, 40), (97, 97), (1000, 7)])
def test_bootstrap_means_equal_concatenated_blocks_cut_to_n(N, b):
    d = np.random.default_rng(1).normal(size=N) + 0.3
    means = ar.bootstrap_means(d, b)
    starts = ar.block_starts(N, b)
    for i in (0, 1, 777, ar.BOOT_B - 1):
        resample = np.concatenate([d[s:s + b] for s in starts[i]])[:N]
        assert len(resample) == N and np.isclose(means[i], resample.mean(), rtol=0, atol=1e-12)


def test_bootstrap_p_is_small_for_a_shift_and_never_zero():
    rng = np.random.default_rng(2)
    shifted = rng.normal(size=2000) + 5.0
    p = ar.bootstrap_p(shifted, ar.bootstrap_means(shifted, 50))
    assert p == 1 / (ar.BOOT_B + 1)                                    # (1 + 0) / (B + 1), never 0
    noise = rng.normal(size=2000)
    noise -= noise.mean()                                              # d-bar = 0: every resample counts
    assert ar.bootstrap_p(noise, ar.bootstrap_means(noise, 50)) == 1.0


def test_holm_matches_the_step_down_rule():
    adj = ar.holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])                  # 4 x 0.005, 3 x 0.01, max(2 x 0.03, .), 1 x 0.04
    assert np.allclose(ar.holm([0.5, 0.9]), [1.0, 1.0])


def test_dm_hln_at_lag_zero_is_a_scaled_mean_over_its_standard_error():
    d = np.random.default_rng(3).normal(size=300) + 0.2
    stat, p = ar.dm_hln(d, 1)
    N = len(d)
    expected = d.mean() / np.sqrt(d.var() / N) * np.sqrt((N - 1) / N)   # population variance, HLN factor at H = 1
    assert np.isclose(stat, expected) and 0 < p < 1


def test_dm_hln_reports_a_non_positive_variance_as_missing():
    d = np.tile([1.0, -1.0], 50)                                       # strongly negative lag-1 autocovariance
    assert all(np.isnan(ar.dm_hln(d + 0.1, 2)))


def test_primary_loss_series_reproduce_the_protocol_metrics():
    rng = np.random.default_rng(4)
    y = rng.uniform(20_000, 50_000, size=(400, 24))
    f = (y * (1 + rng.normal(scale=0.02, size=y.shape))).astype(np.float32)
    series, scalars = ar.score(f, y, mase_scale=900.0)
    ref = rx.metrics(y, f.astype(np.float64), np.r_[y[:, 0], y[:, 0]], 400, 24)
    assert np.isclose(series["ape"].mean(), ref["MAPE"], rtol=1e-12)    # mean of per-origin means = overall mean
    assert np.isclose(np.sqrt(series["se"].mean()), ref["RMSE"], rtol=1e-12)
    for k in ("MAPE", "MAE", "RMSE", "bias_pct", "under_share"):
        key = {"bias_pct": "mean_signed_pct_error", "under_share": "under_forecast_share"}.get(k, k)
        assert np.isclose(scalars[k], ref[key], rtol=1e-12)


def test_information_contrasts_run_second_minus_first_within_one_family():
    plan = {"primary_comparisons": [], "secondary_comparisons": [],
            "information_contrasts": {"models": ["M"], "contrasts": [["F0", "F1", "x"], ["F1", "F1A0", "y"], ["F3", "F4", "z"]]}}
    out = ar.comparisons(plan, {"M|F0|H24", "M|F1|H24", "M|F3|H24", "M|F4|H24"})
    assert [(c["A"], c["B"]) for c in out] == [("M|F1|H24", "M|F0|H24"), ("M|F4|H24", "M|F3|H24")]
    assert {c["family"] for c in out} == {"information M H=24"}
    assert out[1]["type"] == "scenario-specific checkpoint of one training run" and not out[0]["exploratory"]


def test_text_numbers_match_at_their_own_precision_and_skip_markup():
    bank = {d: {} for d in range(5)}
    for v, src in ((0.0198438, "inference:diff"), (3.181, "systems:MAPE"), (87671.0, "data:hours")):
        for d in range(5):
            bank[d].setdefault(round(v, d), []).append(src)
    text = ("trailed by 0.020 pp and 3.18% (Table {{tab:T2}}), 87,671 h, 0.021 pp [@kok2022forecasting], 10^−5^, "
            "$$n = 2047$$ results/analysis/figures/p_errors.png and 12 models")
    found = {n["number"]: n["sources"] for n in ar.text_numbers(text, bank)}
    assert found == {"0.020": ["inference:diff"], "3.18": ["systems:MAPE"], "87,671": ["data:hours"], "0.021": []}


def test_tables_and_figures_must_be_numbered_in_order_of_first_mention():
    ar.check_order("Table {{tab:TD}} and {{table:TD}}, then Table {{tab:T4}}; Fig. {{fig:data}} and Fig. {{fig:architecture}}")
    with pytest.raises(SystemExit):
        ar.check_order("Table {{tab:T4}} before Table {{tab:TD}}")


def test_supplementary_tables_are_numbered_in_their_own_series():
    assert ar.table_label("T4") == "2" and ar.table_label("TM") == "S4" and ar.table_label("TF") == "S2"
    assert ar.fill_numbers("Table {{tab:T4}}, Supplementary Table {{stab:design}} and Figs. {{sfig:leads}}–{{sfig:attention}}") \
        == "Table 2, Supplementary Table S3 and Figs. S1–S2"


def test_holiday_names_are_written_in_english_with_the_turkish_name_once():
    assert ar.english_holidays("Ramazan Bayramı (Eid al-Fitr) day 2") == "Eid al-Fitr (Ramazan Bayramı), day 2"
    assert ar.english_holidays("Kurban Bayramı 2025 (eve and 4 days)") == "Eid al-Adha 2025 (eve and 4 days)"
    assert ar.english_holidays("Sunday") == "Sunday" and ar.english_holidays(1.5) == 1.5
    assert ar.english_holidays("Eid al-Fitr (Ramazan Bayramı), day 2") == "Eid al-Fitr (Ramazan Bayramı), day 2"


def test_references_follow_the_journal_style():
    journal = {"authors": ["Saez, M.", "Tilbury, D. M."], "title": "A framework.", "container": "Applied Energy",
               "year": 2022, "volume": "62", "pages_or_article": "925-938", "doi": "10.1/x", "type": "journal"}
    assert ar.format_reference(journal) == "Saez M, Tilbury DM. A framework. Appl Energy 2022;62:925–38. https://doi.org/10.1/x"
    web = {"authors": ["EPİAŞ"], "title": "Platform", "container": "Online data platform", "year": 2026,
           "url": "https://e.x", "accessed": "1 October 2026", "type": "dataset"}
    assert ar.format_reference(web) == "[dataset] EPİAŞ. Platform. Online data platform; 2026. https://e.x [accessed 1 October 2026]"
    talk = {"authors": ["Hong, T."], "title": "A benchmark", "container": "2011 IEEE PES General Meeting", "year": 2011,
            "pages_or_article": "1-6", "type": "conference"}
    assert ar.format_reference(talk) == "Hong T. A benchmark. In: 2011 IEEE PES General Meeting; 2011, p. 1–6."


def test_sci_writes_powers_of_ten_for_word():
    assert ar.sci(1e-5) == "1 × 10⁻⁵" and ar.sci(0.0003) == "3 × 10⁻⁴" and ar.sci(1e-3) == "1 × 10⁻³"
