"""Data-pipeline tests. Run with:  python -m pytest -q tests

The first group needs no files (parser behaviour on synthetic cells). The second group checks
the built dataset (run `python prepare_data.py build` first); it is skipped if the file is absent.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prepare_data import DataError, holidays, parse_local_labels

ROOT = Path(__file__).resolve().parents[1]
BUILT = ROOT / "data" / "processed" / "tr_load_hourly.csv"


# ---------------------------------------------------------------- parser (no files needed)
def _raw(rows):
    return pd.DataFrame(rows, columns=["Date", "Hour", "Consumption (MWh)"]).assign(raw_row=range(2, 2 + len(rows)))


def test_excel_datetime_is_not_read_day_first():
    # An Excel date 2020-02-01 (1 February) must never be read as 2 January (day first).
    labels = parse_local_labels(_raw([[datetime(2020, 2, 1, 5), "05:00", 1.0]]))
    assert labels.iloc[0] == pd.Timestamp("2020-02-01 05:00")


def test_text_date_is_day_month_year():
    labels = parse_local_labels(_raw([["01.02.2019", "13:00", 1.0], ["12.01.2020", "00:00", 1.0]]))
    assert list(labels) == [pd.Timestamp("2019-02-01 13:00"), pd.Timestamp("2020-01-12 00:00")]


def test_hour_disagreeing_with_datetime_cell_is_rejected():
    with pytest.raises(DataError):
        parse_local_labels(_raw([[datetime(2020, 2, 1, 6), "05:00", 1.0]]))


def test_malformed_text_date_is_rejected():
    with pytest.raises(DataError):
        parse_local_labels(_raw([["2020-02-01", "05:00", 1.0]]))


def test_overlapping_holidays_are_merged():
    full, half = holidays([2023])
    assert "Ramazan" in full[pd.Timestamp("2023-04-23").date()][0]       # 23 April 2023 = bayram day 3
    assert pd.Timestamp("2023-10-28").date() in half                       # Republic Day eve


# ---------------------------------------------------------------- built dataset
@pytest.fixture(scope="module")
def df():
    if not BUILT.exists():
        pytest.skip("run `python prepare_data.py build` first")
    return pd.read_csv(BUILT, parse_dates=["datetime_local"])


def test_row_count_and_utc_continuity(df):
    utc = pd.to_datetime(df["datetime_utc"], utc=True)
    assert len(df) == 87_671
    assert (utc.diff().dropna() == pd.Timedelta(1, "h")).all()


def test_rows_per_year(df):
    expected = {y: (8784 if y in (2016, 2020, 2024) else 8760) for y in range(2016, 2026)}
    expected[2016] -= 1                                                    # 27 March 2016 had 23 hours
    assert df.groupby("year").size().to_dict() == expected


def test_single_imputed_hour(df):
    imp = df.loc[df.is_imputed == 1, "datetime_local"]
    assert list(imp) == [pd.Timestamp("2016-03-27 02:00")]


def test_calendar_columns_recompute(df):
    t = df["datetime_local"]
    assert (df["hour"] == t.dt.hour).all() and (df["day_of_week"] == t.dt.dayofweek).all()
    assert (df["is_weekend"] == (t.dt.dayofweek >= 5).astype(int)).all()
    assert np.allclose(df["hour_sin"], np.sin(2 * np.pi * t.dt.hour / 24))


def test_swap_regression_known_days(df):
    # Kurban Bayramı Sunday 8 June 2025 must look like a holiday; Wednesday 6 August 2025 like a summer workday.
    daily = df.set_index("datetime_local")["consumption_mwh"].resample("D").mean()
    assert daily["2025-06-08"] < 32_000 and daily["2025-08-06"] > 45_000
    assert df.loc[df.datetime_local.dt.date == pd.Timestamp("2025-06-08").date(), "is_holiday"].eq(1).all()


def test_record_day_matches_public_report(df):
    total = df.loc[df.datetime_local.dt.date == pd.Timestamp("2024-07-09").date(), "consumption_mwh"].sum()
    assert abs(total / 1_147_848 - 1) < 0.005                             # Anadolu Agency, 10 July 2024
