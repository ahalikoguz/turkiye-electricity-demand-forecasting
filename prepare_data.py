"""Stage 0-2 of the pipeline: raw data -> validated hourly load series with calendar and weather columns.

Usage (run from the project root):
  python prepare_data.py geocode    resolve the 81 province capitals to coordinates
  python prepare_data.py weather    download raw Open-Meteo responses (resumable; safe to re-run)
  python prepare_data.py build      raw EPİAŞ Excel (+ weather) -> data/processed/

The raw file is never modified. Every change to the data is listed in corrections.csv and every
check result is written to data_report.json. If a behavioural plausibility check raises a warning,
the report is marked "approved": false until a person has looked at it and re-runs
`build --accept-warnings`; later stages refuse unapproved data.

Sections: 1 calendar, 2 load series, 3 weather, 4 build and command line.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ==============================================================================================
# 1. Calendar: official public holidays of Türkiye
# ==============================================================================================
# Official public holidays of Türkiye, 2016-2025.
#
# Sources: Law No. 2429 on National Holidays and General Holidays (as amended; 15 July was
# added in 2017) and the religious-holiday dates announced by the Presidency of Religious
# Affairs (Diyanet).
#
# Why a hand-written table: religious holidays follow the lunar Hijri calendar and are fixed
# by official announcement, so computing them algorithmically can be off by one day. The
# dates below are also cross-checked against the load itself: every bayram period shows a
# drop of more than 15 % against neighbouring weeks (see
# plausibility_checks() below and data_report.json).
#
# Only full-day holidays feed the model feature `is_holiday`. Half-day eves are exported as a
# descriptive column for other users.

FIXED_HOLIDAYS = [
    (1, 1, "New Year's Day", 2016),
    (4, 23, "National Sovereignty and Children's Day", 2016),
    (5, 1, "Labour and Solidarity Day", 2016),
    (5, 19, "Commemoration of Atatürk, Youth and Sports Day", 2016),
    (7, 15, "Democracy and National Unity Day", 2017),
    (8, 30, "Victory Day", 2016),
    (10, 29, "Republic Day", 2016),
]

# First day of each religious holiday. Ramazan Bayramı lasts 3 days, Kurban Bayramı 4 days.
RAMAZAN_FIRST_DAY = {
    2016: date(2016, 7, 5), 2017: date(2017, 6, 25), 2018: date(2018, 6, 15), 2019: date(2019, 6, 4),
    2020: date(2020, 5, 24), 2021: date(2021, 5, 13), 2022: date(2022, 5, 2), 2023: date(2023, 4, 21),
    2024: date(2024, 4, 10), 2025: date(2025, 3, 30),
}
KURBAN_FIRST_DAY = {
    2016: date(2016, 9, 12), 2017: date(2017, 9, 1), 2018: date(2018, 8, 21), 2019: date(2019, 8, 11),
    2020: date(2020, 7, 31), 2021: date(2021, 7, 20), 2022: date(2022, 7, 9), 2023: date(2023, 6, 28),
    2024: date(2024, 6, 16), 2025: date(2025, 6, 6),
}

# Law No. 2429: the eves of both bayrams and 28 October are holidays from 13:00 onwards.
HALF_DAY_START_HOUR = 13


def holidays(years) -> tuple[dict[date, tuple[str, str]], dict[date, str]]:
    """Return ({date: (name, type)} for full-day holidays, {date: name} for half-day eves)."""
    full: dict[date, list[tuple[str, str]]] = {}
    half: dict[date, str] = {}

    def add(d: date, name: str, kind: str) -> None:
        full.setdefault(d, []).append((name, kind))

    for y in years:
        for m, d, name, since in FIXED_HOLIDAYS:
            if y >= since:
                add(date(y, m, d), name, "national")
        half[date(y, 10, 28)] = "Republic Day eve (from 13:00)"
        for first, n_days, name in ((RAMAZAN_FIRST_DAY[y], 3, "Ramazan Bayramı (Eid al-Fitr)"),
                                    (KURBAN_FIRST_DAY[y], 4, "Kurban Bayramı (Eid al-Adha)")):
            for k in range(n_days):
                add(first + timedelta(days=k), f"{name} day {k + 1}", "religious")
            half[first - timedelta(days=1)] = f"{name} eve (from 13:00)"

    # Two holidays can fall on the same date (e.g. 23 April 2023 was also Ramazan Bayramı day 3).
    merged = {d: ("; ".join(n for n, _ in v), "religious" if any(k == "religious" for _, k in v) else "national")
              for d, v in full.items()}
    # An eve that is itself a full holiday stays a full holiday.
    half = {d: n for d, n in half.items() if d not in merged}
    return merged, half

# ==============================================================================================
# 2. Load series: raw EPİAŞ export -> validated, corrected hourly series
# ==============================================================================================
# Stage 0-1: raw EPİAŞ export -> validated, corrected hourly load series with calendar columns.
#
# Why this module is strict
# -------------------------
# An earlier version of this project parsed the same Excel file with pandas' day-first
# guessing. The 2020-2025 cells are real Excel dates whose text form is ISO
# ('2020-02-01'); re-reading that text day-first silently swapped day and month for days
# 1-12 (21.7 % of all hours). Row counts, totals, minimum and maximum all stayed the same,
# so nothing looked wrong. This module therefore:
#   * parses each cell type with an explicit rule (no format guessing at all),
#   * checks the time axis in *raw row order*: the export is chronological, and any
#     day/month swap breaks that order (a mathematical invariant, not a heuristic),
#   * accepts only anomalies registered in KNOWN_ANOMALIES and logs every change,
#   * keeps behavioural plausibility checks separate: they only raise warnings.
#
# Time convention: EPİAŞ labels mark the *start* of the hourly interval ("09:00" is the
# consumption of 09:00-10:00 local time). Türkiye used UTC+2 until 2016-03-27 03:00, then
# UTC+3 (permanent since 2016-09-07); conversion uses the IANA zone Europe/Istanbul.

TZ = ZoneInfo("Europe/Istanbul")
VALUE = "Consumption (MWh)"
RAW_COLUMNS = ["Date", "Hour", VALUE]
EXPECTED_HOURS = 87_671          # physical hours from 2016-01-01 00:00 to 2025-12-31 23:00 local
ONE_HOUR_NS = 3_600 * 10**9

# Every accepted deviation from "one row per hour with a positive value".
# Anything not listed here stops the pipeline; a listed item that is NOT found also stops it
# (the input file would then not be the file this pipeline was validated on).
KNOWN_ANOMALIES = {
    ("2016-03-27 02:00", "non_positive"):
        "0 placeholder written by the export on the DST switch day; treated as missing and "
        "linearly interpolated from the neighbouring hours",
    ("2016-03-27 04:00", "duplicate"):
        "exact repetition of the previous row on the DST switch day; the second copy is removed",
}

# Used only to pick reference days for the holiday plausibility check (not a data column):
# nationwide full lockdown of 29 April - 17 May 2021 depressed the load of normal days.
_REFERENCE_EXCLUSIONS = [(pd.Timestamp("2021-04-29"), pd.Timestamp("2021-05-17"))]


class DataError(RuntimeError):
    """Raised when the raw file violates a hard rule; the pipeline must not continue."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_raw(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, dtype={"Hour": str})
    if list(raw.columns) != RAW_COLUMNS:
        raise DataError(f"unexpected columns {list(raw.columns)}, expected {RAW_COLUMNS}")
    raw.insert(0, "raw_row", np.arange(len(raw)) + 2)  # Excel row number (row 1 is the header)
    return raw


def parse_local_labels(raw: pd.DataFrame) -> pd.Series:
    """Explicit parsing of the two cell types found in the export.

    * text 'DD.MM.YYYY' (2016-2019)          -> parsed with that exact format
    * real Excel datetime (2020-2025)        -> taken as a date; its time part must equal `Hour`
    The hour always comes from the `Hour` column ('HH:00').
    """
    hour_txt = raw["Hour"].astype(str)
    if (~hour_txt.str.fullmatch(r"([01]\d|2[0-3]):00")).any():
        raise DataError(f"malformed Hour values at raw rows {raw.loc[~hour_txt.str.fullmatch(r'([01]\d|2[0-3]):00'), 'raw_row'].tolist()[:5]}")
    hour = hour_txt.str[:2].astype(int)

    cells = raw["Date"]
    is_dt = cells.map(lambda v: isinstance(v, datetime))
    is_txt = cells.map(lambda v: isinstance(v, str))
    if not (is_dt | is_txt).all():
        raise DataError(f"unsupported Date cell types at raw rows {raw.loc[~(is_dt | is_txt), 'raw_row'].tolist()[:5]}")

    day = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    txt = cells[is_txt].astype(str)
    if (~txt.str.fullmatch(r"\d{2}\.\d{2}\.\d{4}")).any():
        raise DataError("text dates not in DD.MM.YYYY format")
    day[is_txt] = pd.to_datetime(txt, format="%d.%m.%Y")

    stamps = pd.DatetimeIndex(list(cells[is_dt]))
    if ((stamps.minute != 0) | (stamps.second != 0)).any() or (stamps.hour != hour[is_dt].to_numpy()).any():
        raise DataError("Excel datetime cells whose time part disagrees with the Hour column")
    day[is_dt] = stamps.normalize()

    return day + pd.to_timedelta(hour, unit="h")


def _label(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M")


def build_load_series(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (hourly frame, corrections ledger, hard-check results). Raises DataError on any violation."""
    labels = parse_local_labels(raw)
    values = raw[VALUE].astype(float)

    # --- 1. anomalies must match the registry exactly --------------------------------------
    found = {}
    dup = labels.duplicated(keep="first")
    for i in raw.index[dup]:
        first = raw.index[(labels == labels[i]) & ~dup][0]
        if values[i] != values[first]:
            raise DataError(f"raw row {raw.raw_row[i]}: duplicate timestamp with a different value; manual decision needed")
        found[(_label(labels[i]), "duplicate")] = i
    for i in raw.index[(values <= 0) | values.isna()]:
        found[(_label(labels[i]), "non_positive")] = i
    unknown = set(found) - set(KNOWN_ANOMALIES)
    missing = set(KNOWN_ANOMALIES) - set(found)
    if unknown or missing:
        raise DataError(f"anomalies differ from the registry. unknown: {sorted(unknown)}; not found: {sorted(missing)}")

    # --- 2. time axis in raw row order (catches any day/month swap) ------------------------
    utc = pd.DatetimeIndex(labels).tz_localize(TZ, ambiguous="raise", nonexistent="raise").tz_convert("UTC")
    steps = np.diff(utc.asi8)
    allowed_zero_steps = {found[k] - 1 for k in found if k[1] == "duplicate"}  # step into the duplicate row
    bad = [j for j, s in enumerate(steps) if s != ONE_HOUR_NS and not (s == 0 and j in allowed_zero_steps)]
    if bad:
        rows = [(int(raw.raw_row[j]), int(raw.raw_row[j + 1])) for j in bad[:5]]
        raise DataError(f"time axis is not strictly hourly in raw row order at raw rows {rows}")

    # --- 3. corrections, each written to the ledger ----------------------------------------
    ledger = []
    keep = np.ones(len(raw), dtype=bool)
    for (lab, kind), i in sorted(found.items()):
        if kind == "duplicate":
            keep[i] = False
            ledger.append({"raw_row": int(raw.raw_row[i]), "datetime_local": lab, "datetime_utc": utc[i].strftime("%Y-%m-%dT%H:%MZ"),
                           "action": "removed_duplicate_row", "old_value": values[i], "new_value": np.nan,
                           "reason": KNOWN_ANOMALIES[(lab, kind)]})
    s = pd.Series(values[keep].to_numpy(), index=utc[keep], name="consumption_mwh")
    s[s <= 0] = np.nan
    s = s.interpolate(method="time", limit=1, limit_area="inside")
    if s.isna().any():
        raise DataError("missing values remain after the registered interpolation")
    for (lab, kind), i in sorted(found.items()):
        if kind == "non_positive":
            ledger.append({"raw_row": int(raw.raw_row[i]), "datetime_local": lab, "datetime_utc": utc[i].strftime("%Y-%m-%dT%H:%MZ"),
                           "action": "imputed_linear", "old_value": values[i], "new_value": float(s[utc[i]]),
                           "reason": KNOWN_ANOMALIES[(lab, kind)]})
    ledger = pd.DataFrame(ledger)

    # --- 4. hard checks on the corrected series --------------------------------------------
    checks = {}
    full = pd.date_range(s.index[0], s.index[-1], freq="h")
    checks["continuous_hourly_utc"] = bool(len(full) == len(s) and (full == s.index).all())
    checks["n_hours"] = int(len(s))
    checks["n_hours_expected"] = EXPECTED_HOURS
    local = s.index.tz_convert(TZ).tz_localize(None)
    checks["roundtrip_local_utc_local"] = bool((local == pd.DatetimeIndex(labels[keep])).all())
    # every kept raw row reappears unchanged unless the ledger says otherwise
    imputed_utc = set(pd.to_datetime(ledger.loc[ledger.action == "imputed_linear", "datetime_utc"], utc=True))
    same = np.isclose(s.to_numpy(), values[keep].to_numpy(), rtol=0, atol=1e-9)
    changed = {u for u, ok in zip(s.index, same) if not ok}
    checks["raw_values_preserved_except_ledger"] = bool(changed == imputed_utc)
    removed = ledger.loc[ledger.action == "removed_duplicate_row", "old_value"].sum()
    replaced_old = ledger.loc[ledger.action == "imputed_linear", "old_value"].sum()
    replaced_new = ledger.loc[ledger.action == "imputed_linear", "new_value"].sum()
    checks["total_balance_mwh"] = float(values.sum() - removed - replaced_old + replaced_new - s.sum())
    ok = (checks["continuous_hourly_utc"] and checks["n_hours"] == EXPECTED_HOURS and checks["roundtrip_local_utc_local"]
          and checks["raw_values_preserved_except_ledger"] and abs(checks["total_balance_mwh"]) < 1e-3)
    if not ok:
        raise DataError(f"hard checks failed: {checks}")

    frame = pd.DataFrame({"datetime_local": local, "datetime_utc": s.index, "consumption_mwh": s.to_numpy()})
    frame["is_imputed"] = frame["datetime_utc"].isin(list(imputed_utc)).astype(int)
    return frame, ledger, checks


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    """Calendar columns derived from the local time label only (no information from the load)."""
    t = frame["datetime_local"]
    out = frame.copy()
    out["year"], out["month"], out["day"], out["hour"] = t.dt.year, t.dt.month, t.dt.day, t.dt.hour
    out["day_of_week"] = t.dt.dayofweek                      # 0 = Monday ... 6 = Sunday
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    full, half = holidays(sorted(t.dt.year.unique()))
    d = t.dt.date
    out["is_holiday"] = d.map(lambda x: int(x in full))
    out["holiday_name"] = d.map(lambda x: full.get(x, ("", ""))[0] or half.get(x, ""))
    out["holiday_type"] = d.map(lambda x: full.get(x, ("", ""))[1] or ("half_day" if x in half else ""))
    out["is_half_day_holiday"] = (d.map(lambda x: x in half) & (out["hour"] >= HALF_DAY_START_HOUR)).astype(int)
    # cyclical encodings used as model inputs
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * out["hour"] / 24), np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"], out["dow_cos"] = np.sin(2 * np.pi * out["day_of_week"] / 7), np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out


def plausibility_checks(df: pd.DataFrame, compare_file: Path | None = None) -> tuple[dict, list[str]]:
    """Behavioural checks. They describe the data and raise *warnings*, never stop the pipeline:
    a real change in consumption could violate them, and a wrong series could pass them."""
    s = pd.Series(df["consumption_mwh"].to_numpy(), index=pd.DatetimeIndex(df["datetime_local"]))
    daily = s.resample("D").mean()
    res, warn = {}, []

    # 1. signature of a day/month swap: midnight jumps on days 1-12 vs 13-31
    jump = s.diff().abs()
    mid = jump[jump.index.hour == 0]
    ratio = (mid[mid.index.day <= 12].groupby(mid[mid.index.day <= 12].index.year).mean()
             / mid[mid.index.day > 12].groupby(mid[mid.index.day > 12].index.year).mean())
    res["midnight_jump_ratio_by_year"] = ratio.round(3).to_dict()
    warn += [f"midnight jump ratio {y}: {r:.2f} outside [0.9, 1.1]" for y, r in ratio.items() if not 0.9 <= r <= 1.1]

    # 2. weekly structure: Sunday / Wednesday daily mean
    wk = daily.groupby([daily.index.year, daily.index.dayofweek]).mean().unstack()
    sun_wed = (wk[6] / wk[2]).round(3)
    res["sunday_over_wednesday_by_year"] = sun_wed.to_dict()
    warn += [f"Sunday/Wednesday {y}: {r:.2f} outside [0.80, 0.90]" for y, r in sun_wed.items() if not 0.80 <= r <= 0.90]

    # 3. religious holidays must show a clear drop against the same weekdays of nearby weeks
    hol_days = set(pd.to_datetime(df.loc[df.is_holiday == 1, "datetime_local"]).dt.normalize())
    eve_days = set(pd.to_datetime(df.loc[df.holiday_type == "half_day", "datetime_local"]).dt.normalize())

    def usable(r):
        return r in daily.index and r not in hol_days and r not in eve_days and \
            not any(a <= r <= b for a, b in _REFERENCE_EXCLUSIONS)

    dips = {}
    for label, firsts, n_days in (("Ramazan", RAMAZAN_FIRST_DAY, 3), ("Kurban", KURBAN_FIRST_DAY, 4)):
        for year, first in firsts.items():
            vals = []
            for k in range(n_days):
                d = pd.Timestamp(first + timedelta(days=k))
                refs = [r for r in (d + timedelta(weeks=w) for w in (-1, 1, -2, 2, -3, 3)) if usable(r)][:4]
                vals.append(daily[d] / daily[refs].mean() - 1)
            dips[f"{label} {year}"] = round(100 * float(np.mean(vals)), 1)
    res["religious_holiday_dip_pct"] = dips
    warn += [f"{e}: dip {v}% weaker than -15%" for e, v in dips.items() if v > -15]

    # 4. external anchor: record day reported by Anadolu Agency (9 July 2024: 1,147,848 MWh)
    total = float(s.loc["2024-07-09"].sum())
    res["anchor_2024_07_09_total_mwh"] = round(total, 1)
    if abs(total / 1_147_848 - 1) > 0.005:
        warn.append(f"2024-07-09 total {total:,.0f} MWh differs >0.5% from the reported 1,147,848 MWh")

    # 5. optional: independent re-derivation (e.g. the earlier corrected file) must agree
    if compare_file is not None:
        other = pd.read_csv(compare_file, parse_dates=["datetime"]).drop_duplicates("datetime", keep="first")
        other = pd.Series(other[VALUE].to_numpy(), index=other["datetime"])
        common = s.index.intersection(other.index)
        common = common[common.normalize() != pd.Timestamp("2016-03-27")]  # DST day handled differently
        diff = int((~np.isclose(s[common], other[common], atol=0.005)).sum())
        res["compare_file"] = {"file": compare_file.name, "common_hours": int(len(common)), "differing_hours": diff}
        if diff:
            warn.append(f"{diff} hours differ from {compare_file.name}")
    return res, warn

# ==============================================================================================
# 3. Weather: raw Open-Meteo downloads and national population-weighted features
# ==============================================================================================
# Raw weather acquisition from Open-Meteo (stage 2, download part only).
#
# Why this module exists
# ----------------------
# The load series is national, so temperature must be aggregated over the whole
# country. We download hourly data for the 81 province capitals and aggregate it
# later (population weights) in the data pipeline. This module only fetches and
# archives the *raw* API responses; no transformation happens here, so every
# derived number can be traced back to an untouched file with a checksum.
#
# Two products are fetched:
#   * ERA5 reanalysis (models=era5): observed-like weather for training and for
#     the "oracle" upper bound. The model is pinned explicitly because Open-Meteo's
#     default "best match" switches to a different model in 2017, which would put
#     an artificial break inside the 2016-2025 series.
#   * Archived ECMWF IFS 0.25 deg forecasts (Previous Runs API): what forecasters
#     actually knew 1..7 days ahead. Used for the realistic (ex-ante) evaluation.
#
# All timestamps are requested in GMT/UTC. Türkiye's local time changed from
# UTC+2 to UTC+3 on 2016-03-27, and Open-Meteo's Europe/Istanbul option applies a
# fixed +03:00 even in early 2016, so local-time conversion is done by our own code.
#
# Only the Python standard library is used, so the download can run before (or
# independently of) the heavy ML environment.

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"
PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# The load data start at local 2016-01-01 00:00 (= 2015-12-31 22:00 UTC), so the
# UTC request must begin one day earlier to cover the first local hours.
ERA5_START, ERA5_END = "2015-12-31", "2025-12-31"
ERA5_VARS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "shortwave_radiation"]

# Archived forecasts are needed for the validation (2024) and test (2025) years.
# Coverage before ~March 2024 is expected to be missing; it is recorded, not faked.
PREV_START, PREV_END = "2023-12-31", "2025-12-31"
PREV_MODEL = "ecmwf_ifs025"
PREV_VARS = ["temperature_2m"] + [f"temperature_2m_previous_day{d}" for d in range(1, 8)]

# Türkiye bounding box, used to reject wrong geocoding hits.
LAT_RANGE, LON_RANGE = (35.5, 42.5), (25.5, 45.0)

SLEEP_BETWEEN_CALLS_S = 3      # polite spacing between requests
SLEEP_ON_QUOTA_S = 15 * 60     # Open-Meteo free tier has minutely/hourly/daily quotas
MAX_TRANSIENT_RETRIES = 6


def _fold(s: str) -> str:
    """Case- and diacritic-insensitive key, so 'Kahramanmaraş' matches 'Kahramanmaras'."""
    s = s.replace("ı", "i").replace("İ", "i")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).casefold().replace(" ", "")


def _get_json(url: str, params: dict, log) -> tuple[dict, bytes]:
    """GET with quota-aware retries. Returns (parsed json, raw bytes)."""
    full = f"{url}?{urllib.parse.urlencode(params, safe=',')}"
    transient = 0
    while True:
        try:
            with urllib.request.urlopen(full, timeout=180) as r:
                raw = r.read()
            return json.loads(raw), raw
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:200]
            if e.code == 429:  # quota exhausted: wait instead of failing the whole job
                # The free tier has minutely, hourly and daily limits; wait just long enough.
                wait = 65 if "inutely" in body else 600 if "ourly" in body else SLEEP_ON_QUOTA_S * 4
                log(f"quota reached ({body.strip()}); sleeping {wait // 60} min")
                time.sleep(wait)
                continue
            if e.code >= 500 and transient < MAX_TRANSIENT_RETRIES:
                transient += 1
                time.sleep(30 * transient)
                continue
            raise RuntimeError(f"HTTP {e.code} for {full}: {body}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if transient >= MAX_TRANSIENT_RETRIES:
                raise
            transient += 1
            log(f"network error ({e}); retry {transient}")
            time.sleep(30 * transient)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write to a temp file and rename, so an interrupted run never leaves a half file."""
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def geocode_provinces(cfg_csv: Path, out_csv: Path, log=print) -> None:
    """Resolve each province capital to coordinates and store them with provenance.

    A hit is accepted only if it is an administrative seat (GeoNames PPLA/PPLC) in
    Türkiye, lies inside the country's bounding box and belongs to the expected
    province. Any doubt stops the run: silent wrong coordinates would bias the
    national temperature without any visible error.
    """
    rows, problems = [], []
    with open(cfg_csv, encoding="utf-8") as f:
        provinces = list(csv.DictReader(f))
    for p in provinces:
        params = {"name": p["geocode_name"], "count": 10, "language": "tr", "format": "json", "countryCode": "TR"}
        data, _ = _get_json(GEOCODE_URL, params, log)
        hits = [h for h in data.get("results", []) if h.get("feature_code") in ("PPLA", "PPLC")]
        hits = [h for h in hits if _fold(p["province"]) in _fold(h.get("admin1", "")) or _fold(h.get("admin1", "")) in _fold(p["province"])]
        if len(hits) != 1:
            problems.append(f"{p['plate']} {p['province']}: {len(hits)} matching administrative seats")
            continue
        h = hits[0]
        if not (LAT_RANGE[0] <= h["latitude"] <= LAT_RANGE[1] and LON_RANGE[0] <= h["longitude"] <= LON_RANGE[1]):
            problems.append(f"{p['plate']} {p['province']}: coordinates outside Türkiye")
            continue
        rows.append({"plate": int(p["plate"]), "province": p["province"], "place": h["name"],
                     "admin1": h.get("admin1", ""), "feature_code": h["feature_code"],
                     "latitude": h["latitude"], "longitude": h["longitude"], "geonames_id": h.get("id", "")})
        time.sleep(0.5)
    if problems:
        raise RuntimeError("Geocoding needs manual review:\n  " + "\n  ".join(problems))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log(f"geocoded {len(rows)} provinces -> {out_csv} (source: {GEOCODE_URL}, GeoNames)")


def _expected_hours(start: str, end: str) -> int:
    return ((date.fromisoformat(end) - date.fromisoformat(start)).days + 1) * 24


def _validate(payload: dict, variables: list[str], start: str, end: str, allow_missing: bool) -> str:
    """Return a short coverage note; raise if the payload is structurally wrong."""
    hourly = payload.get("hourly", {})
    n = len(hourly.get("time", []))
    if n != _expected_hours(start, end):
        raise ValueError(f"expected {_expected_hours(start, end)} hours, got {n}")
    notes = []
    for v in variables:
        vals = hourly.get(v)
        if vals is None:
            raise ValueError(f"variable {v} missing from response")
        missing = sum(x is None for x in vals)
        if missing and not allow_missing:
            raise ValueError(f"{v}: {missing} missing values in reanalysis")
        if missing:
            first = next((hourly["time"][i] for i, x in enumerate(vals) if x is not None), "none")
            notes.append(f"{v}: {missing} missing, first value at {first}")
    return "; ".join(notes) or "complete"


def download_weather(provinces_csv: Path, out_dir: Path, log=print) -> None:
    """Fetch archived forecasts first (small, and they decide whether the ex-ante
    design is feasible), then ERA5. Existing valid files are skipped (resumable)."""
    with open(provinces_csv, encoding="utf-8") as f:
        provinces = list(csv.DictReader(f))
    manifest = out_dir / "MANIFEST.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    new_manifest = not manifest.exists()

    jobs = []
    for p in provinces:
        base = {"latitude": p["latitude"], "longitude": p["longitude"], "timezone": "GMT"}
        jobs.append(("previous_runs", p, PREV_RUNS_URL, PREV_VARS, PREV_START, PREV_END, True,
                     {**base, "hourly": ",".join(PREV_VARS), "models": PREV_MODEL,
                      "start_date": PREV_START, "end_date": PREV_END}))
    for p in provinces:
        base = {"latitude": p["latitude"], "longitude": p["longitude"], "timezone": "GMT"}
        jobs.append(("era5", p, ERA5_URL, ERA5_VARS, ERA5_START, ERA5_END, False,
                     {**base, "hourly": ",".join(ERA5_VARS), "models": "era5",
                      "start_date": ERA5_START, "end_date": ERA5_END}))

    with open(manifest, "a", newline="", encoding="utf-8") as mf:
        mw = csv.writer(mf)
        if new_manifest:
            mw.writerow(["file", "sha256", "retrieved_utc", "coverage", "url"])
        for i, (kind, p, url, variables, start, end, allow_missing, params) in enumerate(jobs, 1):
            path = out_dir / kind / f"{int(p['plate']):02d}_{_fold(p['province'])}.json"
            if path.exists():
                continue  # resumable: completed files were validated before being written
            path.parent.mkdir(parents=True, exist_ok=True)
            payload, raw = _get_json(url, params, log)
            coverage = _validate(payload, variables, start, end, allow_missing)
            _atomic_write(path, raw)
            mw.writerow([path.relative_to(out_dir).as_posix(), hashlib.sha256(raw).hexdigest(),
                         datetime.now(timezone.utc).isoformat(timespec="seconds"), coverage,
                         f"{url}?{urllib.parse.urlencode(params, safe=',')}"])
            mf.flush()
            log(f"[{i}/{len(jobs)}] {kind} {p['province']}: {coverage}")
            time.sleep(SLEEP_BETWEEN_CALLS_S)
    log("weather download complete")


# ============================================================================================
# Feature construction (stage 2): raw province responses -> national, population-weighted series
# ============================================================================================

TURKIYE_TOTAL_2022 = 85_279_553   # TÜİK ADNKS 2022 bulletin no. 49685 (6 February 2023)
HEATING_BASE_C, COOLING_BASE_C = 18.0, 22.0   # MGM degree-day conventions, applied per hour


def load_population(xls_path: Path, provinces_cfg: Path) -> pd.DataFrame:
    """Province populations on 31 Dec 2022 (TÜİK ADNKS 2022), returned as weights in plate order.

    A single, fixed weight vector is used for the whole study. It was published in February 2023,
    i.e. before the validation (2024) and test (2025) years, so no future information enters the
    temperature feature. The table must list the 81 provinces in plate order and sum to the
    published national total; otherwise we stop.
    """
    raw = pd.read_excel(xls_path, sheet_name=0, header=None)
    total_rows = raw.index[raw[0].astype(str).str.startswith("Toplam")]
    if len(total_rows) != 1 or int(raw.iloc[total_rows[0], 2]) != TURKIYE_TOTAL_2022:
        raise ValueError("national 2022 total not found or different from the published 85,279,553")
    body = raw.iloc[total_rows[0] + 1: total_rows[0] + 82, [0, 2]]
    body.columns = ["province_tuik", "population_2022"]
    with open(provinces_cfg, encoding="utf-8") as f:
        cfg = list(csv.DictReader(f))
    mismatch = [(c["province"], t) for c, t in zip(cfg, body.province_tuik) if _fold(c["province"]) != _fold(str(t))]
    if len(body) != 81 or mismatch:
        raise ValueError(f"province table does not match configs/provinces_tr.csv: {mismatch[:5]}")
    pop = body.astype({"population_2022": "int64"}).reset_index(drop=True)
    if int(pop.population_2022.sum()) != TURKIYE_TOTAL_2022:
        raise ValueError("province populations do not add up to the national total")
    pop.insert(0, "plate", [int(c["plate"]) for c in cfg])
    pop["province"] = [c["province"] for c in cfg]
    pop["weight"] = pop.population_2022 / TURKIYE_TOTAL_2022
    return pop[["plate", "province", "population_2022", "weight"]]


def _read_hourly(path: Path) -> pd.DataFrame:
    h = json.loads(path.read_text(encoding="utf-8"))["hourly"]
    idx = pd.to_datetime(h.pop("time"), utc=True)          # requested with timezone=GMT
    return pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in h.items()}, index=idx)


def _stack(kind_dir: Path, plates, variable: str) -> pd.DataFrame:
    """hours x provinces matrix of one variable, columns in plate order."""
    cols = {}
    for p in plates:
        files = list(kind_dir.glob(f"{p:02d}_*.json"))
        if len(files) != 1:
            raise FileNotFoundError(f"expected one file for plate {p:02d} in {kind_dir}")
        cols[p] = _read_hourly(files[0])[variable]
    return pd.DataFrame(cols)


def _weighted(matrix: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    # NaN in any province propagates (np.dot), so incomplete hours are never silently re-weighted.
    return matrix.to_numpy() @ w


def era5_features(openmeteo_dir: Path, pop: pd.DataFrame) -> pd.DataFrame:
    """National population-weighted ERA5 features, indexed by UTC hour.

    Degree-hours are computed per province *before* weighting: summer cooling demand in the
    south and mild weather on the Black Sea coast would otherwise cancel out in an average.
    Temperature variables are instantaneous values at the start of the load interval [t, t+1h).
    Open-Meteo radiation is the mean of the *preceding* hour, so it is shifted by one hour to
    describe the same interval as the load.
    """
    d, plates, w = openmeteo_dir / "era5", pop.plate.tolist(), pop.weight.to_numpy()
    t = _stack(d, plates, "temperature_2m")
    out = pd.DataFrame(index=t.index)
    out["t2m_popw_c"] = _weighted(t, w)
    out["hdh18_popw"] = _weighted((HEATING_BASE_C - t).clip(lower=0), w)
    out["cdh22_popw"] = _weighted((t - COOLING_BASE_C).clip(lower=0), w)
    out["apparent_t_popw_c"] = _weighted(_stack(d, plates, "apparent_temperature"), w)
    out["rh_popw_pct"] = _weighted(_stack(d, plates, "relative_humidity_2m"), w)
    out["ssr_popw_wm2"] = pd.Series(_weighted(_stack(d, plates, "shortwave_radiation"), w), index=t.index).shift(-1)
    return out


def forecast_features(openmeteo_dir: Path, pop: pd.DataFrame) -> pd.DataFrame:
    """National population-weighted archived ECMWF IFS 0.25 forecasts by lead day (0 = latest run,
    N = issued about 24*N h before the valid time). Hours without full provincial coverage stay NaN."""
    d, plates, w = openmeteo_dir / "previous_runs", pop.plate.tolist(), pop.weight.to_numpy()
    out = None
    for n in range(0, 8):
        var = "temperature_2m" if n == 0 else f"temperature_2m_previous_day{n}"
        t = _stack(d, plates, var)
        out = pd.DataFrame(index=t.index) if out is None else out
        out[f"t2m_fc_d{n}_popw_c"] = _weighted(t, w)
        out[f"hdh18_fc_d{n}_popw"] = _weighted((HEATING_BASE_C - t).clip(lower=0), w)
        out[f"cdh22_fc_d{n}_popw"] = _weighted((t - COOLING_BASE_C).clip(lower=0), w)
    return out

# ==============================================================================================
# 4. Build and command line
# ==============================================================================================

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "epias_real_time_consumption_2016_2025.xlsx"
EXTERNAL = ROOT / "data" / "external"
PROCESSED = ROOT / "data" / "processed"

COLUMN_NOTES = {
    "datetime_local": "Türkiye local time, start of the hourly interval [t, t+1h). UTC+2 until 2016-03-27 03:00, UTC+3 afterwards.",
    "datetime_utc": "The same instant in UTC.",
    "consumption_mwh": "EPİAŞ Transparency Platform real-time consumption of the hour (MWh).",
    "is_imputed": "1 for the single hour whose raw value was a 0 placeholder (2016-03-27 02:00), linearly interpolated.",
    "year": "Calendar year of datetime_local.", "month": "Month (1-12).", "day": "Day of month.", "hour": "Hour of day (0-23).",
    "day_of_week": "0 = Monday ... 6 = Sunday.",
    "is_weekend": "1 on Saturday and Sunday.",
    "is_holiday": "1 on full-day official public holidays (Law No. 2429; religious dates announced by Diyanet). Model input.",
    "holiday_name": "Name of the holiday or half-day eve (descriptive).",
    "holiday_type": "national / religious / half_day (descriptive).",
    "is_half_day_holiday": "1 from 13:00 on the official half-day eves (28 October, eves of both bayrams). Descriptive, not a model input.",
    "hour_sin": "sin(2*pi*hour/24). Model input.", "hour_cos": "cos(2*pi*hour/24). Model input.",
    "dow_sin": "sin(2*pi*day_of_week/7). Model input.", "dow_cos": "cos(2*pi*day_of_week/7). Model input.",
    # weather (ERA5 via Open-Meteo, 81 province capitals weighted by TÜİK ADNKS 2022 population)
    "t2m_popw_c": "2 m air temperature (degC) at the start of the hour, population-weighted national average.",
    "hdh18_popw": "Heating degree-hours: sum_i w_i*max(0, 18 - T_i), computed per province, then weighted.",
    "cdh22_popw": "Cooling degree-hours: sum_i w_i*max(0, T_i - 22), computed per province, then weighted.",
    "apparent_t_popw_c": "Apparent (feels-like) temperature (degC), population-weighted.",
    "rh_popw_pct": "Relative humidity (%), population-weighted.",
    "ssr_popw_wm2": "Shortwave solar radiation (W/m2), mean over the hour [t, t+1h), population-weighted.",
}
FORECAST_NOTE = ("Archived ECMWF IFS 0.25deg forecasts (Open-Meteo Previous Runs API), population-weighted like the "
                 "ERA5 columns. dN = forecast issued about 24*N h before the valid time (d0 = latest run). Empty where "
                 "the archive is incomplete (before ~Feb 2024 for most leads).")
POP_FILE = EXTERNAL / "tuik_adnks_2022_province_population.xls"


def province_temperatures(openmeteo_dir: Path, pop: pd.DataFrame) -> dict:
    """Hourly 2 m temperature of every province (ERA5 and archived forecasts issued 1-7 days earlier),
    for experiments that correct forecasts per province before computing degree-hours: a bias
    correction of the national degree-hours themselves would break HDH >= max(18 - T, 0)."""
    plates = pop.plate.tolist()
    era5 = _stack(openmeteo_dir / "era5", plates, "temperature_2m")
    fc = [_stack(openmeteo_dir / "previous_runs", plates, f"temperature_2m_previous_day{n}") for n in range(1, 8)]
    if any(not f.index.equals(fc[0].index) for f in fc):
        raise ValueError("forecast lead days do not share one time axis")
    return {"plates": np.asarray(plates, dtype=np.int16), "weights": pop.weight.to_numpy(np.float64),
            "province": np.asarray(pop.province.tolist()),
            "era5_utc_ns": era5.index.as_unit("ns").asi8, "era5_t2m": era5.to_numpy(np.float32),
            "fc_utc_ns": fc[0].index.as_unit("ns").asi8, "fc_t2m": np.stack([f.to_numpy(np.float32) for f in fc]),
            "base_c": np.asarray([HEATING_BASE_C, COOLING_BASE_C])}


def content_sha256(arrays: dict) -> str:
    """Hash of array contents (the .npz container itself stores write times, so its bytes change)."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes() if a.dtype.kind != "U" else "|".join(a.tolist()).encode())
    return h.hexdigest()


def _weather_features():
    """Return (era5 features, forecast features, province temperatures, info). Weather is added only
    when every raw file is present."""
    n = {k: len(list((EXTERNAL / "openmeteo" / k).glob("*.json"))) for k in ("era5", "previous_runs")}
    if n != {"era5": 81, "previous_runs": 81} or not POP_FILE.exists():
        return None, None, None, {"status": f"pending: weather files {n}, population file {'found' if POP_FILE.exists() else 'missing'}"}
    pop = load_population(POP_FILE, ROOT / "configs" / "provinces_tr.csv")
    era5 = era5_features(EXTERNAL / "openmeteo", pop)
    fc = forecast_features(EXTERNAL / "openmeteo", pop)
    prov = province_temperatures(EXTERNAL / "openmeteo", pop)
    info = {"status": "complete",
            "population": {"source": "TÜİK ADNKS 2022, bulletin 49685 (6 Feb 2023)", "total": int(pop.population_2022.sum()),
                           "sha256": sha256(POP_FILE),
                           "largest_weights": pop.nlargest(5, "weight")[["province", "weight"]].round(4).values.tolist()},
            "forecast_first_complete_utc": {c: str(fc[c].first_valid_index()) for c in fc.columns if c.startswith("t2m")}}
    return era5, fc, prov, info


def _logger(path: Path):
    """Log to console and to a file, so long jobs can be checked later."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def _check_raw_checksum() -> str:
    """The raw file must be byte-identical to the archived one; otherwise the registry of known anomalies is void."""
    expected = (ROOT / "data" / "raw" / "SHA256SUMS").read_text().split()[0]
    actual = sha256(RAW_FILE)
    if actual != expected:
        sys.exit(f"raw file checksum mismatch: {actual} != {expected}")
    return actual


def build(accept_warnings: bool, compare_with: Path | None) -> int:
    log = _logger(ROOT / "logs" / "build.log")
    raw_sha = _check_raw_checksum()
    raw = read_raw(RAW_FILE)
    frame, ledger, hard = build_load_series(raw)
    df = add_calendar(frame)
    era5, fc, prov, wx_info = _weather_features()
    if era5 is not None:
        df = df.merge(era5, left_on="datetime_utc", right_index=True, how="left")
        if df[list(era5.columns)].isna().any().any():
            sys.exit("ERA5 features are missing for some load hours; check the weather download")
    df = df[[c for c in COLUMN_NOTES if c in df.columns]]
    plaus, warnings = plausibility_checks(df, compare_with)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["datetime_local"] = out["datetime_local"].dt.strftime("%Y-%m-%d %H:%M")
    out["datetime_utc"] = out["datetime_utc"].dt.strftime("%Y-%m-%dT%H:%MZ")
    csv_path, xlsx_path = PROCESSED / "tr_load_hourly.csv", PROCESSED / "tr_load_hourly.xlsx"
    out.to_csv(csv_path, index=False, float_format="%.10g")
    ledger.to_csv(PROCESSED / "corrections.csv", index=False)
    with pd.ExcelWriter(xlsx_path) as xw:
        out.to_excel(xw, sheet_name="hourly", index=False, freeze_panes=(1, 0))
        pd.DataFrame({"column": list(COLUMN_NOTES), "description": list(COLUMN_NOTES.values())}).to_excel(
            xw, sheet_name="columns", index=False)
        ledger.to_excel(xw, sheet_name="corrections", index=False)
        if fc is not None:
            fco = fc.reset_index(names="datetime_utc")
            fco.insert(1, "datetime_local", fco["datetime_utc"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d %H:%M"))
            fco["datetime_utc"] = fco["datetime_utc"].dt.strftime("%Y-%m-%dT%H:%MZ")
            fco.to_csv(PROCESSED / "tr_temperature_forecasts.csv", index=False, float_format="%.6g")
            fco.to_excel(xw, sheet_name="temperature_forecasts", index=False, freeze_panes=(1, 0))
            pd.DataFrame({"note": [FORECAST_NOTE]}).to_excel(xw, sheet_name="forecast_notes", index=False)
    if prov is not None:
        np.savez_compressed(PROCESSED / "tr_temperature_provinces.npz", **prov)

    per_year = df.groupby("year")["consumption_mwh"].agg(["count", "min", "mean", "max", "sum"])
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_file": {"name": RAW_FILE.name, "sha256": raw_sha, "rows": int(len(raw))},
        "hard_checks": hard,
        "corrections": ledger.to_dict(orient="records"),
        "per_year": {int(y): {"hours": int(r["count"]), "min_mwh": round(r["min"], 2), "mean_mwh": round(r["mean"], 2),
                              "max_mwh": round(r["max"], 2), "total_twh": round(r["sum"] / 1e6, 3)} for y, r in per_year.iterrows()},
        "holiday_hours_by_year": df.groupby("year")["is_holiday"].sum().astype(int).to_dict(),
        "plausibility": plaus,
        "weather": wx_info,
        "annual_mean_temperature_c": (df.groupby("year")["t2m_popw_c"].mean().round(2).to_dict()
                                      if "t2m_popw_c" in df.columns else None),
        "warnings": warnings,
        "approved": not warnings or accept_warnings,
        "outputs_sha256": {p.name: sha256(p) for p in (csv_path, PROCESSED / "corrections.csv",
                                                        PROCESSED / "tr_temperature_forecasts.csv") if p.exists()},
        "outputs_content_sha256": {"tr_temperature_provinces.npz": content_sha256(prov)} if prov is not None else {},
        "environment": {"python": platform.python_version(), "pandas": pd.__version__},
    }
    (PROCESSED / "data_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"built {len(df)} hours; corrections: {len(ledger)}; warnings: {len(warnings)}; approved: {report['approved']}")
    for w in warnings:
        log(f"WARNING: {w}")
    return 0 if report["approved"] else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "geocode", "weather"])
    ap.add_argument("--accept-warnings", action="store_true", help="approve the data after a person reviewed the warnings")
    ap.add_argument("--compare-with", type=Path, help="optional independent CSV to compare values against")
    args = ap.parse_args(argv)
    provinces = EXTERNAL / "provinces_geocoded.csv"
    if args.command == "build":
        return build(args.accept_warnings, args.compare_with)
    if args.command == "geocode":
        geocode_provinces(ROOT / "configs" / "provinces_tr.csv", provinces, _logger(ROOT / "logs" / "geocode.log"))
    elif args.command == "weather":
        if not provinces.exists():
            sys.exit("run 'python prepare_data.py geocode' first")
        download_weather(provinces, EXTERNAL / "openmeteo", _logger(ROOT / "logs" / "weather_download.log"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
