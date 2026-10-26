# Multi-horizon forecasting of national electricity demand in Türkiye

Code and data of the paper *Multi-horizon forecasting of national electricity demand with calendar, holiday and
temperature information: an evaluation for Türkiye* (Rıdvan Çetin, Mustafa Sağlam, Abdulhalık Oğuz). The citation
will be added when the paper is published.

Türkiye's national hourly electricity consumption (EPİAŞ, 2016–2025) is forecast 1, 24, 168 and 720 h ahead from
every hour of 2025, after training on 2016–2023 and model selection on 2024.

## Contents

| Path | Content |
|---|---|
| `prepare_data.py` | Raw data → validated hourly series, calendar and holidays, weather features |
| `models.py` | The TCN architectures |
| `run_experiments.py` | Training, selection, `freeze` and `evaluate` for every model |
| `analyze_results.py` | Statistical comparisons, breakdowns, efficiency profile, tables and figures |
| `configs/` | Experiments, selections, the analysis plan and the freeze record |
| `data/` | Raw EPİAŞ export, processed series, temperatures, sources and checksums (`data/SOURCES.md`) |
| `results/runs.csv` | Validation and test metrics of all 759 runs |
| `tests/` | pytest suite |

## Environment

Python 3.12 and PyTorch 2.11. On Windows, install torch 2.11.0 from the PyTorch CUDA 12.8 index, then
`pip install -r requirements-lock.txt`; on Linux, `pip install -r requirements.in -c constraints-science.txt`.

## Use

```bash
python prepare_data.py build
python -m pytest -q tests
python run_experiments.py run --experiment baselines --devices auto
```

The run records, stored forecasts, trained TCN, recurrent and Chronos-2 LoRA models and raw weather files are archived on Zenodo: https://doi.org/10.5281/zenodo.23143335. The LightGBM and NeuralForecast models are available from the authors on request. The analysis commands of analyze_results.py need these files.

## Notes

- `.gitattributes` keeps every file byte for byte, because the checksums of the data and the configuration are
  checked before any computation.
- For this release, comments and docstrings of the scripts were edited. Run identifiers hash the code without
  comments, so they are unchanged (`tests/test_run_identity.py`); `evaluate` compares raw file hashes and asks
  for a new `freeze`.

## Licence

Code: MIT (`LICENSE`). Data: see `DATA_LICENSE.md`.
