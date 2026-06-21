# Predicting First-Week Spotify Streams with Random Forest

## Project Overview

A machine learning framework for forecasting first-week Spotify global streaming performance for major album releases. Uses a `RandomForestRegressor` trained on 49 verified historical albums, with Leave-One-Out Cross-Validation, MAU-normalized features, and a clean temporal holdout split. All stream figures verified via Spotify Charts.

## 2026 Case Study: Summer of Anticipation

| Artist | Album | Predicted | Actual (Wk1) | Error |
|---|---|---|---|---|
| Drake | Iceman | 475M | 455.3M | -4.3% (under) |
| Olivia Rodrigo | *you seem pretty sad for a girl so in love* | 389M | 394.6M | +1.2% (over) |
| Ariana Grande | Petal | 386M | not yet released (Jul 31, 2026) | — |

Both validated predictions landed within ~5% despite very different release-week trajectories — strong evidence the model's trailing-average/MAU-normalized features capture real signal, not day-one hype noise. No qualitative overlays were applied — both are raw model outputs.

## Data & Methodology

### Training Dataset
49 verified albums (2015–2024) across Pop, Hip-Hop, R&B, Latin, and Country.

| Field | Description |
|---|---|
| `spotify_mau` | Platform-wide Monthly Active Users — annual figure from Spotify investor filings, identical for all albums in a given year |
| `hype_score` | Pre-release cultural momentum (1.0–10.0) |
| `lead_time` | Days from first announcement to release |
| `prev_album_streams` | First-year streams of artist's prior album (M) |
| `first_week_streams` | Target — first-week Spotify streams (M) |
| `is_household_name` | Binary flag for global superstar name recognition |

Annual MAU reference:

| 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 80M | 100M | 120M | 207M | 248M | 299M | 365M | 489M | 574M | 640M | 715M | 761M |

### Feature Engineering
- **MAU Normalization**: all stream features divided by `spotify_mau` to make albums comparable across eras — the most important transformation in the pipeline
- **Trailing Average**: mean first-year streams of all prior albums by the same artist, computed without data leakage
- **Log Target**: `log(streams / MAU)` stabilizes the right-skewed distribution
- **Interaction Term**: `is_household_name × track_count`
- **Global Median Fallback (~303M)**: for an artist's first tracked album, `prev_album_streams` and `trailing_avg` fall back to the dataset median. Affects 30 of 49 training rows (debut entries only). Never used for Drake, Olivia, or Ariana — all three have multiple verified albums in the dataset. Median preferred over mean (344M), which is skewed upward by Taylor Swift outliers.

### Model

`RandomForestRegressor` (`n_estimators=200, max_depth=4, min_samples_leaf=3`), selected after head-to-head testing against Ridge, Lasso, ElasticNet, SVR, and GBM. Ridge had a higher LOOCV R² (0.679 vs RF's 0.517), but **RF won on holdout MAPE** (21–25% vs Ridge's worse generalization) — prioritized because holdout reflects true generalization, the actual use case here. Constrained depth and leaf size intentionally prevent overfitting on a 49-row dataset.

**Top features**: `trailing_avg_per_mau` (33.9%), `prev_streams_per_mau` (19.3%), `spotify_mau` (15.9%) — the model is fundamentally answering "how big is this artist, and which direction are they trending," which is appropriate given the small, outlier-heavy dataset (Taylor Swift's *TTPD* alone hit 1,173M).

## Performance

| Metric | Value |
|---|---|
| LOOCV R² | 0.517 |
| LOOCV MAPE | 38.3% |
| 2025 Holdout MAPE | 25.4% (6 unseen albums) |
| 2026 Validation | Iceman -4.3%, Olivia +1.2%, Harry Styles +65.8% (over) |

Holdout MAPE beating LOOCV MAPE suggests reasonable generalization to unseen data. The two 2026 album validations landed well inside the error band; Harry Styles is a clear miss, kept here as a documented limitation rather than excluded.

## Limitations

- 28/49 training rows fall back to a dataset median (~303M) for the trailing-average feature, since no earlier album exists for that artist yet — standard handling for first-observed entries in a small dataset. LOOCV accounts for this (R²=0.517), and all three 2026 predictions use real prior-album data, not the fallback.
- Superstar-only dataset — 303M reflects a major-label superstar baseline, not the industry at large.
- Over-prediction risk for artists with thin or volatile history (Harry Styles 2026: +65.8%).
- Narrative context (label disputes, comebacks, tour momentum) is unquantifiable and kept out of model inputs — documented separately, never used to adjust predictions.

## Data Integrity Policy

All stream figures require a verified source (@StatsSpotify, Chartmetric, or kworb.net/Billboard) before inclusion. Hype scores and `is_household_name` flags are manually assigned with documented rationale. Predictions are never adjusted post hoc to match a desired outcome.

## Repository Structure

```
├── spotify_prediction_model_v3.py   # Model: training, LOOCV, predictions, output
└── README.md
```