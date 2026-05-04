# Data-Driven Forecasting for High-Anticipation Album Releases

## Project Overview
A machine learning framework for forecasting **first-week Spotify global streaming performance** for major album releases. Uses a `RandomForestRegressor` trained on 49 verified historical albums, with Leave-One-Out Cross-Validation, MAU-normalized features, and a clean temporal holdout split. All stream figures verified via @StatsSpotify and Chartmetric.

---

## 2026 Case Study: Summer of Anticipation

| Artist | Album | Prediction | Adjusted |
|---|---|---|---|
| Drake | *Iceman* | 476M | **526M** (+50M overlay) |
| Olivia Rodrigo | *OR3* | 392M | — |
| Ariana Grande | *Petal* | 383M | — |

Drake's overlay accounts for two factors the model cannot encode: a 3-year absence premium (+25M) and post-beef cultural curiosity (+25M). These are documented separately, not baked into model inputs.

---

## Data & Methodology

### Training Dataset
49 verified albums (2015–2024) across Pop, Hip-Hop, R&B, Latin, and Country. Key fields:

| Field | Description |
|---|---|
| `spotify_mau` | Platform-wide Monthly Active Users — annual figure from Spotify investor filings, identical for all albums in a given year |
| `hype_score` | Pre-release cultural momentum (1.0–10.0) |
| `lead_time` | Days from first announcement to release |
| `prev_album_streams` | First-year streams of artist's prior album (M) |
| `first_week_streams` | **Target** — first-year Spotify streams (M) |
| `is_household_name` | Binary flag for global superstar name recognition |

**Annual MAU reference:**

| 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 80M | 100M | 120M | 207M | 248M | 299M | 365M | 489M | 574M | 640M | 715M | 761M |

### Feature Engineering
- **MAU Normalization:** All stream features divided by `spotify_mau` to make albums comparable across eras — the most important transformation in the pipeline
- **Trailing Average:** Mean first-year streams of all prior albums by the same artist, computed without data leakage
- **Log Target:** `log(streams / MAU)` stabilizes the right-skewed distribution
- **Interaction Term:** `is_household_name × track_count`
- **Global Median Fallback (~303M):** For an artist's first tracked album, `prev_album_streams` and `trailing_avg` fall back to the dataset median. Affects 30 of 49 training rows (debut entries only). **Never used for Drake, Olivia, or Ariana** — all three have multiple verified albums in the dataset. The median is preferred over the mean (344M) because the mean is pulled upward by Taylor Swift outliers.

### Model
`RandomForestRegressor` — `n_estimators=200`, `max_depth=4`, `min_samples_leaf=3`

Constrained depth and leaf size intentionally prevent overfitting on a 49-row dataset. Validated with **Leave-One-Out Cross-Validation** — each of the 49 training albums held out once, model retrained on the remaining 48.

---

## Performance

| Metric | Value |
|---|---|
| LOOCV R² | **0.517** |
| LOOCV MAPE | 38.3% |
| 2025 Holdout MAPE | **25.1%** (6 unseen albums) |
| 2026 Validation | Harry Styles predicted 339M, actual 205M (+66%) |

R² of 0.517 on 49 rows predicting music streaming — one of the noisiest entertainment metrics — is a meaningful result. The holdout MAPE (25.1%) being better than LOOCV (38.3%) confirms the model generalizes well to unseen data. The remaining unexplained variance reflects factors no feature set can capture: viral singles, playlist editorial, cultural timing.

---

## Limitations

- **30/49 training rows use an imputed prior** (303M median fallback for debut entries) — the model's top feature is estimated data for 61% of training rows. Mitigated by LOOCV measuring performance with imputation in place, and by the fact that all 2026 targets use real data only.
- **Superstar-only dataset** — 303M does not represent average music industry baselines; it represents the center of a major-label superstar distribution.
- **Systematic over-prediction** for artists with no prior history in the dataset (Harry Styles 2026: +66% error).
- **Narrative context is unquantifiable** — comeback arcs, controversy premiums, and viral moments require the documented qualitative overlay approach.

---

## Data Integrity Policy
All stream figures require a verified source (@StatsSpotify or Chartmetric) before inclusion. Hype scores and `is_household_name` flags are manually assigned with documented rationale. Qualitative overlays are always shown separately from model outputs — no inputs are adjusted to produce a desired prediction.

---

## Repository Structure
```
├── spotify_prediction_model_v3.py   # Model: training, LOOCV, predictions, output
└── README.md
```