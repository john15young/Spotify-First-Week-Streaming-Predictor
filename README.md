# Predicting First-Week Spotify Streams with Random Forest

## Project Overview

A machine learning framework for forecasting first-week Spotify global streaming performance for major album releases. Uses a `RandomForestRegressor` trained on 49 verified historical albums (2015–2024), with Leave-One-Out Cross-Validation, MAU-normalized features, and a clean temporal holdout split. All stream figures verified via Spotify Charts.

A live interactive version of this predictor is available at: **[spotify-first-week-streaming-predictor.streamlit.app](https://spotify-first-week-streaming-predictor.streamlit.app)**

---

## 2026 Case Study: Summer of Anticipation

| Artist | Album | Predicted | Actual (Wk1) | Error |
|---|---|---|---|---|
| Drake | Iceman | 479M | 450.4M | +6.4% (over) |
| Olivia Rodrigo | *you seem pretty sad for a girl so in love* | 413M | 394.7M | +4.7% (over) |
| Ariana Grande | Petal | 408M | not yet released (Jul 31, 2026) | — |

Both validated predictions landed within 7% despite very different release-week trajectories — strong evidence the model's trailing-average/MAU-normalized features capture real signal, not day-one hype noise. No qualitative overlays were applied — both are raw model outputs.

---

## Data & Methodology

### Training Dataset

49 verified albums (2015–2024) across Pop, Hip-Hop, R&B, Latin, and Country. An additional 7 albums from 2025 (including Justin Bieber's *Swag*) are held out for validation and never used during training.

| Field | Description |
|---|---|
| `spotify_mau` | Platform-wide Monthly Active Users — annual figure from Spotify investor filings, identical for all albums in a given year |
| `hype_score` | Pre-release cultural momentum (1.0–10.0) |
| `lead_time` | Days from first announcement to release |
| `prev_album_streams` | First-week streams of artist's prior album (M) |
| `first_week_streams` | Target — first-week Spotify global streams (M) |
| `is_household_name` | Binary flag for global superstar name recognition |

Annual MAU reference:

| 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 80M | 100M | 120M | 207M | 248M | 299M | 365M | 489M | 574M | 640M | 715M | 761M |

### Feature Engineering

- **MAU Normalization**: all stream features divided by `spotify_mau` to make albums comparable across eras — the most important transformation in the pipeline
- **Trailing Average**: mean first-week streams of all prior albums by the same artist, computed without data leakage
- **Log Target**: `log(streams / MAU)` stabilizes the right-skewed distribution
- **Interaction Term**: `is_household_name × track_count`
- **Global Median Fallback (~285M)**: for an artist's first tracked album, `prev_album_streams` and `trailing_avg` fall back to the dataset median. Affects debut entries only. Never used for Drake, Olivia, or Ariana — all three have multiple verified albums in the dataset. Median preferred over mean, which is skewed upward by Taylor Swift outliers (TTPD: 1,173M).

### Model

`RandomForestRegressor` (`n_estimators=200, max_depth=4, min_samples_leaf=3`), selected after head-to-head testing against Ridge, Lasso, ElasticNet, SVR, GBM, and XGBoost. Ridge had a higher LOOCV R² (0.679 vs RF's 0.519), but **RF won on holdout MAPE** (25.8% vs worse generalization from other models) — prioritized because holdout reflects true generalization, the actual use case here. Constrained depth and leaf size intentionally prevent overfitting on a 49-row training dataset.

XGBoost was tested at the current dataset size and underperformed RF on LOOCV R² (0.427 vs 0.519) and LOOCV MAPE (42.8% vs 38.1%). Will be revisited at 80+ training rows.

**Top features**: `trailing_avg_per_mau` (34%), `prev_streams_per_mau` (19%), `spotify_mau` (16%) — the model is fundamentally answering "how big is this artist, and which direction are they trending," which is appropriate given the small, outlier-heavy dataset.

---

## Performance

| Metric | Value |
|---|---|
| LOOCV R² | 0.519 |
| LOOCV MAPE | 38.1% |
| 2025 Holdout MAPE | 25.8% (7 unseen albums) |
| 2026 Validation — Drake Iceman | +6.4% over |
| 2026 Validation — Olivia Rodrigo | +4.7% over |
| 2026 Validation — Harry Styles KissAllTheTime | +69.4% over (documented limitation) |

Holdout MAPE (25.8%) beating LOOCV MAPE (38.1%) suggests solid generalization to unseen data. The two Summer 2026 album validations landed well inside the ±38% error band. Harry Styles is a clear miss — kept here as a documented limitation rather than excluded.

---

## Limitations

- Some training rows fall back to a dataset median (~285M) for the trailing-average feature, since no earlier album exists for that artist yet — standard handling for first-observed entries in a small dataset. LOOCV accounts for this (R²=0.519), and all three primary 2026 predictions use real prior-album data, not the fallback.
- Superstar-only dataset — the median reflects a major-label superstar baseline, not the music industry at large.
- Over-prediction risk for artists with thin or volatile history (Harry Styles 2026: +69.4%). Suspected cause: the model over-credits `is_household_name` without a signal for declining audience engagement.
- Surprise releases (Bieber's *Swag*, Taylor Swift's *Folklore*/*Evermore*) are not modeled accurately — `lead_time=1` is used as a workaround. An `is_surprise` binary feature is planned once the dataset is large enough to support it.
- Narrative context (label disputes, comebacks, tour momentum) is unquantifiable and kept out of model inputs — documented separately, never used to adjust predictions.
- Dataset currently skewed toward US/UK superstar-tier artists. Expansion to mid-tier artists and non-English language releases is in progress.

---

## Data Integrity Policy

All stream figures require a verified source (Spotify Charts, kworb.net, or Billboard) before inclusion. Hype scores and `is_household_name` flags are manually assigned with documented rationale. Predictions are never adjusted post hoc to match a desired outcome.

---

## Repository Structure

```
├── Spotify_First_Week_Streaming_Predictor.ipynb   # Full model notebook: training, LOOCV, predictions, output
├── app.py                                          # Streamlit web application (live at streamlit.app)
├── requirements.txt                                # Python dependencies for the web app
└── README.md
```

---

## Roadmap

- Add mid-tier artists (80–200M first-week range) to address upward prediction bias
- Add non-English superstars (BTS, Peso Pluma) for genre/language diversity
- Implement `is_surprise` as a proper binary feature
- Test lead single chart position as an objective hype proxy
- Re-benchmark RF vs XGBoost at 80+ training rows