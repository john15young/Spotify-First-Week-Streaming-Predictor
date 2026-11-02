# Predicting First-Week Spotify Streams with Random Forest

## Project Overview

A machine learning framework for forecasting first-week Spotify global streaming performance for major album releases. Uses a `RandomForestRegressor` trained on 48 verified historical albums (2015 to 2024), with Leave-One-Out Cross-Validation, MAU-normalized features, and a temporal holdout of 8 albums from 2025. All stream figures verified via Spotify Charts.

The project's central test is a case study on three summer 2026 albums: each predicted ahead of its release, then checked against the actual first-week figure.

A live interactive version of this predictor is available at: **[spotify-first-week-streaming-predictor.streamlit.app](https://spotify-first-week-streaming-predictor.streamlit.app)**

---

## 2026 Case Study: Summer of Anticipation

| Artist | Album | Predicted | Actual (Wk1) | Error |
|---|---|---|---|---|
| Drake | Iceman | 479M | 450.4M | +6.3% (over) |
| Olivia Rodrigo | *you seem pretty sad for a girl so in love* | 413M | 394.7M | +4.6% (over) |
| Ariana Grande | Petal | 408M | 236.1M | +72.8% (over) |

Two of three landed within 7%. The third did not, and it is the most informative result here.

Drake and Olivia were both predicted ahead of release, from raw model output with no manual adjustment. With only three albums, two close calls say little about the model's real accuracy.

Ariana is a clear miss, kept here as a documented limitation rather than excluded. The model expected a return close to her historical baseline; the album opened at roughly two-thirds of that. It matches the over-prediction pattern recorded for Harry Styles below. The model has no signal for a shrinking audience.

A note on scope: the 2025 holdout of 8 albums is the model's out-of-sample test. These three 2026 albums are a separate, smaller check made under live conditions. They were predicted with the earlier version of the model (see Model Revision below) and are never recomputed.

---

## Data & Methodology

### Training Dataset

48 verified albums (2015 to 2024) across Pop, Hip-Hop, R&B, Latin, and Country. A further 8 albums from 2025 (including Justin Bieber's *Swag* and The Weeknd's *Hurry Up Tomorrow*) are held out for validation and never used in training.

| Field | Description |
|---|---|
| `spotify_mau` | Platform-wide Monthly Active Users. Annual figure from Spotify investor filings, identical for all albums in a given year |
| `lead_time` | Days from the first official release-date announcement to the actual release |
| `history_per_mau` | Artist history signal: prior album's first-week streams and career average, MAU-normalized |
| `is_household_name` | Binary flag for global superstar name recognition |
| `first_week_streams` | Target. First-week Spotify global streams (M) |

The model also uses track count, feature-track count, and years since the artist's last album.

Annual MAU reference:

| 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 H1 | 2026 H2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80M | 100M | 120M | 207M | 248M | 299M | 365M | 489M | 574M | 640M | 715M | 761M | 788M |

Predictions are pinned to the MAU of the album's release period, not the current figure, so the same inputs always reproduce the same number as the platform grows.

### Feature Engineering

- **MAU Normalization**: all stream features divided by `spotify_mau` to make albums comparable across eras. The most important transformation in the pipeline.
- **Combined History Signal**: the artist's prior album and career average, averaged and MAU-normalized, computed without data leakage.
- **Log Target**: `log(streams / MAU)` stabilizes the right-skewed distribution.
- **Interaction Term**: `is_household_name × track_count`.
- **Global Median Fallback (~285M)**: for an artist's first tracked album, the history signal falls back to the dataset median. This affects debut entries only, and was never used for Drake, Olivia, or Ariana, who all have multiple verified albums in the dataset. Median is preferred over mean, which Taylor Swift outliers skew upward (TTPD: 1,173M).

### Model

`RandomForestRegressor` (`n_estimators=200, max_depth=4, min_samples_leaf=3`), compared against Ridge, Lasso, ElasticNet, SVR, GBM, and XGBoost. On the corrected data the picture is mixed: the linear models score slightly better than RF on cross-validation, and RF scores better on the 2025 holdout. With 48 training rows and 8 holdout albums, neither gap is large enough to call a clear winner. I kept RF because it handles the skewed, outlier-heavy data without extra tuning. Constrained depth and leaf size limit overfitting.

XGBoost clearly underperformed at the current dataset size. Will be revisited at 80+ training rows.

**Top features**: `history_per_mau` (about 55%), `spotify_mau` (about 20%), then the household-name interaction (about 7%). The model is mostly answering "how big is this artist, and how big was the platform."

---

## Performance

| Metric | Model | "Always guess the median" baseline |
|---|---|---|
| LOOCV R² (log scale) | 0.432 | n/a |
| LOOCV MAPE | 43.9% | 42.8% |
| LOOCV median error | 29.2% | 25.6% |
| 2025 holdout MAPE (8 albums) | 35.6% | 32.0% |

| Case study / validation | Result |
|---|---|
| 2026 Drake Iceman | +6.3% over |
| 2026 Olivia Rodrigo | +4.6% over |
| 2026 Ariana Grande Petal | +72.8% over (documented limitation) |
| 2026 Harry Styles KissAllTheTime | +80.0% over (documented limitation) |

### Reading the numbers honestly

The short version: **this model is not clearly better than a simple guess.**

The 0.432 R² is measured on `log(streams / MAU)`, so it scores how well the model places an album relative to the platform. On the raw stream scale, the same predictions score slightly below zero (-0.027), meaning they explain no more of the variation in actual streams than the average does. A few outsized albums (TTPD at 1,173M) dominate that variance and the model does not reach them.

The baseline column shows the same thing: always guessing the median album is about as accurate as the model on average, and slightly more accurate on the 2025 holdout and on the typical album. The model's average error is 43.9% and its typical error is 29.2%.

I am keeping the model as a documented, honest framework rather than presenting it as a working forecaster. The 2026 case study is a small sample, and the Ariana and Harry Styles misses show where it breaks.

---

## Limitations

- **No clear edge over a naive baseline.** See above. This is the headline limitation.
- Some training rows fall back to the dataset median (~285M) for the history feature, since no earlier album exists for that artist yet. All three 2026 case study predictions use real prior-album data, not the fallback.
- Superstar-only dataset. The median reflects a major-label superstar baseline, not the music industry at large.
- **Over-prediction for artists with declining momentum.** Ariana Grande 2026 (+72.8%) and Harry Styles 2026 (+80.0%) are the clearest cases. The model over-credits historical scale with no signal for a shrinking audience.
- **Over-prediction for artists with thin history.** With only one prior album, the history signal is close to noise. The live app flags this case before showing a prediction.
- **Front-loaded release assumption.** Every training album is a conventional release where streams concentrate in week one. Slow-burn releases that build over later weeks are not represented.
- Surprise releases (Bieber's *Swag*, Taylor Swift's *Folklore*/*Evermore*) are not modeled accurately. `lead_time=1` is used as a workaround.
- Narrative context (label disputes, comebacks, tour momentum) is unquantifiable and kept out of model inputs. Documented separately, never used to adjust predictions.
- Dataset skews toward US/UK superstar-tier artists.
- **Sample size is the binding constraint.** 48 training rows against 8 features limits what any algorithm can extract. This is a data ceiling, not a tuning problem.
- Two averaging methods are in use. Cross-validation averages the trees in log space, while the holdout and live app average each tree's final stream estimate. The second method gives slightly higher predictions. I have flagged this and not changed it.

---

## Data Integrity Policy

All stream figures require a verified source (Spotify Charts, kworb.net, or Billboard) before inclusion. `is_household_name` flags are manually assigned with documented rationale. Predictions are never adjusted post hoc to match a desired outcome.

Case study predictions are recorded ahead of each release and never recomputed. The model has since been revised and would produce different figures today. The logged numbers are kept as a record of what was actually predicted. Results that went against the model are documented rather than removed.

---

## Model Revision, October 2026

After the summer 2026 case study, I audited the model's inputs. I found two problems.

### Problem 1: `hype_score` was scored with hindsight

`hype_score` was a 1 to 10 rating of pre-release buzz that I assigned by hand. It was the model's most load-bearing input: removing it cost 0.071 of LOOCV R², far more than any other feature.

The problem was when I scored it. All of the historical albums were rated after they had already come out. I gave TTPD a 10 knowing it had opened at 1,173M. So the feature was partly recording the answer.

Cross-validation does not catch this. Holding an album out hides its result from the model, but does nothing about hindsight already baked into that album's inputs.

| Feature set | LOOCV R² | LOOCV MAPE | 2025 holdout MAPE |
|---|---|---|---|
| Original, 11 features | 0.495 | 40.6% | 39.2% |
| `hype_score` removed, 10 features | 0.424 | 44.4% | 32.6% |

Dropping it made the cross-validated numbers worse but the held-out year better. I removed it. I also merged the three near-interchangeable history inputs into one.

### Problem 2: `lead_time` was mostly wrong

I re-checked every album's lead time against its original announcement. Most of the entered values were wrong, often because I had used the wrong start date or counted from a later reschedule. I corrected them all to one definition: days from the first official release-date announcement to the actual release. Postponed albums count to the date they really came out.

With the old values, `lead_time` looked like a useful signal. With the correct ones it is mostly noise: removing it actually nudges cross-validated R² up slightly (+0.013). Part of the earlier model's apparent skill was coming from wrong data, not from real signal.

I also moved *Hurry Up Tomorrow* to the 2025 holdout, since it was released that year and had been sitting in training.

### Where that left the model

| | Original | Revised |
|---|---|---|
| Features | 11 | 8 |
| Training / holdout albums | 48 / 8 (old data) | 48 / 8 (corrected data) |
| LOOCV R² (log scale) | 0.495 | 0.432 |
| LOOCV MAPE | 40.6% | 43.9% |
| LOOCV median error | 29.0% | 29.2% |
| 2025 holdout MAPE | 39.2% | 35.6% |

The headline numbers got worse, which is the correct outcome. The old ones were propped up by a feature that had seen the answers and by lead times that were entered incorrectly. The honest conclusion is that, once both are fixed, the model is roughly at baseline level.

The three logged case study predictions were made before this revision. Re-running them on the corrected data gives essentially the same conclusions, but the logged figures are kept as recorded.

The revised model is cleaner but unproven. It has not yet predicted an album ahead of release. The next major release is the test.

---

## Repository Structure

```
├── Spotify_First_Week_Streaming_Predictor.ipynb   # Full model notebook: training, LOOCV, predictions, output
├── app.py                                          # Streamlit web application (live at streamlit.app)
├── .streamlit/config.toml                          # App theme
├── requirements.txt                                # Python dependencies for the web app
└── README.md
```

---

## Roadmap

- Broader artist coverage (mid-tier and non-English artists) to address upward prediction bias
- Additional pre-release signals to improve accuracy
- Re-benchmark RF vs XGBoost at 80+ training rows
