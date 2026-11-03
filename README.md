# Predicting First-Week Spotify Streams with Random Forest

## Project Overview

A machine learning framework for forecasting first-week Spotify global streaming performance for major album releases. Uses a `RandomForestRegressor` trained on 91 verified historical albums (2015 to 2024), with Leave-One-Out Cross-Validation, platform-size-adjusted features, and a temporal holdout of 20 albums from 2025. All stream figures verified via Spotify Charts.

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

A note on scope: the 2025 holdout of 20 albums is the model's out-of-sample test. These three 2026 albums are a separate, smaller check made under live conditions. They were predicted with the earlier version of the model (see Model Revision below) and are never recomputed.

---

## Data & Methodology

### Training Dataset

91 verified albums (2015 to 2024) from 41 artists across Pop, Hip-Hop, R&B, Latin, Country and K-pop. A further 20 albums from 2025 (including Justin Bieber's *Swag*, The Weeknd's *Hurry Up Tomorrow*, Bad Bunny's *DeBÍ TiRAR MáS FOToS*, Rosalía's *LUX*, Doja Cat's *Vie*, and several debut albums by new artists such as sombr and Addison Rae) are held out for validation and never used in training.

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

- **Platform-size adjustment**: stream features are divided by `spotify_mau` raised to the power 0.25, to make albums comparable across eras. Dividing by MAU in full over-corrects, because first-week streams grew far more slowly than the user base. The exponent was chosen by cross-validation (see Development Notes below).
- **Combined History Signal**: the artist's prior album and career average, averaged and MAU-normalized, computed without data leakage. Column names keep the `_per_mau` suffix for continuity.
- **Log Target**: `log(streams / MAU^0.25)` stabilizes the right-skewed distribution.
- **Interaction Term**: `is_household_name × track_count`.
- **Global Median Fallback (about 205M)**: for an artist's first tracked album, the history signal falls back to the median of the training albums only. This affects first-tracked albums, which is 34 of the 91 training rows, and was never used for Drake, Olivia, or Ariana, who all have multiple verified albums in the dataset. Median is preferred over mean, which Taylor Swift outliers skew upward (TTPD: 1,173M).

### Model

`RandomForestRegressor` (`n_estimators=200, max_depth=4, min_samples_leaf=3`), compared against Ridge, Lasso, ElasticNet, SVR, GBM, and XGBoost. With the platform-size adjustment in place, RF is ahead of the linear models in cross-validation, but by a small margin: a typical error of 39.7% against 41.2 to 42.0% for Ridge, Lasso and ElasticNet. On the 2025 holdout the linear models score better on average error (about 56% against 71.3% for RF using the app's averaging, or 60.7% averaged the cross-validation way). I chose RF on cross-validation alone, as set out in advance, but the holdout is a reminder that the edge is thin. Constrained depth and leaf size limit overfitting.

XGBoost clearly underperformed at 66 training rows. The dataset has now passed 80 rows, so it is due for a re-test.

**Top features**: the household-name interaction (about 39%), `history_per_mau` (about 18%), `lead_time` (about 12%), and `years_since_last_album` (about 10%). Once the platform-size adjustment is weakened, `spotify_mau` itself carries modest weight (about 8%). The model is mostly answering "how big is this artist, and how big was the platform."

---

## Performance

| Metric | Model | "Always guess the median" baseline |
|---|---|---|
| LOOCV MAPE (average error) | 66.7% | 85.6% |
| LOOCV median error | 39.7% | 43.8% |
| LOOCV R² on raw streams | 0.204 | n/a |
| 2025 holdout MAPE (20 albums) | 71.3% | 76.4% |
| 2025 holdout median error | 41.6% | 55.4% |

| Case study / validation | Result |
|---|---|
| 2026 Drake Iceman | +6.3% over |
| 2026 Olivia Rodrigo | +4.6% over |
| 2026 Ariana Grande Petal | +72.8% over (documented limitation) |
| 2026 Harry Styles KissAllTheTime | +42.4% over (documented limitation) |

### Reading the numbers honestly

**The model is modestly better than a simple guess, but its errors are large, the edge is not statistically solid, and holdout numbers are noisy.**

- **Cross-validation (91 albums):** typical error 39.7% against 43.8% for guessing the median. The average error (66.7%) is inflated by a few low-stream albums, above all *Stoney* (see Development Notes), so I rely on the median. R² on raw streams is 0.204.
- **2025 holdout (20 albums):** typical error 41.6% against 55.4% for the guess; average error 71.3% against 76.4%. Averaged the cross-validation way, 41.6% and 60.7%.
- **Bootstrap (5,000 resamples, 90% range):** the typical-error gap in the model's favor is 13.0 points on the holdout (25.6 lower to 1.3 higher) and 4.1 in cross-validation (14.8 lower to 2.8 higher). Both ranges include zero, and the model beats the guess on 54% of cross-validation albums. The holdout average-error gap (5.1 points, 10.3 to 0.2 lower) only just excludes zero and is driven by a few large misses.
- **Holdout noise:** correcting a small fallback leak (see Development Notes) moved the fallback from about 200M to 206M and the holdout average error from 62.4% to 73.4%. Read any holdout figure as accurate to within about 10 points.
- **Main failure, new artists with no history:** 15 of 20 holdout albums were over-predicted. The worst misses are Doja Cat's *Vie* (+259%), Addison Rae (+206%), Sleep Token (+155%), Central Cee (+122%), Olivia Dean (+119%) and sombr (+80%): the model predicts about 180M to 215M for albums that opened at 56M to 100M. The biggest under-predictions are megastars: Taylor Swift's *Life of a Showgirl* (60% under), Playboi Carti's *MUSIC* (43%) and Bad Bunny's *DeBÍ TiRAR MáS FOToS* (37%). The model regresses to the middle.
- **Unseen artists** (all of an artist's albums left out): typical error 41.8% against 45.3% for the guess.

I am keeping this as a documented framework, not a working forecaster. The Ariana and Harry Styles misses show where it breaks.

---

## Limitations

- **No proven edge over a naive baseline.** The headline limitation.
- **Debut and thin-history artists are over-predicted.** 34 of 91 training rows use the fallback (about 205M) for the history feature, and with one prior album the history signal is close to noise. The live app flags this case. All three 2026 case study predictions use real prior-album data.
- **Declining momentum.** Ariana Grande 2026 (+72.8%) and Harry Styles 2026 (+42.4%): the model over-credits historical scale and has no signal for a shrinking audience.
- **Era effects.** Adding older albums (2015 to 2022) made cross-validation error worse; listening habits and release strategies changed.
- **Release type.** Every training album is a conventional release where streams concentrate in week one, so slow burns are not represented. Surprise releases (Bieber's *Swag*, Taylor Swift's *Folklore* and *Evermore*) are not modeled accurately; `lead_time=1` is a workaround.
- **Narrow data.** US/UK superstar-tier artists, so the median reflects a major-label baseline, not the industry.
- **Narrative context** (label disputes, comebacks, tours) is unquantifiable, kept out of inputs, and never used to adjust predictions.
- **Sample size is the binding constraint.** 91 rows against 8 features is a data ceiling, not a tuning problem.
- **Two averaging methods.** Cross-validation averages trees in log space; the holdout and live app average each tree's stream estimate, which gives slightly higher predictions. Flagged, not changed.

---

## Data Integrity Policy

Every stream figure needs a verified source (chartmasters.org, Spotify Charts, kworb.net or Billboard). `is_household_name` flags are assigned by hand with a documented rationale. Predictions are never adjusted after the fact.

Case study predictions are recorded before each release and never recomputed, even though the revised model would give different figures today. Results that went against the model are documented, not removed.

---

## Model Revision, October 2026

After the summer 2026 case study I audited the model's inputs and found two problems.

1. **`hype_score` was scored with hindsight.** It was a hand-assigned 1 to 10 buzz rating and the model's most load-bearing input (removing it cost 0.071 of LOOCV R²). But every historical album was rated after release (I gave TTPD a 10 knowing it opened at 1,173M), so the feature partly recorded the answer. Cross-validation cannot catch this. I removed it and merged the three near-interchangeable history inputs into one.
2. **`lead_time` was mostly wrong.** I re-checked every value against its announcement and corrected them to one definition: days from the first official release-date announcement to the actual release. With the wrong values it looked useful; with the correct ones it is mostly noise.

I also moved *Hurry Up Tomorrow* to the 2025 holdout, since it came out that year.

| (48 training / 8 holdout albums) | Original | Revised |
|---|---|---|
| Features | 11 | 8 |
| LOOCV R² (log scale) | 0.495 | 0.432 |
| LOOCV MAPE | 40.6% | 43.9% |
| LOOCV median error | 29.0% | 29.2% |
| 2025 holdout MAPE | 39.2% | 35.6% |

The headline numbers got worse, which is the correct outcome: the old ones were propped up by a feature that had seen the answers and by wrong lead times. The logged case study predictions predate this revision and are kept as recorded.

---

## Development Notes

- **Dataset growth.** The data grew from 48 to 91 training albums, with every album meeting the rules added whatever the result. Rules: solo studio albums only (joint albums are not training rows but can serve as an artist's previous release), one source (chartmasters.org, Spotify global), and lead time counted from the first official announcement of the date the album actually came out on.
- **Platform-size adjustment.** Stream features are divided by MAU^0.25 instead of full MAU, because first-week streams grew far more slowly than the user base. This cut cross-validated typical error by about 4 points at 66 albums. At 91 albums the choice is within noise: no adjustment is now marginally ahead in cross-validation (39.3% against 39.9% typical error), and I kept 0.25 rather than change the model for a 0.6-point difference.
- **Fallback value.** Artists with no earlier album use the median of the training albums only (about 205M). An earlier version included holdout albums, which was a small leak and is fixed. 34 of 91 training rows use it.
- **Debut-artist flags.** A "has history" flag lowers typical error by about 2 points and average error by about 5 in cross-validation (38.2% against 40.0% typical). It is the lead candidate but is not adopted yet. Dropping `feature_track_count` is a small, inconsistent gain, so all 8 features stay.
- **Stoney is included.** Post Malone's 2016 debut (14.5M, confirmed) is by far the smallest album and the model misses it by about 560%. It meets the rules, so it stays. It raises average error (60.8% to 66.7%) but barely moves typical error (39.5% to 39.7%), which is why typical error is the metric I rely on.

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

- A "has history" flag or a separate treatment for debut artists, the model's largest source of error
- Broader artist coverage: more low-tier and new-artist albums, which the model over-predicts
- Update `app.py` to the platform-size adjustment and the larger dataset (the live app still uses the earlier method and the earlier, smaller training set) and report a typical-error range
- Additional pre-release signals to improve accuracy
- Re-benchmark RF vs XGBoost now that the dataset has passed 80 training rows
- Test a separate treatment for superstar releases and a surprise-release flag
- A larger 2025 to 2026 holdout
