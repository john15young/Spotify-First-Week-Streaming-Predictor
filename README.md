# Predicting First-Week Spotify Streams with Random Forest

## Project Overview

A machine learning framework for forecasting first-week Spotify global streaming performance for major album releases. Uses a `RandomForestRegressor` trained on 49 verified historical albums (2015–2024), with Leave-One-Out Cross-Validation, MAU-normalized features, and a clean temporal holdout split. All stream figures verified via Spotify Charts.

The project's central test is a case study on three summer 2026 albums: each predicted ahead of its release, then checked against the actual first-week figure.

A live interactive version of this predictor is available at: **[spotify-first-week-streaming-predictor.streamlit.app](https://spotify-first-week-streaming-predictor.streamlit.app)**

---

## 2026 Case Study: Summer of Anticipation

| Artist | Album | Predicted | Actual (Wk1) | Error |
|---|---|---|---|---|
| Drake | Iceman | 479M | 450.4M | +6.4% (over) |
| Olivia Rodrigo | *you seem pretty sad for a girl so in love* | 413M | 394.7M | +4.7% (over) |
| Ariana Grande | Petal | 408M | 236.1M | +72.8% (over) |

Two of three landed within 7%. The third did not, and it is the most informative result here.

Drake and Olivia were both predicted ahead of release, from raw model output with no manual adjustment, and both landed well inside the error band despite very different release-week trajectories.

Ariana is a clear miss, kept here as a documented limitation rather than excluded. The model expected a return close to her historical baseline; the album opened at roughly two-thirds of that. It matches the over-prediction pattern recorded for Harry Styles below. The model has no signal for a shrinking audience.

A note on scope: the 2025 holdout of 7 albums is the model's real out-of-sample test. These three 2026 albums are a separate, smaller check made under live conditions.

---

## Data & Methodology

### Training Dataset

49 verified albums (2015–2024) across Pop, Hip-Hop, R&B, Latin, and Country. A further 7 albums from 2025 (including Justin Bieber's *Swag*) are held out for validation and never used in training.

| Field | Description |
|---|---|
| `spotify_mau` | Platform-wide Monthly Active Users. Annual figure from Spotify investor filings, identical for all albums in a given year |
| `lead_time` | Days from first announcement to release |
| `history_per_mau` | Artist history signal: prior album's first-week streams and career average, MAU-normalized |
| `is_household_name` | Binary flag for global superstar name recognition |
| `first_week_streams` | Target. First-week Spotify global streams (M) |

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

`RandomForestRegressor` (`n_estimators=200, max_depth=4, min_samples_leaf=3`), selected after head-to-head testing against Ridge, Lasso, ElasticNet, SVR, GBM, and XGBoost. Ridge posted a higher LOOCV R², but **RF won on holdout MAPE**, which was the deciding metric because holdout reflects true generalization. Constrained depth and leaf size intentionally prevent overfitting on 49 training rows.

XGBoost underperformed RF at the current dataset size. Will be revisited at 80+ training rows.

**Top features**: `history_per_mau` (56%), `spotify_mau` (19%), `lead_time` (6%). The model is fundamentally answering "how big is this artist, and how big was the platform," which is appropriate for a small, outlier-heavy dataset.

---

## Performance

| Metric | Value |
|---|---|
| LOOCV R² (log scale) | 0.483 |
| LOOCV MAPE | 40.0% |
| LOOCV median error | 22.5% |
| 2025 Holdout MAPE | 25.8% (7 unseen albums) |
| 2026 Case Study: Drake Iceman | +6.4% over |
| 2026 Case Study: Olivia Rodrigo | +4.7% over |
| 2026 Case Study: Ariana Grande Petal | +72.8% over (documented limitation) |
| 2026 Validation: Harry Styles KissAllTheTime | +60.7% over (documented limitation) |

Holdout MAPE (25.8%) beating LOOCV MAPE (40.0%) suggests the model generalizes rather than memorizes, though with only 7 holdout albums that gap carries a wide confidence interval.

The 2026 row is a single album, not a validation set. One observation cannot estimate an error rate; it is logged as a data point, and the 2025 holdout is the real out-of-sample test.

### Reading the R² honestly

The 0.483 is measured on `log(streams / MAU)`, so it scores how well the model places an album relative to the platform, not its raw stream count. On the raw scale the same predictions score far lower, because a few outsized albums (TTPD at 1,173M) dominate the variance and the model does not reach them.

Against naive baselines, the model beats "always guess the dataset median" by a modest margin on LOOCV MAPE. With 49 rows and 8 features, that margin is suggestive rather than decisive. The median error of 22.5% describes everyday behavior better than the mean, which the Taylor Swift outliers drag upward.

---

## Limitations

- Some training rows fall back to the dataset median (~285M) for the history feature, since no earlier album exists for that artist yet. This is standard handling for first-observed entries in a small dataset. All three 2026 case study predictions use real prior-album data, not the fallback.
- Superstar-only dataset. The median reflects a major-label superstar baseline, not the music industry at large.
- **Over-prediction for artists with declining momentum.** Ariana Grande 2026 (+72.8%) and Harry Styles 2026 (+60.7%) are the clearest cases. The model over-credits historical scale with no signal for a shrinking audience. This is the single largest known weakness.
- **Over-prediction for artists with thin history.** With only one prior album, the history signal is close to noise. The live app now flags this case before showing a prediction.
- **Front-loaded release assumption.** Every training album is a conventional release where streams concentrate in week one. Slow-burn releases that build over later weeks are not represented, and the model cannot anticipate them.
- Surprise releases (Bieber's *Swag*, Taylor Swift's *Folklore*/*Evermore*) are not modeled accurately. `lead_time=1` is used as a workaround. An `is_surprise` binary feature is planned once the dataset supports it.
- Narrative context (label disputes, comebacks, tour momentum) is unquantifiable and kept out of model inputs. Documented separately, never used to adjust predictions.
- Dataset skews toward US/UK superstar-tier artists. Expansion to mid-tier and non-English releases is in progress.
- **Sample size is the binding constraint.** 49 training rows against 8 features limits what any algorithm can extract. This is a data ceiling, not a tuning problem.

---

## Data Integrity Policy

All stream figures require a verified source (Spotify Charts, kworb.net, or Billboard) before inclusion. `is_household_name` flags are manually assigned with documented rationale. Predictions are never adjusted post hoc to match a desired outcome.

Case study predictions are recorded ahead of each release and never recomputed. The model has since been revised and would produce different figures today. The logged numbers are kept as a record of what was actually predicted. Results that went against the model are documented rather than removed.

---

## Model Revision, October 2026

After the summer 2026 case study, I audited the model's inputs and found a problem with its strongest one.

`hype_score` was a 1 to 10 rating of pre-release buzz that I assigned by hand. It was the model's most load-bearing input: removing it cost 0.042 of LOOCV R², more than double the next feature (`lead_time`, 0.019) and far ahead of everything else.

The problem was when I scored it. All 49 historical albums were rated after they had already come out. I gave TTPD a 10 knowing it had opened at 1,173M. So the feature was not purely measuring anticipation. It was partly recording the answer.

Cross-validation does not catch this. Holding an album out hides its result from the model, but does nothing about hindsight already baked into that album's inputs. Validation looked clean because the problem sat upstream of it.

The test that confirmed it:

| Feature set | LOOCV R² | LOOCV MAPE | 2025 holdout MAPE |
|---|---|---|---|
| Original, 11 features | 0.519 | 38.2% | 25.4% |
| `hype_score` removed, 10 features | 0.477 | 40.8% | **22.4%** |

Dropping it made the cross-validated numbers worse but the held-out year better. A feature that helps on data scored after the fact, and hurts on data the model never saw, is leaking.

So I removed it. I also merged three history inputs into one. Ablation showed they were near-interchangeable: removing `trailing_avg_per_mau` moved R² by 0.003, and removing either `prev_streams_per_mau` or `trailing_spt_per_mau` slightly *improved* it. Whichever one was dropped, the others absorbed the work, which is what three versions of the same signal look like.

Where that left the model:

| | Original | Revised |
|---|---|---|
| Features | 11 | 8 |
| LOOCV R² (log scale) | 0.519 | 0.483 |
| LOOCV MAPE | 38.2% | 40.0% |
| LOOCV median error | 22.5% | 22.5% |
| 2025 holdout MAPE | 25.4% | 25.8% |

The headline numbers got slightly worse, which is the correct outcome. The old ones were propped up by a feature that had seen the answers.

One honest trade-off: if I had removed only `hype_score` and kept the three history inputs separate, the model scored slightly better on the 2025 test albums (22.4% error vs 25.8%). I merged them anyway because they were measuring the same thing, and a simpler model is easier to trust. With only 7 test albums, a gap that small could easily be luck, so I didn't treat it as a reason to keep the extra inputs.

The case study was unaffected. Those three albums had not been released when I scored them, so no hindsight was available to apply. Their predictions are left exactly as recorded.

The cost is that the revised model is cleaner but unproven. It has not yet predicted an album ahead of release. The next major release is the test.

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

- Add mid-tier artists (80–200M first-week range) to address upward prediction bias
- Add non-English superstars (BTS, Peso Pluma) for genre/language diversity
- Add a momentum feature (direction of change across an artist's last two releases) to address the largest known weakness
- Implement `is_surprise` as a proper binary feature
- Re-benchmark RF vs XGBoost at 80+ training rows
