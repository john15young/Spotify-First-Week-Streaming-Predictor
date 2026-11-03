# Predicting First-Week Spotify Streams with Ridge Regression

## Project Overview

A machine learning framework for forecasting first-week Spotify global streaming performance for major album releases. Uses a Ridge regression trained on 91 verified historical albums (2015 to 2024), with Leave-One-Out Cross-Validation, platform-size-adjusted features, a debut-artist flag, and a temporal holdout of 20 albums from 2025. All stream figures verified via Spotify Charts.

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
| `has_hist` | 1 if a real previous-album first-week figure exists for the artist, 0 if the fallback value stands in for one (debuts, new artists, unknown priors) |
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

Ridge regression on standardized features (alpha 1.0, scikit-learn's default, fixed in advance and not tuned), predicting the log target above. I compared it with a Random Forest (200 trees, depth 4, leaf size 3), Lasso and ElasticNet, and chose by leave-one-out cross-validation alone.

Before the debut flag existed, the Random Forest was ahead by a small margin (typical error 39.7% against 41.2% to 42.0% for the linear models). The flag changed that. It lowers the forest's typical error only a little but takes Ridge from 41.2% to 32.1%. A linear model can apply one percentage drop to every album with no history, while a tree needs enough albums in a leaf to learn it.

| Model, 9 features | LOO typical | LOO average | Leave-one-artist-out typical | 2025 holdout typical | 2025 holdout average |
|---|---|---|---|---|---|
| Ridge | 32.1% | 58.8% | 33.4% | 26.1% | 41.9% |
| Random Forest (three-seed average) | 38.9% | 61.0% | 39.5% | 40.1% | 63.6% |
| "Always guess the median" | 43.8% | 85.6% | 45.3% | 55.4% | 76.4% |

Lasso (36.0%) and ElasticNet (33.9%) fall between the two in leave-one-out typical error. Ridge is not sensitive to its one setting: alpha from 0.1 to 10 gives a typical error of 32.9% to 36.6%. XGBoost and gradient boosting underperformed at 66 rows and have not been re-tested with the flag.

**Weights** (standardized coefficients; positive raises the prediction): `has_hist` +0.31, household name × tracks +0.21, `feature_track_count` -0.12, `track_count` +0.11, `history_per_mau` +0.09, `spotify_mau` -0.06, `is_household_name` -0.05, `years_since_last_album` +0.03, `lead_time` -0.03. In plain terms the model is mostly asking whether the artist has a track record and how big their name is. `lead_time` has almost no weight, but dropping it still costs about 1.5 points, so it stays.

---

## Performance

| Metric | Model | "Always guess the median" baseline |
|---|---|---|
| LOOCV MAPE (average error) | 58.8% | 85.6% |
| LOOCV median error | 32.1% | 43.8% |
| Leave-one-artist-out median error | 33.4% | 45.3% |
| LOOCV R² on raw streams | 0.375 | n/a |
| 2025 holdout MAPE (20 albums) | 41.9% | 76.4% |
| 2025 holdout median error | 26.1% | 55.4% |

| Case study / validation | Result |
|---|---|
| 2026 Drake Iceman | +6.3% over |
| 2026 Olivia Rodrigo | +4.6% over |
| 2026 Ariana Grande Petal | +72.8% over (documented limitation) |
| 2026 Harry Styles KissAllTheTime | +51.1% over (documented limitation) |

The three case study rows are the logged predictions from the original model. The Harry Styles row is recomputed live with the current model.

### Reading the numbers honestly

**The model beats a simple guess in every test, and for the first time the 90% bootstrap ranges exclude zero, but typical errors are still about a third and the holdout is small.**

- **Cross-validation (91 albums):** typical error 32.1% against 43.8% for guessing the median; average error 58.8% against 85.6%. The average is pulled up by a few low-stream albums (*Stoney* 647%, *Emails I Can't Send* 536%, J Balvin's *Jose* 274%), so I rely on the median. R² on raw streams is 0.375.
- **2025 holdout (20 albums):** typical error 26.1% against 55.4% for the guess; average error 41.9% against 76.4%. 14 of 20 were over-predicted.
- **Bootstrap (5,000 resamples, 90% range):** the typical-error gap in the model's favor is 10.5 points in cross-validation (17.7 to 0.1 lower, so only just excluding zero) and 27.9 on the holdout (62.1 to 7.3 lower). The average-error gaps are 26.0 (44.7 to 11.5 lower) and 34.0 (54.1 to 17.2 lower). The model beats the guess on 60% of cross-validation albums.
- **How much to trust the holdout:** the model and every setting were chosen by cross-validation, not the holdout. But I looked at holdout numbers throughout development, so treat it as confirmation, not a clean test. It is only 20 albums and includes a few extreme ones, so read it as accurate to within about 10 points.
- **The debut flag fixed most of the new-artist problem.** Before it, the model over-predicted most debut and small artists by 2x or more. With it, Olivia Dean is +5%, Central Cee +6%, sombr +6%, Alex Warren -7% (under) and Sleep Token +27%. Addison Rae is still +72%.
- **Remaining failures:** artists with a known previous album that opened far below it. Doja Cat's *Vie* is +274% (previous album 124.5M, this one 56M), Tyler, the Creator +77% and Karol G +55%. On the other side, the very biggest releases are under-predicted: Taylor Swift's *Life of a Showgirl* (55% under), Playboi Carti's *MUSIC* (54%) and Bad Bunny's *DeBÍ TiRAR MáS FOToS* (40%). The model pulls everything toward the middle.
- **Unseen artists** (all of an artist's albums left out): typical error 33.4% against 45.3% for the guess.

I am presenting this as a modest, documented forecaster: a typical miss of about a third, against about 44% for guessing. The Ariana and Harry Styles misses show where it breaks.

---

## Limitations

- **Errors are still large.** The typical miss is about a third of the true figure, and average error is higher because a few albums are badly missed.
- **Regression to the middle.** The very biggest releases are under-predicted and small, surprising ones over-predicted.
- **The debut flag is crude.** It is binary. It says whether a previous album exists, not whether this release will resemble it (Doja Cat's *Vie*).
- **Declining momentum.** Ariana Grande 2026 (+72.8%) and Harry Styles 2026 (+51.1%): the model over-credits historical scale and has no signal for a shrinking audience.
- **Influential rows.** In a linear model, single extreme albums matter. Without *Stoney* the cross-validated typical error is 35.6% against 43.5% for the guess, so the edge survives, but the number depends on that one row.
- **Era effects.** Adding older albums (2015 to 2022) made cross-validation error worse; listening habits and release strategies changed.
- **Release type.** Every training album is a conventional release where streams concentrate in week one, so slow burns are not represented. Surprise releases (Bieber's *Swag*, Taylor Swift's *Folklore* and *Evermore*) are not modeled accurately; `lead_time=1` is a workaround.
- **Narrow data.** US/UK superstar-tier artists, so the median reflects a major-label baseline, not the industry.
- **Narrative context** (label disputes, comebacks, tours) is unquantifiable, kept out of inputs, and never used to adjust predictions.
- **Sample size.** 91 rows against 9 features, and a model that assumes effects add up in log space.

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

- **Dataset growth.** The data grew from 48 to 91 training albums, with every album meeting the rules added whatever the result. Rules: solo studio albums only (joint albums are not training rows but can serve as an artist's previous release), one source (chartmasters.org, Spotify global), and lead time counted from the announcement of the release date the album actually came out on (for a postponed album, the announcement of the final date).
- **Platform-size adjustment.** Stream features are divided by MAU^0.25 instead of full MAU, because first-week streams grew far more slowly than the user base. Cross-validated typical error for Ridge by exponent: 33.7% at 0, **32.1% at 0.25**, 33.2% at 0.5, 32.7% at 0.75, 33.0% at 1. All five sit within 1.6 points, so the choice matters little. With the earlier Random Forest, 0 and 0.25 traded places as the dataset grew.
- **Fallback value.** Artists with no earlier album use the median of the training albums only (about 205M). An earlier version included holdout albums, which was a small leak and is fixed. 34 of 91 training rows use it.
- **Debut-artist flag.** `has_hist` was added because cross-validation said it helped at two dataset sizes (Random Forest typical error 38.8% to 38.0% at 90 albums, 40.0% to 38.2% at 91). With Ridge it was far larger (41.2% to 32.1%), which is what led to the model change.
- **One prediction method.** An earlier Random Forest version averaged its trees one way in cross-validation and another way in the app, which gave slightly higher live predictions. Ridge has no such mismatch: one prediction in log space, used everywhere.
- **Feature removal** (Ridge, leave-one-out typical error): all 9 features 32.1%; without `lead_time` 33.6%; without `feature_track_count` 36.6%; without `years_since_last_album` 33.2%; without `spotify_mau` 33.4%; without `is_household_name` 32.7% (a tie). All 9 stay.
- **Stoney is included.** Post Malone's 2016 debut (14.5M, confirmed) is by far the smallest album, and the model's cross-validated prediction is 108M, a miss of about 650%. It meets the rules, so it stays. It raises the average error (53.2% without it, 58.8% with) and lowers the typical error (35.6% without, 32.1% with), because it is a clean example of the no-history effect.

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

- Broader artist coverage: more low-tier and new-artist albums
- A softer version of the debut flag, for example prior-album size relative to the artist's other albums
- Re-test XGBoost and gradient boosting now that the dataset has passed 90 training rows and has the debut flag
- Additional pre-release signals (for example lead-single streams) to address under-prediction of the biggest releases
- A surprise-release flag
- A larger 2026 holdout
