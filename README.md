# Spotify First-Week Streaming Predictor
### Data-Driven Forecasting for High-Anticipation Album Releases

## Project Overview
This repository contains a machine learning model designed to forecast **first-week streaming performance** for major album releases. By moving beyond manual speculation, the project utilizes a `RandomForestRegressor` to analyze the intersection of market growth, artist hype, and historical streaming behavior to provide quantitative, data-backed projections.

## 2026 Case Study: The "Summer of Titans"
As of April 2026, the model is currently focused on the highly anticipated release cycle featuring three industry heavyweights:
- **Drake – *Iceman*:** Analyzing his bid to reclaim hip-hop supremacy following recent cryptic teasers and a dominant start to the 2026 streaming year.
- **Ariana Grande – *AG8*:** Forecasting her return to pop dominance based on recent studio confirmations and summer tour synergy.
- **Lana Del Rey – *Stove*:** Predicting the commercial impact of her pivot into country music, factoring in her transition to a "High-Engagement" streaming tier.

## Data & Methodology
- **The Feature Set:** The model processes `albums.csv`, aggregating key performance indicators (KPIs) such as:
    - **Track Count:** The total number of unique assets available for streaming.
    - **Spotify MAU:** Global Monthly Active Users (currently adjusted to the 2026 benchmark of **751M**).
    - **Hype Score:** A weighted variable (1.0–10.0) quantifying social sentiment and promotional intensity.
    - **Rollout Duration:** The length of the marketing campaign prior to release.
- **Analysis:** By utilizing a **Random Forest** algorithm, the model identifies non-linear relationships between market size and track volume, identifying the streaming "ceiling" for different artist tiers.
- **Goal:** To provide a reliable, data-driven baseline for the commercial impact of major releases in an era of massive market expansion.

## How to Run
1. **Clone this repository:**
   ```bash
   git clone [https://github.com/john15young/Spotify-First-Week-Streaming-Predictor.git](https://github.com/john15young/Spotify-First-Week-Streaming-Predictor.git)