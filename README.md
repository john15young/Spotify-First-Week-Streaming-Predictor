# Spotify First-Week Streaming Predictor
### Data-Driven Forecasting for High-Anticipation Album Releases

## Project Overview
This repository contains a machine learning model designed to forecast **first-week streaming performance** for major album releases. The project aims to move beyond manual speculation by using a `RandomForestRegressor` to provide quantitative, data-backed projections.

## The Objective: "Iceman"
The primary case study for this model is **Drake’s *Iceman***—widely considered one of the most anticipated albums of 2026. This project quantifies the "hype" factor by analyzing patterns in promotional timelines, market reach, and historical streaming performance.

## Data & Methodology
- **Data Source:** The model processes `albums.csv`, which aggregates key performance indicators (KPIs) such as:
    - Release rollout duration.
    - Market influence score.
    - Track count and promotional intensity.
- **Analysis:** By utilizing a Random Forest algorithm, the model identifies which variables are the strongest predictors of a massive first-week streaming debut.
- **Goal:** To provide a reliable, data-driven baseline for the commercial impact of *Iceman*.

## How to Run
1. Clone this repository: `git clone https://github.com/john15young/Spotify-First-Week-Streaming-Predictor.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `jupyter notebook Spotify_First_Week_Streaming_Predictor.ipynb`
- 