# Data-Driven Forecasting for High-Anticipation Album Releases

## Project Overview
This repository contains a machine learning framework designed to forecast **first-week streaming performance** for major album releases. By moving beyond manual speculation, the project utilizes a `RandomForestRegressor` to analyze the intersection of market growth, artist hype, and historical streaming behavior to provide quantitative, data-backed projections.

## 2026 Case Study: The "Summer of Titans"
As of April 2026, the model is focused on the highly anticipated release cycle featuring three industry heavyweights:
* **Drake — *Iceman*:** Analyzing his bid to reclaim hip-hop supremacy following a dominant start to the 2026 streaming year.
* **Olivia Rodrigo — *OR3*:** Forecasting her "Efficiency King" status and whether her high Streams-Per-Track (SPT) ratio can break records with the 2026 MAU surge.
* **Ariana Grande — *AG8*:** Predicting the commercial impact of her return to pop dominance factoring in recent studio confirmations.

## Data & Methodology

### **The Feature Set ($X$)**
The model processes `streaming_data_final.csv`, aggregating key performance indicators (KPIs) including:
* **Track Count:** Total songs available for streaming (influencing volume vs. efficiency).
* **Spotify MAU:** Global Monthly Active Users (adjusted to the 2026 benchmark of **751M**).
* **Hype Score:** A weighted variable (1.0–10.0) quantifying social sentiment and promotional intensity.
* **Days to Release:** The length of the marketing rollout.
* **Event Type:** A binary toggle for standard vs. "surprise" or "event-style" releases.
* **Feature Track Count:** The volume of high-profile collaborations on the project.

### **The Machine Learning Model**
The system utilizes a **Random Forest** algorithm, which identifies non-linear relationships between market size and track volume. It employs an ensemble of 100 decision trees to average out predictions, identifying the streaming "ceiling" for different artist tiers while accounting for "Streaming Inflation" caused by Spotify's global user growth.

### **Goal**
To provide a reliable, data-driven baseline for the commercial impact of major releases in an era of massive market expansion.