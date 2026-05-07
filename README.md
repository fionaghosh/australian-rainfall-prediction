# 🌧️ Predicting Next-Day Rainfall Across Australia
### Using Machine Learning on a Decade of Bureau of Meteorology Weather Data

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔗 Live Dashboard
👉 **[Click here to open the interactive dashboard](https://australian-rainfall-prediction-v8mzgyv4ry6gnrnb2ajqxd.streamlit.app)**

---

## 📌 Project Overview

This project uses **10 years of daily weather observations (2008–2018)** from
49 Australian weather stations to predict whether it will rain the next day.
The dataset is sourced from the Bureau of Meteorology via Kaggle and contains
142,193 records across 23 features.

The project goes beyond standard classification modelling to include
explainability, geographic performance analysis, probability calibration,
and a fully deployed interactive dashboard.

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| Best Model | XGBoost |
| AUC-ROC | 0.8905 |
| Optimal Threshold | 0.64 |
| False Alarm Reduction | 43% |
| Top Predictor | Humidity at 3pm |
| Best City | Perth Airport (AUC 0.9561) |
| Hardest City | Norfolk Island (AUC 0.8032) |
| Calibration Improvement | +16.2% via Platt Scaling |

---

## 📁 Project Structure

```plaintext
australian-rainfall-prediction/
│
├── app.py              # Streamlit dashboard (5 pages)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 🔍 Project Sections

### 1. Exploratory Data Analysis
- Class imbalance analysis — 77.6% No Rain vs 22.4% Rain
- Missing value profiling across all 23 columns
- Rainfall distribution and top rainy cities visualised
- Seasonal and geographic patterns across 49 stations

### 2. Data Cleaning & Preprocessing
- Dropped high-missing columns — Sunshine (48%) and Evaporation (43%)
- Per-location median imputation to preserve geographic context
- Global median fallback for locations with fully missing columns
- Date decomposition into Month, Year, Season and Day of Year
- One-hot encoding of wind direction and season features
- Label encoding of 49 city names

### 3. Model Training & Comparison
- Four models compared — Logistic Regression, Random Forest, XGBoost, LightGBM
- Stratified 80/20 train-test split preserving class balance
- Class weight balancing applied to handle 77/22 imbalance
- AUC-ROC used as primary metric — not raw accuracy

| Model | AUC-ROC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| XGBoost | 0.8905 | 0.6612 | 0.5742 | 0.7791 |
| LightGBM | 0.8866 | 0.6483 | 0.5502 | 0.7890 |
| Random Forest | 0.8849 | 0.5928 | 0.7837 | 0.4767 |
| Logistic Regression | 0.8644 | 0.6203 | 0.5170 | 0.7751 |

### 4. SHAP Explainability
- TreeExplainer applied to XGBoost on 2,000 test samples
- Global feature importance bar chart
- Beeswarm plot showing direction and magnitude of each feature
- Waterfall plot explaining individual rainy day predictions

**Top 5 Predictors:**

| Rank | Feature | Mean SHAP Value |
|------|---------|----------------|
| 1 | Humidity3pm | 1.103 |
| 2 | Pressure3pm | 0.613 |
| 3 | WindGustSpeed | 0.454 |
| 4 | Cloud3pm | 0.264 |
| 5 | MinTemp | 0.219 |

### 5. Threshold Optimisation
- Default threshold of 0.50 replaced with optimal threshold of 0.64
- F1 score improved from 0.6612 to 0.6690
- False alarms reduced from 3,683 to 2,105 — a 43% reduction
- Precision-Recall curve used to identify the optimal cutoff point

| Setting | F1 Score | Precision | Recall |
|---------|----------|-----------|--------|
| Default (0.50) | 0.6612 | 0.5742 | 0.7791 |
| Optimal (0.64) | 0.6690 | 0.6694 | 0.6685 |

### 6. City-Level Performance Analysis
- Model evaluated individually across all 49 Australian cities
- 42 out of 49 cities achieved AUC above 0.85
- Geographic performance mapped against local rain frequency
- Arid inland cities easiest to predict — tropical coastal cities hardest

| | City | AUC-ROC |
|--|------|---------|
| Best | Perth Airport | 0.9561 |
| Mean | All 49 Cities | 0.8851 |
| Hardest | Norfolk Island | 0.8032 |

### 7. Rainfall Trend Analysis 2008–2018
- Annual rainfall frequency declined from 31.1% in 2007 to 20.9% in 2017
- 2014 was the driest year on record at just 20.4% rainy days
- Winter months June–August consistently wettest at around 27% rain frequency
- Afternoon humidity rose from 45.8% to 51.9% despite declining rainfall
- Maximum temperatures showed a slight upward trend across the decade

### 8. Probability Calibration
- Raw XGBoost found to be systematically overconfident across all probability bins
- Platt scaling applied to correct probability estimates
- Brier score improved 16.2% from 0.1263 to 0.1058
- Calibrated model used in all dashboard probability outputs

| Model | Brier Score |
|-------|-------------|
| XGBoost Raw | 0.1263 |
| XGBoost Calibrated | 0.1058 |
| Perfect Calibration | 0.0000 |

---

## 🗺️ Interactive Dashboard Pages

| Page | Description |
|------|-------------|
| Home | Project overview, key metrics and results summary |
| EDA Dashboard | City selector with seasonal rain and temperature charts |
| Predict Tomorrow's Rain | Input today's weather and get a next-day rain probability |
| Australia Rain Map | Interactive Folium map of predicted rain probability across all 49 stations |
| Model Performance | Full model comparison, SHAP rankings, calibration results |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Visualisation | Matplotlib, Seaborn, Folium |
| Dashboard | Streamlit |
| Hosting | Streamlit Community Cloud |
| Environment | Google Colab |

---

## 📦 Running Locally

```bash
# Clone the repository
git clone https://github.com/fionaghosh/australian-rainfall-prediction.git
cd australian-rainfall-prediction

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

---

## 📂 Dataset

| Field | Detail |
|-------|--------|
| Source | [Kaggle — Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) |
| Original Source | Australian Bureau of Meteorology |
| Period | 2008–2018 |
| Weather Stations | 49 across Australia |
| Raw Records | 145,460 |
| Records After Cleaning | 142,193 |

---

## 👩‍💻 Author

**Fiona Ghosh**

[![GitHub](https://img.shields.io/badge/GitHub-fionaghosh-black)](https://github.com/fionaghosh)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-fiona--ghosh-blue)](https://linkedin.com/in/fiona-ghosh)

---

## 📄 License
This project is licensed under the MIT License.

---
