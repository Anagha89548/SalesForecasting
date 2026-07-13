# 📊 Sales Forecasting & Demand Analytics Dashboard

An end-to-end Data Analytics and Machine Learning project that forecasts retail sales, detects anomalies, segments product demand, and presents business insights through an interactive Streamlit dashboard.

---

## 📌 Project Overview

This project analyzes historical retail sales data to help businesses improve inventory planning and decision-making. It combines exploratory data analysis, time series forecasting, anomaly detection, demand segmentation, and dashboard visualization into a single solution.

---

## 🚀 Features

- 📈 Sales trend analysis
- 📅 Monthly and yearly sales visualization
- 🤖 Sales forecasting using Prophet
- ⚠️ Anomaly detection using Isolation Forest and Z-Score
- 📦 Product demand segmentation using K-Means Clustering
- 📊 Interactive Streamlit dashboard
- 🌍 Region and category-based sales analysis

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-learn
- Prophet
- Statsmodels
- Streamlit
- KaggleHub

---

## 📂 Project Structure

```
Sales_Forecasting/
│
├── app.py
├── train.csv
├── requirements.txt
├── README.md
│
├── pages/
│   ├── 1_Sales_Overview.py
│   ├── 2_Forecast_Explorer.py
│   ├── 3_Anomaly_Report.py
│   └── 4_Product_Demand_Segments.py
│
├── notebooks/
│   └── Sales_Forecasting.ipynb
│
└── images/
```

---

## 📊 Machine Learning Models

### Sales Forecasting
- SARIMA
- Prophet ✅ (Best Performing Model)
- XGBoost

### Anomaly Detection
- Isolation Forest
- Z-Score

### Demand Segmentation
- K-Means Clustering
- PCA Visualization

---

## 📈 Dashboard Pages

### 1️⃣ Sales Overview
- Total Sales KPI
- Sales by Year
- Monthly Sales Trend
- Region Filter
- Category Filter

### 2️⃣ Forecast Explorer
- Category Forecast
- Region Forecast
- 1–3 Month Forecast
- Forecast Accuracy (MAE & RMSE)

### 3️⃣ Anomaly Report
- Sales Anomaly Visualization
- Detected Anomaly Dates
- Sales Values

### 4️⃣ Product Demand Segments
- Cluster Visualization
- Demand Group Table
- Stocking Recommendations

---

## 📋 Business Insights

- Prophet produced the highest forecasting accuracy.
- Technology products show the strongest future demand.
- The West region is expected to generate higher future sales.
- Promotional periods create significant sales spikes.
- Demand segmentation helps optimize inventory planning.

---

## 📊 Dataset

Dataset:
**Superstore Sales Forecasting Dataset**

Source:
https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting

---

## 👨‍💻 Author

**ANAGHA C.ANTO**

B.Tech Computer Science Engineering

Data Analytics | Data Science | Machine Learning

---

## 📄 License

This project is developed for educational and learning purposes.
