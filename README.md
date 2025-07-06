# 📊 Day 13 – Growth Metrics Tracker

## Overview
This project is designed to help businesses monitor their key growth KPIs, track CLV vs CAC trends, analyze cohort retention, and predict churn using real-time metrics.

## Objective
- Visualize trends in Revenue, CAC, CLV, Retention, and Growth.
- Trigger KPI alerts for CLV/CAC ratio, retention, and churn thresholds.
- Compare monthly vs quarterly metrics and cohort retention.
- Integrate NPS trends and apply churn prediction using logistic regression.

## Dataset
The dataset includes the following columns:
- `Month`, `New_Customers`, `ChurnedCustomers`, `ActiveCustomers`
- `ARPU`, `MRR`, `Revenue`, `Cost`, `CAC`, `CLV`, `NPS`
- `ChurnRate`, `Cohort`, `GrowthRate`, `RetentionRate`, `CLV_CAC_Ratio`, `Quarter`

## Technologies Used
- Python
- pandas, seaborn, matplotlib, scikit-learn

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python growth_metrics_tracker.py
```

## Features
- 📉 Metric trend visualizations
- ⚠️ Automated KPI alerts
- 📆 Monthly vs quarterly comparison
- 📊 Monthly cohort retention
- 🌟 NPS line plot analysis
- 🔮 Logistic regression for churn prediction

## Future Enhancements
- Real-time dashboard (Streamlit)
- Segment-wise NPS breakdown
- Custom KPI benchmarking
