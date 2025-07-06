import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ------------------------------
# 1. Load Final Dataset
# ------------------------------
df = pd.read_csv("Growth_Metrics_Dataset.csv", parse_dates=['Month'])
df = df.sort_values("Month")

# ------------------------------
# 2. 📉 Trend Visualizations
# ------------------------------
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Month', y='Revenue', label='Revenue')
sns.lineplot(data=df, x='Month', y='Cost', label='Cost')
plt.title("Revenue vs Cost Over Time")
plt.xlabel("Month"); plt.ylabel("USD"); plt.legend(); plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Month', y='New_Customers', label='New Users')
sns.lineplot(data=df, x='Month', y='ActiveCustomers', label='Active Users')
plt.title("User Growth vs Active Users")
plt.xlabel("Month"); plt.ylabel("Users"); plt.legend(); plt.tight_layout()
plt.show()

# ------------------------------
# 3. 🧮 KPI Tracker Summary
# ------------------------------
kpi_summary = df[['Month', 'Revenue', 'New_Customers', 'ActiveCustomers',
                  'CLV', 'CAC', 'CLV_CAC_Ratio', 'ARPU', 'RetentionRate', 'ChurnRate']]
print("\n🧾 KPI Snapshot:\n", kpi_summary.tail())

# ------------------------------
# 4. 💸 CLV vs CAC Ratio
# ------------------------------
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Month', y='CLV', label='CLV')
sns.lineplot(data=df, x='Month', y='CAC', label='CAC')
plt.title("CLV vs CAC Over Time"); plt.xlabel("Month"); plt.ylabel("USD")
plt.legend(); plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.lineplot(data=df, x='Month', y='CLV_CAC_Ratio', color='green')
plt.axhline(1, color='red', linestyle='--')
plt.title("CLV/CAC Ratio Threshold"); plt.ylabel("Ratio")
plt.tight_layout(); plt.show()

# ------------------------------
# 5. 🔁 Retention vs Growth
# ------------------------------
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Month', y='GrowthRate', label='Growth Rate')
sns.lineplot(data=df, x='Month', y='RetentionRate', label='Retention Rate')
plt.title("Growth vs Retention"); plt.xlabel("Month"); plt.ylabel("Rate")
plt.legend(); plt.tight_layout(); plt.show()

# ------------------------------
# 6. ⚠️ Automated KPI Alerts
# ------------------------------
alerts = []
for i, row in df.iterrows():
    if row['CLV_CAC_Ratio'] < 1:
        alerts.append(f"⚠️ {row['Month'].strftime('%b %Y')}: CLV/CAC ratio below 1.")
    if row['RetentionRate'] < 0.5:
        alerts.append(f"⚠️ {row['Month'].strftime('%b %Y')}: Retention below 50%.")
    if row['ChurnRate'] > 0.3:
        alerts.append(f"⚠️ {row['Month'].strftime('%b %Y')}: High churn rate ({row['ChurnRate']:.2f}).")

print("\n🔔 KPI Alerts:\n" + "\n".join(alerts))

# ------------------------------
# 7. 📆 Monthly vs Quarterly View
# ------------------------------
df['Quarter'] = df['Month'].dt.to_period("Q")

quarterly = df.groupby('Quarter').agg({
    'Revenue': 'sum',
    'New_Customers': 'sum',
    'CLV_CAC_Ratio': 'mean',
    'RetentionRate': 'mean'
}).reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(data=quarterly, x='Quarter', y='Revenue')
plt.title("Quarterly Revenue"); plt.tight_layout(); plt.show()

# ------------------------------
# 8. 📊 Monthly Cohort Retention
# ------------------------------
cohorts = df[['Month', 'New_Customers', 'ActiveCustomers']].copy()
cohorts['CohortMonth'] = cohorts['Month'].dt.to_period('M')
cohorts['RetentionPercent'] = cohorts['ActiveCustomers'] / cohorts['New_Customers']

plt.figure(figsize=(10, 5))
sns.barplot(data=cohorts, x='CohortMonth', y='RetentionPercent')
plt.title("Monthly Retention Rate"); plt.ylabel("Retention %")
plt.xticks(rotation=45); plt.tight_layout(); plt.show()

# ------------------------------
# 9. 🌟 NPS Analysis
# ------------------------------
plt.figure(figsize=(8, 4))
sns.lineplot(data=df, x='Month', y='NPS', marker='o')
plt.axhline(50, color='green', linestyle='--')
plt.title("Net Promoter Score (NPS)")
plt.tight_layout(); plt.show()

# ------------------------------
# 10. 🔮 Churn Prediction (Logistic Regression)
# ------------------------------
df['HighChurnNextMonth'] = df['ChurnRate'].shift(-1) > 0.3
df.dropna(inplace=True)

features = ['Revenue', 'Cost', 'CLV', 'CAC', 'RetentionRate']
X = df[features]
y = df['HighChurnNextMonth'].astype(int)

if y.nunique() < 2:
    print("\n⚠️ Not enough class diversity for churn prediction model (only one class present).")
    print("   Try updating the dataset to include more variation in 'ChurnRate'.")
else:
    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    print("\n🔮 Churn Prediction Report:\n")
    print(classification_report(y, y_pred, target_names=["Low Risk", "High Risk"]))

