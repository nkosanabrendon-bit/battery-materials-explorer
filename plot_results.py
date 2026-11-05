import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Load and prepare data
df = pd.read_csv("li_batteries.csv")
df = df.dropna(subset=['voltage_V', 'capacity_mAh_g', 'energy_Wh_kg', 'volume_change_pct'])

X = df[['voltage_V', 'capacity_mAh_g', 'volume_change_pct']]
y = df['energy_Wh_kg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3. Generate Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='royalblue', edgecolor='k', s=30)

# Plot the ideal 1:1 line (where Actual == Predicted)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label="Ideal Fit (y = x)")

plt.xlabel("Actual Energy Density (Wh/kg)", fontsize=12)
plt.ylabel("Predicted Energy Density (Wh/kg)", fontsize=12)
plt.title("Random Forest Performance: Actual vs. Predicted", fontsize=14)
plt.legend(loc="upper left")
plt.grid(True, linestyle='--', alpha=0.5)

# Save and show
plt.tight_layout()
plt.savefig("actual_vs_predicted_energy.png", dpi=300)
print("Success! Plot saved locally as 'actual_vs_predicted_energy.png'.")
