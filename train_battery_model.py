import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
print("Loading battery dataset...")
df = pd.read_csv("li_batteries.csv")
print(f"Total rows loaded: {len(df)}")

# Clean up any missing values in our key parameters
df = df.dropna(subset=['voltage_V', 'capacity_mAh_g', 'energy_Wh_kg', 'volume_change_pct'])

# 2. Define Features (X) and Target (y)
X = df[['voltage_V', 'capacity_mAh_g', 'volume_change_pct']]
y = df['energy_Wh_kg']

# 3. Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train a Random Forest Regressor
print("Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model performance
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Results ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Wh/kg")
print(f"R² Score: {r2:.4f}")

# 6. Check Feature Importances
print("\n--- Feature Importance Breakdown ---")
for col, imp in zip(X.columns, model.feature_importances_):
    print(f"- {col}: {imp * 100:.2f}%")
