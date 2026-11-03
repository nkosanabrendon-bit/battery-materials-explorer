import pandas as pd

df = pd.read_csv("li_batteries.csv")

print("Rows:", len(df))
print("Missing values:")
print(df.isna().sum())
print()
print(df.describe().round(2))
print()
print("Top 5 by energy:")
print(df.sort_values("energy_Wh_kg", ascending=False).head(5)[["formula", "voltage_V", "capacity_mAh_g", "energy_Wh_kg"]])
