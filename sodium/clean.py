import pandas as pd

df = pd.read_csv("na_batteries.csv")
print("Before cleaning:", len(df))

keep = (
    (df["voltage_V"] >= 1.0) & (df["voltage_V"] <= 5.5)
    & (df["capacity_mAh_g"] >= 10) & (df["capacity_mAh_g"] <= 700)
    & (df["energy_Wh_kg"] >= 50) & (df["energy_Wh_kg"] <= 2000)
)
clean = df[keep]

print("After cleaning:", len(clean))
print()
print(clean.describe().round(2))
clean.to_csv("na_batteries_clean.csv", index=False)
print()
print("Top 5 by energy:")
print(clean.sort_values("energy_Wh_kg", ascending=False).head(5)[["formula", "voltage_V", "capacity_mAh_g", "energy_Wh_kg"]])
