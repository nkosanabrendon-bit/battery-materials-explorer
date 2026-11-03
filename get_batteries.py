from mp_api.client import MPRester
import pandas as pd

with MPRester() as mpr:
    docs = mpr.materials.insertion_electrodes.search(working_ion="Li")

rows = []
for d in docs:
    rows.append({
        "battery_id": d.battery_id,
        "formula": d.framework_formula,
        "working_ion": str(d.working_ion),
        "voltage_V": d.average_voltage,
        "capacity_mAh_g": d.capacity_grav,
        "energy_Wh_kg": d.energy_grav,
        "volume_change_pct": d.max_delta_volume,
    })

df = pd.DataFrame(rows)
df.to_csv("li_batteries.csv", index=False)
print("Saved", len(df), "battery materials")
print(df.head())
