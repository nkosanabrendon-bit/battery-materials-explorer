from mp_api.client import MPRester
import pandas as pd
import json

clean = pd.read_csv("li_batteries_clean.csv")
wanted = set(clean["battery_id"])

with MPRester() as mpr:
    docs = mpr.materials.insertion_electrodes.search(
        working_ion="Li"
    )
    pairs = {}
    for d in docs:
        if d.battery_id in wanted:
            pairs[d.battery_id] = str(d.id_discharge)
    print("Batteries with an ID:", len(pairs))

    ids = sorted(set(pairs.values()))
    structures = {}
    for i in range(0, len(ids), 200):
        chunk = ids[i:i + 200]
        res = mpr.materials.summary.search(
            material_ids=chunk, fields=["material_id", "structure"]
        )
        for r in res:
            structures[str(r.material_id)] = r.structure.as_dict()
        print("Got", len(structures), "of", len(ids), "structures")

out = {}
for bid, mid in pairs.items():
    if mid in structures:
        out[bid] = structures[mid]

with open("battery_structures.json", "w") as f:
    json.dump(out, f)
print("Saved", len(out), "structures")
