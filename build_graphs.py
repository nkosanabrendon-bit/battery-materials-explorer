import json
import torch
import pandas as pd
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from torch_geometric.data import Data

df = pd.read_csv("li_batteries_clean.csv").set_index("battery_id")
structs = json.load(open("battery_structures.json"))

graphs = []
for bid, sd in structs.items():
    s = Structure.from_dict(sd)

    feats = []
    for site in s:
        el = Element(site.specie.symbol)
        x = el.X
        if x != x:
            x = 0.0
        feats.append([el.Z / 100, x / 4, float(el.atomic_mass) / 200,
                      el.row / 7, el.group / 18])

    src, dst, dist = [], [], []
    for i, nbrs in enumerate(s.get_all_neighbors(5.0)):
        nbrs = sorted(nbrs, key=lambda n: n.nn_distance)[:12]
        for n in nbrs:
            src.append(n.index)
            dst.append(i)
            dist.append(n.nn_distance)
    if len(src) == 0:
        continue

    row = df.loc[bid]
    graphs.append(Data(
        x=torch.tensor(feats, dtype=torch.float),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(dist, dtype=torch.float).view(-1, 1),
        y_voltage=torch.tensor([float(row["voltage_V"])]),
        y_capacity=torch.tensor([float(row["capacity_mAh_g"])]),
        y_energy=torch.tensor([float(row["energy_Wh_kg"])]),
        formula=str(row["formula"]),
    ))

torch.save(graphs, "graphs.pt")
atoms = [g.num_nodes for g in graphs]
edges = [g.num_edges for g in graphs]
print("Graphs saved:", len(graphs))
print("Atoms per crystal: average", round(sum(atoms) / len(atoms), 1), "max", max(atoms))
print("Edges per crystal: average", round(sum(edges) / len(edges), 1), "max", max(edges))
