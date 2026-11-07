import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, global_mean_pool

graphs = torch.load("graphs.pt", weights_only=False)
ids = list(json.load(open("battery_structures.json")).keys())
df = pd.read_csv("li_batteries_clean.csv").set_index("battery_id")
assert len(ids) == len(graphs), "graph count and id count differ"

bad = 0
for g, bid in zip(graphs, ids):
    g.bid = bid
    g.z = torch.round(g.x[:, 0] * 100).long()
    g.y = g.y_voltage
    if abs(float(df.loc[bid, "voltage_V"]) - g.y.item()) > 1e-4:
        bad += 1
print("ID mismatches:", bad)

random.seed(0)
formulas = sorted(set(g.formula for g in graphs))
random.shuffle(formulas)
n = len(formulas)
train_f = set(formulas[: int(0.7 * n)])
val_f = set(formulas[int(0.7 * n): int(0.85 * n)])

centers = torch.linspace(0, 5, 20)


def expand(dist):
    return torch.exp(-((dist - centers) ** 2) / 0.1)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(100, 64)
        self.c1 = CGConv(64, dim=20, aggr="mean")
        self.c2 = CGConv(64, dim=20, aggr="mean")
        self.c3 = CGConv(64, dim=20, aggr="mean")
        self.l1 = torch.nn.Linear(64, 32)
        self.l2 = torch.nn.Linear(32, 1)

    def forward(self, d):
        e = expand(d.edge_attr)
        x = self.emb(d.z)
        x = F.relu(self.c1(x, d.edge_index, e))
        x = F.relu(self.c2(x, d.edge_index, e))
        x = F.relu(self.c3(x, d.edge_index, e))
        x = global_mean_pool(x, d.batch)
        x = F.relu(self.l1(x))
        return self.l2(x).view(-1)


ck = torch.load("voltage_gnn2.pt", weights_only=False)
model = Net()
model.load_state_dict(ck["state"])
model.eval()

preds = []
with torch.no_grad():
    for d in DataLoader(graphs, batch_size=64):
        preds.append((model(d) * ck["std"] + ck["mean"]).numpy())
preds = np.concatenate(preds)

rows = []
for g, p in zip(graphs, preds):
    if g.formula in train_f:
        split = "train"
    elif g.formula in val_f:
        split = "validation"
    else:
        split = "test"
    r = df.loc[g.bid]
    rows.append({
        "battery_id": g.bid,
        "formula": g.formula,
        "voltage_mp": round(float(r["voltage_V"]), 3),
        "voltage_pred": round(float(p), 3),
        "capacity_mAh_g": round(float(r["capacity_mAh_g"]), 1),
        "energy_Wh_kg": round(float(r["energy_Wh_kg"]), 1),
        "split": split,
    })
out = pd.DataFrame(rows)
out["abs_error"] = (out["voltage_mp"] - out["voltage_pred"]).abs().round(3)
out.to_csv("predictions.csv", index=False)
print("Saved", len(out), "rows")
print(out["split"].value_counts())
t = out[out["split"] == "test"]
r2 = 1 - ((t["voltage_mp"] - t["voltage_pred"]) ** 2).sum() / ((t["voltage_mp"] - t["voltage_mp"].mean()) ** 2).sum()
print("Test r2 (should be about 0.48):", round(float(r2), 3))
print("Test MAE (should be about 0.42):", round(float(t["abs_error"].mean()), 3))
