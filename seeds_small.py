import random
import numpy as np
import torch
import torch.nn.functional as F
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, global_mean_pool

graphs = torch.load("graphs.pt", weights_only=False)
random.Random(0).shuffle(graphs)
graphs = graphs[:360]
for g in graphs:
    g.y = g.y_voltage
    g.z = torch.round(g.x[:, 0] * 100).long()
formulas = sorted(set(g.formula for g in graphs))
symbols = sorted({e.symbol for g in graphs for e in Composition(g.formula).elements})
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


def metrics(p, t):
    r2 = 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    return float(r2), float(np.abs(t - p).mean())


def feat(g):
    c = Composition(g.formula).fractional_composition
    return [c[s] for s in symbols]


rf_res, gnn_res = [], []
for seed in [1, 2, 3]:
    rnd = random.Random(seed)
    torch.manual_seed(seed)
    fl = formulas[:]
    rnd.shuffle(fl)
    n = len(fl)
    tf = set(fl[: int(0.7 * n)])
    vf = set(fl[int(0.7 * n): int(0.85 * n)])
    train = [g for g in graphs if g.formula in tf]
    val = [g for g in graphs if g.formula in vf]
    test = [g for g in graphs if g.formula not in tf and g.formula not in vf]

    rf = RandomForestRegressor(n_estimators=300, random_state=seed)
    rf.fit(np.array([feat(g) for g in train]), np.array([g.y.item() for g in train]))
    p = rf.predict(np.array([feat(g) for g in test]))
    t = np.array([g.y.item() for g in test])
    rf_res.append(metrics(p, t))

    ys = torch.cat([g.y for g in train])
    mean, std = ys.mean().item(), ys.std().item()
    tl = DataLoader(train, batch_size=32, shuffle=True)
    vl = DataLoader(val, batch_size=64)
    sl = DataLoader(test, batch_size=64)
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    def predict(loader):
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for d in loader:
                ps.append(model(d) * std + mean)
                ts.append(d.y)
        return torch.cat(ps).numpy(), torch.cat(ts).numpy()

    best, best_state, wait = 1e9, None, 0
    for epoch in range(1, 201):
        model.train()
        for d in tl:
            opt.zero_grad()
            F.mse_loss(model(d), (d.y - mean) / std).backward()
            opt.step()
        p, t2 = predict(vl)
        vloss = ((p - t2) ** 2).mean()
        if vloss < best:
            best, wait = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= 25:
            break
    model.load_state_dict(best_state)
    p, t2 = predict(sl)
    gnn_res.append(metrics(p, t2))
    print("Seed", seed, "| RF r2 %.3f MAE %.3f | GNN r2 %.3f MAE %.3f"
          % (rf_res[-1][0], rf_res[-1][1], gnn_res[-1][0], gnn_res[-1][1]), flush=True)

rf_a, gnn_a = np.array(rf_res), np.array(gnn_res)
print()
print("Random Forest: r2 %.3f +/- %.3f | MAE %.3f" % (rf_a[:, 0].mean(), rf_a[:, 0].std(), rf_a[:, 1].mean()))
print("GNN v2:        r2 %.3f +/- %.3f | MAE %.3f" % (gnn_a[:, 0].mean(), gnn_a[:, 0].std(), gnn_a[:, 1].mean()))
