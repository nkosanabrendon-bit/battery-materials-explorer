import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

random.seed(0)
torch.manual_seed(0)

graphs = torch.load("graphs.pt", weights_only=False)
for g in graphs:
    g.y = g.y_voltage

# Split by formula, so the same material never sits in two piles
formulas = sorted(set(g.formula for g in graphs))
random.shuffle(formulas)
n = len(formulas)
train_f = set(formulas[: int(0.7 * n)])
val_f = set(formulas[int(0.7 * n): int(0.85 * n)])
train = [g for g in graphs if g.formula in train_f]
val = [g for g in graphs if g.formula in val_f]
test = [g for g in graphs if g.formula not in train_f and g.formula not in val_f]
print("Train:", len(train), "Val:", len(val), "Test:", len(test))

ys = torch.cat([g.y for g in train])
mean, std = ys.mean().item(), ys.std().item()

train_loader = DataLoader(train, batch_size=32, shuffle=True)
val_loader = DataLoader(val, batch_size=64)
test_loader = DataLoader(test, batch_size=64)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = GCNConv(5, 64)
        self.c2 = GCNConv(64, 64)
        self.c3 = GCNConv(64, 64)
        self.l1 = torch.nn.Linear(64, 32)
        self.l2 = torch.nn.Linear(32, 1)

    def forward(self, d):
        x = F.relu(self.c1(d.x, d.edge_index))
        x = F.relu(self.c2(x, d.edge_index))
        x = F.relu(self.c3(x, d.edge_index))
        x = global_mean_pool(x, d.batch)
        x = F.relu(self.l1(x))
        return self.l2(x).view(-1)


model = Net()
opt = torch.optim.Adam(model.parameters(), lr=0.001)


def predict(loader):
    model.eval()
    p, t = [], []
    with torch.no_grad():
        for d in loader:
            p.append(model(d) * std + mean)
            t.append(d.y)
    return torch.cat(p).numpy(), torch.cat(t).numpy()


def score(p, t):
    r2 = 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    mae = np.abs(t - p).mean()
    return r2, mae


best, best_state, wait = 1e9, None, 0
for epoch in range(1, 301):
    model.train()
    for d in train_loader:
        opt.zero_grad()
        loss = F.mse_loss(model(d), (d.y - mean) / std)
        loss.backward()
        opt.step()
    p, t = predict(val_loader)
    val_loss = ((p - t) ** 2).mean()
    if val_loss < best:
        best, wait = val_loss, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
    if epoch % 10 == 0:
        print("Epoch", epoch, "val MSE", round(float(val_loss), 4))
    if wait >= 30:
        print("Early stop at epoch", epoch)
        break

model.load_state_dict(best_state)
p, t = predict(test_loader)
r2, mae = score(p, t)
base = np.abs(t - mean).mean()
print()
print("Test r2:", round(float(r2), 3))
print("Test MAE (volts):", round(float(mae), 3))
print("Guessing the average every time would give MAE:", round(float(base), 3))
torch.save({"state": best_state, "mean": mean, "std": std}, "voltage_gnn.pt")
