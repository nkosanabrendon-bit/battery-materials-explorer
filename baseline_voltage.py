import random
import numpy as np
import torch
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor

random.seed(0)
graphs = torch.load("graphs.pt", weights_only=False)

formulas = sorted(set(g.formula for g in graphs))
random.shuffle(formulas)
n = len(formulas)
train_f = set(formulas[: int(0.7 * n)])
val_f = set(formulas[int(0.7 * n): int(0.85 * n)])
train = [g for g in graphs if g.formula in train_f]
test = [g for g in graphs if g.formula not in train_f and g.formula not in val_f]

symbols = sorted({e.symbol for g in graphs for e in Composition(g.formula).elements})


def feat(g):
    c = Composition(g.formula).fractional_composition
    return [c[s] for s in symbols]


Xtr = np.array([feat(g) for g in train])
ytr = np.array([g.y_voltage.item() for g in train])
Xte = np.array([feat(g) for g in test])
yte = np.array([g.y_voltage.item() for g in test])

rf = RandomForestRegressor(n_estimators=300, random_state=0)
rf.fit(Xtr, ytr)
p = rf.predict(Xte)

r2 = 1 - ((yte - p) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
mae = np.abs(yte - p).mean()
print("Test materials:", len(test))
print("Random Forest test r2:", round(float(r2), 3))
print("Random Forest test MAE (volts):", round(float(mae), 3))
