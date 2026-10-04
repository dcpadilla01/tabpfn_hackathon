import pandas as pd, numpy as np
from agent_api import load_saved, train_targets

t = train_targets()
df = load_saved("e013_union.parquet")
d = df.merge(t, on=["household_key","snapshot_day"], how="inner")
print("merged shape:", d.shape)
feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
sd = d["snapshot_day"].values
tr = np.isin(sd, list(range(95,404,28))); va = np.isin(sd, [403,431])
print("tr/va counts:", tr.sum(), va.sum())
X = np.column_stack([pd.to_numeric(d[f], errors="coerce").astype(float).fillna(0).values for f in feats])
print("X shape:", X.shape, "std mean:", X[tr].std(0).mean())
y = d["future_spend_4w"].values.astype(float)
print("y mean train:", y[tr].mean(), "MAE of constant mean on va:", np.abs(y[tr].mean()-y[va]).mean())
# check a simple ridge on 5 features
Z = (X - X[tr].mean(0))/(X[tr].std(0)+1e-9)
A = Z[tr].T@Z[tr] + 100*np.eye(Z.shape[1])
w = np.linalg.solve(A, Z[tr].T@y[tr])
pred = Z[va]@w
print("ridge5-all-feats MAE on 403+431:", np.abs(pred-y[va]).mean())
print("pred std:", pred.std())
