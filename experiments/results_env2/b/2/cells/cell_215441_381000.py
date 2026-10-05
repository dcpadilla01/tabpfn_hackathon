
import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days

df = load_saved("rfm_cadence_v1.parquet")
m = df.merge(train_targets(), on=["household_key","snapshot_day"], how="inner")
tr_all = snapshot_days()["train"]
fitm = m.snapshot_day.isin([d for d in tr_all if d<=347]).values
holdm = ~fitm
num_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
y = m["future_spend_4w"].values.astype(float)

X = m[num_cols].astype(float)
med = pd.Series(X[fitm].median(), index=num_cols)
Xf_raw = X[fitm].fillna(med); Xh_raw = X[holdm].fillna(med)
mu = Xf_raw.mean(); sd = Xf_raw.std().replace(0,1)
Xf = ((Xf_raw-mu)/sd).values; Xh = ((Xh_raw-mu)/sd).values
yf, yh = y[fitm], y[holdm]
ym = yf.mean(); yf0 = yf - ym

G = Xf.T@Xf; b = Xf.T@yf0; n = Xf.shape[1]
best = None
for a in [0.3,1,3,10,30,100,300,1000,3000]:
    w = np.linalg.solve(G + a*np.eye(n), b)
    p = Xh@w + ym
    mae = np.mean(np.abs(p-yh))
    print(f"alpha={a:>6} holdMAE={mae:8.3f} corr={np.corrcoef(p,yh)[0,1]:.3f}")
    if best is None or mae < best[1]: best = (a,mae)
print("best:", best, "| harness E003 val MAE = 61.131")
