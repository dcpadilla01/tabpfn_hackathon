
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
print("NaN counts top:", X.isna().sum().sort_values(ascending=False).head(8).to_dict())
print("max |value| per col (top 8):")
print(X.abs().max().sort_values(ascending=False).head(8))

med = pd.Series(X[fitm].median(), index=num_cols)
Xf_raw = X[fitm].fillna(med); Xh_raw = X[holdm].fillna(med)
mu = Xf_raw.mean(); sd = Xf_raw.std().replace(0,1)
Xf = ((Xf_raw-mu)/sd).values; Xh = ((Xh_raw-mu)/sd).values
yf, yh = y[fitm], y[holdm]
print("any nan in Xf/Xh:", np.isnan(Xf).any(), np.isnan(Xh).any())

G = Xf.T@Xf; b = Xf.T@yf; n = Xf.shape[1]
for a in [1,3,10,30,100,300,1000,3000,1e4,1e5,1e6]:
    w = np.linalg.solve(G + a*np.eye(n), b)
    p = Xh@w
    print(f"alpha={a:>8.0f} holdMAE={np.mean(np.abs(p-yh)):8.3f} corr={np.corrcoef(p,yh)[0,1]:.3f} predmean={p.mean():7.1f}")
# single-feature ridge on spend_28d
j = num_cols.index("spend_28d")
w1 = np.linalg.solve(G[j,j]+np.array([1,10,100]), b[j])
print("1feat:", [(round(float(np.mean(np.abs(x*Xh[:,j]-yh))),2)) for x in w1])
