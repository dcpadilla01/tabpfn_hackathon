import numpy as np, pandas as pd, re
import agent_api as A

e = A.load_saved("e011_table.parquet")
print("e011 shape", e.shape)
cols = [c for c in e.columns if c not in ("household_key","snapshot_day")]
pref = {}
for c in cols:
    p = re.split(r"[_0-9]", c)[0] or c
    pref.setdefault(p, []).append(c)
for p, cs in sorted(pref.items()):
    print(f"{p:12s} n={len(cs):3d} e.g. {cs[:5]}")

tt = A.train_targets()
print("\ntargets", tt.shape)
print(tt.future_spend_4w.describe())

# local target computation from capped snapshot(459), verify against train_targets
v = A.snapshot()
tx = v.transactions
print("tx<=459:", tx.shape, "maxday", tx.day.max())
days = np.arange(1, 460)
piv = tx.groupby(["household_key","day"]).sales_value.sum().unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0)
C = np.hstack([np.zeros((piv.shape[0],1)), piv.values.cumsum(1)])
hidx = {h:i for i,h in enumerate(piv.index)}
tt2 = tt.copy()
tt2["i"] = tt2.household_key.map(hidx)
print("missing hh:", tt2.i.isna().sum())
tt2["i"] = tt2.i.astype(int)
t_ = tt2.snapshot_day.values
yloc = C[tt2.i, t_+28] - C[tt2.i, t_]
print("max |local - train_target| =", np.abs(yloc - tt2.future_spend_4w.values).max())

y = tt2.future_spend_4w.values
print("\nMAE global mean:", np.abs(y-y.mean()).mean())
w1 = C[tt2.i, t_] - C[tt2.i, t_-28]
print("MAE pred=w1(trailing28):", np.abs(y-w1).mean(), "corr(y,w1)=", np.corrcoef(y,w1)[0,1])
# past targets w2..w14 mean as predictor
K = 14
Wm = np.full((len(tt2), K), np.nan)
for k in range(1, K+1):
    end = t_ - 28*(k-1); start = end-27
    ok = start >= 1
    Wm[ok, k-1] = C[tt2.i[ok], end[ok]] - C[tt2.i[ok], start[ok]-1]
past = np.nanmean(Wm[:,1:], axis=1)
print("MAE pred=mean(w2..w14):", np.abs(y-past).mean())
# global ratio per snapshot (aligned windows, past only) -> pred = w1*global_ratio
preds = np.full(len(tt2), np.nan)
for j,t in enumerate(sorted(tt2.snapshot_day.unique())):
    m = tt2.snapshot_day.values==t
    ds = np.array([t-28*k for k in range(1, 14) if t-28*k >= 28])
    f = C[:, ds+28] - C[:, ds]
    b = C[:, ds] - C[:, ds-28]
    r = np.where(b>0, f/np.maximum(b,1e-9), np.nan)
    gm = np.nanmean(np.clip(r,0,25))
    preds[m] = (w1[m]) * gm
print("MAE pred=w1*global_aligned_ratio:", np.abs(y-preds).mean())
print("y quantiles:", np.quantile(y,[.5,.75,.9,.95,.99]).round(1))
