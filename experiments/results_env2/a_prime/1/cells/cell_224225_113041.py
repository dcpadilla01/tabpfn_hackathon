
import pandas as pd, numpy as np
import agent_api as A

# Build prior-snapshot (X=spend_28 at s, y=spend in (s,s+28]) pairs from transactions (analysis only)
snap = A.snapshot(459)
tx = snap.transactions
first = tx.groupby("household_key").day.min()
grid = [95,123,151,179,207,235,263,291,319,347,375,403,431]

pairs = []
for s in grid:
    if s+28 > 459: continue
    w = tx[(tx.day > s-28) & (tx.day <= s)].groupby("household_key").sales_value.sum()
    y = tx[(tx.day > s) & (tx.day <= s+28)].groupby("household_key").sales_value.sum()
    df = pd.DataFrame({"x": w}).join(pd.DataFrame({"y": y})).fillna({"y":0.0})
    df = df[df.index.map(first).notna() & (df.index.map(first) <= s-84)]
    df["s"] = s
    pairs.append(df.reset_index())
P = pd.concat(pairs, ignore_index=True)
print("pairs:", P.shape, "per-snapshot sizes:", P.groupby("s").size().to_dict())

t = A.train_targets()
f = A.load_saved("e006_dynamics.parquet")
m = t.merge(f, on=["household_key","snapshot_day"], how="left")

def calib_pred(rows_d, max_s):
    """rows_d: df with household_key, x=spend_28 at d. Pool pairs with s<=max_s."""
    pool = P[P.s <= max_s]
    x = pool.x.values; y = pool.y.values
    qs = np.quantile(x, np.linspace(0,1,21))
    qs[0] -= 1; qs[-1] += 1
    b = pd.cut(pd.Series(x), qs, labels=False)
    med = pd.Series(y).groupby(b.values).median()
    glb = np.median(y)
    xb = pd.cut(pd.Series(rows_d.x.values), qs, labels=False)
    return xb.map(med).fillna(glb).values

def mae(y,p): return np.mean(np.abs(y-p))

# pure calib predictor on pseudo-val (403,431) and full train
for days,name in [([403,431],"pseudo-val"),([459,487,515,543],"val(no y—skip)")]:
    if name.startswith("val"): continue
    sub = m[m.snapshot_day.isin(days)]
    x = sub.spend_28.fillna(0).values
    p = calib_pred(pd.DataFrame({"x":x}), max_s=max(days)-28)
    print(f"pure calib (spend28-decile median) {name}: MAE={mae(sub.future_spend_4w.values,p):.3f}")

# k-NN calib on (log spend_28, log spend_84) — pseudo-val
def knn_calib(rows_d, max_s, k=60):
    pool = P[P.s <= max_s].copy()
    w84 = []
    # need spend_84 per pair: recompute quickly
    txl = tx
    s84 = {}
    for s in sorted(pool.s.unique()):
        w = txl[(txl.day > s-84) & (txl.day <= s)].groupby("household_key").sales_value.sum()
        s84[s] = w
    pool["x84"] = [s84[s].get(h, 0.0) for s,h in zip(pool.s, pool.household_key)]
    Fp = np.log1p(pool[["x","x84"]].values)
    y = pool.y.values
    Fd = np.log1p(rows_d[["x","x84"]].values)
    out = np.empty(len(rows_d))
    for i in range(len(rows_d)):
        d2 = ((Fp-Fd[i])**2).sum(1)
        idx = np.argpartition(d2, k)[:k]
        out[i] = np.median(y[idx])
    return out

for days in [[403,431]]:
    sub = m[m.snapshot_day.isin(days)]
    rd = pd.DataFrame({"x":sub.spend_28.fillna(0).values, "x84":sub.spend_84.fillna(0).values})
    p = knn_calib(rd, max_s=max(days)-28)
    print(f"pure kNN calib (spend28,spend84) pseudo-val: MAE={mae(sub.future_spend_4w.values,p):.3f}")
    rd2 = pd.DataFrame({"x":sub.spend_28.fillna(0).values})
    p2 = calib_pred(rd2, max_s=max(days)-28)
    print(f"blend kNN*0.5+decile*0.5: MAE={mae(sub.future_spend_4w.values, 0.5*p+0.5*p2):.3f}")
