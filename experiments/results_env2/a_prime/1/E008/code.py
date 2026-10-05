
import pandas as pd, numpy as np
import agent_api as A

for name in ["e003_mktg.parquet","e004_seasonal.parquet","e006_dynamics.parquet","e007_staples.parquet"]:
    df = A.load_saved(name)
    print(name, df.shape)
    print(list(df.columns))
    print()

# correlations of e006 features with target on train rows
t = A.train_targets()
f = A.load_saved("e006_dynamics.parquet")
m = t.merge(f, on=["household_key","snapshot_day"], how="left")
num = [c for c in f.columns if c not in ("household_key","snapshot_day")]
num = [c for c in num if pd.api.types.is_numeric_dtype(m[c])]
cor = m[num].corrwith(m["future_spend_4w"]).sort_values(key=np.abs, ascending=False)
print("TARGET describe:"); print(m["future_spend_4w"].describe())
print("\nTop |corr| with target (e006 features):")
print(cor.head(40).round(3).to_string())


# ---- cell ----

import pandas as pd, numpy as np
import agent_api as A

t = A.train_targets()
f = A.load_saved("e006_dynamics.parquet")
m = t.merge(f, on=["household_key","snapshot_day"], how="left").dropna(subset=["spend_28"])

y = m["future_spend_4w"].values

def mae(p): return np.mean(np.abs(y - p))

print("naive baselines on TRAIN rows:")
for c in ["spend_28","spend_84","spend_56","d_ewma_spend_hl28","d_ewma_spend_hl56","avg28_all","blk_1","decay_mean"]:
    print(f"  {c:22s} MAE={mae(m[c].values):8.3f}")
print("  pred=0                 MAE=%8.3f" % mae(np.zeros(len(y))))
print("  pred=median            MAE=%8.3f" % mae(np.full(len(y), np.median(y))))

# ratio y / spend_84 by bucket of spend_84
b = pd.qcut(m["spend_84"], 10, duplicates="drop")
g = m.groupby(b, observed=True).apply(lambda d: pd.Series({
    "n": len(d), "y_med": d.future_spend_4w.median(), "s84_med": d.spend_84.median(),
    "ratio_med": (d.future_spend_4w/ d.spend_84.clip(lower=1e-9)).median()}), include_groups=False)
print("\nmedian y and y/spend_84 by spend_84 decile:")
print(g.round(2).to_string())

# zero-target households: what do they look like?
z = m[m.future_spend_4w==0]; nz = m[m.future_spend_4w>0]
print(f"\nzero-target: {len(z)} ({len(z)/len(m):.1%})")
for c in ["spend_28","spend_84","recency","trips_28","d_ewma_spend_hl28"]:
    print(f"  {c:18s} zero: med={z[c].median():8.2f}   nonzero: med={nz[c].median():8.2f}")

# among zero-target, share with spend_28==0
print("zero-target & spend_28==0 share:", (z.spend_28==0).mean().round(3))
print("nonzero-target & spend_28==0 share:", (nz.spend_28==0).mean().round(3))

# log-space correlation
ly = np.log1p(y)
for c in ["spend_28","spend_84","d_ewma_spend_hl28","d_ewma_spend_hl56"]:
    print(f"corr log1p({c}) vs log1p(y): {np.corrcoef(np.log1p(m[c].values), ly)[0,1]:.3f}")


# ---- cell ----

import pandas as pd, numpy as np
import agent_api as A

t = A.train_targets()
f = A.load_saved("e006_dynamics.parquet")
m = t.merge(f, on=["household_key","snapshot_day"], how="left")
val = f[~f.snapshot_day.isin(t.snapshot_day.unique())]

# ---- drift check ----
print("mean future_spend_4w by train snapshot_day:")
print(m.groupby("snapshot_day").future_spend_4w.agg(["mean","median","size"]).round(1).to_string())
print("\nmean spend_28 by snapshot_day (train rows + val rows):")
a = m.groupby("snapshot_day").spend_28.mean().rename("train")
b = val.groupby("snapshot_day").spend_28.mean().rename("val")
print(pd.concat([a,b],axis=1).round(1).to_string())

# ---- local ridge proxy to identify model family ----
cat_cols = ["classification_1","classification_2","classification_3","classification_4",
            "classification_5","homeowner_desc","kid_category_desc"]
num_cols = [c for c in f.columns if c not in ("household_key","snapshot_day")+tuple(cat_cols)]

def design(df, num, cats, log=False, ref=None):
    Xn = df[num].copy().astype(float)
    if log:
        Xn = np.log1p(Xn.clip(lower=0))
    if ref is None:
        mu, sd = Xn.mean(), Xn.std().replace(0,1)
        Xn = (Xn-mu)/sd
    else:
        mu, sd = ref
        Xn = (Xn-mu)/sd
    Xn = Xn.fillna(0.0).values
    blocks=[Xn]
    for c in cats:
        d = pd.get_dummies(df[c].astype(str).fillna("NA"), prefix=c[:6], dtype=float)
        blocks.append(d.values)
    return np.hstack(blocks)

tr = m.dropna(subset=["spend_28"])
ytr = tr.future_spend_4w.values
va = val
yva = None

def fit_ridge(X, y, alpha):
    A_ = X.T@X + alpha*np.eye(X.shape[1]); b = X.T@y
    return np.linalg.solve(A_, b)

def val_mae(Xtr, ytr, Xva, yva, alphas=(0.1,1,10,100)):
    best=(1e9,None)
    for al in alphas:
        w = fit_ridge(Xtr,ytr,al)
        p = Xva@w
        mae = np.mean(np.abs(yva-p))
        if mae<best[0]: best=(mae,al)
    return best

# need val targets to compare with harness MAE -> approximate with our own split instead:
# hold out last 2 train snapshots (403,431) as pseudo-val
ptr = m[m.snapshot_day<=375].dropna(subset=["spend_28"])
pv = m[m.snapshot_day.isin([403,431])].dropna(subset=["spend_28"])
ytr2, yva2 = ptr.future_spend_4w.values, pv.future_spend_4w.values

for name, log in [("raw",False),("log1p",True)]:
    Xtr = design(ptr, num_cols, cat_cols, log=log)
    Xva = design(pv, num_cols, cat_cols, log=log, ref=None)  # ref misuse; recompute properly below
    # recompute with consistent ref
    Xn_tr = np.log1p(ptr[num_cols].astype(float).clip(lower=0)) if log else ptr[num_cols].astype(float)
    mu,sd = Xn_tr.mean(), Xn_tr.std().replace(0,1)
    def dm(df):
        Xn = np.log1p(df[num_cols].astype(float).clip(lower=0)) if log else df[num_cols].astype(float)
        Xn = ((Xn-mu)/sd).fillna(0.0).values
        blocks=[Xn]
        for c in cat_cols:
            blocks.append(pd.get_dummies(df[c].astype(str).fillna("NA"), prefix=c[:6], dtype=float).reindex(columns=cols[c], fill_value=0).values)
        return np.hstack(blocks)
    cols = {c: pd.get_dummies(ptr[c].astype(str).fillna("NA"), prefix=c[:6], dtype=float).columns for c in cat_cols}
    Xtr, Xva = dm(ptr), dm(pv)
    mae, al = val_mae(Xtr, ytr2, Xva, yva2)
    print(f"\nridge pseudo-val (snap 403/431) [{name}]: MAE={mae:.3f} alpha={al}  n_feat={Xtr.shape[1]}")


# ---- cell ----

import pandas as pd, numpy as np
import agent_api as A

t = A.train_targets()
f = A.load_saved("e006_dynamics.parquet")
m = t.merge(f, on=["household_key","snapshot_day"], how="left")

cat_cols = ["classification_1","classification_2","classification_3","classification_4",
            "classification_5","homeowner_desc","kid_category_desc"]
num_cols = [c for c in f.columns if c not in ("household_key","snapshot_day")+tuple(cat_cols)]
num_cols = [c for c in num_cols if pd.api.types.is_numeric_dtype(f[c])]

ptr = m[m.snapshot_day<=375].dropna(subset=["spend_28"])
pv  = m[m.snapshot_day.isin([403,431])].dropna(subset=["spend_28"])
ytr2, yva2 = ptr.future_spend_4w.values, pv.future_spend_4w.values

Xn_tr = ptr[num_cols].astype(float); mu,sd = Xn_tr.mean(), Xn_tr.std().replace(0,1)
cols = {c: pd.get_dummies(ptr[c].astype(str).fillna("NA"), prefix=c[:6], dtype=float).columns for c in cat_cols}
def dm(df):
    Xn = ((df[num_cols].astype(float)-mu)/sd).fillna(0.0).values
    blocks=[Xn]
    for c in cat_cols:
        blocks.append(pd.get_dummies(df[c].astype(str).fillna("NA"), prefix=c[:6], dtype=float).reindex(columns=cols[c], fill_value=0).values)
    return np.hstack(blocks)
Xtr, Xva = dm(ptr), dm(pv)
w = np.linalg.solve(Xtr.T@Xtr + 100*np.eye(Xtr.shape[1]), Xtr.T@ytr2)
p = Xva@w
print("base pseudo-val MAE:", round(np.mean(np.abs(yva2-p)),3))

# multiplicative shrink
for s in [0.7,0.8,0.9,1.0,1.1]:
    print(f"  shrink x{s}: {np.mean(np.abs(yva2 - s*p)):.3f}")

# blend with spend_28 and with 0
for wt in [0.0,0.1,0.2,0.3,0.5]:
    print(f"  blend p*(1-w)+spend28*w (w={wt}): {np.mean(np.abs(yva2 - ((1-wt)*p + wt*pv.spend_28.values))):.3f}")

# error structure: over/under by y bucket
df = pd.DataFrame({"y":yva2,"p":p})
df["b"] = pd.qcut(df.y, [0,.25,.5,.75,.9,1.0], duplicates="drop")
print(df.groupby("b", observed=True).apply(lambda d: pd.Series(
    {"n":len(d),"y_med":d.y.median(),"p_med":d.p.median(),"bias":(d.p-d.y).mean(),"mae":np.abs(d.p-d.y).mean()}), include_groups=False).round(1).to_string())

# error structure by spend_28==0
z = pv.spend_28.values==0
print("\nspend_28==0 rows:", z.mean().round(3), " MAE:", round(np.mean(np.abs(yva2[z]-p[z])),2), " mean y:", round(yva2[z].mean(),2), " mean p:", round(p[z].mean(),2))
print("spend_28>0  rows: MAE:", round(np.mean(np.abs(yva2[~z]-p[~z])),2))


# ---- cell ----

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
