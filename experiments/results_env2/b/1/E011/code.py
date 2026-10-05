
import pandas as pd, numpy as np

names = ["e001_history","e002_mix","e003_full","e003_momentum","e004_marketing","e006_cadence","e007_temporal","e008_level_shape","e009_target_enc","e010_gross"]
for n in names:
    df = load_saved(n + ".parquet")
    print("==", n, df.shape)
    print(sorted([c for c in df.columns if c not in ("household_key","snapshot_day")]))

t = train_targets()
print("\ntargets", t.shape)
print(t[TARGET].describe())
print("zero frac:", (t[TARGET]==0).mean())
print("snapdays:", snapshot_days())


# ---- cell ----

import pandas as pd, numpy as np

base = load_saved("e008_level_shape.parquet")
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
print("merged", df.shape)
y = df[TARGET].values.astype(float)

feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")
for c in cat_cols:
    d = pd.get_dummies(X[c].astype("category"), prefix=c[:10], dummy_na=True)
    Xn = pd.concat([Xn, d.astype(float)], axis=1)
med = Xn.median()
Xn = Xn.fillna(med)
print("design", Xn.shape, "cats:", cat_cols)

print("\nBaselines on train rows:")
print("  MAE predict-0   :", np.abs(y).mean().round(2))
print("  MAE global mean :", np.abs(y-y.mean()).mean().round(2))
if "sp28" in df: print("  MAE sp28 (last4w):", np.abs(y-df["sp28"]).mean().round(2))
if "sp84_rate" in df: print("  MAE sp84_rate*4 :", np.abs(y-df["sp84_rate"]*4).mean().round(2))
if "z_med4w_hist" in df: print("  MAE med4w hist  :", np.abs(y-df["z_med4w_hist"]).mean().round(2))

# ridge CV
idx = np.random.RandomState(0).permutation(len(y)); K=5
def ridge_cv(Xv, yv, alphas=(1,10,100,1000), logt=False):
    n=len(yv); maes={a:0.0 for a in alphas}
    yt = np.log1p(yv) if logt else yv
    for k in range(K):
        va = idx[k::K]; tr = np.setdiff1d(idx, va)
        mu=Xv[tr].mean(0); sd=Xv[tr].std(0)+1e-9
        A=np.hstack([(Xv[tr]-mu)/sd, np.ones((len(tr),1))])
        B=np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))])
        G=A.T@A; p=G.shape[0]
        for a in alphas:
            Gg=G.copy(); Gg[np.arange(p-1),np.arange(p-1)]+=a
            w=np.linalg.solve(Gg, A.T@yt[tr])
            pred=B@w
            if logt: pred=np.clip(np.expm1(pred),0,None)
            maes[a]+=np.abs(yv[va]-pred).sum()
    return {a: round(m/n,2) for a,m in maes.items()}

Xv = Xn.values.astype(np.float64)
print("\nRidge CV MAE (raw target):", ridge_cv(Xv, y))
print("Ridge CV MAE (log target):", ridge_cv(Xv, y, logt=True))

# feature correlations with target
cors = {}
for c in Xn.columns:
    cors[c] = abs(np.corrcoef(Xn[c].values, y)[0,1]) if Xn[c].std()>0 else 0
top = sorted(cors.items(), key=lambda kv:-kv[1])[:25]
print("\nTop-25 |corr| with target:")
for c,v in top: print(f"  {c:24s} {v:.3f}")


# ---- cell ----

import pandas as pd, numpy as np

base = load_saved("e008_level_shape.parquet")
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
y = df[TARGET].values.astype(float)
feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")
for c in cat_cols:
    d = pd.get_dummies(X[c].astype("category"), prefix=c[:10], dummy_na=True)
    Xn = pd.concat([Xn, d.astype(float)], axis=1)
Xn = Xn.fillna(Xn.median())
Xv = Xn.values.astype(np.float64)
n = len(y)

# correlations (fixed)
cors = {}
for c in Xn.columns:
    v = Xn[c].values
    if np.nanstd(v) > 1e-12:
        cors[c] = abs(float(np.corrcoef(v, y)[0,1]))
top = sorted(cors.items(), key=lambda kv: -kv[1])[:25]
print("Top-25 |corr| with target:")
for c, val in top: print(f"  {c:22s} {val:.3f}")

# --- structural tests ---
idx = np.random.RandomState(0).permutation(n); K=5
def cv_mae(pred_fn):
    s = 0.0
    for k in range(K):
        va = idx[k::K]; tr = np.setdiff1d(idx, va)
        p = pred_fn(tr, va)
        s += np.abs(y[va]-p).sum()
    return round(s/n, 2)

def ridge_fit(Xtr, ytr, a=100):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-9
    A=np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))])
    G=A.T@A; p=G.shape[0]
    Gg=G.copy(); Gg[np.arange(p),np.arange(p)]+=a*np.r_[np.ones(p-1),0]
    return np.linalg.solve(Gg, A.T@ytr), mu, sd

# 1) plain ridge
def f_ridge(tr, va):
    w, mu, sd = ridge_fit(Xv[tr], y[tr])
    return np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))]) @ w
print("\nridge all:", cv_mae(f_ridge))

# 2) two-piece: inactive (sp28==0) -> constant, active -> ridge
sp28 = df["sp28"].values.astype(float)
inact = sp28 <= 0
def f_2piece(tr, va):
    c = y[tr][inact[tr]].mean() if inact[tr].any() else 0.0
    w, mu, sd = ridge_fit(Xv[tr][~inact[tr]], y[tr][~inact[tr]])
    p = np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))]) @ w
    p[inact[va]] = c
    return p
print("two-piece (const for sp28==0):", cv_mae(f_2piece))

# 3) ridge on log target, exp back (two-piece too)
def f_2piece_log(tr, va):
    c = np.expm1(np.log1p(y[tr][inact[tr]]).mean()) if inact[tr].any() else 0.0
    w, mu, sd = ridge_fit(Xv[tr][~inact[tr]], np.log1p(y[tr][~inact[tr]]))
    p = np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))]) @ w
    p = np.clip(np.expm1(p), 0, None); p[inact[va]] = c
    return p
print("two-piece log-target:", cv_mae(f_2piece_log))

# 4) blend ridge with med4w hist
med4 = df["z_med4w_hist"].values.astype(float)
def f_blend(tr, va):
    w, mu, sd = ridge_fit(Xv[tr], y[tr])
    p = np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))]) @ w
    return 0.5*p + 0.5*med4[va]
print("blend ridge+med4w 50/50:", cv_mae(f_blend))

# 5) quantile cap
def f_cap(tr, va):
    w, mu, sd = ridge_fit(Xv[tr], y[tr])
    p = np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))]) @ w
    hi = np.quantile(y[tr], 0.97)
    return np.clip(p, 0, hi)
print("ridge capped at q97:", cv_mae(f_cap))


# ---- cell ----

import pandas as pd, numpy as np
base = load_saved("e008_level_shape.parquet")
feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
dup = [c for c in set(feat_cols) if feat_cols.count(c) > 1]
print("dup cols in table:", dup)
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
y = df[TARGET].values.astype(float)
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")
for c in cat_cols:
    d = pd.get_dummies(X[c].astype("category"), prefix=c[:10], dummy_na=True)
    Xn = pd.concat([Xn, d.astype(float)], axis=1)
Xn = Xn.fillna(Xn.median())
# drop duplicate labels
Xn = Xn.loc[:, ~Xn.columns.duplicated()]
print("design", Xn.shape)
Xv = Xn.values.astype(np.float64)
n = len(y)
cors = {}
for c in Xn.columns:
    v = Xn[c].values
    if np.nanstd(v) > 1e-12:
        cors[c] = abs(float(np.corrcoef(v, y)[0,1]))
top = sorted(cors.items(), key=lambda kv: -kv[1])[:25]
print("Top-25 |corr| with target:")
for c, val in top: print(f"  {c:22s} {val:.3f}")

idx = np.random.RandomState(0).permutation(n); K=5
def cv_mae(pred_fn):
    s = 0.0
    for k in range(K):
        va = idx[k::K]; tr = np.setdiff1d(idx, va)
        s += np.abs(y[va]-pred_fn(tr, va)).sum()
    return round(s/n, 2)
def ridge_fit(Xtr, ytr, a=100):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-9
    A=np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))])
    G=A.T@A; p=G.shape[0]
    Gg=G.copy(); Gg[np.arange(p),np.arange(p)]+=a*np.r_[np.ones(p-1),0]
    return np.linalg.solve(Gg, A.T@ytr), mu, sd
def f_ridge(tr, va):
    w, mu, sd = ridge_fit(Xv[tr], y[tr])
    return np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))]) @ w
print("\nridge all:", cv_mae(f_ridge))
sp28 = df["sp28"].values.astype(float)
inact = sp28 <= 0
def f_2piece(tr, va):
    c = y[tr][inact[tr]].mean() if inact[tr].any() else 0.0
    w, mu, sd = ridge_fit(Xv[tr][~inact[tr]], y[tr][~inact[tr]])
    p = np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))]) @ w
    p[inact[va]] = c
    return p
print("two-piece const-for-inactive:", cv_mae(f_2piece))
def f_2piece_log(tr, va):
    c = float(np.expm1(np.log1p(y[tr][inact[tr]]).mean())) if inact[tr].any() else 0.0
    w, mu, sd = ridge_fit(Xv[tr][~inact[tr]], np.log1p(y[tr][~inact[tr]]))
    p = np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))]) @ w
    p = np.clip(np.expm1(p), 0, None); p[inact[va]] = c
    return p
print("two-piece log-target:", cv_mae(f_2piece_log))
med4 = df["z_med4w_hist"].values.astype(float)
def f_blend(tr, va):
    w, mu, sd = ridge_fit(Xv[tr], y[tr])
    p = np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))]) @ w
    return 0.5*p + 0.5*med4[va]
print("blend ridge+med4w:", cv_mae(f_blend))
def f_cap(tr, va):
    w, mu, sd = ridge_fit(Xv[tr], y[tr])
    p = np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))]) @ w
    return np.clip(p, 0, np.quantile(y[tr], 0.97))
print("ridge cap q97:", cv_mae(f_cap))


# ---- cell ----

import pandas as pd, numpy as np, agent_api

base = load_saved("e008_level_shape.parquet")
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
y = df[TARGET].values.astype(float)

feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")
for i, c in enumerate(cat_cols):
    d = pd.get_dummies(X[c].astype("category"), prefix=f"cat{i}", dummy_na=True)
    Xn = pd.concat([Xn, d.astype(float)], axis=1)
Xn = Xn.fillna(Xn.median())
Xv = Xn.values.astype(np.float64)
print("design", Xn.shape)

sp28 = df["sp28"].values.astype(float)
bins = [0, 0.01, 25, 50, 100, 200, 400, 1e9]
labs = pd.cut(sp28, bins, labels=False)
print("\nE[y|sp28 bucket]:")
for b in sorted(pd.unique(labs)):
    m = labs==b
    print(f"  bucket {b}: n={m.sum():6d}  mean sp28={sp28[m].mean():7.1f}  mean y={y[m].mean():7.1f}  med y={np.median(y[m]):7.1f}")

snap = agent_api.snapshot(459)
tx = snap.transactions[["household_key","day","sales_value","week_no"]].copy()
print("\ntx rows:", len(tx), "neg sales:", (tx.sales_value<0).sum())

tx = tx.sort_values(["household_key","day"])
keys = tx.household_key.values
days = tx.day.values.astype(int)
sv = tx.sales_value.values.astype(float)
order = np.argsort(keys, kind="stable")
keys, days, sv = keys[order], days[order], sv[order]
uniq, start = np.unique(keys, return_index=True)
h2idx = {h:i for i,h in enumerate(uniq)}
ends = np.r_[start[1:], len(keys)]
cs = np.concatenate([[0.0], np.cumsum(sv)])

def win_sum(h, lo, hi):
    i = h2idx.get(h)
    if i is None: return 0.0
    a = np.searchsorted(days[start[i]:ends[i]], lo, "left") + start[i]
    b = np.searchsorted(days[start[i]:ends[i]], hi, "right") + start[i]
    return cs[b]-cs[a]

hh = df["household_key"].values; sd = df["snapshot_day"].values
seas1y = np.array([win_sum(h, s-364, s-336) for h, s in zip(hh, sd)])
seas1y_b = np.array([win_sum(h, s-392, s-364) for h, s in zip(hh, sd)])
print("corr seas1y aligned:", round(np.corrcoef(seas1y, y)[0,1],3), "| misaligned:", round(np.corrcoef(seas1y_b, y)[0,1],3))

wk = tx.groupby("week_no").sales_value.sum()
wk_cnt = tx.groupby("week_no").household_key.nunique()
wk_per_hh = (wk / wk_cnt).fillna(0)
seas_idx, mkt_now, mkt_yoy = [], [], []
for s in sd:
    fw = list(range((s+9)//7, (s+28+8)//7 + 1))
    prev  = [wk_per_hh[w-52] for w in fw if (w-52) in wk_per_hh.index]
    prev2 = [wk_per_hh[w-104] for w in fw if (w-104) in wk_per_hh.index]
    seas_idx.append(np.mean(prev) if prev else np.nan)
    mkt_yoy.append(np.mean(prev)/np.mean(prev2) if prev and prev2 and np.mean(prev2)>0 else np.nan)
    now = [wk_per_hh[w] for w in range((s+8)//7-8, (s+8)//7) if w in wk_per_hh.index]
    mkt_now.append(np.mean(now) if now else np.nan)
seas_idx = np.array(seas_idx); mkt_now = np.array(mkt_now); mkt_yoy = np.array(mkt_yoy)
for s in np.unique(sd):
    m = sd==s
    print(f"  s={s}: seas_idx={np.nanmean(seas_idx[m]):6.1f} mkt_now={np.nanmean(mkt_now[m]):6.1f} yoy={np.nanmean(mkt_yoy[m]):5.2f} mean_y={y[m].mean():6.1f}")


# ---- cell ----

import pandas as pd, numpy as np
base = load_saved("e008_level_shape.parquet")
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
y = df[TARGET].values.astype(float)
feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")
for i, c in enumerate(cat_cols):
    Xn = pd.concat([Xn, pd.get_dummies(X[c].astype("category"), prefix=f"cat{i}", dummy_na=True).astype(float)], axis=1)
Xn = Xn.fillna(Xn.median())
Xv = Xn.values.astype(np.float64); n = len(y)
sp28 = df["sp28"].values.astype(float)
med4 = df["z_med4w_hist"].values.astype(float)

idx = np.random.RandomState(0).permutation(n); K=5
def cv_mae(pred_fn):
    s=0.0
    for k in range(K):
        va=idx[k::K]; tr=np.setdiff1d(idx,va)
        s+=np.abs(y[va]-pred_fn(tr,va)).sum()
    return round(s/n,2)
def ridge_fit(Xtr,ytr,a=100):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-9
    A=np.hstack([(Xtr-mu)/sd,np.ones((len(Xtr),1))])
    G=A.T@A; p=G.shape[0]
    Gg=G.copy(); Gg[np.arange(p),np.arange(p)]+=a*np.r_[np.ones(p-1),0]
    return np.linalg.solve(Gg,A.T@ytr),mu,sd
def ridge_pred(tr,va,Xall):
    w,mu,sd=ridge_fit(Xall[tr],y[tr])
    return np.hstack([(Xall[va]-mu)/sd,np.ones((len(va),1))])@w

# 1) scaling predictions toward conditional median
p0 = np.zeros(n)
for k in range(K):
    va=idx[k::K]; tr=np.setdiff1d(idx,va); p0[va]=ridge_pred(tr,va,Xv)
for c in [0.80,0.85,0.90,0.95,1.0]:
    print(f"scale {c}: {np.abs(y-c*p0).mean().round(2)}")

# 2) bucket-calibration feature (out-of-fold bucket means)
bins=[0,0.01,25,50,100,200,400,1e9]
labs=pd.cut(sp28,bins,labels=False).values
cal=np.zeros(n)
for k in range(K):
    va=idx[k::K]; tr=np.setdiff1d(idx,va)
    bm={b:y[tr][labs[tr]==b].mean() for b in np.unique(labs[tr])}
    cal[va]=[bm.get(l, y[tr].mean()) for l in labs[va]]
X2=np.hstack([Xv, cal[:,None]])
print("ridge + cal feature:", cv_mae(lambda tr,va: ridge_pred(tr,va,X2)))
# cal alone
print("cal alone MAE:", round(np.abs(y-cal).mean(),2))
# blend cal with ridge
for wgt in [0.3,0.5,0.7]:
    print(f"blend ridge+cal {wgt}:", round(np.abs(y-(wgt*p0+(1-wgt)*cal)).mean(),2))

# 3) blend with med4w finer
for wgt in [0.3,0.4,0.5,0.6]:
    print(f"blend ridge+med4w {wgt}:", round(np.abs(y-(wgt*p0+(1-wgt)*med4)).mean(),2))
# blend all three
for a,b in [(0.45,0.2),(0.4,0.25),(0.5,0.15)]:
    print(f"blend ridge {a} med4w {b} cal {round(1-a-b,2)}:", round(np.abs(y-(a*p0+b*med4+(1-a-b)*cal)).mean(),2))


# ---- cell ----

import pandas as pd, numpy as np
base = load_saved("e008_level_shape.parquet")
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
y = df[TARGET].values.astype(float)
feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")
for i, c in enumerate(cat_cols):
    Xn = pd.concat([Xn, pd.get_dummies(X[c].astype("category"), prefix=f"cat{i}", dummy_na=True).astype(float)], axis=1)
Xn = Xn.fillna(Xn.median())
Xv = Xn.values.astype(np.float64); n = len(y)
sp28 = df["sp28"].values.astype(float)
med4 = df["z_med4w_hist"].values.astype(float)

idx = np.random.RandomState(0).permutation(n); K=5
def cv_mae(pred_fn):
    s=0.0
    for k in range(K):
        va=idx[k::K]; tr=np.setdiff1d(idx,va)
        s+=np.abs(y[va]-pred_fn(tr,va)).sum()
    return round(s/n,2)
def ridge_fit(Xtr,ytr,a=100):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-9
    A=np.hstack([(Xtr-mu)/sd,np.ones((len(Xtr),1))])
    G=A.T@A; p=G.shape[0]
    Gg=G.copy(); Gg[np.arange(p),np.arange(p)]+=a*np.r_[np.ones(p-1),0]
    return np.linalg.solve(Gg,A.T@ytr),mu,sd
def ridge_pred(tr,va,Xall):
    w,mu,sd=ridge_fit(Xall[tr],y[tr])
    return np.hstack([(Xv[va]-mu)/sd,np.ones((len(va),1))])@w if False else np.hstack([(Xall[va]-mu)/sd,np.ones((len(va),1))])@w

p0=np.zeros(n)
for k in range(K):
    va=idx[k::K]; tr=np.setdiff1d(idx,va); p0[va]=ridge_pred(tr,va,Xv)
print("oof ridge:", round(np.abs(y-p0).mean(),2))

bins=[0,0.01,25,50,100,200,400,1e9]
labs=pd.cut(sp28,bins,labels=False).values.astype(float)
cal=np.zeros(n)
for k in range(K):
    va=idx[k::K]; tr=np.setdiff1d(idx,va)
    bm={b:y[tr][labs[tr]==b].mean() for b in np.unique(labs[tr])}
    gm=y[tr].mean()
    cal[va]=[bm.get(l,gm) for l in labs[va]]
print("cal alone:", round(np.abs(y-cal).mean(),2))
X2=np.hstack([Xv, cal[:,None]])
print("ridge + cal feature:", cv_mae(lambda tr,va: ridge_pred(tr,va,X2)))
for wgt in [0.3,0.5,0.7]:
    print(f"blend ridge+cal {wgt}:", round(np.abs(y-(wgt*p0+(1-wgt)*cal)).mean(),2))
for wgt in [0.3,0.4,0.5,0.6]:
    print(f"blend ridge+med4w {wgt}:", round(np.abs(y-(wgt*p0+(1-wgt)*med4)).mean(),2))
for a,b in [(0.45,0.2),(0.4,0.25),(0.5,0.15)]:
    print(f"ridge {a} med4w {b} cal {round(1-a-b,2)}:", round(np.abs(y-(a*p0+b*med4+(1-a-b)*cal)).mean(),2))


# ---- cell ----

import pandas as pd, numpy as np
base = load_saved("e008_level_shape.parquet")
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
y = df[TARGET].values.astype(float)
feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")
for i, c in enumerate(cat_cols):
    Xn = pd.concat([Xn, pd.get_dummies(X[c].astype("category"), prefix=f"cat{i}", dummy_na=True).astype(float)], axis=1)
Xn = Xn.fillna(Xn.median())
Xv = Xn.values.astype(np.float64); n = len(y)
sp28 = df["sp28"].values.astype(float)
med4 = df["z_med4w_hist"].values.astype(float)

idx = np.random.RandomState(0).permutation(n); K=5
def ridge_fit(Xtr,ytr,a=100):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-9
    A=np.hstack([(Xtr-mu)/sd,np.ones((len(Xtr),1))])
    G=A.T@A; p=G.shape[0]
    Gg=G.copy(); Gg[np.arange(p),np.arange(p)]+=a*np.r_[np.ones(p-1),0]
    return np.linalg.solve(Gg,A.T@ytr),mu,sd
def ridge_pred(tr,va,Xall):
    w,mu,sd=ridge_fit(Xall[tr],y[tr])
    return np.hstack([(Xall[va]-mu)/sd,np.ones((len(va),1))])@w
def cv_mae(pred_fn):
    s=0.0
    for k in range(K):
        va=idx[k::K]; tr=np.setdiff1d(idx,va)
        s+=np.abs(y[va]-pred_fn(tr,va)).sum()
    return round(s/n,2)

p0=np.zeros(n)
for k in range(K):
    va=idx[k::K]; tr=np.setdiff1d(idx,va); p0[va]=ridge_pred(tr,va,Xv)
print("oof ridge:", round(np.abs(y-p0).mean(),2))

bins=[0,0.01,25,50,100,200,400,1e9]
labs=np.asarray(pd.cut(sp28,bins,labels=False),dtype=float)
cal=np.zeros(n)
for k in range(K):
    va=idx[k::K]; tr=np.setdiff1d(idx,va)
    bm={b:y[tr][labs[tr]==b].mean() for b in np.unique(labs[tr])}
    gm=y[tr].mean()
    cal[va]=[bm.get(l,gm) for l in labs[va]]
print("cal alone:", round(np.abs(y-cal).mean(),2))
X2=np.hstack([Xv, cal[:,None]])
print("ridge + cal feature:", cv_mae(lambda tr,va: ridge_pred(tr,va,X2)))
for wgt in [0.3,0.5,0.7]:
    print(f"blend ridge+cal {wgt}:", round(np.abs(y-(wgt*p0+(1-wgt)*cal)).mean(),2))
for wgt in [0.3,0.4,0.5,0.6]:
    print(f"blend ridge+med4w {wgt}:", round(np.abs(y-(wgt*p0+(1-wgt)*med4)).mean(),2))
for a,b in [(0.45,0.2),(0.4,0.25),(0.5,0.15)]:
    print(f"ridge {a} med4w {b} cal {round(1-a-b,2)}:", round(np.abs(y-(a*p0+b*med4+(1-a-b)*cal)).mean(),2))


# ---- cell ----

import pandas as pd, numpy as np, agent_api

# Probe 1: can load_saved be used inside build_features?
def probe_fn(view, snapshot_day):
    hh = view.households
    try:
        t = load_saved("e008_level_shape.parquet")
        sub = t[t.snapshot_day == snapshot_day].set_index("household_key")
        sub = sub.reindex(hh)
        return sub[["z_med4w_hist", "sp28"]]
    except Exception as e:
        print("ERR", type(e).__name__, e); raise

bf = agent_api.build_features(probe_fn)
print("probe shape:", bf.shape, "cols:", list(bf.columns))
print("nan frac z_med4w_hist:", bf["z_med4w_hist"].isna().mean().round(4))
print(bf.head(3))

# Probe 2: weekly total-spend spikes (holiday-like weeks)?
snap = agent_api.snapshot(459)
tx = snap.transactions
wk = tx.groupby("week_no").sales_value.sum()
print("\nweekly total spend: mean %.0f std %.0f" % (wk.mean(), wk.std()))
print("top-10 weeks:", dict(wk.nlargest(10).round(0)))
print("bottom-5 weeks:", dict(wk.nsmallest(5).round(0)))
# per-active-household weekly to normalize for panel growth
cnt = tx.groupby("week_no").household_key.nunique()
wph = wk/cnt
print("per-hh weekly: mean %.1f std %.1f; top-8:" % (wph.mean(), wph.std()), dict(wph.nlargest(8).round(1)))


# ---- cell ----

import pandas as pd, numpy as np, agent_api

base = load_saved("e008_level_shape.parquet")
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
y = df[TARGET].values.astype(float)
feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")
for i, c in enumerate(cat_cols):
    Xn = pd.concat([Xn, pd.get_dummies(X[c].astype("category"), prefix=f"cat{i}", dummy_na=True).astype(float)], axis=1)
Xn = Xn.fillna(Xn.median())
Xv = Xn.values.astype(np.float64); n = len(y)

# leak-free candidate features from per-household day-sorted tx (filter day <= s per row)
snap = agent_api.snapshot(459)
tx = snap.transactions[["household_key","day","sales_value"]].sort_values(["household_key","day"])
keys = tx.household_key.values; days = tx.day.values.astype(int); sv = tx.sales_value.values.astype(float)
uniq, start = np.unique(keys, return_index=True)
h2idx = {h:i for i,h in enumerate(uniq)}
ends = np.r_[start[1:], len(keys)]
cs = np.concatenate([[0.0], np.cumsum(sv)])
def win_sum(h, lo, hi):
    i = h2idx.get(h)
    if i is None: return 0.0
    a = np.searchsorted(days[start[i]:ends[i]], lo, "left") + start[i]
    b = np.searchsorted(days[start[i]:ends[i]], hi, "right") + start[i]
    return cs[b]-cs[a]
def win_cnt(h, lo, hi):
    i = h2idx.get(h)
    if i is None: return 0.0
    a = np.searchsorted(days[start[i]:ends[i]], lo, "left")
    b = np.searchsorted(days[start[i]:ends[i]], hi, "right")
    return float(b-a)
def act_weeks(h, lo, hi):
    i = h2idx.get(h)
    if i is None: return 0.0
    dd = days[start[i]:ends[i]]
    m = (dd > lo) & (dd <= hi)
    return len(np.unique((dd[m]+8)//7)) if m.any() else 0.0

hh = df.household_key.values; sd = df.snapshot_day.values.astype(int)
sp28v = df.sp28.values.astype(float); sp56v = df.sp56.values.astype(float)
sp84v = df.sp84.values.astype(float); sp364v = df.sp364.values.astype(float)
seas1y = np.array([win_sum(h, s-364, s-336) for h, s in zip(hh, sd)])
aw4  = np.array([act_weeks(h, s-28, s) for h, s in zip(hh, sd)])
aw52 = np.array([act_weeks(h, s-364, s) for h, s in zip(hh, sd)])
ten  = df.tenure.values.astype(float)

eps = 1.0
cand = {}
cand["rr28_364"] = sp28v / np.maximum(sp364v/13.0, eps)
cand["rr28_84"]  = sp28v / np.maximum(sp84v/3.0, eps)
cand["rr56_364"] = sp56v / np.maximum(sp364v/6.5, eps)
cand["rr28_1y"]  = sp28v / np.maximum(seas1y, eps)
cand["aw_ratio"] = aw4 / np.maximum(aw52/13.0, 0.5)
cand["ten_rate"] = sp364v / np.maximum(ten, 28.0)
cand["med4w_x_rr"] = df.z_med4w_hist.values * np.clip(cand["rr28_364"], 0, 3)
for k in list(cand):
    cand["log_"+k] = np.log1p(np.clip(cand[k], 0, 50))

idx = np.random.RandomState(0).permutation(n); K=5
def ridge_fit(Xtr,ytr,a=100):
    mu=Xtr.mean(0); sdv=Xtr.std(0)+1e-9
    A=np.hstack([(Xtr-mu)/sdv,np.ones((len(Xtr),1))])
    G=A.T@A; p=G.shape[0]
    Gg=G.copy(); Gg[np.arange(p),np.arange(p)]+=a*np.r_[np.ones(p-1),0]
    return np.linalg.solve(Gg,A.T@ytr),mu,sdv
def cv_mae(Xall):
    s=0.0
    for k in range(K):
        va=idx[k::K]; tr=np.setdiff1d(idx,va)
        w,mu,sdv=ridge_fit(Xall[tr],y[tr])
        p=np.hstack([(Xall[va]-mu)/sdv,np.ones((len(va),1))])@w
        s+=np.abs(y[va]-p).sum()
    return round(s/n,2)
print("base ridge:", cv_mae(Xv))
# add candidates one block at a time
C = np.column_stack([cand[k] for k in cand])
print("with all candidates:", cv_mae(np.hstack([Xv, C])))
for k in cand:
    print(f"  +{k:14s}: {cv_mae(np.hstack([Xv, cand[k][:,None]]))}")
# correlation of candidates with y
print("\ncand corr with y:", {k: round(float(np.corrcoef(v,y)[0,1]),3) for k,v in cand.items()})


# ---- cell ----

import pandas as pd, numpy as np, agent_api

base = load_saved("e008_level_shape.parquet")
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
y = df[TARGET].values.astype(float)
feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")

# (a) rank-duplicate detection among numeric features
ranks = Xn.rank()
dup_groups = []
cols = list(Xn.columns)
seen = set()
for i, c in enumerate(cols):
    if c in seen: continue
    grp = [c]
    for c2 in cols[i+1:]:
        if c2 in seen: continue
        if ranks[c].equals(ranks[c2]):
            grp.append(c2); seen.add(c2)
    if len(grp) > 1: dup_groups.append(grp)
n_redundant = sum(len(g)-1 for g in dup_groups)
print("rank-duplicate groups:", len(dup_groups), "| redundant cols:", n_redundant)
for g in dup_groups[:15]: print("  ", g)

# (b) churn separation among rows with sp28>0
sp28 = df.sp28.values.astype(float); med4 = df.z_med4w_hist.values.astype(float)
dsl = df.days_since_last.values.astype(float); gapmed = df.gap_med.values.astype(float)
act = sp28 > 0
dorm = dsl / np.maximum(gapmed, 1.0)
print("\nAmong sp28>0 rows (n=%d): y==0 frac %.3f" % (act.sum(), (y[act]==0).mean()))
for name, v in [("dormancy dsl/gapmed", dorm), ("days_since_last", dsl), ("sp28", sp28), ("trend_84", df.trend_84.values)]:
    vv = np.asarray(v, dtype=float)
    print(f"  {name:20s} y==0: mean {vv[act & (y==0)].mean():7.2f} | y>0: mean {vv[act & (y>0)].mean():7.2f}")
# how big is the MAE contribution of zero rows?
print("MAE contribution from y==0 rows (if predict med4w):", round(np.abs(y[(y==0)] - med4[(y==0)]).mean(),1), "n=", (y==0).sum())

# (c) mix stability: cosine between dsh84_* and dsh364_* vectors
d84 = [c for c in df.columns if c.startswith("dsh84_")]
d364 = [c for c in df.columns if c.startswith("dsh364_")]
A = df[d84].fillna(0).values; B = df[d364].fillna(0).values
num = (A*B).sum(1); den = np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1) + 1e-9
mixstab = num/den
print("\nmixstab corr with y:", round(np.corrcoef(mixstab, y)[0,1],3),
      " corr with |y-med4w|:", round(np.corrcoef(mixstab, np.abs(y-med4))[0,1],3))

# (d) market calibration: mean y by snapshot vs week_mod52
print("\nmean y by snapshot:", {int(s): round(y[df.snapshot_day==s].mean(),1) for s in sorted(df.snapshot_day.unique())})
print("week_mod52 by snapshot:", {int(s): int(df.week_mod52[df.snapshot_day==s].iloc[0]) for s in sorted(df.snapshot_day.unique())})


# ---- cell ----

import pandas as pd, numpy as np, agent_api
base = load_saved("e008_level_shape.parquet")
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
y = df[TARGET].values.astype(float)
feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")
for i, c in enumerate(cat_cols):
    Xn = pd.concat([Xn, pd.get_dummies(X[c].astype("category"), prefix=f"cat{i}", dummy_na=True).astype(float)], axis=1)
Xn = Xn.fillna(Xn.median())
Xv = Xn.values.astype(np.float64); n = len(y)
sp28 = df.sp28.values.astype(float); med4 = df.z_med4w_hist.values.astype(float)
dsl = df.days_since_last.values.astype(float); gapmed = df.gap_med.values.astype(float)
act = sp28 > 0
dorm = dsl/np.maximum(gapmed,1.0)
print("dormancy nan frac:", np.isnan(dorm).mean())
actnan = act & ~np.isnan(dorm)
print("dormancy: y==0 mean %.2f | y>0 mean %.2f" % (np.nanmean(dorm[actnan & (y[actnan]==0)]), np.nanmean(dorm[actnan & (y[actnan]>0)])))
# AUROC-like separation for y==0
from numpy import argsort
def auroc(v, mask):
    vv = v[mask]; yy = (y[mask]==0).astype(int)
    ok = ~np.isnan(vv); vv=vv[ok]; yy=yy[ok]
    pos = vv[yy==1]; neg = vv[yy==0]
    if len(pos)==0 or len(neg)==0: return np.nan
    # rank-based
    allv = np.concatenate([pos,neg]); r = allv.argsort().argsort()+1
    rpos = r[:len(pos)].sum()
    return (rpos - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))
print("AUC(y==0) days_since_last:", round(auroc(dsl, act),3), "| dormancy:", round(auroc(dorm, act),3))
print("AUC(y==0) sp28:", round(auroc(sp28, act),3), "| trend_84:", round(auroc(np.asarray(df.trend_84,float), act),3))

# quick ridge on interaction-augmented design (offline, leak-free)
def add_inter(Xv, df):
    sp28v = df.sp28.values.astype(float); med4v = df.z_med4w_hist.values.astype(float)
    dslv = df.days_since_last.values.astype(float)
    inter = np.column_stack([
        sp28v*med4v/100.0, sp28v*dslv/100.0, med4v*dslv/100.0,
        sp28v*np.asarray(df.trend_84,float), med4v*np.asarray(df.trend_84,float),
        sp28v*np.asarray(df.wksp_std,float)/100.0,
        np.log1p(sp28v)*np.log1p(med4v),
        sp28v*np.asarray(df.nact84,float)/10.0, med4v*np.asarray(df.nact84,float)/10.0,
    ])
    return np.hstack([Xv, inter])
X2 = add_inter(Xv, df)
idx = np.random.RandomState(0).permutation(n); K=5
def cv_mae(Xall, a=100):
    s=0.0
    for k in range(K):
        va=idx[k::K]; tr=np.setdiff1d(idx,va)
        mu=Xall[tr].mean(0); sdv=Xall[tr].std(0)+1e-9
        A=np.hstack([(Xall[tr]-mu)/sdv,np.ones((len(tr),1))])
        G=A.T@A; p=G.shape[0]
        Gg=G.copy(); Gg[np.arange(p),np.arange(p)]+=a*np.r_[np.ones(p-1),0]
        w=np.linalg.solve(Gg,A.T@y[tr])
        pr=np.hstack([(Xall[va]-mu)/sdv,np.ones((len(va),1))])@w
        s+=np.abs(y[va]-pr).sum()
    return round(s/n,2)
print("ridge base:", cv_mae(Xv), "| with interactions:", cv_mae(X2))

# market-level calibration check: is there a snapshot-level level shift the model misses?
oof=np.zeros(n)
for k in range(K):
    va=idx[k::K]; tr=np.setdiff1d(idx,va)
    mu=Xv[tr].mean(0); sdv=Xv[tr].std(0)+1e-9
    A=np.hstack([(Xv[tr]-mu)/sdv,np.ones((len(tr),1))])
    G=A.T@A; p=G.shape[0]
    Gg=G.copy(); Gg[np.arange(p),np.arange(p)]+=100*np.r_[np.ones(p-1),0]
    w=np.linalg.solve(Gg,A.T@y[tr])
    oof[va]=np.hstack([(Xv[va]-mu)/sdv,np.ones((len(va),1))])@w
res = y-oof
print("\nresidual mean by snapshot:", {int(s): round(res[df.snapshot_day==s].mean(),1) for s in sorted(df.snapshot_day.unique())})
print("resid mean overall:", round(res.mean(),2), "| resid std:", round(res.std(),1))


# ---- cell ----

import pandas as pd, numpy as np, agent_api
base = load_saved("e008_level_shape.parquet")
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
y = df[TARGET].values.astype(float)
feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")
for i, c in enumerate(cat_cols):
    Xn = pd.concat([Xn, pd.get_dummies(X[c].astype("category"), prefix=f"cat{i}", dummy_na=True).astype(float)], axis=1)
Xn = Xn.fillna(Xn.median())
Xv = Xn.values.astype(np.float64); n = len(y)
sp28 = df.sp28.values.astype(float); med4 = df.z_med4w_hist.values.astype(float)
dsl = df.days_since_last.values.astype(float); gapmed = df.gap_med.values.astype(float)
act = sp28 > 0
dorm = dsl/np.maximum(gapmed,1.0)
def auroc(v, mask):
    vv = np.asarray(v,float)[mask]; yy = (y[mask]==0).astype(int)
    ok = ~np.isnan(vv); vv=vv[ok]; yy=yy[ok]
    if yy.sum()==0 or (1-yy).sum()==0: return np.nan
    allv = np.concatenate([vv[yy==1], vv[yy==0]]); r = allv.argsort().argsort()+1
    rpos = r[:yy.sum()].sum()
    return (rpos - yy.sum()*(yy.sum()+1)/2) / (yy.sum()*(1-yy).sum())
print("AUC(y==0 | act): dsl %.3f dorm %.3f sp28 %.3f trend84 %.3f" % (
    auroc(dsl,act), auroc(dorm,act), auroc(sp28,act), auroc(df.trend_84.values,act)))
print("mean dorm: y==0 %.2f y>0 %.2f" % (np.nanmean(dorm[act&(y==0)]), np.nanmean(dorm[act&(y>0)])))

def add_inter(Xv, df):
    sp28v = df.sp28.values.astype(float); med4v = df.z_med4w_hist.values.astype(float)
    dslv = df.days_since_last.values.astype(float)
    tr84 = np.asarray(df.trend_84, float); wstd = np.asarray(df.wksp_std, float); nact = np.asarray(df.nact84, float)
    inter = np.column_stack([sp28v*med4v/100.0, sp28v*dslv/100.0, med4v*dslv/100.0,
        sp28v*tr84, med4v*tr84, sp28v*wstd/100.0, np.log1p(sp28v)*np.log1p(med4v),
        sp28v*nact/10.0, med4v*nact/10.0])
    return np.hstack([Xv, inter])
X2 = add_inter(Xv, df)
idx = np.random.RandomState(0).permutation(n); K=5
def cv_mae(Xall, a=100):
    s=0.0
    for k in range(K):
        va=idx[k::K]; tr=np.setdiff1d(idx,va)
        mu=Xall[tr].mean(0); sdv=Xall[tr].std(0)+1e-9
        A=np.hstack([(Xall[tr]-mu)/sdv,np.ones((len(tr),1))])
        G=A.T@A; p=G.shape[0]
        Gg=G.copy(); Gg[np.arange(p),np.arange(p)]+=a*np.r_[np.ones(p-1),0]
        w=np.linalg.solve(Gg,A.T@y[tr])
        pr=np.hstack([(Xall[va]-mu)/sdv,np.ones((len(va),1))])@w
        s+=np.abs(y[va]-pr).sum()
    return round(s/n,2)
print("ridge base:", cv_mae(Xv), "| with interactions:", cv_mae(X2))

oof=np.zeros(n)
for k in range(K):
    va=idx[k::K]; tr=np.setdiff1d(idx,va)
    mu=Xv[tr].mean(0); sdv=Xv[tr].std(0)+1e-9
    A=np.hstack([(Xv[tr]-mu)/sdv,np.ones((len(tr),1))])
    G=A.T@A; p=G.shape[0]
    Gg=G.copy(); Gg[np.arange(p),np.arange(p)]+=100*np.r_[np.ones(p-1),0]
    w=np.linalg.solve(Gg,A.T@y[tr])
    oof[va]=np.hstack([(Xv[va]-mu)/sdv,np.ones((len(va),1))])@w
res = y-oof
print("\nresidual mean by snapshot:", {int(s): round(res[df.snapshot_day==s].mean(),1) for s in sorted(df.snapshot_day.unique())})
print("resid mean overall:", round(res.mean(),2), "| resid std:", round(res.std(),1))
