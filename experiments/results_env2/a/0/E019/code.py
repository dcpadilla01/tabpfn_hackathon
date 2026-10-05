
import agent_api, pandas as pd, numpy as np

feats = agent_api.load_saved("feats_v4.parquet")
print("feats_v4:", feats.shape)
print(list(feats.columns))

tt = agent_api.train_targets()
print("\ntargets:", tt.shape, sorted(tt.snapshot_day.unique()))
m = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
print("\nper-snapshot target stats (train):")
print(m[m.future_spend_4w.notna()].groupby("snapshot_day").future_spend_4w.agg(["count","mean","median"]).round(1))

print("\nNaN counts (nonzero):")
nn = feats.isna().sum()
print(nn[nn>0])

# saved validation predictions
names = ["pred_e004","pred_e005","pred_e007","pred_e008","pred_e011","pred_e012","pred_e013","pred_e014","pred_e015","pred_e017"]
base = agent_api.load_saved("pred_e013.parquet")
print("\npred_e013:", base.shape, sorted(base.snapshot_day.unique()))
M = base[["household_key","snapshot_day"]].copy()
for nm in names:
    p = agent_api.load_saved(nm + ".parquet")
    M = M.merge(p[["household_key","snapshot_day","prediction"]].rename(columns={"prediction":nm}), on=["household_key","snapshot_day"])
P = M.drop(columns=["household_key","snapshot_day"])
print("\npred corr:")
print(P.corr().round(3).mean(axis=1).round(4))
print("\npred means per snapshot (e013):")
print(base.groupby("snapshot_day").prediction.agg(["count","mean","median"]).round(1))


# ---- cell ----

import agent_api, pandas as pd, numpy as np

# 1) Aggregate weekly spend seasonality over the 2 years
snap = agent_api.snapshot(459)
tx = snap.transactions
wk = tx.groupby("week_no").sales_value.sum()
print("weekly total spend: mean %.0f std %.0f cv %.2f, min wk %d (%.0f), max wk %d (%.0f)" %
      (wk.mean(), wk.std(), wk.std()/wk.mean(), wk.idxmin(), wk.min(), wk.idxmax(), wk.max()))
# print a coarse profile
prof = wk.reset_index(); prof["blk"] = (prof.week_no-1)//13
print(prof.groupby("blk").sales_value.mean().round(0))

# 2) verify lag1_spend alignment in feats_v3: spend in [s-27, s]?
f3 = agent_api.load_saved("feats_v3.parquet")
hh = f3.household_key.iloc[0]
row = f3.iloc[0]
s = int(row.snapshot_day)
h = agent_api.history(hh, as_of_day=s)
m = (h.day >= s-27) & (h.day <= s)
print("\ncheck lag1: hh", hh, "snap", s, "computed", h[m].sales_value.sum().round(2), "feat", row.lag1_spend)
m2 = (h.day >= s-55) & (h.day <= s-28)
print("check lag2: computed", h[m2].sales_value.sum().round(2), "feat", row.lag2_spend)

# 3) does a year-ago aligned window exist for most rows? tenure distribution
tt = agent_api.train_targets()
ten = f3[["household_key","snapshot_day","tenure"]]
print("\ntenure describe:"); print(ten.tenure.describe().round(0))
print("share tenure>=364:", (ten.tenure>=364).mean().round(3))


# ---- cell ----

import agent_api, pandas as pd, numpy as np

f3 = agent_api.load_saved("feats_v3.parquet")
print("feats_v3 cols:", list(f3.columns)[:40])

# per-household weekly spend seasonality (normalized by # active households)
snap = agent_api.snapshot(459)
tx = snap.transactions
g = tx.groupby(["week_no","household_key"]).sales_value.sum().reset_index()
per_hh = g.groupby("week_no").sales_value.mean()
prof = per_hh.reset_index(); prof["blk"] = (prof.week_no-1)//13
print("\nper-active-household weekly spend by 13wk block:")
print(prof.groupby("blk").sales_value.mean().round(1))
# finer: 4-week blocks
prof["blk4"] = (prof.week_no-1)//4
b4 = prof.groupby("blk4").sales_value.mean().round(1)
print("\nper-hh weekly spend by 4wk block:")
print(b4.to_string())

# lag alignment check
hh = f3.household_key.iloc[0]; s = int(f3.snapshot_day.iloc[0])
h = agent_api.history(hh, as_of_day=s)
print("\ncheck hh", hh, "snap", s,
      "lag1 calc", h[(h.day>=s-27)&(h.day<=s)].sales_value.sum().round(2),
      "feat", f3.lag1_spend.iloc[0],
      "| lag2 calc", h[(h.day>=s-55)&(h.day<=s-28)].sales_value.sum().round(2),
      "feat", f3.lag2_spend.iloc[0])


# ---- cell ----

import agent_api, pandas as pd, numpy as np, time
feats = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
print(feats.dtypes.value_counts())
obj_cols = [c for c in feats.columns if feats[c].dtype==object]
print("object cols:", obj_cols)
for c in obj_cols[:3]: print(c, feats[c].unique()[:8])

m = feats.merge(tt, on=["household_key","snapshot_day"])
y = m.future_spend_4w
print("\ntarget: zero share %.3f, quantiles:" % (y==0).mean(), np.quantile(y,[.1,.25,.5,.75,.9,.95,.99]).round(1))

# timing test: one xgb config on all train rows
import xgboost as xgb
FE = [c for c in feats.columns if c not in ("household_key","snapshot_day")]
X = m[FE].copy()
for c in obj_cols:
    X[c] = X[c].astype("category")
print("X shape", X.shape)
t0=time.time()
mdl = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, max_depth=5, min_child_weight=40,
                       learning_rate=0.08, n_estimators=400, n_jobs=-1, tree_method="hist", enable_categorical=True)
mdl.fit(X, y)
print("fit time %.1fs" % (time.time()-t0))
t0=time.time()
_ = mdl.predict(X.iloc[:5000])
print("pred time %.2fs" % (time.time()-t0))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, time, xgboost as xgb
feats = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
train = feats.merge(tt, on=["household_key","snapshot_day"])
FE = [c for c in feats.columns if c not in ("household_key","snapshot_day")]
obj = [c for c in FE if str(train[c].dtype)=="category"]

def prep(df):
    X = df[FE].copy()
    for c in obj: X[c] = X[c].astype("category")
    return X

configs = [dict(max_depth=d, min_child_weight=w) for d in (4,5,6) for w in (20,40,60)][:8]
print("configs:", configs)

t0 = time.time()
rows = []
for s in sorted(train.snapshot_day.unique()):
    tr = train[train.snapshot_day != s]
    Xtr, ytr = prep(tr), tr.future_spend_4w.values
    va = train[train.snapshot_day == s]
    Xva = prep(va)
    preds = []
    for cfg in configs:
        for seed in (7, 13):
            mdl = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
                                   learning_rate=0.08, n_estimators=400, n_jobs=-1,
                                   tree_method="hist", enable_categorical=True,
                                   random_state=seed, **cfg)
            mdl.fit(Xtr, ytr)
            preds.append(mdl.predict(Xva))
    p = np.mean(preds, axis=0)
    rows.append(pd.DataFrame({"household_key": va.household_key.values, "snapshot_day": s,
                              "oof": p, "y": va.future_spend_4w.values}))
oof = pd.concat(rows, ignore_index=True)
print("OOF done %.0fs" % (time.time()-t0))
print("OOF MAE %.3f" % np.abs(oof.oof - oof.y).mean())

# save OOF for reuse
agent_api.save_table(oof, "oof_e013.parquet")


# ---- cell ----

import agent_api, pandas as pd, numpy as np, time, xgboost as xgb
feats = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
train = feats.merge(tt, on=["household_key","snapshot_day"])
FE = [c for c in feats.columns if c not in ("household_key","snapshot_day")]
obj = [c for c in FE if str(train[c].dtype)=="category"]

def prep(df):
    X = df[FE].copy()
    for c in obj: X[c] = X[c].astype("category")
    return X

configs = [(4,20),(5,40),(6,20),(5,60)]
t0 = time.time()
rows = []
for s in sorted(train.snapshot_day.unique()):
    tr = train[train.snapshot_day != s]
    Xtr, ytr = prep(tr), tr.future_spend_4w.values
    va = train[train.snapshot_day == s]
    Xva = prep(va)
    preds = []
    for d,w in configs:
        for seed in (7,13):
            mdl = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
                                   learning_rate=0.08, n_estimators=400, n_jobs=4,
                                   tree_method="hist", enable_categorical=True,
                                   random_state=seed, max_depth=d, min_child_weight=w)
            mdl.fit(Xtr, ytr)
            preds.append(mdl.predict(Xva))
    p = np.mean(preds, axis=0)
    rows.append(pd.DataFrame({"household_key": va.household_key.values, "snapshot_day": s,
                              "oof": p, "y": va.future_spend_4w.values}))
oof = pd.concat(rows, ignore_index=True)
print("OOF done %.0fs" % (time.time()-t0))
print("OOF MAE %.3f" % np.abs(oof.oof - oof.y).mean())
agent_api.save_table(oof, "oof_e013.parquet")


# ---- cell ----

import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e013.parquet")
y = oof.y.values; p = oof.oof.values

# 1) per-snapshot multiplicative calibration (leave-one-out style: fit on other snapshots)
oof["resid"] = y - p
for s in sorted(oof.snapshot_day.unique()):
    tr = oof[oof.snapshot_day != s]
    a = tr.y.sum()/tr.oof.sum()
    va = oof[oof.snapshot_day == s]
    mae = np.abs(va.oof*a - va.y).mean()
    print("snap %3d: out-scale %.3f  MAE %.3f -> %.3f" % (s, a, np.abs(va.oof-va.y).mean(), mae))

# 2) global isotonic on oof
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds="clip", y_min=0)
iso.fit(p, y)
print("\nisotonic global MAE %.3f" % np.abs(iso.predict(p)-y).mean())
print("iso mapping:", np.round(np.quantile(p,[0,.1,.25,.5,.75,.9,.99]),1),
      "->", np.round(iso.predict(np.quantile(p,[0,.1,.25,.5,.75,.9,.99])),1))

# 3) residual vs prediction size (slope of y on p)
lo = pd.qcut(p, 10, duplicates="drop")
print("\nbin means: p, y, mean resid")
print(oof.groupby(lo, observed=True).agg(p=("oof","mean"), y=("y","mean"), n=("y","size")).round(1))


# ---- cell ----

import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e013.parquet")
y = oof.y.values; p = oof.oof.values
base = np.abs(p-y).mean()
print("base OOF MAE %.3f" % base)

# median resid by pred-bin (out-of-fold per snapshot)
oof["bin"] = pd.qcut(p, 10, duplicates="drop")
print("\nmedian resid per bin:")
print(oof.groupby("bin", observed=True).agg(p=("oof","median"), medres=("resid","median"), n=("y","size")).round(1))

# grid: global affine a*p+b (MAE-optimal shrink)
best=(1e9,None)
for a in np.arange(0.90,1.11,0.02):
    for b in range(-6,7,2):
        m = np.abs(a*p+b-y).mean()
        if m<best[0]: best=(m,(round(a,2),b))
print("\nbest affine:", best)

# blend with raw predictors
f4 = agent_api.load_saved("feats_v4.parquet")
oof2 = oof.merge(f4[["household_key","snapshot_day","exp4w_blend","lag1_spend","spend_28","wk_mean8"]], on=["household_key","snapshot_day"])
for col in ["exp4w_blend","lag1_spend","spend_28","wk_mean8"]:
    r = oof2[col].fillna(0).values
    print("%s alone MAE %.3f" % (col, np.abs(r-y).mean()))
    bestw=(1e9,None)
    for w in np.arange(0.6,1.01,0.05):
        m = np.abs(w*p+(1-w)*r-y).mean()
        if m<bestw[0]: bestw=(m,round(w,2))
    print("  best blend w=%.2f MAE %.3f" % (bestw[1], bestw[0]))

# per-bin additive median correction (fit on other snapshots, LOO)
tot=0; n=0
for s in sorted(oof.snapshot_day.unique()):
    tr = oof[oof.snapshot_day!=s]; va = oof[oof.snapshot_day==s]
    corr = tr.groupby("bin", observed=True).resid.median()
    adj = va.bin.map(corr).fillna(0).values
    tot += np.abs(va.oof.values+adj-va.y.values).sum(); n += len(va)
print("\nper-bin median correction MAE %.3f" % (tot/n))


# ---- cell ----

import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e013.parquet")
oof["resid"] = oof.y - oof.oof
y = oof.y.values; p = oof.oof.values
print("base OOF MAE %.3f" % np.abs(p-y).mean())

oof["bin"] = pd.qcut(p, 10, duplicates="drop")
print("median resid per bin:")
print(oof.groupby("bin", observed=True).agg(p=("oof","median"), medres=("resid","median"), n=("y","size")).round(1))

best=(1e9,None)
for a in np.arange(0.90,1.11,0.02):
    for b in range(-6,7,2):
        m = np.abs(a*p+b-y).mean()
        if m<best[0]: best=(m,(round(a,2),b))
print("best affine:", best)

f4 = agent_api.load_saved("feats_v4.parquet")
oof2 = oof.merge(f4[["household_key","snapshot_day","exp4w_blend","lag1_spend","spend_28","wk_mean8"]], on=["household_key","snapshot_day"])
for col in ["exp4w_blend","lag1_spend","spend_28","wk_mean8"]:
    r = oof2[col].fillna(0).values
    bestw=(1e9,None)
    for w in np.arange(0.6,1.01,0.05):
        m = np.abs(w*p+(1-w)*r-y).mean()
        if m<bestw[0]: bestw=(m,round(w,2))
    print("%s: alone MAE %.3f | best blend w=%.2f MAE %.3f" % (col, np.abs(r-y).mean(), bestw[1], bestw[0]))

tot=0; n=0
for s in sorted(oof.snapshot_day.unique()):
    tr = oof[oof.snapshot_day!=s]; va = oof[oof.snapshot_day==s]
    corr = tr.groupby("bin", observed=True).resid.median()
    adj = va.bin.map(corr).fillna(0).values
    tot += np.abs(va.oof.values+adj-va.y.values).sum(); n += len(va)
print("per-bin median correction MAE %.3f" % (tot/n))


# ---- cell ----

import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e013.parquet").copy()
oof["resid"] = oof.y - oof.oof
y = oof.y.values; p = oof.oof.values
print("base OOF MAE %.3f | pred min %.1f max %.1f" % (np.abs(p-y).mean(), p.min(), p.max()))

# A) clip negatives
pc = np.clip(p, 0, None)
print("A clip@0 MAE %.3f" % np.abs(pc-y).mean())

# B) per-pred-decile median-resid correction, fit on OTHER snapshots (LOO)
oof["bin"] = pd.qcut(p, 10, duplicates="drop").astype(str)
tot=np.zeros(3); n=0
for s in sorted(oof.snapshot_day.unique()):
    tr = oof[oof.snapshot_day!=s]; va = oof[oof.snapshot_day==s]
    corr = tr.groupby("bin").resid.median()
    adj = va.bin.map(corr).astype(float).fillna(0).values
    pv = va.oof.values
    tot[0]+=np.abs(pv+adj-va.y.values).sum()
    tot[1]+=np.abs(np.clip(pv+adj,0,None)-va.y.values).sum()
    tot[2]+=np.abs(pv-va.y.values).sum(); n+=len(va)
print("B bin-corr MAE %.3f | +clip %.3f | base %.3f" % (tot[0]/n, tot[1]/n, tot[2]/n))

# C) aggregate bias: OOF mean vs actual mean per snapshot
print("\nC per-snapshot mean: actual vs oof")
g = oof.groupby("snapshot_day").agg(y=("y","mean"), p=("oof","mean"))
g["ratio"] = (g.y/g.p).round(3)
print(g.round(1))

# D) oracle household fixed effect (cheating diagnostic): predict hh mean of y over other snapshots
tot=0; n=0
for s in sorted(oof.snapshot_day.unique()):
    tr = oof[oof.snapshot_day!=s].groupby("household_key").y.mean()
    va = oof[oof.snapshot_day==s]
    pr = va.household_key.map(tr).fillna(va.y.mean()).values
    tot += np.abs(pr-va.y.values).sum(); n+=len(va)
print("\nD oracle hh-FE MAE %.3f (vs model 61.5)" % (tot/n))

# E) LOO blend with spend_28
f4 = agent_api.load_saved("feats_v4.parquet")
oof = oof.merge(f4[["household_key","snapshot_day","spend_28"]], on=["household_key","snapshot_day"])
tot=0; n=0
for s in sorted(oof.snapshot_day.unique()):
    tr = oof[oof.snapshot_day!=s]; va = oof[oof.snapshot_day==s]
    r = va.spend_28.fillna(0).values
    best=(1e9,1)
    for w in np.arange(0.85,1.001,0.01):
        m = np.abs(w*va.oof.values+(1-w)*r-va.y.values).mean()
        if m<best[0]: best=(m,w)
    tot+=best[0]*len(va); n+=len(va)
print("E LOO blend spend_28 MAE %.3f" % (tot/n))


# ---- cell ----

import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e013.parquet").copy()
oof["resid"] = oof.y - oof.oof
f4 = agent_api.load_saved("feats_v4.parquet")
oof = oof.merge(f4[["household_key","snapshot_day","spend_28"]], on=["household_key","snapshot_day"])
oof["bin"] = pd.qcut(oof.oof, 10, duplicates="drop").astype(str)
snaps = sorted(oof.snapshot_day.unique())

def eval_stack(s, use_bin=True, use_clip=True, w=0.97, kappa=0.0, hh_shrink=False):
    tr = oof[oof.snapshot_day!=s]; va = oof[oof.snapshot_day==s]
    pv = va.oof.values.copy(); yv = va.y.values
    if use_bin:
        corr = tr.groupby("bin").resid.median()
        pv = pv + va.bin.map(corr).astype(float).fillna(0).values
    if hh_shrink:
        g = tr.groupby("household_key").agg(my=("y","mean"), mp=("oof","mean"))
        g["r"] = (g.my/g.mp).clip(0.5,2.0)
        r = va.household_key.map(g.r).fillna(1.0).values**kappa
        pv = pv*r
    if use_clip: pv = np.clip(pv, 0, None)
    r28 = va.spend_28.fillna(0).values
    pv = w*pv + (1-w)*r28
    return np.abs(pv-yv).mean(), len(va)

# grid over stack options (honest LOO per snapshot)
res = []
for use_bin in [True, False]:
    for w in [0.95, 0.97, 1.0]:
        for kappa in [0.0, 0.5]:
            tot=0; n=0
            for s in snaps:
                m,k = eval_stack(s, use_bin, True, w, kappa, kappa>0)
                tot+=m*k; n+=k
            res.append((round(tot/n,3), use_bin, w, kappa))
for r in sorted(res): print(r)


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb

feats = agent_api.load_saved("feats_v4.parquet")
oof = agent_api.load_saved("oof_e013.parquet")
oof["resid"] = oof.y - oof.oof
oof["bin"] = pd.qcut(oof.oof, 10, duplicates="drop").astype(str)
corr = oof.groupby("bin").resid.median()
edges = np.unique(np.quantile(oof.oof, np.linspace(0, 1, 11)))
print("corr:", corr.round(1).to_dict())

tt = agent_api.train_targets()
train = feats.merge(tt[["household_key","snapshot_day"]], on=["household_key","snapshot_day"])
val = feats[feats.snapshot_day >= 459].copy()
FE = [c for c in feats.columns if c not in ("household_key","snapshot_day")]
obj = [c for c in FE if str(train[c].dtype) == "category"]

def prep(df):
    X = df[FE].copy()
    for c in obj: X[c] = X[c].astype("category")
    return X

Xtr, ytr = prep(train), train.future_spend_4w.values
Xva = prep(val)
preds = []
for d, w in [(4,20),(5,40),(6,20),(5,60)]:
    for seed in (7, 13):
        m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, learning_rate=0.08,
                             n_estimators=400, n_jobs=4, tree_method="hist", enable_categorical=True,
                             random_state=seed, max_depth=d, min_child_weight=w)
        m.fit(Xtr, ytr)
        preds.append(m.predict(Xva))
p = np.mean(preds, axis=0)

bins = pd.cut(pd.Series(p), edges, include_lowest=True).astype(str)
adj = bins.map(corr).astype(float).fillna(0).values
p2 = np.clip(p + adj, 0, None)
r28 = val.spend_28.fillna(0).values
final = 0.97 * p2 + 0.03 * r28

out = pd.DataFrame({"household_key": val.household_key.values,
                    "snapshot_day": val.snapshot_day.values, "prediction": final})
print("rows:", len(out), "| mean %.1f median %.1f min %.1f" % (final.mean(), np.median(final), final.min()))
agent_api.save_table(out, "pred_e019.parquet")


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb

feats = agent_api.load_saved("feats_v4.parquet")
oof = agent_api.load_saved("oof_e013.parquet")
oof["resid"] = oof.y - oof.oof
oof["bin"] = pd.qcut(oof.oof, 10, duplicates="drop").astype(str)
corr = oof.groupby("bin").resid.median()
edges = np.unique(np.quantile(oof.oof, np.linspace(0, 1, 11)))

tt = agent_api.train_targets()
train = feats.merge(tt, on=["household_key","snapshot_day"])
val = feats[feats.snapshot_day >= 459].copy()
FE = [c for c in feats.columns if c not in ("household_key","snapshot_day")]
obj = [c for c in FE if str(train[c].dtype) == "category"]

def prep(df):
    X = df[FE].copy()
    for c in obj: X[c] = X[c].astype("category")
    return X

Xtr, ytr = prep(train), train.future_spend_4w.values
Xva = prep(val)
preds = []
for d, w in [(4,20),(5,40),(6,20),(5,60)]:
    for seed in (7, 13):
        m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, learning_rate=0.08,
                             n_estimators=400, n_jobs=4, tree_method="hist", enable_categorical=True,
                             random_state=seed, max_depth=d, min_child_weight=w)
        m.fit(Xtr, ytr)
        preds.append(m.predict(Xva))
p = np.mean(preds, axis=0)

bins = pd.cut(pd.Series(p), edges, include_lowest=True).astype(str)
adj = bins.map(corr).astype(float).fillna(0).values
p2 = np.clip(p + adj, 0, None)
r28 = val.spend_28.fillna(0).values
final = 0.97 * p2 + 0.03 * r28

out = pd.DataFrame({"household_key": val.household_key.values,
                    "snapshot_day": val.snapshot_day.values, "prediction": final})
print("rows:", len(out), "| mean %.1f median %.1f min %.1f max %.1f" % (final.mean(), np.median(final), final.min(), final.max()))
agent_api.save_table(out, "pred_e019.parquet")
