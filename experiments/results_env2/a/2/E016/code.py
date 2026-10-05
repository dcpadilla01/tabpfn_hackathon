import agent_api as A
import pandas as pd, numpy as np

for name in ["pred_seasonal","pred_e008","pred_e009","pred_e010_blend","pred_e006","oof_e008","oof_harness"]:
    try:
        df = A.load_saved(name+".parquet")
        print(name, df.shape, list(df.columns)[:8])
        print(df.head(3))
        print()
    except Exception as e:
        print(name, "ERR", type(e).__name__, e)


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
print(tt.shape, tt.columns.tolist())

oof8 = A.load_saved("oof_e008.parquet")
print(oof8.shape, oof8.columns.tolist())
m = tt.merge(oof8, on=["household_key","snapshot_day"], how="inner")
print("merged", m.shape)

def mae(p, y): return np.mean(np.abs(p-y))
y = m.future_spend_4w.values
for c in ["oof_sq","oof_med","oof_log"]:
    print(c, round(mae(m[c].values, y),3))

# grid search blend weights over the three oof components
best=(1e9,None)
import itertools
for w1 in np.arange(0,1.01,0.1):
    for w2 in np.arange(0,1.01-w1,0.1):
        w3=1-w1-w2
        p=w1*m.oof_med.values+w2*m.oof_sq.values+w3*m.oof_log.values
        v=mae(p,y)
        if v<best[0]: best=(v,(round(w1,2),round(w2,2),round(w3,2)))
print("best blend oof:", best)


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

v3 = A.load_saved("feats_v3.parquet")
se = A.load_saved("feats_seasonal.parquet")
print("v3", v3.shape, v3.columns.tolist()[:12], "...")
print("se", se.shape, se.columns.tolist())
tt = A.train_targets()
print("train rows", tt.shape)
# check val/train split of feature tables
print(v3.snapshot_day.value_counts().sort_index())
print(se.snapshot_day.value_counts().sort_index())
# overlap of keys
k_v3 = set(map(tuple, v3[["household_key","snapshot_day"]].values))
k_se = set(map(tuple, se[["household_key","snapshot_day"]].values))
print("v3==se keys:", k_v3==k_se)
print("NaN share in seasonal cols:")
print(se.drop(columns=["household_key","snapshot_day"]).isna().mean().round(3))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, time
import xgboost as xgb
print("xgb", xgb.__version__)

v3 = A.load_saved("feats_v3.parquet")
se = A.load_saved("feats_seasonal.parquet")
m = v3.merge(se.drop(columns=["household_key","snapshot_day"]), left_index=True, right_index=True) if False else v3.merge(se, on=["household_key","snapshot_day"], how="inner", suffixes=("","_se"))
print("merged", m.shape)
feat_cols = [c for c in m.columns if c not in ("household_key","snapshot_day")]
print("n_feat", len(feat_cols))
print(m[feat_cols].dtypes.value_counts())
A.save_table(m, "feats_all_e016.parquet")

tt = A.train_targets()
tr = m.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("train rows w/ target:", tr.shape, "target mean", tr.future_spend_4w.mean().round(2), "median", tr.future_spend_4w.median())

# time one quantile fit
X = tr[feat_cols].values.astype(np.float32); y = tr.future_spend_4w.values.astype(np.float32)
t0=time.time()
mdl = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=10,
                       subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                       objective="reg:quantileerror", quantile_alpha=0.5, n_jobs=8, random_state=1)
mdl.fit(X, y)
print("300-tree quantile fit secs:", round(time.time()-t0,1))
t0=time.time()
p = mdl.predict(X[:5000])
print("predict secs:", round(time.time()-t0,2))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

m = A.load_saved("feats_all_e016.parquet")
feat_cols = [c for c in m.columns if c not in ("household_key","snapshot_day")]
tt = A.train_targets()
tr = m.merge(tt, on=["household_key","snapshot_day"], how="inner")
X = tr[feat_cols].values.astype(np.float32); y = tr.future_spend_4w.values.astype(np.float32)
days = tr.snapshot_day.values

def med_xgb(seed, n=300, lr=0.05, md=6):
    return xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        objective="reg:quantileerror", quantile_alpha=0.5, n_jobs=8, random_state=seed)
def sq_xgb(seed=1, n=300, lr=0.05, md=7):
    return xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=8, random_state=seed)
def hgb_q50(seed=1, it=200, lr=0.06):
    return HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=it, learning_rate=lr,
        max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)

folds = [(days<=207), (days>207)&(days<=291), (days>291)&(days<=375), (days>375)]
oof = {k: np.zeros(len(y)) for k in ["med","sq","hgb"]}
t0=time.time()
for fi,f in enumerate(folds):
    Xtr, ytr = X[~f], y[~f]; Xte = X[f]
    for name, mk in [("med", lambda: med_xgb(1)), ("sq", lambda: sq_xgb(1)), ("hgb", lambda: hgb_q50(1))]:
        mod = mk(); mod.fit(Xtr, ytr); oof[name][f] = mod.predict(Xte)
    print("fold", fi, round(time.time()-t0,1), "s")
def mae(p): return np.mean(np.abs(p-y))
for k in oof: print(k, round(mae(oof[k]),3))
best=(1e9,None)
for w1 in np.arange(0,1.01,0.1):
    for w2 in np.arange(0,1.01-w1,0.1):
        w3=round(1-w1-w2,2)
        p=w1*oof["med"]+w2*oof["sq"]+w3*oof["hgb"]
        v=mae(p)
        if v<best[0]: best=(v,(round(w1,2),round(w2,2),w3))
print("best weights (med,sq,hgb):", best)


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

m = A.load_saved("feats_all_e016.parquet")
feat_cols = [c for c in m.columns if c not in ("household_key","snapshot_day")]
tt = A.train_targets()
tr = m.merge(tt, on=["household_key","snapshot_day"], how="inner")
X = tr[feat_cols].values.astype(np.float32); y = tr.future_spend_4w.values.astype(np.float32)
days = tr.snapshot_day.values

def med_xgb(seed=1, n=300, lr=0.05, md=6):
    return xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        objective="reg:quantileerror", quantile_alpha=0.5, n_jobs=8, random_state=seed)
def sq_xgb(seed=1, n=300, lr=0.05, md=7):
    return xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=8, random_state=seed)
def hgb_q50(seed=1, it=200, lr=0.06):
    return HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=it, learning_rate=lr,
        max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)

folds = [(days<=207), (days>207)&(days<=291), (days>291)&(days<=375), (days>375)]
oof = {k: np.zeros(len(y)) for k in ["med","sq","hgb"]}
t0=time.time()
for fi,f in enumerate(folds):
    Xtr, ytr = X[~f], y[~f]; Xte = X[f]
    for name, mk in [("med", lambda: med_xgb(1)), ("sq", lambda: sq_xgb(1)), ("hgb", lambda: hgb_q50(1))]:
        mod = mk()
        if name=="hgb":
            med = np.nanmedian(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            mod.fit(np.where(np.isfinite(Xtr), Xtr, med[None,:]), ytr)
            oof[name][f] = mod.predict(np.where(np.isfinite(Xte), Xte, med[None,:]))
        else:
            mod.fit(Xtr, ytr); oof[name][f] = mod.predict(Xte)
    print("fold", fi, round(time.time()-t0,1), "s")
def mae(p): return np.mean(np.abs(p-y))
for k in oof: print(k, round(mae(oof[k]),3))
best=(1e9,None)
for w1 in np.arange(0,1.01,0.1):
    for w2 in np.arange(0,1.01-w1,0.1):
        w3=round(1-w1-w2,2)
        p=w1*oof["med"]+w2*oof["sq"]+w3*oof["hgb"]
        v=mae(p)
        if v<best[0]: best=(v,(round(w1,2),round(w2,2),w3))
print("best weights (med,sq,hgb):", best)


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

mall = A.load_saved("feats_all_e016.parquet")
v3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()

def prep(df):
    t = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    fc = [c for c in df.columns if c not in ("household_key","snapshot_day")]
    return t[fc].values.astype(np.float32), t.future_spend_4w.values.astype(np.float32), t.snapshot_day.values

Xa, ya, da = prep(mall)   # v3+seasonal
Xv, yv, dv = prep(v3)     # v3 only

def med_xgb(seed): return xgb.XGBRegressor(n_estimators=600, learning_rate=0.04, max_depth=6, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist", objective="reg:quantileerror", quantile_alpha=0.5,
    n_jobs=8, random_state=seed)
def sq_xgb(seed): return xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=8, random_state=seed)
def hgb_q(seed): return HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=300, learning_rate=0.06,
    max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)
def hgb_s(seed): return HistGradientBoostingRegressor(loss="squared_error", max_iter=300, learning_rate=0.06,
    max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)

def fill_nan(Xtr, Xte):
    med = np.nanmedian(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(Xtr), Xtr, med[None,:]), np.where(np.isfinite(Xte), Xte, med[None,:])

def run_cv(X, y, days, learners):
    folds = [(days<=207), (days>207)&(days<=291), (days>291)&(days<=375), (days>375)]
    oof = {k: np.zeros(len(y)) for k in learners}
    t0=time.time()
    for fi,f in enumerate(folds):
        Xtr, ytr, Xte = X[~f], y[~f], X[f]
        for name, mk in learners.items():
            mod = mk()
            if name.startswith("hgb"):
                Xtr2, Xte2 = fill_nan(Xtr, Xte)
                mod.fit(Xtr2, ytr); oof[name][f] = mod.predict(Xte2)
            else:
                mod.fit(Xtr, ytr); oof[name][f] = mod.predict(Xte)
        print("  fold", fi, round(time.time()-t0,1), "s", flush=True)
    return oof

def mae(p,y): return np.mean(np.abs(p-y))

learners = {"med": lambda: med_xgb(1), "sq": lambda: sq_xgb(1), "hgbq": lambda: hgb_q(1), "hgbs": lambda: hgb_s(1)}
print("CV v3+seasonal:"); oa = run_cv(Xa, ya, da, learners)
for k,v in oa.items(): print(" ", k, round(mae(v,ya),3))
print("CV v3 only (med learner):"); ov = run_cv(Xv, yv, dv, {"med": lambda: med_xgb(1)})
print("  med(v3)", round(mae(ov["med"],yv),3))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

mall = A.load_saved("feats_all_e016.parquet")   # v3 + seasonal
v3  = A.load_saved("feats_v3.parquet")
tt = A.train_targets()

def prep(df):
    t = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    fc = [c for c in df.columns if c not in ("household_key","snapshot_day")]
    return t[fc].values.astype(np.float32), t.future_spend_4w.values.astype(np.float32), t.snapshot_day.values

Xa, ya, da = prep(mall)
Xv, yv, dv = prep(v3)
assert (da==dv).all() and (ya==yv).all()
days = da; y = ya

def med_xgb(seed): return xgb.XGBRegressor(n_estimators=600, learning_rate=0.04, max_depth=6, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist", objective="reg:quantileerror", quantile_alpha=0.5,
    n_jobs=8, random_state=seed)
def hgb_q(seed): return HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=300, learning_rate=0.06,
    max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)

def fill_nan(Xtr, Xte):
    med = np.nanmedian(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(Xtr), Xtr, med[None,:]), np.where(np.isfinite(Xte), Xte, med[None,:])

folds = [(days<=207), (days>207)&(days<=291), (days>291)&(days<=375), (days>375)]
oof = {k: np.zeros(len(y)) for k in ["med_v3","med_all","hgbq_v3","hgbq_all"]}
t0=time.time()
for fi,f in enumerate(folds):
    for name, (X, mk) in {"med_v3":(Xv,med_xgb), "med_all":(Xa,med_xgb), "hgbq_v3":(Xv,hgb_q), "hgbq_all":(Xa,hgb_q)}.items():
        mod = mk(1)
        if name.startswith("hgbq"):
            Xtr2, Xte2 = fill_nan(X[~f], X[f]); mod.fit(Xtr2, y[~f]); oof[name][f] = mod.predict(Xte2)
        else:
            mod.fit(X[~f], y[~f]); oof[name][f] = mod.predict(X[f])
    print("fold", fi, round(time.time()-t0,1), "s", flush=True)

def mae(p): return np.mean(np.abs(p-y))
for k in oof: print(k, round(mae(oof[k]),3))
print("last fold only (days>375):")
f = folds[3]
for k in oof: print(" ", k, round(np.mean(np.abs(oof[k][f]-y[f])),3))

# save oof for later weight search
odf = pd.DataFrame({"household_key": tt.household_key.values if False else None})
# rebuild keys: merged train rows order
tr = mall.merge(tt, on=["household_key","snapshot_day"], how="inner")
odf = pd.DataFrame({"household_key": tr.household_key, "snapshot_day": tr.snapshot_day,
                    "y": y, **{k: oof[k] for k in oof}})
A.save_table(odf, "oof_e016_cv.parquet")

best=(1e9,None)
for w1 in np.arange(0,1.01,0.1):
    for w2 in np.arange(0,1.01-w1,0.1):
        for w3 in np.arange(0,1.01-w1-w2+1e-9,0.1):
            w4=round(1-w1-w2-w3,2)
            p=w1*oof["med_v3"]+w2*oof["med_all"]+w3*oof["hgbq_v3"]+w4*oof["hgbq_all"]
            v=mae(p)
            if v<best[0]: best=(v,(round(w1,2),round(w2,2),round(w3,2),w4))
print("best weights (med_v3,med_all,hgbq_v3,hgbq_all):", best)


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

mall = A.load_saved("feats_all_e016.parquet")   # v3 + seasonal
v3  = A.load_saved("feats_v3.parquet")
tt = A.train_targets()

def prep(df):
    t = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    fc = [c for c in df.columns if c not in ("household_key","snapshot_day")]
    return t[fc].values.astype(np.float32), t.future_spend_4w.values.astype(np.float32), t.snapshot_day.values

Xa, ya, da = prep(mall)
Xv, yv, dv = prep(v3)
days = da; y = ya

def med_xgb(seed): return xgb.XGBRegressor(n_estimators=600, learning_rate=0.04, max_depth=6, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist", objective="reg:quantileerror", quantile_alpha=0.5,
    n_jobs=8, random_state=seed)
def hgb_q(seed): return HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=300, learning_rate=0.06,
    max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)

def fill_nan(Xtr, Xte):
    med = np.nanmedian(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(Xtr), Xtr, med[None,:]), np.where(np.isfinite(Xte), Xte, med[None,:])

folds = [(days<=207), (days>207)&(days<=291), (days>291)&(days<=375), (days>375)]
oof = {k: np.zeros(len(y)) for k in ["med_v3","med_all","hgbq_v3","hgbq_all"]}
t0=time.time()
for fi,f in enumerate(folds):
    for name, (X, mk) in {"med_v3":(Xv,med_xgb), "med_all":(Xa,med_xgb), "hgbq_v3":(Xv,hgb_q), "hgbq_all":(Xa,hgb_q)}.items():
        mod = mk(1)
        if name.startswith("hgbq"):
            Xtr2, Xte2 = fill_nan(X[~f], X[f]); mod.fit(Xtr2, y[~f]); oof[name][f] = mod.predict(Xte2)
        else:
            mod.fit(X[~f], y[~f]); oof[name][f] = mod.predict(X[f])
    print("fold", fi, round(time.time()-t0,1), "s", flush=True)

def mae(p): return np.mean(np.abs(p-y))
for k in oof: print(k, round(mae(oof[k]),3))

tr = mall.merge(tt, on=["household_key","snapshot_day"], how="inner")
odf = pd.DataFrame({"household_key": tr.household_key.values, "snapshot_day": tr.snapshot_day.values,
                    "y": y})
for k in oof: odf[k] = oof[k]
A.save_table(odf, "oof_e016_cv.parquet")

best=(1e9,None)
for w1 in np.arange(0,1.01,0.1):
    for w2 in np.arange(0,1.01-w1,0.1):
        for w3 in np.arange(0,1.01-w1-w2+1e-9,0.1):
            w4=round(1-w1-w2-w3,2)
            p=w1*oof["med_v3"]+w2*oof["med_all"]+w3*oof["hgbq_v3"]+w4*oof["hgbq_all"]
            v=mae(p)
            if v<best[0]: best=(v,(round(w1,2),round(w2,2),round(w3,2),w4))
print("best weights (med_v3,med_all,hgbq_v3,hgbq_all):", best)


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

mall = A.load_saved("feats_all_e016.parquet")   # v3 + seasonal (train+val rows)
v3   = A.load_saved("feats_v3.parquet")
tt   = A.train_targets()

def prep(df):
    fc = [c for c in df.columns if c not in ("household_key","snapshot_day")]
    tr = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    va = df[df.snapshot_day>=459]
    return (tr[fc].values.astype(np.float32), tr.future_spend_4w.values.astype(np.float32),
            va[fc].values.astype(np.float32), va[["household_key","snapshot_day"]])

Xa_tr, ya_tr, Xa_va, va_keys = prep(mall)
Xv_tr, yv_tr, Xv_va, va_keys2 = prep(v3)
assert (va_keys.values==va_keys2.values).all()
print("train", Xa_tr.shape, "val", Xa_va.shape)

def med_xgb(seed): return xgb.XGBRegressor(n_estimators=600, learning_rate=0.04, max_depth=6, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist", objective="reg:quantileerror", quantile_alpha=0.5,
    n_jobs=8, random_state=seed)
def hgb_q(seed): return HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=300, learning_rate=0.06,
    max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)

def fill_nan_fitpred(mk, seeds, Xtr, ytr, Xte):
    med = np.nanmedian(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr2 = np.where(np.isfinite(Xtr), Xtr, med[None,:]); Xte2 = np.where(np.isfinite(Xte), Xte, med[None,:])
    out = np.zeros(len(Xte))
    for s in seeds:
        m = mk(s); m.fit(Xtr2, ytr); out += m.predict(Xte2)
    return out/len(seeds)

def xgb_fitpred(mk, seeds, Xtr, ytr, Xte):
    out = np.zeros(len(Xte))
    for s in seeds:
        m = mk(s); m.fit(Xtr, ytr); out += m.predict(Xte)
    return out/len(seeds)

t0=time.time()
p_med_v3 = xgb_fitpred(med_xgb, [1,2], Xv_tr, yv_tr, Xv_va)
print("med_v3 done", round(time.time()-t0,1), flush=True)
p_hgbq_v3 = fill_nan_fitpred(hgb_q, [1,2], Xv_tr, yv_tr, Xv_va)
print("hgbq_v3 done", round(time.time()-t0,1), flush=True)
p_hgbq_all = fill_nan_fitpred(hgb_q, [1,2], Xa_tr, ya_tr, Xa_va)
print("hgbq_all done", round(time.time()-t0,1), flush=True)

# CV-optimal weights: 0.4*med_v3 + 0.2*hgbq_v3 + 0.4*hgbq_all
pred = 0.4*p_med_v3 + 0.2*p_hgbq_v3 + 0.4*p_hgbq_all
out = va_keys.copy()
out["prediction"] = pred
print(out.shape, out.prediction.describe().round(2))
p = A.save_table(out, "pred_e016.parquet")
print(p)
