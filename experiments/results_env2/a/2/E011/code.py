import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")

feats = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
print("feats", feats.shape)
print("cols:", list(feats.columns))
y = tt.future_spend_4w
print("target mean %.1f med %.1f p90 %.1f p99 %.1f zero-rate %.3f" % (y.mean(), y.median(), y.quantile(.9), y.quantile(.99), (y==0).mean()))

oof = A.load_saved("oof_e008.parquet")
print("oof cols:", oof.columns.tolist())
m = oof.merge(tt, on=["household_key","snapshot_day"])
pc = [c for c in oof.columns if c not in ("household_key","snapshot_day")]
for c in pc:
    e = m[c]-m.future_spend_4w
    print(c, "OOF MAE %.2f bias %.2f" % (e.abs().mean(), e.mean()))
c0 = pc[0]
m["dec"] = pd.qcut(m[c0], 10, duplicates="drop")
for k, d in m.groupby("dec", observed=True):
    e = d[c0]-d.future_spend_4w
    print("dec %-12s n %5d pred %6.1f y %6.1f bias %7.1f mae %6.1f" % (str(k), len(d), d[c0].mean(), d.future_spend_4w.mean(), e.mean(), e.abs().mean()))
for sd, d in m.groupby("snapshot_day"):
    e = d[c0]-d.future_spend_4w
    print("snap", sd, "mae %.1f bias %.1f ymean %.1f" % (e.abs().mean(), e.mean(), d.future_spend_4w.mean()))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

feats = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="inner")
FE = [c for c in feats.columns if c not in ("household_key","snapshot_day")]
CAT = [c for c in FE if c.startswith("dem_")]
for c in CAT: df[c] = df[c].astype("category")

def fit_pred(tr, va, kind, seed=7):
    ytr = tr.future_spend_4w.values
    Xtr = tr[FE]; Xva = va[FE]
    if kind=="med":
        m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
                             subsample=0.8, colsample_bytree=0.7, objective="reg:quantileerror", quantile_alpha=0.5,
                             random_state=seed, n_jobs=8, enable_categorical=True, tree_method="hist")
    else:
        m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
                             subsample=0.8, colsample_bytree=0.7, random_state=seed, n_jobs=8,
                             enable_categorical=True, tree_method="hist")
    m.fit(Xtr, ytr)
    return m.predict(Xva)

# local CV: train on snaps <= 375, validate on 403 & 431
trsnaps = [95,123,151,179,207,235,263,291,319,347,375]
vasnaps = [403,431]
tr = df[df.snapshot_day.isin(trsnaps)]; va = df[df.snapshot_day.isin(vasnaps)]
t0=time.time()
p_med = fit_pred(tr, va, "med")
p_sq  = fit_pred(tr, va, "sq")
print("fit time %.1fs" % (time.time()-t0))
for nm, p in [("med",p_med),("sq",p_sq),("blend",0.5*p_med+0.5*p_sq)]:
    e = p - va.future_spend_4w.values
    print(nm, "localCV MAE %.2f bias %.2f" % (np.abs(e).mean(), e.mean()))
    for sd in vasnaps:
        msk = va.snapshot_day.values==sd
        print("   snap", sd, "mae %.1f" % np.abs(e[msk]).mean())

# market-level spend drift check (data up to day 459 only)
v = A.snapshot(459)
tx = v.transactions
wk = tx.groupby("week_no").sales_value.sum()
hh = tx.groupby("week_no").household_key.nunique()
print("\nweekly market spend (per active hh):")
per_hh = (wk/hh)
idx = per_hh.index
for i in range(0, len(idx), 8):
    print("wk %3d-%3d  per-hh %.1f" % (idx[i], idx[min(i+7,len(idx)-1)], per_hh.iloc[i:i+8].mean()))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

feats = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="inner")
FE = [c for c in feats.columns if c not in ("household_key","snapshot_day")]
CAT = [c for c in FE if c.startswith("dem_")]
for c in CAT:
    df[c] = df[c].astype("Float64").fillna(-1).astype(int).astype(str).astype("category")

def fit_pred(tr, va, kind, seed=7):
    ytr = tr.future_spend_4w.values
    kw = dict(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
              subsample=0.8, colsample_bytree=0.7, random_state=seed, n_jobs=8,
              enable_categorical=True, tree_method="hist")
    if kind=="med":
        m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **kw)
    else:
        m = xgb.XGBRegressor(**kw)
    m.fit(tr[FE], ytr)
    return m.predict(va[FE])

trsnaps = [95,123,151,179,207,235,263,291,319,347,375]
vasnaps = [403,431]
tr = df[df.snapshot_day.isin(trsnaps)]; va = df[df.snapshot_day.isin(vasnaps)]
t0=time.time()
p_med = fit_pred(tr, va, "med")
p_sq  = fit_pred(tr, va, "sq")
print("fit time %.1fs" % (time.time()-t0))
for nm, p in [("med",p_med),("sq",p_sq),("blend",0.5*p_med+0.5*p_sq)]:
    e = p - va.future_spend_4w.values
    print(nm, "localCV MAE %.2f bias %.2f" % (np.abs(e).mean(), e.mean()))
    for sd in vasnaps:
        msk = va.snapshot_day.values==sd
        print("   snap", sd, "mae %.1f" % np.abs(e[msk]).mean())

v = A.snapshot(459)
tx = v.transactions
wk = tx.groupby("week_no").sales_value.sum()
hh = tx.groupby("week_no").household_key.nunique()
per_hh = (wk/hh); idx = per_hh.index
print("\nweekly market spend per active hh:")
for i in range(0, len(idx), 8):
    print("wk %3d-%3d  per-hh %.1f" % (idx[i], idx[min(i+7,len(idx)-1)], per_hh.iloc[i:i+8].mean()))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

oof = A.load_saved("oof_e008.parquet"); tt = A.train_targets(); feats = A.load_saved("feats_v3.parquet")
m = oof.merge(tt, on=["household_key","snapshot_day"]).merge(feats, on=["household_key","snapshot_day"])
y = m.future_spend_4w.values

# find blend weights approximating E008
best=None
for wm in [0.4,0.5,0.6]:
    for ws in [0.1,0.2,0.3,0.4]:
        wl = 1-wm-ws
        if wl<0: continue
        p = wm*m.oof_med + ws*m.oof_sq + wl*m.oof_log
        mae = np.abs(p-y).mean()
        if best is None or mae<best[0]: best=(mae,wm,ws,wl)
print("best oof blend:", best)
mae,wm,ws,wl = best
m["blend"] = wm*m.oof_med + ws*m.oof_sq + wl*m.oof_log

trm = m[m.snapshot_day<=347]; vam = m[m.snapshot_day>=375]
SF = ["blend","spend_28","baskets_28","active_28","days_since_last","avg_basket_84",
      "trend_28_56","week_of_year","spend_112","baskets_112","has_demo","spend_7","baskets_84"]
def stacker_fit(d):
    X = d[SF].copy()
    mod = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, min_child_weight=20,
                           subsample=0.8, colsample_bytree=0.8, objective="reg:quantileerror",
                           quantile_alpha=0.5, random_state=0, n_jobs=8)
    mod.fit(X, d.future_spend_4w)
    return mod
st = stacker_fit(trm)
p_st = st.predict(vam[SF])
p_raw = vam.blend.values; yv = vam.future_spend_4w.values
print("raw blend  holdout(375-431) MAE %.2f" % np.abs(p_raw-yv).mean())
print("stacked    holdout(375-431) MAE %.2f" % np.abs(p_st-yv).mean())
for sd in [375,403,431]:
    k = vam.snapshot_day==sd
    print("  snap",sd,"raw %.2f stacked %.2f" % (np.abs(p_raw[k.values]-yv[k.values]).mean(), np.abs(p_st[k.values]-yv[k.values]).mean()))

# simple decile ratio calibration
trm2 = trm.copy(); trm2["bin"] = pd.qcut(trm2.blend, 10, duplicates="drop")
ratio = trm2.groupby("bin", observed=True).apply(lambda d: np.median(d.future_spend_4w/np.maximum(d.blend,1)))
vam2 = vam.copy(); vam2["bin"] = pd.cut(vam2.blend, trm2["bin"].cat.categories)
p_cal = vam2.blend.values * vam2["bin"].map(ratio).astype(float).fillna(1).values
print("decile-ratio calib MAE %.2f" % np.abs(p_cal-yv).mean())
print("stacked bias %.1f raw bias %.1f" % ((p_st-yv).mean(), (p_raw-yv).mean()))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

feats = A.load_saved("feats_v3.parquet"); tt = A.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="inner")
FE = [c for c in feats.columns if c not in ("household_key","snapshot_day")]
CAT = [c for c in FE if c.startswith("dem_")]
for c in CAT:
    df[c] = df[c].astype("Float64").fillna(-1).astype(int).astype(str).astype("category")

def fit_pred(tr, va, seed=7, extra=None, weights=None):
    Xtr = tr[FE].copy(); ytr = tr.future_spend_4w.values
    if extra is not None: Xtr = pd.concat([Xtr, extra[0]], axis=1)
    kw = dict(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
              subsample=0.8, colsample_bytree=0.7, random_state=seed, n_jobs=8,
              enable_categorical=True, tree_method="hist")
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **kw)
    m.fit(Xtr, ytr, sample_weight=weights)
    Xva = va[FE].copy()
    if extra is not None: Xva = pd.concat([Xva, extra[1]], axis=1)
    return m.predict(Xva)

trsnaps = [95,123,151,179,207,235,263,291,319,347,375]
vasnaps = [403,431]
tr = df[df.snapshot_day.isin(trsnaps)].copy(); va = df[df.snapshot_day.isin(vasnaps)].copy()
yv = va.future_spend_4w.values

# extra feature: snapshot day index
ex_tr = pd.DataFrame({"snap_day": tr.snapshot_day.values}); ex_va = pd.DataFrame({"snap_day": va.snapshot_day.values})

# weights: recency decay half-life 168d
w = np.exp(-(431 - tr.snapshot_day.values)/168.0*0.693)

t0=time.time()
p0 = fit_pred(tr, va)
p1 = fit_pred(tr, va, extra=(ex_tr, ex_va))
p2 = fit_pred(tr, va, weights=w)
p3 = fit_pred(tr, va, extra=(ex_tr, ex_va), weights=w)
print("time %.0fs" % (time.time()-t0))
for nm,p in [("base",p0),("+snapday",p1),("+recency_w",p2),("+both",p3)]:
    e = p-yv
    print(nm, "localCV MAE %.2f bias %.1f" % (np.abs(e).mean(), e.mean()))
    for sd in vasnaps:
        k = va.snapshot_day.values==sd
        print("   snap",sd,"mae %.1f" % np.abs(e[k]).mean())


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

feats = A.load_saved("feats_v3.parquet"); tt = A.train_targets()
print("feats", feats.shape, "tt", tt.shape, "tt NaN targets:", tt.future_spend_4w.isna().sum())
print("tt dup keys:", tt.duplicated(["household_key","snapshot_day"]).sum())
df = feats.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("merged", df.shape, "NaN y:", df.future_spend_4w.isna().sum())
FE = [c for c in feats.columns if c not in ("household_key","snapshot_day")]
CAT = [c for c in FE if c.startswith("dem_")]
for c in CAT:
    df[c] = df[c].astype("Float64").fillna(-1).astype(int).astype(str).astype("category")

def fit_pred(tr, va, seed=7, extra=None, weights=None):
    ytr = tr.future_spend_4w.values
    ok = np.isfinite(ytr)
    Xtr = tr.loc[ok, FE].copy(); ytr = ytr[ok]
    if extra is not None: Xtr = pd.concat([Xtr, extra[0][ok] if hasattr(extra[0],'iloc') else extra[0]], axis=1)
    kw = dict(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
              subsample=0.8, colsample_bytree=0.7, random_state=seed, n_jobs=8,
              enable_categorical=True, tree_method="hist")
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **kw)
    m.fit(Xtr, ytr, sample_weight=None if weights is None else (weights[ok] if weights is not None else None))
    Xva = va[FE].copy()
    if extra is not None: Xva = pd.concat([Xva, extra[1]], axis=1)
    return m.predict(Xva)

trsnaps = [95,123,151,179,207,235,263,291,319,347,375]
vasnaps = [403,431]
tr = df[df.snapshot_day.isin(trsnaps)].reset_index(drop=True); va = df[df.snapshot_day.isin(vasnaps)].reset_index(drop=True)
yv = va.future_spend_4w.values
ex_tr = pd.DataFrame({"snap_day": tr.snapshot_day.values}); ex_va = pd.DataFrame({"snap_day": va.snapshot_day.values})
w = np.exp(-(431 - tr.snapshot_day.values)/168.0*0.693)

t0=time.time()
p0 = fit_pred(tr, va)
p1 = fit_pred(tr, va, extra=(ex_tr, ex_va))
p2 = fit_pred(tr, va, weights=w)
p3 = fit_pred(tr, va, extra=(ex_tr, ex_va), weights=w)
print("time %.0fs" % (time.time()-t0))
for nm,p in [("base",p0),("+snapday",p1),("+recency_w",p2),("+both",p3)]:
    e = p-yv
    print(nm, "localCV MAE %.2f bias %.1f" % (np.abs(e).mean(), e.mean()))
    for sd in vasnaps:
        k = va.snapshot_day.values==sd
        print("   snap",sd,"mae %.1f" % np.abs(e[k]).mean())


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

def yearago_fn(view, snapshot_day):
    hh = view.households
    hh = pd.Index(hh)
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hh)]
    d = tx.day
    def win(lo, hi):
        m = (d > lo) & (d <= hi)
        t = tx[m]
        g = t.groupby("household_key")
        return g.sales_value.sum(), g.basket_id.nunique()
    s1,b1 = win(snapshot_day-363, snapshot_day-336)   # year-ago matching 4w
    s2,b2 = win(snapshot_day-391, snapshot_day-336)   # year-ago 8w
    s3,b3 = win(snapshot_day-727, snapshot_day-700)   # two-years-ago 4w
    out = pd.DataFrame({
        "spend_y1_4w": s1, "baskets_y1_4w": b1,
        "spend_y1_8w": s2, "spend_y2_4w": s3,
    }).reindex(hh).fillna(0.0)
    out["active_y1"] = (out.spend_y1_4w > 0).astype(float)
    return out

t0=time.time()
X = A.build_features(yearago_fn)
print("built", X.shape, "%.0fs" % (time.time()-t0))
print(X.head(3))

feats3 = A.load_saved("feats_v3.parquet")
f5 = feats3.merge(X.drop(columns=["household_key"], errors="ignore"), left_on=["household_key","snapshot_day"],
                  right_index=True if X.index.name=="household_key" else ["snapshot_day", X.index.name or X.columns[0]])
print("merge check", f5.shape)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

def yearago_fn(view, snapshot_day):
    hh = pd.Index(view.households)
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hh)]
    d = tx.day
    def win(lo, hi):
        t = tx[(d > lo) & (d <= hi)]
        g = t.groupby("household_key")
        return g.sales_value.sum(), g.basket_id.nunique()
    s1,b1 = win(snapshot_day-363, snapshot_day-336)
    s2,b2 = win(snapshot_day-391, snapshot_day-336)
    s3,b3 = win(snapshot_day-727, snapshot_day-700)
    out = pd.DataFrame({"spend_y1_4w": s1, "baskets_y1_4w": b1, "spend_y1_8w": s2, "spend_y2_4w": s3}).reindex(hh).fillna(0.0)
    out["active_y1"] = (out.spend_y1_4w > 0).astype(float)
    return out

X = A.build_features(yearago_fn)          # 36426 x 7, indexed by household_key per snapshot
X = X.reset_index()                        # household_key, snapshot_day, feats
feats3 = A.load_saved("feats_v3.parquet")
f5 = feats3.merge(X, on=["household_key","snapshot_day"], how="inner")
print("f5", f5.shape)
A.save_table(f5, "feats_v5.parquet")

tt = A.train_targets()
df = f5.merge(tt, on=["household_key","snapshot_day"], how="inner")
FE = [c for c in f5.columns if c not in ("household_key","snapshot_day")]
CAT = [c for c in FE if c.startswith("dem_")]
for c in CAT: df[c] = df[c].astype("Float64").fillna(-1).astype(int).astype(str).astype("category")

def fit_pred(tr, va, cols, seed=7):
    ok = np.isfinite(tr.future_spend_4w.values)
    tr = tr.loc[ok]
    kw = dict(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
              subsample=0.8, colsample_bytree=0.7, random_state=seed, n_jobs=8,
              enable_categorical=True, tree_method="hist")
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **kw)
    m.fit(tr[cols], tr.future_spend_4w)
    return m.predict(va[cols])

trsnaps = [95,123,151,179,207,235,263,291,319,347,375]; vasnaps=[403,431]
tr = df[df.snapshot_day.isin(trsnaps)]; va = df[df.snapshot_day.isin(vasnaps)]
yv = va.future_spend_4w.values
t0=time.time()
p0 = fit_pred(tr, va, FE)
NEW = ["spend_y1_4w","baskets_y1_4w","spend_y1_8w","spend_y2_4w","active_y1"]
p1 = fit_pred(tr, va, FE+NEW)
print("time %.0fs" % (time.time()-t0))
for nm,p in [("base",p0),("+yearago",p1)]:
    e = p-yv
    print(nm, "localCV MAE %.2f bias %.1f" % (np.abs(e).mean(), e.mean()))
    for sd in vasnaps:
        k = va.snapshot_day.values==sd
        print("   snap",sd,"mae %.1f" % np.abs(e[k]).mean())


# ---- cell ----
import agent_api as A, pandas as pd
f3 = A.load_saved("feats_v3.parquet"); f5 = A.load_saved("feats_v5.parquet")
print(f3.shape, f5.shape)
print("f5 dup labels:", f5.columns[f5.columns.duplicated()].tolist())
NEW = ["spend_y1_4w","baskets_y1_4w","spend_y1_8w","spend_y2_4w","active_y1"]
print("overlap with v3:", [c for c in NEW if c in f3.columns])
print([c for c in f5.columns if "y1" in c or "y2" in c])


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

f5 = A.load_saved("feats_v5.parquet")
tt = A.train_targets()
df = f5.merge(tt, on=["household_key","snapshot_day"], how="inner")
FE = [c for c in f5.columns if c not in ("household_key","snapshot_day")]
NEW = ["spend_y1_4w","baskets_y1_4w","spend_y1_8w","spend_y2_4w","active_y1"]
CAT = [c for c in FE if c.startswith("dem_")]
for c in CAT: df[c] = df[c].astype("Float64").fillna(-1).astype(int).astype(str).astype("category")

def fit_pred(tr, va, cols, seed=7):
    kw = dict(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
              subsample=0.8, colsample_bytree=0.7, random_state=seed, n_jobs=8,
              enable_categorical=True, tree_method="hist")
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **kw)
    m.fit(tr[cols], tr.future_spend_4w)
    return m.predict(va[cols])

trsnaps = [95,123,151,179,207,235,263,291,319,347,375]; vasnaps=[403,431]
tr = df[df.snapshot_day.isin(trsnaps)]; va = df[df.snapshot_day.isin(vasnaps)]
yv = va.future_spend_4w.values
t0=time.time()
p0 = fit_pred(tr, va, FE)
p1 = fit_pred(tr, va, FE+NEW)
print("time %.0fs" % (time.time()-t0))
for nm,p in [("base",p0),("+yearago",p1)]:
    e = p-yv
    print(nm, "localCV MAE %.2f bias %.1f" % (np.abs(e).mean(), e.mean()))
    for sd in vasnaps:
        k = va.snapshot_day.values==sd
        print("   snap",sd,"mae %.1f" % np.abs(e[k]).mean())

# feature importance of new feats
kw = dict(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
          subsample=0.8, colsample_bytree=0.7, random_state=7, n_jobs=8,
          enable_categorical=True, tree_method="hist")
m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **kw)
m.fit(tr[FE+NEW], tr.future_spend_4w)
imp = pd.Series(m.feature_importances_, index=FE+NEW).sort_values(ascending=False)
print("new feat ranks:", {k:(i+1, round(v,4)) for i,(k,v) in enumerate(imp.items()) if k in NEW})


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

f5 = A.load_saved("feats_v5.parquet")
tt = A.train_targets()
df = f5.merge(tt, on=["household_key","snapshot_day"], how="inner")
NEW = ["spend_y1_4w","baskets_y1_4w","spend_y1_8w","spend_y2_4w","active_y1"]
FE = [c for c in f5.columns if c not in ("household_key","snapshot_day")+tuple(NEW)]
CAT = [c for c in FE if c.startswith("dem_")]
for c in CAT: df[c] = df[c].astype("Float64").fillna(-1).astype(int).astype(str).astype("category")

def fit_pred(tr, va, cols, seed=7):
    kw = dict(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
              subsample=0.8, colsample_bytree=0.7, random_state=seed, n_jobs=8,
              enable_categorical=True, tree_method="hist")
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **kw)
    m.fit(tr[cols], tr.future_spend_4w)
    return m.predict(va[cols])

trsnaps = [95,123,151,179,207,235,263,291,319,347,375]; vasnaps=[403,431]
tr = df[df.snapshot_day.isin(trsnaps)]; va = df[df.snapshot_day.isin(vasnaps)]
yv = va.future_spend_4w.values
t0=time.time()
p0 = fit_pred(tr, va, FE)
p1 = fit_pred(tr, va, FE+NEW)
print("time %.0fs" % (time.time()-t0))
for nm,p in [("base",p0),("+yearago",p1)]:
    e = p-yv
    print(nm, "localCV MAE %.2f bias %.1f" % (np.abs(e).mean(), e.mean()))
    for sd in vasnaps:
        k = va.snapshot_day.values==sd
        print("   snap",sd,"mae %.1f" % np.abs(e[k]).mean())

kw = dict(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
          subsample=0.8, colsample_bytree=0.7, random_state=7, n_jobs=8,
          enable_categorical=True, tree_method="hist")
m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **kw)
m.fit(tr[FE+NEW], tr.future_spend_4w)
imp = pd.Series(m.feature_importances_, index=FE+NEW).sort_values(ascending=False)
print("new feat ranks:", {k:(i+1, round(v,4)) for i,(k,v) in enumerate(imp.items()) if k in NEW})
