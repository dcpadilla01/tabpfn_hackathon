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
