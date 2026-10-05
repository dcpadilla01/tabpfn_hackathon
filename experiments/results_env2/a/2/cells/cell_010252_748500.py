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
