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
