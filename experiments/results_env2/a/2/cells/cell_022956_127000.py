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
