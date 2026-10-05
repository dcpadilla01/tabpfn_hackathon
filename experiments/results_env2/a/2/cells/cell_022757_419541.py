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
