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
