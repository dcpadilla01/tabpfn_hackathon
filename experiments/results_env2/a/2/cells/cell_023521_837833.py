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
