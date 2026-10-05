import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

f = A.load_saved("feats_v3.parquet").drop(columns=["index"])
tt = A.train_targets()
df = tt.merge(f, on=["household_key","snapshot_day"], how="inner")
FEATS = [c for c in f.columns if c not in ("household_key","snapshot_day")]
print("n feats:", len(FEATS))

def train_pred(train_days, pred_days, params=None, alpha=0.5):
    tr = df[df.snapshot_day.isin(train_days)]
    Xtr, ytr = tr[FEATS], tr.future_spend_4w
    p = dict(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
             colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, n_jobs=4,
             objective="reg:quantileerror", quantile_alpha=alpha, tree_method="hist")
    if params: p.update(params)
    m = xgb.XGBRegressor(**p)
    m.fit(Xtr, ytr)
    pr = df[df.snapshot_day.isin(pred_days)]
    return pr, m.predict(pr[FEATS])

t0=time.time()
# local proxy: train on <=403, predict 431
pr, pred = train_pred([d for d in range(95,432,28) if d<=403], [431])
mae431 = np.abs(pred - pr.future_spend_4w.values).mean()
print("LOCAL 431 MAE (E005 repro):", round(mae431,3), "time", round(time.time()-t0,1))
# also proxy on 403 trained <=375
pr2, pred2 = train_pred([d for d in range(95,404,28) if d<=375], [403])
print("LOCAL 403 MAE:", round(np.abs(pred2-pr2.future_spend_4w.values).mean(),3))
