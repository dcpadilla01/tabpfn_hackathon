import numpy as np, pandas as pd, xgboost as xgb, time

allF = load_saved("allF.parquet")
oof = load_saved("oof_e5.parquet")[["household_key","snapshot_day","future_spend_4w"]]
held = load_saved("e016_held.parquet")[["household_key","snapshot_day","future_spend_4w"]]
feats = [c for c in allF.columns if c not in ("household_key","snapshot_day")]
tr = allF.merge(oof, on=["household_key","snapshot_day"])
he = allF.merge(held, on=["household_key","snapshot_day"])
Xtr, Xhe = tr[feats].values.astype(np.float32), he[feats].values.astype(np.float32)
ytr, yhe = tr.future_spend_4w.values, he.future_spend_4w.values
dtr, dhe = tr.snapshot_day.values, he.snapshot_day.values
def mae(p,y): return np.abs(np.asarray(p)-np.asarray(y)).mean()

def qw(y): return 0.5**((375.0-y)/140.0)

t0=time.time()
params = dict(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist",
              learning_rate=0.03, max_depth=6, min_child_weight=25, subsample=0.8,
              colsample_bytree=0.7, reg_lambda=8.0, reg_alpha=0.5, base_score=0.5,
              n_jobs=4, eval_metric="mae")
preds_he, preds_tr = [], []
for s in [7,17,27]:
    d = xgb.DMatrix(Xtr, label=ytr, weight=qw(ytr))
    b = xgb.train(params, d, num_boost_round=1200, verbose_eval=False)
    preds_he.append(b.predict(xgb.DMatrix(Xhe)))
    preds_tr.append(b.predict(xgb.DMatrix(Xtr)))
    print("seed", s, "held MAE:", round(mae(preds_he[-1], yhe),3), round(time.time()-t0,1),"s")
Phe = np.mean(preds_he, axis=0); Ptr = np.mean(preds_tr, axis=0)
print("ENSEMBLE held MAE:", round(mae(Phe, yhe),3), "| E005 held MAE 62.82")
print("ENSEMBLE train-fit MAE:", round(mae(Ptr, ytr),3))
print("mean pred held:", round(Phe.mean(),1), "target mean:", round(yhe.mean(),1))
