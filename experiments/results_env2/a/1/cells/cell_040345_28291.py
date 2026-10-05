import numpy as np, pandas as pd, xgboost as xgb, time

allF = load_saved("allF.parquet")
oof = load_saved("oof_e5.parquet")[["household_key","snapshot_day","future_spend_4w"]]
held = load_saved("e016_held.parquet")[["household_key","snapshot_day","future_spend_4w"]]
tgt = allF[["household_key","snapshot_day","future_spend_4w"]].rename(columns={"future_spend_4w":"tgt_F"})
feats = [c for c in allF.columns if c not in ("household_key","snapshot_day","future_spend_4w")]

tr = allF.drop(columns=["future_spend_4w"]).merge(oof, on=["household_key","snapshot_day"])
he = allF.drop(columns=["future_spend_4w"]).merge(held, on=["household_key","snapshot_day"])
val = allF[allF.snapshot_day>=459]
Xtr, Xhe, Xval = (tr[feats].values.astype(np.float32), he[feats].values.astype(np.float32),
                  val[feats].values.astype(np.float32))
ytr, yhe = tr.future_spend_4w.values, he.future_spend_4w.values
def mae(p,y): return np.abs(np.asarray(p)-np.asarray(y)).mean()
def qw(y): return 0.5**((375.0-y)/140.0)
params = dict(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist",
              learning_rate=0.03, max_depth=6, min_child_weight=25, subsample=0.8,
              colsample_bytree=0.7, reg_lambda=8.0, reg_alpha=0.5, base_score=0.5,
              n_jobs=4, eval_metric="mae")
t0=time.time()
preds_he, preds_val = [], []
for s in [7,17,27]:
    d = xgb.DMatrix(Xtr, label=ytr, weight=qw(ytr))
    b = xgb.train(params, d, num_boost_round=1200, verbose_eval=False)
    preds_he.append(b.predict(xgb.DMatrix(Xhe)))
    preds_val.append(b.predict(xgb.DMatrix(Xval)))
    print("seed", s, "held MAE:", round(mae(preds_he[-1], yhe),3), round(time.time()-t0,1),"s")
Phe = np.mean(preds_he, axis=0); Pval = np.mean(preds_val, axis=0)
print("ENSEMBLE held MAE:", round(mae(Phe, yhe),3), "| E005 held 62.82")
print("val preds:", Pval.shape, "mean", round(Pval.mean(),1), "finite:", np.isfinite(Pval).all())
out = val[["household_key","snapshot_day"]].copy()
out["prediction"] = Pval.astype(np.float64)
out = out.reset_index(drop=True)
print(out.shape, out.snapshot_day.value_counts().to_dict())
path = save_table(out, "e017_preds")
print("saved:", path)
