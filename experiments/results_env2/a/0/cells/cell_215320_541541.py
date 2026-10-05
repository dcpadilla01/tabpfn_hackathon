
import numpy as np, pandas as pd, xgboost as xgb, time
t0=time.time()

feats = agent_api.load_saved("feats_v3.parquet")
data = feats
tt = agent_api.train_targets()
data = data.merge(tt, on=["household_key","snapshot_day"], how="left")
feat_cols=[c for c in data.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
days = agent_api.snapshot_days()
tr = data[data.snapshot_day.isin(days["train"])]
va = data[data.snapshot_day.isin(days["validation"])]
X_tr, y_tr = tr[feat_cols], tr["future_spend_4w"].values
X_va = va[feat_cols]

obj = "reg:absoluteerror"
base = dict(objective=obj, tree_method="hist", learning_rate=0.05, max_depth=6,
            min_child_weight=10, subsample=0.85, colsample_bytree=0.8, reg_lambda=1.5)

es_tr = tr[tr.snapshot_day<=403]; es_va = tr[tr.snapshot_day==431]
m1 = xgb.XGBRegressor(n_estimators=4000, early_stopping_rounds=150, eval_metric="mae", **base)
m1.fit(es_tr[feat_cols], es_tr["future_spend_4w"].values, eval_set=[(es_va[feat_cols], es_va["future_spend_4w"].values)], verbose=False)
bi = m1.best_iteration
p_es = m1.predict(es_va[feat_cols])
print("best_iter", bi, "MAE@431(es)", round(np.abs(p_es-es_va["future_spend_4w"].values).mean(),3))

n_est = int(bi*1.15)+10
m2 = xgb.XGBRegressor(n_estimators=n_est, **base)
m2.fit(X_tr, y_tr, verbose=False)
pred = np.clip(m2.predict(X_va), 0, None)
print("pred stats: mean", round(pred.mean(),2), "median", round(np.median(pred),2), "max", round(pred.max(),1))

imp = pd.Series(m2.feature_importances_, index=feat_cols).sort_values(ascending=False)
print(imp.head(20).round(4).to_string())

pp = agent_api.save_table(va[["household_key","snapshot_day"]].assign(prediction=pred), "pred_e004")
print("saved preds:", pp, "total secs", round(time.time()-t0,1))
