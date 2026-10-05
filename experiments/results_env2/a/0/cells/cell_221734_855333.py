import pandas as pd, numpy as np, agent_api, xgboost as xgb
from sklearn.metrics import mean_absolute_error

feats = agent_api.load_saved("feats_v3.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
FE = [c for c in feats.columns if c not in ("index","household_key","snapshot_day","future_spend_4w")]
df[FE] = df[FE].astype(float)
BASE = dict(n_estimators=900, learning_rate=0.03, max_depth=6, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=4)

tr = df[df.snapshot_day <= 403]; pv = df[df.snapshot_day == 431].copy()
m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **BASE)
m.fit(tr[FE], tr.future_spend_4w)
pv["pred"] = np.clip(m.predict(pv[FE]),0,None)

# error decomposition by actual bucket
pv["bucket"] = pd.cut(pv.future_spend_4w, [-1, 0.01, 25, 75, 150, 300, 10000])
g = pv.groupby("bucket", observed=True).apply(lambda x: pd.Series({
    "n": len(x), "mean_pred": x.pred.mean(), "mean_act": x.future_spend_4w.mean(),
    "MAE": np.abs(x.pred - x.future_spend_4w).mean(), "err_share": np.abs(x.pred-x.future_spend_4w).sum()/np.abs(pv.pred-pv.future_spend_4w).sum()}))
print(g)

# zero-spend rows: what does the model say?
z = pv[pv.future_spend_4w == 0]
print("\nzero rows:", len(z), "mean pred:", z.pred.mean().round(2), "median pred:", z.pred.median().round(2))
print("their exp4w_blend:", z.exp4w_blend.mean().round(2), "spend_28:", z.spend_28.mean().round(2), "days_since_last:", z.days_since_last.mean().round(1))

# how predictive is a zero-classifier signal? fraction of 4w windows with zero spend by recent activity
df["zero_next"] = (df.future_spend_4w == 0).astype(float)
for c in ["days_since_last","spend_28","wk0","active_weeks8"]:
    print(c, "corr with zero_next:", df[[c,"zero_next"]].corr().iloc[0,1].round(3))
