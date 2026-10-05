
import pandas as pd, numpy as np, agent_api, xgboost as xgb

feats = agent_api.load_saved("e004_features.parquet")
nf = agent_api.load_saved("e005_newfeats.parquet")
df = feats.merge(nf.drop(columns=["household_key","snapshot_day"]), left_index=True, right_index=True)
tt = agent_api.train_targets()
df = df.merge(tt, on=["household_key","snapshot_day"])
feat_cols = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
X = df[feat_cols].astype(float).replace([np.inf,-np.inf], np.nan)
y = df.future_spend_4w.values
day = df.snapshot_day.values

tr = day<=403; te = day==431
w = 0.5**((403-day[tr])/140.0)
m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
    n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=8,
    random_state=0, base_score=70.0)
m.fit(X[tr], y[tr], sample_weight=w)
p431 = m.predict(X[te]); y431 = y[te]
s28_te = df.spend_28.values[te]

z = s28_te==0
print("MAE all:", np.mean(np.abs(p431-y431)))
print("MAE s28==0 raw:", np.mean(np.abs(p431[z]-y431[z])), " if pred=0:", np.mean(y431[z]))
print("n:", z.sum(), "median y z:", np.median(y431[z]), "mean pred z:", p431[z].mean())
print("dist y|z:", np.percentile(y431[z],[10,25,50,75,90]))
print("dist pred|z:", np.percentile(p431[z],[10,25,50,75,90]))
# what does pred look like for z households
print("frac pred>10 among z:", (p431[z]>10).mean())
