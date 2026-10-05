
import pandas as pd, numpy as np, agent_api, xgboost as xgb, time

feats = agent_api.load_saved("e004_features.parquet")
nf = agent_api.load_saved("e005_newfeats.parquet")
df = feats.merge(nf.drop(columns=["household_key","snapshot_day"]), left_index=True, right_index=True)
tt = agent_api.train_targets()
df = df.merge(tt, on=["household_key","snapshot_day"])
print(df.shape)

feat_cols = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
X = df[feat_cols].astype(float).replace([np.inf,-np.inf], np.nan)
y = df.future_spend_4w.values
day = df.snapshot_day.values

def make_model(seed):
    return xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=0.5,
        n_estimators=1200, learning_rate=0.03, max_depth=6,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=8, random_state=seed, base_score=70.0)

t0=time.time()
# diagnostic: train on days<=403, predict 431
tr = day<=403; te = day==431
w = 0.5**((403-day[tr])/140.0)
m = make_model(0); m.fit(X[tr], y[tr], sample_weight=w)
p431 = m.predict(X[te]); y431 = y[te]
print("train time", time.time()-t0)
print("MAE 431:", np.mean(np.abs(p431-y431)))
res = y431-p431
print("median residual:", np.median(res), "mean:", res.mean())

# global shift
print("MAE after global median shift:", np.mean(np.abs(p431+np.median(res)-y431)))

# per-decile-bin median calibration
qs = np.quantile(p431, np.linspace(0,1,11))
bins = np.clip(np.digitize(p431, qs[1:-1]),0,9)
corr = np.array([np.median(res[bins==b]) for b in range(10)])
print("bin med resid:", np.round(corr,1))
print("MAE after bin calib:", np.mean(np.abs(p431+corr[bins]-y431)))

# zero-behavior: households with s28==0
s28_te = X[te][:, feat_cols.index("spend_28")]
print("n s28==0 at 431:", (s28_te==0).sum(), "median y:", np.median(y431[s28_te==0]), "median pred:", np.median(p431[s28_te==0]))
print("MAE on s28==0 subset raw:", np.mean(np.abs(p431[s28_te==0]-y431[s28_te==0])),
      "if pred=0:", np.mean(np.abs(y431[s28_te==0])))
