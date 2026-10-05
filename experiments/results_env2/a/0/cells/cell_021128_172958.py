
import agent_api, pandas as pd, numpy as np, time
feats = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
print(feats.dtypes.value_counts())
obj_cols = [c for c in feats.columns if feats[c].dtype==object]
print("object cols:", obj_cols)
for c in obj_cols[:3]: print(c, feats[c].unique()[:8])

m = feats.merge(tt, on=["household_key","snapshot_day"])
y = m.future_spend_4w
print("\ntarget: zero share %.3f, quantiles:" % (y==0).mean(), np.quantile(y,[.1,.25,.5,.75,.9,.95,.99]).round(1))

# timing test: one xgb config on all train rows
import xgboost as xgb
FE = [c for c in feats.columns if c not in ("household_key","snapshot_day")]
X = m[FE].copy()
for c in obj_cols:
    X[c] = X[c].astype("category")
print("X shape", X.shape)
t0=time.time()
mdl = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, max_depth=5, min_child_weight=40,
                       learning_rate=0.08, n_estimators=400, n_jobs=-1, tree_method="hist", enable_categorical=True)
mdl.fit(X, y)
print("fit time %.1fs" % (time.time()-t0))
t0=time.time()
_ = mdl.predict(X.iloc[:5000])
print("pred time %.2fs" % (time.time()-t0))
