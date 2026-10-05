import pandas as pd, numpy as np, agent_api, xgboost as xgb
print("xgb version:", xgb.__version__)
f1 = agent_api.load_saved("feats_v1.parquet")
if "household_key" not in f1.columns: f1 = f1.reset_index()
print("feats_v1:", f1.shape)
print("cols:", f1.columns.tolist())
print(f1.dtypes.value_counts())
tt = agent_api.train_targets()
print("targets:", tt.shape, tt.columns.tolist())
y = tt.future_spend_4w
print(y.describe())
print("zero frac:", (y==0).mean())
sd = agent_api.snapshot_days(); print(sd)
va_days = sd["validation"]
print("val rows in feats_v1:", (f1.snapshot_day.isin(va_days)).sum())
# objective availability
rng = np.random.RandomState(0)
X = rng.rand(60,3); yy = rng.rand(60)
for obj in ["reg:squarederror","reg:absoluteerror","reg:quantileerror"]:
    try:
        m = xgb.XGBRegressor(n_estimators=2, objective=obj)
        if obj=="reg:quantileerror": m.set_params(quantile_alpha=0.5)
        m.fit(X,yy); print("OK", obj)
    except Exception as e:
        print("FAIL", obj, type(e).__name__, str(e)[:120])
p1 = agent_api.load_saved("pred_e001.parquet")
print("pred_e001:", p1.shape, p1.columns.tolist())
