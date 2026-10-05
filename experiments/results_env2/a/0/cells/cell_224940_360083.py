import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

f = A.load_saved("feats_v3.parquet").drop(columns=["index"])
tt = A.train_targets()
df = tt.merge(f, on=["household_key","snapshot_day"], how="inner")
FEATS = [c for c in f.columns if c not in ("household_key","snapshot_day")]

m = xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=4, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0, n_jobs=4,
    objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist")
m.fit(df[FEATS], df.future_spend_4w)

val = f[f.snapshot_day>=459].copy()
pred = m.predict(val[FEATS])
w = 0.3
val["prediction"] = (1-w)*pred + w*val.exp4w_blend.values
out = val[["household_key","snapshot_day","prediction"]].copy()
out["household_key"]=out.household_key.astype(int); out["snapshot_day"]=out.snapshot_day.astype(int)
print(out.shape, out.prediction.describe().round(1).to_dict())
p = A.save_table(out, "pred_e007.parquet")
print(p)
