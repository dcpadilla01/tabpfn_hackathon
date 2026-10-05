
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

f4 = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
feats = [c for c in f4.columns if c not in ["household_key","snapshot_day"]]
cat_cols = [c for c in feats if not pd.api.types.is_numeric_dtype(f4[c])]
X_full = pd.get_dummies(f4[feats], columns=cat_cols, dummy_na=True).astype(np.float32)
print("X_full", X_full.shape)

tr_mask = f4.snapshot_day.isin(tt.snapshot_day.unique()) & (f4.snapshot_day <= 403)
va_mask = f4.snapshot_day.isin([459,487,515,543])
Xtr = X_full[tr_mask.values].values
ytr = tt.set_index(["household_key","snapshot_day"]).reindex(
    pd.MultiIndex.from_arrays([f4.loc[tr_mask,"household_key"], f4.loc[tr_mask,"snapshot_day"]])
).future_spend_4w.values.astype(float)
print("train rows", len(ytr), "nan y:", np.isnan(ytr).sum())
Xva = X_full[va_mask.values].values
va_keys = f4.loc[va_mask, ["household_key","snapshot_day"]].reset_index(drop=True)
print("val rows", len(va_keys))

cfgs = [(4,20),(4,40),(4,60),(5,20),(5,40),(5,60),(6,20),(6,40)]
preds = []
for d, w in cfgs:
    for s in range(4):
        m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=d, min_child_weight=w,
                         subsample=0.8, colsample_bytree=0.8, objective="reg:quantileerror",
                         quantile_alpha=0.5, random_state=s, n_jobs=4, tree_method="hist")
        m.fit(Xtr, ytr)
        preds.append(np.clip(m.predict(Xva), 0, None))
p = np.mean(preds, axis=0)
out = va_keys.copy()
out["prediction"] = p.astype(float)
print("pred stats: mean", round(p.mean(),2), "median", round(np.median(p),2), "min", round(p.min(),2), "max", round(p.max(),2))
path = agent_api.save_table(out, "pred_e015")
print(path)
