import agent_api, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

v3 = agent_api.load_saved("feats_v3.parquet")
seas = agent_api.load_saved("feats_seasonal.parquet").drop(columns=["snapshot_day"])
df = v3.merge(seas, on="household_key", suffixes=("","_s"))
# sanity: same snapshot_day
assert (df.snapshot_day == df.snapshot_day_s).all()
df = df.drop(columns=["snapshot_day_s"])
# new seasonal interaction features
df["seas_ratio_364"] = df["lag364_spend"]/(df["spend_28"]+10)
df["seas_ratio_308"] = df["lag308_spend"]/(df["spend_28"]+10)
df["seas_trend"] = (df["lag364_spend"]+1)/(df["lag252_spend"]+1)
df["seas_diff_364"] = df["lag364_spend"] - df["spend_28"]
df["active_y1"] = (df["lag364_bask"]>0).astype(np.float32)
df["seas_ratio_364_8w"] = df["lag364_8w"]/(df["spend_56"]+10)
df["seas_avg_y1"] = (df["lag308_spend"]+df["lag252_spend"]+df["lag364_spend"])/3.0
df["seas_vs_year_avg"] = df["lag364_spend"]/(df["seas_avg_y1"]+10)
feats_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
print("n features:", len(feats_cols))
tt = agent_api.train_targets()
df = df.merge(tt, on=["household_key","snapshot_day"], how="left")
tr = df[df.future_spend_4w.notna()].copy()
va = df[df.future_spend_4w.isna()].copy()
print("train rows", len(tr), "val rows", len(va))
Xtr, ytr = tr[feats_cols].values.astype(np.float32), tr.future_spend_4w.values
Xva = va[feats_cols].values.astype(np.float32)

t0=time.time()
preds = []
def add(p, w=1.0):
    preds.append((p, w))
# XGB median x2 seeds
for s in [7, 17]:
    m = XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=7, min_child_weight=10,
                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="reg:quantileerror",
                     quantile_alpha=0.5, tree_method="hist", n_jobs=8, random_state=s, verbosity=0)
    m.fit(Xtr, ytr); add(m.predict(Xva))
print("xgb med done %.0fs"%(time.time()-t0))
# XGB squared x1
m = XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=7, min_child_weight=10,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="reg:squarederror",
                 tree_method="hist", n_jobs=8, random_state=7, verbosity=0)
m.fit(Xtr, ytr); add(m.predict(Xva))
print("xgb sq done %.0fs"%(time.time()-t0))
# HGB quantile x2 seeds
for s in [7, 17]:
    m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=400, learning_rate=0.06,
                                      max_leaf_nodes=31, random_state=s)
    m.fit(Xtr, ytr); add(m.predict(Xva))
print("hgb done %.0fs"%(time.time()-t0))
W = np.array([w for _,w in preds], float)
P = np.vstack([p for p,_ in preds])
pred = (W[:,None]*P).sum(0)/W.sum()
out = va[["household_key","snapshot_day"]].copy()
out["prediction"] = pred
print(out.prediction.describe().round(2))
p = agent_api.save_table(out, "pred_e014.parquet")
print("saved", p, "%.0fs"%(time.time()-t0))
