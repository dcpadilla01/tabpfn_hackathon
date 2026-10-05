
import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb

feats = A.load_saved("feats_prof.parquet")
f3 = A.load_saved("feats_v3.parquet")
fs = A.load_saved("feats_seasonal.parquet")
oof16 = A.load_saved("oof_e016_cv.parquet")

df = feats.merge(f3, on=["household_key","snapshot_day"], how="left", suffixes=("","_v3"))
df = df.merge(fs, on=["household_key","snapshot_day"], how="left")
df = df.merge(oof16[["household_key","snapshot_day","y"]], on=["household_key","snapshot_day"], how="left")
print(df.shape, "missing y:", df.y.isna().sum())

prof_cols = [c for c in feats.columns if c.startswith("prof_")]
Xcols = [c for c in f3.columns if c not in ("household_key","snapshot_day")] + \
        [c for c in fs.columns if c not in ("household_key","snapshot_day")] + prof_cols
X = df[Xcols].astype(float).copy()
y = df["y"].values
days = df["snapshot_day"].values

def run_xgb(Xtr, ytr, Xte, seed=7):
    m = xgb.XGBRegressor(n_estimators=900, learning_rate=0.05, max_depth=6, min_child_weight=8,
                         subsample=0.8, colsample_bytree=0.8, objective="reg:quantileerror",
                         quantile_alpha=0.5, tree_method="hist", random_state=seed, n_jobs=8)
    m.fit(Xtr, ytr)
    return m.predict(Xte)

def run_hgb(Xtr, ytr, Xte, seed=7):
    m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=400, learning_rate=0.06,
                                      max_leaf_nodes=31, random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict(Xte)

tr_mask = ~np.isnan(y)
val_mask = ~tr_mask
pred_med = np.zeros(len(df)); pred_hgb = np.zeros(len(df))
for d in [459,487,515,543]:
    m_te = days==d
    pred_med[m_te] = run_xgb(X[tr_mask], y[tr_mask], X[m_te])
    pred_hgb[m_te] = run_hgb(X[tr_mask], y[tr_mask], X[m_te])
def mae(p): return np.abs(p[val_mask]-y[val_mask]).mean()
print("medXGB+prof MAE:", round(mae(pred_med),3))
print("HGB+prof MAE:", round(mae(pred_hgb),3))
print("blend 0.5/0.5:", round(mae(.5*pred_med+.5*pred_hgb),3))
print("blend 0.4/0.6:", round(mae(.4*pred_med+.6*pred_hgb),3))

# save validation predictions of the best variant
out = df.loc[val_mask, ["household_key","snapshot_day"]].copy()
out["prediction"] = 0.5*pred_med[val_mask] + 0.5*pred_hgb[val_mask]
path = A.save_table(out, "pred_prof.parquet")
print("saved", path)
