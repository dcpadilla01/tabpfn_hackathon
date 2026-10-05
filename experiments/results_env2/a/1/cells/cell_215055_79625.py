
import pandas as pd, numpy as np, xgboost as xgb
f = agent_api.load_saved("e002_features.parquet")
tt = agent_api.train_targets().set_index(["household_key","snapshot_day"])
f = f.join(tt.future_spend_4w.rename("y"), on=["household_key","snapshot_day"])
Xcols = [c for c in f.columns if c not in ["index","household_key","snapshot_day","y"]]
tr = f[f.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403])].dropna(subset=["y"])
ho = f[f.snapshot_day==431].dropna(subset=["y"])
Xtr,ytr,Xho,yho = tr[Xcols], tr.y, ho[Xcols], ho.y
def mae(p): return round(float(np.abs(p-yho).mean()),3)

try:
    m = xgb.XGBRegressor(n_estimators=1200, objective="reg:quantileerror", quantile_alpha=0.5,
                         learning_rate=0.03, max_depth=5, min_child_weight=5, subsample=0.8,
                         colsample_bytree=0.7, reg_lambda=5, tree_method="hist", n_jobs=8)
    m.fit(Xtr, ytr); print("quantile median:", mae(m.predict(Xho)))
except Exception as e: print("quantile err:", type(e).__name__, e)

m = xgb.XGBRegressor(n_estimators=1200, objective="reg:pseudohubererror", learning_rate=0.03, max_depth=5,
                     min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=5, tree_method="hist", n_jobs=8)
m.fit(Xtr, ytr); p = m.predict(Xho)
for q in [0.97,0.99,1.0]:
    c = np.quantile(p, q); print("clip@",q, mae(np.minimum(p,c)))
for lr,num in [(0.02,1800),(0.015,2400)]:
    m2 = xgb.XGBRegressor(n_estimators=num, objective="reg:pseudohubererror", learning_rate=lr, max_depth=5,
                          min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=5, tree_method="hist", n_jobs=8)
    m2.fit(Xtr, ytr); print("huber",lr,num, mae(m2.predict(Xho)))

from sklearn.ensemble import HistGradientBoostingRegressor
hg = HistGradientBoostingRegressor(loss="absolute_error", max_iter=600, learning_rate=0.05, min_samples_leaf=40,
                                   l2_regularization=1.0, random_state=0)
hg.fit(Xtr, ytr); print("skHGB abs:", mae(hg.predict(Xho)))
