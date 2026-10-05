
import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')

full = agent_api.load_saved('feats_v4.parquet')
t = agent_api.train_targets()
tr = full.merge(t, on=['household_key','snapshot_day'], how='inner')
FCOLS = [c for c in full.columns if c not in ('household_key','snapshot_day')]
trfit = tr[tr.snapshot_day <= 403]; hold = tr[tr.snapshot_day == 431]
Xtr, ytr = trfit[FCOLS].values.astype(float), trfit['future_spend_4w'].values
Xh, yh = hold[FCOLS].values.astype(float), hold['future_spend_4w'].values

def fit(Xa, ya, seed=1, obj='reg:squarederror', q=None):
    kw = dict(n_estimators=2000, learning_rate=0.02, max_depth=7, min_child_weight=10,
              subsample=0.8, colsample_bytree=0.8, objective=obj, random_state=seed,
              n_jobs=8, tree_method='hist')
    if q is not None: kw['quantile_alpha'] = q
    m = xgb.XGBRegressor(**kw); m.fit(Xa, ya); return m

m1 = fit(Xtr, ytr, seed=1); m2 = fit(Xtr, ytr, seed=2, obj='reg:quantileerror', q=0.5)
p = 0.5*m1.predict(Xh) + 0.5*m2.predict(Xh)
def mae(a,b): return np.mean(np.abs(np.asarray(a)-np.asarray(b)))
print("holdout(431) blend MAE with 10 new features:", round(mae(p, yh),3), " (base was 64.549)")
print("mean pred:", round(p.mean(),2), "median pred:", round(np.median(p),2))
