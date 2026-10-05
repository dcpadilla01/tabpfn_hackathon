
import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')

full = agent_api.load_saved('feats_v4.parquet')
t = agent_api.train_targets()
tr = full.merge(t, on=['household_key','snapshot_day'], how='inner')
FCOLS = [c for c in full.columns if c not in ('household_key','snapshot_day')]
Xtr = tr[FCOLS].values.astype(float); ytr = tr['future_spend_4w'].values
val = full[full.snapshot_day >= 459]
Xv = val[FCOLS].values.astype(float)
print("train:", Xtr.shape, "val:", Xv.shape)

def fit(seed=1, obj='reg:squarederror', q=None):
    kw = dict(n_estimators=2000, learning_rate=0.02, max_depth=7, min_child_weight=10,
              subsample=0.8, colsample_bytree=0.8, objective=obj, random_state=seed,
              n_jobs=8, tree_method='hist')
    if q is not None: kw['quantile_alpha'] = q
    m = xgb.XGBRegressor(**kw); m.fit(Xtr, ytr); return m

m1 = fit(1); m2 = fit(2, obj='reg:quantileerror', q=0.5)
pred = 0.5*m1.predict(Xv) + 0.5*m2.predict(Xv)
out = val[['household_key','snapshot_day']].copy()
out['prediction'] = pred.astype(np.float32)
print(out.shape, out['prediction'].describe().round(2).to_dict())
agent_api.save_table(out, 'pred_e007.parquet')
