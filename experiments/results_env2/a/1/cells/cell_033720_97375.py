import pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet'); tt = agent_api.train_targets()
FEAT = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = tt.merge(F.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'], how='left')
use_days = [95,123,151,179,207,235,263,291,319,347,375]
w = 0.5 ** ((431 - tr.snapshot_day)/140.0); m = tr.snapshot_day.isin(use_days)
Xw = tr.loc[m,FEAT]; yw = tr.loc[m,'future_spend_4w'].values; sw = w.loc[m].values
t0=time.time()
sq_preds = []
for seed in [7,17,27]:
    msq = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1200, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_lambda=5.0, tree_method='hist', n_jobs=8, random_state=seed)
    msq.fit(Xw, yw, sample_weight=sw)
    sq_preds.append(msq.predict(F[FEAT]))
    print('seed', seed, 'done %.0fs' % (time.time()-t0))
P = F[['household_key','snapshot_day']].copy()
for i,p in enumerate(sq_preds): P['sq%d'%i] = p
agent_api.save_table(P, 'e016_sqpreds.parquet')
print('saved sq preds', P.shape)