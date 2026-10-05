import numpy as np, pandas as pd, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = F[F.future_spend_4w.notna()]
p5 = agent_api.load_saved('e005_preds.parquet')
va = p5[['household_key','snapshot_day']].merge(F[F.snapshot_day>=459], on=['household_key','snapshot_day'], how='left')
print('rows', len(va), 'nan feats:', int(va[FEATS].isna().any(axis=1).sum()))
va[FEATS] = va[FEATS].fillna(0).replace([np.inf,-np.inf], 0)
t0=time.time()
preds=[]
for s in (7, 21, 99):
    w = 0.5 ** ((tr.snapshot_day.max() - tr.snapshot_day.astype(float).values) / 140)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=2400,
        learning_rate=0.03, max_depth=0, max_leaves=31, grow_policy='lossguide', subsample=0.8,
        colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0, n_jobs=8,
        random_state=s, tree_method='hist')
    m.fit(tr[FEATS].values, tr.future_spend_4w.values, sample_weight=w, verbose=False)
    preds.append(m.predict(va[FEATS].values))
va = va.assign(prediction=np.mean(preds, axis=0))
out = va[['household_key','snapshot_day','prediction']]
print(out.shape, 'finite:', np.isfinite(out.prediction).all())
path = agent_api.save_table(out, 'e008_preds.parquet')
print('SAVED', path)
