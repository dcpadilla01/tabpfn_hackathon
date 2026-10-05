import numpy as np, pandas as pd, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
L = agent_api.load_saved('lagfeats.parquet')
F = F.merge(L, on=['household_key','snapshot_day'], how='left')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = F[(F.snapshot_day<=403) & F.future_spend_4w.notna()].copy()
va = F[F.snapshot_day==431].copy()
yv = va.future_spend_4w.values
LAG = ['lag28_56','lag56_84','lag84_112','lag112_140']
print('corr with target (431):', {c: round(np.corrcoef(va[c], yv)[0,1],3) for c in LAG})

def fit_predict(feats, lr=0.03, rounds=2400, half=140, seed=7):
    w = 0.5 ** ((tr.snapshot_day.max() - tr.snapshot_day.astype(float).values) / half)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5,
        n_estimators=rounds, learning_rate=lr, max_depth=0, max_leaves=31,
        grow_policy='lossguide', subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, reg_lambda=1.0, n_jobs=8, random_state=seed, tree_method='hist')
    m.fit(tr[feats].values, tr.future_spend_4w.values, sample_weight=w, verbose=False)
    return m.predict(va[feats].values)

t0=time.time()
pL = fit_predict(FEATS)
print('with lag feats: MAE %.3f (%.0fs)' % (np.abs(pL-yv).mean(), time.time()-t0))
# blend lag model with reference 63.049? just report; also try 4-seed avg of lag model
preds=[fit_predict(FEATS, seed=s) for s in range(4)]
print('lag 4-seed:     MAE %.3f' % np.abs(np.mean(preds,axis=0)-yv).mean())
