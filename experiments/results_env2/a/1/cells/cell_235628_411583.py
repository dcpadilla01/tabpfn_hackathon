import numpy as np, pandas as pd, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = F[(F.snapshot_day<=403) & F.future_spend_4w.notna()]
va = F[F.snapshot_day==431]
yv = va.future_spend_4w.values
print('target mean by train day:')
print(tr.groupby('snapshot_day').future_spend_4w.agg(['mean','median']).round(1).T)

def fit_predict(feats, lr=0.03, rounds=2400, half=140, alpha=0.5, seed=7, leaves=31):
    w = 0.5 ** ((tr.snapshot_day.max() - tr.snapshot_day.astype(float).values) / half)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha,
        n_estimators=rounds, learning_rate=lr, max_depth=0, max_leaves=leaves,
        grow_policy='lossguide', subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, reg_lambda=1.0, n_jobs=8, random_state=seed, tree_method='hist')
    m.fit(tr[feats].values, tr.future_spend_4w.values, sample_weight=w, verbose=False)
    return m.predict(va[feats].values)

t0=time.time()
# A: day-index feature
trA = tr.assign(day_idx=tr.snapshot_day.astype(float)); vaA = va.assign(day_idx=va.snapshot_day.astype(float))
pA = fit_predict(FEATS+['day_idx'])
print('A day_idx:      MAE %.3f (%.0fs)' % (np.abs(pA-yv).mean(), time.time()-t0))

# B: 8-seed ensemble (ref feats)
preds = [fit_predict(FEATS, seed=s) for s in range(8)]
pB = np.mean(preds, axis=0)
print('B 8-seed:       MAE %.3f (%.0fs)' % (np.abs(pB-yv).mean(), time.time()-t0))

# C: alpha blend
pA5 = preds[0]
for a in (0.45, 0.55):
    pa = fit_predict(FEATS, alpha=a, seed=0)
    blend = a*pA5/0.5 + (0.5-a)/0.5*pa
    print('C alpha %.2f blend: MAE %.3f' % (a, np.abs(blend-yv).mean()), '(%.0fs)'%(time.time()-t0))
