import numpy as np, pandas as pd, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = F[(F.snapshot_day<=403) & F.future_spend_4w.notna()]
va = F[F.snapshot_day==431]

def fit_predict(lr, rounds, half, leaves=31, subs=0.8, cols=0.8, mincw=20, seed=7):
    w = 0.5 ** ((tr.snapshot_day.max() - tr.snapshot_day.astype(float).values) / half)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5,
        n_estimators=rounds, learning_rate=lr, max_depth=0, max_leaves=leaves,
        grow_policy='lossguide', subsample=subs, colsample_bytree=cols,
        min_child_weight=mincw, reg_lambda=1.0, n_jobs=8, random_state=seed, tree_method='hist')
    m.fit(tr[FEATS].values, tr.future_spend_4w.values, sample_weight=w, verbose=False)
    return m.predict(va[FEATS].values)

t0=time.time()
cfgs = [
    dict(lr=0.03, rounds=2400, half=140),          # E005 config (reference)
    dict(lr=0.03, rounds=2400, half=277),          # E003 decay
    dict(lr=0.05, rounds=1500, half=140),
    dict(lr=0.03, rounds=2400, half=140, cols=0.6),
    dict(lr=0.03, rounds=2400, half=140, mincw=50),
]
for c in cfgs:
    p = fit_predict(**c)
    print(c, 'MAE %.3f' % np.abs(p-va.future_spend_4w.values).mean(), ' (%.0fs)'%(time.time()-t0))
