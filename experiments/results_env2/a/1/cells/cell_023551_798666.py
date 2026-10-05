import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = allF[allF.snapshot_day<=431].copy()
fit = tr[tr.snapshot_day<=375]
ho  = tr[tr.snapshot_day>=403]

def mae(yt, yp): return np.abs(yt-yp).mean()

# XGB quantile (alpha=.5) like E005 on all features; internal holdout to measure variance
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

def fit_xgb(Xtr, ytr, Xho, seeds=(7,), rounds=1200, lr=0.03, decay=140):
    t0 = (Xtr.snapshot_day.max() if 'snapshot_day' in Xtr else 375)
    w = np.power(0.5, (t0 - Xtr.snapshot_day.values)/decay)
    preds = []
    for s in seeds:
        m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=lr, max_depth=6, min_child_weight=10,
                             subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, reg_alpha=0.0,
                             objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist', n_jobs=8, random_state=s)
        m.fit(Xtr[feat_cols].values, ytr, sample_weight=w)
        preds.append(m.predict(Xho[feat_cols].values))
    return np.mean(preds, axis=0)

Xtr, ytr = fit, fit.future_spend_4w.values
p = fit_xgb(Xtr, ytr, ho, seeds=(7,11,13,19), rounds=1200)
print('internal holdout MAE (E005-style, 4 seeds):', mae(ho.future_spend_4w, p).round(3))
print('bias', (p-ho.future_spend_4w).mean().round(2))
# per-snapshot
hh = ho.copy(); hh['p']=p
print(hh.groupby('snapshot_day').apply(lambda d: mae(d.future_spend_4w, d.p), include_groups=False).round(2))
# compare spend_84/3 on same holdout
print('spend_84/3 MAE:', mae(ho.future_spend_4w, ho.spend_84/3).round(3))
print('blend .35 MAE:', mae(ho.future_spend_4w, .35*ho.spend_28+.65*ho.spend_84/3).round(3))