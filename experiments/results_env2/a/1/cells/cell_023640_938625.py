import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = allF[allF.snapshot_day<=431].copy()
fit = tr[tr.snapshot_day<=375]
ho  = tr[tr.snapshot_day>=403]
import xgboost as xgb

def train_pred(Xtr, ytr, Xho, seeds, rounds=1200, lr=0.03, decay=140, depth=6, mcw=10):
    t0 = Xtr.snapshot_day.max()
    w = np.power(0.5, (t0 - Xtr.snapshot_day.values)/decay)
    ps=[]
    for s in seeds:
        m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=lr, max_depth=depth, min_child_weight=mcw,
                             subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
                             objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist', n_jobs=8, random_state=s)
        m.fit(Xtr[feat_cols].values, ytr, sample_weight=w)
        ps.append(m.predict(Xho[feat_cols].values))
    return np.mean(ps, axis=0)

def mae(yt, yp): return np.abs(yt-yp).mean()

# A) quantile on log1p(y)
p_log = train_pred(fit, np.log1p(fit.future_spend_4w.values), ho)
print('A log-target quantile :', mae(ho.future_spend_4w, np.expm1(p_log)).round(3))

# B) absoluteerror objective
t0 = fit.snapshot_day.max()
w = np.power(0.5, (t0-fit.snapshot_day.values)/140)
m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=10, subsample=0.8,
                     colsample_bytree=0.6, reg_lambda=1.0, objective='reg:absoluteerror', tree_method='hist', n_jobs=8, random_state=7)
m.fit(fit[feat_cols].values, fit.future_spend_4w.values, sample_weight=w)
p_abs = m.predict(ho[feat_cols].values)
print('B absoluteerror       :', mae(ho.future_spend_4w, p_abs).round(3))

# C) squared error
m2 = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=10, subsample=0.8,
                     colsample_bytree=0.6, reg_lambda=1.0, objective='reg:squarederror', tree_method='hist', n_jobs=8, random_state=7)
m2.fit(fit[feat_cols].values, fit.future_spend_4w.values, sample_weight=w)
p_sq = m2.predict(ho[feat_cols].values)
print('C squarederror        :', mae(ho.future_spend_4w, p_sq).round(3))
print('  bias C', (p_sq-ho.future_spend_4w).mean().round(2), ' bias A', (np.expm1(p_log)-ho.future_spend_4w).mean().round(2))

# D) blend A+C
print('D blend .5A+.5C       :', mae(ho.future_spend_4w, .5*np.expm1(p_log)+.5*p_sq).round(3))
print('D blend .7C+.3A       :', mae(ho.future_spend_4w, .7*p_sq+.3*np.expm1(p_log)).round(3))
print('E blend C + spend84/3 :', mae(ho.future_spend_4w, .8*p_sq+.2*ho.spend_84/3).round(3))