import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = allF[allF.snapshot_day<=431].copy()
fit = tr[tr.snapshot_day<=375]
ho  = tr[tr.snapshot_day>=403]
import xgboost as xgb

def train_pred(Xtr, ytr, Xho, seeds=(7,), rounds=1200, lr=0.03, decay=140, depth=6, mcw=10, obj='reg:quantileerror'):
    t0 = Xtr.snapshot_day.max()
    w = np.power(0.5, (t0 - Xtr.snapshot_day.values)/decay)
    ps=[]
    for s in seeds:
        kw = dict(n_estimators=rounds, learning_rate=lr, max_depth=depth, min_child_weight=mcw,
                  subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, tree_method='hist', n_jobs=8, random_state=s)
        if obj=='reg:quantileerror': kw.update(objective=obj, quantile_alpha=0.5)
        else: kw['objective']=obj
        m = xgb.XGBRegressor(**kw)
        m.fit(Xtr[feat_cols].values, ytr, sample_weight=w)
        ps.append(m.predict(Xho[feat_cols].values))
    return np.mean(ps, axis=0)

def mae(yt, yp): return np.abs(yt-yp).mean()

p_log = np.expm1(train_pred(fit, np.log1p(fit.future_spend_4w.values), ho))
# two-model blend grid on internal holdout
p_sq = None
t0 = fit.snapshot_day.max(); w = np.power(0.5,(t0-fit.snapshot_day.values)/140)
m2 = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=10, subsample=0.8,
                     colsample_bytree=0.6, reg_lambda=1.0, objective='reg:squarederror', tree_method='hist', n_jobs=8, random_state=7)
m2.fit(fit[feat_cols].values, fit.future_spend_4w.values, sample_weight=w)
p_sq = m2.predict(ho[feat_cols].values)
best=(None,9e9)
for wa in [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]:
    p = wa*p_log + (1-wa)*p_sq
    v = mae(ho.future_spend_4w, p)
    if v<best[1]: best=(wa,v)
    print('wa=%.1f  %.3f'%(wa,v))
print('best wa', best)
# also blend with spend_84/3
for w3 in [0,.1,.2,.3]:
    p = .7*p_log+.3*p_sq
    p = (1-w3)*p + w3*(ho.spend_84/3)
    print('w3=%.1f'%w3, mae(ho.future_spend_4w,p).round(3))