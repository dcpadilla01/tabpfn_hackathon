import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = allF[allF.snapshot_day<=431].copy()
fit = tr[tr.snapshot_day<=375]
ho  = tr[tr.snapshot_day>=403]
import xgboost as xgb

def train_pred(Xtr, ytr, Xho, seeds=(7,), rounds=1200, lr=0.03, decay=140, depth=6, mcw=10, obj='reg:quantileerror', wmul=1.0):
    t0 = Xtr.snapshot_day.max()
    w = np.power(0.5, (t0 - Xtr.snapshot_day.values)/decay)*wmul
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
# clip log predictions at max of train targets? check
print('log pred max', p_log.max().round(1), 'train y max', fit.future_spend_4w.max())

# clip variants
for cap in [1500, 1200, 1000, 900]:
    print('clip@%d:'%cap, mae(ho.future_spend_4w, np.clip(p_log,0,cap)).round(3))

# per-decile bias of log model
qb = pd.qcut(ho.future_spend_4w, 10, duplicates='drop')
g = ho.assign(p=p_log).groupby(qb, observed=True).apply(lambda d: pd.Series({'n':len(d),'y':d.future_spend_4w.mean(),'p':d.p.mean()}), include_groups=False)
print(g.round(2))
# what if we add a positive shift? (bias -22)
for sh in [0, 5, 10, 15]:
    print('shift+%d:'%sh, mae(ho.future_spend_4w, p_log+sh).round(3))
# multiplicative calibration
for k in [1.0, 1.03, 1.05, 1.08]:
    print('x%.2f:'%k, mae(ho.future_spend_4w, p_log*k).round(3))