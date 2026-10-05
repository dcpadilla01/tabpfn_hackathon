import pandas as pd, numpy as np, xgboost as xgb, time
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = allF[allF.snapshot_day<=375].copy()          # 11 train snapshots, E005 regime
va = allF[allF.snapshot_day>431].copy()           # validation rows
print('train rows', len(tr), 'val rows', len(va))

t0 = 375
w = np.power(0.5, (t0 - tr.snapshot_day.values)/140.0)

def fit_mean(obj, yvals, seeds, rounds=1200):
    ps=[]
    for s in seeds:
        kw = dict(n_estimators=rounds, learning_rate=0.03, max_depth=6, min_child_weight=10,
                  subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, tree_method='hist', n_jobs=8, random_state=s)
        if obj=='reg:quantileerror': kw.update(objective=obj, quantile_alpha=0.5)
        else: kw['objective']=obj
        m = xgb.XGBRegressor(**kw)
        m.fit(tr[feat_cols].values, yvals, sample_weight=w)
        ps.append(m.predict(va[feat_cols].values))
    return np.mean(ps, axis=0)

t=time.time()
ylog = np.log1p(tr.future_spend_4w.values)
p_log = np.expm1(fit_mean('reg:quantileerror', ylog, seeds=(7,11,13,19,23,29,31,37)))
print('A done %.0fs'%(time.time()-t))
p_sq  = fit_mean('reg:squarederror', tr.future_spend_4w.values, seeds=(7,11,13))
print('B done %.0fs'%(time.time()-t))

blend = 0.8*p_log + 0.2*p_sq
anchor = va['spend_84'].fillna(0).values/3.0
pred = np.clip(0.7*blend + 0.3*anchor, 0, None)
print('pred stats', np.round([pred.min(), pred.mean(), pred.median(), pred.max()],2))

out = va[['household_key','snapshot_day']].copy()
out['prediction'] = pred
path = agent_api.save_table(out, 'e013_preds.parquet')
print(path, out.shape)