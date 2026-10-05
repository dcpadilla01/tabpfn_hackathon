import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feats = [c for c in df.columns if c not in drop]
days = sorted(df.snapshot_day.unique())
print(days)

def fit_pred(tr, va, params, feats):
    m = xgb.XGBRegressor(n_estimators=params['n'], learning_rate=params['lr'],
        max_depth=params['md'], min_child_weight=params['mcw'], subsample=params['ss'],
        colsample_bytree=params['cs'], reg_lambda=params['rl'], gamma=params.get('g',0.0),
        objective='reg:absoluteerror', tree_method='hist', n_jobs=8, random_state=0)
    m.fit(tr[feats], tr.future_spend_4w, eval_set=[(va[feats], va.future_spend_4w)], verbose=False)
    return m.predict(va[feats])

t0=time.time()
# time-based OOF: each snapshot predicted by model trained on earlier snapshots
oof = np.zeros(len(df)); seen = np.zeros(len(df))
for d in days:
    va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
    if len(tr)==0: continue
    p = fit_pred(tr, va, dict(n=1200,lr=0.03,md=7,mcw=40,ss=0.8,cs=0.7,rl=1.0), feats)
    oof[df.index.isin(va.index)] = p; seen[df.index.isin(va.index)]=1
mask = seen>0
print('OOF MAE', np.abs(oof[mask]-df.future_spend_4w.values[mask]).mean(), 'time', time.time()-t0)
print('OOF R2', 1-((oof[mask]-df.future_spend_4w.values[mask])**2).sum()/((df.future_spend_4w.values[mask]-df.future_spend_4w.values[mask].mean())**2).sum())
