import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet'); tt = agent_api.train_targets()
cat_cols = [c for c in feats.columns if str(feats[c].dtype)=='category']
codes = feats.copy()
for c in cat_cols:
    codes[c] = codes[c].astype('object').fillna('__NA__').astype('category').cat.codes.astype('int32')+1
FEATS = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = codes[FEATS].astype('float32')
df = codes[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype('float32'); Xtr_all = X.loc[df.index].values
CONFIGS = [dict(n_estimators=400, learning_rate=0.08, max_depth=d, min_child_weight=mcw,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method='hist', n_jobs=4)
           for d in (4,5,6) for mcw in (20,40,60)][:8]
def fit_eval(train_snaps, eval_snaps, wfun, seeds=(0,)):
    tr_mask = df.snapshot_day.isin(train_snaps).values
    Xa, ya = Xtr_all[tr_mask], y[tr_mask]
    sw = wfun(df.snapshot_day[tr_mask].values)
    preds = {s: np.zeros(((df.snapshot_day==s).sum(), len(CONFIGS)*len(seeds))) for s in eval_snaps}
    k=0
    for cfg in CONFIGS:
        for sd in seeds:
            p=dict(cfg); p['seed']=sd; p['quantile_alpha']=0.5
            m=xgb.XGBRegressor(**p); m.fit(Xa,ya,sample_weight=sw)
            for s in eval_snaps:
                preds[s][:,k]=m.predict(Xtr_all[(df.snapshot_day==s).values])
            k+=1
    return {s: np.mean(np.abs(preds[s].mean(axis=1)-df.future_spend_4w[df.snapshot_day==s].values)) for s in eval_snaps}
tr = list(range(95,404,28))
for name, wf in [
    ('32x last2', lambda d: np.where(d>=347,32.,1.)),
    ('64x last2', lambda d: np.where(d>=347,64.,1.)),
    ('lin ramp', lambda d: (431.0-d)/28.0+1.0),
    ('sqrt ramp', lambda d: np.sqrt((431.0-d)/28.0+1.0)),
]:
    r = fit_eval(tr,[403,431],wf)
    print(name, {k: round(v,3) for k,v in r.items()})
r1 = fit_eval(list(range(95,404,28)),[403],lambda d_: np.where(d_>=347,16.,1.))
r2 = fit_eval(list(range(95,432,28)),[431],lambda d_: np.where(d_>=375,16.,1.))
print('16x WF: 403', round(r1[403],3), '431', round(r2[431],3))
r3 = fit_eval(list(range(95,404,28)),[403],lambda d_: np.ones_like(d_,dtype=float))
r4 = fit_eval(list(range(95,432,28)),[431],lambda d_: np.ones_like(d_,dtype=float))
print('unif WF: 403', round(r3[403],3), '431', round(r4[431],3))