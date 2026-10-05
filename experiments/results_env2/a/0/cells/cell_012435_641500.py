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
def fit_eval(configs, train_snaps, eval_snaps, wfun, seeds=(0,1)):
    tr_mask = df.snapshot_day.isin(train_snaps).values
    Xa, ya = Xtr_all[tr_mask], y[tr_mask]
    sw = wfun(df.snapshot_day[tr_mask].values)
    preds = {s: np.zeros(((df.snapshot_day==s).sum(), len(configs)*len(seeds))) for s in eval_snaps}
    k=0
    for cfg in configs:
        for sd in seeds:
            p=dict(cfg); p['seed']=sd; p['quantile_alpha']=0.5; p['tree_method']='hist'; p['n_jobs']=4
            m=xgb.XGBRegressor(**p); m.fit(Xa,ya,sample_weight=sw)
            for s in eval_snaps:
                preds[s][:,k]=m.predict(Xtr_all[(df.snapshot_day==s).values])
            k+=1
    return {s: np.mean(np.abs(preds[s].mean(axis=1)-df.future_spend_4w[df.snapshot_day==s].values)) for s in eval_snaps}
BASE = [dict(n_estimators=400, learning_rate=0.08, max_depth=d, min_child_weight=mcw,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
        for d in (4,5,6) for mcw in (20,40,60)][:8]
SHALLOW = [dict(n_estimators=600, learning_rate=0.05, max_depth=d, min_child_weight=mcw,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0)
        for d in (3,4,5) for mcw in (60,100,150)][:8]
wf_last2 = lambda d_: np.where(d_>=max(d_)-28,1e4,1.)
for name, cfgs in [('base8', BASE), ('shallow8', SHALLOW)]:
    r1 = fit_eval(cfgs, list(range(95,404,28)),[403],wf_last2)
    r2 = fit_eval(cfgs, list(range(95,432,28)),[431],wf_last2)
    print(name, '403', round(r1[403],3), '431', round(r2[431],3))
# 4-seed variance check with base8
r1 = fit_eval(BASE, list(range(95,404,28)),[403],wf_last2,seeds=(0,1,2,3))
r2 = fit_eval(BASE, list(range(95,432,28)),[431],wf_last2,seeds=(0,1,2,3))
print('base8 4seed', '403', round(r1[403],3), '431', round(r2[431],3))