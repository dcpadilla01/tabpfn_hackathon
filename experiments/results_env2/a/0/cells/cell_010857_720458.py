import agent_api, pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')

feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()

cat_cols = [c for c in feats.columns if str(feats[c].dtype)=='category']
codes = feats.copy()
for c in cat_cols:
    codes[c] = codes[c].astype('object').fillna('__NA__').astype('category')
    codes[c] = codes[c].cat.codes.astype('int32') + 1
FEATS = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = codes[FEATS].astype('float32')

df = codes[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype('float32')
Xtr_all = X.loc[df.index].values
print('train matrix', Xtr_all.shape)

def fit_eval(configs, train_snaps, eval_snaps, seeds=(0,), qalpha=0.5, logt=False, wfun=None):
    tr_mask = df.snapshot_day.isin(train_snaps).values
    Xa, ya = Xtr_all[tr_mask], y[tr_mask]
    if logt: ya = np.log1p(ya)
    sw = wfun(df.snapshot_day[tr_mask].values) if wfun else None
    preds = {s: np.zeros(((df.snapshot_day==s).sum(), len(configs)*len(seeds))) for s in eval_snaps}
    k=0; t0=time.time()
    for cfg in configs:
        for sd in seeds:
            p = dict(cfg); p['seed']=sd; p['quantile_alpha']=qalpha
            m = xgb.XGBRegressor(**p)
            m.fit(Xa, ya, sample_weight=sw)
            for s in eval_snaps:
                mask = (df.snapshot_day==s).values
                pr = m.predict(Xtr_all[mask])
                if logt: pr = np.expm1(pr)
                preds[s][:,k]=pr
            k+=1
    print('fit time', round(time.time()-t0,1))
    out={}
    for s in eval_snaps:
        yy = df.future_spend_4w[df.snapshot_day==s].values
        out[s]=np.mean(np.abs(preds[s].mean(axis=1)-yy))
    return out, preds

CONFIGS = [dict(n_estimators=400, learning_rate=0.08, max_depth=d, min_child_weight=mcw,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method='hist', n_jobs=4)
           for d in (4,5,6) for mcw in (20,40,60)][:8]

# internal CV: train on snaps <=375, eval on 403 & 431, two recency weightings
tr = list(range(95,404,28))
res_a,_ = fit_eval(CONFIGS, tr, [403,431], seeds=(0,))
print('uniform wts:', {k: round(v,3) for k,v in res_a.items()})
res_b,_ = fit_eval(CONFIGS, tr, [403,431], seeds=(0,), wfun=lambda d: np.where(d>=347,2.0,1.0))
print('recency 2x on 347+:', {k: round(v,3) for k,v in res_b.items()})