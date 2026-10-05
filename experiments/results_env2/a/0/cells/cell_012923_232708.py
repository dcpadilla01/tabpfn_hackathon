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
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
           for d in (4,5,6) for mcw in (20,40,60)][:8]
def fit_eval(train_snaps, eval_snaps, wfun, seeds=(0,1)):
    tr_mask = df.snapshot_day.isin(train_snaps).values
    Xa, ya = Xtr_all[tr_mask], y[tr_mask]
    sw = wfun(df.snapshot_day[tr_mask].values)
    preds = {s: np.zeros(((df.snapshot_day==s).sum(), len(CONFIGS)*len(seeds))) for s in eval_snaps}
    k=0
    for cfg in CONFIGS:
        for sd in seeds:
            p=dict(cfg); p['seed']=sd; p['quantile_alpha']=0.5; p['tree_method']='hist'; p['n_jobs']=4
            m=xgb.XGBRegressor(**p); m.fit(Xa,ya,sample_weight=sw)
            for s in eval_snaps:
                preds[s][:,k]=m.predict(Xtr_all[(df.snapshot_day==s).values])
            k+=1
    return {s: np.mean(np.abs(preds[s].mean(axis=1)-df.future_spend_4w[df.snapshot_day==s].values)) for s in eval_snaps}
# blend of model prediction with exp4w_blend (household-level persistence) on WF eval
def run_blend(blend_w):
    r1 = fit_eval(list(range(95,404,28)),[403],wf_last2)
    r2 = fit_eval(list(range(95,432,28)),[431],wf_last2)
    return r1,r2
wf_last2 = lambda d_: np.where(d_>=max(d_)-28,1e4,1.)
import numpy as np
# need raw preds; redo with blend: quick approach - compute preds manually
def fit_preds(train_snaps, eval_snaps, wfun, seeds=(0,1)):
    tr_mask = df.snapshot_day.isin(train_snaps).values
    Xa, ya = Xtr_all[tr_mask], y[tr_mask]
    sw = wfun(df.snapshot_day[tr_mask].values)
    store={}
    for cfg in CONFIGS:
        for sd in seeds:
            p=dict(cfg); p['seed']=sd; p['quantile_alpha']=0.5; p['tree_method']='hist'; p['n_jobs']=4
            m=xgb.XGBRegressor(**p); m.fit(Xa,ya,sample_weight=sw)
            for s in eval_snaps:
                pr = m.predict(Xtr_all[(df.snapshot_day==s).values])
                store.setdefault(s,[]).append(pr)
    return store
st = fit_preds(list(range(95,404,28)),[403],wf_last2)
P = np.mean(st[403],axis=0)
eb = feats.loc[df.index][df.snapshot_day==403].exp4w_blend.values
yy = df.future_spend_4w[df.snapshot_day==403].values
for w in [0.0,0.1,0.2,0.3,0.5]:
    print('blend w=%.1f'%w, round(np.mean(np.abs((1-w)*P + w*eb - yy)),3))