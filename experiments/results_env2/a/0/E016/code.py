import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved('feats_v4.parquet')
print('feats_v4', feats.shape)
cols = list(feats.columns)
print(len(cols), cols)
print(feats.dtypes.value_counts())
pred = agent_api.load_saved('pred_e013.parquet')
print('pred_e013', pred.shape, list(pred.columns))
print(pred['prediction'].describe())
tt = agent_api.train_targets()
print('tt', tt.shape, sorted(tt.snapshot_day.unique()))
print(tt.future_spend_4w.describe())
print('val snaps', agent_api.snapshot_days())

# ---- cell ----
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

# ---- cell ----
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
    ('3x last2', lambda d: np.where(d>=347,3.,1.)),
    ('4x last2', lambda d: np.where(d>=347,4.,1.)),
    ('exp hl112', lambda d: 2**(-(431-d)/112.)),
    ('exp hl56', lambda d: 2**(-(431-d)/56.)),
    ('2x last3', lambda d: np.where(d>=319,2.,1.)),
]:
    r = fit_eval(tr,[403,431],wf)
    print(name, {k: round(v,3) for k,v in r.items()})

# ---- cell ----
import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet'); tt = agent_api.train_targets()
cat_cols = [c for c in feats.columns if str(feats[c].dtype)=='category']
codes = feats.copy()
for c in cat_cols:
    codes[c] = codes[c].astype('object').fillna('__NA__').astype('category').cat.codes.astype('int32')+1
FEATS = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = codes[FEATS].astype('float32')
df = codes[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day','future_spend_4w'], how='inner')
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
    ('6x last2', lambda d: np.where(d>=347,6.,1.)),
    ('8x last2', lambda d: np.where(d>=347,8.,1.)),
    ('2x 347+ 2x 375+', lambda d: np.where(d>=375,3.,np.where(d>=347,2.,1.))),
    ('2x 375+ only', lambda d: np.where(d>=375,2.,1.)),
    ('4x 375+ 2x 347+', lambda d: np.where(d>=375,4.,np.where(d>=347,2.,1.))),
]:
    r = fit_eval(tr,[403,431],wf)
    print(name, {k: round(v,3) for k,v in r.items()})

# ---- cell ----
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
    ('6x last2', lambda d: np.where(d>=347,6.,1.)),
    ('8x last2', lambda d: np.where(d>=347,8.,1.)),
    ('2x347 3x375', lambda d: np.where(d>=375,3.,np.where(d>=347,2.,1.))),
    ('2x 375 only', lambda d: np.where(d>=375,2.,1.)),
    ('4x375 2x347', lambda d: np.where(d>=375,4.,np.where(d>=347,2.,1.))),
]:
    r = fit_eval(tr,[403,431],wf)
    print(name, {k: round(v,3) for k,v in r.items()})

# ---- cell ----
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
for name, ts, wf in [
    ('16x last2', tr, lambda d: np.where(d>=347,16.,1.)),
    ('only last2', [347,375], lambda d: np.ones_like(d,dtype=float)),
    ('only last3', [319,347,375], lambda d: np.ones_like(d,dtype=float)),
    ('only last4', [291,319,347,375], lambda d: np.ones_like(d,dtype=float)),
    ('8x last2+2x319', tr, lambda d: np.where(d>=347,8.,np.where(d>=319,2.,1.))),
]:
    r = fit_eval(ts,[403,431],wf)
    print(name, {k: round(v,3) for k,v in r.items()})

# ---- cell ----
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
def fit_eval(train_snaps, eval_snaps, wfun, seeds=(0,), ret=False):
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
    out = {s: np.mean(np.abs(preds[s].mean(axis=1)-df.future_spend_4w[df.snapshot_day==s].values)) for s in eval_snaps}
    return (out, preds) if ret else out
tr = list(range(95,404,28))
def lastN(N, x):
    D = max(x)
    return np.where(d>=D-28*(N-1) if False else x>=D-28*(N-1), 1.0, 1.0)
def mk(N, x):
    D = max(x); return np.where(x>=D-28*(N-1), float(x), 1.0)
for name, wf in [
    ('32x last2', lambda d: mk(2,d)),
    ('64x last2', lambda d: mk(2,d) if False else np.where(d>=347,64.,1.)),
    ('lin ramp', lambda d: (431-d)/28.0+1.0),
    ('sqrt ramp', lambda d: np.sqrt((431-d)/28.0+1.0)),
]:
    r = fit_eval(tr,[403,431],wf)
    print(name, {k: round(v,3) for k,v in r.items()})
# walk-forward sanity for 16x last2 and uniform
d=None
r1 = fit_eval(list(range(95,404,28)),[403],lambda d_: np.where(d_>=347,16.,1.))
r2 = fit_eval(list(range(95,432,28)),[431],lambda d_: np.where(d_>=375,16.,1.))
print('16x WF: 403', round(r1[403],3), '431', round(r2[431],3))
r3 = fit_eval(list(range(95,404,28)),[403],lambda d_: np.ones_like(d_,dtype=float))
r4 = fit_eval(list(range(95,432,28)),[431],lambda d_: np.ones_like(d_,dtype=float))
print('unif WF: 403', round(r3[403],3), '431', round(r4[431],3))

# ---- cell ----
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

# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet'); tt = agent_api.train_targets()
g = tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median','count'])
print(g.round(2))
# proxy for level at validation snapshots: exp4w_blend mean per snapshot (all rows incl. val)
g2 = feats.groupby('snapshot_day').exp4w_blend.agg(['mean','median'])
print(g2.round(2).loc[[403,431,459,487,515,543]])
# also spend_28 mean per snapshot
print(feats.groupby('snapshot_day').spend_28.mean().round(2).loc[[95,207,319,403,431,459,487,515,543]])

# ---- cell ----
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
# 2-seed bags, walk-forward eval, uniform vs 16x-last2 vs 24x-last2
for name, wf in [('unif', lambda d_: np.ones_like(d_,dtype=float)),
                 ('16x', lambda d_: np.where(d_>=max(d_)-28,16.,1.)),
                 ('24x', lambda d_: np.where(d_>=max(d_)-28,24.,1.))]:
    r1 = fit_eval(list(range(95,404,28)),[403],wf,seeds=(0,1))
    r2 = fit_eval(list(range(95,432,28)),[431],wf,seeds=(0,1))
    print(name, '403', round(r1[403],3), '431', round(r2[431],3))

# ---- cell ----
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
for name, wf in [('48x', lambda d_: np.where(d_>=max(d_)-28,48.,1.)),
                 ('96x', lambda d_: np.where(d_>=max(d_)-28,96.,1.)),
                 ('200x', lambda d_: np.where(d_>=max(d_)-28,200.,1.)),
                 ('1e4x', lambda d_: np.where(d_>=max(d_)-28,1e4,1.))]:
    r1 = fit_eval(list(range(95,404,28)),[403],wf,seeds=(0,1))
    r2 = fit_eval(list(range(95,432,28)),[431],wf,seeds=(0,1))
    print(name, '403', round(r1[403],3), '431', round(r2[431],3))

# ---- cell ----
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

# ---- cell ----
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

# ---- cell ----
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
TR = list(range(95,432,28))
tr_mask = df.snapshot_day.isin(TR).values
Xa, ya = Xtr_all[tr_mask], y[tr_mask]
sw = np.where(df.snapshot_day[tr_mask].values>=347, 1e4, 1.0)
val_mask = df.snapshot_day.isin([459,487,515,543]).values
Xv = Xtr_all[val_mask]
P = np.zeros((Xv.shape[0], len(CONFIGS)*2)); k=0
for cfg in CONFIGS:
    for sd in (0,1):
        p=dict(cfg); p['seed']=sd; p['quantile_alpha']=0.5; p['tree_method']='hist'; p['n_jobs']=4
        m=xgb.XGBRegressor(**p); m.fit(Xa,ya,sample_weight=sw)
        P[:,k]=m.predict(Xv); k+=1
pred = P.mean(axis=1)
out = df.loc[val_mask, ['household_key','snapshot_day']].copy()
out['prediction']=pred
print(out.shape, out.snapshot_day.value_counts().to_dict())
print(out.prediction.describe().round(3))
path = agent_api.save_table(out, 'pred_e016.parquet')
print('saved', path)