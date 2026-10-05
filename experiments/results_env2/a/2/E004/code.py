import agent_api, pandas as pd, numpy as np
f3 = agent_api.load_saved('feats_v3.parquet')
print(f3.shape)
print(list(f3.columns))
tt = agent_api.train_targets()
print(tt.shape, tt.head())
print(tt.future_spend_4w.describe())


# ---- cell ----
import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
print(df.shape, df.snapshot_day.unique())
print(df.dtypes.value_counts())
obj_cols = [c for c in df.columns if df[c].dtype==object]
print(obj_cols)
for c in obj_cols: print(c, df[c].nunique())


# ---- cell ----
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


# ---- cell ----
import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feats = [c for c in df.columns if c not in drop]
days = sorted(df.snapshot_day.unique())

def fit_pred(tr, va, params, feats, target='future_spend_4w', objective='reg:absoluteerror'):
    m = xgb.XGBRegressor(n_estimators=params['n'], learning_rate=params['lr'],
        max_depth=params['md'], min_child_weight=params['mcw'], subsample=params['ss'],
        colsample_bytree=params['cs'], reg_lambda=params['rl'], gamma=params.get('g',0.0),
        objective=objective, tree_method='hist', n_jobs=8, random_state=0)
    y = tr[target]
    if objective=='reg:absoluteerror': ytr = y
    else: ytr = y
    m.fit(tr[feats], ytr, eval_set=[(va[feats], va[target])], verbose=False)
    return m.predict(va[feats])

t0=time.time()
def run_oof(mode, params=dict(n=1200,lr=0.03,md=7,mcw=40,ss=0.8,cs=0.7,rl=1.0)):
    oof = np.zeros(len(df)); seen=np.zeros(len(df))
    for d in days:
        va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
        if len(tr)==0: continue
        if mode=='base':
            p = fit_pred(tr,va,params,feats)
        elif mode=='log':
            tr2 = tr.copy(); tr2['y']=np.log1p(tr2.future_spend_4w)
            p = np.expm1(fit_pred(tr2,va,params,feats,target='y'))
        elif mode=='hh_offset':
            p = fit_pred(tr,va,params,feats)
            # per-household mean residual from train
            tr2 = tr.copy(); tr2['res']=tr2.future_spend_4w-p  # careful: p is for va rows
            # need residuals on train: refit quick? instead compute via crossval within train
            # simpler: use per-household mean of (target - pred) where pred from a model trained on other train days
            # approximate: residual = target - model trained on all train
            mfull = xgb.XGBRegressor(n_estimators=params['n'],learning_rate=params['lr'],max_depth=params['md'],
                min_child_weight=params['mcw'],subsample=params['ss'],colsample_bytree=params['cs'],
                reg_lambda=params['rl'],objective='reg:absoluteerror',tree_method='hist',n_jobs=8,random_state=0)
            mfull.fit(tr[feats],tr.future_spend_4w,verbose=False)
            r = tr.future_spend_4w - mfull.predict(tr[feats])
            hh = tr.assign(r=r).groupby('household_key').r.mean()
            shrunken = hh*0.7
            p = p + va.household_key.map(shrunken).fillna(0).values
        oof[df.index.isin(va.index)]=p; seen[df.index.isin(va.index)]=1
    m_ = seen>0
    mae = np.abs(oof[m_]-df.future_spend_4w.values[m_]).mean()
    print(mode,'OOF MAE',round(mae,3),'time',round(time.time()-t0,1)); return mae

run_oof('base')
run_oof('log')
run_oof('hh_offset')


# ---- cell ----
import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feats = [c for c in df.columns if c not in drop]
days = sorted(df.snapshot_day.unique())
P = dict(n=1200,lr=0.03,md=7,mcw=40,ss=0.8,cs=0.7,rl=1.0)

def mk(n,lr,md,mcw,ss,cs,rl,obj):
    return xgb.XGBRegressor(n_estimators=n,learning_rate=lr,max_depth=md,min_child_weight=mcw,
        subsample=ss,colsample_bytree=cs,reg_lambda=rl,objective=obj,tree_method='hist',n_jobs=8,random_state=0)

t0=time.time()
oof_log = np.zeros(len(df)); oof_off = np.zeros(len(df)); seen=np.zeros(len(df))
for d in days:
    va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
    if len(tr)==0: continue
    # log target
    m = mk(P['n'],P['lr'],P['md'],P['mcw'],P['ss'],P['cs'],P['rl'],'reg:squarederror')
    m.fit(tr[feats], np.log1p(tr.future_spend_4w), eval_set=[(va[feats],np.log1p(va.future_spend_4w))], verbose=False)
    p_log = np.expm1(m.predict(va[feats]))
    # per-household residual offset from full-train model
    mfull = mk(P['n'],P['lr'],P['md'],P['mcw'],P['ss'],P['cs'],P['rl'],'reg:absoluteerror')
    mfull.fit(tr[feats],tr.future_spend_4w,verbose=False)
    r = tr.future_spend_4w - mfull.predict(tr[feats])
    hh = tr.assign(r=r).groupby('household_key').r.mean()
    for shrink in (0.5,0.7,1.0):
        p_off = mfull.predict(va[feats]) + shrink*va.household_key.map(hh).fillna(0).values
        idx = df.index.isin(va.index)
        if shrink==0.5: oof_off[idx]=p_off
    oof_log[df.index.isin(va.index)]=p_log
    seen[df.index.isin(va.index)]=1
    if d==days[-1]:
        pass
mask=seen>0
yv = df.future_spend_4w.values[mask]
print('log  OOF MAE', round(np.abs(oof_log[mask]-yv).mean(),3), 'R2', round(1-((oof_log[mask]-yv)**2).sum()/((yv-yv.mean())**2).sum(),4), round(time.time()-t0,1))
print('off  OOF MAE', round(np.abs(oof_off[mask]-yv).mean(),3), 'R2', round(1-((oof_off[mask]-yv)**2).sum()/((yv-yv.mean())**2).sum(),4))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feats = [c for c in df.columns if c not in drop]
days = sorted(df.snapshot_day.unique())

def mk(n,lr,md,mcw,ss,cs,rl):
    return xgb.XGBRegressor(n_estimators=n,learning_rate=lr,max_depth=md,min_child_weight=mcw,
        subsample=ss,colsample_bytree=cs,reg_lambda=rl,objective='reg:absoluteerror',tree_method='hist',n_jobs=8,random_state=0)

def oof_mae(params):
    oof = np.zeros(len(df)); seen=np.zeros(len(df))
    for d in days:
        va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
        if len(tr)==0: continue
        m = mk(**params)
        m.fit(tr[feats],tr.future_spend_4w,verbose=False)
        oof[df.index.isin(va.index)]=m.predict(va[feats]); seen[df.index.isin(va.index)]=1
    mask=seen>0
    return np.abs(oof[mask]-df.future_spend_4w.values[mask]).mean(), oof, mask

t0=time.time()
configs = {
 'md5':  dict(n=1500,lr=0.03,md=5,mcw=40,ss=0.8,cs=0.7,rl=1.0),
 'md9':  dict(n=1200,lr=0.03,md=9,mcw=40,ss=0.8,cs=0.7,rl=1.0),
 'mcw10':dict(n=1200,lr=0.03,md=7,mcw=10,ss=0.8,cs=0.7,rl=1.0),
 'mcw100':dict(n=1200,lr=0.03,md=7,mcw=100,ss=0.8,cs=0.7,rl=1.0),
}
res={}
for k,p in configs.items():
    mae,oof,mask = oof_mae(p)
    res[k]=(mae,oof,mask)
    print(k, round(mae,3), round(time.time()-t0,1))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feats = [c for c in df.columns if c not in drop]
days = sorted(df.snapshot_day.unique())

def mk(obj='reg:absoluteerror',n=1200,lr=0.03,md=7,mcw=10,ss=0.8,cs=0.7,rl=1.0):
    return xgb.XGBRegressor(n_estimators=n,learning_rate=lr,max_depth=md,min_child_weight=mcw,
        subsample=ss,colsample_bytree=cs,reg_lambda=rl,objective=obj,tree_method='hist',n_jobs=8,random_state=0)

def oof_preds(obj='reg:absoluteerror',**kw):
    oof = np.zeros(len(df)); seen=np.zeros(len(df))
    for d in days:
        va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
        if len(tr)==0: continue
        m = mk(obj=obj,**kw)
        m.fit(tr[feats],tr.future_spend_4w,verbose=False)
        oof[df.index.isin(va.index)]=m.predict(va[feats]); seen[df.index.isin(va.index)]=1
    return oof, seen>0

def mae(p,mask): 
    y=df.future_spend_4w.values[mask]; return np.abs(p[mask]-y).mean()

t0=time.time()
p1,m = oof_preds(n=2000,lr=0.02,md=7,mcw=10); print('lr.02', round(mae(p1,m),3), round(time.time()-t0,1))
p2,_ = oof_preds(md=6,mcw=10); print('md6', round(mae(p2,m),3), round(time.time()-t0,1))
p3,_ = oof_preds(cs=0.5,mcw=10); print('cs.5', round(mae(p3,m),3), round(time.time()-t0,1))
p4,m4 = oof_preds(obj='reg:squarederror',mcw=10); print('L2', round(mae(p4,m4),3), round(time.time()-t0,1))
pb = 0.75*p1+0.25*p4
print('blend lr.02+L2', round(mae(pb,m),3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feats = [c for c in df.columns if c not in drop]
days = sorted(df.snapshot_day.unique())

def mk(obj='reg:absoluteerror',n=1200,lr=0.03,md=7,mcw=10,ss=0.8,cs=0.7,rl=1.0):
    return xgb.XGBRegressor(n_estimators=n,learning_rate=lr,max_depth=md,min_child_weight=mcw,
        subsample=ss,colsample_bytree=cs,reg_lambda=rl,objective=obj,tree_method='hist',n_jobs=8,random_state=0)

def oof_preds(obj='reg:absoluteerror',**kw):
    oof = np.zeros(len(df)); seen=np.zeros(len(df))
    for d in days:
        va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
        if len(tr)==0: continue
        m = mk(obj=obj,**kw)
        m.fit(tr[feats],tr.future_spend_4w,verbose=False)
        oof[df.index.isin(va.index)]=m.predict(va[feats]); seen[df.index.isin(va.index)]=1
    return oof, seen>0

def mae(p,mask):
    y=df.future_spend_4w.values[mask]; return np.abs(p[mask]-y).mean()

t0=time.time()
pA,m = oof_preds(n=2000,lr=0.02,md=7,mcw=10); print('lr.02', round(mae(pA,m),3), round(time.time()-t0,1))
pB,_ = oof_preds(obj='reg:squarederror',md=7,mcw=10); print('L2', round(mae(pB,m),3), round(time.time()-t0,1))
for w in (0.5,0.7):
    print('blend',w, round(mae(w*pA+(1-w)*pB,m),3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
g = tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median',lambda x:(x==0).mean()])
g.columns=['mean','median','zero_frac']
print(g.round(2))
# feature drift: spend_28 by snapshot day
f3 = agent_api.load_saved('feats_v3.parquet')
g2 = f3.groupby('snapshot_day')[['spend_28','spend_84','baskets_28']].mean().round(2)
print(g2)
# correlation of snapshot_day with target within household (drift per household)
m = f3[['household_key','snapshot_day','spend_28']].merge(tt,on=['household_key','snapshot_day'])
print('corr(spend_28, target)', m.spend_28.corr(m.future_spend_4w).round(3))
print('corr(snapshot_day, target)', m.snapshot_day.corr(m.future_spend_4w).round(3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feats = [c for c in df.columns if c not in drop]
days = sorted(df.snapshot_day.unique())

def mk(obj='reg:absoluteerror',n=1200,lr=0.03,md=7,mcw=10,ss=0.8,cs=0.7,rl=1.0):
    return xgb.XGBRegressor(n_estimators=n,learning_rate=lr,max_depth=md,min_child_weight=mcw,
        subsample=ss,colsample_bytree=cs,reg_lambda=rl,objective=obj,tree_method='hist',n_jobs=8,random_state=0)

def oof_preds(obj='reg:absoluteerror',**kw):
    oof = np.zeros(len(df)); seen=np.zeros(len(df))
    for d in days:
        va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
        if len(tr)==0: continue
        m = mk(obj=obj,**kw)
        m.fit(tr[feats],tr.future_spend_4w,verbose=False)
        oof[df.index.isin(va.index)]=m.predict(va[feats]); seen[df.index.isin(va.index)]=1
    return oof, seen>0

def mae(p,mask):
    y=df.future_spend_4w.values[mask]; return np.abs(p[mask]-y).mean()

t0=time.time()
pA,m = oof_preds(n=2000,lr=0.02,md=7,mcw=10); print('lr.02', round(mae(pA,m),3), round(time.time()-t0,1))
pB,_ = oof_preds(md=6,mcw=10); print('md6', round(mae(pB,m),3), round(time.time()-t0,1))
pC,_ = oof_preds(cs=0.5,md=7,mcw=10); print('cs.5', round(mae(pC,m),3), round(time.time()-t0,1))
for w in (0.4,0.5,0.6):
    print('blend A/B',w, round(mae(w*pA+(1-w)*pB,m),3))
print('blend A/B/C .5/.25/.25', round(mae(0.5*pA+0.25*pB+0.25*pC,m),3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(200)
tx = v.table('transactions')
print(tx[['sales_value','retail_disc','coupon_disc','coupon_match_disc','quantity']].describe().round(2))
print(tx.shape)


# ---- cell ----
import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feats = [c for c in df.columns if c not in drop]
days = sorted(df.snapshot_day.unique())

def mk(obj='reg:absoluteerror',n=1200,lr=0.03,md=7,mcw=10,ss=0.8,cs=0.7,rl=1.0):
    return xgb.XGBRegressor(n_estimators=n,learning_rate=lr,max_depth=md,min_child_weight=mcw,
        subsample=ss,colsample_bytree=cs,reg_lambda=rl,objective=obj,tree_method='hist',n_jobs=8,random_state=0)

def oof_preds(obj='reg:absoluteerror',**kw):
    oof = np.zeros(len(df)); seen=np.zeros(len(df))
    for d in days:
        va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
        if len(tr)==0: continue
        m = mk(obj=obj,**kw)
        m.fit(tr[feats],tr.future_spend_4w,verbose=False)
        oof[df.index.isin(va.index)]=m.predict(va[feats]); seen[df.index.isin(va.index)]=1
    return oof, seen>0

def mae(p,mask):
    y=df.future_spend_4w.values[mask]; return np.abs(p[mask]-y).mean()

# quick: does adding day index as feature help? and does dropping weak features help?
t0=time.time()
df['day_idx'] = df.snapshot_day.astype(float)
feats2 = feats+['day_idx']
pD,m = oof_preds(n=1200,lr=0.03,md=7,mcw=10)  # base cfg but mcw10 (new best-ish)
# rebuild with feats2
def oof_preds2(fl,**kw):
    oof = np.zeros(len(df)); seen=np.zeros(len(df))
    for d in days:
        va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
        if len(tr)==0: continue
        m = mk(**kw)
        m.fit(tr[fl],tr.future_spend_4w,verbose=False)
        oof[df.index.isin(va.index)]=m.predict(va[fl]); seen[df.index.isin(va.index)]=1
    return oof, seen>0
pE,mE = oof_preds2(feats2,n=1200,lr=0.03,md=7,mcw=10)
print('base mcw10', round(mae(pD,m),3))
print('with day_idx', round(mae(pE,mE),3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feats = [c for c in df.columns if c not in drop]
days = sorted(df.snapshot_day.unique())

def mk(obj='reg:absoluteerror',n=1200,lr=0.03,md=7,mcw=10,ss=0.8,cs=0.7,rl=1.0):
    return xgb.XGBRegressor(n_estimators=n,learning_rate=lr,max_depth=md,min_child_weight=mcw,
        subsample=ss,colsample_bytree=cs,reg_lambda=rl,objective=obj,tree_method='hist',n_jobs=8,random_state=0)

def oof_preds(**kw):
    oof = np.zeros(len(df)); seen=np.zeros(len(df))
    for d in days:
        va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
        if len(tr)==0: continue
        m = mk(**kw)
        m.fit(tr[feats],tr.future_spend_4w,verbose=False)
        oof[df.index.isin(va.index)]=m.predict(va[feats]); seen[df.index.isin(va.index)]=1
    return oof, seen>0

def mae(p,mask):
    y=df.future_spend_4w.values[mask]; return np.abs(p[mask]-y).mean()

t0=time.time()
# final: average of md7 lr.02 n2000 and md6 base — the two best configs — full-train models for val days
preds_val = []
for kw in [dict(n=2000,lr=0.02,md=7,mcw=10), dict(n=1200,lr=0.03,md=6,mcw=10)]:
    m = mk(**kw)
    tr = df[df.snapshot_day<=431]
    m.fit(tr[feats],tr.future_spend_4w,verbose=False)
    fv = agent_api.load_saved('feats_v3.parquet')
    fv = fv[fv.snapshot_day>=459]
    preds_val.append(m.predict(fv[feats]))
p = np.mean(preds_val,axis=0)
out = fv[['household_key','snapshot_day']].copy()
out['prediction']=p
path = agent_api.save_table(out,'pred_e004.parquet')
print(path, out.shape, out.prediction.describe().round(2))
