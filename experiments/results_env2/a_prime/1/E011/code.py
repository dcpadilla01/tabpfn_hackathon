
import agent_api, pandas as pd, numpy as np
np.random.seed(0)

df = agent_api.load_saved('e006_dynamics.parquet')
t = agent_api.train_targets()
d = df.merge(t, on=['household_key','snapshot_day'], how='inner')
cols = [c for c in df.columns if c not in ('household_key','snapshot_day')]
print('rows', len(d), 'snapshots', sorted(d.snapshot_day.unique()), 'nfeat', len(cols))
print(df.dtypes.value_counts().to_dict())
print('target:', t.future_spend_4w.describe().round(2).to_dict())
print('zero share', round((t.future_spend_4w==0).mean(),3))
print('naive MAE median', round(np.abs(t.future_spend_4w - t.future_spend_4w.median()).mean(),2))

mats=[]
for c in cols:
    s=d[c]
    if not np.issubdtype(s.dtype, np.number):
        s=pd.Series(pd.factorize(s)[0], index=s.index)
    mats.append(pd.to_numeric(s, errors='coerce').to_numpy(dtype=float))
X=np.column_stack(mats); y=d.future_spend_4w.to_numpy(float); day=d.snapshot_day.to_numpy()

def fit_pred(Xtr,ytr,Xte,lam):
    mu=np.nanmean(Xtr,axis=0); sd=np.nanstd(Xtr,axis=0); sd[sd==0]=1.0
    Ztr=np.nan_to_num((Xtr-mu)/sd); Zte=np.nan_to_num((Xte-mu)/sd)
    Ztr=np.hstack([Ztr,np.ones((len(Ztr),1))]); Zte=np.hstack([Zte,np.ones((len(Zte),1))])
    P=np.eye(Ztr.shape[1])*lam; P[-1,-1]=0.0
    w=np.linalg.solve(Ztr.T@Ztr+P, Ztr.T@ytr)
    return Zte@w

days=sorted(set(day.tolist()))
best=(None,1e9)
for lam in [3,10,30,100,300]:
    errs=[]
    for k in days:
        tr=day!=k; te=day==k
        p=fit_pred(X[tr],y[tr],X[te],lam)
        errs.append(np.abs(p-y[te]))
    e=np.concatenate(errs).mean()
    print('ridge LOSO lam',lam,'MAE',round(e,3))
    if e<best[1]: best=(lam,e)
lam=best[0]; print('best lam',lam, round(best[1],3))

# log-target variant
for lam in [10,100]:
    errs=[]
    yl=np.log1p(y)
    for k in days:
        tr=day!=k; te=day==k
        p=np.clip(np.expm1(fit_pred(X[tr],yl[tr],X[te],lam)),0,None)
        errs.append(np.abs(p-y[te]))
    print('log-target lam',lam,'LOSO MAE',round(np.concatenate(errs).mean(),3))

# permutation importance on 3 holdout train days
imp=[]
for j,c in enumerate(cols):
    deltas=[]
    for k in [431,403,375]:
        tr=day!=k; te=day==k
        p0=fit_pred(X[tr],y[tr],X[te],lam)
        Xp=X[te].copy(); Xp[:,j]=Xp[np.random.permutation(len(Xp)),j]
        p1=fit_pred(X[tr],y[tr],Xp,lam)
        deltas.append(np.abs(p1-y[te]).mean()-np.abs(p0-y[te]).mean())
    imp.append((c,float(np.mean(deltas))))
imp.sort(key=lambda z:z[1])
print('\nHARMFUL (perm lowers MAE) 15:')
for c,v in imp[:15]: print(round(v,3), c)
print('USEFUL 15:')
for c,v in imp[-15:]: print(round(v,3), c)
print('\nmid-range 30:')
for c,v in imp[15:45]: print(round(v,3), c)


# ---- cell ----
import pandas as pd, numpy as np, agent_api
df=agent_api.load_saved('e006_dynamics.parquet'); t=agent_api.train_targets()
d=df.merge(t,on=['household_key','snapshot_day'],how='inner')
cols=[c for c in df.columns if c not in ('household_key','snapshot_day')]
mats=[]
for c in cols:
    s=d[c]
    if s.dtype.kind not in 'iufcb':
        s=pd.Series(pd.factorize(s)[0],index=s.index)
    mats.append(pd.to_numeric(s,errors='coerce').to_numpy(dtype=float))
X=np.column_stack(mats); y=d.future_spend_4w.to_numpy(float); day=d.snapshot_day.to_numpy()

def fit_pred(Xtr,ytr,Xte,lam):
    mu=np.nanmean(Xtr,axis=0); sd=np.nanstd(Xtr,axis=0); sd[sd==0]=1.0
    Ztr=np.nan_to_num((Xtr-mu)/sd); Zte=np.nan_to_num((Xte-mu)/sd)
    Ztr=np.hstack([Ztr,np.ones((len(Ztr),1))]); Zte=np.hstack([Zte,np.ones((len(Zte),1))])
    P=np.eye(Ztr.shape[1])*lam; P[-1,-1]=0.0
    w=np.linalg.solve(Ztr.T@Ztr+P, Ztr.T@ytr)
    return Zte@w

days=sorted(set(day.tolist()))
best=(None,1e9)
for lam in [3,10,30,100,300]:
    errs=[]
    for k in days:
        tr=day!=k; te=day==k
        p=fit_pred(X[tr],y[tr],X[te],lam)
        errs.append(np.abs(p-y[te]))
    e=np.concatenate(errs).mean()
    print('ridge LOSO lam',lam,'MAE',round(e,3))
    if e<best[1]: best=(lam,e)
lam=best[0]; print('best lam',lam, round(best[1],3))

for lam in [10,100]:
    errs=[]; yl=np.log1p(y)
    for k in days:
        tr=day!=k; te=day==k
        p=np.clip(np.expm1(fit_pred(X[tr],yl[tr],X[te],lam)),0,None)
        errs.append(np.abs(p-y[te]))
    print('log-target lam',lam,'LOSO MAE',round(np.concatenate(errs).mean(),3))

imp=[]
for j,c in enumerate(cols):
    deltas=[]
    for k in [431,403,375]:
        tr=day!=k; te=day==k
        p0=fit_pred(X[tr],y[tr],X[te],lam)
        Xp=X[te].copy(); Xp[:,j]=Xp[np.random.permutation(len(Xp)),j]
        p1=fit_pred(X[tr],y[tr],Xp,lam)
        deltas.append(np.abs(p1-y[te]).mean()-np.abs(p0-y[te]).mean())
    imp.append((c,float(np.mean(deltas))))
imp.sort(key=lambda z:z[1])
print('\nHARMFUL 15 (perm lowers MAE):')
for c,v in imp[:15]: print(round(v,3), c)
print('\nUSEFUL 15:')
for c,v in imp[-15:]: print(round(v,3), c)
print('\nMID 30:')
for c,v in imp[15:45]: print(round(v,3), c)


# ---- cell ----
import pandas as pd, numpy as np, agent_api
df=agent_api.load_saved('e006_dynamics.parquet'); t=agent_api.train_targets()
d=df.merge(t,on=['household_key','snapshot_day'],how='inner')
cols=[c for c in df.columns if c not in ('household_key','snapshot_day')]
print(len(cols),'cols:')
print(cols)

# target by snapshot (drift check)
print('\ntrain target mean/median by snapshot:')
print(d.groupby('snapshot_day').future_spend_4w.agg(['mean','median','count']).round(1))
print('\nspend_28 mean by snapshot (all rows):')
print(df.groupby('snapshot_day').spend_28.mean().round(1) if 'spend_28' in df.columns else 'no spend_28 col')


# ---- cell ----
import pandas as pd, numpy as np, agent_api
df=agent_api.load_saved('e006_dynamics.parquet'); t=agent_api.train_targets()
d=df.merge(t,on=['household_key','snapshot_day'],how='inner')
cols=[c for c in df.columns if c not in ('household_key','snapshot_day')]

def to_mats(d, cols):
    mats=[]
    for c in cols:
        s=d[c]
        if s.dtype.kind not in 'iufcb':
            s=pd.Series(pd.factorize(s)[0],index=s.index)
        mats.append(pd.to_numeric(s,errors='coerce').to_numpy(dtype=float))
    return np.column_stack(mats)

y=d.future_spend_4w.to_numpy(float); day=d.snapshot_day.to_numpy()
def fit_pred(Xtr,ytr,Xte,lam=10):
    mu=np.nanmean(Xtr,axis=0); sd=np.nanstd(Xtr,axis=0); sd[sd==0]=1.0
    Ztr=np.nan_to_num((Xtr-mu)/sd); Zte=np.nan_to_num((Xte-mu)/sd)
    Ztr=np.hstack([Ztr,np.ones((len(Ztr),1))]); Zte=np.hstack([Zte,np.ones((len(Zte),1))])
    P=np.eye(Ztr.shape[1])*lam; P[-1,-1]=0.0
    w=np.linalg.solve(Ztr.T@Ztr+P, Ztr.T@ytr)
    return Zte@w
def loso(X,y,day):
    errs=[]
    for k in sorted(set(day.tolist())):
        tr=day!=k; te=day==k
        errs.append(np.abs(fit_pred(X[tr],y[tr],X[te])-y[te]))
    return np.concatenate(errs).mean()

# permutation importance ranking (all 13 folds, faster convergence)
rng=np.random.default_rng(1)
X=to_mats(d,cols)
base=[]
for k in sorted(set(day.tolist())):
    tr=day!=k; te=day==k
    p0=fit_pred(X[tr],y[tr],X[te]); base.append(np.abs(p0-y[te]).mean())
print('base LOSO', round(np.mean(base),3))
imp={}
for j,c in enumerate(cols):
    deltas=[]
    for k in sorted(set(day.tolist())):
        tr=day!=k; te=day==k
        Xp=X[te].copy(); Xp[:,j]=Xp[rng.permutation(len(Xp)),j]
        p1=fit_pred(X[tr],y[tr],Xp)
        deltas.append(np.abs(p1-y[te]).mean()-np.abs(p0-y[te]).mean())
    imp[c]=np.mean(deltas)
rank=sorted(cols,key=lambda c:imp[c],reverse=True)
print('top30:',[(c,round(imp[c],2)) for c in rank[:30]])

# subset tests
for K in [100,80,60,40,25]:
    sub=rank[:K]
    print('top',K,'LOSO',round(loso(to_mats(d,sub),y,day),3))

# cross-sectional level feature: mean spend_28 across households at same snapshot
lvl=d.groupby('snapshot_day').spend_28.mean().rename('era_mean_spend')
d2=d.merge(lvl,left_on='snapshot_day',right_index=True,how='left')
X2=np.hstack([X,d2[['era_mean_spend']].to_numpy(float)])
print('with era_mean LOOC', round(loso(X2,y,day),3))

# era-relative: spend_28 minus era mean
d2['rel_spend28']=d2.spend_28-d2.era_mean_spend
X3=np.hstack([X,d2[['era_mean_spend','rel_spend28']].to_numpy(float)])
print('with era+rel LOSO', round(loso(X3,y,day),3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api
df=agent_api.load_saved('e006_dynamics.parquet'); t=agent_api.train_targets()
d=df.merge(t,on=['household_key','snapshot_day'],how='inner')
cols=[c for c in df.columns if c not in ('household_key','snapshot_day')]

def to_mats(d, cols):
    mats=[]
    for c in cols:
        s=d[c]
        if s.dtype.kind not in 'iufcb':
            s=pd.Series(pd.factorize(s)[0],index=s.index)
        mats.append(pd.to_numeric(s,errors='coerce').to_numpy(dtype=float))
    return np.column_stack(mats)

y=d.future_spend_4w.to_numpy(float); day=d.snapshot_day.to_numpy()
def fit_pred(Xtr,ytr,Xte,lam=10):
    mu=np.nanmean(Xtr,axis=0); sd=np.nanstd(Xtr,axis=0); sd[sd==0]=1.0
    Ztr=np.nan_to_num((Xtr-mu)/sd); Zte=np.nan_to_num((Xte-mu)/sd)
    Ztr=np.hstack([Ztr,np.ones((len(Ztr),1))]); Zte=np.hstack([Zte,np.ones((len(Zte),1))])
    P=np.eye(Ztr.shape[1])*lam; P[-1,-1]=0.0
    w=np.linalg.solve(Ztr.T@Ztr+P, Ztr.T@ytr)
    return Zte@w
def loso(X,y,day):
    errs=[]
    for k in sorted(set(day.tolist())):
        tr=day!=k; te=day==k
        errs.append(np.abs(fit_pred(X[tr],y[tr],X[te])-y[te]))
    return np.concatenate(errs).mean()

rng=np.random.default_rng(1)
X=to_mats(d,cols)
days=sorted(set(day.tolist()))
imp={}
for j,c in enumerate(cols):
    deltas=[]
    for k in days:
        tr=day!=k; te=day==k
        p0=fit_pred(X[tr],y[tr],X[te])
        Xp=X[te].copy(); Xp[:,j]=Xp[rng.permutation(len(Xp)),j]
        p1=fit_pred(X[tr],y[tr],Xp)
        deltas.append(np.abs(p1-y[te]).mean()-np.abs(p0-y[te]).mean())
    imp[c]=float(np.mean(deltas))
rank=sorted(cols,key=lambda c:imp[c],reverse=True)
print('top30:',[(c,round(imp[c],2)) for c in rank[:30]])
print('bottom10:',[(c,round(imp[c],2)) for c in rank[-10:]])

for K in [147,100,80,60,40,25]:
    print('top',K,'LOSO',round(loso(to_mats(d,rank[:K]),y,day),3))

lvl=d.groupby('snapshot_day').spend_28.mean().rename('era_mean_spend')
d2=d.merge(lvl,left_on='snapshot_day',right_index=True,how='left')
d2['rel_spend28']=d2.spend_28-d2.era_mean_spend
X2=np.hstack([X,d2[['era_mean_spend']].to_numpy(float)])
X3=np.hstack([X,d2[['era_mean_spend','rel_spend28']].to_numpy(float)])
print('with era_mean LOSO', round(loso(X2,y,day),3))
print('with era+rel LOSO', round(loso(X3,y,day),3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api
df=agent_api.load_saved('e006_dynamics.parquet'); t=agent_api.train_targets()
d=df.merge(t,on=['household_key','snapshot_day'],how='inner')
cols=[c for c in df.columns if c not in ('household_key','snapshot_day')]
def to_mats(d, cols):
    mats=[]
    for c in cols:
        s=d[c]
        if s.dtype.kind not in 'iufcb': s=pd.Series(pd.factorize(s)[0],index=s.index)
        mats.append(pd.to_numeric(s,errors='coerce').to_numpy(dtype=float))
    return np.column_stack(mats)
y=d.future_spend_4w.to_numpy(float); day=d.snapshot_day.to_numpy()
def fit_pred(Xtr,ytr,Xte,lam=10):
    mu=np.nanmean(Xtr,axis=0); sd=np.nanstd(Xtr,axis=0); sd[sd==0]=1.0
    Ztr=np.nan_to_num((Xtr-mu)/sd); Zte=np.nan_to_num((Xte-mu)/sd)
    Ztr=np.hstack([Ztr,np.ones((len(Ztr),1))]); Zte=np.hstack([Zte,np.ones((len(Zte),1))])
    P=np.eye(Ztr.shape[1])*lam; P[-1,-1]=0.0
    return Zte@np.linalg.solve(Ztr.T@Ztr+P, Ztr.T@ytr)
X=to_mats(d,cols); days=sorted(set(day.tolist()))
pred=np.zeros(len(y))
for k in days:
    tr=day!=k; te=day==k
    pred[te]=fit_pred(X[tr],y[tr],X[te])
err=np.abs(pred-y)
print('proxy MAE',round(err.mean(),2))
# references
print('MAE pred=spend_28:', round(np.abs(d.spend_28-y).mean(),2))
print('MAE pred=0.9*spend_28:', round(np.abs(0.9*d.spend_28-y).mean(),2))
# bias by predicted decile
q=pd.qcut(pred,10,labels=False,duplicates='drop')
print('\ndecile: pred_mean actual_mean bias  MAE  n')
for q_ in sorted(set(q)):
    m=q==q_
    print(q_, round(pred[m].mean(),1), round(y[m].mean(),1), round((pred[m]-y[m]).mean(),1), round(err[m].mean(),1), m.sum())
# optimal global scale/offset
for c in [0.8,0.9,0.95,1.0]:
    print('scale',c,'MAE',round(np.abs(pred*c-y).mean(),2))
# zeros
z=y==0
print('\nzero rows:',z.sum(),'MAE on them',round(err[z].mean(),1),'mean pred on them',round(pred[z].mean(),1))
print('their spend_28 median',d.loc[z,'spend_28'].median(),'recency median',d.loc[z,'recency'].median())
print('nonzero spend_28 median',d.loc[~z,'spend_28'].median(),'recency median',d.loc[~z,'recency'].median())
# rule: predict 0 if spend_28 < thr
for thr in [0,5,10,20]:
    p2=np.where(d.spend_28<thr,0,pred)
    print('zero-rule thr',thr,'MAE',round(np.abs(p2-y).mean(),2))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings('ignore')

e6 = agent_api.load_saved('e006_dynamics.parquet')

def fn(view, snapshot_day):
    tx = view.table('transactions')
    wk = int(view.week)
    lo = max(1, wk-77)
    tx = tx[(tx.week_no >= lo) & (tx.week_no <= wk)]
    g = tx.groupby(['household_key','week_no']).sales_value.sum()
    weeks = np.arange(lo, wk+1)
    ages = wk - weeks
    w28 = 0.5**(ages/4.0); w56 = 0.5**(ages/8.0)
    out = {}
    for h, sub in g.groupby(level=0):
        s = sub.droplevel(0)
        a = pd.Series(0.0, index=weeks)
        a.loc[s.index] = (s > 0).astype(float).values
        out[h] = dict(
            act_ewma28=float((a*w28).sum()/w28.sum()),
            act_ewma56=float((a*w56).sum()/w56.sum()),
            act_share_13w=float(a.iloc[-13:].mean()),
            act_share_26w=float(a.iloc[-26:].mean()),
            act_share_52w=float(a.iloc[-52:].mean()),
        )
    df = pd.DataFrame.from_dict(out, orient='index')
    df.index.name='household_key'
    return df.reindex(view.households)

probe = agent_api.build_features(fn)
print('probe shape', probe.shape)
agent_api.save_table(probe, 'e011_probe.parquet')

# assemble candidates
p = probe.copy()
base = e6.merge(p.reset_index(), on=['household_key','snapshot_day'], how='inner')
th=5.0
base['sp28_lo']=np.minimum(base.spend_28,th); base['sp28_hi']=np.maximum(base.spend_28-th,0)
base['ew28_lo']=np.minimum(base.d_ewma_spend_hl28,th); base['ew28_hi']=np.maximum(base.d_ewma_spend_hl28-th,0)
base['ew28_lo25']=np.minimum(base.d_ewma_spend_hl28,25.0); base['ew28_hi25']=np.maximum(base.d_ewma_spend_hl28-25.0,0)
base['lapsed']=(base.spend_28<th).astype(float)
base['exp_spend']=base.spend_28*base.act_ewma28
base['exp_spend2']=base.d_ewma_spend_hl28*base.act_ewma28
base['rec_lo']=np.minimum(base.recency,28.0); base['rec_hi']=np.maximum(base.recency-28.0,0)
agent_api.save_table(base, 'e011_cand.parquet')
t = agent_api.train_targets()
d = base.merge(t, on=['household_key','snapshot_day'])
print('cand rows', len(d))
print(d[['act_ewma28','act_share_26w','sp28_lo','lapsed','exp_spend']].describe().round(3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings('ignore')
d=agent_api.load_saved('e011_cand.parquet'); t=agent_api.train_targets()
d=d.merge(t,on=['household_key','snapshot_day'],how='inner')
cols=[c for c in d.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
def to_mats(d, cols):
    mats=[]
    for c in cols:
        s=d[c]
        if s.dtype.kind not in 'iufcb': s=pd.Series(pd.factorize(s)[0],index=s.index)
        mats.append(pd.to_numeric(s,errors='coerce').to_numpy(dtype=float))
    return np.column_stack(mats)
y=d.future_spend_4w.to_numpy(float); day=d.snapshot_day.to_numpy()
def fit_pred(Xtr,ytr,Xte,lam=10):
    mu=np.nanmean(Xtr,axis=0); sd=np.nanstd(Xtr,axis=0); sd[sd==0]=1.0
    Ztr=np.nan_to_num((Xtr-mu)/sd); Zte=np.nan_to_num((Xte-mu)/sd)
    Ztr=np.hstack([Ztr,np.ones((len(Ztr),1))]); Zte=np.hstack([Zte,np.ones((len(Zte),1))])
    P=np.eye(Ztr.shape[1])*lam; P[-1,-1]=0.0
    return Zte@np.linalg.solve(Ztr.T@Ztr+P, Ztr.T@ytr)
def loso(cols_):
    X=to_mats(d,cols_)
    errs=[]
    for k in sorted(set(day.tolist())):
        tr=day!=k; te=day==k
        errs.append(np.abs(fit_pred(X[tr],y[tr],X[te])-y[te]))
    return np.concatenate(errs).mean()

print('full cand', round(loso(cols),3), len(cols))
for K in [60,50,45,40,35,30]:
    print('top',K,round(loso(cols[:K]),3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings('ignore')
d=agent_api.load_saved('e011_cand.parquet'); t=agent_api.train_targets()
d=d.merge(t,on=['household_key','snapshot_day'],how='inner')
E6=[c for c in agent_api.load_saved('e006_dynamics.parquet').columns if c not in ('household_key','snapshot_day')]
NEW=['sp28_lo','sp28_hi','ew28_lo','ew28_hi','ew28_lo25','ew28_hi25','lapsed','exp_spend','exp_spend2','rec_lo','rec_hi','act_ewma28','act_ewma56','act_share_13w','act_share_26w','act_share_52w']
def to_mats(d_, cols):
    mats=[]
    for c in cols:
        s=d_[c]
        if s.dtype.kind not in 'iufcb': s=pd.Series(pd.factorize(s)[0],index=s.index)
        mats.append(pd.to_numeric(s,errors='coerce').to_numpy(dtype=float))
    return np.column_stack(mats)
y=d.future_spend_4w.to_numpy(float); day=d.snapshot_day.to_numpy(); days=sorted(set(day.tolist()))
def fit_pred(Xtr,ytr,Xte,lam):
    mu=np.nanmean(Xtr,axis=0); sd=np.nanstd(Xtr,axis=0); sd[sd==0]=1.0
    Ztr=np.nan_to_num((Xtr-mu)/sd); Zte=np.nan_to_num((Xte-mu)/sd)
    Ztr=np.hstack([Ztr,np.ones((len(Ztr),1))]); Zte=np.hstack([Zte,np.ones((len(Zte),1))])
    P=np.eye(Ztr.shape[1])*lam; P[-1,-1]=0.0
    return Zte@np.linalg.solve(Ztr.T@Ztr+P, Ztr.T@ytr)
def loso(cols_,lam=10):
    X=to_mats(d,cols_); errs=[]
    for k in days:
        tr=day!=k; te=day==k
        errs.append(np.abs(fit_pred(X[tr],y[tr],X[te],lam)-y[te]))
    return np.concatenate(errs).mean()

# rank E6 cols
rng=np.random.default_rng(2); X=to_mats(d,E6)
imp={}
for j,c in enumerate(E6):
    dl=[]
    for k in days:
        tr=day!=k; te=day==k
        p0=fit_pred(X[tr],y[tr],X[te],10)
        Xp=X[te].copy(); Xp[:,j]=Xp[rng.permutation(len(Xp)),j]
        p1=fit_pred(X[tr],y[tr],Xp,10)
        dl.append(np.abs(p1-y[te]).mean()-np.abs(p0-y[te]).mean())
    imp[c]=float(np.mean(dl))
rank=sorted(E6,key=lambda c:imp[c],reverse=True)
print('ranks 30-50:',[(c,round(imp[c],2)) for c in rank[30:50]])

V={'top30':rank[:30],'top45':rank[:45],'top60':rank[:60]}
for name,base in V.items():
    for lam in [3,10,30,100]:
        print(name,'lam',lam,round(loso(base,lam),3))
print()
H1=['sp28_lo','sp28_hi','ew28_lo','ew28_hi','rec_lo','rec_hi']
H2=H1+['lapsed','act_ewma28','act_share_26w']
H3=H1+['lapsed','act_ewma28','act_share_26w','exp_spend','ew28_lo25','ew28_hi25']
for hn,H in [('H1',H1),('H2',H2),('H3',H3)]:
    for lam in [3,10,30,100]:
        print('top45+',hn,'lam',lam,round(loso(rank[:45]+H,lam),3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings('ignore')
d=agent_api.load_saved('e011_cand.parquet'); t=agent_api.train_targets()
d=d.merge(t,on=['household_key','snapshot_day'],how='inner')
print('cand cols sample:', [c for c in d.columns if 'index' in c])
E6=[c for c in agent_api.load_saved('e006_dynamics.parquet').columns if c not in ('household_key','snapshot_day')]
E6=[c+'__x' if c+'__x' in d.columns else c for c in E6]
E6=[c for c in E6 if c in d.columns and c!='index']
NEW=['sp28_lo','sp28_hi','ew28_lo','ew28_hi','ew28_lo25','ew28_hi25','lapsed','exp_spend','exp_spend2','rec_lo','rec_hi','act_ewma28','act_ewma56','act_share_13w','act_share_26w','act_share_52w']
NEW=[c for c in NEW if c in d.columns]
def to_mats(d_, cols):
    mats=[]
    for c in cols:
        s=d_[c]
        if s.dtype.kind not in 'iufcb': s=pd.Series(pd.factorize(s)[0],index=s.index)
        mats.append(pd.to_numeric(s,errors='coerce').to_numpy(dtype=float))
    return np.column_stack(mats)
y=d.future_spend_4w.to_numpy(float); day=d.snapshot_day.to_numpy(); days=sorted(set(day.tolist()))
def fit_pred(Xtr,ytr,Xte,lam):
    mu=np.nanmean(Xtr,axis=0); sd=np.nanstd(Xtr,axis=0); sd[sd==0]=1.0
    Ztr=np.nan_to_num((Xtr-mu)/sd); Zte=np.nan_to_num((Xte-mu)/sd)
    Ztr=np.hstack([Ztr,np.ones((len(Ztr),1))]); Zte=np.hstack([Zte,np.ones((len(Zte),1))])
    P=np.eye(Ztr.shape[1])*lam; P[-1,-1]=0.0
    return Zte@np.linalg.solve(Ztr.T@Ztr+P, Ztr.T@ytr)
def loso(cols_,lam=10):
    X=to_mats(d,cols_); errs=[]
    for k in days:
        tr=day!=k; te=day==k
        errs.append(np.abs(fit_pred(X[tr],y[tr],X[te],lam)-y[te]))
    return np.concatenate(errs).mean()
rng=np.random.default_rng(2); X=to_mats(d,E6)
imp={}
for j,c in enumerate(E6):
    dl=[]
    for k in days:
        tr=day!=k; te=day==k
        p0=fit_pred(X[tr],y[tr],X[te],10)
        Xp=X[te].copy(); Xp[:,j]=Xp[rng.permutation(len(Xp)),j]
        p1=fit_pred(X[tr],y[tr],Xp,10)
        dl.append(np.abs(p1-y[te]).mean()-np.abs(p0-y[te]).mean())
    imp[c]=float(np.mean(dl))
rank=sorted(E6,key=lambda c:imp[c],reverse=True)
print('ranks 25-50:',[(c,round(imp[c],2)) for c in rank[25:50]])
V={'top30':rank[:30],'top45':rank[:45],'top60':rank[:60]}
for name,base in V.items():
    for lam in [3,10,30,100]:
        print(name,'lam',lam,round(loso(base,lam),3))
print()
H1=['sp28_lo','sp28_hi','ew28_lo','ew28_hi','rec_lo','rec_hi']
H2=H1+['lapsed','act_ewma28','act_share_26w']
H3=H1+['lapsed','act_ewma28','act_share_26w','exp_spend','ew28_lo25','ew28_hi25']
for hn,H in [('H1',H1),('H2',H2),('H3',H3)]:
    for lam in [3,10,30,100]:
        print('top45+',hn,'lam',lam,round(loso(rank[:45]+H,lam),3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings('ignore')
d=agent_api.load_saved('e011_cand.parquet')
TOP45=['d_ewma_spend_hl28','d_ewma_spend_hl56','d_ewma_spend_hl14','d_ewma_spend_hl112','trips_84','avg28_all','total_all','d_block_std','days_84','spend_182','blk_1','days_28','spend_28','n_depts_28d','spend_rate_182','trips_56','trips_28','trips_112','d_trips_14d','d_spend_14d','d_block_max','d_zero_wk_streak_10w','d_wk_active_12w','std6','items_mean_84','days_56','d_trips_7d','trend_28_84','slope6','spend_365','spend_56','d_gap_mean_84d','days_365','classification_1','kid_category_desc','blk_5','d_gap_std_84d','days_182','trips_365','blk_11','blk_9','spend_s392','classification_3','spend_rate_84','spend_84']
NEW=['sp28_lo','sp28_hi','ew28_lo','ew28_hi','rec_lo','rec_hi','lapsed','act_ewma28','act_share_26w','exp_spend']
keep=['household_key','snapshot_day']+TOP45+NEW
tab=d[keep].copy()
print(tab.shape, 'missing:', tab.isna().mean().mean().round(3))
agent_api.save_table(tab,'e011_lapsed.parquet')
print('saved', agent_api.snapshot_days())
