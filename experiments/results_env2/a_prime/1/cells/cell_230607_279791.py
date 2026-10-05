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
