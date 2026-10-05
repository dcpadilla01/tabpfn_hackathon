
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
