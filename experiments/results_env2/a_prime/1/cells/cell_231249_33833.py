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
