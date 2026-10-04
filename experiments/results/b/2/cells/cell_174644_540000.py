import agent_api, pandas as pd, numpy as np

tt = agent_api.train_targets()
merged = agent_api.load_saved('e019_exposure')
e18cols = [c for c in agent_api.load_saved('e018_hinge_prune.parquet').columns if c not in ('household_key','snapshot_day')]
newcols = [c for c in merged.columns if c not in ('household_key','snapshot_day') and c not in e18cols]

SPLITS=[('S1',[95,123,151,179,207,235,263,291,319,347],[375,403,431]),
        ('S3',[95,123,151,179,207,235,263,291,319,347,375,403],[431])]
def ridge_eval(Xtr,ytr,Xva,alpha):
    A=np.hstack([Xtr.reshape(len(Xtr),-1),np.ones((len(Xtr),1))]); B=np.hstack([Xva.reshape(len(Xva),-1),np.ones((len(Xva),1))])
    d=A.shape[1]; lam=alpha*np.eye(d); lam[-1,-1]=0.0
    w=np.linalg.solve(A.T@A+lam, A.T@ytr); return B@w
t2=merged.merge(tt,on=['household_key','snapshot_day'],how='left')
def prep(cols, sname):
    trd,vad=dict(SPLITS)[sname]
    tr=t2[t2.snapshot_day.isin(trd)]; va=t2[t2.snapshot_day.isin(vad)]
    Xtr=tr[cols].astype(float); Xva=va[cols].astype(float)
    med=Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
    mu=Xtr.mean(); sd=Xtr.std().replace(0,1.0)
    return ((Xtr-mu)/sd).values, ((Xva-mu)/sd).values, tr.future_spend_4w.values, va.future_spend_4w.values

res=[]
for c in newcols:
    for sname,base_cols in [('S1',e18cols),('S3',e18cols)]:
        cols=base_cols+[c]
        Xtr,Xva,ytr,yva=prep(cols,sname)
        a=20.0 if sname=='S1' else 50.0
        p=ridge_eval(Xtr,ytr,Xva,a)
        Xb,Yb,_,_=prep(base_cols,sname)
        pb=ridge_eval(Xb,ytr,Yb,a)
        res.append((c,sname,float(np.abs(p-yva).mean()-np.abs(pb-yva).mean())))
d=pd.DataFrame(res).pivot(index=0,columns=1,values=2)
print('delta MAE (positive = feature HELPS):')
print(d.round(3).sort_values('S1',ascending=False).to_string())
print()
print('sum of deltas:', d.sum(axis=1).sort_values(ascending=False).round(3).to_dict())