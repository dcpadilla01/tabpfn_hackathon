import agent_api, pandas as pd, numpy as np

tt = agent_api.train_targets()
SPLITS=[('S1',[95,123,151,179,207,235,263,291,319,347],[375,403,431]),
        ('S2',[95,123,151,179,207,235,263,291,319],[347,375,403,431]),
        ('S3',[95,123,151,179,207,235,263,291,319,347,375,403],[431])]

def ridge_fit_pred(Xtr,ytr,Xva,alpha):
    A=np.hstack([Xtr,np.ones((len(Xtr),1))])
    B=np.hstack([Xva,np.ones((len(Xva),1))])
    d=A.shape[1]
    G=A.T@A
    lam=alpha*np.eye(d); lam[-1,-1]=0.0
    w=np.linalg.solve(G+lam, A.T@ytr)
    return B@w

def local_eval(tab, alphas=(10.0,), label=''):
    feat=[c for c in tab.columns if c not in ('household_key','snapshot_day')]
    t2=tab.merge(tt,on=['household_key','snapshot_day'],how='left')
    rows=[]
    for sname,trd,vad in SPLITS:
        tr=t2[t2.snapshot_day.isin(trd)]; va=t2[t2.snapshot_day.isin(vad)]
        Xtr=tr[feat].astype(float); Xva=va[feat].astype(float)
        med=Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
        mu=Xtr.mean(); sd=Xtr.std().replace(0,1.0)
        Xtr=((Xtr-mu)/sd).values; Xva=((Xva-mu)/sd).values
        ytr=tr.future_spend_4w.values; yva=va.future_spend_4w.values
        r={'split':sname,'n_va':len(va)}
        for a in alphas:
            p=ridge_fit_pred(Xtr,ytr,Xva,a)
            r[f'r{a:g}']=float(np.abs(p-yva).mean())
        rows.append(r)
    df=pd.DataFrame(rows)
    print(label); print(df.round(3).to_string(index=False))
    print('MEAN:',{k:round(x,3) for k,x in df.mean(numeric_only=True).items()})
    return df

e18=agent_api.load_saved('e018_hinge_prune.parquet')
local_eval(e18,alphas=(5.0,10.0,20.0,50.0),label='E018 (official val 60.746)')