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
