import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
d=A.load_saved('cand_v1.parquet').merge(A.train_targets(),on=['household_key','snapshot_day'])
m431=(d.snapshot_day==431).values; tr431=(d.snapshot_day<=403).values
y=d.future_spend_4w.values
print(d[['ew28','pr_s28','tp_act8','tp_nzmean','y' if 'y' in d else 'future_spend_4w']].describe().round(2))
sub=d[m431]
print('corr with y@431:')
for c in ['spend_28','ew28','tp_pred','pr_s28','tp_nzmean','tp_act8']:
    print(c, round(float(np.corrcoef(sub[c].fillna(0),sub.future_spend_4w)[0,1]),3))
def prep1(cols):
    X=d[cols].copy()
    for c in cols:
        if X[c].dtype==object: X[c]=X[c].astype('category').cat.codes.astype(float)
    Xv=X.values.astype(float); med=np.nanmedian(Xv,0); Xv=np.where(np.isnan(Xv),med,Xv)
    return (Xv-Xv.mean(0))/(Xv.std(0)+1e-9)
def fitpred(Xs,lam):
    n=int(tr431.sum()); Xd=np.hstack([np.ones((n,1)),Xs[tr431]])
    M=Xd.T@Xd+lam*np.eye(Xd.shape[1]); M[0,0]-=lam
    w=np.linalg.solve(M,Xd.T@y[tr431])
    return np.hstack([np.ones((int(m431.sum()),1)),Xs[m431]])@w
for c in ['spend_28','ew28','tp_pred','pr_s28']:
    p=fitpred(prep1([c]),10)
    print('single',c,'MAE@431',round(float(np.abs(p-y[m431]).mean()),3))
