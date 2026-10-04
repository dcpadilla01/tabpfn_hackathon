import pandas as pd, numpy as np, agent_api

df = agent_api.load_saved('e013_denoise.parquet')
tt = agent_api.train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
feat = [c for c in m.columns if c not in ('household_key','snapshot_day','future_spend_4w')]

# find duplicate / constant columns on train rows
sub = m[m.snapshot_day<347][feat]
nun = sub.nunique()
const = nun[nun<=1].index.tolist()
print('constant cols:', const)
corr = sub.corr().abs()
dup = [(a,b) for i,a in enumerate(feat) for b in feat[i+1:] if corr.loc[a,b]>0.999]
print('near-dup pairs:', dup[:40], '... total', len(dup))

def prep(tr, te, cols, alpha, logskew=False):
    Xtr = tr[cols].to_numpy(float); Xte = te[cols].to_numpy(float)
    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    if logskew:
        sk = np.abs(pd.DataFrame(Xtr).skew().to_numpy())
        ms = sk > 2
        Xtr = np.where(ms[None,:], np.sign(Xtr)*np.log1p(np.abs(Xtr)), Xtr)
        Xte = np.where(ms[None,:], np.sign(Xte)*np.log1p(np.abs(Xte)), Xte)
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
    Ztr = (Xtr-mu)/sd; Zte = (Xte-mu)/sd
    Ztr = np.hstack([Ztr, np.ones((len(Ztr),1))]); Zte = np.hstack([Zte, np.ones((len(Zte),1))])
    ytr = tr.future_spend_4w.to_numpy(float)
    p = Ztr.shape[1]
    A = Ztr.T@Ztr + alpha*np.eye(p); A[-1,-1] -= alpha
    w = np.linalg.pinv(A) @ (Ztr.T@ytr)
    return Zte@w

def cv(m, cols, alphas, eval_days=(347,375,403,431), logskew=False):
    out={}
    for a in alphas:
        per=[]
        for D in eval_days:
            tr=m[m.snapshot_day<D]; te=m[m.snapshot_day==D]
            pr=prep(tr,te,cols,a,logskew)
            per.append(np.mean(np.abs(pr-te.future_spend_4w.to_numpy(float))))
        out[a]=round(float(np.mean(per)),3)
    return out

alphas=[0,1,3,10,30,100,300,1000]
print('E013 alpha curve:', cv(m, feat, alphas))
print('E013 alpha curve logskew:', cv(m, feat, alphas, logskew=True))