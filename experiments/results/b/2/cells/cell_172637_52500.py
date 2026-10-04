import pandas as pd, numpy as np, agent_api

df = agent_api.load_saved('e013_denoise.parquet')
tt = agent_api.train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
feat = [c for c in m.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
print('rows', len(m), 'feat', len(feat))

def fit_predict(tr, te, cols, alpha, clip_q=None, logskew=False):
    Xtr = tr[cols].to_numpy(float); Xte = te[cols].to_numpy(float)
    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    if clip_q is not None:
        lo = np.quantile(Xtr, clip_q, axis=0); hi = np.quantile(Xtr, 1-clip_q, axis=0)
        Xtr = np.clip(Xtr, lo, hi); Xte = np.clip(Xte, lo, hi)
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
    w = np.linalg.solve(A, Ztr.T@ytr)
    return Zte@w

def cv(m, cols, alphas, eval_days=(347,375,403,431), **kw):
    out={}
    for a in alphas:
        per=[]
        for D in eval_days:
            tr=m[m.snapshot_day<D]; te=m[m.snapshot_day==D]
            pr=fit_predict(tr,te,cols,a,**kw)
            per.append(np.mean(np.abs(pr-te.future_spend_4w.to_numpy(float))))
        out[a]=round(float(np.mean(per)),3)
    return out

alphas=[0,0.3,1,3,10,30,100,300,1000]
print('E013 baseline alpha curve:', cv(m, feat, alphas))

# bias structure at alpha=3
tr=m[m.snapshot_day<431]; te=m[m.snapshot_day==431]
pr=fit_predict(tr,te,feat,3)
y=te.future_spend_4w.to_numpy(float)
qc=pd.qcut(pr,10,duplicates='drop')
g=pd.DataFrame({'p':pr,'y':y,'q':qc}).groupby('q',observed=True).agg(pmean=('p','mean'),ymean=('y','mean'),mae=('y',lambda s:s))
g['mae']=np.abs(pd.DataFrame({'p':pr,'y':y}).groupby(pd.qcut(pr,10,duplicates='drop').codes).apply(lambda d: np.mean(np.abs(d.p-d.y))))
print(g.round(1))
print('mean pred', pr.mean().round(1), 'mean y', y.mean().round(1), 'median pred', np.median(pr).round(1), 'median y', np.median(y).round(1))