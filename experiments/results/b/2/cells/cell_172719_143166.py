import pandas as pd, numpy as np, agent_api

df = agent_api.load_saved('e013_denoise.parquet')
tt = agent_api.train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
feat = [c for c in m.columns if c not in ('household_key','snapshot_day','future_spend_4w')]

def prep(tr, te, cols, alpha, logskew=False, clip_q=None):
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
    w = np.linalg.pinv(A) @ (Ztr.T@ytr)
    return Zte@w

def cv(m, cols, alphas, eval_days=(347,375,403,431), **kw):
    out={}
    for a in alphas:
        per=[]
        for D in eval_days:
            tr=m[m.snapshot_day<D]; te=m[m.snapshot_day==D]
            pr=prep(tr,te,cols,a,**kw)
            per.append(np.mean(np.abs(pr-te.future_spend_4w.to_numpy(float))))
        out[a]=round(float(np.mean(per)),3)
    return out

alphas=[1000,3000,10000,30000,100000]
print('E013 high alpha:', cv(m, feat, alphas))

# decile bias at alpha=1000
tr=m[m.snapshot_day<431]; te=m[m.snapshot_day==431]
pr=prep(tr,te,feat,1000); y=te.future_spend_4w.to_numpy(float)
qs=np.quantile(pr,np.linspace(0,1,11))
idx=np.clip(np.digitize(pr,qs[1:-1]),0,9)
for i in range(10):
    s=idx==i
    print(f'dec{i}: n={s.sum():5d} pred={pr[s].mean():7.1f} y={y[s].mean():7.1f} mae={np.abs(pr[s]-y[s]).mean():6.1f}')
print('overall pred mean', pr.mean().round(1), 'y mean', y.mean().round(1))

# prune dups/consts
sub = m[m.snapshot_day<347][feat]
nun = sub.nunique(); const = nun[nun<=1].index.tolist()
corr = sub.corr().abs().fillna(0)
drop=set(const)
for i,a in enumerate(feat):
    if a in drop: continue
    for b in feat[i+1:]:
        if b in drop: continue
        if corr.loc[a,b]>0.999: drop.add(b)
feat2=[c for c in feat if c not in drop]
print('pruned to', len(feat2))
print('pruned alpha curve:', cv(m, feat2, [100,1000,3000,10000]))
print('pruned+clip:', cv(m, feat2, [1000,3000], clip_q=0.01))
print('pruned+logskew:', cv(m, feat2, [1000,3000], logskew=True))