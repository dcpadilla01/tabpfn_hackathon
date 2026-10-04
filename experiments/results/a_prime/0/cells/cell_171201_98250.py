import numpy as np, pandas as pd

def prep(path):
    t = agent_api.load_saved(path)
    tt = agent_api.train_targets()
    df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
    days = agent_api.snapshot_days()
    trm = df.snapshot_day.isin(days['train']).values
    feat=[c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    seen=set(); fd=[]
    for c in feat:
        if c not in seen: seen.add(c); fd.append(c)
    num=df[fd].select_dtypes(include=[np.number,'bool']).columns.tolist()
    cat=[c for c in fd if c not in num]
    Xn=df[num].astype(float); Xn=Xn.replace([np.inf,-np.inf],np.nan)
    Xn=Xn.fillna(Xn[trm].median()).fillna(0)
    parts=[Xn]
    for c in cat:
        parts.append(pd.get_dummies(df[c].astype('category'), prefix=c, dummy_na=True).astype(float))
    X=pd.concat(parts,axis=1)
    mu=X[trm].mean(); sd=X[trm].std().replace(0,1.0)
    Xs=((X-mu)/sd).fillna(0).values
    y=df['future_spend_4w'].values
    sd_days=sorted(df.snapshot_day[trm].unique())
    fold=np.zeros(len(y),int)
    for i,d in enumerate(sd_days): fold[df.snapshot_day.values==d]=i%5
    return Xs,y,trm,fold[trm],df

def ridge_cv(Xs,y,trm,fold_tr,tfwd,tinv,alphas):
    Xtr,ytr=Xs[trm],tfwd(y[trm])
    eye=np.eye(Xtr.shape[1])
    out=[]
    for a in alphas:
        errs=[]
        for k in range(5):
            m=fold_tr!=k
            w=np.linalg.solve(Xtr[m].T@Xtr[m]+a*eye, Xtr[m].T@ytr[m])
            p=tinv(Xtr[~m]@w)
            errs.append(np.abs(p-y[trm][~m]).mean())
        out.append((np.mean(errs),a))
    out.sort()
    return out[0]

Xs,y,trm,fold_tr,df=prep('e013_stock.parquet')
print('y: mean %.1f med %.1f p90 %.1f p99 %.1f max %.1f zero-frac %.3f'%(y[trm].mean(),np.median(y[trm]),np.quantile(y[trm],.9),np.quantile(y[trm],.99),y[trm].max(),(y[trm]==0).mean()))
for name,tf,tinv in [('identity',lambda v:v,lambda v:v),
                     ('log1p',np.log1p,lambda v:np.expm1(np.clip(v,0,12))),
                     ('sqrt',np.sqrt,lambda v:np.square(np.clip(v,0,None)))]:
    m,a=ridge_cv(Xs,y,trm,fold_tr,tf,tinv,[1,3,10,30,100,300,1000,3000])
    print(f'{name:9s} best CV-MAE {m:.3f} at alpha {a}')
