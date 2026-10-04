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
    Xs=((X-mu)/sd).replace([np.inf,-np.inf],0).fillna(0).values
    y=df['future_spend_4w'].values
    sd_days=sorted(df.snapshot_day[trm].unique())
    fold=np.zeros(len(y),int)
    for i,d in enumerate(sd_days): fold[df.snapshot_day.values==d]=i%5
    return Xs,y,trm,fold[trm],df,X.columns.tolist()

Xs,y,trm,fold_tr,df,cols=prep('e013_stock.parquet')
Xtr,ytr=Xs[trm],y[trm]
yt=np.sqrt(ytr)
eye=np.eye(Xtr.shape[1])
def cv_for(a,Xm,ym,fold):
    errs=[]
    for k in range(5):
        m=fold!=k
        w=np.linalg.solve(Xm[m].T@Xm[m]+a*eye, Xm[m].T@ym[m])
        p=np.square(np.clip(Xm[~m]@w,0,None))
        errs.append(np.abs(p-y[trm][~m]).mean())
    return np.mean(errs)
best=None
for a in [0.3,1,3,10,30,100,300,1000]:
    c=cv_for(a,Xtr,yt,fold_tr)
    if best is None or c<best[1]: best=(a,c)
a=best[0]; print('best alpha',a,'CV',round(best[1],3))
w=np.linalg.solve(Xtr.T@Xtr+a*eye, Xtr.T@yt)
imp=np.abs(w)
order=np.argsort(-imp)
print('--- top 60 features by |coef| (sqrt-target ridge) ---')
for i in order[:60]: print(f'{cols[i]:34s} {imp[i]:.4f}')
# correlation clusters on train
Xdf=pd.DataFrame(Xtr,columns=cols)
C=Xdf.corr().abs().values
np.fill_diagonal(C,0)
clusters=[]; assigned=set()
for i in order:
    if i in assigned: continue
    grp=[j for j in range(len(cols)) if C[i,j]>0.90]
    clusters.append((cols[i],[cols[j] for j in grp]))
    assigned.add(i); assigned.update(grp)
print('n clusters (|r|>0.9):',len(clusters),'from',len(cols),'features')
print('--- largest clusters ---')
for rep,g in sorted(clusters,key=lambda z:-len(z[1]))[:15]:
    print(f'{rep:30s} <- {g[:8]}{"..." if len(g)>8 else ""} (n={len(g)})')
