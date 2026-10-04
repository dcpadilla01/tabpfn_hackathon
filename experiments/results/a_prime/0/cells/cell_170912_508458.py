import numpy as np, pandas as pd
from collections import defaultdict
t = agent_api.load_saved('e013_stock.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
days = agent_api.snapshot_days()
trm = df.snapshot_day.isin(days['train']).values
y = df['future_spend_4w'].values
feat = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
num = df[feat].select_dtypes(include=[np.number,'bool']).columns.tolist()
cat = [c for c in feat if c not in num]
Xn = df[num].astype(float).copy()
Xn = Xn.fillna(Xn[trm].median()).fillna(0)
parts=[('num',Xn)]; group=defaultdict(list)
for c in num: group['num'].append(c)
for c in cat:
    p=c.split('_')[0]
    d=pd.get_dummies(df[c].astype('category'), prefix=p+'__', dummy_na=True).astype(float)
    parts.append((p,d))
    group[p].extend(d.columns)
X=pd.concat([d for _,d in parts],axis=1)
assert not X.columns.duplicated().any()
mu=X[trm].mean(); sd=X[trm].std().replace(0,1.0)
Xs=((X-mu)/sd).values
Xtr,Xva=Xs[trm],Xs[~trm]
ytr=y[trm]
print('Xtr',Xtr.shape,'Xva',Xva.shape)
eye=np.eye(Xtr.shape[1])
pos={c:i for i,c in enumerate(X.columns)}
def fitw(a,Xm,ym):
    return np.linalg.solve(Xm.T@Xm+a*eye, Xm.T@ym)
# 5-fold CV on train for alpha
rng=np.random.RandomState(0); fold=rng.permutation(len(ytr))%5
best=None
for a in [0.1,0.3,1,3,10,30,100,300]:
    errs=[]
    for k in range(5):
        m=fold!=k
        w=fitw(a,Xtr[m],ytr[m])
        errs.append(np.abs(Xtr[~m]@w-ytr[~m]).mean())
    cv=np.mean(errs)
    w=fitw(a,Xtr,ytr)
    print(f'alpha {a}: cv {cv:.3f} trainMAE {np.abs(Xtr@w-ytr).mean():.2f}')
    if best is None or cv<best[1]: best=(a,cv)
a_best=best[0]; print('best alpha',a_best,best[1])
w=fitw(a_best,Xtr,ytr)
res=[]
for p,cols in group.items():
    idx=[pos[c] for c in cols]
    Xz=Xtr.copy(); Xz[:,idx]=0.0
    wz=fitw(a_best,Xz,ytr)
    # CV-MAE with block zeroed
    errs=[]
    for k in range(5):
        m=fold!=k
        wz2=fitw(a_best,Xz[m],ytr[m])
        errs.append(np.abs(Xz[~m]@wz2-ytr[~m]).mean())
    res.append((np.mean(errs)-best[1],p,len(cols)))
res.sort(reverse=True)
print('--- block ablation (delta CV-MAE when zeroed; positive=block helps) ---')
for d,p,n in res: print(f'{p:22s} n={n:3d} delta={d:+.3f}')
cols=list(X.columns)
wi=np.argsort(-np.abs(w))[:25]
print('--- top |coef| ---')
for i in wi: print(f'{cols[i]:34s} {w[i]:+.3f}')
