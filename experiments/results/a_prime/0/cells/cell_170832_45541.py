import numpy as np, pandas as pd
from collections import defaultdict
t = agent_api.load_saved('e013_stock.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
days = agent_api.snapshot_days()
trm = df.snapshot_day.isin(days['train']).values
print('rows', len(df), 'train', trm.sum(), 'val', (~trm).sum(), 'missing target in train', df.loc[trm,'future_spend_4w'].isna().sum())
y = df['future_spend_4w'].values
feat = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
num = df[feat].select_dtypes(include=[np.number,'bool']).columns.tolist()
cat = [c for c in feat if c not in num]
Xn = df[num].astype(float).copy()
Xn = Xn.fillna(Xn[trm].median()).fillna(0)
Xparts=[Xn]; group=defaultdict(list)
for c in num: group[c.split('_')[0]].append(c)
for c in cat:
    p=c.split('_')[0]
    d=pd.get_dummies(df[c].astype('category'), prefix=p, dummy_na=True).astype(float)
    Xparts.append(d); group[p].extend(d.columns)
X=pd.concat(Xparts,axis=1)
mu=X[trm].mean(); sd=X[trm].std().replace(0,1.0)
Xs=(X-mu)/sd
Xtr,Xva=Xs[trm].values,Xs[~trm].values
ytr,yva=y[trm],y[~trm]
print('Xtr',Xtr.shape,'Xva',Xva.shape)
eye=np.eye(Xtr.shape[1])
def fit(a,Xtr_=Xtr,Xva_=Xva):
    w=np.linalg.solve(Xtr_.T@Xtr_+a*eye, Xtr_.T@ytr)
    return np.abs(Xtr_@w-ytr).mean(), np.abs(Xva_@w-yva).mean(), w
best=None
for a in [0.03,0.1,0.3,1,3,10,30,100,300]:
    mtr,mva,_=fit(a)
    print(f'alpha {a}: train {mtr:.2f} val {mva:.3f}')
    if best is None or mva<best[1]: best=(a,mva)
a_best=best[0]; print('best alpha',a_best,best[1])
res=[]
for p,cols in group.items():
    idx=[Xs.columns.get_loc(c) for c in cols if c in Xs.columns]
    Xz_tr=Xtr.copy(); Xz_va=Xva.copy()
    Xz_tr[:,idx]=0.0; Xz_va[:,idx]=0.0
    _,mva,_=fit(a_best,Xz_tr,Xz_va)
    res.append((mva-best[1],p,len(cols)))
res.sort(reverse=True)
print('--- block ablation (delta val MAE when zeroed; positive=block helps) ---')
for d,p,n in res: print(f'{p:22s} n={n:3d} delta={d:+.3f}')
_,_,w=fit(a_best)
cols=list(Xs.columns)
wi=np.argsort(-np.abs(w))[:25]
print('--- top |coef| ---')
for i in wi: print(f'{cols[i]:32s} {w[i]:+.3f}')
