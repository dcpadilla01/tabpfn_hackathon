import numpy as np, pandas as pd
from collections import defaultdict
t = agent_api.load_saved('e013_stock.parquet')
print('e013 shape', t.shape)
print(t.dtypes.value_counts())
print('sample cols:', list(t.columns)[:50])
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
days = agent_api.snapshot_days()
trm = df.snapshot_day.isin(days['train']).values
y = df['future_spend_4w'].values
feat = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
num = df[feat].select_dtypes(include=[np.number,'bool']).columns.tolist()
cat = [c for c in feat if c not in num]
print('n num', len(num), 'n cat', len(cat), cat)
Xn = df[num].astype(float).copy()
med = Xn[trm].median()
Xn = Xn.fillna(med).fillna(0)
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
    Xz_tr=Xtr.copy(); Xz_va=Xva.copy()
    idx=[Xs.columns.get_loc(c) for c in cols]
    Xz_tr[:,idx]=0; Xz_va[:,idx]=0
    _,mva,_=fit(a_best,Xz_tr,Xz_va)
    res.append((mva-best[1],p,len(cols)))
res.sort(reverse=True)
print('--- block ablation (delta val MAE when zeroed; positive=block helps) ---')
for d,p,n in res: print(f'{p:22s} n={n:3d} delta={d:+.3f}')
_,_,w=fit(a_best)
wi=np.argsort(-np.abs(w))[:25]
cols=list(Xs.columns)
print('--- top |coef| ---')
for i in wi: print(f'{cols[i]:32s} {w[i]:+.3f}')


# ---- cell ----
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


# ---- cell ----
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


# ---- cell ----
import numpy as np, pandas as pd
from collections import defaultdict
t = agent_api.load_saved('e013_stock.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
days = agent_api.snapshot_days()
trm = df.snapshot_day.isin(days['train']).values
y = df['future_spend_4w'].values
feat = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
print('dup feats:', [c for c in feat if feat.count(c)>1][:20])
# dedupe: keep first occurrence
seen=set(); feat_d=[]
for c in feat:
    if c not in seen: seen.add(c); feat_d.append(c)
num = df[feat_d].select_dtypes(include=[np.number,'bool']).columns.tolist()
cat = [c for c in feat_d if c not in num]
Xn = df[num].astype(float).copy()
Xn = Xn.fillna(Xn[trm].median()).fillna(0)
parts=[Xn]; group=defaultdict(list)
for c in num: group['num'].append(c)
for c in cat:
    p=c.split('_')[0]
    d=pd.get_dummies(df[c].astype('category'), prefix=p+'__', dummy_na=True).astype(float)
    parts.append(d); group[p].extend(d.columns)
X=pd.concat(parts,axis=1)
print('X cols', X.shape, 'dups', X.columns.duplicated().sum())
mu=X[trm].mean(); sd=X[trm].std().replace(0,1.0)
Xs=((X-mu)/sd).values
Xtr,Xva=Xs[trm],Xs[~trm]
ytr=y[trm]
eye=np.eye(Xtr.shape[1])
pos={c:i for i,c in enumerate(X.columns)}
def fitw(a,Xm,ym):
    return np.linalg.solve(Xm.T@Xm+a*eye, Xm.T@ym)
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
res=[]
for p,cols in group.items():
    idx=[pos[c] for c in cols]
    Xz=Xtr.copy(); Xz[:,idx]=0.0
    errs=[]
    for k in range(5):
        m=fold!=k
        wz=fitw(a_best,Xz[m],ytr[m])
        errs.append(np.abs(Xz[~m]@wz-ytr[~m]).mean())
    res.append((np.mean(errs)-best[1],p,len(cols)))
res.sort(reverse=True)
print('--- block ablation (delta CV-MAE when zeroed; positive=block helps) ---')
for d,p,n in res: print(f'{p:22s} n={n:3d} delta={d:+.3f}')
w=fitw(a_best,Xtr,ytr)
cols=list(X.columns)
wi=np.argsort(-np.abs(w))[:25]
print('--- top |coef| ---')
for i in wi: print(f'{cols[i]:34s} {w[i]:+.3f}')


# ---- cell ----
import numpy as np, pandas as pd
from collections import defaultdict

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
    return Xs,y,trm,fold,df

def ridge_cv(Xs,y,fold,trm,tfwd,tinv,alphas):
    Xtr,ytr=Xs[trm],tfwd(y[trm])
    eye=np.eye(Xtr.shape[1])
    out=[]
    for a in alphas:
        errs=[]
        for k in range(5):
            m=fold!=k
            w=np.linalg.solve(Xtr[m].T@Xtr[m]+a*eye, Xtr[m].T@ytr[m])
            p=tinv(Xtr[~m]@w)
            errs.append(np.abs(p-y[~m]).mean())
        out.append((np.mean(errs),a))
    out.sort()
    return out[0]

t_all=prep('e013_stock.parquet')
Xs,y,trm,fold,df=t_all
print('y stats: mean %.1f med %.1f p90 %.1f p99 %.1f max %.1f zero-frac %.3f'%(y[trm].mean(),np.median(y[trm]),np.quantile(y[trm],.9),np.quantile(y[trm],.99),y[trm].max(),(y[trm]==0).mean()))
for name,tf,tinv in [('identity',lambda v:v,lambda v:v),
                     ('log1p',np.log1p,lambda v:np.expm1(np.clip(v,0,12))),
                     ('sqrt',np.sqrt,lambda v:np.square(np.clip(v,0,None)))]:
    m,a=ridge_cv(Xs,y,fold,trm,tf,tinv,[1,3,10,30,100,300,1000,3000])
    print(f'{name:9s} best CV-MAE {m:.3f} at alpha {a}')


# ---- cell ----
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


# ---- cell ----
import numpy as np, pandas as pd
tt = agent_api.train_targets()
print('mean/median y by snapshot:'); print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']).round(1))
for p in ['stock_v1','rhythm_v1','mkt_v2','rank_block' ]:
    try:
        b=agent_api.load_saved(p+'.parquet'); print(p, b.shape, list(b.columns)[:40])
    except Exception as e: print(p,'ERR',e)
t=agent_api.load_saved('e013_stock.parquet')
cols=[c for c in t.columns if c not in ('household_key','snapshot_day')]
print('e013 n cols',len(cols))
print([c for c in cols if c.startswith('m_') or c.startswith('rk_') or c.startswith('stk_') or c.startswith('rs_') or c.startswith('rz_')])


# ---- cell ----
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


# ---- cell ----
import numpy as np, pandas as pd
t=agent_api.load_saved('e013_stock.parquet')
tt=agent_api.train_targets()
df=t.merge(tt,on=['household_key','snapshot_day'],how='left')
days=agent_api.snapshot_days(); trm=df.snapshot_day.isin(days['train']).values
# 1) m_spend28 vs spend_28
sub=df.loc[trm,['m_spend28','spend_28','future_spend_4w']]
print('corr(m_spend28,spend_28)', sub.m_spend28.corr(sub.spend_28).round(4))
d=(sub.m_spend28-sub.spend_28)
print('diff describe:', d.describe().round(3).to_dict())
print('frac equal:', (d.abs()<1e-9).mean().round(3))
# quantiles of each
for c in ['m_spend28','spend_28']:
    print(c, np.quantile(sub[c],[.5,.9,.99,.999]).round(1), 'max', sub[c].max().round(1))
# corr with target
print('corr with y: m_spend28 %.3f spend_28 %.3f'%(sub.m_spend28.corr(sub.future_spend_4w), sub.spend_28.corr(sub.future_spend_4w)))
# spearman
print('spearman with y: m %.3f s %.3f'%(sub.m_spend28.corr(sub.future_spend_4w,method='spearman'), sub.spend_28.corr(sub.future_spend_4w,method='spearman')))


# ---- cell ----
import numpy as np, pandas as pd
t13=agent_api.load_saved('e013_stock.parquet')
tt=agent_api.train_targets()
df=t13.merge(tt,on=['household_key','snapshot_day'],how='left')
days=agent_api.snapshot_days(); trm=df.snapshot_day.isin(days['train']).values
na=df.m_spend28.isna()
print('m_spend28 NaN frac %.3f; spend_28==0 frac %.3f; overlap %.3f'%((na.mean()), (df.spend_28==0).mean(), (na&(df.spend_28==0)).mean()))
print('y mean where m_spend28 NaN: %.1f | not-NaN: %.1f'%(df.future_spend_4w[trm&na].mean(), df.future_spend_4w[trm&~na].mean()))
print('spend_28 stats where NaN:', df.spend_28[na].describe().round(1).to_dict())
feat=[c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
nafrac=df[feat].isna().mean().sort_values(ascending=False)
print('--- top NaN fractions ---'); print(nafrac[nafrac>0].head(20).round(3).to_dict())
pat=na.values
same=[c for c in feat if df[c].isna().values.tobytes()==pat.tobytes()]
print('cols with identical NaN pattern:', same[:20])


# ---- cell ----
import numpy as np, pandas as pd
t = agent_api.load_saved('e013_stock.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
days = agent_api.snapshot_days()
trm = df.snapshot_day.isin(days['train']).values
feat = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
num = df[feat].select_dtypes(include=[np.number,'bool']).columns.tolist()
cat = [c for c in feat if c not in num]
Xtr = df.loc[trm, num].astype(float).replace([np.inf,-np.inf], np.nan)
ytr = df.future_spend_4w[trm].values.astype(float)
Xf = Xtr.fillna(Xtr.median()).fillna(0)
var = Xf.var(); zv = var[var<=1e-12].index.tolist()
num2 = [c for c in num if c not in zv]
Xf = Xf[num2]
corry = {}
for c in num2:
    x = Xf[c].values
    r = np.corrcoef(x, ytr)[0,1]
    corry[c] = abs(r) if np.isfinite(r) else 0.0
order = sorted(num2, key=lambda c: -corry[c])
Cm = np.nan_to_num(np.corrcoef(Xf[order].values, rowvar=False))
kept=[]; dropped=[]
for i,c in enumerate(order):
    if any(abs(Cm[i,j])>0.95 for j in kept): dropped.append(c)
    else: kept.append(c)
print('kept %d | dropped %d | zero-var dropped %d'%(len(kept),len(dropped),len(zv)))
print('DROPPED:', dropped)
out = df[['household_key','snapshot_day'] + kept + cat].copy()
print('out shape', out.shape, '| dtypes', out.dtypes.value_counts().to_dict())
path = agent_api.save_table(out, 'pruned_v1.parquet')
print('saved:', path)


# ---- cell ----
import numpy as np, pandas as pd
t = agent_api.load_saved('e013_stock.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
days = agent_api.snapshot_days()
trm = df.snapshot_day.isin(days['train']).values
feat = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
num = df[feat].select_dtypes(include=[np.number,'bool']).columns.tolist()
cat = [c for c in feat if c not in num]
Xtr = df.loc[trm, num].astype(float).replace([np.inf,-np.inf], np.nan)
ytr = df.future_spend_4w[trm].values.astype(float)
Xf = Xtr.fillna(Xtr.median()).fillna(0)
var = Xf.var(); zv = var[var<=1e-12].index.tolist()
num2 = [c for c in num if c not in zv]
Xf = Xf[num2]
corry = {}
for c in num2:
    r = np.corrcoef(Xf[c].values, ytr)[0,1]
    corry[c] = abs(r) if np.isfinite(r) else 0.0
order = sorted(num2, key=lambda c: -corry[c])
Cm = np.corrcoef(Xf[order].values, rowvar=False)
Cm = np.nan_to_num(Cm)
pos = {c:i for i,c in enumerate(order)}
kept=[]; dropped=[]
for c in order:
    i = pos[c]
    if any(abs(Cm[i,pos[k]])>0.95 for k in kept): dropped.append(c)
    else: kept.append(c)
print('kept %d | dropped %d | zero-var dropped %d'%(len(kept),len(dropped),len(zv)))
print('DROPPED:', dropped)
out = df[['household_key','snapshot_day'] + kept + cat].copy()
print('out shape', out.shape, '| dtypes', out.dtypes.value_counts().to_dict())
path = agent_api.save_table(out, 'pruned_v1.parquet')
print('saved:', path)
