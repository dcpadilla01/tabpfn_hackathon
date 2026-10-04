import agent_api, numpy as np, pandas as pd

t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys = ['household_key','snapshot_day']
feat_cols = [c for c in t.columns if c not in keys + ['index']]
tt = agent_api.train_targets()
m = t.merge(tt, on=keys, how='inner')
y = m['future_spend_4w'].astype(float).values

Xdf = m[feat_cols].copy()
nonnum = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(Xdf[c])]
for c in nonnum:
    Xdf[c] = pd.to_numeric(Xdf[c], errors='coerce')
X = Xdf.values.astype(float)
keep = [j for j in range(len(feat_cols)) if not np.isnan(X[:,j]).all()]
feat_cols2 = [feat_cols[j] for j in keep]
X = X[:, keep]
mask = np.isnan(X)
cm = np.nanmean(X, axis=0)
X[mask] = np.take(cm, np.where(mask)[1])
sd = X.std(0); bad = sd <= 1e-12
Xz = (X - cm)/np.where(bad, 1, sd)

corr = np.nan_to_num(np.array([np.corrcoef(Xz[:,j], y)[0,1] if not bad[j] else 0.0 for j in range(len(feat_cols2))]))
order = np.argsort(-np.abs(corr))

folds = m['snapshot_day'].values.astype(int)
days = sorted(set(folds))
G={};B={};Xv={};yv={}
for d in days:
    tr = folds != d
    G[d]=Xz[tr].T@Xz[tr]; B[d]=Xz[tr].T@y[tr]; Xv[d]=Xz[~tr]; yv[d]=y[~tr]

def loso(idx, alpha, clip=True, ridge=True):
    idx=list(idx); errs=[]
    for d in days:
        A=G[d][np.ix_(idx,idx)]
        if ridge: A=A+alpha*np.eye(len(idx))
        beta=np.linalg.solve(A,B[d][idx])
        p=Xv[d][:,idx]@beta
        if clip: p=np.maximum(p,0.0)
        errs.append(np.abs(p-yv[d]).mean())
    return float(np.mean(errs))

full=list(range(len(feat_cols2)))
print('ridge LOSO MAE (all %d feats):'%len(feat_cols2))
best_a,best_m=None,1e9
for a in [3,10,30,100,300,1000,3000]:
    mm=loso(full,a)
    print('alpha %5d -> %.3f'%(a,mm))
    if mm<best_m: best_m,best_a=mm,a
print('best alpha',best_a,round(best_m,3))

print('\nLOSO MAE top-k |corr| (alpha=%d):'%best_a)
for k in [10,20,40,60,80,120,160,200,240,280,320,len(feat_cols2)]:
    print('k=%3d -> %.3f'%(k,loso(order[:k],best_a)))

print('\nLOSO MAE excluding bottom-k |corr| (alpha=%d):'%best_a)
for k in [0,20,40,60,80,100,140,180]:
    drop=set(order[-k:].tolist()) if k>0 else set()
    idx=[j for j in full if j not in drop]
    print('drop worst %3d -> n=%3d MAE %.3f'%(k,len(idx),loso(idx,best_a)))