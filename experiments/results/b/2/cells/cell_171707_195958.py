
import agent_api as A, numpy as np, pandas as pd

e13 = A.load_saved('e013_denoise.parquet')
st  = A.load_saved('e014_stack.parquet')[['household_key','snapshot_day','gbm_pred']]
tt  = A.train_targets()
df = e13.merge(st, on=['household_key','snapshot_day'], how='left').merge(tt, on=['household_key','snapshot_day'], how='inner')
tr = df[df.snapshot_day.isin(A.snapshot_days()['train'])].copy()
print('train rows', tr.shape)

def design(frame, cols):
    X = frame[cols].copy()
    num = X.select_dtypes(include=[np.number]).astype(float)
    cat = X.select_dtypes(exclude=[np.number])
    parts = [num]
    if cat.shape[1]:
        parts.append(pd.get_dummies(cat.astype(str), dummy_na=True).astype(float))
    X = pd.concat(parts, axis=1)
    return X.replace([np.inf,-np.inf], np.nan).fillna(X.median())

def ridge_cv(X, y, alpha=100.0, folds=5, seed=0):
    Xv = X.values.astype(float)
    mu = np.nanmean(Xv,0); sd = Xv.std(0); sd[sd==0]=1
    Xv = (Xv-mu)/sd
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    fs = np.array_split(idx, folds)
    maes=[]
    for f in range(folds):
        b = fs[f]; a = np.concatenate([fs[j] for j in range(folds) if j!=f])
        Xa = np.hstack([Xv[a], np.ones((len(a),1))])
        Xb = np.hstack([Xv[b], np.ones((len(b),1))])
        A_ = Xa.T@Xa + alpha*np.eye(Xa.shape[1]); A_[-1,-1] -= alpha
        w = np.linalg.solve(A_, Xa.T@y[a])
        maes.append(np.abs(Xb@w - y[b]).mean())
    return np.mean(maes), np.std(maes)

num = [c for c in e13.columns if c not in ('household_key','snapshot_day')]
y = tr.future_spend_4w.values
for cols, name in [(num,'E013'), (num+['gbm_pred'],'E013+gbm_pred')]:
    X = design(tr, cols)
    m,s = ridge_cv(X, y)
    print(f'{name}: {m:.3f} +/- {s:.3f}')
tr['gbm_sq'] = tr.gbm_pred**2
X = design(tr, num+['gbm_pred','gbm_sq']); m,s = ridge_cv(X,y); print(f'E013+gbm+sq: {m:.3f} +/- {s:.3f}')
X = design(tr, num+['gbm_pred','gbm_sq','gbm_cu']); m,s = ridge_cv(X,y); print(f'E013+gbm+sq+cu: {m:.3f} +/- {s:.3f}') if False else None
tr['gbm_cu'] = tr.gbm_pred**3
X = design(tr, num+['gbm_pred','gbm_sq','gbm_cu']); m,s = ridge_cv(X,y); print(f'E013+gbm+sq+cu: {m:.3f} +/- {s:.3f}')
