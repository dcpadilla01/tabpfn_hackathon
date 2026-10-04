
import agent_api as A, numpy as np, pandas as pd

e13 = A.load_saved('e013_denoise.parquet')
st  = A.load_saved('e014_stack.parquet')[['household_key','snapshot_day','gbm_pred']]
tt  = A.train_targets()
df = e13.merge(st, on=['household_key','snapshot_day'], how='left').merge(tt, on=['household_key','snapshot_day'], how='inner')
tr = df[df.snapshot_day.isin(A.snapshot_days()['train'])].copy()
y = tr.future_spend_4w.values

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
    idx = rng.permutation(len(y)); fs = np.array_split(idx, folds)
    maes=[]
    for f in range(folds):
        b = fs[f]; a = np.concatenate([fs[j] for j in range(folds) if j!=f])
        Xa = np.hstack([Xv[a], np.ones((len(a),1))]); Xb = np.hstack([Xv[b], np.ones((len(b),1))])
        M = Xa.T@Xa + alpha*np.eye(Xa.shape[1]); M[-1,-1] -= alpha
        w = np.linalg.solve(M, Xa.T@y[a])
        maes.append(np.abs(Xb@w - y[b]).mean())
    return np.mean(maes)

num = [c for c in e13.columns if c not in ('household_key','snapshot_day')]
tr['gbm_log'] = np.log1p(tr.gbm_pred.clip(lower=0))
tr['gbm_sqrt'] = np.sqrt(tr.gbm_pred.clip(lower=0))
for cols, name in [
    (['gbm_pred'], 'gbm_pred alone'),
    (num+['gbm_log'], 'E013+gbm_log'),
    (num+['gbm_sqrt'], 'E013+gbm_sqrt'),
    (num+['gbm_pred','gbm_log'], 'E013+gbm_pred+gbm_log'),
]:
    print(f'{name}: {ridge_cv(design(tr, cols), y):.3f}')
