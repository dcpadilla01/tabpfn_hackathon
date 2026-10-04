
import agent_api, numpy as np, pandas as pd

t = agent_api.load_saved('e009_ewma_longlags.parquet')
print('saved shape', t.shape)
keys = ['household_key','snapshot_day']
feat_cols = [c for c in t.columns if c not in keys + ['index']]
print('n feature cols', len(feat_cols))
print('dtypes', dict(t.dtypes.value_counts()))
nonnum = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(t[c])]
print('nonnum count', len(nonnum), nonnum[:25])

tt = agent_api.train_targets()
print('targets', tt.shape)
print('y stats', {k: round(float(v),2) for k,v in tt['future_spend_4w'].describe().items()})
m = t.merge(tt, on=keys, how='inner')
print('merged', m.shape)
y = m['future_spend_4w'].astype(float).values
print('y mean %.2f median %.2f zero%% %.3f' % (y.mean(), np.median(y), (y==0).mean()))

Xdf = m[feat_cols].copy()
for c in nonnum:
    Xdf[c] = pd.to_numeric(Xdf[c], errors='coerce')
X = Xdf.values.astype(float)
mask = np.isnan(X)
cm = np.nanmean(X, axis=0)
X[mask] = np.take(cm, np.where(mask)[1])
sd = X.std(0); sd[sd==0] = 1
Xz = (X - cm)/sd

corr = np.zeros(len(feat_cols))
for j in range(len(feat_cols)):
    if sd[j] > 0:
        c = np.corrcoef(Xz[:,j], y)[0,1]
        corr[j] = 0 if np.isnan(c) else c
order = np.argsort(-np.abs(corr))
print('\ntop 45 by |corr with target| (train rows):')
for j in order[:45]:
    print('%-38s % .3f' % (feat_cols[j], corr[j]))

folds = m['snapshot_day'].values
days = sorted(set(folds))
print('\ntrain snapshot days:', days)
G = {}; B = {}; Xv = {}; yv = {}; trmask = {}
for d in days:
    tr = folds != d
    G[d] = Xz[tr].T @ Xz[tr]
    B[d] = Xz[tr].T @ y[tr]
    Xv[d] = Xz[~tr]; yv[d] = y[~tr]

def loso(idx, alpha, clip=True):
    idx = list(idx); errs = []
    for d in days:
        A = G[d][np.ix_(idx, idx)] + alpha*np.eye(len(idx))
        beta = np.linalg.solve(A, B[d][idx])
        p = Xv[d][:, idx] @ beta
        if clip: p = np.maximum(p, 0.0)
        errs.append(np.abs(p - yv[d]).mean())
    return float(np.mean(errs))

full = list(range(len(feat_cols)))
print('\nridge LOSO MAE by alpha (all %d feats):' % len(feat_cols))
best_a, best_m = None, 1e9
for a in [1, 3, 10, 30, 100, 300, 1000]:
    mm = loso(full, a)
    print('alpha %5d -> %.3f' % (a, mm))
    if mm < best_m: best_m, best_a = mm, a
print('best alpha', best_a, round(best_m,3))
print('mean-baseline LOSO MAE %.3f' % np.mean([np.abs(yv[d] - y[folds!=d].mean()).mean() for d in days]))

print('\nLOSO MAE by top-k |corr| features (alpha=%d):' % best_a)
for k in [5,10,20,40,60,80,120,180,250,320,len(feat_cols)]:
    print('k=%3d -> %.3f' % (k, loso(order[:k], best_a)))
