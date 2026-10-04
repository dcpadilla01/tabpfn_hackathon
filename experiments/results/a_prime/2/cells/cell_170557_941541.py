import agent_api, numpy as np, pandas as pd

t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys = ['household_key','snapshot_day']
feat_cols = [c for c in t.columns if c not in keys + ['index']]
tt = agent_api.train_targets()
m = t.merge(tt, on=keys, how='inner')
y = m['future_spend_4w'].astype(float).values
folds = m['snapshot_day'].values.astype(int)

Xdf = m[feat_cols].copy()
nonnum = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(Xdf[c])]
for c in nonnum:
    Xdf[c] = pd.to_numeric(Xdf[c], errors='coerce')
X = Xdf.values.astype(float)
keep = [j for j in range(len(feat_cols)) if not np.isnan(X[:,j]).all()]
feat_cols2 = [feat_cols[j] for j in keep]
X = X[:, keep]
cm = np.nanmean(X, axis=0); mask = np.isnan(X)
X[mask] = np.take(cm, np.where(mask)[1])
sd = X.std(0); bad = sd <= 1e-12
Xz = (X - cm)/np.where(bad,1,sd)
corr = np.nan_to_num(np.array([np.corrcoef(Xz[:,j], y)[0,1] if not bad[j] else 0.0 for j in range(len(feat_cols2))]))
order = np.argsort(-np.abs(corr))

def fit_eval(train_mask, val_mask, idx, alpha, logt=False, clip=True):
    Xtr, ytr = Xz[train_mask][:, idx], y[train_mask]
    Xva, yva = Xz[val_mask][:, idx], y[val_mask]
    yt = np.log1p(ytr) if logt else ytr
    A = Xtr.T@Xtr + alpha*np.eye(len(idx))
    b = Xtr.T@yt
    beta = np.linalg.solve(A, b)
    p = Xva@beta
    if logt: p = np.expm1(p)
    if clip: p = np.clip(p, 0, None)
    return np.abs(p - yva).mean()

# time-aware: train on snapshots <= 375, validate on 403 & 431
trm = folds <= 375
print('per-day val MAE (full feats, ridge a=1000, raw):')
for d in [375, 403, 431]:
    vm = folds == d
    print(' day', d, 'n=%d ymean=%.1f MAE %.2f' % (vm.sum(), y[vm].mean(), fit_eval(trm, vm, list(range(len(feat_cols2))), 1000)))

vam = (folds == 403) | (folds == 431)
full = list(range(len(feat_cols2)))
print('\nTIME-AWARE proxy (train <=375, val {403,431}):')
print('%-28s %s' % ('variant','MAE'))
for a in [30, 100, 300, 1000, 3000]:
    print('ridge raw a=%-5d all      %.3f' % (a, fit_eval(trm, vam, full, a)))
for a in [30, 100, 300, 1000, 3000]:
    print('ridge log a=%-5d all      %.3f' % (a, fit_eval(trm, vam, full, a, logt=True)))
print('mean-baseline            %.3f' % np.abs(y[vam] - y[trm].mean()).mean())

print('\ntop-k |corr| raw a=1000:')
for k in [20,40,60,80,120,160,200,240,280,360]:
    print('k=%3d -> %.3f' % (k, fit_eval(trm, vam, order[:k], 1000)))
print('\ntop-k |corr| log a=1000:')
for k in [20,40,60,80,120,160,200,240,280,360]:
    print('k=%3d -> %.3f' % (k, fit_eval(trm, vam, order[:k], 1000, logt=True)))