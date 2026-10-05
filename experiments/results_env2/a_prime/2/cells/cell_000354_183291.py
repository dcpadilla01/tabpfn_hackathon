
import numpy as np, pandas as pd

tt = train_targets()
feats_all = [c for c in load_saved('e011_discounts.parquet').columns if c not in ('household_key','snapshot_day')]

def prep(d, feats):
    X = d[feats].copy()
    for c in feats:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype('category').cat.codes.astype(float)
            X[c] = X[c].where(X[c] >= 0, np.nan)
    return X.astype(float)

def pseudo_eval(df, feats, lam=30, fit_days=None, test_day=431):
    if fit_days is None:
        fit_days = [d for d in snapshot_days()['train'] if d < test_day]
    tr = df[df.snapshot_day.isin(fit_days)]; te = df[df.snapshot_day == test_day]
    Xtr, Xte = prep(tr, feats), prep(te, feats)
    mu, sg = Xtr.mean(), Xtr.std().replace(0,1)+1e-9
    Xtr = ((Xtr-mu)/sg).fillna(0.0).values; Xte = ((Xte-mu)/sg).fillna(0.0).values
    ytr, yte = tr.future_spend_4w.values, te.future_spend_4w.values
    A = np.hstack([Xtr, np.ones((len(Xtr),1))]); Av = np.hstack([Xte, np.ones((len(Xte),1))])
    D = np.eye(A.shape[1]); D[-1,-1]=0
    w = np.linalg.solve(A.T@A + lam*D, A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(Av@w-yte).mean(), len(te)

base = load_saved('e011_discounts.parquet').merge(tt, on=['household_key','snapshot_day'])
# sanity: pseudo-val on 431 for E011 feats; also test 403 as target
for lam in [10, 30, 100, 300]:
    trm, tem, n = pseudo_eval(base, feats_all, lam=lam)
    print('lam', lam, 'train MAE', round(trm,3), 'pseudo-val(431) MAE', round(tem,3), 'n', n)
# also pseudo-val at 403 (fit 95..375)
trm, tem, n = pseudo_eval(base, feats_all, lam=30, test_day=403)
print('test403: train MAE', round(trm,3), 'pseudo MAE', round(tem,3), 'n', n)
