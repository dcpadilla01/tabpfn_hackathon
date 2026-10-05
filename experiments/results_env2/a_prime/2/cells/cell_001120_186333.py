
import numpy as np, pandas as pd
out = load_saved('e015_full_superset.parquet')
print('dupes:', out.columns[out.columns.duplicated()].tolist())
print('shape', out.shape)
print('has index col:', 'index' in out.columns)
# quick pseudo with dedup
out = out.loc[:, ~out.columns.duplicated()]
tt = train_targets()
df = out.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in out.columns if c not in ('household_key','snapshot_day')]
def prep(d, fs):
    X = d[fs].copy()
    for c in fs:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype('category').cat.codes.astype(float); X[c] = X[c].where(X[c]>=0, np.nan)
    return X.astype(float)
def pseudo(fs, lam, fit_max, test_day):
    fit_days = [d for d in snapshot_days()['train'] if d <= fit_max]
    tr = df[df.snapshot_day.isin(fit_days)]; te = df[df.snapshot_day==test_day]
    Xtr, Xte = prep(tr, fs), prep(te, fs)
    mu, sg = Xtr.mean(), Xtr.std().replace(0,1)+1e-9
    Xtr = ((Xtr-mu)/sg).fillna(0.).values; Xte = ((Xte-mu)/sg).fillna(0.).values
    ytr, yte = tr.future_spend_4w.values, te.future_spend_4w.values
    A_ = np.hstack([Xtr, np.ones((len(Xtr),1))]); Av = np.hstack([Xte, np.ones((len(Xte),1))])
    D = np.eye(A_.shape[1]); D[-1,-1]=0
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr)
    return np.abs(Av@w-yte).mean()
for fit_max, tests in [(347,[403,431]), (375,[403,431]), (403,[431])]:
    print('fit<=%d superset: %s' % (fit_max, '  '.join('%d:%.2f'%(t,pseudo(feats,30,fit_max,t)) for t in tests)))
