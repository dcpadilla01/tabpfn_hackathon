
import numpy as np, pandas as pd

out = load_saved('e015_full_superset.parquet')
tt = train_targets()
df = out.merge(tt, on=['household_key','snapshot_day'])
base_feats = [c for c in out.columns if c not in ('household_key','snapshot_day')]  # = no_index base
DSL = ['dsl_le_%d'%k for k in [7,14,21,35,60,120]] + ['s28_x_dsl%d'%k for k in [7,14,21,35]] + ['ew28_x_dsl14','s28_x_churnrisk']
SNAP = ['snap_mean_s28','snap_med_s28','snap_mean_ew28','snap_mean_s84','snap_zeroshare','snap_mean_dsl','snap_p90_s28']
no_trend = [c for c in base_feats if c != 'snap_day']
no_cal = [c for c in no_trend if c not in ['wk_of_year','ann_sin','ann_cos']]

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

sets = {'base': base_feats, 'base+snap': base_feats+SNAP,
        'base+snap-notrend': no_trend+SNAP, 'base+snap-nocal': no_cal+SNAP,
        'base+snap+dsl': base_feats+SNAP+DSL}
for fit_max, tests in [(347,[403,431]), (375,[403,431]), (403,[431])]:
    for name, fs in sets.items():
        vals = '  '.join('%d:%.2f'%(t, pseudo(fs,30,fit_max,t)) for t in tests)
        print('fit<=%d %-18s %s' % (fit_max, name, vals))
    print()
