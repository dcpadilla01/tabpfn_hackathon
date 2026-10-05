
import numpy as np, pandas as pd

BASE = load_saved('e011_discounts.parquet')
base_feats = [c for c in BASE.columns if c not in ('household_key','snapshot_day')]
tt = train_targets()

def fn(view, sd):
    b = BASE[BASE['snapshot_day'] == sd].drop(columns=['snapshot_day']).set_index('household_key')
    hh = pd.Index(b.index); f = pd.DataFrame(index=hh)
    s28, dsl = b['spend_28'], b['days_since_last']
    # hinge/cumulative dummies on spend_28
    for k in [0,10,25,50,100,200,400,800]:
        f['s28_gt_%d'%k] = (s28 > k).astype(float)
    # hinges on other key predictors
    for col, ks in [('ew_28',[10,25,50,100,200]), ('spend_84',[50,150,400,800]),
                    ('spend_365',[200,600,1200]), ('lr_mean28',[10,30,80,150]),
                    ('avg_basket_84',[10,25,50,100])]:
        for k in ks:
            f['%s_gt_%d'%(col,k)] = (b[col] > k).astype(float)
    # interactions
    f['s28_x_rec14'] = s28 * (dsl <= 14); f['s28_x_rec7'] = s28 * (dsl <= 7)
    f['s28_x_freq'] = s28 * b['trips_per_wk_84']; f['s28_x_basket'] = s28 * b['avg_basket_84'] / 50
    f['s28_x_stab'] = s28 * (b['spend_28'] / (b['spend_28_prior'] + 1)).clip(0,3)
    f['s28_x_active'] = s28 * b['active_28']
    f['ew28_x_rec14'] = b['ew_28'] * (dsl <= 14)
    f['s84_x_freq'] = b['spend_84'] * b['trips_per_wk_84']
    f['s365_x_tenure'] = b['spend_365'] * (b['days_since_first'] / 365).clip(0, 1.5)
    return b.join(f, how='left')

out = build_features(fn)
H = [c for c in out.columns if c not in BASE.columns]
print('n new', len(H))
df = out.merge(tt, on=['household_key','snapshot_day'])

def prep(d, fs):
    X = d[fs].copy()
    for c in fs:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype('category').cat.codes.astype(float); X[c] = X[c].where(X[c]>=0, np.nan)
    return X.astype(float)

def pseudo(fs, lam=30, test_day=431):
    fit_days = [d for d in snapshot_days()['train'] if d < test_day]
    tr = df[df.snapshot_day.isin(fit_days)]; te = df[df.snapshot_day==test_day]
    Xtr, Xte = prep(tr, fs), prep(te, fs)
    mu, sg = Xtr.mean(), Xtr.std().replace(0,1)+1e-9
    Xtr = ((Xtr-mu)/sg).fillna(0.).values; Xte = ((Xte-mu)/sg).fillna(0.).values
    ytr, yte = tr.future_spend_4w.values, te.future_spend_4w.values
    A_ = np.hstack([Xtr, np.ones((len(Xtr),1))]); Av = np.hstack([Xte, np.ones((len(Xte),1))])
    D = np.eye(A_.shape[1]); D[-1,-1]=0
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr)
    return np.abs(Av@w-yte).mean()

no_index = [c for c in base_feats if c != 'index']
subsets = {
 'base78': base_feats,
 'no_index': no_index,
 'no_index+H': no_index + H,
 'no_index+H,drop_cal': no_index + H and [c for c in no_index if c not in ['snap_day','wk_of_year','ann_sin','ann_cos']] + H,
 'no_index,drop_cal': [c for c in no_index if c not in ['snap_day','wk_of_year','ann_sin','ann_cos']],
}
for name, fs in subsets.items():
    print('%-22s 431: %.3f  403: %.3f' % (name, pseudo(fs,30,431), pseudo(fs,30,403)))
for lam in [10, 100]:
    print('lam', lam, 'no_index+H 431:', round(pseudo(no_index+H, lam, 431),3), '403:', round(pseudo(no_index+H, lam, 403),3))
save_table(out, 'e015_hinges.parquet')
print('saved')
