
import numpy as np, pandas as pd

BASE = load_saved('e011_discounts.parquet')
base_feats = [c for c in BASE.columns if c not in ('household_key','snapshot_day')]
tt = train_targets()
# fixed clip bounds from full base (feature-scale constants only)
for c in ['spend_28','spend_84','spend_365','lr_max28','basket_max_84']:
    print(c, 'p99', round(BASE[c].quantile(0.99),1), 'max', round(BASE[c].max(),1))

def fn(view, sd):
    b = BASE[BASE['snapshot_day'] == sd].drop(columns=['snapshot_day']).set_index('household_key')
    hh = pd.Index(b.index); f = pd.DataFrame(index=hh)
    s28, dsl = b['spend_28'], b['days_since_last']
    # dsl hinges + interactions
    for k in [7,14,21,35,60,120]:
        f['dsl_le_%d'%k] = (dsl <= k).astype(float)
    for k in [7,14,21,35]:
        f['s28_x_dsl%d'%k] = s28 * (dsl <= k)
    f['ew28_x_dsl14'] = b['ew_28'] * (dsl <= 14)
    f['s28_x_churnrisk'] = s28 * (dsl > 28).astype(float)
    # snapshot-level aggregates (constant per snapshot)
    f['snap_mean_s28'] = b['spend_28'].mean(); f['snap_med_s28'] = b['spend_28'].median()
    f['snap_mean_ew28'] = b['ew_28'].mean(); f['snap_mean_s84'] = b['spend_84'].mean()
    f['snap_zeroshare'] = (b['spend_28'] == 0).mean(); f['snap_mean_dsl'] = b['days_since_last'].mean()
    f['snap_p90_s28'] = b['spend_28'].quantile(0.9)
    # clipped heavy tails
    f['s28_c'] = s28.clip(upper=900); f['s84_c'] = b['spend_84'].clip(upper=2200)
    f['s365_c'] = b['spend_365'].clip(upper=5000); f['lrmax_c'] = b['lr_max28'].clip(upper=1600)
    return b.join(f, how='left')

out = build_features(fn)
DSL = ['dsl_le_%d'%k for k in [7,14,21,35,60,120]] + ['s28_x_dsl%d'%k for k in [7,14,21,35]] + ['ew28_x_dsl14','s28_x_churnrisk']
SNAP = ['snap_mean_s28','snap_med_s28','snap_mean_ew28','snap_mean_s84','snap_zeroshare','snap_mean_dsl','snap_p90_s28']
CLIP = ['s28_c','s84_c','s365_c','lrmax_c']
print('out shape', out.shape)

df = out.merge(tt, on=['household_key','snapshot_day'])
no_index = [c for c in base_feats if c != 'index']

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

sets = {'base78': base_feats, 'no_index': no_index, '+dsl': no_index+DSL, '+snap': no_index+SNAP,
        '+clip': no_index+CLIP, '+dsl+snap': no_index+DSL+SNAP, '+dsl+snap+clip': no_index+DSL+SNAP+CLIP}
for name, fs in sets.items():
    print('%-16s 431: %.3f  403: %.3f' % (name, pseudo(fs,30,431), pseudo(fs,30,403)))
best = min(sets, key=lambda n: pseudo(sets[n],30,431)+pseudo(sets[n],30,403))
print('BEST:', best)
save_table(out[ ['household_key','snapshot_day'] + sets['+dsl+snap+clip'] ], 'e015_full_superset.parquet')
save_table(out[ ['household_key','snapshot_day'] + sets[best] ], 'e015_best_pseudo.parquet')
print('saved both')
