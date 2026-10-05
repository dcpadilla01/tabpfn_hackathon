
import numpy as np, pandas as pd
print('e009 cols:', load_saved('e009_ar_season.parquet').columns.tolist())

BASE = load_saved('e011_discounts.parquet')
base_feats = [c for c in BASE.columns if c not in ('household_key','snapshot_day')]

def fn(view, sd):
    b = BASE[BASE['snapshot_day'] == sd].drop(columns=['snapshot_day']).set_index('household_key')
    hh = pd.Index(b.index); f = pd.DataFrame(index=hh)
    s28 = b['spend_28']; dsl = b['days_since_last']
    # A: linear-spline cumulative dummies on spend_28 + interactions
    for k in [0,10,25,50,100,200,400,800]:
        f['s28_gt_%d'%k] = (s28 > k).astype(float)
    f['s28_x_rec14'] = s28 * (dsl <= 14); f['s28_x_rec7'] = s28 * (dsl <= 7)
    f['ew28_x_rec14'] = b['ew_28'] * (dsl <= 14)
    f['s28_x_freq'] = s28 * b['trips_per_wk_84']; f['s28_x_basket'] = s28 * b['avg_basket_84'] / 50
    # B: forward-looking campaign exposure
    camps = view.campaigns
    cmap_start = camps.set_index('campaign')['start_day']; cmap_end = camps.set_index('campaign')['end_day']
    ct = view.campaign_targets; ct = ct[ct.household_key.isin(hh)].copy()
    ct['cstart'] = ct.campaign.map(cmap_start); ct['cend'] = ct.campaign.map(cmap_end)
    g = ct.groupby('household_key')
    f['targ_still_running'] = g.apply(lambda x: ((x.cstart<=sd)&(x.cend>sd)).sum())
    f['targ_fut_overlap_days'] = g.apply(lambda x: np.clip(np.minimum(x.cend, sd+28)-sd, 0, None).clip(upper=28).sum())
    f['targ_active_now'] = g.apply(lambda x: ((x.cstart<=sd)&(x.cend>=sd)).sum())
    # C: demo interactions
    f['s28_x_hasdemo'] = s28 * b['has_demo']; f['ew28_x_hasdemo'] = b['ew_28'] * b['has_demo']
    f['s28_x_c4'] = s28 * pd.to_numeric(b['demo_c4'], errors='coerce')
    f['s28_x_owner'] = s28 * (b['homeowner'].astype(str)=='Homeowner')
    return b.join(f, how='left')

out = build_features(fn)
A = ['s28_gt_%d'%k for k in [0,10,25,50,100,200,400,800]] + ['s28_x_rec14','s28_x_rec7','ew28_x_rec14','s28_x_freq','s28_x_basket']
B = ['targ_still_running','targ_fut_overlap_days','targ_active_now']
C = ['s28_x_hasdemo','ew28_x_hasdemo','s28_x_c4','s28_x_owner']
print('out shape', out.shape)

tt = train_targets()
df = out.merge(tt, on=['household_key','snapshot_day'])
mom_cols = [c for c in base_feats if any(k in c for k in ['mom','accel','recent_vs','wcv','slope','ann_','wk_of_year','index'])]
disc_cols = ['rd_84','cd_84','disc_share_84','coup_trips_84','rd_364','cd_364','disc_share_364','coup_trips_364','coup_trip_share_364','net_spend_364','units_per_trip_364','days_since_redem','redem_84']
print('mom_cols', mom_cols)

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

for name, fs in [('base78', base_feats), ('+A_spline', base_feats+A), ('+B_futcamp', base_feats+B),
                 ('+C_demoint', base_feats+C), ('+ABC', base_feats+A+B+C),
                 ('drop_mom', [c for c in base_feats if c not in mom_cols]),
                 ('drop_disc', [c for c in base_feats if c not in disc_cols])]:
    print('%-12s 431: %.3f  403: %.3f' % (name, pseudo(fs,30,431), pseudo(fs,30,403)))
