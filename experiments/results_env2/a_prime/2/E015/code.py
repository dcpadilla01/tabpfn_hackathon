
base = load_saved('e011_discounts.parquet')
print(base.shape)
print(base.columns.tolist())
print(base.dtypes.value_counts())

# verify load_saved works inside build_features and rows line up
def fn(view, sd):
    b = load_saved('e011_discounts.parquet')
    b = b[b['snapshot_day'] == sd].set_index('household_key')
    return b.drop(columns=['snapshot_day'])

out = build_features(fn)
print('built:', out.shape)
print('n snapshots:', out.snapshot_day.nunique() if 'snapshot_day' in out.columns else 'n/a')
print(out.head(3))


# ---- cell ----

# Test: are outer-scope globals visible inside fn?
SENTINEL = 'visible'
def fn(view, sd):
    try:
        val = SENTINEL
    except NameError:
        val = 'NameError'
    return pd.DataFrame({'g_test': [val]}, index=view.households)

out = build_features(fn)
print('global test:', out.g_test.unique())

# Data details needed for reconstruction
s = snapshot(459)
t = s.transactions
print('\ntrans shape', t.shape)
print(t[['sales_value','coupon_disc','coupon_match_disc','retail_disc','quantity']].describe().round(3))
print('\ncampaigns head'); print(s.campaigns.head(8).to_string())
print('\ncampaign_targets head'); print(s.campaign_targets.head(5).to_string())
print('campaign_targets desc counts:'); print(s.campaign_targets.description.value_counts())
print('\ndemographics classification_1 uniques:', sorted(s.demographics.classification_1.dropna().unique())[:10])
print('homeowner:', s.demographics.homeowner_desc.value_counts().to_dict())
print('kid_cat:', s.demographics.kid_category_desc.value_counts().to_dict())
print('c4:', s.demographics.classification_4.value_counts().to_dict())
print('snapshot_days:', snapshot_days())


# ---- cell ----

import numpy as np, pandas as pd

# --- facts: campaign timing coverage ---
c = snapshot(459).campaigns
print('campaigns n=', len(c), 'min start', c.start_day.min(), 'max end', c.end_day.max())
print(c.sort_values('start_day').to_string())
for d in [95, 207, 319, 347, 431]:
    cc = snapshot(d).campaigns
    print(d, 'n campaigns', len(cc), 'starts', sorted(cc.start_day.tolist())[:6])
r = snapshot(459).coupon_redemptions
print('redemptions day range', r.day.min(), r.day.max(), 'n', len(r))

tt = train_targets()
print(tt.future_spend_4w.describe().round(2))
print('zero share', round((tt.future_spend_4w==0).mean(),3), 'n rows', len(tt))

# --- ridge proxy on E011 base ---
base = load_saved('e011_discounts.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'])
tr_days = snapshot_days()['train']; va_days = snapshot_days()['validation']
feats = [c for c in base.columns if c not in ('household_key','snapshot_day')]

def prep(d):
    X = d[feats].copy()
    for c in feats:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype('category').cat.codes.astype(float)
            X[c] = X[c].where(X[c] >= 0, np.nan)
    return X.astype(float)

tr_df, va_df = df[df.snapshot_day.isin(tr_days)], df[df.snapshot_day.isin(va_days)]
Xtr_raw, Xva_raw = prep(tr_df), prep(va_df)
mu, sg = Xtr_raw.mean(), Xtr_raw.std().replace(0,1)+1e-9
Xtr = ((Xtr_raw-mu)/sg).fillna(0.0).values; Xva = ((Xva_raw-mu)/sg).fillna(0.0).values
ytr, yva = tr_df.future_spend_4w.values, va_df.future_spend_4w.values

def ridge_eval(lam):
    A = np.hstack([Xtr, np.ones((len(Xtr),1))]); Av = np.hstack([Xva, np.ones((len(Xva),1))])
    D = np.eye(A.shape[1]); D[-1,-1] = 0
    w = np.linalg.solve(A.T@A + lam*D, A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(Av@w-yva).mean()

for lam in [1,3,10,30,100,300,1000,3000]:
    trm, vam = ridge_eval(lam)
    print('lam', lam, 'train MAE', round(trm,3), 'val MAE', round(vam,3))


# ---- cell ----

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


# ---- cell ----

import numpy as np, pandas as pd

BASE = load_saved('e011_discounts.parquet')
dm_check = snapshot(459).display_mailer
print('dm shape', dm_check.shape, dm_check.dtypes.to_dict())
print(dm_check.head(4).to_string())
print('display vals', dm_check.display.value_counts().head().to_dict())
print('mailer vals', dm_check.mailer.value_counts().head().to_dict())
print('trans product_id dtype', snapshot(459).transactions.product_id.dtype, 'store', snapshot(459).transactions.store_id.dtype)

def fn(view, sd):
    b = BASE[BASE['snapshot_day'] == sd].drop(columns=['snapshot_day']).set_index('household_key')
    hh = pd.Index(b.index)
    t = view.table('transactions')
    t = t[t.household_key.isin(hh)]

    def k(s):
        try: return s.astype('int64')
        except Exception: return s.astype(str)

    camps = view.campaigns
    cmap_type = camps.set_index('campaign')['description']
    cmap_start = camps.set_index('campaign')['start_day']
    cmap_end = camps.set_index('campaign')['end_day']
    ct = view.campaign_targets
    ct = ct[ct.household_key.isin(hh)].copy()
    ct['ctype'] = ct.campaign.map(cmap_type)
    ct['cstart'] = ct.campaign.map(cmap_start)
    ct['cend'] = ct.campaign.map(cmap_end)
    f = pd.DataFrame(index=hh)
    g = ct.groupby('household_key')
    f['n_targ_camps'] = g['campaign'].nunique()
    for typ in ['TypeA','TypeB','TypeC']:
        sub = ct[ct.ctype == typ]
        n = sub.groupby('household_key')['campaign'].nunique()
        f['targ_n_' + typ] = n
        f['targ_ever_' + typ] = (n > 0).astype(float)
    act = ct[(ct.cstart <= sd) & (ct.cend >= sd)]
    f['n_targ_active'] = act.groupby('household_key')['campaign'].nunique()
    f['n_targ_recent56'] = ct[(sd - ct.cstart) <= 56].groupby('household_key')['campaign'].nunique()
    f['n_targ_recent84'] = ct[(sd - ct.cstart) <= 84].groupby('household_key')['campaign'].nunique()

    cr = view.coupon_redemptions
    cr = cr[cr.household_key.isin(hh)].copy()
    cr['ctype'] = cr.campaign.map(cmap_type)
    g2 = cr.groupby('household_key')
    f['n_redem_total'] = g2['day'].count()
    for w in [28, 56, 180]:
        f['n_redem_' + str(w)] = cr[cr.day > sd - w].groupby('household_key')['day'].count()
    f['days_since_first_redem'] = sd - g2['day'].min()
    f['n_redem_camps'] = g2['campaign'].nunique()
    for typ in ['TypeA','TypeB','TypeC']:
        f['n_redem_' + typ] = cr[cr.ctype == typ].groupby('household_key')['day'].count()
    eng = cr.merge(ct[['household_key','campaign']].drop_duplicates(), on=['household_key','campaign'])
    f['n_redem_targ_camps'] = eng.groupby('household_key')['campaign'].nunique()

    dm = view.display_mailer
    dm = dm.assign(product_id=k(dm.product_id), store_id=k(dm.store_id))
    for w, tag in [(84,''), (28,'28')]:
        tw = t[t.day > sd - w].copy()
        tw['week_no'] = (tw.day + 8) // 7
        tw['product_id'] = k(tw.product_id); tw['store_id'] = k(tw.store_id)
        m = tw.merge(dm, on=['product_id','store_id','week_no'], how='left')
        disp = m['display'].notna() & (m['display'] != 0)
        mail = m['mailer'].notna() & (m['mailer'] != 0) & (m['mailer'] != '')
        sp = m.groupby('household_key')['sales_value'].sum()
        f['disp_share' + tag] = m[disp].groupby('household_key')['sales_value'].sum() / sp
        f['mail_share' + tag] = m[mail].groupby('household_key')['sales_value'].sum() / sp
        f['n_disp_baskets' + tag] = m[disp].groupby('household_key')['basket_id'].nunique()
        f['n_mail_baskets' + tag] = m[mail].groupby('household_key')['basket_id'].nunique()
    return b.join(f, how='left')

out = build_features(fn)
print('out shape', out.shape)
new_cols = [c for c in out.columns if c not in BASE.columns]
print('new cols', len(new_cols), new_cols)
print(out[new_cols].isna().mean().round(3).to_string())

# pseudo-eval
tt = train_targets()
df = out.merge(tt, on=['household_key','snapshot_day'])
base_feats = [c for c in BASE.columns if c not in ('household_key','snapshot_day')]
feats = base_feats + new_cols

def prep(d, fs):
    X = d[fs].copy()
    for c in fs:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype('category').cat.codes.astype(float)
            X[c] = X[c].where(X[c] >= 0, np.nan)
    return X.astype(float)

def pseudo(df, fs, lam=30, test_day=431):
    fit_days = [d for d in snapshot_days()['train'] if d < test_day]
    tr = df[df.snapshot_day.isin(fit_days)]; te = df[df.snapshot_day == test_day]
    Xtr, Xte = prep(tr, fs), prep(te, fs)
    mu, sg = Xtr.mean(), Xtr.std().replace(0,1)+1e-9
    Xtr = ((Xtr-mu)/sg).fillna(0.).values; Xte = ((Xte-mu)/sg).fillna(0.).values
    ytr, yte = tr.future_spend_4w.values, te.future_spend_4w.values
    A = np.hstack([Xtr, np.ones((len(Xtr),1))]); Av = np.hstack([Xte, np.ones((len(Xte),1))])
    D = np.eye(A.shape[1]); D[-1,-1] = 0
    w = np.linalg.solve(A.T@A + lam*D, A.T@ytr)
    return np.abs(Av@w-yte).mean()

for lam in [10, 30, 100]:
    print('lam', lam, 'base', round(pseudo(df, base_feats, lam),3), 'cand', round(pseudo(df, feats, lam),3))

save_table(out, 'e015_marketing2.parquet')
print('saved')


# ---- cell ----

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


# ---- cell ----

import numpy as np, pandas as pd

BASE = load_saved('e011_discounts.parquet')
base_feats = [c for c in BASE.columns if c not in ('household_key','snapshot_day')]

def fn(view, sd):
    b = BASE[BASE['snapshot_day'] == sd].drop(columns=['snapshot_day']).set_index('household_key')
    hh = pd.Index(b.index); f = pd.DataFrame(index=hh)
    s28 = b['spend_28']; dsl = b['days_since_last']
    for k in [0,10,25,50,100,200,400,800]:
        f['s28_gt_%d'%k] = (s28 > k).astype(float)
    f['s28_x_rec14'] = s28 * (dsl <= 14); f['s28_x_rec7'] = s28 * (dsl <= 7)
    f['ew28_x_rec14'] = b['ew_28'] * (dsl <= 14)
    f['s28_x_freq'] = s28 * b['trips_per_wk_84']; f['s28_x_basket'] = s28 * b['avg_basket_84'] / 50
    camps = view.campaigns
    cmap_start = camps.set_index('campaign')['start_day']; cmap_end = camps.set_index('campaign')['end_day']
    ct = view.campaign_targets; ct = ct[ct.household_key.isin(hh)].copy()
    ct['cstart'] = ct.campaign.map(cmap_start); ct['cend'] = ct.campaign.map(cmap_end)
    ct['still'] = ((ct.cstart <= sd) & (ct.cend > sd)).astype(float)
    ct['ovl'] = np.clip(np.minimum(ct.cend, sd+28) - sd, 0, None).clip(upper=28)
    ct['actnow'] = ((ct.cstart <= sd) & (ct.cend >= sd)).astype(float)
    g = ct.groupby('household_key')
    f['targ_still_running'] = g['still'].sum()
    f['targ_fut_overlap_days'] = g['ovl'].sum()
    f['targ_active_now'] = g['actnow'].sum()
    f['s28_x_hasdemo'] = s28 * b['has_demo']; f['ew28_x_hasdemo'] = b['ew_28'] * b['has_demo']
    f['s28_x_c4'] = s28 * pd.to_numeric(b['demo_c4'], errors='coerce')
    f['s28_x_owner'] = s28 * (b['homeowner'].astype(str)=='Homeowner')
    return b.join(f, how='left')

out = build_features(fn)
A = ['s28_gt_%d'%k for k in [0,10,25,50,100,200,400,800]] + ['s28_x_rec14','s28_x_rec7','ew28_x_rec14','s28_x_freq','s28_x_basket']
B = ['targ_still_running','targ_fut_overlap_days','targ_active_now']
C = ['s28_x_hasdemo','ew28_x_hasdemo','s28_x_c4','s28_x_owner']
tt = train_targets()
df = out.merge(tt, on=['household_key','snapshot_day'])
mom_cols = [c for c in base_feats if any(k in c for k in ['mom','accel','recent_vs','wcv','slope','ann_','wk_of_year','index'])]
disc_cols = ['rd_84','cd_84','disc_share_84','coup_trips_84','rd_364','cd_364','disc_share_364','coup_trips_364','coup_trip_share_364','net_spend_364','units_per_trip_364','days_since_redem','redem_84']

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


# ---- cell ----

import numpy as np, pandas as pd

BASE = load_saved('e011_discounts.parquet')
base_feats = [c for c in BASE.columns if c not in ('household_key','snapshot_day')]
print('index col sample:', BASE['index'].describe().to_dict())
print('snap_day sample:', BASE['snap_day'].describe().to_dict())

tt = train_targets()
df = BASE.merge(tt, on=['household_key','snapshot_day'])

mom = ['mom28_delta','mom28_ratio','recent_vs_typ','accel','slope_13w','wcv_26']
cal = ['snap_day','wk_of_year','ann_sin','ann_cos','index']

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

sets = {
 'base78': base_feats,
 'drop_mom_only': [c for c in base_feats if c not in mom],
 'drop_cal_only': [c for c in base_feats if c not in cal],
 'drop_mom+cal': [c for c in base_feats if c not in mom+cal],
 'drop_index_snapday': [c for c in base_feats if c not in ['index','snap_day']],
 'drop_wkofyr_ann': [c for c in base_feats if c not in ['wk_of_year','ann_sin','ann_cos']],
}
for name, fs in sets.items():
    print('%-20s 431: %.3f  403: %.3f' % (name, pseudo(fs,30,431), pseudo(fs,30,403)))


# ---- cell ----

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


# ---- cell ----

import numpy as np, pandas as pd
tt = train_targets()
g = tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median','count'])
print(g.round(2).to_string())
BASE = load_saved('e011_discounts.parquet')
b = BASE.groupby('snapshot_day')[['spend_28','spend_365','ew_28']].mean().round(2)
print(b.to_string())
# correlation of day with mean target
days = g.index.values.astype(float); m = g['mean'].values
print('corr(day, mean target):', np.corrcoef(days, m)[0,1].round(3))
sl = np.polyfit(days, m, 1); print('slope per day:', sl[0].round(4), '=> per 28d:', (sl[0]*28).round(2))
# trend in spend_28 (behavioral level)
b2 = BASE.groupby('snapshot_day').spend_28.mean()
print('corr(day, mean spend_28):', np.corrcoef(b2.index.values.astype(float), b2.values)[0,1].round(3))
sl2 = np.polyfit(b2.index.values.astype(float), b2.values, 1); print('spend_28 slope/day:', sl2[0].round(4))
# zero share by day
z = tt.groupby('snapshot_day').future_spend_4w.apply(lambda x: (x==0).mean()).round(3)
print(z.to_string())


# ---- cell ----

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


# ---- cell ----

import numpy as np, pandas as pd

out = load_saved('e015_full_superset.parquet')
tt = train_targets()
df = out.merge(tt, on=['household_key','snapshot_day'])
base_feats = [c for c in load_saved('e011_discounts.parquet').columns if c not in ('household_key','snapshot_day')]
DSL = ['dsl_le_%d'%k for k in [7,14,21,35,60,120]] + ['s28_x_dsl%d'%k for k in [7,14,21,35]] + ['ew28_x_dsl14','s28_x_churnrisk']
SNAP = ['snap_mean_s28','snap_med_s28','snap_mean_ew28','snap_mean_s84','snap_zeroshare','snap_mean_dsl','snap_p90_s28']
no_index = [c for c in base_feats if c != 'index']
no_trend = [c for c in no_index if c != 'snap_day']
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

sets = {'base': base_feats, 'noix': no_index, 'noix+snap': no_index+SNAP,
        'noix+snap-notrend': no_trend+SNAP, 'noix+snap-nocal': no_cal+SNAP,
        'noix+snap+dsl': no_index+SNAP+DSL}
for fit_max, tests in [(347,[403,431]), (375,[403,431]), (403,[431])]:
    for name, fs in sets.items():
        vals = ['%.2f'%pseudo(fs,30,fm,td) for fm,td in [(fit_max,t) for t in tests]]
        print('fit<=%d %-18s %s' % (fit_max, name, '  '.join('%d:%s'%(t,v) for (t,_),v in zip([(t,0) for t in tests], vals))))
    print()


# ---- cell ----

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


# ---- cell ----

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
