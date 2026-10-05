
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
