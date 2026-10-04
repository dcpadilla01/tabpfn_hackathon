import agent_api as api, pandas as pd, numpy as np, re

EPS = 1e-6
TOP_DEPTS = ['GROCERY','DRUG GM','MEAT','PRODUCE','KIOSK-GAS','MEAT-PCKGD','DELI','PASTRY']
def san(s): return re.sub(r'\W+','_',s).strip('_')

def sz_parse(s):
    if isinstance(s, str):
        m = re.search(r'(\d+(\.\d+)?)', s)
        if m: return float(m.group(1))
    return np.nan

def build(view, day):
    hh = view.households
    if isinstance(hh, pd.DataFrame):
        idx = pd.Index(pd.unique(hh['household_key']), name='household_key')
    else:
        idx = pd.Index(pd.unique(np.asarray(hh)), name='household_key')
    out = pd.DataFrame(index=idx)

    tx = view.transactions
    p = view.products.set_index('product_id')
    t = pd.DataFrame({
        'hk': tx['household_key'].values,
        'bid': tx['basket_id'].values,
        'd': tx['day'].values,
        'sv': tx['sales_value'].values,
        'store': tx['store_id'].values,
        'tt': tx['trans_time'].values,
        'brand': tx['product_id'].map(p['brand']).values,
        'dept': tx['product_id'].map(p['department']).values,
        'pkg': tx['product_id'].map(p['curr_size_of_product'].map(sz_parse)).values,
        'mfr': tx['product_id'].map(p['manufacturer']).fillna('__NA__').values,
    })
    d, sv = t['d'].values, t['sv'].values
    m28 = d > day - 28
    m112 = d > day - 112
    priv = t['brand'].values == 'Private'

    spend28 = t.loc[m28].groupby('hk')['sv'].sum().reindex(idx).fillna(0.0)
    # --- private label ---
    for name, mask in [('28', m28), ('112', m112), ('lt', np.ones(len(t), bool))]:
        ps = t.loc[mask & priv].groupby('hk')['sv'].sum().reindex(idx).fillna(0.0)
        tot = t.loc[mask].groupby('hk')['sv'].sum().reindex(idx).fillna(0.0)
        out['priv_share' + name] = ps / (tot + EPS)
    out['priv_share_trend'] = out['priv_share28'] - out['priv_share112']
    out['priv_trips28'] = t.loc[m28 & priv].groupby('hk')['bid'].nunique().reindex(idx).fillna(0.0)
    # --- manufacturer concentration ---
    for wname, mask in [('28', m28), ('112', m112)]:
        sub = t.loc[mask]
        tot = sub.groupby('hk')['sv'].sum().reindex(idx).fillna(0.0)
        gs = sub.groupby(['hk', 'mfr'])['sv'].sum().reset_index()
        gs['rk'] = gs.groupby('hk')['sv'].rank(ascending=False, method='first')
        out['mfr_top1_share' + wname] = (gs.loc[gs.rk == 1].groupby('hk')['sv'].sum().reindex(idx).fillna(0.0) / (tot + EPS))
        if wname == '28':
            out['mfr_top3_share28'] = (gs.loc[gs.rk <= 3].groupby('hk')['sv'].sum().reindex(idx).fillna(0.0) / (tot + EPS))
            out['n_mfr28'] = gs.groupby('hk').size().reindex(idx).fillna(0.0)
    # --- package size / bulk ---
    def wavg_pkg(mask):
        sub = t.loc[mask & t['pkg'].notna()]
        if len(sub) == 0: return pd.Series(0.0, index=idx)
        g = sub.assign(wx=sub['sv'] * sub['pkg']).groupby('hk')[['wx', 'sv']].sum()
        return (g['wx'] / (g['sv'] + EPS)).reindex(idx)
    w28, w112 = wavg_pkg(m28), wavg_pkg(m112)
    out['pkg_wavg28'] = w28
    out['pkg_ratio28_112'] = w28 / (w112 + EPS)
    bulk = t['pkg'].values >= 24
    for name, mask in [('28', m28), ('112', m112)]:
        bs = t.loc[mask & bulk].groupby('hk')['sv'].sum().reindex(idx).fillna(0.0)
        tot = t.loc[mask].groupby('hk')['sv'].sum().reindex(idx).fillna(0.0)
        out['bulk_share' + name] = bs / (tot + EPS)
    # --- per-department absolute 28d spend + momentum ---
    for dd in TOP_DEPTS:
        s = san(dd)
        dm = t['dept'].values == dd
        d28 = t.loc[m28 & dm].groupby('hk')['sv'].sum().reindex(idx).fillna(0.0)
        d112 = t.loc[m112 & dm].groupby('hk')['sv'].sum().reindex(idx).fillna(0.0)
        out['d28_' + s] = d28
        out['mom_' + s] = np.minimum(d28 / (d112 / 4.0 + 1.0), 100.0)
    # --- basket tails ---
    b28 = t.loc[m28].groupby(['hk', 'bid'])['sv'].sum().reset_index()
    out['bk_med28'] = b28.groupby('hk')['sv'].median().reindex(idx)
    out['nline28'] = t.loc[m28].groupby('hk').size().reindex(idx).fillna(0.0)
    ntrips28 = b28.groupby('hk').size().reindex(idx).fillna(0.0)
    out['lines_per_trip28'] = out['nline28'] / (ntrips28 + EPS)
    b112 = t.loc[m112].groupby(['hk', 'bid'])['sv'].sum().reset_index()
    med112 = b112.groupby('hk')['sv'].median().reindex(idx)
    b28j = b28.join(med112.rename('med112'), on='hk')
    big = b28j.loc[b28j['sv'] > 1.5 * b28j['med112'].fillna(b28j['sv'].median())]
    out['big_basket_share28'] = big.groupby('hk')['sv'].sum().reindex(idx).fillna(0.0) / (spend28 + EPS)
    # --- store switching ---
    s28 = t.loc[m28].groupby(['hk', 'store'])['sv'].sum()
    tot28 = s28.groupby('hk').sum().reindex(idx).fillna(0.0)
    p28 = (s28 / (tot28.reindex(s28.index.get_level_values(0)).values + EPS) + EPS)
    out['store_entropy28'] = (-p28 * np.log(p28)).groupby('hk').sum().reindex(idx).fillna(0.0)
    prior = t[(d <= day - 28) & (d > day - 140)][['hk', 'store']].drop_duplicates()
    prior['f'] = 1
    b = t.loc[m28].groupby(['hk', 'bid'])['store'].first().reset_index().merge(prior, on=['hk', 'store'], how='left')
    newt = b.loc[b['f'].isna()].groupby('hk').size().reindex(idx).fillna(0.0)
    out['new_store_share28'] = newt / (ntrips28 + EPS)
    # --- time of day ---
    hour = (t['tt'].fillna(-1) // 100).values
    sub = t.loc[m28]
    h = hour[m28]
    cnt = sub.groupby('hk').size().reindex(idx).fillna(0.0) + EPS
    out['morning_share28'] = sub.assign(m=(h < 12).astype(float)).groupby('hk')['m'].sum().reindex(idx).fillna(0.0) / cnt
    out['evening_share28'] = sub.assign(e=(h >= 18).astype(float)).groupby('hk')['e'].sum().reindex(idx).fillna(0.0) / cnt
    return out

df = api.build_features(build)
print('built', df.shape, 'feats', df.shape[1] - 2)
print(df.isna().mean().sort_values(ascending=False).head(8))
e18 = api.load_saved('e018_basestab.parquet')
m = e18.merge(df.reset_index(), on=['household_key', 'snapshot_day'], how='inner')
print('merged', m.shape)
m['ix_sp28_season'] = m['spend28'] * m['season_lift']
m['ix_sp28_tgt'] = m['spend28'] * m['targeted_flag']
m['ix_sp28_rec'] = m['spend28'] / (1.0 + m['recency'].clip(lower=0))
m['ix_sp28_act'] = m['spend28'] * m['act_rate_4w'].fillna(0.0)
print('final feats', m.shape[1] - 2)
path = api.save_table(m, 'e020_rawmix.parquet')
print(path)