import pandas as pd, numpy as np, time

TOPD = ['GROCERY','DRUG GM','PRODUCE','COSMETICS','NUTRITION','MEAT','MEAT-PCKGD','DELI','PASTRY','FLORAL','SEAFOOD-PCKGD','MISC. TRANS.','SPIRITS','SEAFOOD']

def new_features(view, snapshot_day):
    day = int(view.day)
    t_all = view.table('transactions')
    first = t_all.groupby('household_key')['day'].min()
    hh = pd.Index(first[first <= day - 84].index, name='household_key')
    hset = set(hh)
    t = t_all[t_all['household_key'].isin(hset)][['household_key','day','sales_value','basket_id','product_id','quantity','coupon_disc','coupon_match_disc','retail_disc','trans_time']].copy()
    t['age'] = day - t['day']
    t['disc'] = t['coupon_disc'].fillna(0) + t['coupon_match_disc'].fillna(0) + t['retail_disc'].fillna(0)
    f = pd.DataFrame(index=hh)
    for w in (7, 14):
        m = t[t['age'] < w]
        f[f'sp{w}'] = m.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
        f[f'trips{w}'] = m.groupby('household_key')['basket_id'].nunique().reindex(hh).fillna(0.0)
    m84 = t[t['age'] < 84]
    for hl in (14, 28, 56):
        wgt = np.power(0.5, m84['age'] / hl)
        s = pd.Series(m84['sales_value'].values * wgt.values).groupby(m84['household_key'].values).sum()
        f[f'ewma{hl}'] = s.reindex(hh).fillna(0.0)
    prod = view.table('products')
    dept = prod.set_index('product_id')['department']
    t['department'] = t['product_id'].map(dept)
    for w, tag in ((84, '84'), (364, '364')):
        m = t[t['age'] < w]
        piv = m.pivot_table(index='household_key', columns='department', values='sales_value', aggfunc='sum')
        tot = m.groupby('household_key')['sales_value'].sum()
        for d in TOPD:
            if d in piv.columns:
                f[f'dsh{tag}_{d[:5]}'] = (piv[d] / tot).reindex(hh).fillna(0.0)
        f[f'ndept{tag}'] = m.groupby('household_key')['department'].nunique().reindex(hh).fillna(0.0)
    m364 = t[t['age'] < 364].copy()
    br = prod.set_index('product_id')['brand']
    m364['brand'] = m364['product_id'].map(br)
    pv = m364.pivot_table(index='household_key', columns='brand', values='sales_value', aggfunc='sum')
    tot = m364.groupby('household_key')['sales_value'].sum()
    f['privshare'] = (pv['Private'] / tot).reindex(hh).fillna(0.0) if 'Private' in pv.columns else 0.0
    dsum = m364.groupby('household_key')['disc'].sum()
    f['discshare364'] = (dsum / tot).reindex(hh).fillna(0.0)
    q364 = m364.groupby('household_key')['quantity'].sum()
    f['avgprice364'] = (tot / q364).reindex(hh)
    f['evshare84'] = m84.assign(ev=(m84['trans_time'] >= 1800).astype(float)).groupby('household_key')['ev'].mean().reindex(hh)
    ct = view.table('campaign_targets')
    ct = ct[ct['household_key'].isin(hset)]
    for typ in ('TypeA', 'TypeB', 'TypeC'):
        f[f'ncamp{typ}'] = ct[ct['description'] == typ].groupby('household_key').size().reindex(hh).fillna(0.0)
    camps = view.table('campaigns')
    active = set(camps[(camps['start_day'] <= day) & (camps['end_day'] >= day)]['campaign'])
    f['ncamp_active'] = ct[ct['campaign'].isin(active)].groupby('household_key').size().reindex(hh).fillna(0.0)
    cr = view.table('coupon_redemptions')
    cr = cr[cr['household_key'].isin(hset)].copy()
    cr['age'] = day - cr['day']
    f['nred84'] = cr[cr['age'] < 84].groupby('household_key').size().reindex(hh).fillna(0.0)
    f['nred364'] = cr[cr['age'] < 364].groupby('household_key').size().reindex(hh).fillna(0.0)
    f['days_since_red'] = cr.groupby('household_key')['age'].min().reindex(hh)
    m168 = t[t['age'] < 168]
    bt = m168[['household_key', 'basket_id', 'day']].drop_duplicates().sort_values(['household_key', 'day'])
    bt['gap'] = bt.groupby('household_key')['day'].diff()
    f['gap_mean'] = bt.groupby('household_key')['gap'].mean().reindex(hh)
    f['gap_med'] = bt.groupby('household_key')['gap'].median().reindex(hh)
    mc = m168.copy(); mc['wk'] = mc['day'] // 7
    ws = mc.groupby(['household_key', 'wk'])['sales_value'].sum()
    f['wksp_mean'] = ws.groupby('household_key').mean().reindex(hh)
    f['wksp_std'] = ws.groupby('household_key').std().reindex(hh)
    m84b = t[t['age'] < 84].copy(); m84b['wk'] = m84b['day'] // 7
    f['nwk84'] = m84b.groupby('household_key')['wk'].nunique().reindex(hh).fillna(0.0)
    return f

for d in (95, 459):
    t0 = time.time()
    v = agent_api.snapshot(d)
    f = new_features(v, d)
    print(f"day {d}: {f.shape}, {time.time()-t0:.1f}s")
    print("NaN cols:", dict(f.isna().sum()[f.isna().sum() > 0]))
print(f.describe().T[['mean','50%','max']].round(2).head(60))
