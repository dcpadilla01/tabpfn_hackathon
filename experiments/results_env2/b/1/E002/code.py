import pandas as pd, numpy as np
e1 = agent_api.load_saved('e001_history.parquet')
print("E001 shape:", e1.shape)
print(list(e1.columns))
tt = agent_api.train_targets()
print(tt['future_spend_4w'].describe())
v = agent_api.snapshot(459)
prod = v.table('products')
print(prod['department'].value_counts().head(16))
print(prod['brand'].value_counts())
camp = v.table('campaigns')
print(camp['description'].value_counts())
ct = v.table('campaign_targets')
print(ct['description'].value_counts())
cr = v.table('coupon_redemptions')
print("redemptions:", cr.shape, "max day:", cr['day'].max())
tx = v.table('transactions')
print("tx:", tx.shape, tx['day'].min(), tx['day'].max())
print(tx['sales_value'].describe())


# ---- cell ----
import pandas as pd, numpy as np

TOPD = ['GROCERY','DRUG GM','PRODUCE','COSMETICS','NUTRITION','MEAT','MEAT-PCKGD','DELI','PASTRY','FLORAL','SEAFOOD-PCKGD','MISC. TRANS.','SPIRITS','SEAFOOD']

def new_features(view, snapshot_day):
    day = int(view.day)
    hh = pd.Index(view.households, name='household_key')
    hset = set(hh)
    t = view.table('transactions')
    t = t[t['household_key'].isin(hset)][['household_key','day','sales_value','basket_id','product_id','quantity','coupon_disc','coupon_match_disc','retail_disc','trans_time']].copy()
    t['age'] = day - t['day']
    t['disc'] = t['coupon_disc'].fillna(0) + t['coupon_match_disc'].fillna(0) + t['retail_disc'].fillna(0)
    f = pd.DataFrame(index=hh)

    # finer recent windows
    for w in (7, 14):
        m = t[t['age'] < w]
        f[f'sp{w}'] = m.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
        f[f'trips{w}'] = m.groupby('household_key')['basket_id'].nunique().reindex(hh).fillna(0.0)

    # EWMA spend over last 84 days (recency-weighted)
    m84 = t[t['age'] < 84]
    for hl in (14, 28, 56):
        wgt = np.power(0.5, m84['age'] / hl)
        s = pd.Series(m84['sales_value'].values * wgt.values).groupby(m84['household_key'].values).sum()
        f[f'ewma{hl}'] = s.reindex(hh).fillna(0.0)

    # department mix shares
    prod = view.table('products')
    dept = prod.set_index('product_id')['department']
    t = t.assign(department=t['product_id'].map(dept))
    for w, tag in ((84, '84'), (364, '364')):
        m = t[t['age'] < w]
        piv = m.pivot_table(index='household_key', columns='department', values='sales_value', aggfunc='sum')
        tot = m.groupby('household_key')['sales_value'].sum()
        for d in TOPD:
            if d in piv.columns:
                f[f'dsh{tag}_{d[:5]}'] = (piv[d] / tot).reindex(hh).fillna(0.0)
        f[f'ndept{tag}'] = m.groupby('household_key')['department'].nunique().reindex(hh).fillna(0.0)

    # brand mix + discount reliance + avg unit price
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

    # campaigns / coupons
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

    # cadence + volatility
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

v = agent_api.snapshot(95)
f = new_features(v, 95)
print("snapshot 95:", f.shape)
print(f.head(3).T.head(30))
v2 = agent_api.snapshot(459)
f2 = new_features(v2, 459)
print("snapshot 459:", f2.shape)
print("NaN cols:", f2.isna().sum()[f2.isna().sum() > 0])


# ---- cell ----
v = agent_api.snapshot(95)
print(type(v.households))
print(v.households[:5] if hasattr(v.households,'__getitem__') else v.households)
print(type(v.day), v.day, getattr(v,'week',None))
print(type(v.table('transactions')))


# ---- cell ----
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


# ---- cell ----
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
    t['disc'] = -(t['coupon_disc'].fillna(0) + t['coupon_match_disc'].fillna(0) + t['retail_disc'].fillna(0))
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

t0 = time.time()
def full_fn(view, snapshot_day):
    base = agent_api.baseline_features()
    e1 = agent_api.load_saved('e001_history.parquet')
    nf = new_features(view, snapshot_day)
    idx = nf.index
    e1s = e1[(e1['snapshot_day'] == snapshot_day)].set_index('household_key').drop(columns=['snapshot_day'], errors='ignore')
    out = e1s.reindex(idx).join(nf)
    out = out.reset_index()
    out.insert(1, 'snapshot_day', snapshot_day)
    return out

tbl = agent_api.build_features(full_fn)
print("built:", tbl.shape, f"{time.time()-t0:.1f}s")
print(tbl.head(2).T.head(8))
p = agent_api.save_table(tbl, 'e002_mix.parquet')
print(p)


# ---- cell ----
import pandas as pd, numpy as np, time

TOPD = ['GROCERY','DRUG GM','PRODUCE','COSMETICS','NUTRITION','MEAT','MEAT-PCKGD','DELI','PASTRY','FLORAL','SEAFOOD-PCKGD','MISC. TRANS.','SPIRITS','SEAFOOD']
WINS = (28, 56, 84, 168, 364, 728)

def full_fn(view, snapshot_day):
    day = int(view.day)
    t_all = view.table('transactions')
    first = t_all.groupby('household_key')['day'].min()
    hh = pd.Index(first[first <= day - 84].index, name='household_key')
    hset = set(hh)
    t = t_all[t_all['household_key'].isin(hset)].copy()
    t['age'] = day - t['day']
    t['cdisc'] = -(t['coupon_disc'].fillna(0) + t['coupon_match_disc'].fillna(0))
    t['rdisc'] = -t['retail_disc'].fillna(0)
    t['disc'] = t['cdisc'] + t['rdisc']
    f = pd.DataFrame(index=hh)
    g = t.groupby('household_key')
    f['tenure'] = (day - g['day'].min()).reindex(hh)
    f['days_since_last'] = (day - g['day'].max()).reindex(hh)
    for w in WINS:
        m = t[t['age'] < w]
        gm = m.groupby('household_key')
        sp = gm['sales_value'].sum()
        trips = gm['basket_id'].nunique()
        f[f'sp{w}'] = sp.reindex(hh).fillna(0.0)
        f[f'trips{w}'] = trips.reindex(hh).fillna(0.0)
        f[f'prods{w}'] = gm['product_id'].nunique().reindex(hh).fillna(0.0)
        f[f'stores{w}'] = gm['store_id'].nunique().reindex(hh).fillna(0.0)
        f[f'qty{w}'] = gm['quantity'].sum().reindex(hh).fillna(0.0)
        f[f'avgbs{w}'] = (sp / trips).reindex(hh)
        f[f'maxbs{w}'] = m.groupby(['household_key','basket_id'])['sales_value'].sum().groupby('household_key').max().reindex(hh).fillna(0.0)
        f[f'nact{w}'] = gm['day'].nunique().reindex(hh).fillna(0.0)
    f['cdisc84'] = t[t['age'] < 84].groupby('household_key')['cdisc'].sum().reindex(hh).fillna(0.0)
    f['rdisc84'] = t[t['age'] < 84].groupby('household_key')['rdisc'].sum().reindex(hh).fillna(0.0)
    f['cdisc364'] = t[t['age'] < 364].groupby('household_key')['cdisc'].sum().reindex(hh).fillna(0.0)
    f['rdisc364'] = t[t['age'] < 364].groupby('household_key')['rdisc'].sum().reindex(hh).fillna(0.0)
    f['trend_28_56'] = f['sp28'] / f['sp56'].replace(0, np.nan)
    f['trend_84'] = (f['sp28'] * 3) / f['sp84'].replace(0, np.nan)
    f['sp28_rate'] = f['sp28'] / 28.0
    f['sp84_rate'] = f['sp84'] / 84.0
    f['sp364_rate'] = f['sp364'] / 364.0
    lag = t[(t['age'] >= 364) & (t['age'] < 392)].groupby('household_key')['sales_value'].sum()
    f['sp_lag1y'] = lag.reindex(hh).fillna(0.0)
    # demographics
    demo = view.table('demographics')
    demo = demo.set_index('household_key')
    for c in demo.columns:
        f['d_' + c] = demo[c].reindex(hh)
    f['has_demo'] = f['d_classification_1'].notna().astype(float)
    f['week_mod52'] = int(view.week) % 52

    # ---- NEW: mix / brand / discount / marketing / cadence ----
    for w in (7, 14):
        m = t[t['age'] < w]
        f[f'nsp{w}'] = m.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
        f[f'ntrips{w}'] = m.groupby('household_key')['basket_id'].nunique().reindex(hh).fillna(0.0)
    m84 = t[t['age'] < 84]
    for hl in (14, 28, 56):
        wgt = np.power(0.5, m84['age'] / hl)
        s = pd.Series(m84['sales_value'].values * wgt.values).groupby(m84['household_key'].values).sum()
        f[f'newma{hl}'] = s.reindex(hh).fillna(0.0)
    prod = view.table('products')
    dept = prod.set_index('product_id')['department']
    br = prod.set_index('product_id')['brand']
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
    m84b = m84.copy(); m84b['wk'] = m84b['day'] // 7
    f['nwk84'] = m84b.groupby('household_key')['wk'].nunique().reindex(hh).fillna(0.0)
    out = f.reset_index()
    out.insert(1, 'snapshot_day', snapshot_day)
    return out

t0 = time.time()
tbl = agent_api.build_features(full_fn)
print("built:", tbl.shape, f"{time.time()-t0:.1f}s")
p = agent_api.save_table(tbl, 'e002_mix.parquet')
print(p)
