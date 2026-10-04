import agent_api, pandas as pd, numpy as np

def make_feats(view, snapshot_day):
    tx = view.transactions
    hh = sorted(set(view.households))
    idx = pd.Index(hh, name='household_key')
    g = tx.groupby('household_key')
    first = g['day'].min().reindex(hh)
    lt_spend = g['sales_value'].sum().reindex(hh)
    lt_trips = g['basket_id'].nunique().reindex(hh)
    ten = (snapshot_day - first + 1).clip(lower=1)
    out = pd.DataFrame(index=idx)
    out['lt_spend_per_day'] = lt_spend/ten
    out['lt_trips_per_day'] = lt_trips/ten
    out['lt_spend_per_trip'] = lt_spend/lt_trips.clip(lower=1)
    tx364 = tx[tx.day > snapshot_day-364]
    g3 = tx364.groupby('household_key')
    sp3 = g3['sales_value'].sum().reindex(hh).fillna(0.0)
    tr3 = g3['basket_id'].nunique().reindex(hh).fillna(0.0)
    out['spend_per_day364'] = sp3/364.0
    out['spend_per_trip364'] = sp3/tr3.clip(lower=1)
    # weekly CV over last 52 weeks (zero weeks included)
    w0 = snapshot_day//7
    weeks = np.arange(max(w0-51,0), w0+1)
    tw = tx.assign(wk=(tx.day//7).astype(int))
    wsum = tw[tw.wk.isin(weeks)].groupby(['household_key','wk'])['sales_value'].sum().unstack(fill_value=0.0)
    wsum = wsum.reindex(index=hh, columns=weeks).fillna(0.0)
    mu = wsum.mean(axis=1); sd = wsum.std(axis=1)
    out['cv_weekly_52'] = np.where(mu>0, sd/np.maximum(mu,1e-9), np.nan)
    # 4-week windows over past year
    L = {}
    for k in range(1,14):
        hi = snapshot_day-28*(k-1); lo = snapshot_day-28*k
        msk = (tx.day>lo)&(tx.day<=hi)
        L[k] = tx[msk].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
    Ldf = pd.DataFrame(L)
    m1 = Ldf.mean(axis=1); s1 = Ldf.std(axis=1)
    out['cv_window_1y'] = np.where(m1>0, s1/np.maximum(m1,1e-9), np.nan)
    # gaps / overdue
    last = g['day'].max().reindex(hh)
    rec = (snapshot_day-last).astype(float)
    tx112 = tx[tx.day > snapshot_day-112]
    bd = tx112.groupby('household_key')['day'].apply(lambda s: np.sort(pd.unique(s.values)))
    def gs(a):
        if a is None or len(a)<2: return (np.nan, np.nan)
        d = np.diff(a); return (float(d.max()), float(np.median(d)))
    g2 = bd.apply(gs)
    gmed = g2.apply(lambda t: t[1])
    out['max_gap_112'] = g2.apply(lambda t: t[0])
    out['overdue_days'] = rec - gmed
    out['overdue_ratio'] = rec/(gmed.fillna(3)+1.0)
    # top store health
    ts112 = tx112.groupby(['household_key','store_id']).size().rename('n').reset_index()
    topmap = ts112.loc[ts112.groupby('household_key')['n'].idxmax()].set_index('household_key')
    tot = ts112.groupby('household_key')['n'].sum()
    out['top_store_share112'] = (topmap['n']/tot)
    store_last = tx.groupby('store_id')['day'].max()
    dsa = snapshot_day - topmap['store_id'].map(store_last)
    out['days_since_top_store_active'] = dsa
    out['top_store_active14'] = (dsa<=14).astype(float)
    # recent-week share of 28d spend
    s7 = tx[tx.day > snapshot_day-7].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
    s28 = tx[tx.day > snapshot_day-28].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
    out['fw_recent7_share'] = s7/(s28+1.0)
    return out

tab = agent_api.build_features(make_feats)
print('built:', tab.shape)
newcols = [c for c in tab.columns if c not in ('household_key','snapshot_day')]
print(tab[newcols].describe().round(3).T[['mean','std','min','max']])

e11 = agent_api.load_saved('e011_price.parquet')
m = e11.merge(tab, on=['household_key','snapshot_day'], how='inner')
print('after basestab merge:', m.shape)

nf = agent_api.load_saved('newfeat.parquet')
print('newfeat:', nf.shape, list(nf.columns))
ok = False
if nf.shape[0]==len(e11) and set(['household_key','snapshot_day']).issubset(nf.columns):
    nfc = [c for c in nf.columns if c not in ('household_key','snapshot_day')]
    if not set(nfc) & set(m.columns):
        m = m.merge(nf, on=['household_key','snapshot_day'], how='inner'); ok=True
print('after newfeat merge:', m.shape, 'merged_newfeat:', ok)
print('dups:', int(m.duplicated(['household_key','snapshot_day']).sum()))

tt = agent_api.train_targets()
mm = m.merge(tt, on=['household_key','snapshot_day'])
tr = mm.snapshot_day<=403
for c in newcols:
    print('corr', c, round(float(mm.loc[tr,c].corr(mm.loc[tr,'future_spend_4w'])),3))
path = agent_api.save_table(m, 'e018_basestab.parquet')
print('PATH:', path)
