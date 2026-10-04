import agent_api as A, pandas as pd, numpy as np

tt = A.train_targets()
print(tt.future_spend_4w.describe())
print('zero share', (tt.future_spend_4w==0).mean(), 'median', tt.future_spend_4w.median())

def rhythm(view, snapshot_day):
    hh = pd.Index(view.households)
    out = pd.DataFrame(index=hh)
    tx = view.transactions
    tx = tx[tx.household_key.isin(set(hh))]
    if len(tx)==0:
        return out
    # basket level
    b = tx.groupby('basket_id', as_index=False, sort=False).agg(
        h=('household_key','first'), day=('day','first'), spend=('sales_value','sum'),
        store=('store_id','first'), ttime=('trans_time','first'))
    b['dow'] = b.day % 7
    b84  = b[b.day > snapshot_day-84]
    b56  = b[b.day > snapshot_day-56]
    b112 = b[b.day > snapshot_day-112]
    # gaps between basket days (112d)
    bs = b112.sort_values(['h','day'])
    bs = bs.assign(gap=bs.groupby('h')['day'].diff())
    g = bs.groupby('h')['gap']
    out['rs_gap_mean'] = g.mean(); out['rs_gap_std'] = g.std(); out['rs_gap_max'] = g.max()
    out['rs_gap_p90'] = g.quantile(0.9)
    # basket value stats (84d)
    g = b84.groupby('h')['spend']
    out['rs_bmax_84'] = g.max(); out['rs_bstd_84'] = g.std()
    m = g.mean(); s = g.std()
    out['rs_bcv_84'] = s/m
    # active days
    ad = b84.groupby('h')['day'].nunique()
    out['rs_active_share_84'] = ad/84.0
    span = b84.groupby('h')['day'].agg(lambda x: (x.max()-x.min())//7+1)
    nb = b84.groupby('h').size()
    out['rs_bpw_act'] = nb/span.replace(0,np.nan)
    # day-of-week concentration (56d)
    def ent(x):
        c = x.value_counts(); p = c/c.sum(); return -(p*np.log(p)).sum()
    dd = b56.groupby('h')['dow']
    out['rs_dow_ent_56'] = dd.apply(ent)
    out['rs_dow_top_56'] = dd.apply(lambda x: x.value_counts(normalize=True).iloc[0])
    # time of day (84d)
    out['rs_tt_mean_84'] = b84.groupby('h')['ttime'].mean()
    out['rs_tt_am_84'] = b84.groupby('h')['ttime'].apply(lambda x:(x<1200).mean())
    # longest daily streak (56d)
    def streak(x):
        d = np.unique(x.values)
        if len(d)==0: return 0
        seg = np.split(d, np.where(np.diff(d)>1)[0]+1)
        return max(len(s) for s in seg)
    out['rs_streak_56'] = b56.groupby('h')['day'].apply(streak)
    # zero-basket weeks out of last 12
    b84 = b84.assign(wk=(snapshot_day-b84.day)//7)
    out['rs_zero_wk'] = 12 - b84.groupby('h')['wk'].nunique()
    # store loyalty (84d)
    st = b84.groupby(['h','store'])['spend'].sum().reset_index()
    tot = st.groupby('h')['spend'].sum()
    st['sh'] = st.spend/st.tot
    top = st.sort_values('spend', ascending=False).groupby('h', sort=False).head(2)
    out['sl_top_share'] = top.groupby('h')['sh'].apply(lambda x: x.iloc[0])
    out['sl_top2_share'] = top.groupby('h')['sh'].sum()
    out['sl_nstores'] = st.groupby('h').size()
    out['sl_store_ent'] = st.groupby('h')['spend'].apply(lambda x: (lambda p: -(p*np.log(p)).sum())(x/x.sum()))
    out['sl_top_spend'] = top.groupby('h')['spend'].apply(lambda x: x.iloc[0])
    # deal / discount intensity (84d, line level)
    t = tx[tx.day > snapshot_day-84].copy()
    d = -(t.retail_disc + t.coupon_disc + t.coupon_match_disc).clip(lower=0)
    cp = (-(t.coupon_disc + t.coupon_match_disc)).clip(lower=0)
    t = t.assign(dd=d, cp=cp, deal=d>0, spend_deal=np.where(d>0, t.sales_value, 0.0))
    agg = t.groupby('household_key').agg(sp=('sales_value','sum'), dd=('dd','sum'), cp=('cp','sum'),
                                         n=('sales_value','size'), ndeal=('deal','sum'), spd=('spend_deal','sum'))
    den = (agg.sp+agg.dd).replace(0,np.nan)
    out['dl_disc_rate'] = agg.dd/den
    out['dl_cp_rate'] = agg.cp/den
    out['dl_item_deal_share'] = agg.ndeal/agg.n
    out['dl_deal_spend'] = agg.spd
    bd = t.groupby('basket_id')['dd'].max()
    b84 = b84.assign(hasd=b84.basket_id.map((bd>0).astype(float)))
    out['dl_basket_deal_share'] = b84.groupby('h')['hasd'].mean()
    return out

res = A.build_features(rhythm)
print('rhythm block:', res.shape, 'nan frac', res.isna().mean().mean().round(3))
A.save_table(res, 'rhythm_v1.parquet')
base = A.load_saved('e009_demo.parquet')
m = base.merge(res.reset_index().rename(columns={'index':'household_key'}) if res.index.name is None else res.reset_index(),
               on=['household_key','snapshot_day'], how='inner')
print('merged:', m.shape)
A.save_table(m, 'e010_rhythm.parquet')
print('saved')