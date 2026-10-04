import numpy as np, pandas as pd

base = agent_api.load_saved('e013_stock.parquet')
print("base", base.shape)

def fn(view, snapshot_day):
    s = snapshot_day; w = (s+8)//7
    idx = view.households
    out = pd.DataFrame(index=idx)
    phase = (w-1)%52+1
    out['phase_sin'] = np.sin(2*np.pi*phase/52.0)
    out['phase_cos'] = np.cos(2*np.pi*phase/52.0)
    out['phase_cat'] = 'p%02d' % phase
    out['year_frac'] = s/364.0
    t = view.transactions
    tw = t.assign(wk=(t.day+8)//7).groupby('wk').agg(sp=('sales_value','sum'),
                                                     bk=('basket_id','nunique'),
                                                     hh=('household_key','nunique'))
    tw['perhh'] = tw.sp/tw.hh
    tw['bkphh'] = tw.bk/tw.hh
    def mkt(col, weeks, minc):
        v = tw.reindex(pd.Index(weeks))[col]
        return v.mean() if v.notna().sum() >= minc else np.nan
    out['mkt_spend4']  = mkt('perhh', range(w-4, w), 3)
    out['mkt_spend12'] = mkt('perhh', range(w-12, w), 8)
    out['mkt_bask4']   = mkt('bkphh', range(w-4, w), 3)
    out['mkt_all']     = tw.perhh.mean()
    a4 = tw.reindex(pd.Index(range(w-4, w)))['perhh']
    a48 = tw.reindex(pd.Index(range(w-52, w-4)))['perhh']
    out['mkt4_vs_all'] = out.mkt_spend4/out.mkt_all
    out['mkt4_vs_48'] = (a4.mean()/a48.mean()) if (a4.notna().sum()>=3 and a48.notna().sum()>=24) else np.nan
    # market same-phase-last-year 4-week block (complete weeks w-56..w-53)
    ly = tw.reindex(pd.Index(range(w-56, w-52)))['perhh']
    out['mkt_ly4'] = ly.mean() if ly.notna().sum()>=3 else np.nan
    out['mkt_ly_ratio'] = out.mkt_spend4/out.mkt_ly4 if out.mkt_ly4 is not None else np.nan
    # active share of cohort
    act4 = t[t.day >= s-27].household_key.nunique()
    out['mkt_active_share'] = act4/len(idx)
    # household aligned lag-13 block (364 days back)
    if s-391 >= 1:
        t13 = t[(t.day >= s-391) & (t.day <= s-364)].groupby('household_key').sales_value.sum()
        out['p13'] = t13.reindex(idx).fillna(0.0)
        t28 = t[t.day >= s-27].groupby('household_key').sales_value.sum().reindex(idx).fillna(0.0)
        out['p13_vs_now'] = out.p13/(t28+1.0)
    else:
        out['p13'] = np.nan; out['p13_vs_now'] = np.nan
    return out

blk = agent_api.build_features(fn)
print("blk", blk.shape)
m = base.merge(blk.reset_index(), on=['household_key','snapshot_day'], how='inner')
print("merged", m.shape, "ncols", len(m.columns))
print(m[['mkt_spend4','mkt_spend12','mkt4_vs_48','mkt_ly_ratio','mkt_active_share','p13']].groupby(m.snapshot_day).mean().round(3).to_string())
path = agent_api.save_table(m, 'e014_season.parquet')
print(path)