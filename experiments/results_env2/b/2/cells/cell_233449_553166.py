def timing_fn(view, snapshot_day):
    hh = view.households
    if hasattr(hh, 'columns') and 'household_key' in getattr(hh, 'columns', []):
        idx = pd.Index(hh.household_key)
    else:
        idx = pd.Index(hh)
    tr = view.table('transactions')
    tr = tr[tr.day <= snapshot_day]
    tr = tr[tr.day > snapshot_day - 112]
    out = pd.DataFrame(index=idx)
    if len(tr) == 0:
        return out
    tr = tr.assign(hour=(tr.trans_time // 100).clip(0, 23).astype(float),
                   dow=(tr.day % 7).astype(int))
    # basket-level style
    bs = tr.groupby(['household_key', 'basket_id']).agg(
        sp=('sales_value', 'sum'), hour=('hour', 'first'), dow=('dow', 'first')).reset_index()
    g = bs.groupby('household_key')
    out['t_hour_mean'] = g.hour.mean()
    out['t_hour_std'] = g.hour.std()
    # spend shares by time of day
    sp_tot = tr.groupby('household_key').sales_value.sum().rename('tot')
    m = tr[(tr.hour >= 7) & (tr.hour <= 11)].groupby('household_key').sales_value.sum()
    e = tr[(tr.hour >= 17) & (tr.hour <= 22)].groupby('household_key').sales_value.sum()
    out['t_morning_sp_share'] = (m / sp_tot)
    out['t_evening_sp_share'] = (e / sp_tot)
    # day-of-week concentration of spend
    dow_sp = tr.groupby(['household_key', 'dow']).sales_value.sum()
    share = dow_sp / dow_sp.groupby('household_key').transform('sum')
    out['t_dow_hhi'] = (share ** 2).groupby('household_key').sum()
    out['t_n_dow'] = tr[tr.sales_value > 0].groupby('household_key').dow.nunique()
    # store breadth in window
    out['t_n_stores'] = tr.groupby('household_key').store_id.nunique()
    st = tr.groupby(['household_key', 'store_id']).sales_value.sum().reset_index()
    top = st.sort_values('sales_value', ascending=False).groupby('household_key').head(1).set_index('household_key')
    out['t_store_share'] = top.sales_value / sp_tot
    return out.reindex(idx)

bt = build_features(timing_fn)
print(bt.shape)
print(bt.head(3))
print(bt.isna().mean().round(3).to_dict())
save_table(bt, 'timing_v1')
print('saved timing_v1')
