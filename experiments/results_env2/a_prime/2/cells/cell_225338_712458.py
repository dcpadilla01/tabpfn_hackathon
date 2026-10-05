
import agent_api, pandas as pd, numpy as np

def fn(view, snapshot_day):
    hh = view.households
    hh_idx = pd.Index(hh)
    tx = view.table('transactions')
    tx = tx[tx.household_key.isin(hh_idx)]
    day = tx['day'].values
    out = pd.DataFrame(index=hh_idx)

    def spend_in(lo, hi):
        m = (day > lo) & (day <= hi)
        if not m.any():
            return pd.Series(0.0, index=hh_idx)
        s = tx.loc[m].groupby('household_key')['sales_value'].sum()
        return s.reindex(hh_idx).fillna(0.0)

    def baskets_in(lo, hi):
        m = (day > lo) & (day <= hi)
        if not m.any():
            return pd.Series(0.0, index=hh_idx)
        s = tx.loc[m].groupby('household_key')['basket_id'].nunique()
        return s.reindex(hh_idx).fillna(0.0)

    # multi-lag 28d spend sequence (lag-2 and lag-3 windows)
    out['spend_28_prior2'] = spend_in(snapshot_day-84, snapshot_day-56)
    out['spend_28_prior3'] = spend_in(snapshot_day-112, snapshot_day-84)

    # trip-frequency trend: last-28d trips vs 84d run-rate
    b28 = baskets_in(snapshot_day-28, snapshot_day)
    b84 = baskets_in(snapshot_day-84, snapshot_day)
    out['trip_freq_ratio'] = b28 / (b84*(28.0/84.0) + 0.5)

    # typical (median) 28d-window spend over trailing 182d
    first_day = tx.groupby('household_key')['day'].min().reindex(hh_idx)
    wins = [spend_in(snapshot_day-28*k, snapshot_day-28*(k-1)) for k in range(1, 7)]
    W = pd.concat(wins, axis=1)
    starts = np.array([snapshot_day-28*k for k in range(1, 7)])
    mask = starts[None, :] >= first_day.values[:, None]
    W = W.where(pd.DataFrame(mask, index=hh_idx, columns=W.columns))
    out['med28_182'] = W.median(axis=1, skipna=True)

    # household-specific seasonal index: historical spend share in the upcoming window's week-of-year slots
    wk = tx['week_no'].values
    wy = (wk - 1) % 52
    twys = np.array(sorted({(((dd + 8)//7) - 1) % 52 for dd in range(snapshot_day+1, snapshot_day+29)}))
    m = np.isin(wy, twys)
    sm = tx.loc[m].groupby('household_key')['sales_value'].sum().reindex(hh_idx).fillna(0.0)
    sa = tx.groupby('household_key')['sales_value'].sum().reindex(hh_idx).fillna(0.0)
    span = (snapshot_day - first_day)
    idx = sm / (sa * (len(twys)/52.0)).replace(0, np.nan)
    out['hh_seasonal_index'] = idx.where(span >= 180)

    if snapshot_day <= 431:
        print(snapshot_day, out.describe().loc[['mean','50%']].round(3).to_dict())
    return out

t = agent_api.build_features(fn)
print(t.shape, t.columns.tolist())
print(t.isna().mean().round(3).to_string())
p = agent_api.save_table(t, 'e009_ar_season.parquet')
print(p)
