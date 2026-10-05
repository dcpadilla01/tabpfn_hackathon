import numpy as np, pandas as pd, time

def make_fn(verbose=False):
    def fn(view, snapshot_day):
        t0 = time.time()
        def log(msg):
            if verbose: print(f'[{snapshot_day}] {msg} t={time.time()-t0:.1f}s', flush=True)
        hh = view.households
        log(f'hh type={type(hh).__name__} len={len(hh) if hasattr(hh,"__len__") else "?"}')
        if isinstance(hh, pd.DataFrame):
            hh_idx = pd.Index(hh['household_key'].unique())
        else:
            hh_idx = pd.Index(pd.unique(np.asarray(hh).ravel()))
        out = pd.DataFrame(index=hh_idx)
        tx = view.table('transactions')
        log(f'tx rows={len(tx)}')
        tx = tx[['household_key','basket_id','day','sales_value']].copy()
        trips = tx[['household_key','basket_id','day']].drop_duplicates()
        log('trips')

        out['cs_days_since_trip'] = snapshot_day - trips.groupby('household_key')['day'].max()
        g = trips.sort_values(['household_key','day'])
        days = g['day'].to_numpy(float)
        gaps = days - g.groupby('household_key')['day'].shift().to_numpy()
        ok = ~np.isnan(gaps)
        med_gap_all = pd.Series(gaps[ok]).groupby(g['household_key'].to_numpy()[ok]).median()
        rec = trips[trips['day'] > snapshot_day - 84]
        g2 = rec.sort_values(['household_key','day'])
        days2 = g2['day'].to_numpy(float)
        gaps2 = days2 - g2.groupby('household_key')['day'].shift().to_numpy()
        ok2 = ~np.isnan(gaps2)
        med_gap_84 = pd.Series(gaps2[ok2]).groupby(g2['household_key'].to_numpy()[ok2]).median()
        mg = med_gap_84.reindex(med_gap_all.index).fillna(med_gap_all)
        out['cs_med_gap_84'] = med_gap_84
        out['cs_med_gap_all'] = med_gap_all
        out['cs_exp_trips_28'] = (28.0 / mg).clip(upper=28.0)
        sp84 = tx[tx['day'] > snapshot_day - 84].groupby('household_key')['sales_value'].sum()
        tr84 = rec.groupby('household_key')['basket_id'].nunique().replace(0, np.nan)
        spt = sp84 / tr84
        out['cs_spend_per_trip_84'] = spt
        out['cs_pred_cadence'] = spt * (28.0 / mg).clip(upper=28.0)
        out['cs_recency_ratio'] = out['cs_days_since_trip'] / mg
        log('cadence')

        tx['wk'] = (tx['day'] - 1) // 7
        wk_max = (snapshot_day - 1) // 7
        weeks = np.arange(wk_max - 11, wk_max + 1)
        w = tx.groupby(['household_key','wk'])['sales_value'].sum().unstack(fill_value=0.0)
        w = w.reindex(columns=weeks, fill_value=0.0).reindex(hh_idx, fill_value=0.0)
        x = np.arange(12, dtype=float); den = ((x - x.mean())**2).sum()
        yv = w.to_numpy(float)
        out['cs_slope_sp_12w'] = (yv - yv.mean(1, keepdims=True)) @ (x - x.mean()) / den
        tw = tx[['household_key','basket_id','wk']].drop_duplicates().groupby(['household_key','wk']).size().unstack(fill_value=0)
        tw = tw.reindex(columns=weeks, fill_value=0).reindex(hh_idx, fill_value=0)
        ytv = tw.to_numpy(float)
        out['cs_slope_tr_12w'] = (ytv - ytv.mean(1, keepdims=True)) @ (x - x.mean()) / den
        log('weekly')

        tx['bin'] = (snapshot_day - tx['day']) // 28
        b = tx[tx['bin'] < 6].groupby(['household_key','bin'])['sales_value'].sum().unstack(fill_value=0.0)
        b = b.reindex(columns=range(6), fill_value=0.0).reindex(hh_idx, fill_value=0.0)
        bv = b.to_numpy(float)
        out['cs_w28_med'] = np.median(bv, axis=1)
        out['cs_w28_max'] = bv.max(1)
        out['cs_w28_min'] = bv.min(1)
        m_ = bv.mean(1); s_ = bv.std(1)
        out['cs_w28_cv'] = np.where(m_ > 0, s_ / np.maximum(m_, 1e-9), np.nan)
        out['cs_w28_zero'] = (bv == 0).sum(1).astype(float)
        sp28 = b[0]
        log('windows')

        def lag_win(lo_off, hi_off, name):
            lo, hi = snapshot_day - lo_off, snapshot_day - hi_off
            if lo < 1:
                out[name] = np.nan
            else:
                s = tx[(tx['day'] >= lo) & (tx['day'] <= hi)].groupby('household_key')['sales_value'].sum()
                out[name] = s.reindex(hh_idx).fillna(0.0)
        lag_win(363, 336, 'cs_sp_ly364')
        lag_win(335, 308, 'cs_sp_lag336')
        out['cs_r_sp28_ly'] = sp28 / (out['cs_sp_ly364'].fillna(0.0) + 1.0)
        r3 = tx[tx['day'] > snapshot_day - 3]
        out['cs_sp3'] = r3.groupby('household_key')['sales_value'].sum().reindex(hh_idx).fillna(0.0)
        out['cs_trips3'] = r3.groupby('household_key')['basket_id'].nunique().reindex(hh_idx).fillna(0.0)
        log('lags')

        tx['woy'] = tx['wk'] % 52
        t_woys = set(int(((d - 1) // 7) % 52) for d in range(snapshot_day + 1, snapshot_day + 29))
        mwoy = tx[tx['woy'].isin(t_woys)].groupby(['household_key','wk'])['sales_value'].sum()
        agg = mwoy.groupby('household_key').agg(['mean','max','count'])
        out['cs_sp_woy_mean'] = agg['mean']
        out['cs_sp_woy_max'] = agg['max']
        out['cs_n_woy_weeks'] = agg['count'].astype(float)
        out['cs_r_recent_woy'] = sp28 / (out['cs_sp_woy_mean'].fillna(0.0) + 1.0)
        log('woy')
        return out
    return fn

fn = make_fn(verbose=True)
v459 = agent_api.snapshot(459)
t0 = time.time()
res = fn(v459, 459)
print('direct 459 OK:', res.shape, f'{time.time()-t0:.1f}s')
print(res.dtypes.value_counts())
print('index sample:', res.index[:3].tolist(), type(res.index[0]))
num = res.to_numpy(dtype=float)
print('inf:', np.isinf(num).sum(), 'all-nan cols:', [c for c in res.columns if res[c].isna().all()])
