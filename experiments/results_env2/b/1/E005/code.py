import numpy as np, pandas as pd

base = agent_api.load_saved('e003_full.parquet')
print('base shape:', base.shape, '| n feature cols:', base.shape[1]-2)

def fn(view, snapshot_day):
    hh = view.households
    if isinstance(hh, pd.DataFrame):
        hh_idx = pd.Index(pd.unique(hh['household_key']))
    elif isinstance(hh, (pd.Series, pd.Index)):
        hh_idx = pd.Index(pd.unique(hh))
    else:
        hh_idx = pd.Index(pd.unique(np.asarray(hh).ravel()))
    out = pd.DataFrame(index=hh_idx)
    tx = view.table('transactions')
    if len(tx) == 0:
        return out
    tx = tx[['household_key','basket_id','day','sales_value']]
    trips = tx[['household_key','basket_id','day']].drop_duplicates()

    # --- cadence ---
    out['cs_days_since_trip'] = snapshot_day - trips.groupby('household_key')['day'].max()
    def med_gap(s):
        d = np.sort(np.asarray(s, dtype=float))
        return float(np.median(np.diff(d))) if d.size >= 2 else np.nan
    gap_all = trips.groupby('household_key')['day'].apply(med_gap)
    rec = trips[trips['day'] > snapshot_day - 84]
    gap_84 = rec.groupby('household_key')['day'].apply(med_gap)
    mg = gap_84.fillna(gap_all)
    sp84 = tx[tx['day'] > snapshot_day - 84].groupby('household_key')['sales_value'].sum()
    tr84 = rec.groupby('household_key')['basket_id'].nunique().replace(0, np.nan)
    spt = sp84 / tr84
    et = (28.0 / mg).clip(upper=28.0)
    out['cs_med_gap_84'] = gap_84
    out['cs_med_gap_all'] = gap_all
    out['cs_exp_trips_28'] = et
    out['cs_spend_per_trip_84'] = spt
    out['cs_pred_cadence'] = spt * et
    out['cs_recency_ratio'] = out['cs_days_since_trip'] / mg

    # --- weekly trend ---
    t2 = tx.copy()
    t2['wk'] = (t2['day'] - 1) // 7
    wk_max = (snapshot_day - 1) // 7
    weeks = np.arange(wk_max - 11, wk_max + 1)
    w = t2.groupby(['household_key','wk'])['sales_value'].sum().unstack(fill_value=0.0)
    w = w.reindex(columns=weeks, fill_value=0.0).reindex(hh_idx, fill_value=0.0)
    x = np.arange(12, dtype=float); den = ((x - x.mean())**2).sum()
    yv = w.to_numpy(float)
    out['cs_slope_sp_12w'] = (yv - yv.mean(1, keepdims=True)) @ (x - x.mean()) / den
    tw = t2[['household_key','basket_id','wk']].drop_duplicates().groupby(['household_key','wk']).size().unstack(fill_value=0)
    tw = tw.reindex(columns=weeks, fill_value=0).reindex(hh_idx, fill_value=0)
    ytv = tw.to_numpy(float)
    out['cs_slope_tr_12w'] = (ytv - ytv.mean(1, keepdims=True)) @ (x - x.mean()) / den

    # --- last 6 non-overlapping 28d windows ---
    t2['bin'] = (snapshot_day - t2['day']) // 28
    b = t2[t2['bin'] < 6].groupby(['household_key','bin'])['sales_value'].sum().unstack(fill_value=0.0)
    b = b.reindex(columns=range(6), fill_value=0.0).reindex(hh_idx, fill_value=0.0)
    bv = b.to_numpy(float)
    out['cs_w28_med'] = np.median(bv, axis=1)
    out['cs_w28_max'] = bv.max(1)
    out['cs_w28_min'] = bv.min(1)
    m_ = bv.mean(1); s_ = bv.std(1)
    out['cs_w28_cv'] = np.where(m_ > 0, s_ / np.maximum(m_, 1e-9), np.nan)
    out['cs_w28_zero'] = (bv == 0).sum(1).astype(float)
    sp28 = b[0]

    # --- same 4-week window one year (364d) / 336d earlier ---
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

    # last 3 days
    r3 = tx[tx['day'] > snapshot_day - 3]
    out['cs_sp3'] = r3.groupby('household_key')['sales_value'].sum().reindex(hh_idx).fillna(0.0)
    out['cs_trips3'] = r3.groupby('household_key')['basket_id'].nunique().reindex(hh_idx).fillna(0.0)

    # --- week-of-year seasonal profile ---
    t2['woy'] = t2['wk'] % 52
    t_woys = set(int(((d - 1) // 7) % 52) for d in range(snapshot_day + 1, snapshot_day + 29))
    mwoy = t2[t2['woy'].isin(t_woys)].groupby(['household_key','wk'])['sales_value'].sum()
    out['cs_sp_woy_mean'] = mwoy.groupby('household_key').mean()
    out['cs_sp_woy_max'] = mwoy.groupby('household_key').max()
    out['cs_n_woy_weeks'] = mwoy.groupby('household_key').count().astype(float)
    out['cs_r_recent_woy'] = sp28 / (out['cs_sp_woy_mean'].fillna(0.0) + 1.0)
    return out

feats = agent_api.build_features(fn)
print('feats shape:', feats.shape)
new_cols = [c for c in feats.columns if str(c).startswith('cs_')]
print('n new cols:', len(new_cols))
print('NaN fraction per new feature:')
print(feats[new_cols].isna().mean().round(3).to_string())

key = ['household_key','snapshot_day']
merged = base.merge(feats, on=key, how='inner')
print('merged:', merged.shape)
assert len(merged) == len(base)

tt = agent_api.train_targets()
diag = merged.merge(tt, on=key)
corr = diag[new_cols].corrwith(diag['future_spend_4w']).sort_values()
print('train-row correlation with target:')
print(corr.round(3).to_string())

path = agent_api.save_table(merged, 'e005_cadence_seasonal')
print('saved:', path)


# ---- cell ----
import numpy as np, pandas as pd

base = agent_api.load_saved('e003_full.parquet')
print('base shape:', base.shape)

def fn(view, snapshot_day):
    hh = view.households
    hh_idx = pd.Index(pd.unique(np.asarray(hh).ravel()))
    out = pd.DataFrame(index=hh_idx)
    tx = view.table('transactions')
    if len(tx) == 0:
        return out
    tx = tx[['household_key','basket_id','day','sales_value']].copy()
    trips = tx[['household_key','basket_id','day']].drop_duplicates()

    out['cs_days_since_trip'] = snapshot_day - trips.groupby('household_key')['day'].max()

    # median gap via vectorized sort per group
    g = trips.sort_values(['household_key','day'])
    d1 = g.groupby('household_key')['day']
    days = g['day'].to_numpy(float)
    prev = d1.shift()
    gaps = (days - prev.to_numpy())
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

    t2 = tx.copy()
    t2['wk'] = (t2['day'] - 1) // 7
    wk_max = (snapshot_day - 1) // 7
    weeks = np.arange(wk_max - 11, wk_max + 1)
    w = t2.groupby(['household_key','wk'])['sales_value'].sum().unstack(fill_value=0.0)
    w = w.reindex(columns=weeks, fill_value=0.0).reindex(hh_idx, fill_value=0.0)
    x = np.arange(12, dtype=float); den = ((x - x.mean())**2).sum()
    yv = w.to_numpy(float)
    out['cs_slope_sp_12w'] = (yv - yv.mean(1, keepdims=True)) @ (x - x.mean()) / den
    tw = t2[['household_key','basket_id','wk']].drop_duplicates().groupby(['household_key','wk']).size().unstack(fill_value=0)
    tw = tw.reindex(columns=weeks, fill_value=0).reindex(hh_idx, fill_value=0)
    ytv = tw.to_numpy(float)
    out['cs_slope_tr_12w'] = (ytv - ytv.mean(1, keepdims=True)) @ (x - x.mean()) / den

    t2['bin'] = (snapshot_day - t2['day']) // 28
    b = t2[t2['bin'] < 6].groupby(['household_key','bin'])['sales_value'].sum().unstack(fill_value=0.0)
    b = b.reindex(columns=range(6), fill_value=0.0).reindex(hh_idx, fill_value=0.0)
    bv = b.to_numpy(float)
    out['cs_w28_med'] = np.median(bv, axis=1)
    out['cs_w28_max'] = bv.max(1)
    out['cs_w28_min'] = bv.min(1)
    m_ = bv.mean(1); s_ = bv.std(1)
    out['cs_w28_cv'] = np.where(m_ > 0, s_ / np.maximum(m_, 1e-9), np.nan)
    out['cs_w28_zero'] = (bv == 0).sum(1).astype(float)
    sp28 = b[0]

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

    t2['woy'] = t2['wk'] % 52
    t_woys = set(int(((d - 1) // 7) % 52) for d in range(snapshot_day + 1, snapshot_day + 29))
    mwoy = t2[t2['woy'].isin(t_woys)].groupby(['household_key','wk'])['sales_value'].sum()
    agg = mwoy.groupby('household_key').agg(['mean','max','count'])
    out['cs_sp_woy_mean'] = agg[('sales_value','mean')]
    out['cs_sp_woy_max'] = agg[('sales_value','max')]
    out['cs_n_woy_weeks'] = agg[('sales_value','count')].astype(float)
    out['cs_r_recent_woy'] = sp28 / (out['cs_sp_woy_mean'].fillna(0.0) + 1.0)
    return out

feats = agent_api.build_features(fn)
print('feats shape:', feats.shape)
new_cols = [c for c in feats.columns if str(c).startswith('cs_')]
print('n new cols:', len(new_cols))
print(feats[new_cols].isna().mean().round(3).to_string())

key = ['household_key','snapshot_day']
merged = base.merge(feats.reset_index().rename(columns={'index':'household_key'}), on=key, how='inner')
print('merged:', merged.shape)
assert len(merged) == len(base)

tt = agent_api.train_targets()
diag = merged.merge(tt, on=key)
corr = diag[new_cols].corrwith(diag['future_spend_4w']).sort_values()
print('corr with target:')
print(corr.round(3).to_string())

path = agent_api.save_table(merged, 'e005_cadence_seasonal')
print('saved:', path)


# ---- cell ----
import numpy as np, pandas as pd

base = agent_api.load_saved('e003_full.parquet')
print('base shape:', base.shape)

def fn(view, snapshot_day):
    hh = view.households
    hh_idx = pd.Index(pd.unique(np.asarray(hh).ravel()))
    out = pd.DataFrame(index=hh_idx)
    tx = view.table('transactions')
    if len(tx) == 0:
        return out
    tx = tx[['household_key','basket_id','day','sales_value']].copy()
    trips = tx[['household_key','basket_id','day']].drop_duplicates()

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

    t2 = tx.copy()
    t2['wk'] = (t2['day'] - 1) // 7
    wk_max = (snapshot_day - 1) // 7
    weeks = np.arange(wk_max - 11, wk_max + 1)
    w = t2.groupby(['household_key','wk'])['sales_value'].sum().unstack(fill_value=0.0)
    w = w.reindex(columns=weeks, fill_value=0.0).reindex(hh_idx, fill_value=0.0)
    x = np.arange(12, dtype=float); den = ((x - x.mean())**2).sum()
    yv = w.to_numpy(float)
    out['cs_slope_sp_12w'] = (yv - yv.mean(1, keepdims=True)) @ (x - x.mean()) / den
    tw = t2[['household_key','basket_id','wk']].drop_duplicates().groupby(['household_key','wk']).size().unstack(fill_value=0)
    tw = tw.reindex(columns=weeks, fill_value=0).reindex(hh_idx, fill_value=0)
    ytv = tw.to_numpy(float)
    out['cs_slope_tr_12w'] = (ytv - ytv.mean(1, keepdims=True)) @ (x - x.mean()) / den

    t2['bin'] = (snapshot_day - t2['day']) // 28
    b = t2[t2['bin'] < 6].groupby(['household_key','bin'])['sales_value'].sum().unstack(fill_value=0.0)
    b = b.reindex(columns=range(6), fill_value=0.0).reindex(hh_idx, fill_value=0.0)
    bv = b.to_numpy(float)
    out['cs_w28_med'] = np.median(bv, axis=1)
    out['cs_w28_max'] = bv.max(1)
    out['cs_w28_min'] = bv.min(1)
    m_ = bv.mean(1); s_ = bv.std(1)
    out['cs_w28_cv'] = np.where(m_ > 0, s_ / np.maximum(m_, 1e-9), np.nan)
    out['cs_w28_zero'] = (bv == 0).sum(1).astype(float)
    sp28 = b[0]

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

    t2['woy'] = t2['wk'] % 52
    t_woys = set(int(((d - 1) // 7) % 52) for d in range(snapshot_day + 1, snapshot_day + 29))
    mwoy = t2[t2['woy'].isin(t_woys)].groupby(['household_key','wk'])['sales_value'].sum()
    agg = mwoy.groupby('household_key').agg(['mean','max','count'])
    out['cs_sp_woy_mean'] = agg['mean']
    out['cs_sp_woy_max'] = agg['max']
    out['cs_n_woy_weeks'] = agg['count'].astype(float)
    out['cs_r_recent_woy'] = sp28 / (out['cs_sp_woy_mean'].fillna(0.0) + 1.0)
    return out

feats = agent_api.build_features(fn)
print('feats shape:', feats.shape)
new_cols = [c for c in feats.columns if str(c).startswith('cs_')]
print('n new cols:', len(new_cols))
print(feats[new_cols].isna().mean().round(3).to_string())

key = ['household_key','snapshot_day']
merged = base.merge(feats.reset_index().rename(columns={'index':'household_key'}), on=key, how='inner')
print('merged:', merged.shape)
assert len(merged) == len(base)

tt = agent_api.train_targets()
diag = merged.merge(tt, on=key)
corr = diag[new_cols].corrwith(diag['future_spend_4w']).sort_values()
print('corr with target:')
print(corr.round(3).to_string())

path = agent_api.save_table(merged, 'e005_cadence_seasonal')
print('saved:', path)


# ---- cell ----
import numpy as np, pandas as pd, time

base = agent_api.load_saved('e003_full.parquet')
print('base shape:', base.shape)

def make_fn(verbose=False):
    def fn(view, snapshot_day):
        t0 = time.time()
        def log(msg):
            if verbose: print(f'[{snapshot_day}] {msg} t={time.time()-t0:.1f}s')
        hh = view.households
        log(f'households type={type(hh).__name__} repr={repr(hh)[:120]}')
        if isinstance(hh, pd.DataFrame):
            hh_idx = pd.Index(hh['household_key'].unique())
        else:
            hh_idx = pd.Index(pd.unique(np.asarray(hh).ravel()))
        out = pd.DataFrame(index=hh_idx)
        tx = view.table('transactions')
        log(f'tx rows={len(tx)}')
        tx = tx[['household_key','basket_id','day','sales_value']].copy()
        trips = tx[['household_key','basket_id','day']].drop_duplicates()
        log('trips done')

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
        log('cadence done')

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
        log('weekly done')

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
        log('windows done')

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
        log('lags done')

        tx['woy'] = tx['wk'] % 52
        t_woys = set(int(((d - 1) // 7) % 52) for d in range(snapshot_day + 1, snapshot_day + 29))
        mwoy = tx[tx['woy'].isin(t_woys)].groupby(['household_key','wk'])['sales_value'].sum()
        agg = mwoy.groupby('household_key').agg(['mean','max','count'])
        out['cs_sp_woy_mean'] = agg['mean']
        out['cs_sp_woy_max'] = agg['max']
        out['cs_n_woy_weeks'] = agg['count'].astype(float)
        out['cs_r_recent_woy'] = sp28 / (out['cs_sp_woy_mean'].fillna(0.0) + 1.0)
        log('woy done')
        return out
    return fn

fn = make_fn(verbose=True)
v543 = agent_api.snapshot(543)
t0 = time.time()
res = fn(v543, 543)
print('direct 543 OK:', res.shape, f'{time.time()-t0:.1f}s')
print(res.dtypes.value_counts())
print('index sample:', res.index[:3].tolist(), type(res.index[0]))
print('inf count:', np.isinf(res.to_numpy(float)).sum() if res.shape[1] else 0)


# ---- cell ----
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


# ---- cell ----
import numpy as np, pandas as pd

base = agent_api.load_saved('e003_full.parquet')
print('base shape:', base.shape)

def fn(view, snapshot_day):
    tx = view.table('transactions')
    tx = tx[['household_key','basket_id','day','sales_value']].copy()
    hh = view.households
    if hh is None:
        first = tx.groupby('household_key')['day'].min()
        hh_idx = pd.Index(first[first <= snapshot_day - 84].index)
    elif isinstance(hh, pd.DataFrame):
        hh_idx = pd.Index(hh['household_key'].unique())
    else:
        hh_idx = pd.Index(pd.unique(np.asarray(hh).ravel()))
    out = pd.DataFrame(index=hh_idx)
    trips = tx[['household_key','basket_id','day']].drop_duplicates()

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

    tx['woy'] = tx['wk'] % 52
    t_woys = set(int(((d - 1) // 7) % 52) for d in range(snapshot_day + 1, snapshot_day + 29))
    mwoy = tx[tx['woy'].isin(t_woys)].groupby(['household_key','wk'])['sales_value'].sum()
    agg = mwoy.groupby('household_key').agg(['mean','max','count'])
    out['cs_sp_woy_mean'] = agg['mean']
    out['cs_sp_woy_max'] = agg['max']
    out['cs_n_woy_weeks'] = agg['count'].astype(float)
    out['cs_r_recent_woy'] = sp28 / (out['cs_sp_woy_mean'].fillna(0.0) + 1.0)
    return out

feats = agent_api.build_features(fn)
print('feats shape:', feats.shape)
new_cols = [c for c in feats.columns if str(c).startswith('cs_')]
print('n new cols:', len(new_cols))
print(feats[new_cols].isna().mean().round(3).to_string())

key = ['household_key','snapshot_day']
merged = base.merge(feats.reset_index().rename(columns={'index':'household_key'}), on=key, how='inner')
print('merged:', merged.shape)
assert len(merged) == len(base)

tt = agent_api.train_targets()
diag = merged.merge(tt, on=key)
corr = diag[new_cols].corrwith(diag['future_spend_4w']).sort_values()
print('corr with target:')
print(corr.round(3).to_string())

path = agent_api.save_table(merged, 'e005_cadence_seasonal')
print('saved:', path)


# ---- cell ----
import pandas as pd, numpy as np
def fn(view, snapshot_day):
    tx = view.table('transactions')
    hh = view.households
    print('day', snapshot_day, 'tx rows', len(tx), 'hh type', type(hh).__name__, flush=True)
    if hh is None:
        first = tx.groupby('household_key')['day'].min()
        idx = pd.Index(first[first <= snapshot_day - 84].index)
    elif isinstance(hh, pd.DataFrame):
        idx = pd.Index(hh['household_key'].unique())
    else:
        idx = pd.Index(pd.unique(np.asarray(hh).ravel()))
    return pd.DataFrame({'cs_test': 1.0}, index=idx)

feats = agent_api.build_features(fn)
print('trivial OK:', feats.shape)


# ---- cell ----
import numpy as np, pandas as pd
def fn(view, snapshot_day):
    print('day', snapshot_day, 'start', flush=True)
    a = np.zeros(50_000_000)  # 400MB
    print('  alloc 400MB ok', flush=True)
    b = np.zeros(190_000_000)  # ~1.5GB
    b[0] = 1.0
    print('  alloc 1.5GB ok', flush=True)
    del a, b
    tx = view.table('transactions')
    return pd.DataFrame({'cs_test': float(len(tx))}, index=pd.Index(['A'], name='household_key'))

feats = agent_api.build_features(fn)
print('memtest OK:', feats.shape)


# ---- cell ----
import numpy as np, pandas as pd

def fn(view, snapshot_day):
    tx = view.table('transactions')
    tx = tx[['household_key','basket_id','day','sales_value']].copy()
    hh = view.households
    if hh is None:
        first = tx.groupby('household_key')['day'].min()
        hh_idx = pd.Index(first[first <= snapshot_day - 84].index)
    elif isinstance(hh, pd.DataFrame):
        hh_idx = pd.Index(hh['household_key'].unique())
    else:
        hh_idx = pd.Index(pd.unique(np.asarray(hh).ravel()))
    out = pd.DataFrame(index=hh_idx)
    trips = tx[['household_key','basket_id','day']].drop_duplicates()
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
    tx['woy'] = tx['wk'] % 52
    t_woys = set(int(((d - 1) // 7) % 52) for d in range(snapshot_day + 1, snapshot_day + 29))
    mwoy = tx[tx['woy'].isin(t_woys)].groupby(['household_key','wk'])['sales_value'].sum()
    agg = mwoy.groupby('household_key').agg(['mean','max','count'])
    out['cs_sp_woy_mean'] = agg['mean']
    out['cs_sp_woy_max'] = agg['max']
    out['cs_n_woy_weeks'] = agg['count'].astype(float)
    out['cs_r_recent_woy'] = sp28 / (out['cs_sp_woy_mean'].fillna(0.0) + 1.0)
    return out

for d in [95, 123, 151, 179, 207, 235, 263, 291, 319, 347, 375, 403, 431, 459]:
    v = agent_api.snapshot(d)
    res = fn(v, d)
    print(d, 'OK', res.shape, flush=True)


# ---- cell ----
import numpy as np, pandas as pd

def fn(view, snapshot_day):
    print(f'fn start day={snapshot_day}', flush=True)
    tx = view.table('transactions')
    tx = tx[['household_key','basket_id','day','sales_value']].copy()
    hh = view.households
    if hh is None:
        first = tx.groupby('household_key')['day'].min()
        hh_idx = pd.Index(first[first <= snapshot_day - 84].index)
    elif isinstance(hh, pd.DataFrame):
        hh_idx = pd.Index(hh['household_key'].unique())
    else:
        hh_idx = pd.Index(pd.unique(np.asarray(hh).ravel()))
    out = pd.DataFrame(index=hh_idx)
    trips = tx[['household_key','basket_id','day']].drop_duplicates()
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
    tx['woy'] = tx['wk'] % 52
    t_woys = set(int(((d - 1) // 7) % 52) for d in range(snapshot_day + 1, snapshot_day + 29))
    mwoy = tx[tx['woy'].isin(t_woys)].groupby(['household_key','wk'])['sales_value'].sum()
    agg = mwoy.groupby('household_key').agg(['mean','max','count'])
    out['cs_sp_woy_mean'] = agg['mean']
    out['cs_sp_woy_max'] = agg['max']
    out['cs_n_woy_weeks'] = agg['count'].astype(float)
    out['cs_r_recent_woy'] = sp28 / (out['cs_sp_woy_mean'].fillna(0.0) + 1.0)
    print(f'fn end day={snapshot_day} shape={out.shape}', flush=True)
    return out

feats = agent_api.build_features(fn)
print('OK:', feats.shape)


# ---- cell ----
import numpy as np, pandas as pd

def fn(view, snapshot_day):
    P = lambda m: print(f'  [{snapshot_day}] {m}', flush=True)
    P('begin')
    tx = view.table('transactions')
    tx = tx[['household_key','basket_id','day','sales_value']].copy()
    hh = view.households
    if hh is None:
        first = tx.groupby('household_key')['day'].min()
        hh_idx = pd.Index(first[first <= snapshot_day - 84].index)
    else:
        hh_idx = pd.Index(pd.unique(np.asarray(hh).ravel()))
    out = pd.DataFrame(index=hh_idx)
    P('tx ready')
    trips = tx[['household_key','basket_id','day']].drop_duplicates()
    P('trips')
    out['cs_days_since_trip'] = snapshot_day - trips.groupby('household_key')['day'].max()
    P('last_trip')
    g = trips.sort_values(['household_key','day'])
    days = g['day'].to_numpy(float)
    gaps = days - g.groupby('household_key')['day'].shift().to_numpy()
    ok = ~np.isnan(gaps)
    med_gap_all = pd.Series(gaps[ok]).groupby(g['household_key'].to_numpy()[ok]).median()
    P('med_gap_all')
    rec = trips[trips['day'] > snapshot_day - 84]
    g2 = rec.sort_values(['household_key','day'])
    days2 = g2['day'].to_numpy(float)
    gaps2 = days2 - g2.groupby('household_key')['day'].shift().to_numpy()
    ok2 = ~np.isnan(gaps2)
    med_gap_84 = pd.Series(gaps2[ok2]).groupby(g2['household_key'].to_numpy()[ok2]).median()
    P('med_gap_84')
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
    P('cadence')
    tx['wk'] = (tx['day'] - 1) // 7
    wk_max = (snapshot_day - 1) // 7
    weeks = np.arange(wk_max - 11, wk_max + 1)
    w = tx.groupby(['household_key','wk'])['sales_value'].sum().unstack(fill_value=0.0)
    P('weekly unstack')
    w = w.reindex(columns=weeks, fill_value=0.0).reindex(hh_idx, fill_value=0.0)
    x = np.arange(12, dtype=float); den = ((x - x.mean())**2).sum()
    yv = w.to_numpy(float)
    out['cs_slope_sp_12w'] = (yv - yv.mean(1, keepdims=True)) @ (x - x.mean()) / den
    tw = tx[['household_key','basket_id','wk']].drop_duplicates().groupby(['household_key','wk']).size().unstack(fill_value=0)
    P('trips weekly unstack')
    tw = tw.reindex(columns=weeks, fill_value=0).reindex(hh_idx, fill_value=0)
    ytv = tw.to_numpy(float)
    out['cs_slope_tr_12w'] = (ytv - ytv.mean(1, keepdims=True)) @ (x - x.mean()) / den
    P('weekly done')
    tx['bin'] = (snapshot_day - tx['day']) // 28
    b = tx[tx['bin'] < 6].groupby(['household_key','bin'])['sales_value'].sum().unstack(fill_value=0.0)
    P('28d unstack')
    b = b.reindex(columns=range(6), fill_value=0.0).reindex(hh_idx, fill_value=0.0)
    bv = b.to_numpy(float)
    out['cs_w28_med'] = np.median(bv, axis=1)
    out['cs_w28_max'] = bv.max(1)
    out['cs_w28_min'] = bv.min(1)
    m_ = bv.mean(1); s_ = bv.std(1)
    out['cs_w28_cv'] = np.where(m_ > 0, s_ / np.maximum(m_, 1e-9), np.nan)
    out['cs_w28_zero'] = (bv == 0).sum(1).astype(float)
    sp28 = b[0]
    P('windows done')
    def lag_win(lo_off, hi_off, name):
        lo, hi = snapshot_day - lo_off, snapshot_day - hi_off
        if lo < 1:
            out[name] = np.nan
        else:
            s = tx[(tx['day'] >= lo) & (tx['day'] <= hi)].groupby('household_key')['sales_value'].sum()
            out[name] = s.reindex(hh_idx).fillna(0.0)
    lag_win(363, 336, 'cs_sp_ly364')
    P('lag364')
    lag_win(335, 308, 'cs_sp_lag336')
    P('lag336')
    out['cs_r_sp28_ly'] = sp28 / (out['cs_sp_ly364'].fillna(0.0) + 1.0)
    r3 = tx[tx['day'] > snapshot_day - 3]
    out['cs_sp3'] = r3.groupby('household_key')['sales_value'].sum().reindex(hh_idx).fillna(0.0)
    out['cs_trips3'] = r3.groupby('household_key')['basket_id'].nunique().reindex(hh_idx).fillna(0.0)
    P('lags done')
    tx['woy'] = tx['wk'] % 52
    t_woys = set(int(((d - 1) // 7) % 52) for d in range(snapshot_day + 1, snapshot_day + 29))
    mwoy = tx[tx['woy'].isin(t_woys)].groupby(['household_key','wk'])['sales_value'].sum()
    P('woy groupby')
    agg = mwoy.groupby('household_key').agg(['mean','max','count'])
    out['cs_sp_woy_mean'] = agg['mean']
    out['cs_sp_woy_max'] = agg['max']
    out['cs_n_woy_weeks'] = agg['count'].astype(float)
    out['cs_r_recent_woy'] = sp28 / (out['cs_sp_woy_mean'].fillna(0.0) + 1.0)
    P('done')
    return out

feats = agent_api.build_features(fn)
print('OK:', feats.shape)


# ---- cell ----
import numpy as np, pandas as pd

def fn(view, snapshot_day):
    P = lambda m: print(f'  [{snapshot_day}] {m}', flush=True)
    if snapshot_day != 207:
        tx0 = view.table('transactions')
        first = tx0.groupby('household_key')['day'].min()
        idx = pd.Index(first[first <= snapshot_day - 84].index)
        return pd.DataFrame({'cs_probe': 1.0}, index=idx)
    tx = view.table('transactions')
    tx = tx[['household_key','basket_id','day','sales_value']].copy()
    P(f'tx {tx.shape} dtypes {dict(tx.dtypes.astype(str))}')
    trips = tx[['household_key','basket_id','day']].drop_duplicates()
    P('trips')
    g = trips.sort_values(['household_key','day'])
    P('sorted')
    days = g['day'].to_numpy(float)
    gaps = days - g.groupby('household_key')['day'].shift().to_numpy()
    ok = ~np.isnan(gaps)
    med = pd.Series(gaps[ok]).groupby(g['household_key'].to_numpy()[ok]).median()
    P(f'med_gap n={len(med)}')
    w = tx.groupby(['household_key','wk' if 'wk' in tx else 'day'])['sales_value'].sum()
    P('groupby sum')
    # the exact op that died:
    tx['wk'] = (tx['day'] - 1) // 7
    w = tx.groupby(['household_key','wk'])['sales_value'].sum().unstack(fill_value=0.0)
    P(f'unstack {w.shape}')
    return pd.DataFrame({'cs_probe': w.sum(axis=1)})

feats = agent_api.build_features(fn)
print('OK:', feats.shape)


# ---- cell ----
import numpy as np, pandas as pd

base = agent_api.load_saved('e003_full.parquet')
print('base:', base.shape, flush=True)

def fn(view, snapshot_day):
    d = snapshot_day
    hh_idx = pd.Index(view.households)
    tx = view.table('transactions')
    t = tx[['household_key', 'day', 'sales_value', 'basket_id']]
    wmax = (d - 1) // 7
    weeks = np.arange(max(0, wmax - 52), wmax + 1)
    wk = ((t['day'] - 1) // 7).astype(np.int32)

    # weekly spend matrix (small)
    sw = t.groupby(['household_key', wk])['sales_value'].sum()
    swu = sw.unstack(fill_value=0.0).reindex(columns=weeks, fill_value=0.0).reindex(hh_idx, fill_value=0.0)
    M = swu.to_numpy(float)

    # baskets per trip-day -> trips per week
    nb = t.groupby(['household_key', 'day'])['basket_id'].nunique()
    nh = nb.index.get_level_values(0); nd = nb.index.get_level_values(1)
    wk_nb = ((nd - 1) // 7).astype(np.int32)
    tw = pd.Series(nb.to_numpy(), index=[nh, wk_nb]).groupby(level=[0, 1]).sum()
    twu = tw.unstack(fill_value=0.0).reindex(columns=weeks, fill_value=0).reindex(hh_idx, fill_value=0)
    TV = twu.to_numpy(float)

    out = pd.DataFrame(index=hh_idx)
    x = np.arange(12, dtype=float); den = ((x - x.mean()) ** 2).sum()
    yv = M[:, -12:]
    out['cs_slope_sp_12w'] = (yv - yv.mean(1, keepdims=True)) @ (x - x.mean()) / den
    ytv = TV[:, -12:]
    out['cs_slope_tr_12w'] = (ytv - ytv.mean(1, keepdims=True)) @ (x - x.mean()) / den

    # 6 non-overlapping 28d bins (4 weeks each)
    B = np.stack([M[:, -(4 * (b + 1)):-(4 * b) if b > 0 else None].sum(1) for b in range(6)], axis=1)
    out['cs_w28_med'] = np.median(B, axis=1)
    out['cs_w28_max'] = B.max(1)
    out['cs_w28_min'] = B.min(1)
    m_ = B.mean(1); s_ = B.std(1)
    out['cs_w28_cv'] = np.where(m_ > 0, s_ / np.maximum(m_, 1e-9), np.nan)
    out['cs_w28_zero'] = (B == 0).sum(1).astype(float)
    sp28 = pd.Series(B[:, 0], index=hh_idx)

    def week_sum(lo_off, hi_off):
        lo_w, hi_w = (d - lo_off) // 7, (d - hi_off) // 7
        cols = [w for w in weeks if lo_w <= w <= hi_w]
        return M[:, [list(weeks).index(w) for w in cols]].sum(1) if cols else np.full(len(hh_idx), np.nan)
    ly = week_sum(363, 336)
    out['cs_sp_ly364'] = ly
    out['cs_sp_lag336'] = week_sum(335, 308)
    out['cs_r_sp28_ly'] = sp28 / (pd.Series(ly, index=hh_idx).fillna(0.0) + 1.0)

    # week-of-year seasonal profile
    woys = np.array([w % 52 for w in weeks])
    tset = set(int(((dd - 1) // 7) % 52) for dd in range(d + 1, d + 29))
    sel = np.array([w in tset for w in woys])
    if sel.any():
        S = M[:, sel]
        cnt = (S > 0).sum(1).astype(float)
        out['cs_sp_woy_mean'] = np.where(cnt > 0, S.sum(1) / np.maximum(cnt, 1), np.nan)
        out['cs_sp_woy_max'] = S.max(1)
        out['cs_n_woy_weeks'] = cnt
    else:
        out['cs_sp_woy_mean'] = np.nan; out['cs_sp_woy_max'] = np.nan; out['cs_n_woy_weeks'] = 0.0
    out['cs_r_recent_woy'] = sp28 / (out['cs_sp_woy_mean'].fillna(0.0) + 1.0)

    # cadence from trip-days
    tdf = pd.DataFrame({'hh': nh.to_numpy(), 'day': nd.to_numpy()}).sort_values(['hh', 'day'])
    last = tdf.groupby('hh')['day'].max()
    out['cs_days_since_trip'] = (d - last).reindex(hh_idx)
    def med_gap(sub):
        g = sub.groupby('hh')['day']
        gaps = sub['day'].to_numpy(float) - g.shift().to_numpy()
        ok = ~np.isnan(gaps)
        if ok.sum() == 0:
            return pd.Series(dtype=float)
        return pd.Series(gaps[ok]).groupby(sub['hh'].to_numpy()[ok]).median()
    mg_all = med_gap(tdf)
    rec = tdf[tdf['day'] > d - 84]
    mg_84 = med_gap(rec)
    mg = mg_84.reindex(mg_all.index).fillna(mg_all)
    out['cs_med_gap_84'] = mg_84.reindex(hh_idx)
    out['cs_med_gap_all'] = mg_all.reindex(hh_idx)
    out['cs_exp_trips_28'] = (28.0 / mg.reindex(hh_idx)).clip(upper=28.0)
    sp84 = t[t['day'] > d - 84].groupby('household_key')['sales_value'].sum().reindex(hh_idx).fillna(0.0)
    tr84 = rec.groupby('hh')['day'].count().reindex(hh_idx).replace(0, np.nan)
    spt = sp84 / tr84
    out['cs_spend_per_trip_84'] = spt
    out['cs_pred_cadence'] = spt * (28.0 / mg.reindex(hh_idx)).clip(upper=28.0)
    out['cs_recency_ratio'] = out['cs_days_since_trip'] / mg.reindex(hh_idx)
    out['cs_trips3'] = tdf[tdf['day'] > d - 3].groupby('hh')['day'].count().reindex(hh_idx).fillna(0.0)
    out['cs_sp3'] = t[t['day'] > d - 3].groupby('household_key')['sales_value'].sum().reindex(hh_idx).fillna(0.0)
    del sw, swu, tw, twu, tdf, rec, nb
    return out

feats = agent_api.build_features(fn)
print('feats:', feats.shape, flush=True)
new_cols = [c for c in feats.columns if c not in ('household_key', 'snapshot_day')]
print('n new:', len(new_cols), 'nan max:', feats[new_cols].isna().mean().max().round(3) if hasattr(feats[new_cols].isna().mean().max(), 'round') else feats[new_cols].isna().mean().max())

merged = base.merge(feats, on=['household_key', 'snapshot_day'], how='inner')
assert len(merged) == len(base), (len(merged), len(base))
path = agent_api.save_table(merged, 'e005_cadence_seasonal')
print('saved:', path)


# ---- cell ----
import numpy as np, pandas as pd

base = agent_api.load_saved('e003_full.parquet')
print('base:', base.shape, flush=True)

def fn(view, d):
    P = lambda m: print(f'[{d}] {m}', flush=True)
    hh_idx = pd.Index(view.households)
    tx = view.table('transactions')
    P('tx')
    wmax = (d - 1) // 7
    weeks = np.arange(max(0, wmax - 52), wmax + 1)
    wk = ((tx['day'] - 1) // 7).astype(np.int32)
    sw = tx.groupby(['household_key', wk])['sales_value'].sum()
    swu = sw.unstack(fill_value=0.0).reindex(columns=weeks, fill_value=0.0).reindex(hh_idx, fill_value=0.0)
    M = swu.to_numpy(float)
    del sw, swu
    P('weekly matrix')
    td = tx[['household_key', 'day', 'basket_id']].drop_duplicates()
    tdd = td[['household_key', 'day']].drop_duplicates().sort_values(['household_key', 'day'])
    P('tripdays')
    gaps = tdd.groupby('household_key')['day'].diff().to_numpy()
    hh_arr = tdd['household_key'].to_numpy()
    ok = ~np.isnan(gaps)
    mg_all = pd.Series(gaps[ok]).groupby(hh_arr[ok]).median()
    sub = tdd[tdd['day'] > d - 84]
    g84 = sub.groupby('household_key')['day'].diff().to_numpy()
    ok2 = ~np.isnan(g84)
    mg_84 = pd.Series(g84[ok2]).groupby(sub['household_key'].to_numpy()[ok2]).median()
    mg = mg_84.reindex(mg_all.index).fillna(mg_all)
    P('med gaps')
    out = pd.DataFrame(index=hh_idx)
    out['cs_med_gap_84'] = mg_84.reindex(hh_idx)
    out['cs_med_gap_all'] = mg_all.reindex(hh_idx)
    out['cs_exp_trips_28'] = (28.0 / mg.reindex(hh_idx)).clip(upper=28.0)
    last = tdd.groupby('household_key')['day'].max()
    out['cs_days_since_trip'] = d - last.reindex(hh_idx)
    sp84 = tx[tx['day'] > d - 84].groupby('household_key')['sales_value'].sum().reindex(hh_idx).fillna(0.0)
    tr84 = sub.groupby('household_key')['day'].count().reindex(hh_idx).replace(0, np.nan)
    spt = sp84 / tr84
    out['cs_spend_per_trip_84'] = spt
    out['cs_pred_cadence'] = spt * (28.0 / mg.reindex(hh_idx)).clip(upper=28.0)
    out['cs_recency_ratio'] = out['cs_days_since_trip'] / mg.reindex(hh_idx)
    del sub
    P('cadence')
    x = np.arange(12, dtype=float); den = ((x - x.mean()) ** 2).sum()
    yv = M[:, -12:]
    out['cs_slope_sp_12w'] = (yv - yv.mean(1, keepdims=True)) @ (x - x.mean()) / den
    td2 = td.assign(wkc=((td['day'] - 1) // 7).astype(np.int32))
    tw = td2.groupby(['household_key', 'wkc'])['day'].count()
    twu = tw.unstack(fill_value=0).reindex(columns=weeks, fill_value=0).reindex(hh_idx, fill_value=0)
    TV = twu.to_numpy(float)
    ytv = TV[:, -12:]
    out['cs_slope_tr_12w'] = (ytv - ytv.mean(1, keepdims=True)) @ (x - x.mean()) / den
    del td, td2, tw, twu, TV
    P('slopes')
    B = np.stack([M[:, -(4 * (b + 1)):-(4 * b) if b > 0 else None].sum(1) for b in range(6)], axis=1)
    out['cs_w28_med'] = np.median(B, axis=1)
    out['cs_w28_max'] = B.max(1)
    out['cs_w28_min'] = B.min(1)
    m_ = B.mean(1); s_ = B.std(1)
    out['cs_w28_cv'] = np.where(m_ > 0, s_ / np.maximum(m_, 1e-9), np.nan)
    out['cs_w28_zero'] = (B == 0).sum(1).astype(float)
    sp28 = pd.Series(B[:, 0], index=hh_idx)
    P('bins')
    wpos = {int(w): i for i, w in enumerate(weeks)}
    def week_sum(lo_off, hi_off):
        lo_w, hi_w = (d - lo_off) // 7, (d - hi_off) // 7
        cols = [wpos[w] for w in range(int(weeks[0]), int(weeks[-1]) + 1) if lo_w <= w <= hi_w]
        return M[:, cols].sum(1) if cols else np.full(len(hh_idx), np.nan)
    ly = week_sum(363, 336)
    out['cs_sp_ly364'] = ly
    out['cs_sp_lag336'] = week_sum(335, 308)
    out['cs_r_sp28_ly'] = sp28 / (pd.Series(ly, index=hh_idx).fillna(0.0) + 1.0)
    P('lags')
    woys = np.array([w % 52 for w in weeks])
    tset = set(int(((dd - 1) // 7) % 52) for dd in range(d + 1, d + 29))
    selv = np.array([w in tset for w in woys])
    if selv.any():
        S = M[:, selv]
        cnt = (S > 0).sum(1).astype(float)
        out['cs_sp_woy_mean'] = np.where(cnt > 0, S.sum(1) / np.maximum(cnt, 1), np.nan)
        out['cs_sp_woy_max'] = S.max(1)
        out['cs_n_woy_weeks'] = cnt
    else:
        out['cs_sp_woy_mean'] = np.nan
        out['cs_sp_woy_max'] = np.nan
        out['cs_n_woy_weeks'] = 0.0
    out['cs_r_recent_woy'] = sp28 / (out['cs_sp_woy_mean'].fillna(0.0) + 1.0)
    r3 = tdd[tdd['day'] > d - 3]
    out['cs_trips3'] = r3.groupby('household_key')['day'].count().reindex(hh_idx).fillna(0.0)
    out['cs_sp3'] = tx[tx['day'] > d - 3].groupby('household_key')['sales_value'].sum().reindex(hh_idx).fillna(0.0)
    P('done')
    return out

feats = agent_api.build_features(fn)
print('feats:', feats.shape, flush=True)
new_cols = [c for c in feats.columns if c not in ('household_key', 'snapshot_day')]
print('n new:', len(new_cols), '| max NaN frac:', float(feats[new_cols].isna().mean().max()))

merged = base.merge(feats, on=['household_key', 'snapshot_day'], how='inner')
assert len(merged) == len(base), (len(merged), len(base))
path = agent_api.save_table(merged, 'e005_cadence_seasonal')
print('saved:', path)
