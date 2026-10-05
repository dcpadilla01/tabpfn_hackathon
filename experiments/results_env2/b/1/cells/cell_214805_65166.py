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
