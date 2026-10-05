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
