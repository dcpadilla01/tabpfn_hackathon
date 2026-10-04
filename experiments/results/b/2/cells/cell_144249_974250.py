import pandas as pd, numpy as np

def build(v, snap):
    s = snap
    idx = pd.Index(list(v.households), name='household_key')
    tx = v.transactions
    F = pd.DataFrame(index=idx)
    day = tx['day'].values; sv = tx['sales_value'].values; hhv = tx['household_key'].values

    def wsum(lo, hi, name):
        m = (day >= lo) & (day <= hi)
        g = tx.loc[m].groupby('household_key')['sales_value'].sum()
        F[name] = g.reindex(idx).fillna(0.0)

    # weekly spend series, last 8 weeks (wk0 = current week)
    wk = (v.week - tx['week_no']).values
    m = wk <= 7
    tmp = pd.DataFrame({'hh': hhv[m], 'wk': wk[m], 'sv': sv[m]})
    pv = tmp.groupby(['hh','wk'])['sv'].sum().unstack().reindex(columns=range(8)).reindex(idx).fillna(0.0)
    for i in range(8): F['wk%d' % i] = pv[i]
    arr = pv.values
    F['wk_avg_4'] = arr[:, :4].mean(1)
    F['wk_avg_8'] = arr.mean(1)
    F['wk_std_8'] = arr.std(1)
    x = np.arange(8); xm = 3.5
    F['wk_slope'] = ((arr - arr.mean(1, keepdims=True)) * (x - xm)).sum(1) / ((x - xm) ** 2).sum()
    F['n_active_weeks_8'] = (arr > 0).sum(1)

    # monthly 28-day windows w0..w13 (w0 = [s-27,s])
    wi = (s - day) // 28
    m = wi <= 13
    tmp = pd.DataFrame({'hh': hhv[m], 'wi': wi[m], 'sv': sv[m]})
    pw = tmp.groupby(['hh','wi'])['sv'].sum().unstack().reindex(columns=range(14)).reindex(idx).fillna(0.0)
    for k in range(3, 9): F['w%d' % k] = pw[k]
    Z = (pw.loc[:, range(1, 14)].values == 0)
    F['zero_frac_13'] = Z.mean(1)
    nz = ~Z
    F['zero_streak'] = np.where(nz.all(1), 13, nz.argmax(1))

    # seasonality: same 4-week window one year earlier
    wsum(s - 363, s - 336, 'spend_same_ly')
    F['ratio_seas'] = pw[0] / (F['spend_same_ly'] + 10.0)

    wsum(s - 6, s, 'spend_7'); wsum(s - 13, s, 'spend_14'); wsum(s - 20, s, 'spend_21')
    wsum(s - 167, s, 'spend_168'); wsum(s - 251, s, 'spend_252'); wsum(s - 503, s, 'spend_504')

    for wlen, tag in [(84, '84'), (168, '168')]:
        m = day > s - wlen
        t = tx.loc[m]
        F['trips_%s' % tag] = t.groupby('household_key')['basket_id'].nunique().reindex(idx).fillna(0.0)
        F['active_days_%s' % tag] = t.groupby('household_key')['day'].nunique().reindex(idx).fillna(0.0)

    # gaps between purchase days
    dd = tx[['household_key','day']].drop_duplicates().sort_values(['household_key','day'])
    t2 = dd.groupby('household_key').tail(2)
    d2 = t2.groupby('household_key')['day'].diff()
    sub = t2.assign(_d=d2.values)
    gp = sub[sub['_d'].notna()].set_index('household_key')['_d']
    F['gap_prev'] = gp.reindex(idx)
    m84 = day > s - 84
    dd84 = tx.loc[m84, ['household_key','day']].drop_duplicates().sort_values(['household_key','day'])
    di = dd84.groupby('household_key')['day'].diff()
    tmp = dd84.assign(_d=di.values)
    gm = tmp[tmp['_d'].notna()].groupby('household_key')['_d'].mean()
    F['gap_mean_84'] = gm.reindex(idx)
    return F

out = agent_api.build_features(build)
print('build done', out.shape)
e1 = agent_api.load_saved('e001_recent_spend.parquet')
print('e1 dtypes:', e1[['household_key','snapshot_day']].dtypes.values, '| out dtypes:',
      out[['household_key','snapshot_day']].dtypes.values)
for c in ['household_key','snapshot_day']:
    e1[c] = e1[c].astype('int64'); out[c] = out[c].astype('int64')
mg = out.merge(e1, on=['household_key','snapshot_day'], how='inner')
print('merged', mg.shape)
print('sanity w0 vs spend_28 maxdiff', np.abs(mg['w0'] if 'w0' in mg else 0).max() if 'w0' in mg else 'n/a')
print('cols:', list(mg.columns))
print(mg[['spend_28','spend_lag1','wk0','wk1','w3','zero_frac_13','zero_streak','gap_prev','spend_same_ly','ratio_seas']].describe().T[['mean','std','min','max']])
path = agent_api.save_table(mg, 'e004_temporal')
print('saved', path)