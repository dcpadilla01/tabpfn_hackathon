import agent_api as api
import pandas as pd, numpy as np

base = api.load_saved('e006_composition.parquet')
print('base:', base.shape)

def compute(view, s):
    try:
        hh = pd.Index(view.households)
    except Exception:
        hh = pd.Index(view.households['household_key'])
    tx = view.transactions
    tx = tx.assign(lagk=(s - tx['day']) // 28)
    tx = tx[tx['lagk'].between(0, 13)]
    sp = tx.groupby(['household_key', 'lagk'])['sales_value'].sum().unstack(fill_value=0.0)
    sp = sp.reindex(columns=range(14), fill_value=0.0).reindex(hh, fill_value=0.0)
    tr = tx.groupby(['household_key', 'lagk'])['basket_id'].nunique().unstack(fill_value=0.0)
    tr = tr.reindex(columns=range(14), fill_value=0.0).reindex(hh, fill_value=0.0)
    fd = tx.groupby('household_key')['day'].min().reindex(hh).fillna(s + 1)

    f = pd.DataFrame(index=hh)
    S = {}
    for k in range(1, 14):
        col = sp[k].where(fd <= s - 27 - 28 * k)   # NaN before household tenure covers window
        S[k] = col
        f[f'lag_spend_{k}'] = col
    allsp = pd.concat([S[k] for k in range(1, 14)], axis=1)
    f['lag_mean_1_2'] = (S[1] + S[2]) / 2.0
    f['lag_mean_3_4'] = (S[3] + S[4]) / 2.0
    f['lag_mean_1_4'] = allsp.iloc[:, 0:4].mean(axis=1)
    f['lag_mean_5_8'] = allsp.iloc[:, 4:8].mean(axis=1)
    f['lag_mean_9_13'] = allsp.iloc[:, 8:13].mean(axis=1)
    f['lag_std_1_13'] = allsp.std(axis=1)
    f['lag_max_1_13'] = allsp.max(axis=1)
    f['lag_min_1_13'] = allsp.min(axis=1)
    f['lag_nonzero_cnt_1_13'] = (allsp > 0).sum(axis=1)
    zmat = (allsp <= 0) & allsp.notna()
    streak = np.zeros(len(f)); run = np.ones(len(f), dtype=bool)
    for j in range(13):
        zk = zmat.iloc[:, j].values & run
        streak = streak + zk
        run = zk
    f['lag_zero_streak'] = streak
    f['lag_trend_12_34'] = (S[1] + S[2]) / 2.0 - (S[3] + S[4]) / 2.0
    f['lag_ratio_1_2'] = S[1] / (S[2] + 1.0)
    f['lag_ratio_1_13'] = S[1] / (S[13] + 1.0)
    f['lag1_is_zero'] = (S[1] <= 0)
    trm = pd.concat([tr[k].where(fd <= s - 27 - 28 * k) for k in (1, 2, 3, 4)], axis=1)
    f['lag_trips_2'] = tr[2].where(fd <= s - 27 - 56)
    f['lag_trips_mean_1_4'] = trm.mean(axis=1)
    return f

full = api.build_features(compute)
print('full:', full.shape)

tt = api.train_targets()
chk = tt.merge(full, on=['household_key', 'snapshot_day'])
print('corr with target on train rows:')
for c in ['lag_spend_1','lag_spend_2','lag_spend_13','lag_mean_1_4','lag_mean_5_8','lag_mean_9_13',
          'lag_trend_12_34','lag_zero_streak','lag_ratio_1_2','lag_nonzero_cnt_1_13','lag_std_1_13']:
    print('  ', c, round(chk[c].corr(chk['future_spend_4w']), 3))

overlap = [c for c in full.columns if c in base.columns and c not in ('household_key', 'snapshot_day')]
print('overlapping cols:', overlap)
if overlap:
    full = full.rename(columns={c: 'ls_' + c for c in overlap})
merged = base.merge(full, on=['household_key', 'snapshot_day'], how='left')
print('merged:', merged.shape, 'dups:', merged.duplicated(['household_key', 'snapshot_day']).sum())
path = api.save_table(merged, 'e007_lagseq')
print('saved:', path)