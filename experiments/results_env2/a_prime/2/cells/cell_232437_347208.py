import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
t12 = A.load_saved('e012_hh_target_enc.parquet')
t11 = A.load_saved('e011_discounts.parquet')
tt = A.train_targets()
print('t12', t12.shape, 't11', t11.shape)
new = [c for c in t12.columns if c not in t11.columns]
print('new cols:', new)
m = t12.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431]; va = m[m.snapshot_day>431]
print('train rows', len(tr), 'val rows', len(va))
for c in new:
    print('\n--', c)
    print(' train: nan%%=%.1f mean=%.2f std=%.2f' % (tr[c].isna().mean()*100, tr[c].mean(), tr[c].std()))
    print(' val  : nan%%=%.1f mean=%.2f std=%.2f' % (va[c].isna().mean()*100, va[c].mean(), va[c].std()))
    v = tr[c].values; y = tr['future_spend_4w'].values
    ok = ~np.isnan(v)
    print(' train corr with y: %.3f' % np.corrcoef(v[ok], y[ok])[0,1])
# check hh_mean_loo construction on a sample household
hh = tr.household_key.iloc[0]
sub = m[m.household_key==hh][['snapshot_day','future_spend_4w','hh_mean_loo','hh_n','hh_mean_causal','hh_mean_last3']]
print('\nsample household rows:\n', sub.to_string())
# alignment check: does t12 have same rows as t11?
k11 = set(map(tuple, t11[['household_key','snapshot_day']].values))
k12 = set(map(tuple, t12[['household_key','snapshot_day']].values))
print('\nrows equal:', k11==k12, '| only in 12:', len(k12-k11), '| only in 11:', len(k11-k12))