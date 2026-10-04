import pandas as pd, numpy as np
tt = train_targets()
# Upper bound: predict each household's train-mean target
hh_mean = tt.groupby('household_key')['future_spend_4w'].mean()
p = tt.household_key.map(hh_mean).values
y = tt.future_spend_4w.values
print('in-sample MAE predicting household train-mean:', np.mean(np.abs(p-y)))

# Ratio stability: for each (hh, snapshot), r = future_spend_4w / spend_28_at_that_snapshot
e5 = load_saved('e005_longrun.parquet')
m = tt.merge(e5[['household_key','snapshot_day','spend_28']], on=['household_key','snapshot_day'])
m['r'] = m.future_spend_4w / m.spend_28.clip(lower=1e-6)
m['lr'] = np.log1p(m.future_spend_4w) - np.log1p(m.spend_28)   # log-ratio approx
g = m.groupby('household_key')['lr']
print('\nlog-ratio: mean within-hh std:', g.std().mean(), 'overall std:', m.lr.std())
print('within-hh std by n_obs:')
n = g.size()
for lo,hi in [(1,3),(3,6),(6,9),(9,13)]:
    sel = (n>=lo)&(n<hi)
    print(f'  n_obs {lo}-{hi-1}: mean within-std={g.std()[sel].mean():.3f} (n hh={sel.sum()})')

# autocorrelation of lr across snapshots within household
m2 = m.sort_values(['household_key','snapshot_day'])
m2['lr_prev'] = m2.groupby('household_key')['lr'].shift(1)
print('\ncorr(lr, lr_prev):', m2[['lr','lr_prev']].corr().iloc[0,1].round(3))
print('corr(r, r_prev) raw:', m2.assign(r_prev=m2.groupby('household_key')['r'].shift(1))[['r','r_prev']].corr().iloc[0,1].round(3))
print('median lr:', m.lr.median().round(3), 'median r:', m.r.median().round(3))