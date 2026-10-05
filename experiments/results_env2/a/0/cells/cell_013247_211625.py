
import pandas as pd, numpy as np, agent_api as A

tt = train_targets()
f = load_saved('feats_v4.parquet')
tr = f[f.snapshot_day.isin(A.snapshot_days()['train'])].merge(tt, on=['household_key','snapshot_day'])
print('train rows', len(tr), 'zero frac %.3f' % (tr.future_spend_4w==0).mean())

# trivial predictor MAEs on train
y = tr.future_spend_4w.values
for name, p in [('zero', np.zeros(len(y))), ('median', np.full(len(y), np.median(y))),
                ('lag1', tr.lag1_spend.values), ('exp4w_blend', tr.exp4w_blend.values),
                ('spend_28', tr.spend_28.values)]:
    print('%-12s MAE %.2f' % (name, np.abs(y-p).mean()))

# zero-inflation: by exp4w_blend bucket
tr['b'] = pd.qcut(tr.exp4w_blend.clip(lower=0), 10, duplicates='drop')
g = tr.groupby('b', observed=True).agg(n=('future_spend_4w','size'), zero=('future_spend_4w', lambda s:(s==0).mean()),
                                       med=('future_spend_4w','median'), mean=('future_spend_4w','mean'),
                                       med_lag=('lag1_spend','median'))
print(g.round(2).to_string())

# seasonal lag: spend in [snap-363, snap-336] (year-ago target window)
snap = A.snapshot(459)
tx = snap.transactions
txg = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
def yearago(day):
    lo, hi = day-363, day-336
    m = txg[(txg.day>=lo)&(txg.day<=hi)].groupby('household_key').sales_value.sum()
    return m
yl = {}
for d in [375,403,431]:
    yl[d] = yearago(d)
sub = tr[tr.snapshot_day.isin(yl.keys())].copy()
sub['ylag'] = [yl[d].get(h,0.0) for h,d in zip(sub.household_key, sub.snapshot_day)]
s = sub[sub.tenure>=364]
print('rows w/ full year-ago:', len(s), 'corr(ylag, y) = %.3f' % s[['ylag','future_spend_4w']].corr().iloc[0,1])
print('MAE lag1 %.2f | ylag %.2f | blend(0.5) %.2f | zero %.2f' % (
    np.abs(s.future_spend_4w-s.lag1_spend).mean(), np.abs(s.future_spend_4w-s.ylag).mean(),
    np.abs(s.future_spend_4w-0.5*(s.lag1_spend+s.ylag)).mean(), np.abs(s.future_spend_4w).mean()))
print('mean ylag %.1f vs mean y %.1f' % (s.ylag.mean(), s.future_spend_4w.mean()))
