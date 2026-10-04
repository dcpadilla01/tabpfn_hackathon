import agent_api, numpy as np, pandas as pd

snap = agent_api.snapshot()
tr = snap.transactions
wk = ((tr['day']+8)//7).astype(int)
g = tr.groupby(wk)['sales_value'].sum().reindex(range(1,103)).fillna(0)
print('weekly market spend, weeks 1-102:')
print(g.round(0).values)
# seasonal index: analog weeks (target - 52) vs (snapshot week - 52)
def idx(target_wk0, cur_wk0, span=4):
    # target window weeks target_wk0..target_wk0+3 (year2), analog year1 weeks -52
    a = g[[w-52 for w in range(target_wk0, target_wk0+4)]].sum()
    b = g[[w-52 for w in range(cur_wk0-3, cur_wk0+1)]].sum()
    return a/b
print('\nseasonal index (year1 analog of target window / year1 analog of current week):')
for sd in [95,123,151,179,207,235,263,291,319,347,375,403,431,459,487,515,543]:
    cw = (sd+8)//7; tw = (sd+29+8)//7  # first target week
    print('  snap %3d curwk %2d tgtwk %2d idx %.2f' % (sd, cw, tw, idx(tw, cw)))

# bias of dec65 by snapshot
t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys=['household_key','snapshot_day']
W = np.nan_to_num(t[['spend_28']+['tlag_%d'%i for i in range(2,14)]].astype(float).values)
w = 0.65**np.arange(13); w/=w.sum(); dec65 = W@w
ttg = agent_api.train_targets()
mn = t[['household_key','snapshot_day']].merge(ttg, on=keys)
mn['dec65'] = dec65
y = mn['future_spend_4w'].values; p = mn['dec65'].values
folds = mn['snapshot_day'].values.astype(int)
print('\ndec65 bias (mean pred - mean y) & MAE by snapshot:')
for d in sorted(set(folds)):
    mm = folds==d
    print('  %3d: pred %7.1f y %7.1f bias %6.1f MAE %6.1f' % (d, p[mm].mean(), y[mm].mean(), p[mm].mean()-y[mm].mean(), np.abs(p[mm]-y[mm]).mean()))