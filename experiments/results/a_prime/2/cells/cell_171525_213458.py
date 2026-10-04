import agent_api, numpy as np, pandas as pd

snap = agent_api.snapshot()
tr = snap.transactions
tr = tr[tr.day <= 459]
wk = ((tr['day']+8)//7).astype(int)  # 1..66
# stable households: active in both year1 (wk1-52) and year2 (wk53-66)
hh_wk = tr.groupby(['household_key', wk])['sales_value'].sum()
piv = hh_wk.unstack(fill_value=0.0)
y1 = piv[[w for w in range(1,53) if w in piv.columns]]
y2 = piv[[w for w in range(53,67) if w in piv.columns]]
act1 = (y1>0).sum(1); act2 = (y2>0).sum(1)
stable = piv[(act1>=20)&(act2>=10)]
print('households total %d, stable %d' % (len(piv), len(stable)))
s1 = stable[[w for w in range(1,53) if w in stable.columns]].mean(0)
s2 = stable[[w for w in range(53,67) if w in stable.columns]].mean(0)
s2.index = [w-52 for w in s2.index]
prof = pd.concat([s1, s2], axis=1).mean(1)  # per-household avg weekly spend by week-of-year 1..14
prof_full = s1.copy()
print('\nper-household avg weekly spend by week-of-year (year1), every 4 wks:')
print(prof_full.round(1).iloc[::4])
overall = prof_full.mean()
seas = prof_full/overall
print('\nseasonal index by week-of-year (year1), selected:')
for w in [1,5,9,13,17,21,25,29,33,37,41,45,49,52]:
    print('  wk %2d: %.2f' % (w, seas[w]))
# index for validation target windows: weeks 67-70 -> woy 15-18; current wks 63-66 -> woy 11-14
def sidx(woy_list): return seas[woy_list].mean()/seas.mean()
print('\nseasonal index target-window/current-window:')
for sd in [95,123,151,179,207,235,263,291,319,347,375,403,431,459,487,515,543]:
    cw = (sd+8)//7; tw = (sd+29+8)//7
    cw_y = [w-52 if w>52 else w for w in range(cw-3,cw+1)]
    tw_y = [w-52 if w>52 else w for w in range(tw,tw+4)]
    print('  snap %3d -> %.3f' % (sd, sidx(tw_y)/sidx(cw_y)))

# dec65 bias by snapshot
t = agent_api.load_saved('e009_ewma_longlags.parquet')
W = np.nan_to_num(t[['spend_28']+['tlag_%d'%i for i in range(2,14)]].astype(float).values)
w = 0.65**np.arange(13); w/=w.sum(); dec65 = W@w
ttg = agent_api.train_targets()
mn = t[['household_key','snapshot_day']].merge(ttg, on=['household_key','snapshot_day'])
mn['dec65'] = dec65
y = mn['future_spend_4w'].values; p = mn['dec65'].values
folds = mn['snapshot_day'].values.astype(int)
print('\ndec65 by snapshot: pred / y / MAE:')
for d in sorted(set(folds)):
    mm = folds==d
    print('  %3d: %7.1f %7.1f %6.1f' % (d, p[mm].mean(), y[mm].mean(), np.abs(p[mm]-y[mm]).mean()))