
import pandas as pd, numpy as np, agent_api as A

# 1) aggregate spend by 4-week window over all households (view capped at 459)
snap = A.snapshot(459)
tx = snap.transactions
txg = tx.groupby('day').sales_value.sum()
tot = txg.reindex(range(1,460), fill_value=0)
w = pd.Series([tot.iloc[d-1:d+27].sum() for d in range(1, 433)], index=range(1,433))
print('4w-window total spend (all hh), selected windows:')
for d in [1,29,57,85,113,141,169,197,225,253,281,309,337,365,393,421]:
    print('  start %3d: %9.0f' % (d, w[d]))
print('mean of first 6 windows %.0f | windows 4-7 (95-179 starts: 85,113,141,169) %.0f | last 6 %.0f' % (
    w.iloc[:6].mean(), w.loc[[85,113,141,169]].mean(), w.iloc[-6:].mean()))
# seasonality ratio: window starting d vs d+364 impossible (data shorter); check within-year pattern via week
wk = txg.reindex(range(1,460), fill_value=0).groupby((np.arange(1,460)-1)//7).sum()
print('weekly total: first 13 wks %.0f, weeks 14-26 %.0f, last 13 wks %.0f' % (wk.iloc[:13].mean(), wk.iloc[13:26].mean(), wk.iloc[-13:].mean()))

# 2) E013 error breakdown on train (in-sample, indicative)
tt = train_targets()
f = load_saved('feats_v4.parquet')
tr = f[f.snapshot_day.isin(A.snapshot_days()['train'])].merge(tt, on=['household_key','snapshot_day'])
p13 = load_saved('pred_e013.parquet').rename(columns={'prediction':'p13'})
tr = tr.merge(p13, on=['household_key','snapshot_day'])
tr['err'] = (tr.future_spend_4w - tr.p13).abs()
tr['dec'] = pd.qcut(tr.p13, 10, duplicates='drop')
g = tr.groupby('dec', observed=True).agg(n=('err','size'), pred=('p13','mean'), y=('future_spend_4w','mean'),
                                         ymed=('future_spend_4w','median'), mae=('err','mean'))
print(g.round(1).to_string())
print('overall train MAE %.2f (in-sample)' % tr.err.mean())
# error by actual zero
for z in [0,1]:
    s = tr[tr.future_spend_4w==0] if z==0 else tr[tr.future_spend_4w>0]
    print('actual %s: n=%d MAE %.2f meanpred %.1f' % ('zero' if z==0 else '>0', len(s), s.err.mean(), s.p13.mean()))
