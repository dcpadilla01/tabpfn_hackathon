
import pandas as pd, numpy as np, agent_api as A

snap = A.snapshot(459)
tx = snap.transactions
# stable cohort: first purchase <= day 94 (has rows at train snaps 95-179 AND late snaps)
first = tx.groupby('household_key').day.min()
cohort = first[first <= 94].index
print('cohort size:', len(cohort))
ctx = tx[tx.household_key.isin(cohort)]
cg = ctx.groupby(['household_key','day']).sales_value.sum().reset_index()

def win_spend(lo, hi):
    m = cg[(cg.day>=lo)&(cg.day<=hi)].groupby('household_key').sales_value.sum()
    return m.reindex(cohort).fillna(0.0)

# same cycle positions, early vs late: windows starting d and d+364 (364-day cycle)
pairs = [(85,449),(113,441),(141,433),(169,425),(197,397),(225,369)]
rows=[]
for e,l in pairs:
    a, b = win_spend(e,e+27), win_spend(l,l+27)
    rows.append((e, l, a.mean(), b.mean(), b.mean()/max(a.mean(),1e-9), (a==0).mean(), (b==0).mean()))
df = pd.DataFrame(rows, columns=['early_start','late_start','early_mean','late_mean','ratio','early_zero','late_zero'])
print(df.round(3).to_string())
print('avg ratio (late/early) = %.3f' % df.ratio.mean())

# also mid-period same positions: d+182
rows2=[]
for e in [85,113,141,169,197,225]:
    a, b = win_spend(e,e+27), win_spend(e+182,e+209)
    rows2.append((e, a.mean(), b.mean(), b.mean()/max(a.mean(),1e-9)))
print(pd.DataFrame(rows2, columns=['start','early','mid','ratio']).round(3).to_string())

# cohort spend by cycle position (day mod 364) across full history: mean 4w spend per window start every 28d
allstarts = list(range(85, 433, 28))
vals = [win_spend(d, d+27).mean() for d in allstarts]
s = pd.Series(vals, index=allstarts)
pos = (s.index - 1) % 364 + 1
sm = pd.DataFrame({'start': s.index, 'cycle_pos': pos, 'mean4w': s.values})
print(sm.round(1).to_string())
