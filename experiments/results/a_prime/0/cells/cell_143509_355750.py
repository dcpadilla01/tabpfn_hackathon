
import agent_api, pandas as pd, numpy as np
tx = agent_api.snapshot(459).transactions[['household_key','day','sales_value']]
def bs(lo,hi):
    m = tx[(tx.day>=lo)&(tx.day<=hi)]
    return m.groupby('household_key').sales_value.sum()
s = agent_api.snapshot_days()['train']
# history blocks: last block before snapshot (sd-27..sd), same season one year back (sd-391..sd-364), one block back (sd-55..sd-28)
rows=[]
for sd in s:
    cur = bs(sd-27, sd); yr = bs(sd-391, sd-364); prev = bs(sd-55, sd-28)
    d = pd.DataFrame({'cur':cur,'yr':yr,'prev':prev}).dropna()
    r_yr = np.corrcoef(np.log1p(d.cur), np.log1p(d.yr))[0,1]
    r_prev = np.corrcoef(np.log1p(d.cur), np.log1p(d.prev))[0,1]
    rows.append((sd, r_yr, r_prev, len(d)))
print(pd.DataFrame(rows, columns=['sd','r_lag364','r_lag28','n']).round(3).to_string())
# how many households have data 364 days back at each snapshot
hh_first = tx.groupby('household_key').day.min()
for sd in [95,123,207,347,431,459,487,515,543]:
    print(sd, 'eligible:', (hh_first<=sd-84).sum(), 'with 364d hist:', (hh_first<=sd-364).sum())
