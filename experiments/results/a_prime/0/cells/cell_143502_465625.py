
import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tx = v.transactions
prod = v.products
print('brand values:', prod.brand.value_counts(dropna=False).head().to_dict())
# target seasonality
tt = agent_api.train_targets()
print(tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median','count']).round(1).to_string())
# per-household 4-week block autocorrelation (same-season lag 13 blocks)
tx = tx[['household_key','day','sales_value']]
def block_spend(lo, hi):
    m = tx[(tx.day>=lo)&(tx.day<=hi)]
    return m.groupby('household_key').sales_value.sum()
s = agent_api.snapshot_days()
blocks = {}
for sd in s['train']:
    blocks[sd] = block_spend(sd+1, sd+28)
B = pd.DataFrame(blocks)
# correlate block sd with block sd-364 and sd-28
for lag, pairs in [(28,[(sd, sd-28) for sd in s['train'] if sd-28 in B.columns]),
                   (364,[(sd, sd-364) for sd in s['train'] if sd-364 in B.columns])]:
    cs = []
    for a,b in pairs:
        x = B[a]; y = B[b]
        ok = x.notna()&y.notna()
        if ok.sum()>50: cs.append(np.corrcoef(np.log1p(x[ok]), np.log1p(y[ok]))[0,1])
    print('lag',lag,'mean log-corr %.3f (n pairs %d)'%(np.mean(cs), len(cs)))
