import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')

# target mean per train snapshot
print("target mean/median per snap:")
print(df.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','mean']).round(1).T)

e11 = agent_api.load_saved('e011_pruned_basket.parquet')
print("\ne011 shape", e11.shape)
# tenure-like cols in e011
cands = [c for c in e11.columns if any(k in c for k in ('tenure','first','days_since'))]
print("tenure-ish cols:", cands)

m = e11.merge(tt, on=['household_key','snapshot_day'], how='inner')
g = m.groupby('snapshot_day')
show = ['tenure','days_since_first','days_since_last','spend_364','trips_364','weekly_mean_12','dec_112','z_dec_224','spend_28','future_spend_4w']
show = [c for c in show if c in m.columns]
print("\nper-snapshot means (drift check):")
print(g[show].mean().round(2))

# how much of spend_364 drift is tenure? corr of feature with tenure, and partial
tr = m[m.snapshot_day<=431]
for c in ['spend_364','trips_364','dec_112','weekly_mean_12','z_dec_224','days_since_last']:
    x = tr[c]; tn = tr['tenure']
    ok = x.notna()&tn.notna()
    print(c, "corr w/ tenure:", round(np.corrcoef(x[ok],tn[ok])[0,1],3))
