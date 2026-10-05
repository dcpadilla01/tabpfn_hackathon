import numpy as np, pandas as pd
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
df = T.merge(tt, on=['household_key','snapshot_day'], how='inner')
print("merged", df.shape)

y = df.future_spend_4w.values
sp28 = df.spend_28.values
sp84 = df.spend_84.values
sp182 = df.spend_182.values

# overall correlations
def corr(a,b):
    m = np.isfinite(a)&np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0,1]
for c in ['spend_28','spend_84','spend_182','spend_365','d_ewma_spend_hl28','d_ewma_spend_hl112','blk_1','spend_s364','total_all','avg28_all','tenure','recency']:
    print(f"corr(y, {c}) = {corr(df[c].values.astype(float), y):.3f}")

# ratio y / trailing spend by snapshot day
for day in sorted(df.snapshot_day.unique()):
    sub = df[df.snapshot_day==day]
    m = sub.spend_28.values > 0
    r = sub.future_spend_4w.values[m] / sub.spend_28.values[m]
    print(f"day {day}: n={len(sub)}, median ratio y/sp28={np.median(r):.2f}, mean sp28={sub.spend_28.mean():.0f}, mean y={sub.future_spend_4w.mean():.0f}")

# per-household: does trailing 28d spend persist across snapshots?
T2 = T[['household_key','snapshot_day','spend_28','spend_84','spend_182','spend_365','tenure','recency']].sort_values(['household_key','snapshot_day'])
g = T2.groupby('household_key')
for c in ['spend_28','spend_84','spend_182','spend_365']:
    T2[c+'_prev'] = g[c].shift(1)
m = T2.spend_28_prev.notna() & (T2.spend_28_prev>0)
print("corr(prev spend_28, spend_28) =", np.corrcoef(T2.spend_28_prev[m], T2.spend_28[m])[0,1])
m84 = T2.spend_84_prev.notna() & (T2.spend_84_prev>0)
print("corr(prev spend_84, spend_84) =", np.corrcoef(T2.spend_84_prev[m84], T2.spend_84[m84])[0,1])
m182 = T2.spend_182_prev.notna()
print("corr(prev spend_182, spend_182) =", np.corrcoef(T2.spend_182_prev[m182], T2.spend_182[m182])[0,1])
