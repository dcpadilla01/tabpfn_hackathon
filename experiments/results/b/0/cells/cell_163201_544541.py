import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e011_price.parquet')
tt = agent_api.train_targets()
print('tt shape', tt.shape, 'dups:', int(tt.duplicated(['household_key','snapshot_day']).sum()))
print('tt snapshots:', sorted(tt.snapshot_day.unique()))
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', df.shape, 'dups:', int(df.duplicated(['household_key','snapshot_day']).sum()))
g = df.groupby('snapshot_day')['future_spend_4w']
stats = pd.DataFrame({'n': g.size(), 'mean': g.mean(), 'med': g.median(), 'p90': g.quantile(.9), 'max': g.max(), 'zero': g.apply(lambda s:(s==0).mean())})
print(stats.round(1))
# simple baselines per snapshot: predict global train mean / median
yv = df.future_spend_4w.values
for sd in [403, 431]:
    m = df.snapshot_day==sd
    tr = df.snapshot_day<sd
    print(f'day {sd}: mean-pred MAE {np.abs(yv[tr].mean()-yv[m]).mean():.2f}, med-pred {np.abs(np.median(yv[tr])-yv[m]).mean():.2f}, top5 targets {np.sort(yv[m])[-5:]}')
# check huge targets
big = df.nlargest(10,'future_spend_4w')[['household_key','snapshot_day','future_spend_4w','spend28','spend364','lt_spend']]
print(big)
