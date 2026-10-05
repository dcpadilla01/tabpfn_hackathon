import agent_api as api
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
print("snapdays:", api.snapshot_days())
tt = api.train_targets()
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median'])
g['zero_frac'] = tt.groupby('snapshot_day')['future_spend_4w'].apply(lambda x: (x==0).mean())
print(g.round(2))
d = g.reset_index()
sl, ic = np.polyfit(d.snapshot_day, d['mean'], 1)
print("trend slope/day %.4f intercept %.2f" % (sl, ic))
allF = api.load_saved('allF.parquet')
print("allF", allF.shape)
cols = list(allF.columns)
print("ncol", len(cols))
for i in range(0, len(cols), 12):
    print(i, cols[i:i+12])
m = tt.merge(allF, on=['household_key','snapshot_day'])
print("merged", m.shape)
corr = m.corr(numeric_only=True)['future_spend_4w'].drop('future_spend_4w')
corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
print("top |corr| with target:")
print(corr.head(30).round(3))
for name in ['e005_preds.parquet','e011_preds.parquet','e004_preds.parquet','repro_e5.parquet']:
    df = api.load_saved(name)
    print(name, df.shape)
    print(df.groupby('snapshot_day')['prediction'].agg(['count','mean','median']).round(2))
key = [c for c in cols if 'spend_28' in c.lower() or 'sp28' in c.lower() or 'spend_84' in c.lower()][:4]
print("key feats:", key)
if key:
    print(allF.groupby('snapshot_day')[key].mean().round(1))
