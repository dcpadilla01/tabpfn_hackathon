import numpy as np, pandas as pd
t = agent_api.load_saved('rich_behavioral.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print("train rows:", len(df))
y = df['future_spend_4w']
print("target stats:", y.describe().to_dict())
print("target by snapshot day:")
print(df.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']))
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
corr = df[feats+['future_spend_4w']].corr()['future_spend_4w'].drop('future_spend_4w').sort_values(key=abs, ascending=False)
print("\ntop |corr| with target:")
print(corr.head(20))
print("\nlowest:")
print(corr.tail(10))
