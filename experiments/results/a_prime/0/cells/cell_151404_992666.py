import agent_api, pandas as pd, numpy as np
mkt = agent_api.load_saved('mkt_v2.parquet')
print("E003 table shape:", mkt.shape)
print("columns:", list(mkt.columns))
tt = agent_api.train_targets()
print("targets shape:", tt.shape)
df = mkt.merge(tt, on=['household_key','snapshot_day'], how='inner')
print("merged:", df.shape)
y = df['future_spend_4w']
print(y.describe())
print("zero share:", (y==0).mean())
num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
cor = df[num].corrwith(y, method='spearman').sort_values()
print(cor.to_string())
