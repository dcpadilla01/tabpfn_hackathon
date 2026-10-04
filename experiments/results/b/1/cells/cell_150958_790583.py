import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e001_history.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', df.shape)

num = [c for c in df.columns if df[c].dtype.kind in 'ifb' and c not in ('household_key','snapshot_day','future_spend_4w','index')]
cat = [c for c in df.columns if df[c].dtype.name=='category']
print('num', len(num), 'cat', cat)

# correlation with target
for c in num:
    print(f"{c:20s} corr={df[c].corr(df.future_spend_4w): .3f}  nan={df[c].isna().mean():.2f}")