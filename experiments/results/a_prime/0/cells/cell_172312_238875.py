import numpy as np, pandas as pd
t13=agent_api.load_saved('e013_stock.parquet')
tt=agent_api.train_targets()
df=t13.merge(tt,on=['household_key','snapshot_day'],how='left')
days=agent_api.snapshot_days(); trm=df.snapshot_day.isin(days['train']).values
na=df.m_spend28.isna()
print('m_spend28 NaN frac %.3f; spend_28==0 frac %.3f; overlap %.3f'%((na.mean()), (df.spend_28==0).mean(), (na&(df.spend_28==0)).mean()))
print('y mean where m_spend28 NaN: %.1f | not-NaN: %.1f'%(df.future_spend_4w[trm&na].mean(), df.future_spend_4w[trm&~na].mean()))
print('spend_28 stats where NaN:', df.spend_28[na].describe().round(1).to_dict())
feat=[c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
nafrac=df[feat].isna().mean().sort_values(ascending=False)
print('--- top NaN fractions ---'); print(nafrac[nafrac>0].head(20).round(3).to_dict())
pat=na.values
same=[c for c in feat if df[c].isna().values.tobytes()==pat.tobytes()]
print('cols with identical NaN pattern:', same[:20])
