
import agent_api as A, pandas as pd, numpy as np
e013 = A.load_saved('e013_stock.parquet')
tt = A.train_targets()
df = e013.merge(tt, on=['household_key','snapshot_day'])
y = df.future_spend_4w
df = df.sort_values(['household_key','snapshot_day'])
g = df.groupby('household_key').future_spend_4w
df['y_prev'] = g.shift(1); df['y_next'] = g.shift(-1)
print('corr(y, y_prev)', df[['future_spend_4w','y_prev']].corr().iloc[0,1])
print('corr(y, y_next)', df[['future_spend_4w','y_next']].corr().iloc[0,1])
print('corr(y, mean(prev,next))', ((df.y_prev+df.y_next)/2).corr(df.future_spend_4w))
hm = df.groupby('household_key').future_spend_4w.transform('mean')
print('R2 household mean (oracle)', 1-((df.future_spend_4w-hm)**2).sum()/((df.future_spend_4w-df.future_spend_4w.mean())**2).sum())
# per-household std vs between-household std
hs = df.groupby('household_key').future_spend_4w.agg(['mean','std','count'])
print('within std mean', hs['std'].mean(), 'between std', hs['mean'].std())
