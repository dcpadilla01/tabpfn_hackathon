
import agent_api as A, pandas as pd, numpy as np
e013 = A.load_saved('e013_stock.parquet')
tt = A.train_targets()
df = e013.merge(tt, on=['household_key','snapshot_day'])
# does a household appear at multiple snapshots? how correlated are consecutive blocks?
df = df.sort_values(['household_key','snapshot_day'])
g = df.groupby('household_key').future_spend_4w
df['y_prev'] = g.shift(1)
df['y_next'] = g.shift(-1)
sub = df.dropna(subset=['y_prev','y_next'])
print('n households', df.household_key.nunique(), 'rows', len(df))
print('corr(y, y_prev)', df[['y','y_prev']].corr().iloc[0,1])
print('corr(y, y_next)', df[['y','y_next']].corr().iloc[0,1])
print('corr(y, y_prev+y_next)/2', ((df.y_prev+df.y_next)/2).corr(df.y))
# how much of target variance is household identity? R2 of household-mean predictor
hm = df.groupby('household_key').y.transform('mean')
print('R2 household mean', 1-((df.y-hm)**2).sum()/((df.y-df.y.mean())**2).sum())
# but that uses future info; use only PAST blocks mean as pure past predictor
