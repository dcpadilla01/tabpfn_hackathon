
import agent_api as A, pandas as pd, numpy as np
e8 = A.load_saved('e008_level_shape.parquet')
tt = A.train_targets()
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values
# Quantile-bucket analysis: how well does sp28 (best single) predict in each bucket?
q = pd.qcut(m['sp28'].fillna(0), 10, duplicates='drop')
g = m.groupby(q, observed=True).agg(y=('future_spend_4w','mean'), pred=('sp28','mean'), n=('future_spend_4w','size'))
g['y_med'] = m.groupby(q, observed=True)['future_spend_4w'].median()
print(g)
# zero rate
print('zero rate', (y==0).mean())
# key: does a log-target view help? check corr of log1p(y) with log1p(sp28)
print('corr log', np.corrcoef(np.log1p(m['sp28'].fillna(0)), np.log1p(y))[0,1])
# what does sp28 look like vs y in raw scale
print(m[['sp28','future_spend_4w']].describe())
# check per-snapshot: is val harder?
print('train sp28 mae by snapshot:')
m['mae_sp28'] = np.abs(m['sp28'].fillna(0)-y)
print(m.groupby('snapshot_day')['mae_sp28'].mean())
# distribution of sp28 vs y at each snapshot - drift?
for d in [95, 431, 459, 543]:
    pass
# look at raw data around snapshots: any structural break in spend level over time?
v = A.snapshot()
tx = v.table('transactions')
tx = tx[tx.day<=459]
wk = tx.groupby('day')['sales_value'].sum()
print('total spend by 28d period:')
print((wk.groupby((wk.index-1)//28).sum()).round(0).to_string())
