import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('weekly_history.parquet')
print('E007 table:', t.shape)
cols = list(t.columns)
for i in range(0, len(cols), 10):
    print('|', ', '.join(cols[i:i+10]))

tt = agent_api.train_targets()
print('\ntargets:', tt.shape, tt.columns.tolist())
print(tt['future_spend_4w'].describe())
print('zero frac:', float((tt['future_spend_4w']==0).mean()))

m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
num = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).select_dtypes(include=[np.number])
cor = num.corrwith(m['future_spend_4w']).abs().sort_values(ascending=False)
print('\ntop-30 |corr|:')
print(cor.head(30).to_string())
print('features with |corr|<0.01:', int((cor<0.01).sum()), '/', len(cor))

v = agent_api.snapshot()
print('\nview attrs:', [a for a in dir(v) if not a.startswith('_')])
hh = v.households
print('households type:', str(pd.Series([hh]).dtype), hh.shape if hasattr(hh,'shape') else '')
print('cols', list(hh.columns)[:6])
print('index name:', hh.index.name)
print(hh.head(3))
print('day', v.day, 'week', v.week)
tx = v.transactions
print('tx', tx.shape, list(tx.columns))
print('has products:', hasattr(v, 'products'))