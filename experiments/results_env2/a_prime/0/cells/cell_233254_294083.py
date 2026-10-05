
import agent_api as A
e13 = A.load_saved('e013_storeprod.parquet')
b1 = A.load_saved('cand_batch1.parquet')
print('e13', e13.shape)
print('b1', b1.shape)
print('e13 cols:', list(e13.columns))
print('b1 cols:', list(b1.columns))
print(b1.head(3).T)
t = A.train_targets()
print(t['future_spend_4w'].describe())
print(A.snapshot_days())
