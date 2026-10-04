
import agent_api as A
t = A.load_saved('e014_stack.parquet')
print(t['gbm_pred'].describe())
print(t.groupby('snapshot_day')['gbm_pred'].mean())
print(t.groupby('snapshot_day').size())
