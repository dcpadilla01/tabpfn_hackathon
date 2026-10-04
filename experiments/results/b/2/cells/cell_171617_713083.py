
import agent_api as A
t = A.load_saved('e014_gbm_oob.parquet')
print(t['gbm_corr'].describe())
print('nan frac', t['gbm_corr'].isna().mean())
print('per-snapshot corr mean:')
print(t.groupby('snapshot_day')['gbm_corr'].mean())
