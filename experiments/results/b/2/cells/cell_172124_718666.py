import agent_api as A
import pandas as pd, numpy as np

stack = A.load_saved('e014_stack.parquet')
print('cols tail:', stack.columns.tolist()[-3:])
df = stack.copy()
df['gbm_pred'] = df['gbm_pred'] / 100.0
df['gbm_log'] = np.log1p(df['gbm_pred']*100.0)/100.0
p = A.save_table(df, 'e017_stack_scaled.parquet')
print(p, df.shape)
print(df[['household_key','snapshot_day','gbm_pred','gbm_log']].head())