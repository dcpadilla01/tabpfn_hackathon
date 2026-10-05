
import agent_api, pandas as pd, numpy as np
e15 = agent_api.load_saved('e015_stack.parquet')
print('e15 stack cols:', [c for c in e15.columns if 'stack' in c])
print(e15[['household_key','snapshot_day','stack_ridge_log']].head())
print(e15['stack_ridge_log'].describe())
print('NaN in stack:', e15['stack_ridge_log'].isna().sum())
