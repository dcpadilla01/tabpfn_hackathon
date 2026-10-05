import agent_api as api
import pandas as pd, numpy as np

e15 = api.load_saved('e015_stack.parquet')
tt = api.train_targets()
df = e15.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged:', df.shape)

y = df.future_spend_4w.values
# OOF stack MAE on train rows (stack is LOSO, so this estimates val performance)
mae_stack = np.abs(df.stack_ridge - y).mean()
mae_sp28 = np.abs(df.spend_28 - y).mean()
mae_ew28 = np.abs(df.d_ewma_spend_hl28 - y).mean()
print(f'MAE stack_ridge (train, OOF): {mae_stack:.3f}')
print(f'MAE spend_28: {mae_sp28:.3f}   MAE ewma_hl28: {mae_ew28:.3f}')
for w in [0.3,0.5,0.7]:
    b = w*df.stack_ridge + (1-w)*df.spend_28
    print(f'blend stack+spend28 w={w}: {np.abs(b-y).mean():.3f}')

# error by target level
df['abs_err'] = np.abs(df.stack_ridge - y)
df['ybin'] = pd.qcut(y.replace(0, np.nan), q=8, duplicates='drop')
print(df.groupby('ybin', observed=True).agg(n=('abs_err','size'), mae=('abs_err','mean'), med_pred=('stack_ridge','median'), med_y=('future_spend_4w','median')))

# zero rows: how much MAE comes from them
z = df.future_spend_4w==0
print('\nzero rows:', z.mean(), 'MAE on zero rows:', df.loc[z,'abs_err'].mean(), '-> contributes', z.mean()*df.loc[z,'abs_err'].mean())
print('MAE on nonzero rows:', df.loc[~z,'abs_err'].mean())
print('mean pred on zero rows:', df.loc[z,'stack_ridge'].mean())

# per snapshot
print('\nper-snapshot MAE:')
print(df.groupby('snapshot_day').agg(n=('abs_err','size'), mae=('abs_err','mean'), mean_y=('future_spend_4w','mean'), mean_pred=('stack_ridge','mean')))
