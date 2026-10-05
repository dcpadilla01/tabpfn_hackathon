import agent_api as api
import pandas as pd, numpy as np

e15 = api.load_saved('e015_stack.parquet')
tt = api.train_targets()
df = e15.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values
df['abs_err'] = np.abs(df.stack_ridge - y)

# error by target level
tmp = df[y>0].copy()
tmp['ybin'] = pd.qcut(tmp.future_spend_4w, q=8, duplicates='drop')
print(tmp.groupby('ybin', observed=True).agg(n=('abs_err','size'), mae=('abs_err','mean'), med_pred=('stack_ridge','median'), med_y=('future_spend_4w','median')).round(1))

z = y==0
print('\nzero rows:', round(z.mean(),3), 'MAE on zero rows:', round(df.loc[z,'abs_err'].mean(),2), 'contrib:', round(z.mean()*df.loc[z,'abs_err'].mean(),2))
print('MAE nonzero:', round(df.loc[~z,'abs_err'].mean(),2))
print('mean pred on zero rows:', round(df.loc[z,'stack_ridge'].mean(),2))

print('\nper-snapshot:')
print(df.groupby('snapshot_day').agg(n=('abs_err','size'), mae=('abs_err','mean'), mean_y=('future_spend_4w','mean'), mean_pred=('stack_ridge','mean')).round(1))

# simple peer-median check on train rows: nearest neighbors in (log spend_28, log trips_28) space
from collections import defaultdict
feat = np.column_stack([np.log1p(df.spend_28.values), np.log1p(df.trips_28.values), np.log1p(df.recency.values)])
# scale
mu, sd = feat.mean(0), feat.std(0)+1e-9
F = (feat-mu)/sd
Y = y
# subsample reference for speed
rng = np.random.RandomState(0)
idx_ref = rng.choice(len(F), 8000, replace=False)
Fr, Yr = F[idx_ref], Y[idx_ref]
# for a sample of queries, find 50 NN and take median y
k=50
mae_nn = []; mae_base=[]
qs = rng.choice(len(F), 4000, replace=False)
from numpy.linalg import norm
for q in qs:
    d = ((Fr - F[q])**2).sum(1)
    nn = np.argpartition(d, k)[:k]
    mae_nn.append(abs(np.median(Yr[nn]) - Y[q]))
    mae_base.append(abs(df.stack_ridge.values[q] - Y[q]))
print('\npeer-median kNN(50) MAE (sample):', round(np.mean(mae_nn),2), ' vs stack:', round(np.mean(mae_base),2))
