
import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
print('targets', tt.shape)
print(tt.future_spend_4w.describe())
print(tt.groupby('snapshot_day').size())

# households per snapshot: first purchase >= 84 days earlier
s = A.snapshot()
tx = s.transactions
first = tx.groupby('household_key').day.min()
print('first day quantiles:', first.quantile([0,.25,.5,.75,1]).values)

# quick predictive check at snapshot 431
sub = tt[tt.snapshot_day == 431].set_index('household_key')
t431 = tx[tx.day <= 431]
sp28 = t431[t431.day > 431-28].groupby('household_key').sales_value.sum().reindex(sub.index).fillna(0)
sp84 = t431[t431.day > 431-84].groupby('household_key').sales_value.sum().reindex(sub.index).fillna(0)
sp364 = t431[t431.day > 431-364].groupby('household_key').sales_value.sum().reindex(sub.index).fillna(0)
y = sub.future_spend_4w
print('corr sp28', np.corrcoef(sp28, y)[0,1], 'corr sp84', np.corrcoef(sp84, y)[0,1], 'corr sp364', np.corrcoef(sp364, y)[0,1])
print('MAE 0:', y.abs().mean(), 'MAE mean:', (y.mean()-y).abs().mean())
print('MAE sp28:', (sp28-y).abs().mean(), 'MAE sp84/3:', (sp84/3-y).abs().mean(), 'MAE sp364/13:', (sp364/13-y).abs().mean())
# blend
for w in [0.3,0.5,0.7]:
    print(f'MAE blend {w}*sp28+(1-w)*sp364/13:', (w*sp28+(1-w)*sp364/13-y).abs().mean())
print('zero fraction', (y==0).mean())
