
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
base = agent_api.load_saved('e011_rank.parquet')
print('same household order:', (base.household_key.values == tt.household_key.values).mean())
print('same snapshot order:', (base.snapshot_day.values == tt.snapshot_day.values).mean())
print(base[['household_key','snapshot_day','spend_28']].head(4))
print(tt.head(4))
# proper merge on keys
m = tt.merge(base, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values
print('corr(spend_28,y) after key merge:', np.corrcoef(m.spend_28.fillna(0), y)[0,1])
va = m.snapshot_day==431
print('MAE spend28 on 431:', np.abs(m.spend_28[va].fillna(0).values - y[va]).mean())
