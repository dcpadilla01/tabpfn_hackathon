
import numpy as np, pandas as pd
tt = train_targets()
g = tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median','count'])
print(g.round(2).to_string())
BASE = load_saved('e011_discounts.parquet')
b = BASE.groupby('snapshot_day')[['spend_28','spend_365','ew_28']].mean().round(2)
print(b.to_string())
# correlation of day with mean target
days = g.index.values.astype(float); m = g['mean'].values
print('corr(day, mean target):', np.corrcoef(days, m)[0,1].round(3))
sl = np.polyfit(days, m, 1); print('slope per day:', sl[0].round(4), '=> per 28d:', (sl[0]*28).round(2))
# trend in spend_28 (behavioral level)
b2 = BASE.groupby('snapshot_day').spend_28.mean()
print('corr(day, mean spend_28):', np.corrcoef(b2.index.values.astype(float), b2.values)[0,1].round(3))
sl2 = np.polyfit(b2.index.values.astype(float), b2.values, 1); print('spend_28 slope/day:', sl2[0].round(4))
# zero share by day
z = tt.groupby('snapshot_day').future_spend_4w.apply(lambda x: (x==0).mean()).round(3)
print(z.to_string())
