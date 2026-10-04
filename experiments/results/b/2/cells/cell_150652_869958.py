
import pandas as pd, numpy as np

tt = train_targets()
# target level by snapshot day (train only) - seasonality/drift?
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median',lambda s:(s==0).mean()])
g.columns=['mean','median','zerofrac']
print(g.round(1))

e5 = load_saved('e005_longrun.parquet')
m = tt.merge(e5[['household_key','snapshot_day','spend_28','wk_avg_8','spend_84']], on=['household_key','snapshot_day'])
print()
print(m.groupby('snapshot_day')[['future_spend_4w','spend_28','wk_avg_8']].median().round(1))
