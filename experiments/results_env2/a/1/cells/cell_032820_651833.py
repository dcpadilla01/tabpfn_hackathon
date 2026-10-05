import pandas as pd, numpy as np
tt = agent_api.train_targets()
print('targets', tt.shape)
y = tt.future_spend_4w.values
print('mean %.2f med %.2f zero %.3f' % (y.mean(), np.median(y), (y==0).mean()))
print('q', np.round(np.quantile(y,[.05,.1,.25,.5,.75,.9,.95,.99]),1))
F = agent_api.load_saved('allF.parquet')
print('allF', F.shape)
print('days', sorted(F.snapshot_day.unique()))
print('cols', F.shape[1])
print(list(F.columns))
p5 = agent_api.load_saved('e005_preds.parquet')
print('e005_preds', p5.shape, sorted(p5.snapshot_day.unique()))
print(p5.head(3))
