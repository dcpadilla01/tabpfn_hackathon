import pandas as pd, numpy as np
P = agent_api.load_saved('e016_allpreds.parquet')
# final E016 prediction: 0.7*quantile + 0.3*two-part (best held-out variant, 62.77)
P['prediction'] = 0.7*P.pq + 0.3*np.where(P.pc < 0.5, 0.0, P.pa)
out = P[P.snapshot_day.isin([459,487,515,543])][['household_key','snapshot_day','prediction']]
print(out.shape, sorted(out.snapshot_day.unique()))
print('mean pred %.1f (e5 was 127.1)' % out.prediction.mean())
print('finite:', np.isfinite(out.prediction).all())
agent_api.save_table(out, 'e016_preds.parquet')