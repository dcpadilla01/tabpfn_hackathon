import agent_api, pandas as pd, numpy as np
def mae(y, p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
tt = agent_api.train_targets()
e = agent_api.load_saved('e009_ewma_longlags.parquet')
m = e.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431]; va = m[m.snapshot_day>=459]
print('train rows', len(tr), 'val rows', len(va))
for c in ['spend_84','spend_28','spend_56','ewma_4']:
    print(c, 'val MAE:', round(mae(va['future_spend_4w'], va[c]),3))
z = va[va['spend_28']==0]
print('val zero28 rows:', len(z), 'target mean:', round(z['future_spend_4w'].mean(),1), 'median:', z['future_spend_4w'].median())
print('val overall mean target:', round(va['future_spend_4w'].mean(),1))
pred = np.where(va['spend_28']==0, 0, va['spend_84'])
print('hybrid MAE:', round(mae(va['future_spend_4w'], pred),3))
pred2 = np.where(va['spend_28']==0, 0, tr[tr['spend_28']>0]['future_spend_4w'].mean())
print('hybrid2 MAE:', round(mae(va['future_spend_4w'], pred2),3))
print('corr spend_84-target (zero28 rows):', m[m['spend_28']==0][['spend_84','future_spend_4w']].corr().iloc[0,1].round(3))
print('corr spend_84-target (active rows):', m[m['spend_28']>0][['spend_84','future_spend_4w']].corr().iloc[0,1].round(3))
# zero28 rows: distribution of future target
print('zero28 target quantiles:', m[m['spend_28']==0]['future_spend_4w'].quantile([0.25,0.5,0.75,0.9]).round(0).to_dict())
