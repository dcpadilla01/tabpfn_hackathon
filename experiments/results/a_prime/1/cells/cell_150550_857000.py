import numpy as np, pandas as pd, agent_api
t1 = agent_api.load_saved('e001_txhist.parquet')
tr = agent_api.train_targets()
d = t1.merge(tr, on=['household_key','snapshot_day'])
y = d['future_spend_4w']
# spend_rate28 is spend in last 28d / 28 * 28 = spend_l1? check
print('corr spend_l1 vs spend_rate28:', d[['spend_l1','spend_rate28']].corr().iloc[0,1].round(4))
print('identical?', np.allclose(d['spend_l1'], d['spend_rate28']))
# distribution of spend_l1 vs y
for c in ['spend_l1','spend_l2','spend_l3','spend_l123_mean']:
    print(c, 'proxy MAE:', round(np.abs(d[c]-y).mean(),2))
# multiplicative structure: y/spend_l1 by spend_l1 quantile
q = pd.qcut(d['spend_l1'], 12, duplicates='drop')
prof = d.groupby(q, observed=True).apply(lambda g: pd.Series({
    'n': len(g), 'mean_y': g['future_spend_4w'].mean(), 'mean_l1': g['spend_l1'].mean(),
    'ratio': g['future_spend_4w'].mean()/max(g['spend_l1'].mean(),1e-9)}))
print(prof.round(3))
# zero-recent households: what do they buy next?
z = d[d['spend_l1']==0]
print('spend_l1==0:', len(z), 'mean y:', z['future_spend_4w'].mean().round(2), 'zero-y share:', (z['future_spend_4w']==0).mean().round(3))
print(z['future_spend_4w'].describe().round(2))
nz = d[d['spend_l1']>0]
print('spend_l1>0 mean y:', nz['future_spend_4w'].mean().round(2))
# does momentum help within spend_l1 bins?
d['bin'] = pd.qcut(d['spend_l1'], 10, duplicates='drop')
print(d.groupby('bin', observed=True)[['momentum','future_spend_4w']].mean().round(2))
# y vs spend_l1 log-log slope
m = (d['spend_l1']>0)&(y>0)
import numpy as np
sl = np.polyfit(np.log(d.loc[m,'spend_l1']), np.log(y[m]), 1)
print('log-log slope y~spend_l1:', sl.round(3))
