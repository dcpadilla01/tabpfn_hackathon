import numpy as np, pandas as pd
t=agent_api.load_saved('e013_stock.parquet')
tt=agent_api.train_targets()
df=t.merge(tt,on=['household_key','snapshot_day'],how='left')
days=agent_api.snapshot_days(); trm=df.snapshot_day.isin(days['train']).values
# 1) m_spend28 vs spend_28
sub=df.loc[trm,['m_spend28','spend_28','future_spend_4w']]
print('corr(m_spend28,spend_28)', sub.m_spend28.corr(sub.spend_28).round(4))
d=(sub.m_spend28-sub.spend_28)
print('diff describe:', d.describe().round(3).to_dict())
print('frac equal:', (d.abs()<1e-9).mean().round(3))
# quantiles of each
for c in ['m_spend28','spend_28']:
    print(c, np.quantile(sub[c],[.5,.9,.99,.999]).round(1), 'max', sub[c].max().round(1))
# corr with target
print('corr with y: m_spend28 %.3f spend_28 %.3f'%(sub.m_spend28.corr(sub.future_spend_4w), sub.spend_28.corr(sub.future_spend_4w)))
# spearman
print('spearman with y: m %.3f s %.3f'%(sub.m_spend28.corr(sub.future_spend_4w,method='spearman'), sub.spend_28.corr(sub.future_spend_4w,method='spearman')))
