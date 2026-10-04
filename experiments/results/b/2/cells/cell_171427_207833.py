import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET, snapshot_days, save_table
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
D = F.merge(tt, on=key)
trd = sorted(snapshot_days()['train'])
feat0 = [c for c in F.columns if c not in key+['gbm_pred']]
mkt28 = D.groupby('snapshot_day')['spend_28'].mean()
mkt_prev = mkt28.shift(1)
D['mkt28'] = D['snapshot_day'].map(mkt28)
D['mkt_mom'] = (D['snapshot_day'].map(mkt28)/D['snapshot_day'].map(mkt_prev)).replace([np.inf,-np.inf],np.nan)
D['drift28'] = D['spend_28'].astype(float)/D['mkt28']
gmed = D[D.snapshot_day.isin(trd)][TARGET].median()
k=16; w_ = D['trips_84'].astype(float)/(D['trips_84'].astype(float)+k)
D['shr'] = w_*D['spend_28'].astype(float) + (1-w_)*gmed
newcols = ['mkt28','mkt_mom','drift28','shr']
OUT = F[key+feat0].merge(D[key+newcols], on=key, how='left')
OUT['mkt_mom'] = OUT['mkt_mom'].fillna(OUT['mkt_mom'].median())
assert OUT.shape[0]==36426 and not OUT[key].duplicated().any()
assert sorted(OUT.snapshot_day.unique())==sorted(F.snapshot_day.unique())
path = save_table(OUT, 'e015_market_ctx')
print('saved:', path, OUT.shape, '| nan%:', round(OUT.isna().mean().mean()*100,2))
print('snap rows:', OUT.groupby('snapshot_day').size().to_dict())