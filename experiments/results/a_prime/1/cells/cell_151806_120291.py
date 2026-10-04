import agent_api as api
import pandas as pd, numpy as np

E3 = api.load_saved('e003_catmix.parquet')
E1 = api.load_saved('e001_txhist.parquet')
NF = api.load_saved('nf_candidates.parquet')
greedy = ['nwmax12','nspend_l12','nspend7','nf_pow90_ewm4','nunits84','nf_nspend28_pow90']
NFs = NF[['household_key','snapshot_day']+greedy]
E1s = E1[['household_key','snapshot_day','spend_l123_mean','zero_recent']]
df = E3.merge(NFs, on=['household_key','snapshot_day'], how='left').merge(E1s, on=['household_key','snapshot_day'], how='left')
s123 = df.spend_l123_mean.values.astype(float)
act = (df.zero_recent==0).astype(float).values
df['h_act_s123'] = s123*act
df['h_inact_s123'] = s123*(1-act)
df['h_act_pow90'] = np.power(1+s123,0.9)*act
df['h_inact_pow90'] = np.power(1+s123,0.9)*(1-act)
df['day_idx'] = df.snapshot_day.values.astype(float)
df = df.drop(columns=['spend_l123_mean','zero_recent'])
print(df.shape)
assert df[['household_key','snapshot_day']].duplicated().sum()==0
path = api.save_table(df, 'nf_p1.parquet')
print(path)
print(df.columns.tolist())