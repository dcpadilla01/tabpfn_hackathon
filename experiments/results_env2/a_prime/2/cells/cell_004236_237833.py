
import agent_api as A
import pandas as pd, numpy as np

t = A.load_saved('e016_churn_gapratio.parquet')
g = t.groupby('snapshot_day')

new = pd.DataFrame(index=t.index)
new['pctl_s28']    = g['spend_28'].rank(pct=True)
new['pctl_ew28']   = g['ew_28'].rank(pct=True)
new['pctl_s84']    = g['spend_84'].rank(pct=True)
new['pctl_dsl']    = g['days_since_last'].rank(pct=True)
new['rel_s28_med']  = t['spend_28'] / t['snap_med_s28']
new['rel_ew28_med'] = t['ew_28'] / t['snap_med_ew28']
new['rel_s84_med']  = t['spend_84'] / t['snap_med_s84']
new['rel_dsl_med']  = t['dsl'] / t['snap_mean_dsl']
new['s28_minus_med'] = t['spend_28'] - t['snap_med_s28']
new['ew28_over_p90'] = t['ew_28'] / t['snap_p90_s28']

out = pd.concat([t, new], axis=1)
print(out.shape, out[['snapshot_day']].nunique().values)
print(new.describe().T[['mean','std','min','max']])
path = A.save_table(out, 'e017_xsec_rank.parquet')
print(path)
