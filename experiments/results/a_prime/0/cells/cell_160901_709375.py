
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
print(A.snapshot_days())
v = A.snapshot(431)
print('campaigns cols', v.campaigns.columns.tolist())
print(v.campaigns.head(3))
print('campaign_targets', v.campaign_targets.shape, v.campaign_targets.columns.tolist())
print(v.campaign_targets.head(3))
print('descriptions', v.campaign_targets.description.value_counts())
print('campaign ranges by type:')
print(v.campaigns.groupby('description').agg(n=('campaign','size'), smin=('start_day','min'), smax=('start_day','max'), emin=('end_day','min'), emax=('end_day','max')))
