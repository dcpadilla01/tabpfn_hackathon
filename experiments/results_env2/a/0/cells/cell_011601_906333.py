import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet'); tt = agent_api.train_targets()
g = tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median','count'])
print(g.round(2))
# proxy for level at validation snapshots: exp4w_blend mean per snapshot (all rows incl. val)
g2 = feats.groupby('snapshot_day').exp4w_blend.agg(['mean','median'])
print(g2.round(2).loc[[403,431,459,487,515,543]])
# also spend_28 mean per snapshot
print(feats.groupby('snapshot_day').spend_28.mean().round(2).loc[[95,207,319,403,431,459,487,515,543]])