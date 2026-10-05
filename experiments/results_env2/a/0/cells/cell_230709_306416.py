
import agent_api, pandas as pd, numpy as np

feats = agent_api.load_saved('feats_v3.parquet')
print("feats_v3 shape:", feats.shape)
print("columns:", list(feats.columns))

tt = agent_api.train_targets()
print("\ntrain_targets shape:", tt.shape)
m = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w']
print("\ntarget describe:\n", y.describe())
print("zero fraction:", (y==0).mean(), " zeros:", (y==0).sum(), "of", len(y))
print("quantiles:", np.percentile(y, [50,75,90,95,99]))

pred7 = agent_api.load_saved('pred_e007.parquet')
print("\npred_e007 shape:", pred7.shape, "cols:", list(pred7.columns))
print(pred7.head(3))
print("pred range:", pred7['prediction'].min(), pred7['prediction'].max())
