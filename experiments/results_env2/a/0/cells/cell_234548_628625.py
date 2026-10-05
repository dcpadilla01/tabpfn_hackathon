
import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
print("feats dup rows:", feats.duplicated(['household_key','snapshot_day']).sum())
print("feats nunique hh:", feats.household_key.nunique(), "snap days:", sorted(feats.snapshot_day.unique()))
print("tt rows:", len(tt), "tt dup:", tt.duplicated(['household_key','snapshot_day']).sum())
# manual check on one snapshot
d = 95
f95 = feats[feats.snapshot_day==d].set_index('household_key')
t95 = tt[tt.snapshot_day==d].set_index('household_key')['future_spend_4w']
common = f95.index.intersection(t95.index)
print("snap95: n feat", len(f95), "n target", len(t95), "common", len(common))
print("corr spend_28:", np.corrcoef(f95.loc[common,'spend_28'], t95.loc[common])[0,1])
print("corr spend_84:", np.corrcoef(f95.loc[common,'spend_84'], t95.loc[common])[0,1])
print("corr exp4w_blend:", np.corrcoef(f95.loc[common,'exp4w_blend'], t95.loc[common])[0,1])
# check whether same household appears at multiple snapshot days with same feature values
sub = feats[feats.household_key==feats.household_key.iloc[0]][['snapshot_day','spend_28','spend_84','exp4w_blend']]
print(sub)
