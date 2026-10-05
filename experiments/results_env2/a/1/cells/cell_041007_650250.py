import agent_api as api, pandas as pd, numpy as np
# Check e016_allpreds columns on train rows: are they OOF? compare to target
ap = api.load_saved('e016_allpreds.parquet')
tt = api.train_targets()
m = ap.merge(tt, on=['household_key','snapshot_day'])
print(m.shape)
for c in ['pq','pl','pc','pa']:
    print(c, "MAE:", np.abs(m[c]-m['future_spend_4w']).mean().round(3), "bias:", (m[c]-m['future_spend_4w']).mean().round(3))
# blend of pq..pa on train rows
m['blend'] = m[['pq','pl','pc','pa']].mean(axis=1)
print("mean blend MAE:", np.abs(m['blend']-m['future_spend_4w']).mean().round(3))
sq = api.load_saved('e016_sqpreds.parquet')
m2 = sq.merge(tt, on=['household_key','snapshot_day'])
for c in ['sq0','sq1','sq2']:
    print(c, "MAE:", np.abs(m2[c]-m2['future_spend_4w']).mean().round(3))
