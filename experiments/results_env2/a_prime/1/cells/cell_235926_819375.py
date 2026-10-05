
import pandas as pd, numpy as np
e012 = agent_api.load_saved('e012_style.parquet')
r = agent_api.load_saved('rawrec.parquet').rename(columns={'g':'snapshot_day'})
M = e012.merge(r, on=['household_key','snapshot_day'], how='left')
first = 'sp28'
print("rows:", len(M), "missing:", int(M[first].isna().sum()))
print("corr sp28 vs spend_28:", round(float(np.corrcoef(M.sp28, M.spend_28)[0,1]),4))
print("mean abs diff sp28 vs spend_28:", round(float((M.sp28-M.spend_28).abs().mean()),3))
print(M[['household_key','snapshot_day','sp28','spend_28','tr28','trips_28']].head(6))
# check zero-target households: does sp28==0 align?
tt = agent_api.train_targets()
M2 = M.merge(tt, on=['household_key','snapshot_day'])
print("n train rows:", len(M2))
print("P(y=0 | sp28=0):", round(float((M2.loc[M2.sp28==0,'future_spend_4w']==0).mean()),3))
print("mean y | sp28=0:", round(float(M2.loc[M2.sp28==0,'future_spend_4w'].mean()),2))
print("mean y | sp28>0:", round(float(M2.loc[M2.sp28>0,'future_spend_4w'].mean()),2))
