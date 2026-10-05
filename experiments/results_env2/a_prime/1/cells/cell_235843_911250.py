
import pandas as pd, numpy as np, time
t0=time.time()
e012 = agent_api.load_saved('e012_style.parquet')
r = agent_api.load_saved('rawrec.parquet').rename(columns={'g':'snapshot_day'})
M = e012.merge(r, on=['household_key','snapshot_day'], how='left')
print(M.shape)
first = [c for c in r.columns if c not in ('household_key','snapshot_day')][0]
print("missing rows:", int(M[first].isna().sum()))
print("corr sp28 vs spend_28:", round(float(np.corrcoef(M.sp28, M.spend_28)[0,1]),4))
print("mean abs diff:", round(float((M.sp28-M.spend_28).abs().mean()),3))
print(M[['household_key','snapshot_day','sp28','spend_28','tr28','trips_28','ew','d_ewma_spend_hl28']].head(5))
print("secs", round(time.time()-t0,1))
