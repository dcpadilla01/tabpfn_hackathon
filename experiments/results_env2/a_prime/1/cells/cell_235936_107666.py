
import pandas as pd, numpy as np
e012 = agent_api.load_saved('e012_style.parquet')
r = agent_api.load_saved('rawrec.parquet').rename(columns={'g':'snapshot_day'})
M = e012.merge(r, on=['household_key','snapshot_day'], how='left')
print("rows:", len(M), "missing:", int(M['sp28'].isna().sum()))
print("corr sp28 vs spend_28:", round(float(np.corrcoef(M.sp28, M.spend_28)[0,1]),4))
print("mean abs diff:", round(float((M.sp28-M.spend_28).abs().mean()),3))
# how many features total now
feats = [c for c in M.columns if c not in ('household_key','snapshot_day')]
print("total feature cols:", len(feats))
agent_api.save_table(M, 'e016_merged')
