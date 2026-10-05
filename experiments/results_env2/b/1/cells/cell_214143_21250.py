
import agent_api, pandas as pd
e2 = agent_api.load_saved('e002_mix.parquet')
e3 = agent_api.load_saved('e003_momentum.parquet')
new_cols = [c for c in e3.columns if c.startswith(('m_','r_','i_','topstore','wkstd','wkmax','splag1y','nact14','spt_','spd_'))]
print("new:", new_cols)
mm = e2.merge(e3[['household_key','snapshot_day']+new_cols], on=['household_key','snapshot_day'], how='inner')
print(mm.shape)
print("rows per snapshot:", mm.groupby('snapshot_day').size().to_dict())
p = agent_api.save_table(mm, 'e003_full.parquet')
print(p)
