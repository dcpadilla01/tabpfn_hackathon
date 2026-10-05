import agent_api, pandas as pd
e8 = agent_api.load_saved('e008_level_shape.parquet')
m = e8
for name in ['candB_g','candD_su','candE_td']:
    t = agent_api.load_saved(name+'.parquet')
    newc = [c for c in t.columns if c not in m.columns]
    m = m.merge(t[['household_key','snapshot_day']+newc], on=['household_key','snapshot_day'], how='left')
assert len(m)==len(e8)
p = agent_api.save_table(m, 'candGST')
print('candGST', m.shape, list(m.columns)[-11:])
