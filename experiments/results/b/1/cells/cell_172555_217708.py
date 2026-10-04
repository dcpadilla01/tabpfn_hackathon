import pandas as pd, numpy as np, agent_api
base = agent_api.load_saved('e015_base.parquet')
med  = agent_api.load_saved('e019_medcombo.parquet')
print('base', base.shape, 'med', med.shape)
new_cols = [c for c in med.columns if c not in base.columns]
print('new cols', len(new_cols), new_cols)
print(med[new_cols].describe().T[['mean','std','min','max']].round(2).to_string())
print('days', sorted(med.snapshot_day.unique()))
print('dupkeys', med.duplicated(['household_key','snapshot_day']).sum())
consts = [c for c in med.columns if med[c].nunique(dropna=False) <= 1]
print('consts', consts)
med2 = med.drop(columns=consts)
p = agent_api.save_table(med2, 'e019_medcombo_final')
print('PATH', p, med2.shape)