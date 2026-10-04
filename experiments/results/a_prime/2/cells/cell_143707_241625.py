import pandas as pd
t = agent_api.load_saved('temporal_structure.parquet')
if 'index' in t.columns: t = t.drop(columns=['index'])
print(t.shape)
print(agent_api.save_table(t, 'temporal_structure.parquet'))