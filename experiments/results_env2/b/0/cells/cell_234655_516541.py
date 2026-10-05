import agent_api, pandas as pd
t2 = agent_api.load_saved('e015_norm_windows.parquet').drop(columns=['index'])
print(t2.shape, t2.columns.tolist()[-8:])
path = agent_api.save_table(t2, 'e015_norm_windows')
print(path)
