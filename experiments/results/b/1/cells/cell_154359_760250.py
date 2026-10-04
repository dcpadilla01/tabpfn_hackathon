t = agent_api.load_saved('e009_macro.parquet')
d = t.describe().T[['mean','std','min','max']]
print(d.to_string())
