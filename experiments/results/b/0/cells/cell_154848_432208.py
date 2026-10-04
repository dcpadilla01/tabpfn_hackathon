import pandas as pd, numpy as np
v = agent_api.snapshot()
print("week", v.week, "day", v.day)
print("KEYS", agent_api.KEYS, "TARGET", agent_api.TARGET)
e11 = agent_api.load_saved('e011_price.parquet')
print("E011", e11.shape)
print(list(e11.columns))
print(e11.head(2).T.head(30))