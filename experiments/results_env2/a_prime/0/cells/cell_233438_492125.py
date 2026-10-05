import agent_api, pandas as pd
e13 = agent_api.load_saved("e013_storeprod.parquet")
e12 = agent_api.load_saved("e012_outcome2.parquet")
c13, c12 = set(e13.columns), set(e12.columns)
print("e013 extra vs e012:", sorted(c13-c12))
print("e012 extra vs e013:", sorted(c12-c13))
print("e013 tail cols:", list(e13.columns)[-40:])
b1 = agent_api.load_saved("cand_batch1.parquet")
print("cand_batch1 extra vs e013:", sorted(set(b1.columns)-c13))
