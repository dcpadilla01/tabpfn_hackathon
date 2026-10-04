import agent_api, numpy as np, pandas as pd

logs = []
def peek(view, sd):
    logs.append((sd, len(view.households), view.day, view.transactions.day.max(), view.transactions.shape[0]))
    return pd.DataFrame({'x':[1]}, index=[1,2,3])

df = agent_api.build_features(peek)
print("DONE", len(logs))
for o in logs: print(o)
