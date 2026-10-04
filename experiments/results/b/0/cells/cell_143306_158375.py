import agent_api, numpy as np, pandas as pd

out = []
def peek(view, sd):
    out.append((sd, type(view.households), list(view.households[:5]) if view.households is not None else None,
                len(view.households) if view.households is not None else 0, view.day,
                view.transactions.day.max(), view.transactions.shape))
    return pd.DataFrame({'x':[1]}, index=[1,2,3])

df = agent_api.build_features(peek)
print("DONE", len(out))
for o in out: print(o)
