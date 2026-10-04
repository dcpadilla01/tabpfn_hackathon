import agent_api, numpy as np, pandas as pd

def peek(view, sd):
    print("sd:", sd, "households:", type(view.households), len(view.households) if view.households is not None else None)
    print("day:", view.day, "week:", view.week)
    return pd.DataFrame({'x':[1]}, index=view.households[:5])

df = agent_api.build_features(peek)
print(df.head())
