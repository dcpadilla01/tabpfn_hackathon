import agent_api, numpy as np, pandas as pd

def peek(view, sd):
    print("sd:", sd, "households type:", type(view.households))
    print("first 5:", list(view.households[:5]) if view.households is not None else None)
    print("n:", len(view.households) if view.households is not None else 0)
    print("day:", view.day)
    # transactions inside a snapshot: check max day
    t = view.transactions
    print("txn max day:", t.day.max(), "shape:", t.shape)
    return pd.DataFrame({'x':[1]}, index=view.households[:5])

df = agent_api.build_features(peek)
