import numpy as np, pandas as pd, agent_api as A
def fn(view, snapshot_day):
    tx = view.table("transactions")
    hhs = view.households
    print("snap", snapshot_day, "n hhs", len(hhs), "tx rows", len(tx))
    print("hhs type/sample:", type(hhs[0]) if len(hhs) else None, hhs[:2])
    print("tx hh dtype:", tx.household_key.dtype, "sample:", tx.household_key.iloc[:2].tolist() if len(tx) else None)
    print("tx day max:", tx.day.max() if len(tx) else None)
    sub = tx[tx.household_key.isin(set(hhs))]
    print("after isin:", len(sub))
    hh0 = hhs[0]
    print("rows for hh0:", len(tx[tx.household_key==hh0]))
    return pd.DataFrame(index=pd.Index(hhs, name="household_key"), data={"x":1.0})
aw = A.build_features(fn)
