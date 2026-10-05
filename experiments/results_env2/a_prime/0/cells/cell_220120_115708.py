import numpy as np, pandas as pd, agent_api as A
aw = A.load_saved("aw_hist.parquet") if False else None
# rebuild quickly (cheap) and inspect structure
def fn(view, snapshot_day):
    tx = view.table("transactions")
    hhs = view.households
    g = tx[tx.household_key.isin(set(hhs))].groupby(["household_key","day"])["sales_value"].sum().reset_index()
    R = pd.DataFrame(index=pd.Index(hhs, name="household_key"))
    for k in range(1,14):
        lo, hi = snapshot_day-28*k+1, snapshot_day-28*k
        w = g[(g.day>=lo)&(g.day<=hi)].groupby("household_key")["sales_value"].sum()
        R[f"aw_{k}"] = w.reindex(R.index).fillna(0.0)
    return R
aw = A.build_features(fn)
print(aw.columns.tolist()[:6], aw.shape)
print(aw.head(3))
