def grid_trips(view, snapshot_day):
    import numpy as np, pandas as pd
    KS = np.arange(14)
    s = int(snapshot_day)
    hh = pd.Index(view.households)
    tx = view.table("transactions")
    tx = tx[tx["day"] <= s]
    b = ((s - tx["day"]) // 28).astype(int)
    tx = tx.assign(_b=b)
    tx = tx[tx["_b"] < 14]
    sp = tx.groupby(["household_key","_b"])["sales_value"].sum().unstack("_b").reindex(index=hh, columns=range(14))
    tr = tx.groupby(["household_key","_b"])["basket_id"].nunique().unstack("_b").reindex(index=hh, columns=range(14))
    fd = tx.groupby("household_key")["day"].min().reindex(hh)
    ld = tx.groupby("household_key")["day"].max().reindex(hh)
    S = np.nan_to_num(sp.values.astype(float))
    T = np.nan_to_num(tr.values.astype(float))
    start = s - 28*KS - 27
    V = (start[None, :] >= fd.values[:, None])
    F = {}
    for k in range(14):
        F[f"g_lw{k}"] = np.where(V[:, k], np.log1p(S[:, k]), np.nan)
    for k in range(6):
        F[f"g_lt{k}"] = np.where(V[:, k], np.log1p(T[:, k]), np.nan)
    dsl = (s - ld).values.astype(float)
    F["g_dsl"] = dsl
    F["g_expact"] = np.exp(-dsl/56.0)
    out = pd.DataFrame(F, index=hh)
    out.index.name = "household_key"
    return out

f = agent_api.build_features(grid_trips)
print("grid+trips ok", f.shape)
