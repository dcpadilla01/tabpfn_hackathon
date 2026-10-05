def grid_only(view, snapshot_day):
    s = int(snapshot_day)
    hh = pd.Index(view.households)
    tx = view.table("transactions")
    tx = tx[tx["day"] <= s]
    b = ((s - tx["day"]) // 28).astype(int)
    tx = tx.assign(_b=b)
    tx = tx[tx["_b"] < 14]
    sp = tx.groupby(["household_key","_b"])["sales_value"].sum().unstack("_b").reindex(index=hh, columns=range(14))
    S = np.nan_to_num(sp.values.astype(float))
    fd = tx.groupby("household_key")["day"].min().reindex(hh)
    start = s - 28*KS - 27
    V = (start[None, :] >= fd.values[:, None])
    F = {f"g_lw{k}": np.where(V[:, k], np.log1p(S[:, k]), np.nan) for k in range(14)}
    out = pd.DataFrame(F, index=hh)
    out.index.name = "household_key"
    return out

f = agent_api.build_features(grid_only)
print("grid ok", f.shape)
