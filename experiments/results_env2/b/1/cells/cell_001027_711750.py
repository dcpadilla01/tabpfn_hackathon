def grid_vec(view, snapshot_day):
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
    fd = tx.groupby("household_key")["day"].min().reindex(hh)
    S = np.nan_to_num(sp.values.astype(float))
    start = s - 28*KS - 27
    V = (start[None, :] >= fd.values[:, None])
    Vf = V.astype(float); V9 = Vf[:, :10]
    m9 = V9.sum(1)
    sum9 = (S[:, :10]*V9).sum(1)
    mean9 = sum9/np.maximum(m9, 1)
    zero9 = ((S[:, :10] <= 0)*V9).sum(1)/np.maximum(m9, 1)
    sq9 = ((S[:, :10]**2)*V9).sum(1)
    var9 = np.maximum(sq9/np.maximum(m9, 1) - mean9**2, 0)
    Sk = V9 @ KS[:10]; Skk = V9 @ (KS[:10]**2); Skv = (S[:, :10]*V9) @ KS[:10]
    den = m9*Skk - Sk**2
    slope = np.where(den > 1e-9, (m9*Skv - Sk*sum9)/np.maximum(den, 1e-9), 0.0)
    G = S[:, 0].mean()
    F = {}
    F["g_wmean"] = mean9; F["g_wstd"] = np.sqrt(var9)
    F["g_wzero"] = zero9; F["g_wslope"] = slope
    F["g_shr1"] = (m9*mean9 + 1*G)/(m9 + 1)
    F["g_shr3"] = (m9*mean9 + 3*G)/(m9 + 3)
    out = pd.DataFrame(F, index=hh)
    out.index.name = "household_key"
    return out

f = agent_api.build_features(grid_vec)
print("vec ok", f.shape)
