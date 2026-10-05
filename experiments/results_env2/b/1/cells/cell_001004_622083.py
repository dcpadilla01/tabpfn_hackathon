def grid_stats(view, snapshot_day):
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
    n = len(hh)
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
    med = np.full(n, np.nan); mn = np.full(n, np.nan); mx = np.full(n, np.nan)
    ew6 = np.full(n, np.nan); ew85 = np.full(n, np.nan)
    for i in range(n):
        kk = KS[:10][V[i, :10]]; vv = S[i, :10][V[i, :10]]
        if vv.size:
            med[i] = np.median(vv); mn[i] = vv.min(); mx[i] = vv.max()
            w6 = 0.6**kk; w85 = 0.85**kk
            ew6[i] = (w6*vv).sum()/w6.sum(); ew85[i] = (w85*vv).sum()/w85.sum()
    G = S[:, 0].mean()
    F = {}
    F["g_wmean"] = mean9; F["g_wmed"] = med; F["g_wstd"] = np.sqrt(var9)
    F["g_wmin"] = mn; F["g_wmax"] = mx
    F["g_wzero"] = zero9; F["g_wslope"] = slope
    F["g_ew6"] = ew6; F["g_ew85"] = ew85
    F["g_shr1"] = (m9*mean9 + 1*G)/(m9 + 1)
    F["g_shr3"] = (m9*mean9 + 3*G)/(m9 + 3)
    out = pd.DataFrame(F, index=hh)
    out.index.name = "household_key"
    return out

f = agent_api.build_features(grid_stats)
print("stats ok", f.shape)
