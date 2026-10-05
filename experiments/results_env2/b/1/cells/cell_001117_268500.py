import numpy as np, pandas as pd

def base_grid(view, snapshot_day):
    s = int(snapshot_day)
    hh = pd.Index(view.households)
    tx = view.table("transactions")
    tx = tx[tx["day"] <= s]
    b = ((s - tx["day"]) // 28).astype(int)
    tx = tx.assign(_b=b); tx = tx[tx["_b"] < 14]
    sp = tx.groupby(["household_key","_b"])["sales_value"].sum().unstack("_b").reindex(index=hh, columns=range(14))
    fd = tx.groupby("household_key")["day"].min().reindex(hh)
    S = np.nan_to_num(sp.values.astype(float))
    KS = np.arange(14)
    start = s - 28*KS - 27
    V = (start[None, :] >= fd.values[:, None])
    return hh, S, V, KS

def mk_variant(name):
    def fn(view, snapshot_day, name=name):
        import numpy as np, pandas as pd
        hh, S, V, KS = base_grid(view, snapshot_day)
        Vf = V.astype(float); V9 = Vf[:, :10]
        m9 = V9.sum(1)
        F = {}
        if name == "A_matmul":
            Sk = V9 @ KS[:10]; Skk = V9 @ (KS[:10]**2)
            F["x"] = Sk + Skk
        elif name == "B_slope":
            Sk = V9 @ KS[:10]; Skk = V9 @ (KS[:10]**2)
            sum9 = (S[:, :10]*V9).sum(1); Skv = (S[:, :10]*V9) @ KS[:10]
            den = m9*Skk - Sk**2
            slope = np.where(den > 1e-9, (m9*Skv - Sk*sum9)/np.maximum(den, 1e-9), 0.0)
            F["x"] = slope
        elif name == "C_zerovar":
            sum9 = (S[:, :10]*V9).sum(1); mean9 = sum9/np.maximum(m9, 1)
            zero9 = ((S[:, :10] <= 0)*V9).sum(1)/np.maximum(m9, 1)
            sq9 = ((S[:, :10]**2)*V9).sum(1)
            var9 = np.maximum(sq9/np.maximum(m9, 1) - mean9**2, 0)
            F["x"] = zero9; F["y"] = np.sqrt(var9)
        elif name == "D_shrink":
            sum9 = (S[:, :10]*V9).sum(1); mean9 = sum9/np.maximum(m9, 1)
            G = S[:, 0].mean()
            F["x"] = (m9*mean9 + 1*G)/(m9 + 1)
            F["y"] = (m9*mean9 + 3*G)/(m9 + 3)
        elif name == "E_loop":
            n = len(hh)
            med = np.full(n, np.nan); ew6 = np.full(n, np.nan)
            for i in range(n):
                kk = KS[:10][V[i, :10]]; vv = S[i, :10][V[i, :10]]
                if vv.size:
                    med[i] = np.median(vv)
                    w6 = 0.6**kk
                    ew6[i] = (w6*vv).sum()/w6.sum()
            F["x"] = med; F["y"] = ew6
        out = pd.DataFrame(F, index=hh)
        out.index.name = "household_key"
        return out
    return fn

for name in ["A_matmul", "B_slope", "C_zerovar", "D_shrink", "E_loop"]:
    try:
        f = agent_api.build_features(mk_variant(name))
        print(name, "OK", f.shape)
    except Exception as e:
        print(name, "FAIL", type(e).__name__, str(e)[:120])
