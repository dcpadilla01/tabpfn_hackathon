import numpy as np, pandas as pd
import agent_api as A

def fn(view, t):
    tx = view.table("transactions")
    piv = tx.groupby(["household_key","day"]).sales_value.sum().unstack(fill_value=0.0)
    piv = piv.reindex(columns=np.arange(1, t+1), fill_value=0.0)
    hh = piv.index.to_numpy(); H = len(hh)
    C = np.zeros((H, t+1)); C[:,1:] = piv.values.cumsum(1)
    def sp(a,b): return C[:, np.clip(b,0,t)] - C[:, np.clip(a-1,0,t)]
    act = piv.values > 0
    def last_le(s):
        a2 = act.copy(); a2[:, s:] = False
        has = a2.any(1)
        return np.where(has, t - a2[:,::-1].argmax(1), 0)
    l_t = last_le(t)
    ss = [t-28] + ([t-56] if t-56 >= 57 else [])
    Xs, Ys, Rs, Gs = [], [], [], []
    for s in ss:
        F = np.zeros((H, 10))
        for j in range(8): F[:,j] = np.log1p(np.maximum(sp(s-7*(j+1)+1, s-7*j), 0))
        F[:,8] = np.log1p(np.maximum(sp(s-27, s), 0))
        F[:,9] = np.clip(s - last_le(s), 0, 56)/56.0
        yref = sp(s+1, s+28); w1ref = sp(s-27, s)
        ok = (last_le(s) >= 57)
        Xs.append(F[ok]); Ys.append(yref[ok])
        Rs.append(np.clip(yref[ok]/np.maximum(w1ref[ok], 20.0), 0, 4))
        Gs.append(np.where(ok)[0])
    Xs = np.vstack(Xs); Ys = np.concatenate(Ys); Rs = np.concatenate(Rs); Gs = np.concatenate(Gs)
    mu = Xs.mean(0); sd = Xs.std(0) + 1e-9
    Zr = (Xs - mu)/sd
    Ft = np.zeros((H, 10))
    for j in range(8): Ft[:,j] = np.log1p(np.maximum(sp(t-7*(j+1)+1, t-7*j), 0))
    Ft[:,8] = np.log1p(np.maximum(sp(t-27, t), 0))
    Ft[:,9] = np.clip(t - l_t, 0, 56)/56.0
    Zt = (Ft - mu)/sd
    D = np.sqrt(np.maximum((Zt**2).sum(1)[:,None] + (Zr**2).sum(1)[None,:] - 2*Zt@Zr.T, 0))
    w1t = sp(t-27, t)
    out30 = np.zeros(H); out10 = np.zeros(H); outrat = np.zeros(H)
    for a in range(H):
        d = D[a].copy(); d[Gs == a] = np.inf
        order = np.argsort(d)[:30]
        w30 = 1.0/(d[order] + 0.5)
        out30[a] = (w30*Ys[order]).sum()/w30.sum()
        outrat[a] = w1t[a]*np.clip((w30*Rs[order]).sum()/w30.sum(), 0, 4)
        w10 = 1.0/(d[order[:10]] + 0.5)
        out10[a] = (w10*Ys[order[:10]]).sum()/w10.sum()
    out = pd.DataFrame({"tw_k30": out30, "tw_k10": out10, "tw_ratio": outrat,
                        "tw_diff": out30 - w1t}, index=pd.Index(hh, name="household_key"))
    return out.reindex(view.households)

feats = A.build_features(fn)
print("built:", feats.shape, "| nan:", feats.isna().sum().sum())

e = A.load_saved("e011_table.parquet")
mrg = e.merge(feats.reset_index(), on=["household_key","snapshot_day"], how="left")
print("merged:", mrg.shape, "| tw nan:", mrg[["tw_k30","tw_k10","tw_ratio","tw_diff"]].isna().sum().to_dict())
assert mrg.shape[0] == e.shape[0]
path = A.save_table(mrg, "twin_v1")
print("saved:", path)
