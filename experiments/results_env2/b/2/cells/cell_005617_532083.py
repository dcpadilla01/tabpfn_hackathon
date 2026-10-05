
import agent_api, pandas as pd, numpy as np

def make_cal2(view, D):
    tx = view.table("transactions")
    hh = view.households
    try: hh = list(hh)
    except: pass
    tx = tx[tx["household_key"].isin(set(hh))]
    rows = {}
    for h, sub in tx.groupby("household_key", sort=False):
        days = sub["day"].values.astype(float)
        sp = sub["sales_value"].values.astype(float)
        o = np.argsort(days, kind="stable")
        days, sp = days[o], sp[o]
        first = days[0]
        cs = np.concatenate([[0.0], np.cumsum(sp)])
        def wsum(a, b):
            i = np.searchsorted(days, a, "left"); j = np.searchsorted(days, b, "right")
            return cs[j] - cs[i]
        Tcur = wsum(D-27, D)
        ts, Ts, Fs = [], [], []
        t = D - 28
        while t >= first + 84:
            Ts.append(wsum(t-27, t)); Fs.append(wsum(t+1, t+28)); ts.append(t)
            t -= 28
        n = len(ts)
        d = dict(c2_n=float(n), c2_tcur=Tcur)
        if n:
            T = np.array(Ts); F = np.array(Fs)
            fm, fmed, fsd = F.mean(), np.median(F), F.std()
            d.update(c2_f_mean=fm, c2_f_med=fmed, c2_f_std=fsd, c2_t_mean=T.mean())
            w = 0.5 ** np.arange(n)          # most recent pair weight 1
            d["c2_f_ew"] = float((F*w).sum()/w.sum())
            d["c2_ratio_ew"] = float((F*w).sum()/max((T*w).sum(), 1e-9))
            d["c2_ratio_last"] = float(F[-1]/max(T[-1], 1e-9))
            m0 = T <= 1e-9
            d["c2_f_zero"] = float(F[m0].mean()) if m0.any() else np.nan
            d["c2_n_zero"] = float(m0.sum())
            if n >= 2 and T.var() > 1e-9:
                sl, ic = np.polyfit(T, F, 1)
                k = n/(n+3.0)
                sl_sh = sl*k
                ic_sh = F.mean() - sl_sh*T.mean()
                d["c2_slope"] = float(sl_sh); d["c2_ic"] = float(ic_sh)
                d["c2_pred"] = float(ic_sh + sl_sh*Tcur)
                d["c2_resid"] = float((F - (ic+sl*T)).std())
                d["c2_corr"] = float(np.corrcoef(T, F)[0,1])
                m = min(n, 6)
                if np.var(T[-m:]) > 1e-9:
                    sl6, ic6 = np.polyfit(T[-m:], F[-m:], 1)
                    d["c2_slope6"] = float(sl6*k)
                    d["c2_pred6"] = float(ic6 + sl6*Tcur)
        rows[h] = d
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "household_key"
    return df.reindex(hh)

tab = agent_api.build_features(make_cal2)
print(tab.shape)
print(tab.columns.tolist())
print(tab.isna().mean().round(3).to_dict())
p = agent_api.save_table(tab, "cal2_v1")
print("saved:", p)
