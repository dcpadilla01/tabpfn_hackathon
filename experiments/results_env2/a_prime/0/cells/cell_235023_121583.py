
import pandas as pd, numpy as np

def dyn_fn(view, T):
    tx = view.table("transactions")
    hh_list = pd.Index(view.households, name="household_key")
    hhset = set(hh_list)
    tx = tx[tx["household_key"].isin(hhset)]
    out = pd.DataFrame(index=hh_list)

    def wsum(lo, hi):
        m = tx[(tx["day"] >= lo) & (tx["day"] <= hi)]
        return m.groupby("household_key")["sales_value"].sum().reindex(hh_list).fillna(0.0)

    w = [wsum(T - 28*k - 27, T - 28*k) for k in range(6)]   # w0 newest .. w5
    s4, s4l1 = w[0], w[1]
    s8  = wsum(T-55, T); s12 = wsum(T-83, T); s16 = wsum(T-111, T)

    out["dyn_mom1"]   = s4/(s4l1 + 1.0)
    out["dyn_r4_12"]  = s4/(s12/3.0 + 1.0)
    out["dyn_r8_16"]  = (s8/2.0)/(s16/4.0 + 1.0)
    out["dyn_d4_l1"]  = s4 - s4l1

    def weekly(n):
        start = T - 7*n + 1
        sub = tx[tx["day"] >= start]
        b = ((sub["day"] - start)//7).astype(int)
        pv = sub.groupby(["household_key", b])["sales_value"].sum().unstack()
        return pv.reindex(columns=range(n)).reindex(hh_list).fillna(0.0)

    p13, p26 = weekly(13), weekly(26)
    X13, X26 = p13.to_numpy(), p26.to_numpy()
    x13 = np.arange(13); c13 = x13 - x13.mean()
    out["dyn_slope13"] = (X13*c13).sum(1)/((c13**2).sum())/(X13.mean(1) + 1.0)
    x26 = np.arange(26); c26 = x26 - x26.mean()
    out["dyn_slope26"] = (X26*c26).sum(1)/((c26**2).sum())/(X26.mean(1) + 1.0)
    out["dyn_cv13"] = X13.std(1)/(X13.mean(1) + 1e-9)
    out["dyn_cv26"] = X26.std(1)/(X26.mean(1) + 1e-9)
    xr = np.arange(8) - 3.5
    sr = (X26[:, 18:26]*xr).sum(1)/((xr**2).sum())
    sp = (X26[:, 10:18]*xr).sum(1)/((xr**2).sum())
    out["dyn_accel"] = (sr - sp)/(X26[:, 10:26].mean(1) + 1.0)
    part = np.partition(X13, -2, axis=1)[:, -2:]
    out["dyn_top2"] = part.sum(1)/(X13.sum(1) + 1e-9)
    Z = X26 <= 1e-9
    run = np.zeros(len(Z)); mx = np.zeros(len(Z))
    for j in range(26):
        run = np.where(Z[:, j], run + 1, 0); mx = np.maximum(mx, run)
    out["dyn_zerorun_max"] = mx; out["dyn_zerorun_cur"] = run
    m26 = X26.mean(1); sd26 = X26.std(1)
    A = (X26 - m26[:, None])/np.where(sd26 > 1e-9, sd26, np.nan)[:, None]
    out["dyn_ac1"] = np.nan_to_num(np.nanmean(A[:, :-1]*A[:, 1:], axis=1))

    W = np.stack([x.to_numpy() for x in w], axis=1)
    d = -np.diff(W, axis=1)
    up = np.zeros(len(W)); dn = np.zeros(len(W))
    for j in range(5):
        up = np.where(d[:, j] > 0, up + 1, 0)
        dn = np.where(d[:, j] < 0, dn + 1, 0)
    out["dyn_n_up"] = up; out["dyn_n_down"] = dn
    out["dyn_pct_cur"] = (W <= W[:, [0]]).sum(1)/6.0

    med4, med8 = s4.median(), s8.median()
    out["dyn_pr4"] = s4/(med4 + 1.0)
    out["dyn_pr8"] = s8/(med8 + 1.0)

    bd = tx[["household_key", "day", "basket_id"]].drop_duplicates().sort_values(["household_key", "day"])
    bd["gap"] = bd.groupby("household_key")["day"].diff()
    grec = bd[(bd["day"] > T-84) & (bd["gap"] > 0)].groupby("household_key")["gap"].mean().reindex(hh_list)
    gpri = bd[(bd["day"] > T-168) & (bd["day"] <= T-84) & (bd["gap"] > 0)].groupby("household_key")["gap"].mean().reindex(hh_list)
    out["dyn_gap_trend"] = grec - gpri
    g112 = bd[(bd["day"] > T-112) & (bd["gap"] > 0)].groupby("household_key")["gap"].median().reindex(hh_list)
    dsl = (T - tx.groupby("household_key")["day"].max()).reindex(hh_list).astype(float)
    out["dyn_overdue"] = g112 - dsl
    out["dyn_overdue_flag"] = ((dsl + g112) <= 28).astype(float)
    out["dyn_bk7"]  = tx[tx["day"] > T-7].groupby("household_key")["basket_id"].nunique().reindex(hh_list).fillna(0)
    out["dyn_bk14"] = tx[tx["day"] > T-14].groupby("household_key")["basket_id"].nunique().reindex(hh_list).fillna(0)

    bk = tx.groupby(["household_key", "basket_id"]).agg(day=("day", "first"), units=("quantity", "sum"), nprod=("product_id", "nunique"))
    rec_b = bk[bk["day"] > T-28]; pri_b = bk[(bk["day"] > T-56) & (bk["day"] <= T-28)]
    out["dyn_upb_trend"] = (rec_b.groupby(level=0)["units"].mean() - pri_b.groupby(level=0)["units"].mean()).reindex(hh_list)
    out["dyn_dpp_trend"] = (rec_b.groupby(level=0)["nprod"].mean() - pri_b.groupby(level=0)["nprod"].mean()).reindex(hh_list)
    bd84 = bd[bd["day"] > T-84].copy()
    bd84["is_wknd"] = (bd84["day"] % 7 >= 5)
    out["dyn_wknd_share"] = bd84.groupby("household_key")["is_wknd"].mean().reindex(hh_list)
    t84 = tx[tx["day"] > T-84].copy()
    t84["is_eve"] = (t84["trans_time"] >= 1700)
    out["dyn_eve_share"] = t84.groupby("household_key")["is_eve"].mean().reindex(hh_list)
    return out

df = build_features(dyn_fn)
print(df.shape, df.dtypes.astype(str).value_counts().to_dict())
nn = df.isna().mean()
print("cols with NaN:", dict(nn[nn > 0].round(3)))
print(df.drop(columns=["household_key","snapshot_day"]).describe().T.round(3)[["mean","50%","min","max"]].to_string())
save_table(df, "e018_dyn.parquet")
