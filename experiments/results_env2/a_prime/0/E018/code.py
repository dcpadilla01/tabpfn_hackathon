
import pandas as pd, numpy as np

names = ["e016_merged.parquet","e017_merged.parquet","e013_storeprod.parquet",
         "e017_phase.parquet","e016_batch2.parquet","e008_spendproc.parquet"]
tabs = {}
for nm in names:
    t = load_saved(nm)
    tabs[nm] = t
    dups = t.duplicated(["household_key","snapshot_day"]).sum()
    print(nm, t.shape, "dups:", dups)

t16, t17, t13 = tabs["e016_merged.parquet"], tabs["e017_merged.parquet"], tabs["e013_storeprod.parquet"]
k16 = set(map(tuple, t16[["household_key","snapshot_day"]].to_numpy()))
k17 = set(map(tuple, t17[["household_key","snapshot_day"]].to_numpy()))
print("key sets equal 16 vs 17:", k16==k17, "n_keys:", len(k16))

c13, c16, c17 = set(t13.columns), set(t16.columns), set(t17.columns)
print("c13<=c16:", c13<=c16, " c13<=c17:", c13<=c17)
incr16, incr17 = c16-c13, c17-c13
print("incr16:", len(incr16), "incr17:", len(incr17), "overlap:", len(incr16&incr17))
print("incr16 sample:", sorted(incr16)[:10])
print("incr17 sample:", sorted(incr17)[:10])
print("dtypes16:", dict(t16.dtypes.astype(str).value_counts()))
print("dtypes17:", dict(t17.dtypes.astype(str).value_counts()))
print("object cols16:", [c for c in t16.columns if t16[c].dtype==object][:12])
print("object cols17:", [c for c in t17.columns if t17[c].dtype==object][:12])


# ---- cell ----

import pandas as pd, numpy as np

tt = train_targets()
print("train target rows:", tt.shape)
g = tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median","std","count"])
print(g.round(2))

t17 = load_saved("e017_merged.parquet")
cols = list(t17.columns)
import re
# find RFM-ish spend columns to reuse for relative features
cand = [c for c in cols if re.search(r"(spend|sp_|s28|s56|ewma|ma4|mean)", c, re.I)]
print(len(cand))
print(sorted(cand)[:60])
print([c for c in cols if "tenure" in c or "day" in c][:20])

# departments/commodities for possible later use
snap = snapshot()
tx = snap.transactions
print("tx shape", tx.shape)
print("n departments", tx.product_id.map(dict(zip(snap.products.product_id, snap.products.department))).nunique() if hasattr(snap,'products') else "n/a")


# ---- cell ----

import pandas as pd, numpy as np, re
t8  = load_saved("e008_spendproc.parquet")
t12 = load_saved("e012_outcome2.parquet")
t17 = load_saved("e017_merged.parquet")

print("E008 incr cols:", [c for c in t8.columns if c.startswith(("sp_","roll","w28","proc"))][:40])
print()
print("E012 te2 cols:", [c for c in t12.columns if c.startswith("te2")])
print()
print("E017 c17 cols:", sorted([c for c in t17.columns if c.startswith("c17")]))
print()
print("E013 store/prod cols:", sorted([c for c in t17.columns if c.startswith(("st_","pr_","bk"))]))


# ---- cell ----

import pandas as pd, numpy as np

def dyn_fn(view, T):
    tx = view.table("transactions")
    hs = view.households
    if hasattr(hs, "columns") and "household_key" in getattr(hs, "columns", []):
        hh_list = pd.Index(hs["household_key"], name="household_key")
    else:
        hh_list = pd.Index(hs.index, name="household_key")
    hhset = set(hh_list)
    tx = tx[(tx["day"] <= T) & (tx["household_key"].isin(hhset))]
    out = pd.DataFrame(index=hh_list)

    def wsum(lo, hi, col="sales_value"):
        m = tx[(tx["day"] >= lo) & (tx["day"] <= hi)]
        return m.groupby("household_key")[col].sum().reindex(hh_list).fillna(0.0)

    w = [wsum(T - 28*k - 27, T - 28*k) for k in range(6)]   # w0 newest .. w5
    s4, s4l1 = w[0], w[1]
    s8  = wsum(T-55, T); s12 = wsum(T-83, T); s16 = wsum(T-111, T)

    out["dyn_mom1"]   = s4/(s4l1 + 1.0)
    out["dyn_r4_12"]  = s4/(s12/3.0 + 1.0)
    out["dyn_r8_16"]  = (s8/2.0)/(s16/4.0 + 1.0)
    out["dyn_d4_l1"]  = s4 - s4l1

    # weekly bucket matrices (chronological: col 0 oldest)
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

    W = np.stack([x.to_numpy() for x in w], axis=1)      # cols w0..w5
    d = -np.diff(W, axis=1)                               # newer minus older
    up = np.zeros(len(W)); dn = np.zeros(len(W))
    for j in range(5):
        up = np.where(d[:, j] > 0, up + 1, 0)
        dn = np.where(d[:, j] < 0, dn + 1, 0)
    out["dyn_n_up"] = up; out["dyn_n_down"] = dn
    out["dyn_pct_cur"] = (W <= W[:, [0]]).sum(1)/6.0

    med4, med8 = s4.median(), s8.median()
    out["dyn_pr4"] = s4/(med4 + 1.0)
    out["dyn_pr8"] = s8/(med8 + 1.0)

    bd = tx[["household_key", "day"]].drop_duplicates().sort_values(["household_key", "day"])
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

# dry-run on three snapshots
for day in [95, 263, 459]:
    v = snapshot(day)
    df = dyn_fn(v, day)
    nn = df.isna().mean().round(3)
    print(day, df.shape, "max NaN frac:", nn.max(), "| cols with NaN>0:", dict(nn[nn > 0]))
    print(df[["dyn_slope13","dyn_mom1","dyn_pr4","dyn_overdue","dyn_zerorun_cur"]].describe().round(2).loc[["mean","50%","min","max"]])


# ---- cell ----

v = snapshot(95)
print([a for a in dir(v) if not a.startswith("_")])
print("households:", type(v.households), v.households)
print("day:", v.day, "week:", v.week)


# ---- cell ----

def probe(view, T):
    print("inside fn: day", view.day, "households type:", type(view.households))
    hs = view.households
    if hs is not None:
        print("households sample:", hs[:5] if hasattr(hs, "__getitem__") else hs)
    return pd.DataFrame({"x": 1.0}, index=pd.Index(sorted(view.transactions.household_key.unique()), name="household_key"))

out = build_features(probe)
print(out.shape, out.columns.tolist())


# ---- cell ----

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


# ---- cell ----

import pandas as pd
t17 = load_saved("e017_merged.parquet")
dyn = load_saved("e018_dyn.parquet")
m = t17.merge(dyn.drop(columns=["snapshot_day"]), on="household_key", how="left")
print(m.shape, "dups:", m.duplicated(["household_key","snapshot_day"]).sum())
print("NaN frac max:", m.isna().mean().max().round(3))
save_table(m, "e018_merged.parquet")


# ---- cell ----

import pandas as pd
dyn = load_saved("e018_dyn.parquet")
print(type(dyn), dyn.shape)
print(dyn.index.dtype, dyn.index.name)
print("cols:", dyn.columns.tolist()[:5])
print("is household_key a column?", "household_key" in dyn.columns)
print("index dups:", dyn.index.duplicated().sum())
t17 = load_saved("e017_merged.parquet")
print("t17 hk dtype:", t17.household_key.dtype, "dyn idx dtype:", dyn.index.dtype)


# ---- cell ----

import pandas as pd
t17 = load_saved("e017_merged.parquet")
dyn = load_saved("e018_dyn.parquet")
m = t17.merge(dyn, on=["household_key","snapshot_day"], how="left")
print(m.shape, "dups:", m.duplicated(["household_key","snapshot_day"]).sum())
print("NaN frac max:", round(m.isna().mean().max(),3))
save_table(m, "e018_merged.parquet")


# ---- cell ----

import pandas as pd
m = load_saved("e018_merged.parquet")
nn = m.isna().mean().sort_values(ascending=False)
print(nn.head(10).round(3))
# check the same col in the dyn table itself
dyn = load_saved("e018_dyn.parquet")
print("dyn NaN:", dyn.isna().mean().sort_values(ascending=False).head(5).round(3))
print("t17 NaN max:", t17.isna().mean().max().round(3))
