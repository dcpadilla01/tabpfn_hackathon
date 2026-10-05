import numpy as np, pandas as pd
e003 = agent_api.load_saved("e003_union.parquet")
print("e003 shape:", e003.shape)
print("cols:", e003.columns.tolist())
tt = agent_api.train_targets()
m = e003.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("merged:", m.shape)
y = m["future_spend_4w"]
print("\ntarget describe:\n", y.describe())
print("\nzero share:", (y==0).mean())
num = m.select_dtypes(include=[np.number]).drop(columns=["snapshot_day","future_spend_4w"], errors="ignore")
corr = num.corrwith(y).sort_values()
print("\ntop positive corr:\n", corr.tail(15).to_string())
print("\ntop negative corr:\n", corr.head(8).to_string())


# ---- cell ----
v = agent_api.snapshot()
for name in ["campaigns","campaign_targets","coupon_redemptions","coupons","display_mailer","transactions","products"]:
    t = v.table(name)
    print(name, t.shape, t.columns.tolist())
print("\ncampaigns by description:")
print(v.table("campaigns").description.value_counts())
print("\nredemptions head:\n", v.table("coupon_redemptions").head())
print("\nredemptions per household describe:")
r = v.table("coupon_redemptions").groupby("household_key").size()
print(r.describe())
print("\ndisplay/mailer values:", v.table("display_mailer").display.value_counts().head(10).to_dict(), v.table("display_mailer").mailer.value_counts().head(10).to_dict())
print("\nday range tx:", v.table("transactions").day.min(), v.table("transactions").day.max())


# ---- cell ----
import numpy as np, pandas as pd
v = agent_api.snapshot()  # capped at 459
tx = v.table("transactions")[["household_key","day","sales_value","basket_id"]]

def win_spend(d0, d1):
    m = tx[(tx.day>=d0)&(tx.day<=d1)]
    return m.groupby("household_key").sales_value.sum()

# check: spend at snapshot 431 (window 432..459) vs lag13 window (68..95), lag1 (404..431)
fut = win_spend(432,459)
lag1 = win_spend(404,431); lag13 = win_spend(68,95); lag26 = win_spend(-260,-233)  # lag26 mostly unavailable
hh = fut.index
df = pd.DataFrame({"y":fut}).join(lag1.rename("lag1")).join(lag13.rename("lag13")).fillna(0.0)
df = df[df.index.isin(v.table("transactions").household_key.unique())]
print("n hh:", len(df))
print("corr lag1:", df.y.corr(df.lag1).round(3), " corr lag13:", df.y.corr(df.lag13).round(3))
# partial: does lag13 add beyond lag1?
import numpy as np
A = np.c_[np.ones(len(df)), df.lag1, df.lag13]
b,_,_,_ = np.linalg.lstsq(A, df.y, rcond=None)
pred = A@b
print("lag1-only MAE:", np.abs(df.lag1-df.y).mean().round(2), " lag1+lag13 OLS MAE:", np.abs(pred-df.y).mean().round(2))
# zero structure: P(y=0 | lag1=0)
print("P(y=0|lag1=0):", (df[df.lag1==0].y==0).mean().round(3), " P(y=0|lag1>0):", (df[df.lag1>0].y==0).mean().round(3))


# ---- cell ----
import numpy as np, pandas as pd

def fn(view, snap):
    tx = view.table("transactions")
    hh = pd.Index(view.households, name="household_key")
    out = pd.DataFrame(index=hh)
    day = tx["day"].values
    hv = tx["household_key"].values
    sv = tx["sales_value"].values
    bk = tx["basket_id"].values

    # --- seasonal lags: 4w window shifted back k cycles ---
    lags = np.full((len(hh), 13), np.nan)
    nb13 = np.zeros(len(hh))
    hpos = pd.Series(np.arange(len(hh)), index=hh)
    for k in range(1, 14):
        lo, hi = snap + 1 - 28*k, snap + 28 - 28*k
        if hi < 1:
            continue
        m = (day >= max(lo,1)) & (day <= hi)
        if not m.any():
            continue
        s = pd.Series(sv[m]).groupby(pd.Series(hv[m])).sum()
        s = s.reindex(hh).fillna(0.0)
        if lo >= 1:
            lags[:, k-1] = s.values
        else:
            lags[:, k-1] = np.nan  # partial window -> missing
        if k == 13:
            nb = pd.Series(bk[m]).groupby(pd.Series(hv[m])).nunique().reindex(hh).fillna(0.0)
            nb13 = nb.values
    for k in range(1, 14):
        out[f"seas_lag{k}_4w"] = lags[:, k-1]
    avail = ~np.isnan(lags)
    out["seas_lag13_ok"] = avail[:, 12]
    out["seas_lag12_ok"] = avail[:, 11]
    out["seas_lag11_ok"] = avail[:, 10]
    sm = np.nanmean(lags[:, 10:13], axis=1)
    sx = np.nanmax(lags[:, 10:13], axis=1)
    out["seas_mean_l11_13"] = sm
    out["seas_max_l11_13"] = sx
    out["seas_lag13_nbask"] = nb13
    out["seas_lag13_active"] = (np.nan_to_num(lags[:,12]) > 0).astype(float)
    sp4 = lags[:, 0]
    out["seas_ratio_l13_l1"] = np.nan_to_num(lags[:,12]) / (sp4 + 1.0)

    # --- zero-cycle share over history (inactivity regularity) ---
    first_day = pd.Series(day).groupby(pd.Series(hv)).min().reindex(hh)
    K = np.minimum(((snap - first_day.values + 1) // 28).astype(float), 13.0)
    K = np.maximum(K, 0)
    zeros = (np.nan_to_num(lags) == 0)
    ok = avail
    kk = np.arange(1, 14)[None, :].repeat(len(hh), 0).astype(float)
    valid = ok & (kk <= K[:, None]) & (K[:, None] > 0)
    zshare = np.where(valid.any(1), (zeros & valid).sum(1) / np.maximum(valid.sum(1), 1), np.nan)
    out["zero_cycle_share"] = zshare

    # --- recency-decayed spend ---
    rec = day >= snap - 167
    w = snap - day[rec]
    hh_r = pd.Series(hv[rec])
    for tau, nm in [(28, "decay28"), (56, "decay56")]:
        d = pd.Series(sv[rec] * (0.5 ** (w / tau))).groupby(hh_r).sum().reindex(hh).fillna(0.0)
        out[nm] = d.values

    # --- fine recency ---
    for win in [7, 14]:
        m = day > snap - win
        s = pd.Series(sv[m]).groupby(pd.Series(hv[m])).sum().reindex(hh).fillna(0.0)
        nb = pd.Series(bk[m]).groupby(pd.Series(hv[m])).nunique().reindex(hh).fillna(0.0)
        out[f"spend_{win}d"] = s.values
        out[f"nbask_{win}d"] = nb.values

    # --- basket-level structure (112d) ---
    m112 = day > snap - 112
    b = pd.Series(sv[m112]).groupby([pd.Series(hv[m112]), pd.Series(bk[m112])]).sum()
    bsum = b.groupby(level=0).sum().reindex(hh).fillna(0.0)
    bmax = b.groupby(level=0).max().reindex(hh).fillna(0.0)
    bstd = b.groupby(level=0).std().reindex(hh)
    out["basket_max_112"] = bmax.values
    out["basket_std_112"] = bstd.values
    out["basket_top1_share_112"] = bmax.values / (bsum.values + 1.0)

    # --- daily activity + weekend share (112d) ---
    dact = pd.Series(day[m112]).groupby(pd.Series(hv[m112])).nunique().reindex(hh).fillna(0.0)
    out["active_day_rate_112"] = (dact.values / 112.0)
    dow = (day % 7)
    mwk = m112 & (dow >= 5)
    swk = pd.Series(sv[mwk]).groupby(pd.Series(hv[mwk])).sum().reindex(hh).fillna(0.0)
    s112 = pd.Series(sv[m112]).groupby(pd.Series(hv[m112])).sum().reindex(hh).fillna(0.0)
    out["weekend_spend_share_112"] = swk.values / (s112.values + 1.0)
    return out

tbl = agent_api.build_features(fn)
print(tbl.shape)
print(tbl.head(3).iloc[:, :12])
p = agent_api.save_table(tbl, "e004_seasonal.parquet")
print(p)
