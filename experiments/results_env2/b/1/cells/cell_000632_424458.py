import numpy as np, pandas as pd

KS = np.arange(14)

def make_feats(view, snapshot_day):
    s = int(snapshot_day)
    hh_obj = view.households
    hh = pd.Index(hh_obj.index) if isinstance(hh_obj, pd.DataFrame) else pd.Index(hh_obj)
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
    n = len(hh)
    start = s - 28*KS - 27                      # window k = [s-28k-27, s-28k]
    V = (start[None, :] >= fd.values[:, None])  # n x 14 validity
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
    for k in range(14):
        F[f"g_lw{k}"] = np.where(V[:, k], np.log1p(S[:, k]), np.nan)
    F["g_w0r"], F["g_w1r"], F["g_w2r"] = S[:, 0], S[:, 1], S[:, 2]
    F["g_wmean"] = mean9; F["g_wmed"] = med; F["g_wstd"] = np.sqrt(var9)
    F["g_wmin"] = mn; F["g_wmax"] = mx
    F["g_wzero"] = zero9; F["g_wslope"] = slope
    F["g_ew6"] = ew6; F["g_ew85"] = ew85
    F["g_shr1"] = (m9*mean9 + 1*G)/(m9 + 1)
    F["g_shr3"] = (m9*mean9 + 3*G)/(m9 + 3)
    F["g_dlog_01"] = F["g_lw0"] - F["g_lw1"]
    F["g_dlog_rm"] = np.log1p(S[:, 0] + S[:, 1]) - np.log1p(S[:, 2] + S[:, 3] + S[:, 4] + S[:, 5])
    for k in range(6):
        F[f"g_lt{k}"] = np.where(V[:, k], np.log1p(T[:, k]), np.nan)
    F["g_tmean"] = (T[:, :10]*V9).sum(1)/np.maximum(m9, 1)
    F["g_tzero"] = ((T[:, :10] <= 0)*V9).sum(1)/np.maximum(m9, 1)
    dsl = (s - ld).values.astype(float)
    F["g_dsl"] = dsl
    F["g_expact"] = np.exp(-dsl/56.0)
    F["g_dlog_seas"] = F["g_lw0"] - F["g_lw12"]
    out = pd.DataFrame(F, index=hh)
    out.index.name = "household_key"
    return out

feats = agent_api.build_features(make_feats)
print("feats", feats.shape)

# sanity: correlation of key grid features with train target
tt = agent_api.train_targets()
chk = feats.merge(tt, on=["household_key", "snapshot_day"])
print(chk[["g_lw0","g_lw1","g_wmed","g_ew85","g_shr3","g_wzero","g_dsl","g_wslope","future_spend_4w"]]
      .corr()["future_spend_4w"].round(3))

base = agent_api.load_saved("e008_level_shape.parquet")
print("base", base.shape)
merged = base.merge(feats, on=["household_key", "snapshot_day"], how="inner")
print("merged", merged.shape, "max nan frac:", merged.filter(like="g_").isna().mean().max().round(3))
path = agent_api.save_table(merged, "e016_grid.parquet")
print(path)
