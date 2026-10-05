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


# ---- cell ----
import numpy as np, pandas as pd

KS = np.arange(14)

def make_feats(view, snapshot_day):
    s = int(snapshot_day)
    hh = pd.Index(view.households)
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


# ---- cell ----
def tiny(view, snapshot_day):
    hh = pd.Index(view.households)
    out = pd.DataFrame({"tiny": np.ones(len(hh))}, index=hh)
    out.index.name = "household_key"
    return out

f = agent_api.build_features(tiny)
print("tiny ok", f.shape)


# ---- cell ----
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


# ---- cell ----
def make_feats(view, snapshot_day):
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
    tr = tx.groupby(["household_key","_b"])["basket_id"].nunique().unstack("_b").reindex(index=hh, columns=range(14))
    fd = tx.groupby("household_key")["day"].min().reindex(hh)
    ld = tx.groupby("household_key")["day"].max().reindex(hh)
    S = np.nan_to_num(sp.values.astype(float))
    T = np.nan_to_num(tr.values.astype(float))
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
print("max nan frac:", feats.filter(like="g_").isna().mean().max().round(4))


# ---- cell ----
def grid_trips(view, snapshot_day):
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
    tr = tx.groupby(["household_key","_b"])["basket_id"].nunique().unstack("_b").reindex(index=hh, columns=range(14))
    fd = tx.groupby("household_key")["day"].min().reindex(hh)
    ld = tx.groupby("household_key")["day"].max().reindex(hh)
    S = np.nan_to_num(sp.values.astype(float))
    T = np.nan_to_num(tr.values.astype(float))
    start = s - 28*KS - 27
    V = (start[None, :] >= fd.values[:, None])
    F = {}
    for k in range(14):
        F[f"g_lw{k}"] = np.where(V[:, k], np.log1p(S[:, k]), np.nan)
    for k in range(6):
        F[f"g_lt{k}"] = np.where(V[:, k], np.log1p(T[:, k]), np.nan)
    dsl = (s - ld).values.astype(float)
    F["g_dsl"] = dsl
    F["g_expact"] = np.exp(-dsl/56.0)
    out = pd.DataFrame(F, index=hh)
    out.index.name = "household_key"
    return out

f = agent_api.build_features(grid_trips)
print("grid+trips ok", f.shape)


# ---- cell ----
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


# ---- cell ----
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


# ---- cell ----
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


# ---- cell ----
def make_feats(view, snapshot_day):
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
    tr = tx.groupby(["household_key","_b"])["basket_id"].nunique().unstack("_b").reindex(index=hh, columns=range(14))
    fd = tx.groupby("household_key")["day"].min().reindex(hh)
    ld = tx.groupby("household_key")["day"].max().reindex(hh)
    S = np.nan_to_num(sp.values.astype(float))
    T = np.nan_to_num(tr.values.astype(float))
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
    # slope via elementwise ops (matmul crashes the isolated runner)
    K10 = KS[:10]
    Sk = (V9*K10).sum(1); Skk = (V9*K10**2).sum(1); Skv = (S[:, :10]*V9*K10).sum(1)
    den = m9*Skk - Sk**2
    slope = np.where(den > 1e-9, (m9*Skv - Sk*sum9)/np.maximum(den, 1e-9), 0.0)
    med = np.full(n, np.nan); mn = np.full(n, np.nan); mx = np.full(n, np.nan)
    ew6 = np.full(n, np.nan); ew85 = np.full(n, np.nan)
    for i in range(n):
        kk = K10[V[i, :10]]; vv = S[i, :10][V[i, :10]]
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
print("max nan frac:", round(float(feats.filter(like="g_").isna().mean().max()), 4))

tt = agent_api.train_targets()
chk = feats.merge(tt, on=["household_key", "snapshot_day"])
print(chk[["g_lw0","g_lw1","g_wmed","g_ew85","g_shr3","g_wzero","g_dsl","g_wslope","g_dlog_rm","future_spend_4w"]]
      .corr()["future_spend_4w"].round(3))

base = agent_api.load_saved("e008_level_shape.parquet")
merged = base.merge(feats, on=["household_key", "snapshot_day"], how="inner")
print("base", base.shape, "merged", merged.shape)
path = agent_api.save_table(merged, "e016_grid.parquet")
print(path)
