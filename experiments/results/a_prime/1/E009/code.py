import agent_api as A
import pandas as pd, numpy as np

print("== snapshot days ==")
print(A.snapshot_days())

tt = A.train_targets()
y = tt.future_spend_4w
print("== target train ==")
print(y.describe())
print("zero share:", round(float((y==0).mean()),4), " median:", float(y.median()))

names = ["e001_txhist","e002_channel","e003_catmix","e004_mkt","e004_mkt_full","e006_catmix_mkt","e007_logratio","nf_candidates","nf_p1","nf_seasonal","nf_transforms"]
tabs = {}
for nm in names:
    try:
        df = A.load_saved(nm + ".parquet")
        tabs[nm] = df
        fcols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
        print(f"{nm}: shape={df.shape} nfeat={len(fcols)}")
    except Exception as e:
        print(nm, "ERR", repr(e))

print("\nE003 cols:", list(tabs["e003_catmix"].columns))
print("\nE001 cols:", list(tabs["e001_txhist"].columns))
print("\nnf_candidates cols:", list(tabs["nf_candidates"].columns))

v = A.snapshot()
tx = v.transactions
print("\ntransactions shape:", tx.shape)
print(tx.head(3))
print("discount signs (means):", tx[["coupon_disc","coupon_match_disc","retail_disc","sales_value","quantity"]].mean())


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()

names = ["e001_txhist","e003_catmix","e004_mkt","e004_mkt_full","nf_p1","nf_candidates","nf_transforms","nf_seasonal"]
tabs = {nm: A.load_saved(nm+".parquet") for nm in names}

def prep(df):
    fcols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
    m = tt.merge(df, on=["household_key","snapshot_day"], how="left")
    X = m[fcols].copy()
    for c in fcols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf,-np.inf], np.nan)
    return X, fcols, m

def ridge_fit(X, y, lam=10.0):
    mu = X.mean(); sd = X.std().replace(0,1.0)
    Z = ((X-mu)/sd).fillna(0.0).values
    Z = np.hstack([Z, np.ones((len(Z),1))])
    A_ = Z.T@Z + lam*np.eye(Z.shape[1]); A_[-1,-1] -= lam
    w = np.linalg.solve(A_, Z.T@y.values)
    return mu, sd, w

def ridge_pred(X, mu, sd, w):
    Z = ((X-mu)/sd).fillna(0.0).values
    Z = np.hstack([Z, np.ones((len(Z),1))])
    return Z@w

TRAIN_DAYS = [95,123,151,179,207,235,263,291,319,347,375,403,431]
def internal_eval(df, hold=(403,431), lam=10.0, logt=False, name=""):
    X, fcols, m = prep(df)
    y = m.future_spend_4w
    tr_days = [d for d in TRAIN_DAYS if d not in hold]
    tr = m.snapshot_day.isin(tr_days)
    va = m.snapshot_day.isin(hold)
    if logt:
        mu,sd,w = ridge_fit(X[tr], np.log1p(y[tr]), lam)
        p = np.expm1(np.clip(ridge_pred(X[va],mu,sd,w),0,15))
    else:
        mu,sd,w = ridge_fit(X[tr], y[tr], lam)
        p = np.clip(ridge_pred(X[va],mu,sd,w),0,None)
    mae = float(np.abs(p - y[va]).mean())
    return mae, len(fcols)

print(f"{'table':16s} {'lin':>8s} {'log':>8s} nfeat")
for nm in names:
    df = tabs[nm]
    m1,_ = internal_eval(df, logt=False)
    m2,_ = internal_eval(df, logt=True)
    print(f"{nm:16s} {m1:8.3f} {m2:8.3f} {df.shape[1]-2}")

# combos
e1, e3, mkt = tabs["e001_txhist"], tabs["e003_catmix"], tabs["e004_mkt"]
nf = tabs["nf_p1"]
key = ["household_key","snapshot_day"]
def merge(a,b): return a.merge(b, on=key, how="inner", suffixes=("","_d"))
c_e3_nf = merge(e3, nf)
m1,_ = internal_eval(c_e3_nf); m2,_ = internal_eval(c_e3_nf, logt=True)
print(f"{'E003+nf_p1':16s} {m1:8.3f} {m2:8.3f} {c_e3_nf.shape[1]-2}")
c_all = merge(c_e3_nf, mkt)
m1,_ = internal_eval(c_all); m2,_ = internal_eval(c_all, logt=True)
print(f"{'E003+nf+mkt':16s} {m1:8.3f} {m2:8.3f} {c_all.shape[1]-2}")


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
key = ["household_key","snapshot_day"]
names = ["e001_txhist","e002_channel","e003_catmix","e004_mkt","e004_mkt_full","e006_catmix_mkt","e007_logratio","nf_candidates","nf_p1","nf_transforms"]
tabs = {nm: A.load_saved(nm+".parquet") for nm in names}

pool = tabs["e003_catmix"].copy()
for nm in ["nf_p1","e004_mkt_full","e002_channel","nf_transforms","nf_candidates"]:
    df = tabs[nm]
    newc = [c for c in df.columns if c not in pool.columns and c not in key]
    pool = pool.merge(df[key+newc], on=key, how="left")
fcols = [c for c in pool.columns if c not in key]
print("pool features:", len(fcols))

m = tt.merge(pool, on=key, how="left")
Xdf = m[fcols].copy()
for c in fcols:
    if Xdf[c].dtype == object:
        Xdf[c] = pd.factorize(Xdf[c])[0].astype(float)
    Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
Xdf = Xdf.replace([np.inf,-np.inf], np.nan)
X = Xdf.values
y = m.future_spend_4w.values
days = m.snapshot_day.values
TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]

mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0); sd[sd==0]=1
Z = (X-mu)/sd
Z = np.where(np.isnan(Z), 0.0, Z)
Z = np.clip(Z, -8, 8)
Z1 = np.hstack([Z, np.ones((len(Z),1))])

def ridge_from_gram(Ztr, ytr, Zva, lam):
    G = Ztr.T@Ztr; b = Ztr.T@ytr
    G = G + lam*np.eye(G.shape[0]); G[-1,-1] -= lam
    w = np.linalg.solve(G, b)
    return Zva@w

def loo_mae(sub, lam=30.0):
    errs = []
    for d in TRAIN:
        tr = days != d
        va = days == d
        cols = [*sub, -1] if len(sub) else [-1]
        p = ridge_from_gram(Z1[tr][:, cols], y[tr], Z1[va][:, cols], lam)
        errs.append(np.abs(np.clip(p,0,None) - y[va]))
    return float(np.concatenate(errs).mean())

base_mae = loo_mae([])
print("predict-train-mean LOO MAE:", round(base_mae,3))

uni = []
for j in range(len(fcols)):
    uni.append((loo_mae([j]), fcols[j]))
uni.sort()
print("top 25 univariate:")
for v,c in uni[:25]: print(f"  {c:28s} {v:7.3f}")
print("worst 8:")
for v,c in uni[-8:]: print(f"  {c:28s} {v:7.3f}")

e3_idx = [fcols.index(c) for c in tabs["e003_catmix"].columns if c not in key]
print("E003 full LOO:", round(loo_mae(e3_idx),3))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
key = ["household_key","snapshot_day"]
names = ["e001_txhist","e002_channel","e003_catmix","e004_mkt","e004_mkt_full","e006_catmix_mkt","e007_logratio","nf_candidates","nf_p1","nf_transforms"]
tabs = {nm: A.load_saved(nm+".parquet") for nm in names}
pool = tabs["e003_catmix"].copy()
for nm in ["nf_p1","e004_mkt_full","e002_channel","nf_transforms","nf_candidates"]:
    df = tabs[nm]
    newc = [c for c in df.columns if c not in pool.columns and c not in key]
    pool = pool.merge(df[key+newc], on=key, how="left")
fcols = [c for c in pool.columns if c not in key]
m = tt.merge(pool, on=key, how="left")
Xdf = m[fcols].copy()
for c in fcols:
    if Xdf[c].dtype == object: Xdf[c] = pd.factorize(Xdf[c])[0].astype(float)
    Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
Xdf = Xdf.replace([np.inf,-np.inf], np.nan)
X = Xdf.values; y = m.future_spend_4w.values; days = m.snapshot_day.values
TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]
mu = np.nanmean(X,0); sd = np.nanstd(X,0); sd[sd==0]=1
Z = np.clip(np.where(np.isnan((X-mu)/sd),0.0,(X-mu)/sd),-8,8)
Z1 = np.hstack([Z, np.ones((len(Z),1))])
K = Z1.shape[1]

# precompute per-snapshot Grams
Gd, bd, Zv, yv, nv = {}, {}, {}, {}, {}
for d in TRAIN:
    tr = days != d; va = days == d
    Gd[d] = Z1[tr].T@Z1[tr]; bd[d] = Z1[tr].T@y[tr]
    Zv[d] = Z1[va]; yv[d] = y[va]; nv[d] = va.sum()

def loo_mae(sub, lam=30.0):
    errs = []
    for d in TRAIN:
        s = [*sub, K-1]
        G = Gd[d][np.ix_(s,s)].copy(); b = bd[d][s].copy()
        G += lam*np.eye(len(s)); G[-1,-1] -= lam
        w = np.linalg.solve(G, b)
        p = np.clip(Zv[d][:,s]@w, 0, None)
        errs.append(np.abs(p - yv[d]))
    return float(np.concatenate(errs).mean())

# forward selection from empty
cur = []; best = loo_mae(cur); print("start (intercept only):", round(best,3))
hist = []
for step in range(18):
    cand_best = None
    for j in range(len(fcols)):
        if j in cur: continue
        v = loo_mae(cur+[j])
        if cand_best is None or v < cand_best[0]: cand_best = (v, j)
    v, j = cand_best
    hist.append((v, fcols[j]))
    print(f"step {step+1}: +{fcols[j]:28s} LOO={v:.3f} (delta {best-v:+.3f})")
    if v < best - 1e-4:
        cur = cur+[j]; best = v
    else:
        cur = cur+[j]; best = v  # keep adding anyway, monitor
print("\nselected:", [fcols[j] for j in cur])
print("final LOO:", round(best,3))

# compare: E003's 59 features
e3_idx = [fcols.index(c) for c in tabs["e003_catmix"].columns if c not in key]
print("E003 LOO:", round(loo_mae(e3_idx),3))
# E003 + top selected extras
extras = [fcols.index(c) for c in ["nf_pow90_ewm4","newm4","nf_pow90_ewm8","nf_pow90","h_act_pow90","nf_pow75_ewm","nf_pow75","newm8","nwmean12","h_act_s123","nf_pow90_ewm13","newm13"] if c in fcols]
print("E003+extras LOO:", round(loo_mae(e3_idx+extras),3))
print("extras-only LOO:", round(loo_mae(extras),3))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

def fn(view, s):
    tx = view.transactions
    hh = pd.Index(view.households)
    tx = tx[tx.household_key.isin(hh)]
    g = tx.groupby("household_key")
    # weekly windows w=1..13 (w=1: days s-6..s)
    rel = s - tx.day
    wk = (rel // 7 + 1).clip(1, 13)
    mo = (rel // 28 + 1).clip(1, 13)
    sv = tx.sales_value
    piv_w = pd.pivot_table(pd.DataFrame({"h":tx.household_key,"w":wk,"v":sv}),
                           index="h", columns="w", values="v", aggfunc="sum").reindex(hh)
    piv_m = pd.pivot_table(pd.DataFrame({"h":tx.household_key,"m":mo,"v":sv}),
                           index="h", columns="m", values="v", aggfunc="sum").reindex(hh)
    W = piv_w.reindex(columns=range(1,14)).fillna(0.0).values  # n x 13 weekly
    M = piv_m.reindex(columns=range(1,14)).fillna(0.0).values  # n x 13 28-day
    out = pd.DataFrame(index=hh)
    out["r_med13"] = np.median(W, axis=1)
    out["r_q25_13"] = np.percentile(W, 25, axis=1)
    out["r_q75_13"] = np.percentile(W, 75, axis=1)
    out["r_med4"] = np.median(W[:, :4], axis=1)
    out["r_maxmed13"] = W.max(1) / (np.median(W,1)+1.0)
    out["r_zerow13"] = (W == 0).sum(1)
    out["r_zerom13"] = (M == 0).sum(1)
    out["r_zerow4"] = (W[:, :4] == 0).sum(1)
    # lag-1 autocorr of weekly series
    ac = np.full(len(out), np.nan)
    X0, X1 = W[:, 1:-1], W[:, :-2]
    sd0, sd1 = X0.std(1), X1.std(1)
    ok = (sd0 > 1e-9) & (sd1 > 1e-9)
    ac[ok] = ((X0[ok]-X0[ok].mean(1,keepdims=True))*(X1[ok]-X1[ok].mean(1,keepdims=True))).mean(1)/(sd0[ok]*sd1[ok])
    out["r_autocorr"] = ac
    # normalized trend slope over 13 weeks
    t = np.arange(13, dtype=float); tc = t - t.mean()
    slope = (W * tc).sum(1) / (tc**2).sum()
    out["r_slope13"] = slope / (W.mean(1) + 1.0)
    # recency / gaps
    last = g.day.max().reindex(hh).fillna(0)
    out["r_dsl"] = s - last
    days_u = g.day.apply(lambda x: np.sort(x.unique())).reindex(hh)
    gaps = []
    for d in days_u.values:
        gaps.append(np.diff(d).mean() if len(d) > 1 else np.nan)
    out["r_gap_mean"] = gaps
    out["r_gap_ratio"] = out["r_dsl"] / (out["r_gap_mean"] + 1.0)
    # cross-sectional percentile ranks (drift-free level signal)
    l1 = M[:,0]; l4 = M[:, :4].sum(1); l13 = M.sum(1)
    for nm, v in [("pct_l1", l1), ("pct_l4", l4), ("pct_l13", l13)]:
        out[nm] = pd.Series(v, index=hh).rank(pct=True)
    # snapshot-level global aggregates (drift calibration)
    out["g_mean_l4"] = float(np.mean(l4)); out["g_med_l4"] = float(np.median(l4))
    out["g_mean_l1"] = float(np.mean(l1))
    # seasonality
    wky = ((s + 8) // 7) % 52
    out["k_sin1"] = np.sin(2*np.pi*wky/52.0); out["k_cos1"] = np.cos(2*np.pi*wky/52.0)
    out["k_sin2"] = np.sin(4*np.pi*wky/52.0); out["k_cos2"] = np.cos(4*np.pi*wky/52.0)
    out["s_day"] = float(s)
    return out

df = A.build_features(fn)
print("built:", df.shape)
p = A.save_table(df, "nf_robust.parquet")
print("saved:", p)

# drift diagnostics on train targets
tt = A.train_targets()
print(tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median"]).round(1))
# global aggregate drift
print(df.groupby("snapshot_day")[["g_mean_l4","g_med_l4","g_mean_l1"]].first().round(1))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
key = ["household_key","snapshot_day"]
pool = A.load_saved("e003_catmix.parquet").copy()
for nm in ["nf_p1","e004_mkt_full","e002_channel","nf_transforms","nf_candidates","nf_robust"]:
    df = A.load_saved(nm+".parquet")
    newc = [c for c in df.columns if c not in pool.columns and c not in key]
    pool = pool.merge(df[key+newc], on=key, how="left")
fcols = [c for c in pool.columns if c not in key]
m = tt.merge(pool, on=key, how="left")
Xdf = m[fcols].copy()
for c in fcols:
    if Xdf[c].dtype == object: Xdf[c] = pd.factorize(Xdf[c])[0].astype(float)
    Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
Xdf = Xdf.replace([np.inf,-np.inf], np.nan)
X = Xdf.values; y = m.future_spend_4w.values; days = m.snapshot_day.values
TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]
mu = np.nanmean(X,0); sd = np.nanstd(X,0); sd[sd==0]=1
Z = np.clip(np.where(np.isnan((X-mu)/sd),0.0,(X-mu)/sd),-8,8)
Z1 = np.hstack([Z, np.ones((len(Z),1))])
K = Z1.shape[1]
Gd, bd, Zv, yv = {}, {}, {}, {}
for d in TRAIN:
    tr = days != d; va = days == d
    Gd[d] = Z1[tr].T@Z1[tr]; bd[d] = Z1[tr].T@y[tr]
    Zv[d] = Z1[va]; yv[d] = y[va]

def loo_mae(sub, lam=30.0):
    errs = []
    for d in TRAIN:
        s = [*sub, K-1]
        G = Gd[d][np.ix_(s,s)].copy(); b = bd[d][s].copy()
        G += lam*np.eye(len(s)); G[-1,-1] -= lam
        w = np.linalg.solve(G, b)
        errs.append(np.abs(np.clip(Zv[d][:,s]@w,0,None) - yv[d]))
    return float(np.concatenate(errs).mean())

rob = [c for c in fcols if c.startswith(("r_","pct_","g_","k_","s_day"))]
print("robust features:", len(rob))
uni = sorted([(loo_mae([fcols.index(c)]), c) for c in rob])
for v,c in uni: print(f"  {c:14s} {v:7.3f}")

sel_prev = ['nf_pow90_ewm4','spend_rate28','nspend7','days_active_l1','h_act_pow90','trips_l3','nbask_max84','ntrips7','spend28_DELI','ndow5','spend_l456_mean','nf_ratio123v456','mkt_tgt_TypeA_n','nf_wsin1','nf_wsin2','nf_pow90_ewm13','h_inact_s123','nwmax12']
sel_prev = [c for c in sel_prev if c in fcols]
e3 = [c for c in A.load_saved("e003_catmix.parquet").columns if c not in key]
tests = {
 "prev18": sel_prev,
 "prev18+robust_all": sel_prev + rob,
 "E003+robust_all": e3 + rob,
 "robust_all_only": rob,
}
for nm, cols in tests.items():
    print(f"{nm:20s} {loo_mae([fcols.index(c) for c in cols]):.3f}  n={len(cols)}")

# greedy forward from prev18 with robust candidates
cur = [fcols.index(c) for c in sel_prev]; best = loo_mae(cur)
print("\nforward from prev18:")
for step in range(10):
    cb = None
    for c in rob:
        j = fcols.index(c)
        if j in cur: continue
        v = loo_mae(cur+[j])
        if cb is None or v < cb[0]: cb = (v, j)
    v, j = cb
    print(f"  +{fcols[j]:14s} LOO={v:.3f} delta {best-v:+.3f}")
    cur = cur+[j]; best = v

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
key = ["household_key","snapshot_day"]
pool = A.load_saved("e003_catmix.parquet").copy()
for nm in ["nf_p1","e004_mkt_full","e002_channel","nf_transforms","nf_candidates","nf_robust"]:
    df = A.load_saved(nm+".parquet")
    newc = [c for c in df.columns if c not in pool.columns and c not in key]
    pool = pool.merge(df[key+newc], on=key, how="left")
fcols = [c for c in pool.columns if c not in key]
m = tt.merge(pool, on=key, how="left")
Xdf = m[fcols].copy()
for c in fcols:
    if Xdf[c].dtype == object: Xdf[c] = pd.factorize(Xdf[c])[0].astype(float)
    Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
Xdf = Xdf.replace([np.inf,-np.inf], np.nan)
X = Xdf.values; y = m.future_spend_4w.values; days = m.snapshot_day.values
TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]
mu = np.nanmean(X,0); sd = np.nanstd(X,0); sd[sd==0]=1
Z = np.clip(np.where(np.isnan((X-mu)/sd),0.0,(X-mu)/sd),-8,8)
Z1 = np.hstack([Z, np.ones((len(Z),1))])
K = Z1.shape[1]

sel_prev = ['nf_pow90_ewm4','spend_rate28','nspend7','days_active_l1','h_act_pow90','trips_l3','nbask_max84','ntrips7','spend28_DELI','ndow5','spend_l456_mean','nf_ratio123v456','mkt_tgt_TypeA_n','nf_wsin1','nf_wsin2','nf_pow90_ewm13','h_inact_s123','nwmax12']
sel_prev = [c for c in sel_prev if c in fcols]
sub = [fcols.index(c) for c in sel_prev] + [K-1]

def fit_pred(d, lam=30.0):
    tr = days != d; va = days == d
    G = Z1[tr][:,sub].T@Z1[tr][:,sub]; b = Z1[tr][:,sub].T@y[tr]
    G += lam*np.eye(len(sub)); G[-1,-1] -= lam
    w = np.linalg.solve(G, b)
    return np.clip(Z1[va][:,sub]@w,0,None), y[va], va

# error decomposition for last snapshot as example
p, yy, va = fit_pred(431)
zmask = yy==0
print("MAE overall:", round(float(np.abs(p-yy).mean()),2))
print("MAE on zero-target rows:", round(float(np.abs(p[zmask]-0).mean()),2), " share:", round(float(zmask.mean()),3))
print("MAE on positive rows:", round(float(np.abs(p[~zmask]-yy[~zmask]).mean()),2))
print("mean pred on zero rows:", round(float(p[zmask].mean()),1))
# by target bucket
for lo,hi in [(0,0),(1,50),(50,100),(100,200),(200,400),(400,2200)]:
    mm = (yy>=lo)&(yy<=hi)
    print(f"  y in [{lo},{hi}]: n={mm.sum():5d} mean_pred={p[mm].mean():7.1f} mean_y={yy[mm].mean():7.1f} MAE={np.abs(p[mm]-yy[mm]).mean():7.1f}")

# how predictable is zero? univariate AUC-ish with simple features
zall = (y==0).astype(float)
from statistics import mean as _m
def auc_score(s, z):
    s = np.asarray(s, float); z = np.asarray(z, float)
    ok = ~np.isnan(s)
    s, z = s[ok], z[ok]
    r = pd.Series(s).rank().values
    n1, n0 = z.sum(), (1-z).sum()
    return (r[z==1].sum() - n1*(n1+1)/2) / (n1*n0)
cands = ["r_dsl","r_zerom13","r_zerow13","r_zerow4","r_gap_ratio","days_since_last","zero_recent","active_share_l1","nf_pow90_ewm4","spend_l1","tenure","r_gap_mean","h_inact_s123"]
for c in cands:
    if c in fcols:
        a = auc_score(X[:, fcols.index(c)], zall)
        print(f"AUC({c:16s}) = {a:.3f}")

# two-part model test (LOO on last few snapshots): p0 linear prob + level ridge on positives
def two_part(d, lam=30.0):
    tr = days != d; va = days == d
    # p0
    G = Z1[tr][:,sub].T@Z1[tr][:,sub]; b = Z1[tr][:,sub].T@zall[tr]
    G += lam*np.eye(len(sub)); G[-1,-1] -= lam
    w0 = np.linalg.solve(G, b)
    p0 = np.clip(Z1[va][:,sub]@w0, 0, 1)
    # level on positives
    trp = tr & (y>0)
    G = Z1[trp][:,sub].T@Z1[trp][:,sub]; b = Z1[trp][:,sub].T@y[trp]
    G += lam*np.eye(len(sub)); G[-1,-1] -= lam
    wl = np.linalg.solve(G, b)
    pl = np.clip(Z1[va][:,sub]@wl, 0, None)
    return (1-p0)*pl, y[va]

errs1, errs2 = [], []
for d in TRAIN:
    p1, yy1 = fit_pred(d)[:2]
    p2, yy2 = two_part(d)
    errs1.append(np.abs(p1-yy1)); errs2.append(np.abs(p2-yy2))
print("\nLOO MAE single ridge:", round(float(np.concatenate(errs1).mean()),3))
print("LOO MAE two-part    :", round(float(np.concatenate(errs2).mean()),3))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
key = ["household_key","snapshot_day"]
pool = A.load_saved("e003_catmix.parquet").copy()
for nm in ["nf_p1","e004_mkt_full","e002_channel","nf_transforms","nf_candidates","nf_robust"]:
    df = A.load_saved(nm+".parquet")
    newc = [c for c in df.columns if c not in pool.columns and c not in key]
    pool = pool.merge(df[key+newc], on=key, how="left")
fcols = [c for c in pool.columns if c not in key]
m = tt.merge(pool, on=key, how="left")
Xdf = m[fcols].copy()
for c in fcols:
    if Xdf[c].dtype == object: Xdf[c] = pd.factorize(Xdf[c])[0].astype(float)
    Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
Xdf = Xdf.replace([np.inf,-np.inf], np.nan)
X = Xdf.values; y = m.future_spend_4w.values; days = m.snapshot_day.values
TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]
mu = np.nanmean(X,0); sd = np.nanstd(X,0); sd[sd==0]=1
Z = np.clip(np.where(np.isnan((X-mu)/sd),0.0,(X-mu)/sd),-8,8)
Z1 = np.hstack([Z, np.ones((len(Z),1))])
K = Z1.shape[1]

sel_prev = ['nf_pow90_ewm4','spend_rate28','nspend7','days_active_l1','h_act_pow90','trips_l3','nbask_max84','ntrips7','spend28_DELI','ndow5','spend_l456_mean','nf_ratio123v456','mkt_tgt_TypeA_n','nf_wsin1','nf_wsin2','nf_pow90_ewm13','h_inact_s123','nwmax12']
sel_prev = [c for c in sel_prev if c in fcols]
SP = [fcols.index(c) for c in sel_prev]

def fit_pred(sub, tr_mask, va_mask, lam=30.0):
    G = Z1[tr_mask][:,sub].T@Z1[tr_mask][:,sub]; b = Z1[tr_mask][:,sub].T@y[tr_mask]
    G += lam*np.eye(len(sub)); G[-1,-1] -= lam
    w = np.linalg.solve(G, b)
    return np.clip(Z1[va_mask][:,sub]@w,0,None)

def loo(sub, lam=30.0):
    errs=[]
    for d in TRAIN:
        p = fit_pred(sub, days!=d, days==d, lam)
        errs.append(np.abs(p-y[days==d]))
    return float(np.concatenate(errs).mean())

def extrap(sub, lam=30.0):
    """train <=375, validate 403+431"""
    tr = days<=375; va = (days==403)|(days==431)
    p = fit_pred(sub, tr, va, lam)
    return float(np.abs(p-y[va]).mean())

# candidate new features
def addf(name, vec):
    global Z1, fcols
    v = np.asarray(vec, float)
    v = (v-np.nanmean(v))/ (np.nanstd(v)+1e-12)
    v = np.clip(np.where(np.isnan(v),0,v),-8,8)
    Z1 = np.hstack([Z1, v.reshape(-1,1)])
    fcols = fcols+[name]
    return len(fcols)-1

i_zw13 = addf("x_zerow13", X[:, fcols.index("r_zerow13")])
i_zw4  = addf("x_zerow4",  X[:, fcols.index("r_zerow4")])
i_dsl  = addf("x_dsl",     X[:, fcols.index("r_dsl")])
i_zm13 = addf("x_zerom13", X[:, fcols.index("r_zerom13")])
i_sday = addf("x_sday",    days.astype(float))
i_gml4 = addf("x_gml4",    X[:, fcols.index("g_mean_l4")])
top = X[:, fcols.index("nf_pow90_ewm4")]
topz = np.clip(np.where(np.isnan(top),0,(top-np.nanmean(top))/(np.nanstd(top)+1e-12)),-8,8)
i_sx   = addf("x_sday_x_top", days.astype(float)/100*topz)
i_gxt  = addf("x_gml4_x_top", X[:, fcols.index("g_mean_l4")]/100*topz)
# conditional level: spend_l1 per active day
cl = X[:, fcols.index("spend_l1")]/(X[:, fcols.index("days_active_l1")].clip(1,None))
i_cl  = addf("x_cond_level", cl)
# decile dummies of top feature
qs = np.nanquantile(top, np.linspace(0.1,0.9,9))
binned = np.searchsorted(qs, np.nan_to_num(top, nan=-1e9))
for b in range(1,9):
    addf(f"x_bin{b}", (binned==b).astype(float))

names = {i_zw13:"x_zerow13",i_zw4:"x_zerow4",i_dsl:"x_dsl",i_zm13:"x_zerom13",i_sday:"x_sday",i_gml4:"x_gml4",i_sx:"x_sday_x_top",i_gxt:"x_gml4_x_top",i_cl:"x_cond_level"}
print(f"{'variant':34s} {'LOO':>7s} {'EXTR':>7s}")
base_loo, base_ex = loo(SP), extrap(SP)
print(f"{'prev18':34s} {base_loo:7.3f} {base_ex:7.3f}")
tests = {
 "prev18+zerow13": SP+[i_zw13],
 "prev18+zerow13+zerow4+dsl": SP+[i_zw13,i_zw4,i_dsl],
 "prev18+zerow13+zerow4+dsl+zm13": SP+[i_zw13,i_zw4,i_dsl,i_zm13],
 "prev18+sday": SP+[i_sday],
 "prev18+gml4": SP+[i_gml4],
 "prev18+sday+gml4": SP+[i_sday,i_gml4],
 "prev18+sday_x_top": SP+[i_sx],
 "prev18+gml4_x_top": SP+[i_gxt],
 "prev18+cond_level": SP+[i_cl],
 "prev18+zerow..+sday+gml4": SP+[i_zw13,i_zw4,i_dsl,i_sday,i_gml4],
 "prev18+zerow..+cond_level": SP+[i_zw13,i_zw4,i_dsl,i_cl],
 "prev18+allbins": SP+[j for j in range(K, len(fcols)) if fcols[j].startswith("x_bin")],
 "prev18+zerow13+allbins": SP+[i_zw13]+[j for j in range(K, len(fcols)) if fcols[j].startswith("x_bin")],
}
for nm, sub in tests.items():
    print(f"{nm:34s} {loo(sub):7.3f} {extrap(sub):7.3f}")
print("\nnote: EXTR = train<=375, val={403,431}; LOO = leave-one-snapshot-out")

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
key = ["household_key","snapshot_day"]
pool = A.load_saved("e003_catmix.parquet").copy()
for nm in ["nf_p1","e004_mkt_full","e002_channel","nf_transforms","nf_candidates","nf_robust"]:
    df = A.load_saved(nm+".parquet")
    newc = [c for c in df.columns if c not in pool.columns and c not in key]
    pool = pool.merge(df[key+newc], on=key, how="left")
fcols = [c for c in pool.columns if c not in key]
m = tt.merge(pool, on=key, how="left")
Xdf = m[fcols].copy()
for c in fcols:
    if Xdf[c].dtype == object: Xdf[c] = pd.factorize(Xdf[c])[0].astype(float)
    Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
Xdf = Xdf.replace([np.inf,-np.inf], np.nan)
X = Xdf.values; y = m.future_spend_4w.values; days = m.snapshot_day.values
hh = m.household_key.values
TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]
mu = np.nanmean(X,0); sd = np.nanstd(X,0); sd[sd==0]=1
Z = np.clip(np.where(np.isnan((X-mu)/sd),0.0,(X-mu)/sd),-8,8)

sel_prev = ['nf_pow90_ewm4','spend_rate28','nspend7','days_active_l1','h_act_pow90','trips_l3','nbask_max84','ntrips7','spend28_DELI','ndow5','spend_l456_mean','nf_ratio123v456','mkt_tgt_TypeA_n','nf_wsin1','nf_wsin2','nf_pow90_ewm13','h_inact_s123','nwmax12']
SP = [fcols.index(c) for c in sel_prev if c in fcols]
i_zw13 = fcols.index("r_zerow13")

codes = pd.factorize(hh)[0]
H = np.zeros((len(hh), codes.max()+1), dtype=np.float32)
H[np.arange(len(hh)), codes] = 1.0
print("hh dummy matrix:", H.shape)

def loo_hh(sub, lam_f=30.0, lam_h=100.0, use_hh=True, inter_zw=False):
    F = np.hstack([Z[:,sub], np.ones((len(Z),1))]).astype(np.float64)
    if inter_zw:
        zw = (Z[:,i_zw13] > 0.5).astype(float)
        F = np.hstack([F, Z[:,sub]*zw[:,None]])
    errs = []
    for d in TRAIN:
        tr = days != d; va = days == d
        Atr = np.hstack([F[tr]] + ([H[tr].astype(np.float64)] if use_hh else []))
        Ava = np.hstack([F[va]] + ([H[va].astype(np.float64)] if use_hh else []))
        lam = np.array([lam_f]*F.shape[1] + ([lam_h]*H.shape[1] if use_hh else []))
        G = Atr.T@Atr + np.diag(lam)
        w = np.linalg.solve(G, Atr.T@y[tr])
        p = np.clip(Ava@w, 0, None)
        errs.append(np.abs(p - y[va]))
    return float(np.concatenate(errs).mean())

print("prev18 (no hh):          ", round(loo_hh(SP, use_hh=False),3))
print("prev18 + hh dummies:     ", round(loo_hh(SP, use_hh=True, lam_h=100.0),3))
print("prev18 + hh (lam_h=30):  ", round(loo_hh(SP, use_hh=True, lam_h=30.0),3))
print("prev18 + hh (lam_h=300): ", round(loo_hh(SP, use_hh=True, lam_h=300.0),3))
print("prev18 + zw-interactions:", round(loo_hh(SP, use_hh=False, inter_zw=True),3))
print("prev18 + hh + zw-inter:  ", round(loo_hh(SP, use_hh=True, inter_zw=True),3))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
key = ["household_key","snapshot_day"]
pool = A.load_saved("e003_catmix.parquet").copy()
for nm in ["nf_p1","e004_mkt_full","e002_channel","nf_transforms","nf_candidates","nf_robust"]:
    df = A.load_saved(nm+".parquet")
    newc = [c for c in df.columns if c not in pool.columns and c not in key]
    pool = pool.merge(df[key+newc], on=key, how="left")
fcols = [c for c in pool.columns if c not in key]
m = tt.merge(pool, on=key, how="left")

# --- build new candidate features on the FULL train rows (approx; ok for screening) ---
v = A.snapshot()
tx = v.transactions
demo = v.demographics
# store tier: mean sales per basket per store (up to day 459)
sb = tx.groupby(["store_id","basket_id"]).sales_value.sum().reset_index()
store_tier = sb.groupby("store_id").sales_value.mean()
store_n = sb.groupby("store_id").size()
# household main store over last 84d
hh_days = [95,123,151,179,207,235,263,291,319,347,375,403,431,459]
rows = []
for s in [95,123,151,179,207,235,263,291,319,347,375,403,431]:
    t = tx[(tx.day<=s)&(tx.day>s-84)]
    b = t.groupby(["household_key","basket_id","store_id"]).sales_value.sum().reset_index()
    hs = b.groupby(["household_key","store_id"]).agg(sp=("sales_value","sum")).reset_index()
    tot = hs.groupby("household_key").sp.transform("sum")
    hs["share"] = hs.sp/tot
    main = hs.loc[hs.groupby("household_key").share.idxmax(), ["household_key","store_id","share"]]
    main["snapshot_day"] = s
    rows.append(main)
mainst = pd.concat(rows).rename(columns={"store_id":"main_store","share":"main_share"})
m = m.merge(mainst[["household_key","snapshot_day","main_store","main_share"]], on=key, how="left")
m["store_tier"] = m.main_store.map(store_tier)
m["store_logn"] = m.main_store.map(np.log1p(store_n))
# store-mean shrinkage of recent spend: store-level mean of spend_l1
m["store_mean_l1"] = m.groupby("snapshot_day").spend_l1.transform(lambda x: x.groupby(m.main_store).transform("mean"))
# demographics cell means
m2 = m.merge(demo, on="household_key", how="left")
m2["cell"] = m2.classification_4.fillna("NA")+"|"+m2.homeowner_desc.fillna("NA")+"|"+m2.kid_category_desc.fillna("NA")
m2["cell_mean_l1"] = m2.groupby(["snapshot_day","cell"]).spend_l1.transform("mean")
m2["cell_mean_l4"] = m2.groupby(["snapshot_day","cell"]).spend_l123_mean.transform("mean")
m = m2

newf = ["main_share","store_tier","store_logn","store_mean_l1","cell_mean_l1","cell_mean_l4"]
Xdf = m[fcols+newf].copy()
for c in fcols+newf:
    if Xdf[c].dtype == object: Xdf[c] = pd.factorize(Xdf[c])[0].astype(float)
    Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
Xdf = Xdf.replace([np.inf,-np.inf], np.nan)
X = Xdf.values; y = m.future_spend_4w.values; days = m.snapshot_day.values
TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]
mu = np.nanmean(X,0); sd = np.nanstd(X,0); sd[sd==0]=1
Z = np.clip(np.where(np.isnan((X-mu)/sd),0.0,(X-mu)/sd),-8,8)
Z1 = np.hstack([Z, np.ones((len(Z),1))])
K = Z1.shape[1]
Gd, bd, Zv, yv = {}, {}, {}, {}
for d in TRAIN:
    tr = days != d; va = days == d
    Gd[d] = Z1[tr].T@Z1[tr]; bd[d] = Z1[tr].T@y[tr]
    Zv[d] = Z1[va]; yv[d] = y[va]
def loo(sub, lam=30.0):
    errs=[]
    for d in TRAIN:
        s=[*sub,K-1]
        G=Gd[d][np.ix_(s,s)].copy(); b=bd[d][s].copy()
        G+=lam*np.eye(len(s)); G[-1,-1]-=lam
        errs.append(np.abs(np.clip(Zv[d][:,s]@np.linalg.solve(G,b),0,None)-yv[d]))
    return float(np.concatenate(errs).mean())

sel_prev = ['nf_pow90_ewm4','spend_rate28','nspend7','days_active_l1','h_act_pow90','trips_l3','nbask_max84','ntrips7','spend28_DELI','ndow5','spend_l456_mean','nf_ratio123v456','mkt_tgt_TypeA_n','nf_wsin1','nf_wsin2','nf_pow90_ewm13','h_inact_s123','nwmax12']
SP = [fcols.index(c) for c in sel_prev if c in fcols]
zw = fcols.index("r_zerow13")
print("base prev18+zw13:", round(loo(SP+[zw]),3))
for c in newf:
    print(f"+{c:14s}", round(loo(SP+[zw, fcols.index(c)]),3))
print("+store_tier+store_mean_l1+cell_mean_l1:", round(loo(SP+[zw, fcols.index("store_tier"), fcols.index("store_mean_l1"), fcols.index("cell_mean_l1")]),3))
# demographics columns themselves
democ = ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc","has_demographics"]
democ = [c for c in democ if c in fcols]
print("demo cols found:", democ)
print("+demo cols:", round(loo(SP+[zw]+[fcols.index(c) for c in democ]),3))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

key = ["household_key","snapshot_day"]
want = ['nf_pow90_ewm4','spend_rate28','nspend7','days_active_l1','h_act_pow90','trips_l3','nbask_max84','ntrips7','spend28_DELI','ndow5','spend_l456_mean','nf_ratio123v456','mkt_tgt_TypeA_n','nf_wsin1','nf_wsin2','nf_pow90_ewm13','h_inact_s123','nwmax12','r_zerow13']
src = {}
for nm in ["e003_catmix","nf_p1","nf_transforms","nf_candidates","e004_mkt_full","nf_robust"]:
    df = A.load_saved(nm+".parquet")
    have = [c for c in want if c in df.columns and c not in src]
    if have:
        src[nm] = df[key+have]
        print(nm, "->", have)
missing = [c for c in want if c not in pd.concat([s.drop(columns=key) for s in src.values()], axis=1).columns] if False else []
out = None
for nm, df in src.items():
    out = df if out is None else out.merge(df, on=key, how="inner")
out = out[key + want]
print("compact table:", out.shape, "cols ok:", set(out.columns)==set(key+want))
p = A.save_table(out, "nf_compact19.parquet")
print("saved:", p)
print(out[want].describe().T[["mean","std"]].round(2))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

key = ["household_key","snapshot_day"]
want = ['nf_pow90_ewm4','spend_rate28','nspend7','days_active_l1','h_act_pow90','trips_l3','nbask_max84','ntrips7','spend28_DELI','ndow5','spend_l456_mean','nf_ratio123v456','mkt_tgt_TypeA_n','nf_wsin1','nf_wsin2','nf_pow90_ewm13','h_inact_s123','nwmax12','r_zerow13']
order = ["e003_catmix","nf_p1","nf_transforms","nf_candidates","e004_mkt_full","nf_robust"]
tabs = {nm: A.load_saved(nm+".parquet") for nm in order}
out = None
for nm in order:
    df = tabs[nm]
    have = [c for c in want if c in df.columns and c not in (set(out.columns) if out is not None else set())]
    if not have: continue
    part = df[key+have].copy()
    out = part if out is None else out.merge(part, on=key, how="inner")
print("cols:", sorted(out.columns) == sorted(key+want), out.shape)
out = out[key+want]
p = A.save_table(out, "nf_compact19.parquet")
print("saved:", p)
print(out.isna().mean().round(3).sort_values(ascending=False).head(5))