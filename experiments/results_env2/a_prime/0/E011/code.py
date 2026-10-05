import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
print("e007:", df.shape)
print(df.dtypes.value_counts())
print("tt:", tt.shape)
print(tt.future_spend_4w.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]))
print("rows per snapshot:"); print(df.groupby("snapshot_day").size())
m = df.merge(tt, on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"]
print("missing target after merge (should be val rows):", int(y.isna().sum()))
feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
rows = []
for c in feats:
    s = m[c]
    if not pd.api.types.is_numeric_dtype(s):
        s = pd.factorize(pd.Series(s))[0]
    s = pd.to_numeric(pd.Series(s), errors="coerce").astype(float)
    rows.append((c, round(s.notna().mean(),3), s.corr(y), s.corr(y, method="spearman")))
r = pd.DataFrame(rows, columns=["feat","nonnull","pear","spear"]).set_index("feat")
r["a"] = r.spear.abs()
print(r.sort_values("a", ascending=False).to_string())


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
y = m["future_spend_4w"].values
feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
X = m[feats].copy()
obj = [c for c in feats if X[c].dtype == object]
for c in obj:
    X[c] = pd.factorize(X[c])[0]
X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(float)
print("X", X.shape, "obj cols:", obj)

def mae(p, t): return np.mean(np.abs(p - t))

# --- conditional structure ---
print("\n== y by recent-activity ==")
for col in ["spend_4w_recent","nbask_4w","spend_4w_lag1"]:
    if col in m.columns:
        g = (m[col].fillna(0) > 0)
        print(col, "| inactive n=%d mean=%.1f med=%.1f zero%%=%.1f || active n=%d mean=%.1f med=%.1f zero%%=%.1f" % (
            (~g).sum(), y[~g].mean(), np.median(y[~g]), (y[~g]==0).mean()*100,
            g.sum(), y[g].mean(), np.median(y[g]), (y[g]==0).mean()*100))

print("\n== simple predictor MAE on snapshot 431 rows (last train snapshot) ==")
t431 = m[m.snapshot_day==431]; y431 = t431.future_spend_4w.values
tr = m[m.snapshot_day<=403]
cands = {
 "global_median(<=403)": np.full(len(t431), np.median(tr.future_spend_4w)),
 "te_hh_mean": t431.te_hh_mean.values,
 "te_hh_shrunk": t431.te_hh_shrunk.values,
 "spend_112": t431.spend_112.values,
 "spend_4w_recent": t431.spend_4w_recent.values,
 "blend(te,spend112)": 0.5*(t431.te_hh_mean.values + t431.spend_112.values),
}
for k,v in cands.items(): print(f"{k:24s} {mae(v, y431):8.3f}")

print("\n== calendar: mean/median y by snapshot_day (train) ==")
cal = m.groupby("snapshot_day").future_spend_4w.agg(["mean","median","size"])
print(cal.round(1))

# --- local ridge: raw vs log1p, fit <=403, eval 431 ---
def ridge_eval(Xa, ya, mask_tr, mask_te, lam):
    mu = Xa[mask_tr].mean(0); sd = Xa[mask_tr].std(0)+1e-9
    A = (Xa[mask_tr]-mu)/sd; A = np.c_[np.ones(len(A)), A]
    b = (Xa[mask_te]-mu)/sd; b = np.c_[np.ones(len(b)), b]
    I = np.eye(A.shape[1]); I[0,0]=0
    beta = np.linalg.solve(A.T@A + lam*I, A.T@ya[mask_tr])
    return mae(b@beta, ya[mask_te])

trmask = (m.snapshot_day<=403).values; temask = (m.snapshot_day==431).values
Xs = np.sign(X)*np.log1p(np.abs(X))
print("\n== local ridge (fit<=403, eval@431) ==")
for lam in [1.0, 10.0, 100.0]:
    print(f"lam={lam:6.1f} raw {ridge_eval(X,y,trmask,temask,lam):8.3f}   logsigned {ridge_eval(Xs,y,trmask,temask,lam):8.3f}")

# duplicate columns count
Xdf = pd.DataFrame(X, columns=feats)
dup = Xdf.T.duplicated()
print("\nexact duplicate feature columns:", int(dup.sum()), feats[dup.values][:20] if dup.sum() else "")


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
te_cols = [c for c in df.columns if c.startswith("te_")]
print("TE features:", te_cols)
print(m.groupby("snapshot_day")[te_cols].mean().round(2).tail(6))

# Build leak-free outcome history per household from TRAIN targets only
tr = m[m.snapshot_day<=431][["household_key","snapshot_day","future_spend_4w"]].sort_values(["household_key","snapshot_day"])
g = tr.groupby("household_key")
tr["prev_outcomes"] = g["future_spend_4w"].cumcount()
tr["hist_mean"] = g["future_spend_4w"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
tr["hist_median"] = g["future_spend_4w"].apply(lambda s: s.shift(1).expanding().median()).reset_index(level=0, drop=True)
tr["hist_last"] = g["future_spend_4w"].shift(1)
tr["hist_ewm"] = g["future_spend_4w"].apply(lambda s: s.shift(1).ewm(halflife=2).mean()).reset_index(level=0, drop=True)
tr["hist_min"] = g["future_spend_4w"].apply(lambda s: s.shift(1).expanding().min()).reset_index(level=0, drop=True)
tr["hist_max"] = g["future_spend_4w"].apply(lambda s: s.shift(1).expanding().max()).reset_index(level=0, drop=True)
tr["hist_std"] = g["future_spend_4w"].apply(lambda s: s.shift(1).expanding().std()).reset_index(level=0, drop=True)
tr["hist_n"] = tr["prev_outcomes"]
gm = tr.future_spend_4w.expanding().mean().shift(1)  # not per hh; placeholder global
tr["hist_slope"] = tr["hist_last"] - tr["hist_mean"]

t431 = tr[tr.snapshot_day==431].merge(m[m.snapshot_day==431][["household_key","te_hh_mean","te_hh_shrunk","spend_112","spend_4w_recent","nbask_4w"]], on="household_key")
y431 = t431.future_spend_4w.values
def mae(p,t): return np.mean(np.abs(np.asarray(p)-t))
print("\n== standalone predictors @431 (n=%d) ==" % len(t431))
for c in ["hist_mean","hist_median","hist_last","hist_ewm","hist_min","hist_max","hist_slope","te_hh_mean","te_hh_shrunk","spend_4w_recent"]:
    v = t431[c].fillna(t431[c].median() if t431[c].notna().any() else 0).values
    print(f"{c:14s} {mae(v,y431):8.3f}")
print("hist_n distribution @431:", t431.hist_n.describe().round(1).to_dict())

# optimal convex blend of hist stats (grid on <=403, eval 431)
tr403 = tr[tr.snapshot_day<=403]
t403 = tr403.merge(m[m.snapshot_day<=403][["household_key","snapshot_day","te_hh_mean","te_hh_shrunk"]], on=["household_key","snapshot_day"])
cands = ["hist_mean","hist_median","hist_ewm","hist_last","te_hh_mean"]
A = t403[cands].values; ya = t403.future_spend_4w.values
best=None
for w in itertools.product(*[np.linspace(0,1,6)]*len(cands)):
    w = np.array(w)
    if w.sum()==0: continue
    w = w/w.sum()
    p = A@w
    e = mae(p, ya)
    if best is None or e<best[1]: best=(w,e)
print("\nbest blend weights (fit<=403):", dict(zip(cands, best[0].round(2))), "MAE=%.3f"%best[1])
w = best[0]
p431 = t431[cands].values@w
print("blend @431 MAE:", round(mae(p431,y431),3))


# ---- cell ----
import numpy as np, pandas as pd, warnings, itertools
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
tr = m[m.snapshot_day<=431][["household_key","snapshot_day","future_spend_4w"]].sort_values(["household_key","snapshot_day"])
g = tr.groupby("household_key")["future_spend_4w"]
for hl in [1, 1.5, 2, 3, 4, 6]:
    tr[f"ewm{hl}"] = g.apply(lambda s: s.shift(1).ewm(halflife=hl).mean()).reset_index(level=0, drop=True)
tr["last1"] = g.shift(1)
tr["last2m"] = g.apply(lambda s: s.shift(1).rolling(2).mean()).reset_index(level=0, drop=True)
tr["last3m"] = g.apply(lambda s: s.shift(1).rolling(3).mean()).reset_index(level=0, drop=True)
tr["hist_mean"] = g.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)
tr["hist_median"] = g.apply(lambda s: s.shift(1).expanding().median()).reset_index(level=0, drop=True)
tr["hist_min"] = g.apply(lambda s: s.shift(1).expanding().min()).reset_index(level=0, drop=True)
tr["hist_max"] = g.apply(lambda s: s.shift(1).expanding().max()).reset_index(level=0, drop=True)
tr["hist_std"] = g.apply(lambda s: s.shift(1).expanding().std()).reset_index(level=0, drop=True)
tr["hist_slope"] = g.apply(lambda s: s.shift(1).diff()).reset_index(level=0, drop=True)

def mae(p,t): return np.mean(np.abs(np.asarray(p)-np.asarray(t)))
print("== standalone @431 ==")
for c in [c for c in tr.columns if c.startswith("ewm") or c in ("last1","last2m","last3m","hist_mean","hist_median","hist_min","hist_max","hist_std","hist_slope")]:
    t431 = tr[tr.snapshot_day==431]
    print(f"{c:10s} {mae(t431[c].values, t431.future_spend_4w.values):8.3f}")

t431 = tr[tr.snapshot_day==431].merge(m[m.snapshot_day==431][["household_key","te_hh_mean","te_hh_shrunk","spend_4w_recent","nbask_4w","spend_112"]], on="household_key")
t403 = tr[tr.snapshot_day<=403].merge(m[m.snapshot_day<=403][["household_key","snapshot_day","te_hh_mean","te_hh_shrunk","spend_4w_recent","nbask_4w","spend_112"]], on=["household_key","snapshot_day"])
cands = ["ewm1","ewm1.5","ewm2","ewm3","ewm4","ewm6","last1","last2m","last3m","hist_mean","hist_median","hist_min","hist_max","hist_std","hist_slope","te_hh_mean","te_hh_shrunk","spend_4w_recent"]
A = t403[cands].values; ya = t403.future_spend_4w.values
rng = np.random.default_rng(0)
best_overall = (None, 1e9, None)
for size in range(1,6):
    for combo in itertools.combinations(range(len(cands)), size):
        Ac = A[:, list(combo)]
        wbest=None; ebest=1e9
        for it in range(200):
            w = rng.dirichlet(np.ones(size)*2) if it>0 else np.ones(size)/size
            e = mae(Ac@w, ya)
            if e<ebest: ebest, wbest = e, w.copy()
        if ebest < best_overall[1]:
            best_overall = ([cands[i] for i in combo], ebest, wbest)
names, e, w = best_overall
print("\nbest blend fit<=403:", names, "MAE=%.3f"%e, np.round(w,2))
print("same blend @431 MAE:", round(mae(t431[names].values@w, t431.future_spend_4w.values),3))
for pair in [("ewm2","te_hh_mean"),("ewm1.5","te_hh_mean"),("ewm2","spend_4w_recent"),("ewm1.5","ewm2","te_hh_mean"),("ewm1","ewm2","te_hh_mean")]:
    A2 = t403[list(pair)].values; wbest=None; ebest=1e9
    for it in range(400):
        w2 = rng.dirichlet(np.ones(len(pair))*2)
        e2 = mae(A2@w2, ya)
        if e2<ebest: ebest,wbest=e2,w2
    print(pair, "fit<=403 MAE=%.3f"%ebest, "@431:", round(mae(t431[list(pair)].values@wbest, t431.future_spend_4w.values),3), np.round(wbest,2))


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")

# outcome-history features from past targets (leak-free: shift(1) within household)
tr = m[["household_key","snapshot_day","future_spend_4w"]].sort_values(["household_key","snapshot_day"])
g = tr.groupby("household_key")["future_spend_4w"]
S = pd.DataFrame({f"lag{k}": g.shift(k) for k in range(1,14)})
W = {hl: 0.5**((np.arange(1,14)-1)/hl) for hl in [1,1.5,2,3,4,6]}
for hl,w in W.items():
    M = S[list(range(1,14))].notna().values.astype(float) * w[None,:]
    tr[f"oh_ewm{hl}"] = (S[list(range(1,14))].fillna(0).values * (M/ M.sum(1,keepdims=True))).sum(1)
tr["oh_last2m"] = g.apply(lambda s: s.shift(1).rolling(2).mean()).reset_index(level=0,drop=True)
tr["oh_last3m"] = g.apply(lambda s: s.shift(1).rolling(3).mean()).reset_index(level=0,drop=True)
tr["oh_mean"] = g.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0,drop=True)
tr["oh_median"] = g.apply(lambda s: s.shift(1).expanding().median()).reset_index(level=0,drop=True)
tr["oh_std"] = g.apply(lambda s: s.shift(1).expanding().std()).reset_index(level=0,drop=True)
tr["oh_min"] = g.apply(lambda s: s.shift(1).expanding().min()).reset_index(level=0,drop=True)
tr["oh_max"] = g.apply(lambda s: s.shift(1).expanding().max()).reset_index(level=0,drop=True)
tr["oh_slope"] = g.apply(lambda s: s.shift(1).diff()).reset_index(level=0,drop=True)
tr["oh_n"] = g.cumcount()
oh_cols = [c for c in tr.columns if c.startswith("oh_")]
m2 = m.merge(tr[["household_key","snapshot_day"]+oh_cols], on=["household_key","snapshot_day"], how="left")

feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
def prep(mm, cols):
    X = mm[cols].copy()
    for c in cols:
        if X[c].dtype == object: X[c] = pd.factorize(X[c])[0]
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(float)
def ridge_eval(cols, fit_max, eval_day, lam=10.0):
    mm = m2[m2.snapshot_day<=eval_day]
    trm = (mm.snapshot_day<=fit_max).values; tem = (mm.snapshot_day==eval_day).values
    Xa = prep(mm, cols); ya = mm.future_spend_4w.values
    mu, sd = Xa[trm].mean(0), Xa[trm].std(0)+1e-9
    A = np.c_[np.ones(trm.sum()), (Xa[trm]-mu)/sd]; B = np.c_[np.ones(tem.sum()), (Xa[tem]-mu)/sd]
    I = np.eye(A.shape[1]); I[0,0]=0
    beta = np.linalg.solve(A.T@A+lam*I, A.T@ya[trm])
    return np.mean(np.abs(B@beta - ya[tem]))
def mae(p,t): return np.mean(np.abs(np.asarray(p)-np.asarray(t)))

base = feats[:]
print("n base feats:", len(base), "n oh feats:", len(oh_cols))
for lam in [1.0, 10.0, 100.0]:
    print(f"lam={lam}: base@431 {ridge_eval(base,403,431,lam):.3f}  base+oh@431 {ridge_eval(base+oh_cols,403,431,lam):.3f}")
print("eval@403(fit<=375): base", round(ridge_eval(base,375,403),3), " base+oh", round(ridge_eval(base+oh_cols,375,403),3))
print("eval@375(fit<=347): base", round(ridge_eval(base,347,375),3), " base+oh", round(ridge_eval(base+oh_cols,347,375),3))

# LEAN table: te block + demo + calendar + core rfm + oh
lean = ["te_hh_mean","te_hh_n","te_hh_shrunk","te_bin","te_prior","te_size","te_home","te_kid",
        "spend_112","spend_56w","spend_28w","spend_12w","spend_4w_recent","nbask_4w","nbask_12w","nbask_112w",
        "nprod_112","nweek_active_112","days_since_last_basket","tenure_days","spend_per_active_week_112",
        "sum_retail_disc","sum_coupon_disc","has_demographics","snapshot_day","week_of_year"]
lean = [c for c in lean if c in df.columns]
print("\nlean cols found:", len(lean), [c for c in lean if c not in df.columns])
for lam in [1.0,10.0,100.0]:
    print(f"lam={lam}: lean@431 {ridge_eval(lean,403,431,lam):.3f}  lean+oh@431 {ridge_eval(lean+oh_cols,403,431,lam):.3f}")
print("eval@403: lean", round(ridge_eval(lean,375,403),3), " lean+oh", round(ridge_eval(lean+oh_cols,375,403),3))


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")

tr = m[["household_key","snapshot_day","future_spend_4w"]].sort_values(["household_key","snapshot_day"])
g = tr.groupby("household_key")["future_spend_4w"]
lagcols = [f"lag{k}" for k in range(1,14)]
S = pd.DataFrame({c: g.shift(int(c[3:])) for c in lagcols})
for hl in [1,1.5,2,3,4,6]:
    w = 0.5**((np.arange(1,14)-1)/hl)
    M = S.notna().values.astype(float) * w[None,:]
    tr[f"oh_ewm{hl}"] = (S.fillna(0).values * (M/ M.sum(1,keepdims=True))).sum(1)
tr["oh_last2m"] = g.apply(lambda s: s.shift(1).rolling(2).mean()).reset_index(level=0,drop=True)
tr["oh_last3m"] = g.apply(lambda s: s.shift(1).rolling(3).mean()).reset_index(level=0,drop=True)
tr["oh_mean"] = g.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0,drop=True)
tr["oh_median"] = g.apply(lambda s: s.shift(1).expanding().median()).reset_index(level=0,drop=True)
tr["oh_std"] = g.apply(lambda s: s.shift(1).expanding().std()).reset_index(level=0,drop=True)
tr["oh_min"] = g.apply(lambda s: s.shift(1).expanding().min()).reset_index(level=0,drop=True)
tr["oh_max"] = g.apply(lambda s: s.shift(1).expanding().max()).reset_index(level=0,drop=True)
tr["oh_slope"] = g.apply(lambda s: s.shift(1).diff()).reset_index(level=0,drop=True)
tr["oh_n"] = g.cumcount()
oh_cols = [c for c in tr.columns if c.startswith("oh_")]
m2 = m.merge(tr[["household_key","snapshot_day"]+oh_cols], on=["household_key","snapshot_day"], how="left")

feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
def prep(mm, cols):
    X = mm[cols].copy()
    for c in cols:
        if X[c].dtype == object: X[c] = pd.factorize(X[c])[0]
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(float)
def ridge_eval(cols, fit_max, eval_day, lam=10.0):
    mm = m2[m2.snapshot_day<=eval_day]
    trm = (mm.snapshot_day<=fit_max).values; tem = (mm.snapshot_day==eval_day).values
    Xa = prep(mm, cols); ya = mm.future_spend_4w.values
    mu, sd = Xa[trm].mean(0), Xa[trm].std(0)+1e-9
    A = np.c_[np.ones(trm.sum()), (Xa[trm]-mu)/sd]; B = np.c_[np.ones(tem.sum()), (Xa[tem]-mu)/sd]
    I = np.eye(A.shape[1]); I[0,0]=0
    beta = np.linalg.solve(A.T@A+lam*I, A.T@ya[trm])
    return np.mean(np.abs(B@beta - ya[tem]))
def mae(p,t): return np.mean(np.abs(np.asarray(p)-np.asarray(t)))

base = feats[:]
print("n base feats:", len(base), "n oh feats:", len(oh_cols))
for lam in [1.0, 10.0, 100.0]:
    print(f"lam={lam}: base@431 {ridge_eval(base,403,431,lam):.3f}  base+oh@431 {ridge_eval(base+oh_cols,403,431,lam):.3f}")
print("eval@403(fit<=375): base", round(ridge_eval(base,375,403),3), " base+oh", round(ridge_eval(base+oh_cols,375,403),3))
print("eval@375(fit<=347): base", round(ridge_eval(base,347,375),3), " base+oh", round(ridge_eval(base+oh_cols,347,375),3))

lean = ["te_hh_mean","te_hh_n","te_hh_shrunk","te_bin","te_prior","te_size","te_home","te_kid",
        "spend_112","spend_56w","spend_28w","spend_12w","spend_4w_recent","nbask_4w","nbask_12w","nbask_112w",
        "nprod_112","nweek_active_112","days_since_last_basket","tenure_days","spend_per_active_week_112",
        "sum_retail_disc","sum_coupon_disc","has_demographics","snapshot_day","week_of_year"]
lean = [c for c in lean if c in df.columns]
print("\nlean cols found:", len(lean), "missing:", [c for c in lean if c not in df.columns])
for lam in [1.0,10.0,100.0]:
    print(f"lam={lam}: lean@431 {ridge_eval(lean,403,431,lam):.3f}  lean+oh@431 {ridge_eval(lean+oh_cols,403,431,lam):.3f}")
print("eval@403: lean", round(ridge_eval(lean,375,403),3), " lean+oh", round(ridge_eval(lean+oh_cols,375,403),3))
# standalone oh in ridge (only oh cols)
print("oh-only@431:", round(ridge_eval(oh_cols,403,431),3))


# ---- cell ----
import numpy as np, pandas as pd, warnings, itertools
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")

tr = m[["household_key","snapshot_day","future_spend_4w"]].sort_values(["household_key","snapshot_day"])
g = tr.groupby("household_key")["future_spend_4w"]
lagcols = [f"lag{k}" for k in range(1,14)]
S = pd.DataFrame({c: g.shift(int(c[3:])) for c in lagcols})
for hl in [1.5, 4]:
    w = 0.5**((np.arange(1,14)-1)/hl)
    M = S.notna().values.astype(float) * w[None,:]
    tr[f"oh_ewm{hl}"] = (S.fillna(0).values * (M/ M.sum(1,keepdims=True))).sum(1)
tr["oh_n"] = g.cumcount()
m2 = m.merge(tr[["household_key","snapshot_day","oh_ewm1.5","oh_ewm4","oh_n"]], on=["household_key","snapshot_day"], how="left")

feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
# exact duplicate columns
Xd = m2[feats].copy()
for c in feats:
    if Xd[c].dtype == object: Xd[c] = pd.factorize(Xd[c])[0]
Xd = Xd.apply(pd.to_numeric, errors="coerce").fillna(0.0)
dupmask = Xd.T.duplicated().values
dups = [f for f,d in zip(feats,dupmask) if d]
print("exact dup cols (%d):"%len(dups), dups)

def prep(mm, cols):
    X = mm[cols].copy()
    for c in cols:
        if X[c].dtype == object: X[c] = pd.factorize(X[c])[0]
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(float)
def ridge_eval(cols, fit_max, eval_day, lam=10.0):
    mm = m2[m2.snapshot_day<=eval_day]
    trm = (mm.snapshot_day<=fit_max).values; tem = (mm.snapshot_day==eval_day).values
    Xa = prep(mm, cols); ya = mm.future_spend_4w.values
    mu, sd = Xa[trm].mean(0), Xa[trm].std(0)+1e-9
    A = np.c_[np.ones(trm.sum()), (Xa[trm]-mu)/sd]; B = np.c_[np.ones(tem.sum()), (Xa[tem]-mu)/sd]
    I = np.eye(A.shape[1]); I[0,0]=0
    beta = np.linalg.solve(A.T@A+lam*I, A.T@ya[trm])
    return np.mean(np.abs(B@beta - ya[tem]))

# greedy forward selection on proxy (fit<=403, eval@431), starting from empty, candidate pool = strongest 40 by |spear|
y_all = m2.future_spend_4w.values
pool = []
for c in feats + ["oh_ewm1.5","oh_ewm4","oh_n"]:
    s = m2[c]
    if s.dtype == object: s = pd.factorize(s)[0]
    s = pd.to_numeric(pd.Series(s), errors="coerce")
    pool.append((abs(s.corr(m2.future_spend_4w, method="spearman")), c))
pool.sort(reverse=True)
cands = [c for _,c in pool[:45]]
print("top45:", cands[:15], "...")
sel = []
cur = 1e9
for it in range(14):
    best = None
    for c in cands:
        if c in sel: continue
        e = ridge_eval(sel+[c], 403, 431)
        if best is None or e < best[1]: best = (c, e)
    if best[1] < cur - 1e-4:
        sel.append(best[0]); cur = best[1]
        print(f"step{it+1}: +{best[0]:26s} MAE={best[1]:.3f}  (n={len(sel)})")
    else:
        print("no improvement; stop"); break
print("\nfinal sel:", sel, "MAE=", round(cur,3))
print("full base+oh:", round(ridge_eval(feats+["oh_ewm1.5","oh_ewm4","oh_n"],403,431),3))
print("base dedup:", round(ridge_eval([f for f in feats if f not in dups],403,431),3))


# ---- cell ----
import pandas as pd
df = agent_api.load_saved("e007_te.parquet")
feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
print(len(feats))
for i in range(0, len(feats), 8):
    print(" | ".join(f"{c:28s}" for c in feats[i:i+8]))


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
tr = m[["household_key","snapshot_day","future_spend_4w"]].sort_values(["household_key","snapshot_day"])
g = tr.groupby("household_key")["future_spend_4w"]
lagcols = [f"lag{k}" for k in range(1,14)]
S = pd.DataFrame({c: g.shift(int(c[3:])) for c in lagcols})
for hl in [1.5, 4]:
    w = 0.5**((np.arange(1,14)-1)/hl)
    M = S.notna().values.astype(float) * w[None,:]
    tr[f"oh_ewm{hl}"] = (S.fillna(0).values * (M/ M.sum(1,keepdims=True))).sum(1)
tr["oh_n"] = g.cumcount()
m2 = m.merge(tr[["household_key","snapshot_day","oh_ewm1.5","oh_ewm4","oh_n"]], on=["household_key","snapshot_day"], how="left")

# raw demographics (static) — merge on household
demo = agent_api.snapshot().demographics
print("demo rows:", len(demo), demo.columns.tolist())
d2 = demo.copy()
for c in d2.columns:
    if c != "household_key": d2[c] = d2[c].astype(str)
m2 = m2.merge(d2, on="household_key", how="left")
m2["has_demo"] = m2["household_key"].isin(set(demo.household_key)).astype(float)

def prep(mm, cols):
    X = mm[cols].copy()
    for c in cols:
        if X[c].dtype == object: X[c] = pd.factorize(X[c])[0]
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(float)
def ridge_eval(cols, fit_max, eval_day, lam=10.0):
    mm = m2[m2.snapshot_day<=eval_day]
    trm = (mm.snapshot_day<=fit_max).values; tem = (mm.snapshot_day==eval_day).values
    Xa = prep(mm, cols); ya = mm.future_spend_4w.values
    mu, sd = Xa[trm].mean(0), Xa[trm].std(0)+1e-9
    A = np.c_[np.ones(trm.sum()), (Xa[trm]-mu)/sd]; B = np.c_[np.ones(tem.sum()), (Xa[tem]-mu)/sd]
    I = np.eye(A.shape[1]); I[0,0]=0
    beta = np.linalg.solve(A.T@A+lam*I, A.T@ya[trm])
    return np.mean(np.abs(B@beta - ya[tem]))

te = ["te_hh_mean","te_hh_n","te_hh_shrunk","te_bin","te_prior","te_size","te_home","te_kid"]
core = ["spend_4w","nbask_4w","spend_8w","nbask_8w","spend_12w","nbask_12w","spend_28w","nbask_28w",
        "spend_56w","nbask_56w","spend_112w","nbask_112w","spend_total","nbask_total","tenure_days",
        "days_since_last","avg_basket_12w","trend_4_8","trend_4_28","active_4w",
        "spend_4w_lag1","spend_4w_lag2","spend_4w_lag3","spend_4w_lag4","spend_4w_lag5","spend_4w_lag6",
        "spend_112_mean4","spend_112_min4","spend_112_max4","spend_112_std","spend_112_cv",
        "gap_mean_112","gap_max_112","gap_std_112","nweek_active_112","week_active_share_112",
        "spend_per_week_112","spend_per_active_week_112","spend_per_basket_112","spend_per_basket_total",
        "spend_per_day_total","avg_price_line_112","avg_units_line_112","disc_share_112","neg_spend_share_112",
        "nprod_112","nstore_112","mean_trans_time_112"]
dsh = [c for c in df.columns if c.startswith("dsh_") and c!="dsh_ "] + ["bsh_National","bsh_Private"]
mkt = ["n_campaign_targets","days_since_last_tgt_start","tgt_active_now","tgt_active_future4w","tgt_TypeA","tgt_TypeB","tgt_TypeC",
       "n_redemptions_total","days_since_last_redemption","n_redemptions_28d","n_redemptions_56d","n_redemptions_112d"]
disp = ["share_disp_28","share_mail_28","share_lines_disp_28","share_baskets_disp_28","share_disp_112","share_mail_112",
        "share_lines_disp_112","share_baskets_disp_112","share_spend_nowdisp_28"]
demo_cols = [c for c in d2.columns if c!="household_key"] + ["has_demo"]
cal = ["snapshot_day"]
oh = ["oh_ewm1.5","oh_ewm4","oh_n"]
print("sizes: te%d core%d dsh%d mkt%d disp%d demo%d"%(len(te),len(core),len(dsh),len(mkt),len(disp),len(demo_cols)))

variants = {
 "A core+te+demo+cal+oh": te+core+demo_cols+cal+oh,
 "B A+dsh": te+core+demo_cols+cal+oh+dsh,
 "C B+mkt": te+core+demo_cols+cal+oh+dsh+mkt,
 "D C+disp(=E007lean)": te+core+demo_cols+cal+oh+dsh+mkt+disp,
 "E core+te+cal+oh (no demo)": te+core+cal+oh,
 "F E+mkt+disp": te+core+cal+oh+mkt+disp,
}
for name, cols in variants.items():
    e431 = ridge_eval(cols,403,431); e403 = ridge_eval(cols,375,403)
    print(f"{name:30s} n={len(cols):3d} @431 {e431:7.3f}  @403 {e403:7.3f}")
print("E007 full + oh + demo:", round(ridge_eval([c for c in df.columns if c not in ('household_key','snapshot_day','day','week','tenure','snap_day','snap_week','index')]+oh+demo_cols,403,431),3))


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
tr = m[["household_key","snapshot_day","future_spend_4w"]].sort_values(["household_key","snapshot_day"])
g = tr.groupby("household_key")["future_spend_4w"]
lagcols = [f"lag{k}" for k in range(1,14)]
S = pd.DataFrame({c: g.shift(int(c[3:])) for c in lagcols})
for hl in [1.5, 4]:
    w = 0.5**((np.arange(1,14)-1)/hl)
    M = S.notna().values.astype(float) * w[None,:]
    tr[f"oh_ewm{hl}"] = (S.fillna(0).values * (M/ M.sum(1,keepdims=True))).sum(1)
tr["oh_n"] = g.cumcount()
m2 = m.merge(tr[["household_key","snapshot_day","oh_ewm1.5","oh_ewm4","oh_n"]], on=["household_key","snapshot_day"], how="left")
demo = agent_api.snapshot().demographics
d2 = demo.copy()
for c in d2.columns:
    if c != "household_key": d2[c] = d2[c].astype(str)
m2 = m2.merge(d2, on="household_key", how="left")
m2["has_demo"] = m2["household_key"].isin(set(demo.household_key)).astype(float)

def prep(mm, cols):
    X = mm[cols].copy()
    for c in cols:
        if X[c].dtype == object: X[c] = pd.factorize(X[c])[0]
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(float)
def ridge_eval(cols, fit_max, eval_day, lam=10.0):
    mm = m2[m2.snapshot_day<=eval_day]
    trm = (mm.snapshot_day<=fit_max).values; tem = (mm.snapshot_day==eval_day).values
    Xa = prep(mm, cols); ya = mm.future_spend_4w.values
    mu, sd = Xa[trm].mean(0), Xa[trm].std(0)+1e-9
    A = np.c_[np.ones(trm.sum()), (Xa[trm]-mu)/sd]; B = np.c_[np.ones(tem.sum()), (Xa[tem]-mu)/sd]
    I = np.eye(A.shape[1]); I[0,0]=0
    beta = np.linalg.solve(A.T@A+lam*I, A.T@ya[trm])
    return np.mean(np.abs(B@beta - ya[tem]))

te = ["te_hh_mean","te_hh_n","te_hh_shrunk","te_bin","te_prior","te_size","te_home","te_kid"]
core = ["spend_4w","nbask_4w","spend_8w","nbask_8w","spend_12w","nbask_12w","spend_28w","nbask_28w",
        "spend_56w","nbask_56w","spend_112w","nbask_112w","spend_total","nbask_total","tenure_days",
        "days_since_last","avg_basket_12w","trend_4_8","trend_4_28","active_4w",
        "spend_4w_lag1","spend_4w_lag2","spend_4w_lag3","spend_112_mean4","spend_112_min4","spend_112_max4",
        "spend_112_std","spend_112_cv","gap_mean_112","gap_max_112","gap_std_112","nweek_active_112",
        "week_active_share_112","spend_per_week_112","spend_per_active_week_112","spend_per_basket_112",
        "spend_per_basket_total","spend_per_day_total","avg_price_line_112","avg_units_line_112",
        "disc_share_112","neg_spend_share_112","nprod_112","nstore_112","mean_trans_time_112"]
dsh = [c for c in df.columns if c.startswith("dsh_") and c!="dsh_ "] + ["bsh_National","bsh_Private"]
mkt = ["n_campaign_targets","days_since_last_tgt_start","tgt_active_now","tgt_active_future4w","tgt_TypeA","tgt_TypeB","tgt_TypeC",
       "n_redemptions_total","days_since_last_redemption","n_redemptions_28d","n_redemptions_56d","n_redemptions_112d"]
disp = ["share_disp_28","share_mail_28","share_lines_disp_28","share_baskets_disp_28","share_disp_112","share_mail_112",
        "share_lines_disp_112","share_baskets_disp_112","share_spend_nowdisp_28"]
demo_cols = [c for c in d2.columns if c!="household_key"] + ["has_demo"]
cal = ["snapshot_day"]
oh = ["oh_ewm1.5","oh_ewm4","oh_n"]
variants = {
 "A core+te+demo+cal+oh": te+core+demo_cols+cal+oh,
 "B A+dsh": te+core+demo_cols+cal+oh+dsh,
 "C B+mkt": te+core+demo_cols+cal+oh+dsh+mkt,
 "D C+disp(=E007lean)": te+core+demo_cols+cal+oh+dsh+mkt+disp,
 "E core+te+cal+oh": te+core+cal+oh,
 "F E+mkt+disp": te+core+cal+oh+mkt+disp,
 "G full E007+oh+demo": [c for c in df.columns if c not in ('household_key','snapshot_day','day','week','tenure','snap_day','snap_week','index')]+oh+demo_cols,
}
for name, cols in variants.items():
    print(f"{name:24s} n={len(cols):3d} @431 {ridge_eval(cols,403,431):7.3f}  @403 {ridge_eval(cols,375,403):7.3f}")


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
tr = m[["household_key","snapshot_day","future_spend_4w"]].sort_values(["household_key","snapshot_day"])
g = tr.groupby("household_key")["future_spend_4w"]
lagcols = [f"lag{k}" for k in range(1,14)]
S = pd.DataFrame({c: g.shift(int(c[3:])) for c in lagcols})
for hl in [1.5, 4]:
    w = 0.5**((np.arange(1,14)-1)/hl)
    M = S.notna().values.astype(float) * w[None,:]
    tr[f"oh_ewm{hl}"] = (S.fillna(0).values * (M/ M.sum(1,keepdims=True))).sum(1)
tr["oh_n"] = g.cumcount()
# new: zero-lag count among available, zero share of 4w windows (last 6 lags), churn-adjusted forecast
Z = S[["lag1","lag2","lag3","lag4","lag5","lag6"]]
tr["oh_zero_share6"] = (Z==0).sum(1) / Z.notna().sum(1).clip(lower=1)
tr["oh_zero_any6"] = ((Z==0).sum(1) > 0).astype(float)
tr["oh_ewm15_x_active"] = tr["oh_ewm1.5"] * (1 - tr["oh_zero_share6"].fillna(0))
tr["oh_ewm15_x_zeroflag"] = tr["oh_ewm1.5"] * (1 - tr["oh_zero_any6"])
tr["oh_min3"] = S[["lag1","lag2","lag3"]].min(1)
tr["oh_max3"] = S[["lag1","lag2","lag3"]].max(1)
tr["oh_trend3"] = S["lag1"] - S["lag3"]
tr["oh_cv3"] = S[["lag1","lag2","lag3"]].std(1) / S[["lag1","lag2","lag3"]].mean(1).replace(0, np.nan)
new_cols = ["oh_zero_share6","oh_zero_any6","oh_ewm15_x_active","oh_ewm15_x_zeroflag","oh_min3","oh_max3","oh_trend3","oh_cv3"]
m2 = m.merge(tr[["household_key","snapshot_day","oh_ewm1.5","oh_ewm4","oh_n"]+new_cols], on=["household_key","snapshot_day"], how="left")

def prep(mm, cols):
    X = mm[cols].copy()
    for c in cols:
        if X[c].dtype == object: X[c] = pd.factorize(X[c])[0]
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(float)
def ridge_eval(cols, fit_max, eval_day, lam=10.0):
    mm = m2[m2.snapshot_day<=eval_day]
    trm = (mm.snapshot_day<=fit_max).values; tem = (mm.snapshot_day==eval_day).values
    Xa = prep(mm, cols); ya = mm.future_spend_4w.values
    mu, sd = Xa[trm].mean(0), Xa[trm].std(0)+1e-9
    A = np.c_[np.ones(trm.sum()), (Xa[trm]-mu)/sd]; B = np.c_[np.ones(tem.sum()), (Xa[tem]-mu)/sd]
    I = np.eye(A.shape[1]); I[0,0]=0
    beta = np.linalg.solve(A.T@A+lam*I, A.T@ya[trm])
    return np.mean(np.abs(B@beta - ya[tem]))

te = ["te_hh_mean","te_hh_n","te_hh_shrunk","te_bin","te_prior","te_size","te_home","te_kid"]
core = ["spend_4w","nbask_4w","spend_8w","nbask_8w","spend_12w","nbask_12w","spend_28w","nbask_28w",
        "spend_56w","nbask_56w","spend_112w","nbask_112w","spend_total","nbask_total","tenure_days",
        "days_since_last","avg_basket_12w","trend_4_8","trend_4_28","active_4w",
        "spend_4w_lag1","spend_4w_lag2","spend_4w_lag3","spend_112_mean4","spend_112_min4","spend_112_max4",
        "spend_112_std","spend_112_cv","gap_mean_112","gap_max_112","gap_std_112","nweek_active_112",
        "week_active_share_112","spend_per_week_112","spend_per_active_week_112","spend_per_basket_112",
        "spend_per_basket_total","spend_per_day_total","avg_price_line_112","avg_units_line_112",
        "disc_share_112","neg_spend_share_112","nprod_112","nstore_112","mean_trans_time_112"]
oh = ["oh_ewm1.5","oh_ewm4","oh_n"]
baseE = te+core+["snapshot_day"]+oh
print("baseE:", round(ridge_eval(baseE,403,431),3), round(ridge_eval(baseE,375,403),3))
for c in new_cols:
    v = baseE+[c]
    print(f"+{c:22s} @431 {ridge_eval(v,403,431):7.3f}  @403 {ridge_eval(v,375,403):7.3f}")
allnew = baseE+new_cols
print("baseE+allnew:", round(ridge_eval(allnew,403,431),3), round(ridge_eval(allnew,375,403),3), round(ridge_eval(allnew,347,375),3))
print("baseE @375:", round(ridge_eval(baseE,347,375),3))
# also standalone quality of new cols
def mae(p,t): return np.mean(np.abs(np.asarray(p)-np.asarray(t)))
t431 = m2[m2.snapshot_day==431]
for c in new_cols+["oh_ewm1.5"]:
    print("standalone", c, round(mae(t431[c].fillna(t431[c].median()).values, t431.future_spend_4w.values),3))


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

JUNK = ["day","week","tenure","snap_day","snap_week","index"]

def make_feats(view, snapshot_day):
    s = int(snapshot_day)
    hh = pd.Index(list(view.households), name="household_key")
    e7 = agent_api.load_saved("e007_te.parquet")
    e7s = e7[e7["snapshot_day"] == s].set_index("household_key")
    e7s = e7s.drop(columns=[c for c in JUNK + ["snapshot_day"] if c in e7s.columns])
    e7s = e7s.reindex(hh)
    # leak-free outcome history from earlier snapshots' realized targets
    tt = agent_api.train_targets()
    tt = tt[tt["snapshot_day"] < s].sort_values(["household_key", "snapshot_day"])
    gm = tt["future_spend_4w"].mean() if len(tt) else np.nan
    g = tt.groupby("household_key")["future_spend_4w"]
    lagcols = ["lag%d" % k for k in range(1, 14)]
    S = pd.DataFrame({c: g.shift(int(c[3:])) for c in lagcols})
    oh = pd.DataFrame(index=tt.index)
    for hl, name in [(1.5, "oh_ewm15"), (4.0, "oh_ewm4")]:
        w = 0.5 ** ((np.arange(1, 14) - 1.0) / hl)
        M = S.notna().values.astype(float) * w[None, :]
        den = M.sum(1, keepdims=True); den[den == 0] = 1.0
        oh[name] = (S.fillna(0).values * (M / den)).sum(1)
    oh["oh_n"] = g.cumcount().astype(float)
    Z = S[["lag1", "lag2", "lag3", "lag4", "lag5", "lag6"]]
    oh["oh_zero_share6"] = (Z == 0).sum(1) / Z.notna().sum(1).clip(lower=1)
    oh["oh_zero_any6"] = ((Z == 0).sum(1) > 0).astype(float)
    oh["oh_min3"] = S[["lag1", "lag2", "lag3"]].min(1)
    oh["oh_max3"] = S[["lag1", "lag2", "lag3"]].max(1)
    oh["oh_trend3"] = S["lag1"] - S["lag3"]
    m3 = S[["lag1", "lag2", "lag3"]]
    oh["oh_cv3"] = m3.std(1) / m3.mean(1).replace(0, np.nan)
    oh["oh_ewm15_x_active"] = oh["oh_ewm15"] * (1 - oh["oh_zero_share6"].fillna(0))
    oh["oh_ewm15_x_zeroflag"] = oh["oh_ewm15"] * (1 - oh["oh_zero_any6"])
    oh = oh.reindex(hh)
    nohist = oh["oh_n"].fillna(0) == 0
    oh.loc[nohist, ["oh_zero_share6","oh_zero_any6","oh_min3","oh_max3","oh_trend3","oh_cv3",
                    "oh_ewm15_x_active","oh_ewm15_x_zeroflag"]] = np.nan
    for c in ["oh_ewm15", "oh_ewm4", "oh_min3", "oh_max3"]:
        oh[c] = oh[c].fillna(gm)
    df = e7s.join(oh)
    demo = view.table("demographics")
    d2 = demo.set_index("household_key").astype(str)
    df = df.join(d2, how="left")
    df["has_demo"] = df.index.isin(set(demo["household_key"])).astype(float)
    df["snapshot_day"] = float(s)
    return df

res = agent_api.build_features(make_feats)
print("shape:", res.shape)
print("rows/snapshot:", res.groupby("snapshot_day").size().to_dict())
print("n feature cols:", res.shape[1] - 2)
print("oh_ewm15 nonnull:", round(res.oh_ewm15.notna().mean(), 3), " te_hh_mean nonnull:", round(res.te_hh_mean.notna().mean(), 3))
print("any NaN keys:", res.household_key.isna().any(), res.snapshot_day.isna().any())
print(res[["oh_ewm15","oh_n","te_hh_mean","snapshot_day"]].describe().round(2))
path = agent_api.save_table(res, "e011_outcomehist.parquet")
print("saved:", path)


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

E7 = agent_api.load_saved("e007_te.parquet")
JUNK = ["day","week","tenure","snap_day","snap_week","index","snapshot_day"]

def make_feats(view, snapshot_day):
    s = int(snapshot_day)
    hh = pd.Index(list(view.households), name="household_key")
    e7s = E7[E7["snapshot_day"] == s].set_index("household_key")
    e7s = e7s.drop(columns=[c for c in JUNK if c in e7s.columns])
    e7s = e7s.reindex(hh)
    tt = agent_api.train_targets()
    tt = tt[tt["snapshot_day"] < s].sort_values(["household_key", "snapshot_day"])
    gm = tt["future_spend_4w"].mean() if len(tt) else np.nan
    g = tt.groupby("household_key")["future_spend_4w"]
    lagcols = ["lag%d" % k for k in range(1, 14)]
    S = pd.DataFrame({c: g.shift(int(c[3:])) for c in lagcols})
    oh = pd.DataFrame(index=tt.index)
    for hl, name in [(1.5, "oh_ewm15"), (4.0, "oh_ewm4")]:
        w = 0.5 ** ((np.arange(1, 14) - 1.0) / hl)
        M = S.notna().values.astype(float) * w[None, :]
        den = M.sum(1, keepdims=True); den[den == 0] = 1.0
        oh[name] = (S.fillna(0).values * (M / den)).sum(1)
    oh["oh_n"] = g.cumcount().astype(float)
    Z = S[["lag1", "lag2", "lag3", "lag4", "lag5", "lag6"]]
    oh["oh_zero_share6"] = (Z == 0).sum(1) / Z.notna().sum(1).clip(lower=1)
    oh["oh_zero_any6"] = ((Z == 0).sum(1) > 0).astype(float)
    oh["oh_min3"] = S[["lag1", "lag2", "lag3"]].min(1)
    oh["oh_max3"] = S[["lag1", "lag2", "lag3"]].max(1)
    oh["oh_trend3"] = S["lag1"] - S["lag3"]
    m3 = S[["lag1", "lag2", "lag3"]]
    oh["oh_cv3"] = m3.std(1) / m3.mean(1).replace(0, np.nan)
    oh["oh_ewm15_x_active"] = oh["oh_ewm15"] * (1 - oh["oh_zero_share6"].fillna(0))
    oh["oh_ewm15_x_zeroflag"] = oh["oh_ewm15"] * (1 - oh["oh_zero_any6"])
    oh = oh.reindex(hh)
    nohist = oh["oh_n"].fillna(0) == 0
    oh.loc[nohist, ["oh_zero_share6","oh_zero_any6","oh_min3","oh_max3","oh_trend3","oh_cv3",
                    "oh_ewm15_x_active","oh_ewm15_x_zeroflag"]] = np.nan
    for c in ["oh_ewm15", "oh_ewm4", "oh_min3", "oh_max3"]:
        oh[c] = oh[c].fillna(gm)
    df = e7s.join(oh)
    demo = view.table("demographics")
    d2 = demo.set_index("household_key").astype(str)
    df = df.join(d2, how="left")
    df["has_demo"] = df.index.isin(set(demo["household_key"])).astype(float)
    df["snapshot_day"] = float(s)
    return df

res = agent_api.build_features(make_feats)
print("shape:", res.shape)
print("rows/snapshot:", res.groupby("snapshot_day").size().to_dict())
print("n feature cols:", res.shape[1] - 2)
print("oh_ewm15 nonnull:", round(res.oh_ewm15.notna().mean(), 3), " te_hh_mean nonnull:", round(res.te_hh_mean.notna().mean(), 3))
print("any NaN keys:", res.household_key.isna().any(), res.snapshot_day.isna().any())
print(res[["oh_ewm15","oh_n","te_hh_mean","snapshot_day"]].describe().round(2))
path = agent_api.save_table(res, "e011_outcomehist.parquet")
print("saved:", path)
