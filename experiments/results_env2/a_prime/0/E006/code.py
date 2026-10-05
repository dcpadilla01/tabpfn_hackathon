import numpy as np, pandas as pd, agent_api as A

df = A.load_saved("e005_marketing.parquet")
print("shape", df.shape)
cols = list(df.columns)
print("n cols", len(cols))
for i in range(0, len(cols), 6):
    print(" | ".join(cols[i:i+6]))

tt = A.train_targets()
y = tt["future_spend_4w"].astype(float)
print("\ntarget describe:\n", y.describe())
print("zero share:", float((y==0).mean()))
print("snapshot_days:", A.snapshot_days())

m = tt.merge(df, on=["household_key","snapshot_day"], how="left")
print("merged shape", m.shape)
ymed = float(y.median())
print("const median MAE:", round(float(np.abs(y-ymed).mean()),2), "median:", ymed)
print("const mean MAE:", round(float(np.abs(y-y.mean()).mean()),2))

feat_cols = [c for c in cols if c not in ("household_key","snapshot_day")]
spend_cols = [c for c in feat_cols if "spend" in c.lower()]
print("\nspend cols:", spend_cols)
for c in spend_cols:
    p = pd.to_numeric(m[c], errors="coerce").fillna(0).values
    if p.std()>0:
        print(f"{c}: rawMAE={np.abs(y-p).mean():.2f} corr={np.corrcoef(y,p)[0,1]:.3f} logcorr={np.corrcoef(y,np.log1p(p))[0,1]:.3f}")

c28 = [c for c in spend_cols if "28" in c and "share" not in c]
if c28:
    p = pd.to_numeric(m[c28[0]], errors="coerce").fillna(0).values
    al = np.linspace(0,2,41); maes=[np.abs(y-a*p).mean() for a in al]; i=int(np.argmin(maes))
    print("\nbest scalar on", c28[0], "alpha", round(float(al[i]),2), "MAE", round(float(maes[i]),2))

num = m[feat_cols].select_dtypes(include=[np.number])
corr = num.corrwith(y).dropna()
top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(30)
print("\ntop |corr| with target:\n", top)


# ---- cell ----
import numpy as np, pandas as pd, agent_api as A

df = A.load_saved("e005_marketing.parquet")
tt = A.train_targets()
m = tt.merge(df, on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"].astype(float).values

tr_days = A.snapshot_days()["train"]; va_days = A.snapshot_days()["validation"]
is_val = m["snapshot_day"].isin(va_days).values
is_tr = m["snapshot_day"].isin(tr_days).values

# Oracle: household's own median of y across TRAIN snapshots only (no val leakage)
tr = m[is_tr]
med = tr.groupby("household_key")["future_spend_4w"].median()
va = m[is_val]
pred_or = va["household_key"].map(med).fillna(tr["future_spend_4w"].median()).values
print("oracle (household median from train snaps) val MAE:", round(float(np.abs(va["future_spend_4w"]-pred_or).mean()),2))

# Oracle 2: household median including val snapshots (upper bound of persistence)
med2 = m.groupby("household_key")["future_spend_4w"].median()
pred_or2 = va["household_key"].map(med2).fillna(m["future_spend_4w"].median()).values
print("oracle (all snaps) val MAE:", round(float(np.abs(va["future_spend_4w"]-pred_or2).mean()),2))

feat_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
X = m[feat_cols].copy()
for c in X.columns:
    if not np.issubdtype(X[c].dtype, np.number):
        X[c] = X[c].astype("category").cat.codes
X = X.replace([np.inf,-np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True)).fillna(0).values.astype(float)

def ridge_eval(X, y, is_tr, is_val, alpha=100.0, standardize=True, logy=False):
    mu, sd = X[is_tr].mean(0), X[is_tr].std(0)+1e-9
    Z = (X-mu)/sd if standardize else X
    yt = np.log1p(y) if logy else y
    A_ = Z[is_tr]; b = yt[is_tr]
    d = A_.shape[1]
    G = A_.T@A_ + alpha*np.eye(d); G[-1,-1] -= alpha  # don't penalize intercept-ish? simple
    w = np.linalg.solve(G, A_.T@b)
    p = Z[is_val]@w
    if logy: p = np.expm1(np.clip(p,0,8))
    return float(np.abs(y[is_val]-p).mean()), float(np.abs(y[is_tr]-Z[is_tr]@w).mean() if not logy else np.abs(y[is_tr]-np.expm1(Z[is_tr]@w)).mean())

for a in [10,100,1000]:
    mae, trm = ridge_eval(X, y, is_tr, is_val, alpha=a)
    print(f"ridge alpha={a}: val MAE {mae:.2f} (train {trm:.2f})")
mae, trm = ridge_eval(X, y, is_tr, is_val, alpha=100, logy=True)
print("ridge log-target alpha=100: val MAE", round(mae,2), "(train", round(trm,2), ")")


# ---- cell ----
import numpy as np, pandas as pd, agent_api as A

df = A.load_saved("e005_marketing.parquet")
tt = A.train_targets()
m = tt.merge(df, on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"].astype(float).values

tr_days = A.snapshot_days()["train"]
print("train rows with NaN y:", int(np.isnan(y).sum()))
PV = 431
is_pv = m["snapshot_day"]==PV
is_fit = m["snapshot_day"].isin([d for d in tr_days if d<PV])
print("fit rows", int(is_fit.sum()), "pv rows", int(is_pv.sum()))

# oracle: household median from fit snaps -> predict pv
med = m[is_fit].groupby("household_key")["future_spend_4w"].median()
pred_or = m[is_pv]["household_key"].map(med).fillna(m[is_fit]["future_spend_4w"].median()).values
print("oracle hh-median pv MAE:", round(float(np.abs(m[is_pv]["future_spend_4w"].values-pred_or).mean()),2))

feat_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
X = m[feat_cols].copy()
for c in X.columns:
    if not np.issubdtype(X[c].dtype, np.number):
        X[c] = X[c].astype("category").cat.codes
X = X.replace([np.inf,-np.inf], np.nan).values.astype(float)

def ridge_eval(X, y, is_fit, is_pv, alpha=100.0, logy=False):
    Xf, Xv = X[is_fit], X[is_pv]
    mu = np.nanmean(Xf,0); sd = np.nanstd(Xf,0)+1e-9
    Zf = np.where(np.isnan(Xf),(Xf-mu),Xf)/sd; Zf=np.nan_to_num(Zf)
    Zv = np.where(np.isnan(Xv),(Xv-mu),Xv)/sd; Zv=np.nan_to_num(Zv)
    yt = np.log1p(y) if logy else y
    d=Zf.shape[1]; G=Zf.T@Zf+alpha*np.eye(d); w=np.linalg.solve(G, Zf.T@yt[is_fit])
    p = Zv@w
    if logy: p=np.expm1(np.clip(p,0,8))
    return float(np.abs(y[is_pv]-p).mean())

for a in [10,100,1000,3000]:
    print(f"ridge a={a}: pv MAE {ridge_eval(X,y,is_fit,is_pv,a):.2f}")
print("ridge logy a=1000:", round(ridge_eval(X,y,is_fit,is_pv,1000,logy=True),2))

# single feature spend_112_mean4 on pv
f = m["spend_112_mean4"].values
p = np.nan_to_num(f)
print("spend_112_mean4 raw pv MAE:", round(float(np.abs(y[is_pv]-p[is_pv]).mean()),2))
al=np.linspace(0,2,41); maes=[np.abs(y[is_pv]-a*p[is_pv]).mean() for a in al]
i=int(np.argmin(maes)); print("scaled best alpha",round(float(al[i]),2),"MAE",round(float(maes[i]),2))


# ---- cell ----
import numpy as np, pandas as pd, agent_api as A

df = A.load_saved("e005_marketing.parquet")
tt = A.train_targets()
m = tt.merge(df, on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"].astype(float).values
tr_days = A.snapshot_days()["train"]; PV=431
is_pv = m["snapshot_day"]==PV
is_fit = m["snapshot_day"].isin([d for d in tr_days if d<PV])

feat_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
X = m[feat_cols].copy()
for c in X.columns:
    if not np.issubdtype(X[c].dtype, np.number): X[c]=X[c].astype("category").cat.codes
X = X.replace([np.inf,-np.inf],np.nan).values.astype(float)

def ridge_pred(X,y,is_fit,is_pv,alpha=3000):
    Xf,Xv=X[is_fit],X[is_pv]
    mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0)+1e-9
    Zf=np.nan_to_num(np.where(np.isnan(Xf),mu,Xf)/sd); Zv=np.nan_to_num(np.where(np.isnan(Xv),mu,Xv)/sd)
    G=Zf.T@Zf+alpha*np.eye(Zf.shape[1]); w=np.linalg.solve(G,Zf.T@y[is_fit])
    return Zv@w, Zf@w

p_pv, p_fit = ridge_pred(X,y,is_fit,is_pv,3000)
yv = y[is_pv]
mae0 = np.abs(yv-p_pv).mean()
print("proxy ridge MAE:", round(mae0,2))

# error by predicted-spend decile
q = pd.qcut(p_fit, 10, duplicates="drop")
err_fit = pd.DataFrame({"q":q,"y":y[is_fit],"p":p_fit})
print("\nfit-set: mean |err| and mean bias by predicted decile:")
g = err_fit.groupby("q",observed=True).apply(lambda d: pd.Series({"mae":np.abs(d.y-d.p).mean(),"bias":(d.p-d.y).mean(),"n":len(d),"ybar":d.y.mean(),"pbar":d.p.mean()}))
print(g.round(1))

# calibration: fit isotonic-like piecewise linear via binning predicted -> mean y on fit set
bins = pd.qcut(p_fit, 50, duplicates="drop")
cal = pd.DataFrame({"b":bins,"y":y[is_fit],"p":p_fit}).groupby("b",observed=True).agg(pmid=("p","mean"),ybar=("y","mean"))
pc = np.interp(p_pv, cal["pmid"].values, cal["ybar"].values)
print("\nbin-calibrated MAE:", round(float(np.abs(yv-pc).mean()),2))

# blend with household median feature
med_hh = m[is_fit].groupby("household_key")["future_spend_4w"].median()
f_med = m["household_key"].map(med_hh).fillna(m[is_fit]["future_spend_4w"].median()).values
for wgt in [0,0.25,0.5,0.75,1.0]:
    pb = (1-wgt)*p_pv + wgt*f_med[is_pv]
    print(f"blend ridge/{wgt} hhmed: pv MAE {np.abs(yv-pb).mean():.2f}")

# simple power scaling of predictions: p^gamma style (p -> a*p + b*p^2?) try p*clip
for g in [0.8,0.9,1.0,1.1,1.2]:
    ps = np.sign(p_pv)*np.abs(p_pv)**g
    print(f"power {g}: MAE {np.abs(yv-ps).mean():.2f}")


# ---- cell ----
import numpy as np, pandas as pd, agent_api as A

df = A.load_saved("e005_marketing.parquet")
df4 = A.load_saved("e004_seasonal.parquet")
tt = A.train_targets()
m = tt.merge(df, on=["household_key","snapshot_day"], how="left")
m = m.merge(df4[["household_key","snapshot_day"]+[c for c in df4.columns if "lag" in c]], on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"].astype(float).values
tr_days = A.snapshot_days()["train"]; PV=431
is_pv = m["snapshot_day"]==PV
is_fit = m["snapshot_day"].isin([d for d in tr_days if d<PV])
yv = y[is_pv]

lag_cols = [c for c in m.columns if c.startswith("spend_4w_lag")]
print("lag cols:", lag_cols)
L = m[lag_cols].replace([np.inf,-np.inf],np.nan)
n_avail = L.notna().sum(1)
med_lags = L.median(1)          # median of household's own prior 4w spends
mean_lags = L.mean(1)
max_lags = L.max(1)
last_lag = L[lag_cols[0]]       # most recent lag (t-28..t]

def ridge_eval(X, y, is_fit, is_pv, alpha=3000):
    Xf,Xv=X[is_fit],X[is_pv]
    mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0)+1e-9
    Zf=np.nan_to_num(np.where(np.isnan(Xf),mu,Xf)/sd); Zv=np.nan_to_num(np.where(np.isnan(Xv),mu,Xv)/sd)
    G=Zf.T@Zf+alpha*np.eye(Zf.shape[1]); w=np.linalg.solve(G,Zf.T@y[is_fit])
    return float(np.abs(y[is_pv]-Zv@w).mean()), Zv@w

base_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
Xb = m[base_cols].copy()
for c in Xb.columns:
    if not np.issubdtype(Xb.columns.dtype if False else Xb[c].dtype, np.number): Xb[c]=Xb[c].astype("category").cat.codes
Xb = Xb.replace([np.inf,-np.inf],np.nan).values.astype(float)

def test(name, extra):
    X = np.column_stack([Xb, extra]) if extra is not None else Xb
    mae,_ = ridge_eval(X, y, is_fit, is_pv, 3000)
    print(f"{name}: pv MAE {mae:.2f}")

test("E005 base", None)
test("+med_lags", med_lags.values.reshape(-1,1))
test("+mean_lags", mean_lags.values.reshape(-1,1))
test("+med+mean+max+last", np.column_stack([med_lags,mean_lags,max_lags,last_lag]))
test("+med+last+n_avail", np.column_stack([med_lags,last_lag,n_avail]))

# log transforms of all spend-ish columns
spendish = [c for c in base_cols if ("spend" in c or c.startswith("sum_")) ]
Xlog = Xb.copy()
idx = [base_cols.index(c) for c in spendish]
Xlog[:, idx] = np.log1p(np.clip(Xlog[:, idx], 0, None))
test("+log(spend cols)", Xlog-Xb if False else np.zeros((len(Xlog),0)))  # replace base
def test2(name, X):
    mae,_ = ridge_eval(X, y, is_fit, is_pv, 3000)
    print(f"{name}: pv MAE {mae:.2f}")
test2("base with log-spend", Xlog)
Xlog2 = np.column_stack([Xlog, med_lags, last_lag])
test2("log-spend + med_lags + last_lag", Xlog2)

# ratios: recent vs baseline
r1 = (m["spend_4w"]/m["spend_28w"].replace(0,np.nan)).fillna(1).values.reshape(-1,1)
r2 = (m["spend_4w"]/m["spend_112w"].replace(0,np.nan)).fillna(1).values.reshape(-1,1)
test2("log-spend + med/last lags + ratios", np.column_stack([Xlog2, r1, r2]))


# ---- cell ----
import numpy as np, pandas as pd, agent_api as A

df = A.load_saved("e005_marketing.parquet")
tt = A.train_targets()
m = tt.merge(df, on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"].astype(float).values
tr_days = A.snapshot_days()["train"]; PV=431
is_pv = m["snapshot_day"]==PV
is_fit = m["snapshot_day"].isin([d for d in tr_days if d<PV])
yv = y[is_pv]

# Build aligned-window 4w spend history per (hh, snapshot) from transactions up to snapshot day.
# For snapshot t and household h: w_k = spend in [t-28k+1, t-28k], k=1..13 (aligned with target windows of earlier snapshots)
# Use view at each snapshot via build_features-like loop: cheaper — compute from history capped per snapshot.
# We can't call build_features here cheaply for all snaps? We can: it calls fn per snapshot. Do it once.
def fn(view, snapshot_day):
    tx = view.table("transactions")
    hhs = view.households
    tx = tx[tx.household_key.isin(set(hhs))]
    g = tx.groupby(["household_key","day"])["sales_value"].sum().reset_index()
    out = {}
    for k in range(1,14):
        lo, hi = snapshot_day-28*k+1, snapshot_day-28*k
        w = g[(g.day>=lo)&(g.day<=hi)].groupby("household_key")["sales_value"].sum()
        out[f"aw_{k}"] = w
    R = pd.DataFrame(index=pd.Index(hhs, name="household_key"))
    for k,v in out.items(): R[f"aw_{k}"] = v.reindex(R.index).fillna(0.0)
    return R

aw = A.build_features(fn)
print("aw shape", aw.shape)

K = aw[["household_key","snapshot_day"]].merge(aw.drop(columns=["household_key","snapshot_day"]), left_index=True, right_index=True)
m = m.merge(aw.reset_index(), on=["household_key","snapshot_day"], how="left")
awc = [f"aw_{k}" for k in range(1,14)]
A_ = m[awc]
n_avail = A_.notna().sum(1)

def ridge_eval(X, y, is_fit, is_pv, alpha=3000):
    Xf,Xv=X[is_fit],X[is_pv]
    mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0)+1e-9
    Zf=np.nan_to_num(np.where(np.isnan(Xf),mu,Xf)/sd); Zv=np.nan_to_num(np.where(np.isnan(Xv),mu,Xv)/sd)
    G=Zf.T@Zf+alpha*np.eye(Zf.shape[1]); w=np.linalg.solve(G,Zf.T@y[is_fit])
    return float(np.abs(y[is_pv]-Zv@w).mean())

base_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
Xb = m[base_cols].copy()
for c in Xb.columns:
    if not np.issubdtype(Xb[c].dtype, np.number): Xb[c]=Xb[c].astype("category").cat.codes
Xb = Xb.replace([np.inf,-np.inf],np.nan).values.astype(float)

print("base:", round(ridge_eval(Xb,y,is_fit,is_pv),2))
for Kk in [3,5,8,13]:
    sub = m[[f"aw_{k}" for k in range(1,Kk+1)]]
    f_med = sub.median(1).values.reshape(-1,1)
    f_mean = sub.mean(1).values.reshape(-1,1)
    print(f"+aw_med(K={Kk}):", round(ridge_eval(np.column_stack([Xb,f_med]),y,is_fit,is_pv),2),
          f"| +aw_mean(K={Kk}):", round(ridge_eval(np.column_stack([Xb,f_mean]),y,is_fit,is_pv),2),
          f"| both:", round(ridge_eval(np.column_stack([Xb,f_med,f_mean]),y,is_fit,is_pv),2))

# raw single-feature check on pv
f_med13 = m[[f"aw_{k}" for k in range(1,14)]].median(1).values
print("aw_med13 raw pv MAE:", round(float(np.abs(yv-np.nan_to_num(f_med13)[is_pv]).mean()),2))
f_mean13 = m[[f"aw_{k}" for k in range(1,14)]].mean(1).values
print("aw_mean13 raw pv MAE:", round(float(np.abs(yv-np.nan_to_num(f_mean13)[is_pv]).mean()),2))


# ---- cell ----
import numpy as np, pandas as pd, agent_api as A
aw = A.load_saved("aw_hist.parquet") if False else None
# rebuild quickly (cheap) and inspect structure
def fn(view, snapshot_day):
    tx = view.table("transactions")
    hhs = view.households
    g = tx[tx.household_key.isin(set(hhs))].groupby(["household_key","day"])["sales_value"].sum().reset_index()
    R = pd.DataFrame(index=pd.Index(hhs, name="household_key"))
    for k in range(1,14):
        lo, hi = snapshot_day-28*k+1, snapshot_day-28*k
        w = g[(g.day>=lo)&(g.day<=hi)].groupby("household_key")["sales_value"].sum()
        R[f"aw_{k}"] = w.reindex(R.index).fillna(0.0)
    return R
aw = A.build_features(fn)
print(aw.columns.tolist()[:6], aw.shape)
print(aw.head(3))


# ---- cell ----
import numpy as np, pandas as pd, agent_api as A
aw = A.load_saved("aw_hist.parquet")
df = A.load_saved("e005_marketing.parquet")
tt = A.train_targets()
m = tt.merge(df, on=["household_key","snapshot_day"], how="left").merge(aw, on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"].astype(float).values
tr_days = A.snapshot_days()["train"]; PV=431
is_pv = m["snapshot_day"]==PV
is_fit = m["snapshot_day"].isin([d for d in tr_days if d<PV])
yv = y[is_pv]
awc = [f"aw_{k}" for k in range(1,14)]
A_ = m[awc]; n_avail = A_.notna().sum(1)

def ridge_eval(X, y, is_fit, is_pv, alpha=3000):
    Xf,Xv=X[is_fit],X[is_pv]
    mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0)+1e-9
    Zf=np.nan_to_num(np.where(np.isnan(Xf),mu,Xf)/sd); Zv=np.nan_to_num(np.where(np.isnan(Xv),mu,Xv)/sd)
    G=Zf.T@Zf+alpha*np.eye(Zf.shape[1]); w=np.linalg.solve(G,Zf.T@y[is_fit])
    return float(np.abs(y[is_pv]-Zv@w).mean())

base_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
Xb = m[base_cols].copy()
for c in Xb.columns:
    if not np.issubdtype(Xb[c].dtype, np.number): Xb[c]=Xb[c].astype("category").cat.codes
Xb = Xb.replace([np.inf,-np.inf],np.nan).values.astype(float)

print("base:", round(ridge_eval(Xb,y,is_fit,is_pv),2))
for Kk in [3,5,8,13]:
    sub = m[[f"aw_{k}" for k in range(1,Kk+1)]]
    f_med = sub.median(1).values.reshape(-1,1); f_mean = sub.mean(1).values.reshape(-1,1)
    print(f"K={Kk}: med {ridge_eval(np.column_stack([Xb,f_med]),y,is_fit,is_pv):.2f} | mean {ridge_eval(np.column_stack([Xb,f_mean]),y,is_fit,is_pv):.2f} | both {ridge_eval(np.column_stack([Xb,f_med,f_mean]),y,is_fit,is_pv):.2f}")
print("aw_med13 raw pv MAE:", round(float(np.abs(yv-np.nan_to_num(m[awc].median(1).values)[is_pv]).mean()),2))
print("aw_mean13 raw pv MAE:", round(float(np.abs(yv-np.nan_to_num(m[awc].mean(1).values)[is_pv]).mean()),2))
# weighted: recent lags weighted more
wts = np.array([1/k for k in range(1,14)])
f_wm = (m[awc]*wts).sum(1)/ (m[awc].notna()*wts).sum(1).values
print("aw_wmean raw pv MAE:", round(float(np.abs(yv-np.nan_to_num(f_wm.values)[is_pv]).mean()),2))


# ---- cell ----
import numpy as np, pandas as pd, agent_api as A
def fn(view, snapshot_day):
    tx = view.table("transactions")
    hhs = view.households
    g = tx[tx.household_key.isin(set(hhs))].groupby(["household_key","day"])["sales_value"].sum().reset_index()
    R = pd.DataFrame(index=pd.Index(hhs, name="household_key"))
    for k in range(1,14):
        lo, hi = snapshot_day-28*k+1, snapshot_day-28*k
        w = g[(g.day>=lo)&(g.day<=hi)].groupby("household_key")["sales_value"].sum()
        R[f"aw_{k}"] = w.reindex(R.index).fillna(0.0)
    return R
aw = A.build_features(fn)
path = A.save_table(aw, "aw_hist")
print(path)

df = A.load_saved("e005_marketing.parquet")
tt = A.train_targets()
m = tt.merge(df, on=["household_key","snapshot_day"], how="left").merge(aw, on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"].astype(float).values
tr_days = A.snapshot_days()["train"]; PV=431
is_pv = m["snapshot_day"]==PV
is_fit = m["snapshot_day"].isin([d for d in tr_days if d<PV])
yv = y[is_pv]
awc = [f"aw_{k}" for k in range(1,14)]

def ridge_eval(X, y, is_fit, is_pv, alpha=3000):
    Xf,Xv=X[is_fit],X[is_pv]
    mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0)+1e-9
    Zf=np.nan_to_num(np.where(np.isnan(Xf),mu,Xf)/sd); Zv=np.nan_to_num(np.where(np.isnan(Xv),mu,Xv)/sd)
    G=Zf.T@Zf+alpha*np.eye(Zf.shape[1]); w=np.linalg.solve(G,Zf.T@y[is_fit])
    return float(np.abs(y[is_pv]-Zv@w).mean())

base_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
Xb = m[base_cols].copy()
for c in Xb.columns:
    if not np.issubdtype(Xb[c].dtype, np.number): Xb[c]=Xb[c].astype("category").cat.codes
Xb = Xb.replace([np.inf,-np.inf],np.nan).values.astype(float)

print("base:", round(ridge_eval(Xb,y,is_fit,is_pv),2))
for Kk in [3,5,8,13]:
    sub = m[[f"aw_{k}" for k in range(1,Kk+1)]]
    f_med = sub.median(1).values.reshape(-1,1); f_mean = sub.mean(1).values.reshape(-1,1)
    print(f"K={Kk}: med {ridge_eval(np.column_stack([Xb,f_med]),y,is_fit,is_pv):.2f} | mean {ridge_eval(np.column_stack([Xb,f_mean]),y,is_fit,is_pv):.2f} | both {ridge_eval(np.column_stack([Xb,f_med,f_mean]),y,is_fit,is_pv):.2f}")
print("aw_med13 raw pv MAE:", round(float(np.abs(yv-np.nan_to_num(m[awc].median(1).values)[is_pv]).mean()),2))
print("aw_mean13 raw pv MAE:", round(float(np.abs(yv-np.nan_to_num(m[awc].mean(1).values)[is_pv]).mean()),2))
wts = np.array([1/k for k in range(1,14)])
f_wm = (m[awc]*wts).sum(1)/ (m[awc].notna()*wts).sum(1).replace(0,np.nan).values
print("aw_wmean raw pv MAE:", round(float(np.abs(yv-np.nan_to_num(f_wm.values)[is_pv]).mean()),2))


# ---- cell ----
import numpy as np, pandas as pd, agent_api as A
def fn(view, snapshot_day):
    tx = view.table("transactions")
    hhs = view.households
    print("snap", snapshot_day, "n hhs", len(hhs), "tx rows", len(tx))
    print("hhs type/sample:", type(hhs[0]) if len(hhs) else None, hhs[:2])
    print("tx hh dtype:", tx.household_key.dtype, "sample:", tx.household_key.iloc[:2].tolist() if len(tx) else None)
    print("tx day max:", tx.day.max() if len(tx) else None)
    sub = tx[tx.household_key.isin(set(hhs))]
    print("after isin:", len(sub))
    hh0 = hhs[0]
    print("rows for hh0:", len(tx[tx.household_key==hh0]))
    return pd.DataFrame(index=pd.Index(hhs, name="household_key"), data={"x":1.0})
aw = A.build_features(fn)


# ---- cell ----
import numpy as np, pandas as pd, agent_api as A
def fn(view, snapshot_day):
    tx = view.table("transactions")
    hhs = view.households
    g = tx[tx.household_key.isin(set(hhs))].groupby(["household_key","day"])["sales_value"].sum().reset_index()
    R = pd.DataFrame(index=pd.Index(hhs, name="household_key"))
    for k in range(1,14):
        lo, hi = snapshot_day-28*k+1, snapshot_day-28*(k-1)
        w = g[(g.day>=lo)&(g.day<=hi)].groupby("household_key")["sales_value"].sum()
        R[f"aw_{k}"] = w.reindex(R.index).fillna(0.0)
    return R
aw = A.build_features(fn)
A.save_table(aw, "aw_hist")
m0 = A.train_targets().merge(A.load_saved("e005_marketing.parquet"), on=["household_key","snapshot_day"], how="left").merge(aw, on=["household_key","snapshot_day"], how="left")
y = m0["future_spend_4w"].astype(float).values
tr_days = A.snapshot_days()["train"]; PV=431
is_pv = m0["snapshot_day"]==PV
is_fit = m0["snapshot_day"].isin([d for d in tr_days if d<PV])
yv = y[is_pv]
awc = [f"aw_{k}" for k in range(1,14)]
print("aw_1==spend_4w_recent?", np.allclose(m0["aw_1"].fillna(0), m0["spend_4w_recent"].fillna(0)))
print("aw_2==spend_4w_lag1?", np.allclose(m0["aw_2"].fillna(0), m0["spend_4w_lag1"].fillna(0)))
print("aw_med13 raw pv MAE:", round(float(np.abs(yv-np.nan_to_num(m0[awc].median(1).values)[is_pv]).mean()),2))
print("aw_mean13 raw pv MAE:", round(float(np.abs(yv-np.nan_to_num(m0[awc].mean(1).values)[is_pv]).mean()),2))

def ridge_eval(X, y, is_fit, is_pv, alpha=3000):
    Xf,Xv=X[is_fit],X[is_pv]
    mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0)+1e-9
    Zf=np.nan_to_num(np.where(np.isnan(Xf),mu,Xf)/sd); Zv=np.nan_to_num(np.where(np.isnan(Xv),mu,Xv)/sd)
    G=Zf.T@Zf+alpha*np.eye(Zf.shape[1]); w=np.linalg.solve(G,Zf.T@y[is_fit])
    return float(np.abs(y[is_pv]-Zv@w).mean())
base_cols = [c for c in A.load_saved("e005_marketing.parquet").columns if c not in ("household_key","snapshot_day")]
Xb = m0[base_cols].copy()
for c in Xb.columns:
    if not np.issubdtype(Xb[c].dtype, np.number): Xb[c]=Xb[c].astype("category").cat.codes
Xb = Xb.replace([np.inf,-np.inf],np.nan).values.astype(float)
print("base:", round(ridge_eval(Xb,y,is_fit,is_pv),2))
for Kk in [3,5,8,13]:
    sub = m0[[f"aw_{k}" for k in range(1,Kk+1)]]
    f_med = sub.median(1).values.reshape(-1,1); f_mean = sub.mean(1).values.reshape(-1,1)
    print(f"K={Kk}: med {ridge_eval(np.column_stack([Xb,f_med]),y,is_fit,is_pv):.2f} | mean {ridge_eval(np.column_stack([Xb,f_mean]),y,is_fit,is_pv):.2f} | both {ridge_eval(np.column_stack([Xb,f_med,f_mean]),y,is_fit,is_pv):.2f}")


# ---- cell ----
import numpy as np, pandas as pd, agent_api as A, warnings
warnings.filterwarnings("ignore")
aw = A.load_saved("aw_hist.parquet")
df = A.load_saved("e005_marketing.parquet")
m = A.train_targets().merge(df, on=["household_key","snapshot_day"], how="left").merge(aw, on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"].astype(float).values
tr_days = A.snapshot_days()["train"]; PV=431
is_pv = m["snapshot_day"]==PV
is_fit = m["snapshot_day"].isin([d for d in tr_days if d<PV])
yv = y[is_pv]
awc = [f"aw_{k}" for k in range(1,14)]
base_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
Xd = m[base_cols].copy()
for c in Xd.columns:
    if not np.issubdtype(Xd[c].dtype, np.number): Xd[c]=Xd[c].astype("category").cat.codes
Xd = Xd.replace([np.inf,-np.inf],np.nan)
Xb = Xd.values.astype(float)

def ridge_eval(X, alpha=3000):
    Xf,Xv=X[is_fit],X[is_pv]
    mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0)+1e-9
    Zf=np.nan_to_num(np.where(np.isnan(Xf),mu,Xf)/sd); Zv=np.nan_to_num(np.where(np.isnan(Xv),mu,Xv)/sd)
    G=Zf.T@Zf+alpha*np.eye(Zf.shape[1]); w=np.linalg.solve(G,Zf.T@y[is_fit])
    return float(np.abs(yv-Zv@w).mean())

print("base:", round(ridge_eval(Xb),2))
f_med8 = m[awc[:8]].median(1).fillna(0).values.reshape(-1,1)
f_mean8 = m[awc[:8]].mean(1).fillna(0).values.reshape(-1,1)
X1 = np.column_stack([Xb, f_med8, f_mean8])
print("base+aw8med/mean:", round(ridge_eval(X1),2))

# rank transform of all features (tree-like invariance) as EXTRA cols for top features only
def rank_col(v):
    r = pd.Series(v).rank(pct=True).values
    return (r-0.5).reshape(-1,1)
top = ["spend_112_mean4","spend_28w","spend_4w","spend_56w","spend_per_day_total","spend_112_max4","spend_112_min4","spend_4w_lag1","nbask_28w","spend_112_std"]
Xr = np.column_stack([X1] + [rank_col(m[c].fillna(0).values) for c in top])
print("base+aw+rank(top10):", round(ridge_eval(Xr),2))

# squared top feature (convexity)
s = m["spend_112_mean4"].fillna(0).values
Xsq = np.column_stack([X1, (s**2/1e4).reshape(-1,1), rank_col(s)])
print("base+aw+sq+rank(top1):", round(ridge_eval(Xsq),2))

# log1p copies of key spend cols
key = ["spend_4w","spend_8w","spend_12w","spend_28w","spend_56w","spend_112w","spend_total","nbask_4w","nbask_8w","nbask_28w","nbask_56w","nbask_112w","spend_per_day_total","spend_per_week_112","spend_112_mean4","spend_112_max4","spend_112_min4","spend_112_std","spend_4w_lag1","spend_4w_lag2","spend_4w_lag3"]
Xlog = np.column_stack([X1] + [np.log1p(np.clip(m[c].fillna(0).values,0,None)).reshape(-1,1) for c in key])
print("base+aw+log1p(21 keys):", round(ridge_eval(Xlog),2))

# combined best guess
Xall = np.column_stack([Xlog] + [rank_col(m[c].fillna(0).values) for c in top[:6]])
print("base+aw+log+rank6:", round(ridge_eval(Xall),2))
print("n cols:", Xall.shape[1])


# ---- cell ----
import numpy as np, pandas as pd, agent_api as A, warnings
warnings.filterwarnings("ignore")
aw = A.load_saved("aw_hist.parquet")
df = A.load_saved("e005_marketing.parquet")
m = A.train_targets().merge(df, on=["household_key","snapshot_day"], how="left").merge(aw, on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"].astype(float).values
tr_days = A.snapshot_days()["train"]; PV=431
is_pv = m["snapshot_day"]==PV
is_fit = m["snapshot_day"].isin([d for d in tr_days if d<PV])
yv = y[is_pv]
awc = [f"aw_{k}" for k in range(1,14)]
base_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
Xd = m[base_cols].copy()
for c in Xd.columns:
    if not np.issubdtype(Xd[c].dtype, np.number): Xd[c]=Xd[c].astype("category").cat.codes
Xd = Xd.replace([np.inf,-np.inf],np.nan)
Xb = Xd.values.astype(float)
def ridge_eval(X, alpha=3000):
    Xf,Xv=X[is_fit],X[is_pv]
    mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0)+1e-9
    Zf=np.nan_to_num(np.where(np.isnan(Xf),mu,Xf)/sd); Zv=np.nan_to_num(np.where(np.isnan(Xv),mu,Xv)/sd)
    G=Zf.T@Zf+alpha*np.eye(Zf.shape[1]); w=np.linalg.solve(G,Zf.T@y[is_fit])
    return float(np.abs(yv-Zv@w).mean())
print("base:", round(ridge_eval(Xb),2))

# drop day/week/snap_day/snap_week/index (snapshot-calendar cols) from base
drop = ["day","week","snap_day","snap_week","index"]
keep = [c for c in base_cols if c not in drop]
Xk = m[keep].copy()
for c in Xk.columns:
    if not np.issubdtype(Xk[c].dtype, np.number): Xk[c]=Xk[c].astype("category").cat.codes
Xk = Xk.replace([np.inf,-np.inf],np.nan).values.astype(float)
print("base w/o calendar cols:", round(ridge_eval(Xk),2))
# calendar only
Xc = m[drop].replace([np.inf,-np.inf],np.nan).values.astype(float)
print("calendar only:", round(ridge_eval(Xc),2))

f_med8 = m[awc[:8]].median(1).fillna(0).values.reshape(-1,1)
f_mean8 = m[awc[:8]].mean(1).fillna(0).values.reshape(-1,1)
X1 = np.column_stack([Xk, f_med8, f_mean8])
print("noCal+aw8:", round(ridge_eval(X1),2))
key = ["spend_4w","spend_8w","spend_12w","spend_28w","spend_56w","spend_112w","spend_total","nbask_4w","nbask_8w","nbask_28w","nbask_56w","nbask_112w","spend_per_day_total","spend_per_week_112","spend_112_mean4","spend_112_max4","spend_112_min4","spend_112_std","spend_4w_lag1","spend_4w_lag2","spend_4w_lag3"]
Xlog = np.column_stack([X1] + [np.log1p(np.clip(m[c].fillna(0).values,0,None)).reshape(-1,1) for c in key])
print("noCal+aw8+log1p:", round(ridge_eval(Xlog),2))


# ---- cell ----
import numpy as np, pandas as pd, agent_api as A, warnings
warnings.filterwarnings("ignore")
df = A.load_saved("e005_marketing.parquet")
aw = A.load_saved("aw_hist.parquet")
m = df.merge(aw, on=["household_key","snapshot_day"], how="left")
awc = [f"aw_{k}" for k in range(1,9)]
m["aw_med8"] = m[awc].median(1)
m["aw_mean8"] = m[awc].mean(1)
key = ["spend_4w","spend_8w","spend_12w","spend_28w","spend_56w","spend_112w","spend_total","nbask_4w","nbask_8w","nbask_28w","nbask_56w","nbask_112w","spend_per_day_total","spend_per_week_112","spend_112_mean4","spend_112_max4","spend_112_min4","spend_112_std","spend_4w_lag1","spend_4w_lag2","spend_4w_lag3"]
for c in key:
    m["lg_"+c] = np.log1p(np.clip(m[c].fillna(0).values,0,None))
drop = ["day","week","snap_day","snap_week","index"]
out = m.drop(columns=[c for c in drop if c in m.columns])
print("cols:", out.shape[1], "rows:", len(out))
print("null rows in key cols:", int(out[["aw_med8","aw_mean8"]].isna().any(1).sum()))
path = A.save_table(out, "e006_awagg")
print(path)
