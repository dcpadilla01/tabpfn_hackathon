import agent_api as A
import pandas as pd, numpy as np

names = ["rfm28","rich_behavioral","season","macro","mkt_demo","e006_composition","e007_lagseq",
         "e010_decay","e011_price","e009_basket","dm_exp","e016_peer","e013_te_clean","cand1","e014_dm","e012_xenc"]
for n in names:
    t = A.load_saved(n + ".parquet")
    cols = list(t.columns)
    print(f"== {n} {t.shape}")
    print("   " + ", ".join(cols)[:1400])

print("\nsnapshot_days:", A.snapshot_days())
base = A.baseline_features()
print("baseline rows:", len(base))

tt = A.train_targets()
print("targets:", tt.shape)
print(tt.groupby("snapshot_day")["future_spend_4w"].agg(["count","mean","median"]).round(1))
print(tt["future_spend_4w"].describe().round(2))
print("zero frac:", round(float((tt.future_spend_4w==0).mean()),4))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

base_cols = None
for n in ["e006_composition","e007_lagseq","e009_basket","e010_decay","e011_price","dm_exp","e016_peer","e013_te_clean","cand1","e014_dm"]:
    t = A.load_saved(n + ".parquet")
    cols = list(t.columns)
    if base_cols is None:
        base_cols = list(A.load_saved("rfm28.parquet").columns)
    extra = [c for c in cols if c not in base_cols]
    print(f"== {n} {t.shape}  extra({len(extra)}):")
    print("   " + ", ".join(extra))
    print()


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

e011 = A.load_saved("e011_price.parquet")
base = list(A.load_saved("rfm28.parquet").columns)
e011_feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
print("E011 n features:", len(e011_feats))

tt = A.train_targets()
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("merged:", df.shape)

y = df["future_spend_4w"].values
ly = np.log1p(y)

num = df[e011_feats].select_dtypes(include=[np.number,bool]).columns
corr = {}
for c in num:
    x = pd.to_numeric(df[c], errors="coerce").values
    ok = ~np.isnan(x)
    if ok.sum() < 100: 
        corr[c] = (np.nan, np.nan); continue
    corr[c] = (np.corrcoef(x[ok], y[ok])[0,1], np.corrcoef(x[ok], ly[ok])[0,1])
c_df = pd.DataFrame(corr, index=["r_y","r_ly"]).T.sort_values("r_y", ascending=False)
print("\nTOP 25 by |corr with y|:")
print(c_df.reindex(c_df.r_y.abs().sort_values(ascending=False).index).head(25).round(3))
print("\nBOTTOM 25 (weakest):")
print(c_df.reindex(c_df.r_y.abs().sort_values().index).head(25).round(3))

# concavity test: linear on raw vs log spend28
def mae(pred, yy): return np.mean(np.abs(pred-yy))
x = pd.to_numeric(df["spend28"], errors="coerce").fillna(0).values
X1 = np.column_stack([np.ones_like(x), x])
b = np.linalg.lstsq(X1, y, rcond=None)[0]; p = X1@b
X2 = np.column_stack([np.ones_like(x), np.log1p(x)])
b2 = np.linalg.lstsq(X2, y, rcond=None)[0]; p2 = X2@b2
X3 = np.column_stack([np.ones_like(x), x, np.log1p(x), np.sqrt(x)])
b3 = np.linalg.lstsq(X3, y, rcond=None)[0]; p3 = X3@b3
print("\nunivariate spend28: MAE lin=%.2f log=%.2f both=%.2f | R2 lin=%.3f log=%.3f" %
      (mae(p,y), mae(p2,y), mae(p3,y), 1-((y-p)**2).sum()/((y-y.mean())**2).sum(), 1-((y-p2)**2).sum()/((y-y.mean())**2).sum()))

# split-half validation of concavity: fit on train snapshot days <=431 (all are train here)
# check per-snapshot: is relationship concave? bin spend28 and show mean y
df["b"] = pd.qcut(df["spend28"].fillna(0), 12, duplicates="drop")
print("\nmean y by spend28 decile:")
print(df.groupby("b", observed=True)["future_spend_4w"].agg(["mean","count"]).round(1))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

e011 = A.load_saved("e011_price.parquet")
tt = A.train_targets()
feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")

# groups
def grp(pre): return [c for c in feats if c.startswith(pre)]
dsp = grp("dsp_"); ap = grp("ap_"); pt = grp("price_trend"); qty = grp("qty_"); sp = grp("spend_p")
iv = grp("iv_")+grp("bk_")+["n_gaps112"]
lag = grp("lag_")+["dec_spend14","dec_spend7","dec_trips","gap_mean","gap_std","gap_med","ntrip112"]
mkt = grp("tA_")+grp("tB_")+grp("tC_")+grp("red_")+["n_tgt_ever","n_tgt_active","targeted_flag","days_since_first_tgt","days_since_last_tgt","tgt_active_remaining","cpn_prod_spend28","cpn_prod_share28"]
demo = grp("demo_")+["has_demo"]
comp_extra = ["ndept28","spend_w0","spend_w1","spend_w2","spend_w3","share_w0","rwspend84","active_weeks112","weekend_share28","zero28","ratio28_364","toptrip_share28","spend28_log","spend56_log","spend112_log","lt_spend_log","nbaskets28"]
core = [c for c in feats if c not in dsp+ap+pt+qty+sp+iv+lag+mkt+demo+comp_extra+["pl_share","disc_share2","c_qty112","macro_ratio","mom_28_112","mom_112_364","upt28","upt112","maxtrip_qty28","share_qty_toptrip"]]
core = [c for c in core if not c.startswith("dm_")]
print("core(%d):"%len(core), core)
print("\ncounts: dsp%d ap%d pt%d qty%d sp%d iv%d lag%d mkt%d demo%d comp%d"%(len(dsp),len(ap),len(pt),len(qty),len(sp),len(iv),len(lag),len(mkt),len(demo),len(comp_extra)))

def ridge_eval(cols, fit_days, eval_days, lam=1.0):
    d = df[df.snapshot_day.isin(fit_days+eval_days)]
    X = d[cols].apply(pd.to_numeric, errors="coerce")
    mu, sd = X.mean(), X.std().replace(0,1)
    Xs = ((X-mu)/sd).fillna(0).values
    y = d["future_spend_4w"].values
    d_tr = d.snapshot_day.isin(fit_days).values
    b = np.linalg.solve(Xs[d_tr].T@Xs[d_tr]+lam*np.eye(len(cols)), Xs[d_tr].T@y[d_tr])
    p = Xs@b
    m = ~d_tr
    return np.mean(np.abs(p[m]-y[m])), 1-((y[m]-p[m])**2).sum()/((y[m]-y[d_tr].mean())**2).sum()

tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
print("\nGROUP DIAG (ridge, fit<=403, eval 431):")
for name, cols in [("ALL228",feats),("core",core),("core+lag",core+lag),("core+lag+decay_in_lag",core+lag),
                   ("core+lag+comp",core+lag+comp_extra),("core+lag+comp+mkt",core+lag+comp_extra+mkt),
                   ("core+lag+comp+mkt+dsp",core+lag+comp_extra+mkt+dsp),
                   ("core+lag+comp+mkt+price(qty,sp_p,pt)",core+lag+comp_extra+mkt+qty+sp+pt),
                   ("core+lag+comp+mkt+ap+iv+misc",core+lag+comp_extra+mkt+ap+iv+["pl_share","disc_share2","c_qty112","macro_ratio","mom_28_112","mom_112_364","upt28","upt112","maxtrip_qty28","share_qty_toptrip"]),
                   ("core+lag+demo",core+lag+demo)]:
    mae, r2 = ridge_eval(cols, tr_days, [431])
    print(f"  {name:38s} n={len(cols):3d}  MAE={mae:7.3f}  R2={r2:.4f}")

# dsp correlations to pick top
r = {}
for c in dsp:
    x = pd.to_numeric(df[c], errors="coerce").fillna(0).values
    r[c] = abs(np.corrcoef(x, df["future_spend_4w"].values)[0,1])
print("\ntop dsp:", sorted(r.items(), key=lambda kv:-kv[1])[:8])

# engineered features check
sp28 = pd.to_numeric(df["spend28"],errors="coerce").fillna(0)
sp56 = pd.to_numeric(df["spend56"],errors="coerce").fillna(0)
sp112 = pd.to_numeric(df["spend112"],errors="coerce").fillna(0)
rec = pd.to_numeric(df["recency"],errors="coerce").fillna(999)
tr28 = pd.to_numeric(df["trips28"],errors="coerce").fillna(0)
cons28 = sp28; cons56 = sp56/2; cons112 = sp112/4
w = np.clip(tr28/4.0, 0, 1)
blend = w*cons28 + (1-w)*cons112
dorm = cons112*np.exp(-rec/21.0)
y = df["future_spend_4w"].values
print("\neng feats corr with y: blend=%.3f dorm=%.3f cons112=%.3f (spend112=%.3f)"%(
    np.corrcoef(blend,y)[0,1], np.corrcoef(dorm,y)[0,1], np.corrcoef(cons112,y)[0,1], np.corrcoef(sp112,y)[0,1]))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

e011 = A.load_saved("e011_price.parquet")
tt = A.train_targets()
feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")

# check dtypes / infs
bad = []
for c in feats:
    x = pd.to_numeric(df[c], errors="coerce")
    if x.isna().all(): bad.append((c,"all-nan"))
    elif np.isinf(x.values).any(): bad.append((c,"inf"))
    elif df[c].dtype == object: bad.append((c,"object"))
print("problem cols:", bad[:20], "..." if len(bad)>20 else "")

def ridge_eval(cols, fit_days, eval_days, lam=3.0, clip=10.0):
    d = df[df.snapshot_day.isin(fit_days+eval_days)]
    X = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0,1)
    Xs = ((X-mu)/sd).clip(-clip, clip).fillna(0).values
    y = d["future_spend_4w"].values
    d_tr = d.snapshot_day.isin(fit_days).values
    b = np.linalg.solve(Xs[d_tr].T@Xs[d_tr]+lam*np.eye(len(cols)), Xs[d_tr].T@y[d_tr])
    p = Xs@b
    m = ~d_tr
    return np.mean(np.abs(p[m]-y[m])), 1-((y[m]-p[m])**2).sum()/((y[m]-y[d_tr].mean())**2).sum()

def grp(pre): return [c for c in feats if c.startswith(pre)]
dsp = grp("dsp_"); ap = grp("ap_"); pt = grp("price_trend"); qty = grp("qty_"); sp = grp("spend_p")
iv = grp("iv_")+grp("bk_")+["n_gaps112"]
lag = grp("lag_")+["dec_spend14","dec_spend7","dec_trips","gap_mean","gap_std","gap_med","ntrip112"]
mkt = grp("tA_")+grp("tB_")+grp("tC_")+grp("red_")+["n_tgt_ever","n_tgt_active","targeted_flag","days_since_first_tgt","days_since_last_tgt","tgt_active_remaining","cpn_prod_spend28","cpn_prod_share28"]
demo = grp("demo_")+["has_demo"]
comp_extra = ["ndept28","spend_w0","spend_w1","spend_w2","spend_w3","share_w0","rwspend84","active_weeks112","weekend_share28","zero28","ratio28_364","toptrip_share28","spend28_log","spend56_log","spend112_log","lt_spend_log","nbaskets28"]
misc = ["pl_share","disc_share2","c_qty112","macro_ratio","mom_28_112","mom_112_364","upt28","upt112","maxtrip_qty28","share_qty_toptrip"]
core = [c for c in feats if c not in dsp+ap+pt+qty+sp+iv+lag+mkt+demo+comp_extra+misc and not c.startswith("dm_")]

tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
print("\nGROUP DIAG (ridge lam=3, fit<=403, eval 431):")
for name, cols in [("core64",core),("core+lag",core+lag),("core+lag+comp",core+lag+comp_extra),
                   ("core+lag+comp+mkt",core+lag+comp_extra+mkt),
                   ("core+lag+comp+mkt+dsp",core+lag+comp_extra+mkt+dsp),
                   ("core+lag+comp+mkt+price",core+lag+comp_extra+mkt+qty+sp+pt),
                   ("ALL228",feats)]:
    mae, r2 = ridge_eval(cols, tr_days, [431])
    print(f"  {name:26s} n={len(cols):3d}  MAE={mae:7.3f}  R2={r2:.4f}")

# single-snapshot eval is noisy; also do leave-one-snapshot-out style: eval on 403 too
print("\ncross-check eval on 403 (fit <=375):")
for name, cols in [("core64",core),("core+lag",core+lag),("ALL228",feats)]:
    mae, r2 = ridge_eval(cols, tr_days[:-1], [403])
    print(f"  {name:26s} n={len(cols):3d}  MAE={mae:7.3f}  R2={r2:.4f}")


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

e011 = A.load_saved("e011_price.parquet")
tt = A.train_targets()
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("merged rows:", len(df), "| dup rows:", df.duplicated(["household_key","snapshot_day"]).sum())

d431 = df[df.snapshot_day==431]
print("day431 rows:", len(d431), "mean y:", d431.future_spend_4w.mean().round(1))
print("spend112 corr with y on 431:", np.corrcoef(pd.to_numeric(d431.spend112,errors="coerce").fillna(0), d431.future_spend_4w)[0,1].round(3))

# minimal ridge: single feature spend112
def ridge_eval(cols, fit_days, eval_days, lam=3.0):
    d = df[df.snapshot_day.isin(fit_days+eval_days)]
    X = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0,1)
    Xs = ((X-mu)/sd).clip(-10,10).fillna(0).values
    y = d["future_spend_4w"].values
    tr = d.snapshot_day.isin(fit_days).values
    b = np.linalg.solve(Xs[tr].T@Xs[tr]+lam*np.eye(len(cols)), Xs[tr].T@y[tr])
    p = Xs@b
    m = ~tr
    return p[m], y[m]

p, y = ridge_eval(["spend112"], [95,123,151,179,207,235,263,291,319,347,375,403], [431])
print("single-feat ridge: MAE=%.2f  mean pred=%.1f mean y=%.1f  pred std=%.1f" % (np.abs(p-y).mean(), p.mean(), y.mean(), p.std()))

# now check Xs magnitudes for core64
feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
sd = X.std()
print("\nfeatures with sd=0:", (sd==0).sum())
print("features with sd>1000:", (sd>1000).sum())
print(sd.sort_values(ascending=False).head(8).round(0))
# check for features with extreme values pre-clip
mx = X.abs().max()
print("\nmax |x| top:", mx.sort_values(ascending=False).head(8).round(0))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

e011 = A.load_saved("e011_price.parquet")
tt = A.train_targets()
feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")

def ridge_eval(cols, fit_days, eval_days, lam=3.0):
    d = df[df.snapshot_day.isin(fit_days+eval_days)]
    X = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0,1)
    Xs = ((X-mu)/sd).clip(-10,10).fillna(0).values
    Xs = np.column_stack([np.ones(len(Xs)), Xs])   # intercept
    y = d["future_spend_4w"].values
    tr = d.snapshot_day.isin(fit_days).values
    b = np.linalg.solve(Xs[tr].T@Xs[tr]+lam*np.eye(len(cols)+1), Xs[tr].T@y[tr])
    p = Xs@b
    m = ~tr
    return np.mean(np.abs(p[m]-y[m])), 1-((y[m]-p[m])**2).sum()/((y[m]-y[tr].mean())**2).sum()

def grp(pre): return [c for c in feats if c.startswith(pre)]
dsp = grp("dsp_"); ap = grp("ap_"); pt = grp("price_trend"); qty = grp("qty_"); sp = grp("spend_p")
iv = grp("iv_")+grp("bk_")+["n_gaps112"]
lag = grp("lag_")+["dec_spend14","dec_spend7","dec_trips","gap_mean","gap_std","gap_med","ntrip112"]
mkt = grp("tA_")+grp("tB_")+grp("tC_")+grp("red_")+["n_tgt_ever","n_tgt_active","targeted_flag","days_since_first_tgt","days_since_last_tgt","tgt_active_remaining","cpn_prod_spend28","cpn_prod_share28"]
demo = grp("demo_")+["has_demo"]
comp = ["ndept28","spend_w0","spend_w1","spend_w2","spend_w3","share_w0","rwspend84","active_weeks112","weekend_share28","zero28","ratio28_364","toptrip_share28","spend28_log","spend56_log","spend112_log","lt_spend_log","nbaskets28"]
misc = ["pl_share","disc_share2","c_qty112","macro_ratio","mom_28_112","mom_112_364","upt28","upt112","maxtrip_qty28","share_qty_toptrip"]
core = [c for c in feats if c not in dsp+ap+pt+qty+sp+iv+lag+mkt+demo+comp+misc and not c.startswith("dm_")]

tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
print("GROUP DIAG (ridge lam=3, WITH intercept; fit<=403, eval 431):")
for name, cols in [("core64",core),("core+lag",core+lag),("core+lag+comp",core+lag+comp),
                   ("core+lag+comp+mkt",core+lag+comp+mkt),
                   ("core+lag+comp+mkt+dsp",core+lag+comp+mkt+dsp),
                   ("core+lag+comp+mkt+price",core+lag+comp+mkt+qty+sp+pt),
                   ("core+lag+comp+mkt+ap+iv+misc",core+lag+comp+mkt+ap+iv+misc),
                   ("ALL228",feats)]:
    mae, r2 = ridge_eval(cols, tr_days, [431])
    print(f"  {name:30s} n={len(cols):3d}  MAE={mae:7.3f}  R2={r2:.4f}")

print("\ncross-check (fit<=375, eval 403):")
for name, cols in [("core64",core),("core+lag",core+lag),("ALL228",feats)]:
    mae, r2 = ridge_eval(cols, tr_days[:-1], [403])
    print(f"  {name:30s} n={len(cols):3d}  MAE={mae:7.3f}  R2={r2:.4f}")


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]

def ridge_eval(df, cols, fit_days, eval_days, lam=3.0):
    d = df[df.snapshot_day.isin(fit_days+eval_days)]
    X = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0,1)
    Xs = ((X-mu)/sd).clip(-10,10).fillna(0).values
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    y = d["future_spend_4w"].values
    tr = d.snapshot_day.isin(fit_days).values
    b = np.linalg.solve(Xs[tr].T@Xs[tr]+lam*np.eye(len(cols)+1), Xs[tr].T@y[tr])
    p = Xs@b
    m = ~tr
    return np.mean(np.abs(p[m]-y[m]))

rb = A.load_saved("rich_behavioral.parquet")   # E003's 53 features
rfm = A.load_saved("rfm28.parquet")            # E001's 3 features
rb_m = rb.merge(tt, on=["household_key","snapshot_day"])
rfm_m = rfm.merge(tt, on=["household_key","snapshot_day"])

print("E001 harness=69.595 | my ridge lam sweep on [spend28,trips28,recency]:")
for lam in [0.3,1,3,10,30]:
    print(f"  lam={lam:5}: {ridge_eval(rfm_m, ['spend28','trips28','recency'], tr_days, [431], lam):.3f}")

rbf = [c for c in rb.columns if c not in ("household_key","snapshot_day")]
print("\nE003 harness=61.525 | my ridge on 53 rich_behavioral feats:")
for lam in [0.3,1,3,10,30]:
    print(f"  lam={lam:5}: {ridge_eval(rb_m, rbf, tr_days, [431], lam):.3f}")

# also eval on 403 for stability
print("\non 403 (fit<=375): rfm3:", {lam: round(ridge_eval(rfm_m, ['spend28','trips28','recency'], tr_days[:-1], [403], lam),3) for lam in [1,3,10]})
print("rich53 403:", {lam: round(ridge_eval(rb_m, rbf, tr_days[:-1], [403], lam),3) for lam in [1,3,10]})


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
rfm = A.load_saved("rfm28.parquet").merge(tt, on=["household_key","snapshot_day"])
cols = ["spend28","trips28","recency"]

def prep(df, cols):
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0,1)
    return ((X-mu)/sd).clip(-10,10).fillna(0).values

d = rfm
Xs = np.column_stack([np.ones(len(d)), prep(d, cols)])
y = d["future_spend_4w"].values
tr = d.snapshot_day.isin(tr_days).values; m = ~tr
Xtr, ytr = Xs[tr], y[tr]

def irls(X, y, delta=None, n_iter=30, lam=1.0):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(n_iter):
        r = y - X@b
        s = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-9
        dl = delta if delta else 1.345*s
        w = np.minimum(1.0, dl/np.maximum(np.abs(r), 1e-9))
        W = np.diag(w) if len(w)<5 else None
        # weighted solve with ridge
        Xw = X * w[:,None]
        b = np.linalg.solve(X.T@Xw + lam*np.eye(X.shape[1]), Xw.T@y)
    return b

# 1) plain ridge (baseline)
b = np.linalg.solve(Xtr.T@Xtr+1*np.eye(4), Xtr.T@ytr)
print("ridge          :", np.abs(Xs[m]@b - y[m]).mean().round(3), "(harness 69.595)")
# 2) Huber/LAD-ridge
for dl in [None, 30, 60]:
    b = irls(Xtr, ytr, delta=dl)
    print(f"huber delta={dl}:", np.abs(Xs[m]@b - y[m]).mean().round(3))
# 3) recency-weighted ridge (weight ~ snapshot day)
w_snap = d.snapshot_day.values / 431.0
Xw = Xtr * w_snap[tr][:,None]
b = np.linalg.solve(Xtr.T@Xw + 1*np.eye(4), Xw.T@ytr)
print("recency-w ridge:", np.abs(Xs[m]@b - y[m]).mean().round(3))
# 4) both
b = irls(Xtr, ytr, delta=60)
r = ytr - Xtr@b; w = np.minimum(1.0, 60/np.maximum(np.abs(r),1e-9)) * w_snap[tr]
Xw = Xtr * w[:,None]
b = np.linalg.solve(Xtr.T@Xw + 1*np.eye(4), Xw.T@ytr)
print("huber+recency  :", np.abs(Xs[m]@b - y[m]).mean().round(3))
# 5) sample weights by day as pure weighting of squared loss (no robustness)
for pw in [0.5, 1.0, 2.0]:
    w = (d.snapshot_day.values/431.0)**pw
    Xw = Xtr * w[tr][:,None]
    b = np.linalg.solve(Xtr.T@Xw + 1*np.eye(4), Xw.T@ytr)
    print(f"day^pw={pw} ridge:", np.abs(Xs[m]@b - y[m]).mean().round(3))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
e011 = A.load_saved("e011_price.parquet")
feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")

def huber_eval(df, cols, fit_days, eval_days, lam=1.0, delta=None, n_iter=25):
    d = df[df.snapshot_day.isin(fit_days+eval_days)]
    X = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0,1)
    Xs = np.column_stack([np.ones(len(d)), ((X-mu)/sd).clip(-10,10).fillna(0).values])
    y = d["future_spend_4w"].values
    tr = d.snapshot_day.isin(fit_days).values; m = ~tr
    Xtr, ytr = Xs[tr], y[tr]
    b = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    for _ in range(n_iter):
        r = ytr - Xtr@b
        s = np.median(np.abs(r-np.median(r)))*1.4826 + 1e-9
        dl = delta if delta else 1.345*s
        w = np.minimum(1.0, dl/np.maximum(np.abs(r),1e-9))
        Xw = Xtr * w[:,None]
        b = np.linalg.solve(Xtr.T@Xw + lam*np.eye(Xtr.shape[1]), Xw.T@ytr)
    p = Xs[m]@b
    return np.abs(p - y[m]).mean()

def grp(pre): return [c for c in feats if c.startswith(pre)]
dsp = grp("dsp_"); ap = grp("ap_"); pt = grp("price_trend"); qty = grp("qty_"); sp = grp("spend_p")
iv = grp("iv_")+grp("bk_")+["n_gaps112"]
lag = grp("lag_")+["dec_spend14","dec_spend7","dec_trips","gap_mean","gap_std","gap_med","ntrip112"]
mkt = grp("tA_")+grp("tB_")+grp("tC_")+grp("red_")+["n_tgt_ever","n_tgt_active","targeted_flag","days_since_first_tgt","days_since_last_tgt","tgt_active_remaining","cpn_prod_spend28","cpn_prod_share28"]
demo = grp("demo_")+["has_demo"]
comp = ["ndept28","spend_w0","spend_w1","spend_w2","spend_w3","share_w0","rwspend84","active_weeks112","weekend_share28","zero28","ratio28_364","toptrip_share28","spend28_log","spend56_log","spend112_log","lt_spend_log","nbaskets28"]
misc = ["pl_share","disc_share2","c_qty112","macro_ratio","mom_28_112","mom_112_364","upt28","upt112","maxtrip_qty28","share_qty_toptrip"]
core = [c for c in feats if c not in dsp+ap+pt+qty+sp+iv+lag+mkt+demo+comp+misc and not c.startswith("dm_")]

print("HUBER probe (delta=30, lam=1; fit<=403, eval 431):")
for name, cols in [("core64",core),("core+lag",core+lag),("core+lag+comp",core+lag+comp),
                   ("E011(228)",feats)]:
    print(f"  {name:16s} n={len(cols):3d}  MAE={huber_eval(df, cols, tr_days, [431]):.3f}")

# engineered candidates
sp28 = pd.to_numeric(df.spend28,errors="coerce").fillna(0); sp56 = pd.to_numeric(df.spend56,errors="coerce").fillna(0)
sp112 = pd.to_numeric(df.spend112,errors="coerce").fillna(0); rec = pd.to_numeric(df.recency,errors="coerce").fillna(999)
tr28 = pd.to_numeric(df.trips28,errors="coerce").fillna(0); tr56 = pd.to_numeric(df.trips56,errors="coerce").fillna(0)
tr112 = pd.to_numeric(df.trips112,errors="coerce").fillna(0)
eng = pd.DataFrame(index=df.index)
eng["blend"] = np.clip(tr28/4,0,1)*sp28 + (1-np.clip(tr28/4,0,1))*(sp112/4)
eng["dorm"] = (sp112/4)*np.exp(-rec/21.0)
eng["sp28_log"] = np.log1p(sp28); eng["sp56_log"] = np.log1p(sp56); eng["sp112_log"] = np.log1p(sp112)
eng["sp364_log"] = np.log1p(pd.to_numeric(df.spend364,errors="coerce").fillna(0))
eng["sqrt28"] = np.sqrt(sp28); eng["sqrt112"] = np.sqrt(sp112)
eng["cons28_log"] = np.log1p(sp28); eng["cons112_log"] = np.log1p(sp112/4)
eng["trips28_log"] = np.log1p(tr28); eng["trips112_log"] = np.log1p(tr112)
eng["rec_sq"] = rec**2; eng["rec_cubert"] = rec**(1/3)
eng["sp28xrec"] = np.log1p(sp28)*rec
eng["sp112xrec"] = np.log1p(sp112)*rec
eng["freq28"] = tr28/28.0; eng["freq112"] = tr112/112.0
eng["spend_per_trip_log"] = np.log1p(sp28/np.maximum(tr28,1))
eng["sp28_cubert"] = sp28**(1/3); eng["sp112_cubert"] = sp112**(1/3)
engf = list(eng.columns)
df2 = pd.concat([df, eng], axis=1)
print("\nengineered (added to core+lag):")
base_cols = core+lag
print(f"  core+lag            {huber_eval(df2, base_cols, tr_days, [431]):.3f}")
print(f"  +eng({len(engf)})      {huber_eval(df2, base_cols+engf, tr_days, [431]):.3f}")
print(f"  E011+eng            {huber_eval(df2, feats+engf, tr_days, [431]):.3f}")
print(f"  eng only on E011    {huber_eval(df2, feats+engf, tr_days, [431]):.3f}")
# which eng feats help individually (added to core+lag)?
print("\nindividual eng feats (added to core+lag):")
for c in engf:
    m1 = huber_eval(df2, base_cols, tr_days, [431]); m2 = huber_eval(df2, base_cols+[c], tr_days, [431])
    print(f"  {c:20s} {m2:.3f} (d={m2-m1:+.3f})")


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
e011 = A.load_saved("e011_price.parquet")
feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")

def huber_eval(df, cols, fit_days, eval_days, lam=1.0, l1=0.0, delta=30.0, n_iter=25):
    d = df[df.snapshot_day.isin(fit_days+eval_days)]
    X = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0,1)
    Xs = np.column_stack([np.ones(len(d)), ((X-mu)/sd).clip(-10,10).fillna(0).values])
    y = d["future_spend_4w"].values
    tr = d.snapshot_day.isin(fit_days).values; m = ~tr
    Xtr, ytr = Xs[tr], y[tr]
    b = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    for _ in range(n_iter):
        r = ytr - Xtr@b
        w = np.minimum(1.0, delta/np.maximum(np.abs(r),1e-9))
        Xw = Xtr * w[:,None]
        b = np.linalg.solve(Xtr.T@Xw + lam*np.eye(Xtr.shape[1]), Xw.T@ytr)
        if l1 > 0:  # simple l1 shrink step
            b[1:] = np.sign(b[1:]) * np.maximum(np.abs(b[1:]) - l1/ (np.diag(Xtr.T@Xw)+1e-9), 0)
    p = Xs[m]@b
    return np.abs(p - y[m]).mean()

def grp(pre): return [c for c in feats if c.startswith(pre)]
dsp = grp("dsp_"); ap = grp("ap_"); pt = grp("price_trend"); qty = grp("qty_"); sp = grp("spend_p")
iv = grp("iv_")+grp("bk_")+["n_gaps112"]
lag = grp("lag_")+["dec_spend14","dec_spend7","dec_trips","gap_mean","gap_std","gap_med","ntrip112"]
mkt = grp("tA_")+grp("tB_")+grp("tC_")+grp("red_")+["n_tgt_ever","n_tgt_active","targeted_flag","days_since_first_tgt","days_since_last_tgt","tgt_active_remaining","cpn_prod_spend28","cpn_prod_share28"]
demo = grp("demo_")+["has_demo"]
comp = ["ndept28","spend_w0","spend_w1","spend_w2","spend_w3","share_w0","rwspend84","active_weeks112","weekend_share28","zero28","ratio28_364","toptrip_share28","spend28_log","spend56_log","spend112_log","lt_spend_log","nbaskets28"]
dm = [c for c in feats if c.startswith("dm_")]
misc = ["pl_share","disc_share2","c_qty112","macro_ratio","mom_28_112","mom_112_364","upt28","upt112","maxtrip_qty28","share_qty_toptrip"]
core = [c for c in feats if c not in dsp+ap+pt+qty+sp+iv+lag+mkt+demo+comp+misc+dm]

cands = {
 "core64": core,
 "core+comp": core+comp,
 "core+lag": core+lag,
 "core+lag+comp": core+lag+comp,
 "core+lag+comp+mkt": core+lag+comp+mkt,
 "core+lag+comp+mkt+dm": core+lag+comp+mkt+dm,
 "core+lag+comp+mkt+dm+demo": core+lag+comp+mkt+dm+demo,
 "E011-dsp(184)": [c for c in feats if c not in dsp],
 "E011-junk(142)": core+lag+comp+mkt+dm+demo,
 "E011(228)": feats,
}
print("HUBER probe, lam=1, delta=30 — eval on 431 and 403:")
print(f"{'cand':28s} {'n':>4s} {'431':>8s} {'403':>8s}")
for name, cols in cands.items():
    m431 = huber_eval(df, cols, tr_days, [431])
    m403 = huber_eval(df, cols, tr_days[:-1], [403])
    print(f"{name:28s} {len(cols):4d} {m431:8.3f} {m403:8.3f}")

print("\nL1 variants (lam=1, l1 shrink) on 431:")
for name, cols in [("core64",core),("E011-junk(142)",core+lag+comp+mkt+dm+demo),("E011(228)",feats)]:
    print(f"  {name:16s} l1=0.02: {huber_eval(df, cols, tr_days, [431], l1=0.02):.3f}  l1=0.05: {huber_eval(df, cols, tr_days, [431], l1=0.05):.3f}")


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

# --- build new features through build_features (sees only data <= snapshot day) ---
def fn(view, snapshot_day):
    hh = pd.Index(view.households)
    t = view.transactions
    out = pd.DataFrame(index=hh)
    # 1) calendar-aligned last-year forward window: [D-363, D-336] mirrors target [D+1, D+28] LY
    lo, hi = snapshot_day-363, snapshot_day-336
    m = (t.day >= lo) & (t.day <= hi)
    sp = t.loc[m].groupby("household_key").sales_value.sum()
    first = t.groupby("household_key").day.min()
    out["fwd_ly"] = sp.reindex(hh).fillna(0.0)
    out["fwd_ly_valid"] = (first.reindex(hh).fillna(10**9) <= lo)
    # 2) empirical 4-week activity rates from weekly rolling sums
    wk = t.groupby(["household_key","week_no"]).sales_value.sum().unstack(fill_value=0.0)
    wk = wk.reindex(index=hh, fill_value=0.0)
    w_end = int((snapshot_day + 8)//7)
    allw = np.arange(1, w_end+1)
    wk = wk.reindex(columns=allw, fill_value=0.0).fillna(0.0)
    R = wk.rolling(4, axis=1, min_periods=4).sum()   # 4-week spend ending at week w
    lo_w = max(4, w_end-51)
    vals = R.loc[:, lo_w:w_end].values
    with np.errstate(invalid="ignore"):
        out["act_rate_4w"] = np.nanmean((vals > 0).astype(float), axis=1)
    lo_w2 = max(4, w_end-15)
    vals2 = R.loc[:, lo_w2:w_end].values
    with np.errstate(invalid="ignore"):
        out["act_rate_16w"] = np.nanmean((vals2 > 0).astype(float), axis=1)
    return out

newf = A.build_features(fn)
print("newfeat:", newf.shape)
print(newf.drop(columns=["household_key","snapshot_day"]).describe().round(3))
A.save_table(newf, "newfeat.parquet")

# --- inert feature audit on E011 ---
e011 = A.load_saved("e011_price.parquet")
tt = A.train_targets()
feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")
X = df[feats].apply(pd.to_numeric, errors="coerce")
sd = X.std()
inert = list(sd[sd==0].index)
print("\ninert (sd=0 on train):", len(inert), inert[:40])

# --- probe: newfeat added to core+lag+comp ---
def huber_eval(df, cols, fit_days, eval_days, lam=1.0, delta=30.0, n_iter=25):
    d = df[df.snapshot_day.isin(fit_days+eval_days)]
    X = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0,1)
    Xs = np.column_stack([np.ones(len(d)), ((X-mu)/sd).clip(-10,10).fillna(0).values])
    y = d["future_spend_4w"].values
    tr = d.snapshot_day.isin(fit_days).values; m = ~tr
    Xtr, ytr = Xs[tr], y[tr]
    b = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    for _ in range(n_iter):
        r = ytr - Xtr@b
        w = np.minimum(1.0, delta/np.maximum(np.abs(r),1e-9))
        Xw = Xtr * w[:,None]
        b = np.linalg.solve(Xtr.T@Xw + lam*np.eye(Xtr.shape[1]), Xw.T@ytr)
    return np.abs(Xs[m]@b - y[m]).mean()

tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
df2 = df.merge(newf.drop(columns=["household_key","snapshot_day"]), left_index=True, right_index=True) if False else df
# merge on keys properly:
df2 = df.merge(newf, on=["household_key","snapshot_day"], how="left")
base = [c for c in feats if c not in inert]  # all E011 features minus inert
def grp(pre): return [c for c in feats if c.startswith(pre)]
dsp = grp("dsp_"); ap = grp("ap_"); pt = grp("price_trend"); qty = grp("qty_"); sp = grp("spend_p")
iv = grp("iv_")+grp("bk_")+["n_gaps112"]
lag = grp("lag_")+["dec_spend14","dec_spend7","dec_trips","gap_mean","gap_std","gap_med","ntrip112"]
mkt = grp("tA_")+grp("tB_")+grp("tC_")+grp("red_")+["n_tgt_ever","n_tgt_active","targeted_flag","days_since_first_tgt","days_since_last_tgt","tgt_active_remaining","cpn_prod_spend28","cpn_prod_share28"]
demo = grp("demo_")+["has_demo"]
comp = ["ndept28","spend_w0","spend_w1","spend_w2","spend_w3","share_w0","rwspend84","active_weeks112","weekend_share28","zero28","ratio28_364","toptrip_share28","spend28_log","spend56_log","spend112_log","lt_spend_log","nbaskets28"]
dm = [c for c in feats if c.startswith("dm_")]
misc = ["pl_share","disc_share2","c_qty112","macro_ratio","mom_28_112","mom_112_364","upt28","upt112","maxtrip_qty28","share_qty_toptrip"]
clp = core_lag_comp = [c for c in feats if c not in dsp+ap+pt+qty+sp+iv+lag+mkt+demo+comp+dm+misc]
nf = ["fwd_ly","fwd_ly_valid","act_rate_4w","act_rate_16w"]
df2["exp_spend"] = df2["act_rate_4w"] * pd.to_numeric(df2["spend112"],errors="coerce").fillna(0)/4.0
df2["exp_spend16"] = df2["act_rate_16w"] * pd.to_numeric(df2["spend112"],errors="coerce").fillna(0)/4.0
df2["fwd_ly_blend"] = df2["fwd_ly"].fillna(0)*0.5 + pd.to_numeric(df2["spend28"],errors="coerce").fillna(0)*0.5

print("\nPROBE (huber delta=30; fit<=403 eval 431 | fit<=375 eval 403):")
for name, cols in [("prune142", clp+lag+comp+mkt+dm+demo),
                   ("prune116", clp+lag+comp),
                   ("prune116+nf", clp+lag+comp+nf),
                   ("prune116+nf+exp", clp+lag+comp+nf+["exp_spend","exp_spend16","fwd_ly_blend"])]:
    m431 = huber_eval(df2, cols, tr_days, [431]); m403 = huber_eval(df2, cols, tr_days[:-1], [403])
    print(f"  {name:20s} n={len(cols):3d}  431={m431:7.3f}  403={m403:7.3f}")
# individual nf deltas on prune116
b0 = huber_eval(df2, clp+lag+comp, tr_days, [431])
for c in nf+["exp_spend","exp_spend16","fwd_ly_blend"]:
    print(f"  +{c:14s} {huber_eval(df2, clp+lag+comp+[c], tr_days, [431]):.3f} (d={huber_eval(df2, clp+lag+comp+[c], tr_days, [431])-b0:+.3f})")


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
e011 = A.load_saved("e011_price.parquet")
feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")

def grp(pre): return [c for c in feats if c.startswith(pre)]
dsp = grp("dsp_"); ap = grp("ap_"); pt = grp("price_trend"); qty = grp("qty_"); sp = grp("spend_p")
iv = grp("iv_")+grp("bk_")+["n_gaps112"]
lag = grp("lag_")+["dec_spend14","dec_spend7","dec_trips","gap_mean","gap_std","gap_med","ntrip112"]
mkt = grp("tA_")+grp("tB_")+grp("tC_")+grp("red_")+["n_tgt_ever","n_tgt_active","targeted_flag","days_since_first_tgt","days_since_last_tgt","tgt_active_remaining","cpn_prod_spend28","cpn_prod_share28"]
demo = grp("demo_")+["has_demo"]
comp = ["ndept28","spend_w0","spend_w1","spend_w2","spend_w3","share_w0","rwspend84","active_weeks112","weekend_share28","zero28","ratio28_364","toptrip_share28","spend28_log","spend56_log","spend112_log","lt_spend_log","nbaskets28"]
dm = [c for c in feats if c.startswith("dm_")]
misc = ["pl_share","disc_share2","c_qty112","macro_ratio","mom_28_112","mom_112_364","upt28","upt112","maxtrip_qty28","share_qty_toptrip"]
keep = [c for c in feats if c not in dsp+ap+pt+qty+sp+iv+lag+mkt+demo+comp+dm+misc]
keep116 = keep + lag + comp
assert len(keep116)==116, len(keep116)
prune = e011[["household_key","snapshot_day"]+keep116]
A.save_table(prune, "e017_prune.parquet")
print("saved e017_prune:", prune.shape)

# probe interactions on prune116 base
def huber_eval(df, cols, fit_days, eval_days, lam=1.0, delta=30.0, n_iter=25):
    d = df[df.snapshot_day.isin(fit_days+eval_days)]
    X = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0,1)
    Xs = np.column_stack([np.ones(len(d)), ((X-mu)/sd).clip(-10,10).fillna(0).values])
    y = d["future_spend_4w"].values
    tr = d.snapshot_day.isin(fit_days).values; m = ~tr
    Xtr, ytr = Xs[tr], y[tr]
    b = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    for _ in range(n_iter):
        r = ytr - Xtr@b
        w = np.minimum(1.0, delta/np.maximum(np.abs(r),1e-9))
        Xw = Xtr * w[:,None]
        b = np.linalg.solve(Xtr.T@Xw + lam*np.eye(Xtr.shape[1]), Xw.T@ytr)
    return np.abs(Xs[m]@b - y[m]).mean()

df["sp112"] = pd.to_numeric(df.spend112,errors="coerce").fillna(0)
df["sp28"]  = pd.to_numeric(df.spend28,errors="coerce").fillna(0)
df["rec"]   = pd.to_numeric(df.recency,errors="coerce").fillna(999)
df["trend"] = pd.to_numeric(df.trend_112_364,errors="coerce").fillna(1).clip(-2,3)
df["oly"]   = pd.to_numeric(df.own_ly_spend4w,errors="coerce").fillna(0)
df["i_sp112_trend"] = df.sp112*df.trend
df["i_sp112_rec"]   = df.sp112*df.rec/100.0
df["i_sp28_rec"]    = df.sp28*df.rec/100.0
df["i_sp112_oly"]   = df.sp112*df.oly.clip(0,500)
df["i_sp28_oly"]    = df.sp28*df.oly.clip(0,500)
df["i_sp112_sq"]    = df.sp112**2/1000.0
df["i_sp28_sq"]     = df.sp28**2/1000.0
ic = ["i_sp112_trend","i_sp112_rec","i_sp28_rec","i_sp112_oly","i_sp28_oly","i_sp112_sq","i_sp28_sq"]
b0_431 = huber_eval(df, keep116, tr_days, [431]); b0_403 = huber_eval(df, keep116, tr_days[:-1], [403])
print(f"\nprune116 base: 431={b0_431:.3f} 403={b0_403:.3f}")
for c in ic:
    m431 = huber_eval(df, keep116+[c], tr_days, [431]); m403 = huber_eval(df, keep116+[c], tr_days[:-1], [403])
    print(f"  +{c:16s} 431={m431:.3f}({m431-b0_431:+.3f})  403={m403:.3f}({m403-b0_403:+.3f})")
all4 = huber_eval(df, keep116+ic, tr_days, [431]); all4b = huber_eval(df, keep116+ic, tr_days[:-1], [403])
print(f"  +all 7          431={all4:.3f}({all4-b0_431:+.3f})  403={all4b:.3f}({all4b-b0_403:+.3f})")
