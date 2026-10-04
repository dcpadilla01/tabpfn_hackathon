
import agent_api as A, pandas as pd, numpy as np
names = ["e018_union_full","e019_true_union","e017_v2","e016_smoothed","e013_peers","e018_timing_hazard","nf_candidates","nf_robust","nf_seasonal","nf_transforms","micro","nf_compact19"]
for n in names:
    try:
        df = A.load_saved(n+".parquet")
        print(n, df.shape)
        print(list(df.columns))
        print("---")
    except Exception as e:
        print(n, "ERR", e)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
base_cols = set(base.columns)
others = ["e019_true_union","e019_everything","e019_full_merged","e017_v2","e017_disc_seasonal","e016_smoothed","e013_peers","e018_timing_hazard","nf_candidates","nf_robust","nf_seasonal","nf_transforms","micro","nf_compact19","e006_catmix_mkt","e007_logratio","e010_l13fix","e011_demo","e012_dorm","e015_peers_demo_l13fix","e002_channel","e003_catmix","e004_mkt_full","e001_txhist"]
missing = {}
for n in others:
    try:
        df = A.load_saved(n+".parquet")
    except Exception as e:
        print(n, "ERR", e); continue
    miss = [c for c in df.columns if c not in base_cols and c not in ("household_key","snapshot_day")]
    if miss:
        missing[n] = miss
        print(n, df.shape, "MISSING:", miss)
print()
print("base feats:", base.shape[1]-2)

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
# 1) nf_seasonal + micro: which columns are NOT in base
base = A.load_saved("e018_union_full.parquet"); bc = set(base.columns)
for n in ["nf_seasonal","micro"]:
    df = A.load_saved(n+".parquet")
    print(n, [c for c in df.columns if c not in bc and c not in ("household_key","snapshot_day")])
# 2) e017_disc_seasonal: normalize _x/_y suffixes and find truly new cols
ds = A.load_saved("e017_disc_seasonal.parquet")
norm = set(c.rstrip("_x").rstrip("_y") if c.endswith(("_x","_y")) else c for c in ds.columns)
print("e017_disc_seasonal truly-new:", sorted(c for c in norm if c not in bc and c not in ("household_key","snapshot_day","index")))
# 3) train targets + target distribution
tt = A.train_targets()
print(tt.shape, tt.future_spend_4w.describe())
print("zero share train:", (tt.future_spend_4w==0).mean())
# 4) display_mailer overview
v = A.snapshot()
dm = v.display_mailer
print("display_mailer", dm.shape); print(dm.head(3)); print(dm.display.value_counts().head()); print(dm.mailer.value_counts().head())

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
tt = A.train_targets()
tr = base.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("train rows", tr.shape)
y = tr.future_spend_4w.values.astype(float)
num = [c for c in tr.columns if c not in ("household_key","snapshot_day","future_spend_4w") and pd.api.types.is_numeric_dtype(tr[c])]
print("n numeric cols", len(num))
cor = {}
for c in num:
    x = tr[c].values.astype(float)
    ok = np.isfinite(x)
    if ok.sum() < 1000: continue
    cor[c] = np.corrcoef(x[ok], y[ok])[0,1]
cor = pd.Series(cor).sort_values(key=np.abs, ascending=False)
print(cor.head(35))
print()
print(cor.tail(10))

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
tt = A.train_targets()
tr = base.merge(tt, on=["household_key","snapshot_day"], how="inner")
y = tr.future_spend_4w.values.astype(float)
# all-NaN / constant columns in base (train rows)
numcols = [c for c in tr.columns if c not in ("household_key","snapshot_day","future_spend_4w") and pd.api.types.is_numeric_dtype(tr[c])]
dead = [c for c in numcols if tr[c].notna().sum()==0 or tr[c].nunique(dropna=True)<=1]
print("dead numeric cols:", dead)
# cheap adds: nf_seasonal (2), coupon_disc_84/match_disc_84 (from e019_true_union), nratio_l/nspend_l/tenure (e017_disc_seasonal)
ns = A.load_saved("nf_seasonal.parquet")[["household_key","snapshot_day","nseas_uplift","nseas_uplift_mean"]]
tu = A.load_saved("e019_true_union.parquet")[["household_key","snapshot_day","coupon_disc_84","match_disc_84"]]
ds = A.load_saved("e017_disc_seasonal.parquet")[["household_key","snapshot_day","nratio_l","nspend_l","tenure"]]
m = tr.merge(ns, on=["household_key","snapshot_day"]).merge(tu, on=["household_key","snapshot_day"]).merge(ds, on=["household_key","snapshot_day"])
for c in ["nseas_uplift","nseas_uplift_mean","coupon_disc_84","match_disc_84","nratio_l","nspend_l","tenure"]:
    x = m[c].values.astype(float); ok = np.isfinite(x)
    print(c, "nan%", round(100*(1-ok.mean()),1), "corr", round(np.corrcoef(x[ok],y[ok])[0,1],4) if ok.sum()>100 else "NA")
# also check spend_ly4w / peer cols corr for context
for c in ["spend_ly4w","peer_p90","zero_frac_full","inactive_run84","n_zero_l13"]:
    x = tr[c].values.astype(float); ok=np.isfinite(x)
    print(c, "nan%", round(100*(1-ok.mean()),1), "corr", round(np.corrcoef(x[ok],y[ok])[0,1],4))

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
tt = A.train_targets()
tr = base.merge(tt, on=["household_key","snapshot_day"], how="inner")
y = tr.future_spend_4w.values.astype(float)
ns = A.load_saved("nf_seasonal.parquet")[["household_key","snapshot_day","nseas_uplift","nseas_uplift_mean"]]
tu = A.load_saved("e019_true_union.parquet")[["household_key","snapshot_day","coupon_disc_84","match_disc_84"]]
ds = A.load_saved("e017_disc_seasonal.parquet")
dscols = [c for c in ds.columns if c.endswith(("_x","_y"))]
print("ds sample cols:", dscols[:6], "...", len(dscols))
# find the ones that are tenure/nspend_l/nratio_l
cand = {}
for c in dscols:
    root = c[:-2]
    if root in ("tenure","nspend_l","nratio_l"):
        cand[c] = root
print(cand)
m = tr.merge(ns, on=["household_key","snapshot_day"]).merge(tu, on=["household_key","snapshot_day"]).merge(ds[["household_key","snapshot_day"]+list(cand)], on=["household_key","snapshot_day"])
for c in ["nseas_uplift","nseas_uplift_mean","coupon_disc_84","match_disc_84"]+list(cand):
    x = m[c].values.astype(float); ok = np.isfinite(x)
    print(c, "nan%", round(100*(1-ok.mean()),1), "corr", round(np.corrcoef(x[ok],y[ok])[0,1],4) if ok.sum()>100 else "NA")

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
ds = A.load_saved("e017_disc_seasonal.parquet")
print([c for c in ds.columns if "tenure" in c or "nspend_l" in c or "nratio_l" in c])

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
print("s_day in base:", "s_day" in base.columns)
rb = A.load_saved("nf_robust.parquet"); bc=set(base.columns)
print("nf_robust new:", [c for c in rb.columns if c not in bc and c not in ("household_key","snapshot_day")])
tt = A.train_targets()
tr = base.merge(tt, on=["household_key","snapshot_day"], how="inner").sort_values(["snapshot_day","household_key"])
y = tr.future_spend_4w.values.astype(float)
# drift check
g = tr.groupby("snapshot_day").agg(y_mean=("future_spend_4w","mean"), l1=("spend_l1","mean"), newm4=("newm4","mean"), m123=("spend_l123_mean","mean"))
print(g.round(1))
# persistence MAEs
def mae(p): return np.mean(np.abs(y-p))
for c in ["spend_l1","newm4","spend_l123_mean","ewm13_d","spend_l456_mean","r_med13","peer_ratio"]:
    print(c, round(mae(tr[c].values.astype(float)),2))
# best simple blend (in-sample, indicative)
cands = ["spend_l1","newm4","spend_l123_mean","ewm13_d","spend_l456_mean","r_med13","nspend28"]
P = np.column_stack([tr[c].values.astype(float) for c in cands])
from numpy.linalg import lstsq
w,_,_,_ = lstsq(P, y, rcond=None)
print("blend weights", np.round(w,3), "MAE", round(mae(P@w),2))
print("median pred MAE", round(mae(np.full_like(y, np.median(y))),2))

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
tt = A.train_targets()
tr = base.merge(tt, on=["household_key","snapshot_day"], how="inner").sort_values(["snapshot_day","household_key"])
y = tr.future_spend_4w.values.astype(float)
# 1) log-space ridge blend of top ~40 features (no tree model; ridge on log1p(y))
from numpy.linalg import lstsq
numcols = [c for c in tr.columns if c not in ("household_key","snapshot_day","future_spend_4w") and pd.api.types.is_numeric_dtype(tr[c])]
numcols = [c for c in numcols if tr[c].nunique(dropna=True)>1]
# standardize, impute median
Z = tr[numcols].astype(float)
med = Z.median()
Z = Z.fillna(med).values
Z = (Z - Z.mean(0)) / (Z.std(0)+1e-9)
Z = np.clip(Z, -8, 8)
ly = np.log1p(y)
# ridge sweep with alpha
def ridge_fit(Zt, yt, alpha):
    n,p = Zt.shape
    A_ = Zt.T@Zt + alpha*np.eye(p)
    return np.linalg.solve(A_, Zt.T@yt)
# leave-one-snapshot-out CV
days = sorted(tr.snapshot_day.unique())
def loso(alpha, target="log"):
    preds = np.zeros(len(tr))
    for d in days:
        te = (tr.snapshot_day.values==d)
        Ztr, ytr = Z[~te], (ly[~te] if target=="log" else y[~te])
        w = ridge_fit(Ztr, ytr, alpha)
        preds[te] = Z[te]@w
        if target=="log": preds[te] = np.expm1(np.clip(preds[te],0,8))
    return np.mean(np.abs(y-preds))
for alpha in [30,100,300,1000]:
    print("alpha",alpha,"log-ridge LOSO MAE", round(loso(alpha,"log"),2), "| linear LOSO MAE", round(loso(alpha,"lin"),2))

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
tt = A.train_targets()
tr = base.merge(tt, on=["household_key","snapshot_day"], how="inner").sort_values(["snapshot_day","household_key"]).reset_index(drop=True)
y = tr.future_spend_4w.values.astype(float)
print("mean y", y.mean())
days = sorted(tr.snapshot_day.unique())
print(days)
te = (tr.snapshot_day.values==days[0])
print("te sum", te.sum(), "dtype", tr.snapshot_day.dtype)

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
tt = A.train_targets()
tr = base.merge(tt, on=["household_key","snapshot_day"], how="inner").sort_values(["snapshot_day","household_key"]).reset_index(drop=True)
y = tr.future_spend_4w.values.astype(float)
numcols = [c for c in tr.columns if c not in ("household_key","snapshot_day","future_spend_4w") and pd.api.types.is_numeric_dtype(tr[c])]
numcols = [c for c in numcols if tr[c].nunique(dropna=True)>1]
Z = tr[numcols].astype(float)
med = Z.median(); Z = Z.fillna(med).values
Z = np.clip((Z - Z.mean(0))/(Z.std(0)+1e-9), -8, 8)
ly = np.log1p(y)
def ridge(Zt, yt, alpha):
    p = Zt.shape[1]
    return np.linalg.solve(Zt.T@Zt + alpha*np.eye(p), Zt.T@yt)
def loso(alpha, mode):
    preds = np.zeros(len(tr))
    for d in days:
        te = (tr.snapshot_day.values == d)
        w = ridge(Z[~te], (ly if mode=="log" else y)[~te], alpha)
        p = Z[te]@w
        if mode=="log": p = np.expm1(np.clip(p,0,8))
        preds[te] = p
    return np.mean(np.abs(y-preds))
for alpha in [10,30,100,300,1000,3000]:
    print(alpha, "log", round(loso(alpha,"log"),2), "lin", round(loso(alpha,"lin"),2))

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
tt = A.train_targets()
tr = base.merge(tt, on=["household_key","snapshot_day"], how="inner").sort_values(["snapshot_day","household_key"]).reset_index(drop=True)
y = tr.future_spend_4w.values.astype(float)
numcols = [c for c in tr.columns if c not in ("household_key","snapshot_day","future_spend_4w") and pd.api.types.is_numeric_dtype(tr[c])]
numcols = [c for c in numcols if tr[c].nunique(dropna=True)>1]
Z = tr[numcols].astype(float)
med = Z.median(); Z = Z.fillna(med).values
Z = np.clip((Z - Z.mean(0))/(Z.std(0)+1e-9), -8, 8)
ly = np.log1p(y)
days = sorted(tr.snapshot_day.unique())
def ridge(Zt, yt, alpha):
    p = Zt.shape[1]
    return np.linalg.solve(Zt.T@Zt + alpha*np.eye(p), Zt.T@yt)
def loso(alpha, mode):
    preds = np.zeros(len(tr))
    for d in days:
        te = (tr.snapshot_day.values == d)
        w = ridge(Z[~te], (ly if mode=="log" else y)[~te], alpha)
        p = Z[te]@w
        if mode=="log": p = np.expm1(np.clip(p,0,8))
        preds[te] = p
    return np.mean(np.abs(y-preds))
for alpha in [10,30,100,300,1000,3000]:
    print(alpha, "log", round(loso(alpha,"log"),2), "lin", round(loso(alpha,"lin"),2))

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
rb = A.load_saved("nf_robust.parquet")
newc = ["s_day","k_sin1","k_cos1","k_sin2","k_cos2","r_autocorr","r_dsl","r_gap_mean","r_gap_ratio","g_mean_l1"]
newc = [c for c in newc if c not in base.columns]
m = base.merge(rb[["household_key","snapshot_day"]+newc], on=["household_key","snapshot_day"], how="left")
print(m.shape, "added:", newc)
print(m[newc].isna().mean().round(3))
print("s_day==snapshot_day:", (m.s_day==m.snapshot_day).mean())
path = A.save_table(m, "e020_final")
print(path)