import agent_api as A, pandas as pd, numpy as np
print("snapshot_days:", A.snapshot_days())
print("KEYS:", A.KEYS, "TARGET:", A.TARGET)
names = ["feats_v3","feats_seasonal","feats_e014","oof_e008","pred_e008","pred_e009","pred_e010_blend","pred_seasonal","pred_e014","pred_log1p","pred_e006","pred_e004","pred_e005"]
for n in names:
    try:
        df = A.load_saved(n + ".parquet")
        cols = list(df.columns)
        print("==", n, df.shape, "| ncols:", len(cols))
        print("   cols:", cols[:10], "..." if len(cols)>10 else "")
        if "snapshot_day" in cols:
            vc = df.snapshot_day.value_counts().sort_index()
            print("   days:", dict(vc))
    except Exception as e:
        print("==", n, "ERR", type(e).__name__, str(e)[:120])
tt = A.train_targets()
print("train_targets:", tt.shape, "zero share:", round(float((tt.future_spend_4w==0).mean()),4))
print(tt.future_spend_4w.describe())
v = A.snapshot()
tx = v.table("transactions")
print("transactions@459:", tx.shape)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, itertools
tt = A.train_targets()
# per-snapshot target stats (drift check)
g = tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median",lambda s:(s==0).mean()])
g.columns=["mean","median","zero_share"]
print(g.round(2))

# saved val predictions: correlations & known val MAEs
preds = {}
maes = {"e004":62.332,"e005":62.014,"e006":61.963,"e008":61.770,"e009":61.828,"e010":61.890,"e012":62.177,"e013":61.698,"e014":61.915}
files = {"e004":"pred_e004","e005":"pred_e005","e006":"pred_e006","e008":"pred_e008","e009":"pred_e009",
         "e010":"pred_e010_blend","e012":"pred_log1p","e013":"pred_seasonal","e014":"pred_e014"}
for k,f in files.items():
    d = A.load_saved(f+".parquet")
    preds[k] = d.set_index(["household_key","snapshot_day"]).prediction
P = pd.DataFrame(preds)
C = P.corr(method="spearman")
print("\nSpearman corr of val predictions (min/max off-diag):")
off = C.where(~np.eye(len(C),dtype=bool))
print("min", round(off.min().min(),4), "max", round(off.max().max(),4))
print(C.round(4).to_string())

# dispersion of predictions vs each other (std of preds per row)
print("\nrow-wise std of predictions: mean", round(P.std(axis=1).mean(),2))
print("\nPer-model val MAE (known):", maes)
# best simple pairwise blends by weighted-avg-of-known-MAE lower bound proxy is weak;
# instead check agreement: rows where models disagree most
print("\npred describe e008:", P.e008.describe().round(1).to_dict())


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
oof = A.load_saved("oof_e008.parquet")
m = oof.merge(tt, on=["household_key","snapshot_day"])
m["res_sq"] = m.future_spend_4w - m.oof_sq
m["res_med"] = m.future_spend_4w - m.oof_med
m["res_log"] = m.future_spend_4w - m.oof_log
g = m.groupby("snapshot_day")[["res_sq","res_med","res_log"]].agg(["median","mean"])
print("OOF residuals (y - pred) by snapshot day:")
print(g.round(2).to_string())
print("\nOOF MAE by day:")
print(m.groupby("snapshot_day")[["res_sq","res_med","res_log"]].apply(lambda d: d.abs().mean()).round(2).to_string())
print("\nOverall median residuals:", m[["res_sq","res_med","res_log"]].median().round(2).to_dict())
print("Overall mean residuals:", m[["res_sq","res_med","res_log"]].mean().round(2).to_dict())
# what shift minimizes OOF MAE for each model?
for c in ["res_sq","res_med","res_log"]:
    r = m[c].values
    best = min(np.arange(-15,15.25,0.25), key=lambda s: np.abs(r-s).mean())
    print(c, "MAE", round(np.abs(r).mean(),3), "-> best shift", best, "MAE after", round(np.abs(r-best).mean(),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
oof = A.load_saved("oof_e008.parquet").merge(tt, on=["household_key","snapshot_day"])
r = oof.future_spend_4w - oof.oof_med
print("Median-objective model: median residual by predicted-value decile")
bins = pd.qcut(oof.oof_med, 10, duplicates="drop")
d = pd.DataFrame({"bin":bins, "res":r, "pred":oof.oof_med, "y":oof.future_spend_4w}).groupby("bin", observed=True)
tab = d.agg(n=("res","size"), pred_med=("pred","median"), y_med=("y","median"), res_med=("res","median"), mae=("res", lambda s: s.abs().mean()))
print(tab.round(2).to_string())
print("\nMAE now:", round(np.abs(r).mean(),3))
# apply per-bin median shift
shift = d.res.median()
adj = r - oof.bin.map(shift) if False else r - bins.map(shift)
print("MAE after per-bin median shift:", round(np.abs(adj).mean(),3))
# also by snapshot day x bin for stability check
print("\nres_med median by day:", oof.groupby("snapshot_day").res_med.median().round(2).to_dict())
# zero-activity segment: households with oof_med == 0
z = oof[oof.oof_med==0]
print("\nrows with pred==0:", len(z), "their y median:", z.future_spend_4w.median(), "MAE contribution:", round(np.abs(z.future_spend_4w).mean(),2))
nz = oof[oof.oof_med>0]
print("rows pred>0:", len(nz), "res median:", round((nz.future_spend_4w-nz.oof_med).median(),2))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
oof = A.load_saved("oof_e008.parquet").merge(tt, on=["household_key","snapshot_day"])
r = (oof.future_spend_4w - oof.oof_med).values
bins = pd.qcut(oof.oof_med, 10, duplicates="drop").astype(str).values
oof["bin"] = bins
sh = oof.groupby("bin").res_med.median() if "res_med" in oof else None
oof["res_med"] = r
sh = oof.groupby("bin").res_med.median()
adj = r - pd.Series(bins).map(sh).values
print("MAE:", round(np.abs(r).mean(),3), "-> after per-decile median shift:", round(np.abs(adj).mean(),3))
print("\nshifts:", sh.round(2).to_dict())

# per-decile shrink: pred' = a_b * pred + b_b fitted to minimize MAE? try scaling toward median
oof["pred"] = oof.oof_med
oof["y"] = oof.future_spend_4w
sc = oof.groupby("bin").apply(lambda d: pd.Series({
    "a": np.median(d.y)/np.median(d.pred) if np.median(d.pred)>1 else 1.0}), include_groups=False)
print("\nmedian-ratio per bin:", sc.a.round(2).to_dict())
adj2 = oof.pred.values * pd.Series(bins).map(sc.a).values
print("MAE after per-decile median-ratio scaling:", round(np.abs(adj2-oof.y.values).mean(),3))

# stability of shifts across days
print("\nshift by day x bin (top bins):")
piv = oof.pivot_table(index="snapshot_day", columns="bin", values="res_med", aggfunc="median")
print(piv.round(1).to_string())


# ---- cell ----
import agent_api as A, pandas as pd
f3 = A.load_saved("feats_v3.parquet")
fs = A.load_saved("feats_seasonal.parquet")
print("feats_v3 columns:")
for c in f3.columns: print(" ", c)
print("\nfeats_seasonal columns:", [c for c in fs.columns if c not in ("household_key","snapshot_day")])


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
# check target autocorrelation: y vs lag364_spend (year-ago same window) on train rows
fs = A.load_saved("feats_seasonal.parquet")
m = fs.merge(tt, on=["household_key","snapshot_day"])
m = m[m.snapshot_day>=263]  # where lag364 exists
print("n:", len(m))
print("corr(lag364_spend, y):", np.corrcoef(m.lag364_spend, m.future_spend_4w)[0,1].round(3))
print("corr(spend_28, y):", np.corrcoef(m.spend_28 if 'spend_28' in m else m.lag84_spend, m.future_spend_4w)[0,1].round(3))
# blend check: y_hat = 0.5*lag364 + 0.5*recent28
m["blend"] = 0.5*m.lag364_spend + 0.5*m.lag84_spend
for c in ["lag364_spend","lag84_spend","blend"]:
    print(c, "MAE:", round(np.abs(m[c]-m.future_spend_4w).mean(),2))
# what about combining recent28? need feats_v3
f3 = A.load_saved("feats_v3.parquet")
m2 = m.merge(f3[["household_key","snapshot_day","spend_28","spend_84"]], on=["household_key","snapshot_day"])
for w in [0.0,0.25,0.5,0.75,1.0]:
    b = w*m2.lag364_spend + (1-w)*m2.spend_28
    print(f"blend w={w} (lag364 vs spend28) MAE:", round(np.abs(b-m2.future_spend_4w).mean(),2))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
f3 = A.load_saved("feats_v3.parquet"); fs = A.load_saved("feats_seasonal.parquet")
F = f3.merge(fs, on=["household_key","snapshot_day"], how="left")
tt = A.train_targets()
data = F.merge(tt, on=["household_key","snapshot_day"]).sort_values(["snapshot_day","household_key"]).reset_index(drop=True)
feat_cols = [c for c in F.columns if c not in ("household_key","snapshot_day")]
X = data[feat_cols].astype(float).values
y = data.future_spend_4w.values
day = data.snapshot_day.values

def mk(obj, n, lr, seed=7):
    return xgb.XGBRegressor(objective=obj, n_estimators=n, learning_rate=lr, max_depth=7,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        n_jobs=-1, random_state=seed)

oof_days=[347,375,403,431]
t0=time.time(); out={}
cfgs = [("med", lambda: mk("reg:absoluteerror",1200,0.03), None),
        ("sq",  lambda: mk("reg:squarederror",2000,0.02), None),
        ("med_w112", lambda: mk("reg:absoluteerror",1200,0.03), 112.),
        ("med_w224", lambda: mk("reg:absoluteerror",1200,0.03), 224.)]
oof_store = {}
for name, mkf, H in cfgs:
    oof = np.full(len(data), np.nan)
    for d in oof_days:
        tr = day < d; te = day == d
        w = None
        if H is not None:
            ref = day[tr].max()
            w = 0.5 ** ((ref - day[tr]) / H)
        m = mkf(); m.fit(X[tr], y[tr], sample_weight=w)
        oof[te] = m.predict(X[te])
    mae = np.nanmean(np.abs(oof-y))
    out[name]=round(mae,3); oof_store[name]=oof
    print(name, "OOF MAE:", round(mae,3), f"({time.time()-t0:.0f}s)", flush=True)

# save oof for stacking later
odf = data[["household_key","snapshot_day"]].copy()
for k,v in oof_store.items(): odf["oof_"+k]=v
p = A.save_table(odf, "oof_harness.parquet")
print("saved:", p, "| MAEs:", out)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
from scipy.optimize import minimize
tt = A.train_targets()
o1 = A.load_saved("oof_e008.parquet").merge(tt, on=["household_key","snapshot_day"])
o2 = A.load_saved("oof_harness.parquet").merge(tt, on=["household_key","snapshot_day"])
o = o1.merge(o2[["household_key","snapshot_day","oof_med","oof_sq","oof_med_w112"]], on=["household_key","snapshot_day"])
y = o.future_spend_4w.values
cols = ["oof_med","oof_sq","oof_log","oof_med","oof_sq","oof_med_w112"]
cols[3] = "oof_med_2"; cols[4]="oof_sq_2"; o["oof_med_2"]=o.oof_med_y; o["oof_sq_2"]=o.oof_sq_y
cols = ["oof_med_x","oof_sq_x","oof_log","oof_med_2","oof_sq_2","oof_med_w112"]
o = o.rename(columns={"oof_med_x":"oof_med_x","oof_sq_x":"oof_sq_x"})
# figure out actual col names
print([c for c in o.columns])


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
from scipy.optimize import minimize
tt = A.train_targets()
o1 = A.load_saved("oof_e008.parquet").merge(tt, on=["household_key","snapshot_day"])
o2 = A.load_saved("oof_harness.parquet").merge(tt, on=["household_key","snapshot_day"])
o = o1.merge(o2[["household_key","snapshot_day","oof_med_y","oof_sq_y","oof_med_w112"]],
              on=["household_key","snapshot_day"], suffixes=("_x","_y"))
o = o.rename(columns={"oof_med_x":"m1","oof_sq_x":"s1","oof_log":"l1","oof_med_y":"m2","oof_sq_y":"s2","oof_med_w112":"w2"})
y = o.future_spend_4w.values
M = o[["m1","s1","l1","m2","s2","w2"]].values
def mae(w, M, y):
    p = M @ w
    return np.abs(p-y).mean()
w0 = np.zeros(M.shape[1]); w0[0]=1
res = minimize(mae, np.full(6,1/6), args=(M,y), method="Nelder-Mead",
               options={"maxiter":4000,"xatol":1e-3,"fatol":1e-3})
w = res.x
print("weights:", dict(zip(["m1","s1","l1","m2","s2","w2"], w.round(3))))
print("stacked OOF MAE:", round(mae(w,M,y),3), " (single med1:", round(np.abs(o.m1-y).mean(),3), ")")
# simpler: nonneg least squares on MAE via repeated Nelder-Mead with clipping
from scipy.optimize import nnls
w2,_ = nnls(M, y)
print("NNLS weights:", w2.round(3), "MAE:", round(np.abs(M@w2-y).mean(),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
from scipy.optimize import minimize, nnls
tt = A.train_targets()
o1 = A.load_saved("oof_e008.parquet").merge(tt, on=["household_key","snapshot_day"])
o2 = A.load_saved("oof_harness.parquet").merge(tt, on=["household_key","snapshot_day"])
o = o1.merge(o2[["household_key","snapshot_day","oof_med","oof_sq","oof_med_w112"]],
              on=["household_key","snapshot_day"], suffixes=("_x","_y"))
o = o.rename(columns={"oof_med_x":"m1","oof_sq_x":"s1","oof_log":"l1","oof_med_y":"m2","oof_sq_y":"s2","oof_med_w112":"w2"})
y = o.future_spend_4w.values
M = o[["m1","s1","l1","m2","s2","w2"]].values
def mae(w): return np.abs(M@w - y).mean()
res = minimize(mae, np.full(6,1/6), method="Nelder-Mead", options={"maxiter":6000,"fatol":1e-4})
w = np.clip(res.x, 0, None); w = w/w.sum()
print("NM weights:", dict(zip(["m1","s1","l1","m2","s2","w2"], w.round(3))), "MAE:", round(mae(w),3))
w2,_ = nnls(M, y)
print("NNLS weights:", w2.round(3), "sum", w2.sum().round(3), "MAE:", round(mae(w2),3))
print("single m1 MAE:", round(np.abs(o.m1-y).mean(),3))
# per-decile shift on the stack? later. Also check shift stability: fit weights on day<=375, eval on 403/431
tr = o.snapshot_day<=375
res2 = minimize(lambda w: np.abs(M[tr]@w - y[tr]).mean(), np.full(6,1/6), method="Nelder-Mead", options={"maxiter":6000})
w_early = np.clip(res2.x,0,None); w_early/=w_early.sum()
print("early-fit weights:", w_early.round(3), "-> late MAE:", round(mae(w_early),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
tt = A.train_targets().sort_values(["household_key","snapshot_day"])
# expanding past-target stats per household (only targets whose window ends <= current snapshot day)
tt["win_end"] = tt.snapshot_day + 28
rows = []
for hk, g in tt.groupby("household_key"):
    ys = g.future_spend_4w.values; ds = g.snapshot_day.values; we = g.win_end.values
    for i in range(len(g)):
        past = (we[:i] <= ds[i])  # windows fully in the past
        pv = ys[:i][past]
        rows.append((hk, ds[i], len(pv), np.mean(pv) if len(pv) else np.nan,
                     np.median(pv) if len(pv) else np.nan,
                     np.std(pv) if len(pv)>1 else np.nan,
                     ys[i-1] if i>0 and we[i-1]<=ds[i] else np.nan))
pt = pd.DataFrame(rows, columns=["household_key","snapshot_day","n_past_tg","mean_past_tg","med_past_tg","std_past_tg","lag1_tg"])
print(pt.shape, pt.head(3).to_string())
print("coverage on rows:", pt.n_past_tg.notna().mean().round(3), "| mean n_past:", pt.n_past_tg.mean().round(2))
p = A.save_table(pt, "past_targets.parquet"); print("saved", p)
# MAE of simple past-median predictor on train rows with >=4 past targets
sub = pt[pt.n_past_tg>=4].merge(tt, on=["household_key","snapshot_day"])
print("rows:", len(sub), "MAE med_past_tg:", round(np.abs(sub.med_past_tg-sub.future_spend_4w).mean(),2),
      "MAE mean_past_tg:", round(np.abs(sub.mean_past_tg-sub.future_spend_4w).mean(),2))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
f3 = A.load_saved("feats_v3.parquet"); fs = A.load_saved("feats_seasonal.parquet")
pt = A.load_saved("past_targets.parquet")
F = f3.merge(fs, on=["household_key","snapshot_day"], how="left").merge(pt, on=["household_key","snapshot_day"], how="left")
tt = A.train_targets()
data = F.merge(tt, on=["household_key","snapshot_day"]).sort_values(["snapshot_day","household_key"]).reset_index(drop=True)
feat_cols = [c for c in F.columns if c not in ("household_key","snapshot_day")]
X = data[feat_cols].astype(float).values
y = data.future_spend_4w.values
day = data.snapshot_day.values
oof_days=[347,375,403,431]
t0=time.time(); oof = np.full(len(data), np.nan)
for d in oof_days:
    tr = day < d; te = day == d
    m = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=1200, learning_rate=0.03,
        max_depth=7, min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=-1, random_state=7)
    m.fit(X[tr], y[tr])
    oof[te] = m.predict(X[te])
    print(d, "done", f"{time.time()-t0:.0f}s", flush=True)
mae = np.nanmean(np.abs(oof-y))
print("med + past-target feats OOF MAE:", round(mae,3), "(harness med was 62.499)")
odf = data[["household_key","snapshot_day"]].copy(); odf["oof"]=oof
A.save_table(odf, "oof_pt.parquet")


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
f3 = A.load_saved("feats_v3.parquet"); fs = A.load_saved("feats_seasonal.parquet")
pt = A.load_saved("past_targets.parquet")
F = f3.merge(fs, on=["household_key","snapshot_day"], how="left").merge(pt, on=["household_key","snapshot_day"], how="left")
tt = A.train_targets()
data = F.merge(tt, on=["household_key","snapshot_day"]).sort_values(["snapshot_day","household_key"]).reset_index(drop=True)
feat_cols = [c for c in F.columns if c not in ("household_key","snapshot_day")]
X = data[feat_cols].astype(float).values
y = data.future_spend_4w.values
day = data.snapshot_day.values
# train on all train rows, predict val
t0=time.time()
m = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=1200, learning_rate=0.03,
    max_depth=7, min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", n_jobs=-1, random_state=7)
m.fit(X, y)
val = F[F.snapshot_day>=459].sort_values(["snapshot_day","household_key"])
Xv = val[feat_cols].astype(float).values
val = val.assign(prediction=m.predict(Xv))
p = A.save_table(val[["household_key","snapshot_day","prediction"]], "pred_pt.parquet")
print("saved", p, f"{time.time()-t0:.0f}s")
