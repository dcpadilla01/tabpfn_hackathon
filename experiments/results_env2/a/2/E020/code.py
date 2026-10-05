import agent_api as A
import pandas as pd, numpy as np

names = ["feats_all_e016","feats_v3","feats_seasonal","oof_e016_cv","oof_e008","oof_harness","oof_pt","oof_tw","past_targets",
"pred_e006","pred_e008","pred_e009","pred_e010","pred_e010_blend","pred_e014","pred_e016","pred_e018","pred_log1p","pred_prof","pred_pt","pred_seasonal"]
for nm in names:
    try:
        df = A.load_saved(nm + ".parquet")
        nonnum = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.number)]
        print(f"{nm:20s} {str(df.shape):14s} cols={list(df.columns)[:12]} nonnum={nonnum[:4]}")
    except Exception as e:
        print(nm, "ERR", type(e).__name__, str(e)[:80])

print()
print("snapshot_days:", A.snapshot_days())
tt = A.train_targets()
print("train_targets", tt.shape, tt.columns.tolist())
print(tt.groupby("snapshot_day")["future_spend_4w"].agg(["count","mean","median"]).round(2))

for nm in ["oof_e016_cv","oof_e008","oof_harness","oof_pt","oof_tw"]:
    df = A.load_saved(nm + ".parquet")
    print("\n--", nm, df.columns.tolist())
    print(df.head(2))
    if "snapshot_day" in df.columns:
        print("days:", sorted(df.snapshot_day.unique()))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet")
tt = A.train_targets()
print("oof rows", len(oof), "tt rows", len(tt))
m = oof.merge(tt, on=["household_key","snapshot_day"], how="left")
print("y mismatch:", (m.y_x.fillna(-1) != m.y_y.fillna(-1)).sum() if "y_y" in m.columns else "cols:", m.columns.tolist())

learners = ["med_v3","med_all","hgbq_v3","hgbq_all"]
y = oof["y"].values
def mae(p): return np.mean(np.abs(p - y))
for c in learners:
    print(f"{c:9s} OOF MAE {mae(oof[c].values):.3f}")
# find E016 blend weights approx: try grid
best=(1e9,None)
import itertools
for w in itertools.product(np.arange(0,1.01,0.05), repeat=3):
    if w[0]+w[1]+w[2] > 1.001: continue
    w4 = 1 - w[0]-w[1]-w[2]
    if w4 < -0.001: continue
    p = w[0]*oof.med_v3 + w[1]*oof.med_all + w[2]*oof.hgbq_v3 + w4*oof.hgbq_all
    mm = mae(p.values)
    if mm < best[0]: best=(mm,w+(round(w4,2),))
print("best fixed-weight blend on OOF:", best)

# per-snapshot MAE of best blend
w = best[1]
p = w[0]*oof.med_v3 + w[1]*oof.med_all + w[2]*oof.hgbq_v3 + w[3]*oof.hgbq_all
oof["blend"] = p
print(oof.groupby("snapshot_day").apply(lambda g: pd.Series({"n":len(g),"y_med":g.y.median(),"mae_blend":np.mean(np.abs(g.blend-g.y)),"mae_medv3":np.mean(np.abs(g.med_v3-g.y))}), include_groups=False).round(2))

# NaN check other oofs
for nm in ["oof_tw","oof_pt","oof_harness"]:
    d = A.load_saved(nm+".parquet")
    col = [c for c in d.columns if c.startswith("oof")][0]
    print(nm, "NaN by day:", d.groupby("snapshot_day")[col].apply(lambda s: s.isna().mean().round(2)).to_dict())


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, itertools

oof = A.load_saved("oof_e016_cv.parquet")
tt = A.train_targets()
oof = oof.merge(tt.rename(columns={"future_spend_4w":"y_tt"}), on=["household_key","snapshot_day"], how="left")
print("y vs y_tt mismatch:", (oof.y.fillna(-9)!=oof.y_tt.fillna(-9)).sum())

tw = A.load_saved("oof_tw.parquet")
oof = oof.merge(tw, on=["household_key","snapshot_day"], how="left")
y = oof.y.values
print("oof_tw MAE:", np.mean(np.abs(oof.oof_tw.values - y)).round(3))

# correlations among val preds
preds = {}
for nm in ["pred_e006","pred_e008","pred_e009","pred_e010","pred_e010_blend","pred_e014","pred_e016","pred_log1p","pred_prof","pred_pt","pred_seasonal","pred_e018"]:
    preds[nm] = A.load_saved(nm+".parquet").set_index(["household_key","snapshot_day"])["prediction"]
P = pd.DataFrame(preds)
print("\nval pred correlations:")
print(P.corr().round(3))
print("\nval pred describe:")
print(P.describe().loc[["mean","50%","std"]].round(2))

# OOF correlations of the 4 learners + tw
L = ["med_v3","med_all","hgbq_v3","hgbq_all","oof_tw"]
print("\nOOF corr:")
print(oof[L].corr().round(3))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet")
tt = A.train_targets()
print("oof dtypes:", oof.dtypes.to_dict())
print("tt dtypes:", tt.dtypes.to_dict())
print("oof y NaN:", oof.y.isna().sum(), " y describe:", oof.y.describe().round(2).to_dict())

m = oof.merge(tt, on=["household_key","snapshot_day"], how="left", suffixes=("","_tt"))
print("merged rows:", len(m))
m["diff"] = m.y - m.future_spend_4w
print("rows equal (|diff|<1e-6):", (m["diff"].abs()<1e-6).sum(), " NaN tt:", m.future_spend_4w.isna().sum())
print(m[["household_key","snapshot_day","y","future_spend_4w","diff"]].head(10))
print("rows where diff!=0 by day:")
print(m.assign(ne=m["diff"].abs()>1e-6).groupby("snapshot_day")["ne"].agg(["sum","count"]))
# maybe y is a shifted/rolled target? check correlation
print("corr(y, tt):", m[["y","future_spend_4w"]].corr().iloc[0,1].round(4))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, itertools

oof = A.load_saved("oof_e016_cv.parquet").drop(columns=["y"])
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
oof = oof.merge(tt, on=["household_key","snapshot_day"], how="left")
tw = A.load_saved("oof_tw.parquet")
oof = oof.merge(tw, on=["household_key","snapshot_day"], how="left")
days = sorted(oof.snapshot_day.unique())
L = ["med_v3","hgbq_v3","hgbq_all","oof_tw"]  # drop med_all (worst, corr .997 w med_v3)
X = oof[L].values; y = oof.y.values

def loso_eval(fit_predict):
    """fit_predict(Xtr,ytr) -> predict(Xval); returns mean MAE over LOSO folds + per-fold"""
    maes=[]
    for d in days:
        tr = oof.snapshot_day.values != d; va = ~tr
        p = fit_predict(X[tr], y[tr], X[va])
        maes.append(np.mean(np.abs(p - y[va])))
    return np.mean(maes), maes

# 1) fixed E016 weights as reference (fit=identity)
def fp_fixed(w):
    def f(Xtr,ytr,Xva): return Xva @ np.array(w)
    return f
w0 = [0.4/0.65, 0.25/0.65, 0.35/0.65, 0.0]
m0,_ = loso_eval(fp_fixed(w0)); print("E016-style fixed weights LOSO MAE:", round(m0,4))

# 2) LOSO grid over weights incl oof_tw
def grid_search(step=0.1, allow_neg=False):
    best=(1e9,None)
    ws = np.arange(0,1.0001,step)
    for w1 in ws:
        for w2 in ws:
            for w3 in ws:
                w4 = 1-w1-w2-w3
                if w4 < -1e-9: continue
                for w5 in ([0.0] if False else [0.0,0.05,0.1,0.15,0.2]):
                    s = w1+w2+w3+w5
                    if s > 1.0001: continue
                    w = np.array([w1,w2,w3,w4*(1-w5) if s<=1 else 0, w5])
                    if abs(w.sum()-1)>1e-6: continue
                    mm,_ = loso_eval(fp_fixed(w))
                    if mm < best[0]: best=(mm,w)
    return best
# too slow with loso inside; use vectorized eval instead
def mae_w(w):
    p = X @ w
    return np.mean(np.abs(p-y))
best=(1e9,None)
for w1 in np.arange(0,1.001,0.1):
    for w2 in np.arange(0,1.001,0.1):
        for w5 in [0,0.05,0.1,0.15,0.2,0.3]:
            for w3 in np.arange(0,1.001,0.1):
                w4 = 1-w1-w2-w3-w5
                if w4 < -1e-9: continue
                w = np.array([w1,w2,w3,w4,w5])
                mm = mae_w(w)
                if mm < best[0]: best=(mm.round(4), w.round(2))
print("full-OOF best 5-learner weights:", best)

# 3) LOSO with per-fold grid on 5 learners (coarse, vectorized per fold)
def fp_grid(Xtr,ytr,Xva):
    best=(1e9,None)
    for w1 in np.arange(0,1.001,0.1):
        for w2 in np.arange(0,1.001,0.1):
            for w5 in [0,0.05,0.1,0.15,0.2]:
                for w3 in np.arange(0,1.001,0.1):
                    w4=1-w1-w2-w3-w5
                    if w4<-1e-9: continue
                    w=np.array([w1,w2,w3,w4,w5])
                    mm=np.mean(np.abs(Xtr@w-ytr))
                    if mm<best[0]: best=(mm,w)
    return Xva@best[1]
m3,f3 = loso_eval(fp_grid); print("LOSO per-fold grid (5 learners):", round(m3,4), [round(x,2) for x in f3])


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, itertools

oof = A.load_saved("oof_e016_cv.parquet").drop(columns=["y"])
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
oof = oof.merge(tt, on=["household_key","snapshot_day"], how="left")
tw = A.load_saved("oof_tw.parquet")
oof = oof.merge(tw, on=["household_key","snapshot_day"], how="left")
days = sorted(oof.snapshot_day.unique())
L = ["med_v3","hgbq_v3","hgbq_all","oof_tw"]
X = oof[L].values; y = oof.y.values

def loso_eval(fit_predict):
    maes=[]
    for d in days:
        tr = oof.snapshot_day.values != d; va = ~tr
        p = fit_predict(X[tr], y[tr], X[va])
        maes.append(np.mean(np.abs(p - y[va])))
    return np.mean(maes), maes

# reference: E016 fixed weights [med_v3 .4, hgbq_v3 .3, hgbq_all .3, tw 0]
m0,_ = loso_eval(lambda a,b,c: c @ np.array([0.4,0.3,0.3,0.0]))
print("E016 weights LOSO MAE:", round(m0,4))
print("full-OOF MAE E016 weights:", round(np.mean(np.abs(X@np.array([0.4,0.3,0.3,0.0])-y)),4))

# full-OOF grid over 5... actually 4 learners incl tw
best=(1e9,None)
for w1 in np.arange(0,1.001,0.05):
    for w2 in np.arange(0,1.001,0.05):
        for w4 in [0,0.05,0.1,0.15,0.2,0.25,0.3]:
            w3 = 1-w1-w2-w4
            if w3 < -1e-9: continue
            w = np.array([w1,w2,w3,w4])
            mm = np.mean(np.abs(X@w-y))
            if mm<best[0]: best=(round(mm,4), w.round(2))
print("full-OOF best 4-learner weights:", best)

# LOSO per-fold grid
def fp_grid(Xtr,ytr,Xva):
    best=(1e9,None)
    for w1 in np.arange(0,1.001,0.05):
        for w2 in np.arange(0,1.001,0.05):
            for w4 in [0,0.05,0.1,0.15,0.2,0.25,0.3]:
                w3=1-w1-w2-w4
                if w3<-1e-9: continue
                w=np.array([w1,w2,w3,w4])
                mm=np.mean(np.abs(Xtr@w-ytr))
                if mm<best[0]: best=(mm,w)
    return Xva@best[1]
m3,f3 = loso_eval(fp_grid); print("LOSO per-fold grid:", round(m3,4), [round(x,2) for x in f3])


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np
from sklearn.isotonic import IsotonicRegression

oof = A.load_saved("oof_e016_cv.parquet").drop(columns=["y"])
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
oof = oof.merge(tt, on=["household_key","snapshot_day"], how="left")
tw = A.load_saved("oof_tw.parquet")
oof = oof.merge(tw, on=["household_key","snapshot_day"], how="left")
days = sorted(oof.snapshot_day.unique())
L = ["med_v3","hgbq_v3","hgbq_all","oof_tw"]
X = oof[L].values; y = oof.y.values; D = oof.snapshot_day.values
W0 = np.array([0.4,0.3,0.3,0.0])
blend = X @ W0

def loso(fit_predict):
    maes=[]
    for d in days:
        tr = D != d; va = ~tr
        maes.append(np.mean(np.abs(fit_predict(X[tr],y[tr],D[tr],X[va],D[va]) - y[va])))
    return round(np.mean(maes),4)

# C) isotonic calibration of blend (fit on train folds, apply to val fold)
def fp_iso(Xtr,ytr,Dtr,Xva,Dva):
    b = Xtr@W0
    iso = IsotonicRegression(out_of_bounds="clip").fit(b, ytr)
    return iso.predict(Xva@W0)
print("C isotonic on blend LOSO:", loso(fp_iso))

# J) linear recalibration a*blend+b
def fp_lin(Xtr,ytr,Dtr,Xva,Dva):
    b = Xtr@W0
    A_ = np.vstack([b, np.ones_like(b)]).T
    coef, *_ = np.linalg.lstsq(A_, ytr, rcond=None)
    return Xva@W0*coef[0]+coef[1]
print("J linear recalib LOSO:", loso(fp_lin))

# K) CDF matching: map blend ranks to train-target quantiles
def fp_cdf(Xtr,ytr,Dtr,Xva,Dva):
    b = Xtr@W0
    qs = np.quantile(b, np.linspace(0.01,0.99,99))
    qy = np.quantile(ytr, np.linspace(0.01,0.99,99))
    return np.interp(Xva@W0, qs, qy)
print("K CDF-match LOSO:", loso(fp_cdf))

# H) median of learners
print("H median-of-4 LOSO:", loso(lambda a,b,c,d,e: np.median(np.vstack([d[:,0],d[:,1],d[:,2],d[:,3]]),axis=0)))

# I) rank averaging -> map back with train y quantiles
def fp_rank(Xtr,ytr,Dtr,Xva,Dva):
    r = np.mean([ (pd.Series(Xva[:,i]).rank(pct=True).values) for i in range(4)],axis=0)
    qy = np.quantile(ytr, np.linspace(0,1,101))
    return np.interp(r, np.linspace(0,1,101), qy)
print("I rank-avg LOSO:", loso(fp_rank))

# G) weights fit on recent snapshots only (last 5 before val fold), fallback to W0
def fp_recent(Xtr,ytr,Dtr,Xva,Dva,K=5):
    recent = np.sort(np.unique(Dtr))[-K:]
    m = np.isin(Dtr, recent)
    best=(1e9,W0)
    for w1 in np.arange(0,1.001,0.1):
        for w2 in np.arange(0,1.001,0.1):
            w4 = 1-w1-w2
            if w4<-1e-9: continue
            w=np.array([w1,w2,0.0,w4])
            mm=np.mean(np.abs(Xtr[m]@w-ytr[m]))
            if mm<best[0]: best=(mm,w)
    return Xva@best[1]
for K in [3,5,8]:
    print(f"G recent-{K} LOSO:", loso(lambda a,b,c,d,e,K=K: fp_recent(a,b,c,d,e,K)))

# naive reference: last-28d spend (spend_all is what window? check) -- just report blend MAE per fold
print("reference W0 LOSO:", loso(lambda a,b,c,d,e: e@W0))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np
from scipy.optimize import nnls

oof = A.load_saved("oof_e016_cv.parquet").drop(columns=["y"])
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
oof = oof.merge(tt, on=["household_key","snapshot_day"], how="left")
tw = A.load_saved("oof_tw.parquet")
oof = oof.merge(tw, on=["household_key","snapshot_day"], how="left")
days = sorted(oof.snapshot_day.unique())
L = ["med_v3","hgbq_v3","hgbq_all","oof_tw"]
X = oof[L].values; y = oof.y.values; D = oof.snapshot_day.values
W0 = np.array([0.4,0.3,0.3,0.0])
blend = X @ W0
hh = oof.household_key.values

# household past realized targets (fully observed before day d): mean over tt rows with snap+28<=d
pt_mean = np.full(len(oof), np.nan)
tt_g = tt.groupby("household_key").apply(lambda g: g[["snapshot_day","y"]].values, include_groups=False)
for i in range(len(oof)):
    d = D[i]
    arr = tt_g.get(hh[i])
    if arr is None: continue
    m = arr[:,0] + 28 <= d
    if m.any(): pt_mean[i] = arr[m,1].mean()

# lagged OOF blend (same household, previous snapshot)
oof_s = oof.sort_values(["household_key","snapshot_day"])
oof_s["blend"] = oof_s[L].values @ W0
oof_s["lag_blend"] = oof_s.groupby("household_key")["blend"].shift(1)
oof_s["lag_y"] = oof_s.groupby("household_key")["y"].shift(1)
lag_blend = oof_s["lag_blend"].reindex(oof.index).values if oof.index.equals(oof_s.index) else oof_s["lag_blend"].values
# reindex safely by position after sorting back
oof_s = oof_s.sort_index()
lag_blend = oof_s["lag_blend"].values
lag_y = oof_s["lag_y"].values

def loso(fit_predict):
    maes=[]
    for d in days:
        tr = D != d; va = ~tr
        maes.append(np.mean(np.abs(fit_predict(tr, va) - y[va])))
    return round(np.mean(maes),4)

# N) blend with household past-target mean
for a in [0.05,0.1,0.2,0.3]:
    f = lambda tr,va,a=a: (1-a)*blend[va] + a*np.nan_to_num(pt_mean[va], nan=blend[va])
    print(f"N a={a} LOSO:", loso(f))

# O) blend with lagged OOF blend
for a in [0.05,0.1,0.2]:
    f = lambda tr,va,a=a: (1-a)*blend[va] + a*np.nan_to_num(lag_blend[va], nan=blend[va])
    print(f"O lagOOF a={a} LOSO:", loso(f))

# P) blend with lagged realized target
for a in [0.05,0.1,0.2]:
    f = lambda tr,va,a=a: (1-a)*blend[va] + a*np.nan_to_num(lag_y[va], nan=blend[va])
    print(f"P lagY a={a} LOSO:", loso(f))

# Q) NNLS weights LOSO
def fp_nnls(tr,va):
    w,_ = nnls(X[tr], y[tr])
    if w.sum()==0: w=W0
    else: w = w/w.sum()
    return X[va]@w
print("Q nnls LOSO:", loso(fp_nnls))

# R) segment weights: low/high blend halves
def fp_seg(tr,va):
    b_tr = blend[tr]
    med = np.median(b_tr)
    for seg in [0,1]:
        m = (b_tr<med) if seg==0 else (b_tr>=med)
        best=(1e9,W0)
        for w1 in np.arange(0,1.001,0.1):
            for w2 in np.arange(0,1.001,0.1):
                w4=1-w1-w2
                if w4<-1e-9: continue
                w=np.array([w1,w2,0.0,w4])
                mm=np.mean(np.abs(X[tr][m]@w-y[tr][m]))
                if mm<best[0]: best=(mm,w)
        if seg==0: wlo=best[1]
    b_va = blend[va]
    return np.where(b_va<med, X[va]@wlo, X[va]@best[1])
print("R segment weights LOSO:", loso(fp_seg))

# S) winsorize blend at caps
for cap in [600,800,1000,1200,1500]:
    f = lambda tr,va,cap=cap: np.minimum(blend[va], cap)
    print(f"S cap={cap} LOSO:", loso(f))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, time

oof = A.load_saved("oof_e016_cv.parquet").drop(columns=["y"])
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
oof = oof.merge(tt, on=["household_key","snapshot_day"], how="left")
W0 = np.array([0.4,0.3,0.3,0.0])
oof["blend"] = oof[["med_v3","hgbq_v3","hgbq_all","oof_tw"]].values @ W0
y = oof.y.values; b = oof.blend.values; err = np.abs(b-y)

# error decomposition by y bucket
cuts = [0,1,10,50,100,200,500,1e9]
lab = ["y=0","0-10","10-50","50-100","100-200","200-500","500+"]
yb = pd.cut(y, bins=cuts, labels=lab, right=False)
df = pd.DataFrame({"yb":yb,"err":err,"b":b,"y":y})
g = df.groupby("yb", observed=True).apply(lambda t: pd.Series({
    "n":len(t), "share_of_MAE": t.err.sum()/err.sum(), "MAE": t.err.mean(),
    "pred_med": t.b.median(), "y_med": t.y.median()}), include_groups=False)
print(g.round(2))

# seed-averaging test on fold day=431: train XGB quantile on feats_v3, 3 seeds
fv = A.load_saved("feats_v3.parquet")
feats = [c for c in fv.columns if c not in ("household_key","snapshot_day")]
tr = fv.snapshot_day.values != 431
va = fv.snapshot_day.values == 431
ytr = tt.set_index(["household_key","snapshot_day"]).reindex(
    pd.MultiIndex.from_frame(fv[["household_key","snapshot_day"]])).values
ytr = ytr.ravel()
import xgboost as xgb
dtr = xgb.DMatrix(fv.loc[tr,feats], label=ytr[tr])
dva = xgb.DMatrix(fv.loc[va,feats])
ps=[]
t0=time.time()
for seed in [7, 42, 2024]:
    m = xgb.train({"objective":"reg:quantileerror","quantile_alpha":0.5,"max_depth":6,
                   "learning_rate":0.03,"n_estimators":1200 if False else None,
                   "eval_metric":"mae","seed":seed,"tree_method":"hist"},
                  dtr, num_boost_round=1200, verbose_eval=False)
    ps.append(m.predict(dva))
    print("seed",seed,"fold431 MAE:", round(np.mean(np.abs(ps[-1]-ytr[va])),3), f"({time.time()-t0:.0f}s)")
print("avg of 3 seeds MAE:", round(np.mean(np.abs(np.mean(ps,axis=0)-ytr[va])),3))
print("med_v3 OOF fold431 MAE:", round(np.mean(np.abs(oof.loc[oof.snapshot_day==431,"med_v3"]-y[oof.snapshot_day.values==431])),3))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, time

oof = A.load_saved("oof_e016_cv.parquet").drop(columns=["y"])
oof = oof.merge(A.load_saved("oof_tw.parquet"), on=["household_key","snapshot_day"], how="left")
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
oof = oof.merge(tt, on=["household_key","snapshot_day"], how="left")
W0 = np.array([0.4,0.3,0.3,0.0])
oof["blend"] = oof[["med_v3","hgbq_v3","hgbq_all","oof_tw"]].values @ W0
y = oof.y.values; b = oof.blend.values; err = np.abs(b-y)

cuts = [0,1,10,50,100,200,500,1e9]
lab = ["y=0","0-10","10-50","50-100","100-200","200-500","500+"]
yb = pd.cut(y, bins=cuts, labels=lab, right=False)
df = pd.DataFrame({"yb":yb,"err":err,"b":b,"y":y})
g = df.groupby("yb", observed=True).apply(lambda t: pd.Series({
    "n":len(t), "share_of_MAE": t.err.sum()/err.sum(), "MAE": t.err.mean(),
    "pred_med": t.b.median(), "y_med": t.y.median()}), include_groups=False)
print(g.round(2))

# seed-averaging test on fold day=431: train XGB quantile on feats_v3, 3 seeds
fv = A.load_saved("feats_v3.parquet")
feats = [c for c in fv.columns if c not in ("household_key","snapshot_day")]
tr = fv.snapshot_day.values != 431
va = fv.snapshot_day.values == 431
ytr = tt.set_index(["household_key","snapshot_day"]).reindex(
    pd.MultiIndex.from_frame(fv[["household_key","snapshot_day"]])).values.ravel()
import xgboost as xgb
dtr = xgb.DMatrix(fv.loc[tr,feats], label=ytr[tr])
dva = xgb.DMatrix(fv.loc[va,feats])
ps=[]
t0=time.time()
for seed in [7, 42, 2024]:
    m = xgb.train({"objective":"reg:quantileerror","quantile_alpha":0.5,"max_depth":6,
                   "learning_rate":0.03,"seed":seed,"tree_method":"hist"},
                  dtr, num_boost_round=1200, verbose_eval=False)
    ps.append(m.predict(dva))
    print("seed",seed,"fold431 MAE:", round(np.mean(np.abs(ps[-1]-ytr[va])),3), f"({time.time()-t0:.0f}s)")
print("avg of 3 seeds MAE:", round(np.mean(np.abs(np.mean(ps,axis=0)-ytr[va])),3))
print("med_v3 OOF fold431 MAE:", round(np.mean(np.abs(oof.loc[oof.snapshot_day==431,"med_v3"]-y[oof.snapshot_day.values==431])),3))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, time
import xgboost as xgb

fv = A.load_saved("feats_v3.parquet")
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
fv = fv.merge(tt, on=["household_key","snapshot_day"], how="left")
print("rows:", len(fv), "y NaN (val rows):", fv.y.isna().sum())
feats = [c for c in fv.columns if c not in ("household_key","snapshot_day","y")]
fv_tr = fv[fv.y.notna()]

tr = fv_tr.snapshot_day.values != 431
va = fv_tr.snapshot_day.values == 431
dtr = xgb.DMatrix(fv_tr.loc[tr,feats], label=fv_tr.loc[tr,"y"].values)
dva = xgb.DMatrix(fv_tr.loc[va,feats])
ps=[]; t0=time.time()
for seed in [7, 42, 2024]:
    m = xgb.train({"objective":"reg:quantileerror","quantile_alpha":0.5,"max_depth":6,
                   "learning_rate":0.03,"seed":seed,"tree_method":"hist"},
                  dtr, num_boost_round=1200, verbose_eval=False)
    ps.append(m.predict(dva))
    print("seed",seed,"fold431 MAE:", round(np.mean(np.abs(ps[-1]-fv_tr.loc[va,"y"].values)),3), f"({time.time()-t0:.0f}s)")
print("avg of 3 seeds MAE:", round(np.mean(np.abs(np.mean(ps,axis=0)-fv_tr.loc[va,"y"].values)),3))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, time
import xgboost as xgb

oof = A.load_saved("oof_e016_cv.parquet").drop(columns=["y"])
oof = oof.merge(A.load_saved("oof_tw.parquet"), on=["household_key","snapshot_day"], how="left")
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
oof = oof.merge(tt, on=["household_key","snapshot_day"], how="left")
fv = A.load_saved("feats_v3.parquet")
fs = A.load_saved("feats_seasonal.parquet")
Xf = fv.merge(fs, on=["household_key","snapshot_day"], how="left")
key_feats = ["spend_all","spend_28","spend_56","spend_84","baskets_28","spend_max_basket_all","lag364_spend","lag308_spend"]
oof = oof.merge(Xf[["household_key","snapshot_day"]+key_feats], on=["household_key","snapshot_day"], how="left")
days = sorted(oof.snapshot_day.unique())
L = ["med_v3","hgbq_v3","hgbq_all","oof_tw"]
W0 = np.array([0.4,0.3,0.3,0.0])
oof["blend"] = oof[L].values @ W0
y = oof.y.values; D = oof.snapshot_day.values; blend = oof.blend.values

# cheap: piecewise multiplicative shrink/expand by blend bucket, median-objective per bucket, LOSO
def fp_pw(tr,va):
    b_tr = blend[tr]; y_tr = y[tr]
    edges = [0,25,75,150,300,600,np.inf]
    corr = []
    for i in range(len(edges)-1):
        m = (b_tr>=edges[i]) & (b_tr<edges[i+1])
        if m.sum()<50: corr.append(1.0); continue
        # median of y/b ratio in bucket
        r = y_tr[m]/np.maximum(b_tr[m],1e-6)
        corr.append(np.clip(np.median(r),0.5,2.0))
    b_va = blend[va]
    idx = np.digitize(b_va, edges[1:-1])
    return b_va * np.array(corr)[idx]
maes=[]
for d in days:
    tr = D!=d; va=~tr
    maes.append(np.mean(np.abs(fp_pw(tr,va)-y[va])))
print("piecewise ratio LOSO:", round(np.mean(maes),4))

# stage-2 XGB quantile on [learners + blend + key feats]
s2cols = L + ["blend"] + key_feats
def fp_s2(tr,va,seed=7,rounds=800,lr=0.05,depth=5):
    dtr = xgb.DMatrix(oof.loc[tr,s2cols], label=y[tr])
    dva = xgb.DMatrix(oof.loc[va,s2cols])
    m = xgb.train({"objective":"reg:quantileerror","quantile_alpha":0.5,"max_depth":depth,
                   "learning_rate":lr,"seed":seed,"tree_method":"hist"}, dtr, num_boost_round=rounds, verbose_eval=False)
    return m.predict(dva)
maes=[]; t0=time.time()
for d in days:
    tr = D!=d; va=~tr
    maes.append(np.mean(np.abs(fp_s2(tr,va)-y[va])))
print("stage2 XGBq LOSO:", round(np.mean(maes),4), [round(x,2) for x in maes], f"({time.time()-t0:.0f}s)")
print("ref W0 LOSO:", round(np.mean([np.mean(np.abs(blend[D!=d][~(D==d)]-y[D!=d])) for d in days]),4))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet").drop(columns=["y"])
oof = oof.merge(A.load_saved("oof_tw.parquet"), on=["household_key","snapshot_day"], how="left")
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
oof = oof.merge(tt, on=["household_key","snapshot_day"], how="left")
fv = A.load_saved("feats_v3.parquet")
oof = oof.merge(fv[["household_key","snapshot_day","spend_28","spend_56","spend_84","baskets_84","baskets_28","spend_all"]],
                on=["household_key","snapshot_day"], how="left")
L = ["med_v3","hgbq_v3","hgbq_all","oof_tw"]
W0 = np.array([0.4,0.3,0.3,0.0])
oof["blend"] = oof[L].values @ W0
tr = oof.y.notna()
d = oof[tr].copy()
y = d.y.values; b = d.blend.values
base = np.mean(np.abs(b-y)); print("train MAE blend:", round(base,3))

for col in ["spend_84","spend_56","spend_28"]:
    z = d[col].values == 0
    print(f"\nrule {col}==0: n={z.sum()} ({z.mean():.1%})")
    print("  of which y==0:", ((y==0)&z).sum(), " y>0:", ((y>0)&z).sum())
    if z.sum():
        p = np.where(z, 0.0, b)
        print("  MAE after zeroing:", round(np.mean(np.abs(p-y)),3), " delta:", round(np.mean(np.abs(p-y))-base,3))
        # partial: only zero if blend below threshold t
        for t in [10,25,50]:
            zz = z & (b<t)
            p = np.where(zz, 0.0, b)
            print(f"    zero if {col}==0 & blend<{t}: n={zz.sum()}, MAE={np.mean(np.abs(p-y)):.3f}, delta={np.mean(np.abs(p-y))-base:+.3f}")

# also: what does blend predict on true-zero rows with spend_84==0?
m = (y==0) & (d.spend_84.values==0)
print("\ntrue-zero & spend_84==0: n=", m.sum(), " blend mean:", b[m].mean().round(1), " median:", np.median(b[m]).round(1))
m2 = (y>0) & (d.spend_84.values==0)
print("y>0 & spend_84==0: n=", m2.sum(), " y mean:", y[m2].mean().round(1), " blend mean:", b[m2].mean().round(1))
# how many val rows would be zeroed?
va = ~tr
zv = oof.loc[va,"spend_84"].values==0
print("\nval rows spend_84==0:", zv.sum(), "of", va.sum())


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

fv = A.load_saved("feats_v3.parquet")
fa = A.load_saved("feats_all_e016.parquet")   # v3 + seasonal
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
fv = fv.merge(tt, on=["household_key","snapshot_day"], how="left")
fa = fa.merge(tt, on=["household_key","snapshot_day"], how="left")
feats_v = [c for c in fv.columns if c not in ("household_key","snapshot_day","y")]
feats_a = [c for c in fa.columns if c not in ("household_key","snapshot_day","y")]
tr_all = fv.y.notna()
print("train rows:", tr_all.sum(), "val rows:", (~tr_all).sum())

def xgb_q(X, yv, Xp, cfg, seed=7):
    dtr = xgb.DMatrix(X, label=yv); dpr = xgb.DMatrix(Xp)
    m = xgb.train({"objective":"reg:quantileerror","quantile_alpha":0.5,"tree_method":"hist",
                   "seed":seed,"learning_rate":cfg[0],"max_depth":cfg[1]}, dtr,
                  num_boost_round=cfg[2], verbose_eval=False)
    return m.predict(dpr)

def hgb_q(X, yv, Xp, cfg, seed=7):
    m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, random_state=seed,
        max_iter=cfg[0], learning_rate=cfg[1], max_leaf_nodes=31, min_samples_leaf=cfg[2],
        l2_regularization=0.1, early_stopping=False)
    m.fit(X, yv)
    return m.predict(Xp)

XCFG = {"a":(0.03,6,1200), "b":(0.02,7,2000), "c":(0.05,6,600)}
HCFG = {"a":(500,0.06,20), "b":(300,0.05,40)}

# ---- fold-431 sanity: pick best config per learner type
fold = 431
trf = tr_all & (fv.snapshot_day.values != fold)
vaf = tr_all & (fv.snapshot_day.values == fold)
oofc = A.load_saved("oof_e016_cv.parquet")
oofc = oofc.merge(A.load_saved("oof_tw.parquet"), on=["household_key","snapshot_day"], how="left")
oc431 = oofc[oofc.snapshot_day==fold].set_index(["household_key","snapshot_day"])
idx_va = pd.MultiIndex.from_frame(fv.loc[vaf,["household_key","snapshot_day"]])
y431 = fv.loc[vaf,"y"].values
ref = {"med_v3":np.mean(np.abs(oc431.loc[idx_va,"med_v3"]-y431)),
       "hgbq_v3":np.mean(np.abs(oc431.loc[idx_va,"hgbq_v3"]-y431)),
       "hgbq_all":np.mean(np.abs(oc431.loc[idx_va,"hgbq_all"]-y431))}
print("OOF fold431 refs:", {k:round(v,3) for k,v in ref.items()})

best_x = min(XCFG, key=lambda k: np.mean(np.abs(xgb_q(fv.loc[trf,feats_v], fv.loc[trf,"y"].values, fv.loc[vaf,feats_v], XCFG[k])-y431)))
print("best xgb cfg:", best_x, "MAE:", round(np.mean(np.abs(xgb_q(fv.loc[trf,feats_v], fv.loc[trf,"y"].values, fv.loc[vaf,feats_v], XCFG[best_x])-y431)),3))

best_hv = min(HCFG, key=lambda k: np.mean(np.abs(hgb_q(fv.loc[trf,feats_v], fv.loc[trf,"y"].values, fv.loc[vaf,feats_v], HCFG[k])-y431)))
print("best hgb(v3) cfg:", best_hv, "MAE:", round(np.mean(np.abs(hgb_q(fv.loc[trf,feats_v], fv.loc[trf,"y"].values, fv.loc[vaf,feats_v], HCFG[best_hv])-y431)),3))
best_ha = min(HCFG, key=lambda k: np.mean(np.abs(hgb_q(fa.loc[trf,feats_a], fa.loc[trf,"y"].values, fa.loc[vaf,feats_a], HCFG[k])-y431)))
print("best hgb(all) cfg:", best_ha, "MAE:", round(np.mean(np.abs(hgb_q(fa.loc[trf,feats_a], fa.loc[trf,"y"].values, fa.loc[vaf,feats_a], HCFG[best_ha])-y431)),3))

# ---- final: train on ALL 13 snapshots, predict val
t0=time.time()
pA = xgb_q(fv.loc[tr_all,feats_v], fv.loc[tr_all,"y"].values, fv.loc[~tr_all,feats_v], XCFG[best_x])
print("A done", f"{time.time()-t0:.0f}s")
pB = hgb_q(fv.loc[tr_all,feats_v], fv.loc[tr_all,"y"].values, fv.loc[~tr_all,feats_v], HCFG[best_hv])
print("B done", f"{time.time()-t0:.0f}s")
pC = hgb_q(fa.loc[tr_all,feats_a], fa.loc[tr_all,"y"].values, fa.loc[~tr_all,feats_a], HCFG[best_ha])
print("C done", f"{time.time()-t0:.0f}s")

pred = 0.4*pA + 0.3*pB + 0.3*pC
out = fv.loc[~tr_all,["household_key","snapshot_day"]].copy()
out["prediction"] = pred
print("val pred: n=",len(out)," mean=",pred.mean().round(2)," med=",np.median(pred).round(2)," max=",pred.max().round(1))

pe = A.load_saved("pred_e016.parquet").set_index(["household_key","snapshot_day"])["prediction"]
out_i = out.set_index(["household_key","snapshot_day"])
common = out_i.index.intersection(pe.index)
print("keys match E016:", len(common)==len(out))
d = out_i.loc[common,"prediction"] - pe.loc[common]
print("vs pred_e016: corr=", np.corrcoef(out_i.loc[common,"prediction"], pe.loc[common])[0,1].round(4),
      " mean diff=", d.mean().round(2), " MAE diff=", np.abs(d).mean().round(2))

path = A.save_table(out.reset_index(drop=True), "pred_e020")
print("saved:", path)
