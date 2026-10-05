
import pandas as pd, numpy as np, agent_api

names = ["pred_e013","pred_e014","pred_e011","pred_e007","pred_e005","pred_e004"]
preds = {}
for n in names:
    try:
        df = agent_api.load_saved(n + ".parquet")
        preds[n] = df
        print(n, df.shape, sorted(df.snapshot_day.unique()))
    except Exception as e:
        print(n, "ERR", type(e).__name__, e)

tt = agent_api.train_targets()
print("train_targets", tt.shape, sorted(tt.snapshot_day.unique()))

# validation rows per pred
val_days = agent_api.snapshot_days()["validation"]
base = preds["pred_e013"]
val = base[base.snapshot_day.isin(val_days)]
print("val rows", val.shape)

# pairwise correlation of predictions on validation rows
m = val[["household_key","snapshot_day","prediction"]].rename(columns={"prediction":"e013"})
for n in names[1:]:
    if n in preds:
        v = preds[n][preds[n].snapshot_day.isin(val_days)][["household_key","snapshot_day","prediction"]].rename(columns={"prediction":n})
        m = m.merge(v, on=["household_key","snapshot_day"], how="inner")
print(m.shape)
print(m.drop(columns=["household_key","snapshot_day"]).corr().round(4))

# train-row MAE of each (in-sample reference)
for n in names:
    if n in preds:
        d = preds[n].merge(tt, on=["household_key","snapshot_day"])
        if len(d):
            print(n, "train MAE", np.abs(d.prediction - d.future_spend_4w).mean().round(3), "rows", len(d))


# ---- cell ----

import pandas as pd, numpy as np, agent_api
f4 = agent_api.load_saved("feats_v4.parquet")
print(f4.shape)
print(list(f4.columns))
print(f4.head(3).T.head(60))


# ---- cell ----

import pandas as pd, numpy as np, agent_api
from scipy.optimize import minimize

tt = agent_api.train_targets()
f4 = agent_api.load_saved("feats_v4.parquet")
names = ["pred_e013","pred_e014","pred_e011","pred_e007","pred_e005","pred_e004"]

# build merged table: keys + preds + a few raw feats
base = f4[["household_key","snapshot_day","exp4w_blend","spend_28","spend_84","trips_28","snap_day"]].copy()
for n in names:
    p = agent_api.load_saved(n + ".parquet")[["household_key","snapshot_day","prediction"]].rename(columns={"prediction":n})
    base = base.merge(p, on=["household_key","snapshot_day"], how="inner")
print("merged", base.shape)

train = base.merge(tt, on=["household_key","snapshot_day"])
print("train rows", train.shape, "days", sorted(train.snapshot_day.unique()))
for n in names:
    print(n, "trainMAE", round(np.abs(train[n]-train.future_spend_4w).mean(),3))

# inner holdout: fit stack on days<=403, eval on 431
inner_tr = train[train.snapshot_day <= 403]
inner_va = train[train.snapshot_day == 431]
print("inner tr/va", len(inner_tr), len(inner_va))

def fit_stack(cols, df_fit, l2=0.0, w0=None):
    X = df_fit[cols].values.astype(float)
    y = df_fit.future_spend_4w.values.astype(float)
    k = len(cols)
    if w0 is None: w0 = np.zeros(k); w0[0] = 1.0
    def obj(w):
        r = y - X @ w
        return np.abs(r).mean() + l2 * np.sum(w**2)
    cons = [{"type":"eq","fun": lambda w: w.sum()-1}]
    bnds = [(0,1)]*k
    res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter":400,"ftol":1e-9})
    return res.x, res.fun

variants = {
 "A6": names,
 "B6+2": names + ["exp4w_blend","spend_28"],
 "C6+4": names + ["exp4w_blend","spend_28","spend_84","trips_28"],
 "D4": ["pred_e013","pred_e011","pred_e007","exp4w_blend","spend_28"],
 "E_e013only_blend": ["pred_e013","exp4w_blend","spend_28"],
}
for v, cols in variants.items():
    w, f = fit_stack(cols, inner_tr)
    mae_va = np.abs(inner_va.future_spend_4w.values - inner_va[cols].values @ w).mean()
    mae_e013 = np.abs(inner_va.future_spend_4w.values - inner_va["pred_e013"].values).mean()
    print(v, "w=", np.round(w,3), "inner431 MAE", round(mae_va,3), "(e013 alone:", round(mae_e013,3), ")")


# ---- cell ----

import pandas as pd, numpy as np, agent_api
tt = agent_api.train_targets()
f4 = agent_api.load_saved("feats_v4.parquet")
p = agent_api.load_saved("pred_e013.parquet")
print(tt.dtypes, "\n", f4.dtypes.head(3), "\n", p.dtypes)
print(tt.household_key.dtype, f4.household_key.dtype, p.household_key.dtype)
m = f4.merge(p, on=["household_key","snapshot_day"])
print("f4+pred", m.shape)
m2 = f4.merge(tt, on=["household_key","snapshot_day"])
print("f4+tt", m2.shape)
m3 = f4.astype({"household_key":"int64","snapshot_day":"int64"}).merge(
     tt.astype({"household_key":"int64","snapshot_day":"int64"}), on=["household_key","snapshot_day"])
print("cast f4+tt", m3.shape)


# ---- cell ----

import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

f4 = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
df = f4.merge(tt, on=["household_key","snapshot_day"])
drop = ["household_key","snapshot_day","future_spend_4w"]
feats = [c for c in df.columns if c not in drop]
# one-hot the categoricals
cat_cols = [c for c in feats if df[c].dtype == object]
X = pd.get_dummies(df[feats], columns=cat_cols, dummy_na=True)
y = df.future_spend_4w.values.astype(float)

tr = df.snapshot_day <= 403
va = df.snapshot_day == 431
Xtr, ytr, Xva, yva = X[tr.values], y[tr.values], X[va.values], y[va.values]
print("inner tr/va", Xtr.shape, Xva.shape)
print("e013-style baseline check: exp4w_blend MAE on 431:",
      round(np.abs(df.loc[va,"exp4w_blend"].values - yva).mean(),3))

def run(tag, obj, transform=None, inv=None, blend=0.3, seed=0, depth=5, mcw=40):
    yt = transform(ytr) if transform else ytr
    m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=depth,
                     min_child_weight=mcw, subsample=0.8, colsample_bytree=0.8,
                     objective=obj, quantile_alpha=0.5 if "quantile" in obj else None,
                     random_state=seed, n_jobs=4, tree_method="hist")
    m.fit(Xtr, yt)
    p = m.predict(Xva)
    p = inv(p) if inv else p
    pb = (1-blend)*p + blend*df.loc[va,"exp4w_blend"].values
    print(f"{tag:28s} MAE {np.abs(p-yva).mean():7.3f}  blended {np.abs(pb-yva).mean():7.3f}")

run("quantile0.5 raw (E011-style)", "reg:quantileerror")
run("sqerr raw", "reg:squarederror")
run("sqerr log1p", "reg:squarederror", transform=np.log1p, inv=np.expm1)
run("quantile0.5 log1p", "reg:quantileerror", transform=np.log1p, inv=np.expm1)
run("pseudohuber raw", "reg:pseudohubererror")
run("sqerr sqrt", "reg:squarederror", transform=np.sqrt, inv=lambda z: np.square(z))


# ---- cell ----

import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

f4 = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
df = f4.merge(tt, on=["household_key","snapshot_day"])
feats = [c for c in df.columns if c not in ["household_key","snapshot_day","future_spend_4w"]]
cat_cols = [c for c in feats if df[c].dtype == object]
X = pd.get_dummies(df[feats], columns=cat_cols, dummy_na=True)
y = df.future_spend_4w.values.astype(float)
day = df.snapshot_day.values.astype(float)
tr = df.snapshot_day <= 403
va = df.snapshot_day == 431
blendv = df.loc[va,"exp4w_blend"].values

def train_eval(w=None, days=None, alpha=0.5, blend=0.0, seed=0, depth=5, mcw=40, lr=0.08, n=400):
    m = tr if days is None else (df.snapshot_day.isin(days))
    Xtr, ytr, dtr = X[m.values], y[m.values], day[m.values]
    if w is None:
        sw = np.ones(len(ytr))
    elif w == "recency":
        sw = np.clip((dtr - 95) / (403 - 95), 0.15, 1.0)
    elif w == "recency2":
        sw = np.clip((dtr - 95) / (403 - 95), 0.05, 1.0) ** 2
    mod = XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, min_child_weight=mcw,
                       subsample=0.8, colsample_bytree=0.8, objective="reg:quantileerror",
                       quantile_alpha=alpha, random_state=seed, n_jobs=4, tree_method="hist")
    mod.fit(Xtr, ytr, sample_weight=sw)
    p = mod.predict(X[va.values])
    p = np.clip(p, 0, None)
    if blend > 0:
        p = (1-blend)*p + blend*blendv
    return np.abs(p - y[va.values]).mean()

print("base (E011-style)      ", round(train_eval(),3))
print("recency w              ", round(train_eval(w="recency"),3))
print("recency w^2            ", round(train_eval(w="recency2"),3))
print("drop day 95            ", round(train_eval(days=[123,151,179,207,235,263,291,319,347,375,403]),3))
print("drop 95, recency       ", round(train_eval(w="recency", days=[123,151,179,207,235,263,291,319,347,375,403]),3))
for b in [0.1,0.2,0.3]:
    print(f"base blend {b}          ", round(train_eval(blend=b),3))
for a in [0.48,0.52,0.54]:
    print(f"alpha {a}              ", round(train_eval(alpha=a),3))
print("deeper bag check d4/d6 ", round(train_eval(depth=4),3), round(train_eval(depth=6),3))


# ---- cell ----

import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

f4 = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
df = f4.merge(tt, on=["household_key","snapshot_day"])
y = df.future_spend_4w.values.astype(float)
print("y stats: zero frac", (y==0).mean().round(4), "| quantiles", np.quantile(y,[.1,.25,.5,.75,.9,.95,.99]).round(1))
for d in sorted(df.snapshot_day.unique()):
    yy = df[df.snapshot_day==d].future_spend_4w
    print(d, "n", len(yy), "zero%", round((yy==0).mean()*100,1), "median", round(yy.median(),1))

feats = [c for c in df.columns if c not in ["household_key","snapshot_day","future_spend_4w"]]
cat_cols = [c for c in feats if df[c].dtype == object]
X = pd.get_dummies(df[feats], columns=cat_cols, dummy_na=True)
tr = df.snapshot_day <= 403; va = df.snapshot_day == 431
ytr, yva = y[tr.values], y[va.values]
Xtr, Xva = X[tr.values], X[va.values]

m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=5, min_child_weight=40,
                 subsample=0.8, colsample_bytree=0.8, objective="reg:quantileerror",
                 quantile_alpha=0.5, random_state=0, n_jobs=4, tree_method="hist")
m.fit(Xtr, ytr)
p = np.clip(m.predict(Xva), 0, None)
print("\nday-431: zero frac", round((yva==0).mean(),3))
print("pred on y==0 rows: median", round(np.median(p[yva==0]),1), "mean", round(p[yva==0].mean(),1))
print("pred on y>0  rows: median", round(np.median(p[yva>0]),1))
print("MAE on y==0 rows", round(np.abs(p[yva==0]).mean(),2), "| share of total abs err",
      round(np.abs(p[yva==0]).sum()/np.abs(p-yva).sum(),3))
print("MAE on y>0 rows", round(np.abs(p[yva>0]-yva[yva>0]).mean(),2))
print("mean pred", round(p.mean(),1), "mean y", round(yva.mean(),1), "median pred", round(np.median(p),1), "median y", round(np.median(yva),1))


# ---- cell ----

import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

f4 = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
df = f4.merge(tt, on=["household_key","snapshot_day"])
y = df.future_spend_4w.values.astype(float)
print("mean/median y by snapshot day:")
g = df.groupby("snapshot_day").future_spend_4w.agg(["mean","median","count"])
print(g.round(1))

feats = [c for c in df.columns if c not in ["household_key","snapshot_day","future_spend_4w"]]
cat_cols = [c for c in feats if df[c].dtype == object]
X = pd.get_dummies(df[feats], columns=cat_cols, dummy_na=True)
tr = df.snapshot_day <= 403; va = df.snapshot_day == 431
ytr, yva = y[tr.values], y[va.values]
Xtr, Xva = X[tr.values], X[va.values]

def fit(obj="reg:quantileerror", alpha=0.5, depth=5, mcw=40, seed=0, n=400, lr=0.08, Xtr_=Xtr, ytr_=ytr):
    m = XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective=obj,
                     quantile_alpha=alpha if "quantile" in obj else None,
                     random_state=seed, n_jobs=4, tree_method="hist")
    m.fit(Xtr_, ytr_)
    return np.clip(m.predict(Xva), 0, None)

p = fit()
# error by y decile
qs = np.quantile(yva, np.linspace(0,1,11))
print("\nday431 error by y-decile (base quantile model):")
for i in range(10):
    lo, hi = qs[i], qs[i+1]
    msk = (yva >= lo) & (yva <= hi) if i==9 else (yva >= lo) & (yva < hi)
    print(f"  dec {i}: y [{lo:6.1f},{hi:6.1f}] n={msk.sum():4d} MAE={np.abs(p[msk]-yva[msk]).mean():7.2f} bias={(p[msk]-yva[msk]).mean():8.2f}")

# bag: quantile + absoluteerror objective, seeds
def bag(configs, seeds=(0,1)):
    ps = []
    for c in configs:
        for s in seeds:
            ps.append(fit(**c, seed=s))
    return np.mean(ps, axis=0)

cfgs = [dict(depth=d, mcw=w) for d in (4,5,6) for w in (20,40,60)][:8]
pb2 = bag(cfgs, seeds=(0,1)); pb4 = bag(cfgs, seeds=(0,1,2,3))
print("\nquantile bag 2 seeds MAE", round(np.abs(pb2-yva).mean(),3))
print("quantile bag 4 seeds MAE", round(np.abs(pb4-yva).mean(),3))
cfgs_l1 = cfgs[:4] + [dict(obj="reg:absoluteerror", depth=d, mcw=w) for d,w in [(4,20),(5,40),(6,40),(4,60)]]
pb_l1 = bag(cfgs_l1, seeds=(0,1))
print("quantile+L1 mixed bag MAE", round(np.abs(pb_l1-yva).mean(),3))


# ---- cell ----

import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

f4 = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
df = f4.merge(tt, on=["household_key","snapshot_day"])
y = df.future_spend_4w.values.astype(float)
feats = [c for c in df.columns if c not in ["household_key","snapshot_day","future_spend_4w"]]
cat_cols = [c for c in feats if df[c].dtype == object]
X = pd.get_dummies(df[feats], columns=cat_cols, dummy_na=True)
tr = df.snapshot_day <= 403; va = df.snapshot_day == 431
ytr, yva = y[tr.values], y[va.values]
Xtr, Xva = X[tr.values], X[va.values]
blendv = df.loc[va,"exp4w_blend"].values

def fit(obj="reg:quantileerror", alpha=0.5, depth=5, mcw=40, seed=0, n=400, lr=0.08):
    m = XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective=obj,
                     quantile_alpha=alpha if "quantile" in obj else None,
                     random_state=seed, n_jobs=4, tree_method="hist")
    m.fit(Xtr, ytr)
    return np.clip(m.predict(Xva), 0, None)

# A) quantile + L1 average
p_q = fit(); p_l1 = fit(obj="reg:absoluteerror")
for w in [0.25,0.5]:
    pm = (1-w)*p_q + w*p_l1
    print(f"avg quantile+L1 w={w}: MAE", round(np.abs(pm-yva).mean(),3))

# B) two-stage: zero classifier + quantile regressor on positives
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
pzero_tr = (ytr == 0).astype(int)
gbc = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=0)
gbc.fit(Xtr, pzero_tr)
pz = gbc.predict_proba(Xva)[:,1]
from sklearn.metrics import roc_auc_score
print("\nzero-classifier AUC on 431:", round(roc_auc_score((yva==0).astype(int), pz),4))
pos = ytr > 0
pq_pos = fit(ytr_=ytr[pos])
best = None
for thr in [0.35,0.45,0.55,0.65]:
    pm = np.where(pz > thr, 0.0, pq_pos)
    mae = np.abs(pm-yva).mean()
    print(f"two-stage thr={thr}: MAE {mae:.3f}")
    if best is None or mae < best[0]: best = (mae, thr)
print("best two-stage:", best)

# C) simple calibrated shrink: p' = a + b*p fit by least squares on inner val
A = np.vstack([np.ones_like(p_q), p_q]).T
coef, *_ = np.linalg.lstsq(A, yva, rcond=None)
print("calibration a,b on 431:", coef.round(4), "-> in-sample MAE", round(np.abs(A@coef - yva).mean(),3))


# ---- cell ----

import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

f4 = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
df = f4.merge(tt, on=["household_key","snapshot_day"])
y = df.future_spend_4w.values.astype(float)
feats = [c for c in df.columns if c not in ["household_key","snapshot_day","future_spend_4w"]]
cat_cols = [c for c in feats if df[c].dtype == object]
print("cat cols:", cat_cols)
X = pd.get_dummies(df[feats], columns=cat_cols, dummy_na=True)
print("object cols left:", [c for c in X.columns if X[c].dtype == object])
Xv = X.astype(np.float32).values
tr = (df.snapshot_day <= 403).values; va = (df.snapshot_day == 431).values
ytr, yva = y[tr], y[va]
Xtr, Xva = Xv[tr], Xv[va]

def fit(obj="reg:quantileerror", alpha=0.5, depth=5, mcw=40, seed=0, n=400, lr=0.08):
    m = XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective=obj,
                     quantile_alpha=alpha if "quantile" in obj else None,
                     random_state=seed, n_jobs=4, tree_method="hist")
    m.fit(Xtr, ytr)
    return np.clip(m.predict(Xva), 0, None)

p_q = fit(); p_l1 = fit(obj="reg:absoluteerror")
for w in [0.25,0.5]:
    print(f"avg quantile+L1 w={w}: MAE", round(np.abs(((1-w)*p_q+w*p_l1)-yva).mean(),3))

# two-stage: zero classifier + quantile on positives
pzero_tr = (ytr == 0).astype(int)
gbc = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=0)
gbc.fit(Xtr, pzero_tr)
pz = gbc.predict_proba(Xva)[:,1]
print("zero-classifier AUC on 431:", round(roc_auc_score((yva==0).astype(int), pz),4))
pos = ytr > 0
pq_pos = fit(ytr_=ytr[pos])
best = None
for thr in [0.3,0.4,0.5,0.6]:
    pm = np.where(pz > thr, 0.0, pq_pos)
    mae = np.abs(pm-yva).mean()
    print(f"two-stage thr={thr}: MAE {mae:.3f}")
    if best is None or mae < best[0]: best = (mae, thr)
print("best two-stage:", best)


# ---- cell ----

import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

f4 = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
df = f4.merge(tt, on=["household_key","snapshot_day"])
y = df.future_spend_4w.values.astype(float)
feats = [c for c in df.columns if c not in ["household_key","snapshot_day","future_spend_4w"]]
cat_cols = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
print("cat cols:", cat_cols)
X = pd.get_dummies(df[feats], columns=cat_cols, dummy_na=True)
Xv = X.astype(np.float32).values
tr = (df.snapshot_day <= 403).values; va = (df.snapshot_day == 431).values
ytr, yva = y[tr], y[va]
Xtr, Xva = Xv[tr], Xv[va]
print("X", Xv.shape)

def fit(obj="reg:quantileerror", alpha=0.5, depth=5, mcw=40, seed=0, n=400, lr=0.08):
    m = XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective=obj,
                     quantile_alpha=alpha if "quantile" in obj else None,
                     random_state=seed, n_jobs=4, tree_method="hist")
    m.fit(Xtr, ytr)
    return np.clip(m.predict(Xva), 0, None)

p_q = fit(); p_l1 = fit(obj="reg:absoluteerror")
for w in [0.25,0.5]:
    print(f"avg quantile+L1 w={w}: MAE", round(np.abs(((1-w)*p_q+w*p_l1)-yva).mean(),3))

pzero_tr = (ytr == 0).astype(int)
gbc = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=0)
gbc.fit(Xtr, pzero_tr)
pz = gbc.predict_proba(Xva)[:,1]
print("zero-classifier AUC on 431:", round(roc_auc_score((yva==0).astype(int), pz),4))
pos = ytr > 0
pq_pos = fit(ytr_=ytr[pos])
best = None
for thr in [0.3,0.4,0.5,0.6]:
    pm = np.where(pz > thr, 0.0, pq_pos)
    mae = np.abs(pm-yva).mean()
    print(f"two-stage thr={thr}: MAE {mae:.3f}")
    if best is None or mae < best[0]: best = (mae, thr)
print("best two-stage:", best)


# ---- cell ----

import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

f4 = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
df = f4.merge(tt, on=["household_key","snapshot_day"])
y = df.future_spend_4w.values.astype(float)
feats = [c for c in df.columns if c not in ["household_key","snapshot_day","future_spend_4w"]]
cat_cols = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
X = pd.get_dummies(df[feats], columns=cat_cols, dummy_na=True)
Xv = X.astype(np.float32).values
tr = (df.snapshot_day <= 403).values; va = (df.snapshot_day == 431).values
ytr, yva = y[tr], y[va]
Xtr, Xva = Xv[tr], Xv[va]

def fit(obj="reg:quantileerror", alpha=0.5, depth=5, mcw=40, seed=0, n=400, lr=0.08):
    m = XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective=obj,
                     quantile_alpha=alpha if "quantile" in obj else None,
                     random_state=seed, n_jobs=4, tree_method="hist")
    m.fit(Xtr, ytr)
    return np.clip(m.predict(Xva), 0, None)

p_q = fit()
pzero_tr = (ytr == 0).astype(int)
hgc = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.08, random_state=0)
hgc.fit(Xtr, pzero_tr)
pz = hgc.predict_proba(Xva)[:,1]
print("zero-classifier AUC on 431:", round(roc_auc_score((yva==0).astype(int), pz),4))
pos = ytr > 0
pq_pos = fit(ytr_=ytr[pos])
best = None
for thr in [0.3,0.4,0.5,0.6]:
    pm = np.where(pz > thr, 0.0, pq_pos)
    mae = np.abs(pm-yva).mean()
    print(f"two-stage thr={thr}: MAE {mae:.3f}")
    if best is None or mae < best[0]: best = (mae, thr)
print("best two-stage:", best)


# ---- cell ----

import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

f4 = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
feats = [c for c in f4.columns if c not in ["household_key","snapshot_day"]]
cat_cols = [c for c in feats if not pd.api.types.is_numeric_dtype(f4[c])]
X_full = pd.get_dummies(f4[feats], columns=cat_cols, dummy_na=True).astype(np.float32)
print("X_full", X_full.shape)

tr_mask = f4.snapshot_day.isin(tt.snapshot_day.unique()) & (f4.snapshot_day <= 403)
va_mask = f4.snapshot_day.isin([459,487,515,543])
Xtr = X_full[tr_mask.values].values
ytr = tt.set_index(["household_key","snapshot_day"]).reindex(
    pd.MultiIndex.from_arrays([f4.loc[tr_mask,"household_key"], f4.loc[tr_mask,"snapshot_day"]])
).future_spend_4w.values.astype(float)
print("train rows", len(ytr), "nan y:", np.isnan(ytr).sum())
Xva = X_full[va_mask.values].values
va_keys = f4.loc[va_mask, ["household_key","snapshot_day"]].reset_index(drop=True)
print("val rows", len(va_keys))

cfgs = [(4,20),(4,40),(4,60),(5,20),(5,40),(5,60),(6,20),(6,40)]
preds = []
for d, w in cfgs:
    for s in range(4):
        m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=d, min_child_weight=w,
                         subsample=0.8, colsample_bytree=0.8, objective="reg:quantileerror",
                         quantile_alpha=0.5, random_state=s, n_jobs=4, tree_method="hist")
        m.fit(Xtr, ytr)
        preds.append(np.clip(m.predict(Xva), 0, None))
p = np.mean(preds, axis=0)
out = va_keys.copy()
out["prediction"] = p.astype(float)
print("pred stats: mean", round(p.mean(),2), "median", round(np.median(p),2), "min", round(p.min(),2), "max", round(p.max(),2))
path = agent_api.save_table(out, "pred_e015")
print(path)
