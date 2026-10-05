import agent_api as A, pandas as pd, numpy as np, time
f = A.load_saved("feats_v3.parquet")
print("feats_v3", f.shape)
print(f.columns.tolist())
print(f.head(3).T)
p = A.load_saved("pred_e005.parquet")
print("pred_e005", p.shape, p.columns.tolist())
print(p.head(3))
tt = A.train_targets()
print("train_targets", tt.shape, tt.columns.tolist())
print(tt.head(3))
print(A.snapshot_days())
t0=time.time()
v = A.snapshot()
tx = v.table("transactions")
print("tx shape", tx.shape, "load time", round(time.time()-t0,2))
print("day range", tx.day.min(), tx.day.max())


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error

f = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
df = tt.merge(f.drop(columns=["index"]), on=["household_key","snapshot_day"], how="left")
print("merged", df.shape, "NaN rows:", df.isna().any(axis=1).sum())

y = df["future_spend_4w"].values
print("target: mean %.1f median %.1f zero-frac %.3f" % (y.mean(), np.median(y), (y==0).mean()))
print("target by snapshot:")
print(df.groupby("snapshot_day")["future_spend_4w"].agg(["mean","median",lambda s:(s==0).mean(),"count"]))

# validation predictions from E005 vs features
p5 = A.load_saved("pred_e005.parquet")
fv = f[f.snapshot_day>=459]
m = fv.merge(p5, on=["household_key","snapshot_day"])
print("pred_e005 stats: mean %.1f median %.1f zero %.3f" % (m.prediction.mean(), m.prediction.median(), (m.prediction==0).mean()))
print("corr pred vs exp4w_blend:", np.corrcoef(m.prediction, m.exp4w_blend)[0,1].round(3))
print("corr pred vs spend_28:", np.corrcoef(m.prediction, m.spend_28)[0,1].round(3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
f = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
print(f.dtypes[["household_key","snapshot_day"]], tt.dtypes.values)
print("f hk unique sample:", f.household_key.unique()[:5], f.household_key.nunique())
print("tt hk unique sample:", tt.household_key.unique()[:5], tt.household_key.nunique())
print("f snap days:", sorted(f.snapshot_day.unique()))
print("tt snap days:", sorted(tt.snapshot_day.unique()))
# try casting
f2 = f.copy(); f2["household_key"]=f2["household_key"].astype(int); f2["snapshot_day"]=f2["snapshot_day"].astype(int)
m = tt.merge(f2.drop(columns=["index"]), on=["household_key","snapshot_day"], how="left")
print("merged after cast:", m.shape, "NaN rows:", m.drop(columns=["future_spend_4w"]).isna().any(axis=1).sum())


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
f = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
m = tt.merge(f.drop(columns=["index"]), on=["household_key","snapshot_day"], how="inner")
print("inner merge:", m.shape)
# check which train rows are missing from feats_v3
key_t = set(zip(tt.household_key, tt.snapshot_day))
key_f = set(zip(f.household_key, f.snapshot_day))
missing = key_t - key_f
print("train keys missing in feats_v3:", len(missing))
import collections
c = collections.Counter(d for h,d in missing)
print(sorted(c.items()))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

f = A.load_saved("feats_v3.parquet").drop(columns=["index"])
tt = A.train_targets()
df = tt.merge(f, on=["household_key","snapshot_day"], how="inner")
FEATS = [c for c in f.columns if c not in ("household_key","snapshot_day")]
print("n feats:", len(FEATS))

def train_pred(train_days, pred_days, params=None, alpha=0.5):
    tr = df[df.snapshot_day.isin(train_days)]
    Xtr, ytr = tr[FEATS], tr.future_spend_4w
    p = dict(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
             colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, n_jobs=4,
             objective="reg:quantileerror", quantile_alpha=alpha, tree_method="hist")
    if params: p.update(params)
    m = xgb.XGBRegressor(**p)
    m.fit(Xtr, ytr)
    pr = df[df.snapshot_day.isin(pred_days)]
    return pr, m.predict(pr[FEATS])

t0=time.time()
# local proxy: train on <=403, predict 431
pr, pred = train_pred([d for d in range(95,432,28) if d<=403], [431])
mae431 = np.abs(pred - pr.future_spend_4w.values).mean()
print("LOCAL 431 MAE (E005 repro):", round(mae431,3), "time", round(time.time()-t0,1))
# also proxy on 403 trained <=375
pr2, pred2 = train_pred([d for d in range(95,404,28) if d<=375], [403])
print("LOCAL 403 MAE:", round(np.abs(pred2-pr2.future_spend_4w.values).mean(),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

f = A.load_saved("feats_v3.parquet").drop(columns=["index"])
tt = A.train_targets()
df = tt.merge(f, on=["household_key","snapshot_day"], how="inner")
FEATS = [c for c in f.columns if c not in ("household_key","snapshot_day")]
BASE = dict(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, n_jobs=4,
        objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist")

def fit(train_days, params=None, sample_weight=None):
    tr = df[df.snapshot_day.isin(train_days)]
    p = dict(BASE)
    if params: p.update(params)
    m = xgb.XGBRegressor(**p)
    w = sample_weight(tr) if sample_weight else None
    m.fit(tr[FEATS], tr.future_spend_4w, sample_weight=w)
    return m

def evaluate(mut_name, params=None, sample_weight=None, blend_fn=None):
    t0=time.time(); out={}
    for held in [403, 431]:
        td = [d for d in range(95,432,28) if d < held]
        m = fit(td, params, sample_weight)
        pr = df[df.snapshot_day==held]
        pred = m.predict(pr[FEATS])
        if blend_fn is not None:
            pred = blend_fn(pred, pr)
        out[held] = np.abs(pred - pr.future_spend_4w.values).mean()
    print(f"{mut_name:42s} 403:{out[403]:7.3f} 431:{out[431]:7.3f} avg:{np.mean(list(out.values())):7.3f} ({time.time()-t0:.0f}s)")
    return out

# 0) repro
evaluate("repro E005")
# 1) blend with exp4w_blend persistence
for w in [0.05, 0.1, 0.2]:
    evaluate(f"blend model + {w}*exp4w_blend", blend_fn=lambda p, pr, w=w: (1-w)*p + w*pr.exp4w_blend.values)
# 2) recency-weighted loss
evaluate("sample_weight exp(-age/112)", sample_weight=lambda tr: np.exp(-(431-tr.snapshot_day)/112))
evaluate("sample_weight exp(-age/56)", sample_weight=lambda tr: np.exp(-(431-tr.snapshot_day)/56))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

f = A.load_saved("feats_v3.parquet").drop(columns=["index"])
tt = A.train_targets()
df = tt.merge(f, on=["household_key","snapshot_day"], how="inner")
FEATS = [c for c in f.columns if c not in ("household_key","snapshot_day")]
BASE = dict(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, n_jobs=4,
        objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist")

def evaluate(name, params=None, blend_fn=None, n_est=800, lr=0.05):
    t0=time.time(); out={}
    for held in [403, 431]:
        td = [d for d in range(95,432,28) if d < held]
        tr = df[df.snapshot_day.isin(td)]
        p = dict(BASE); p["n_estimators"]=n_est; p["learning_rate"]=lr
        if params: p.update(params)
        m = xgb.XGBRegressor(**p).fit(tr[FEATS], tr.future_spend_4w)
        pr = df[df.snapshot_day==held]
        pred = m.predict(pr[FEATS])
        if blend_fn is not None: pred = blend_fn(pred, pr)
        out[held] = np.abs(pred - pr.future_spend_4w.values).mean()
    print(f"{name:44s} 403:{out[403]:7.3f} 431:{out[431]:7.3f} avg:{np.mean(list(out.values())):7.3f} ({time.time()-t0:.0f}s)")
    return out

# blend sweep
for w in [0.3, 0.4, 0.5]:
    evaluate(f"blend + {w}*exp4w_blend", blend_fn=lambda p, pr, w=w: (1-w)*p + w*pr.exp4w_blend.values)
# blend with lag1 instead
evaluate("blend + 0.2*spend28_lag1", blend_fn=lambda p, pr: 0.8*p + 0.2*pr.spend28_lag1.values)
evaluate("blend + 0.2*exp4w_all", blend_fn=lambda p, pr: 0.8*p + 0.2*pr.exp4w_all.values)
# model params
evaluate("depth 8", params=dict(max_depth=8))
evaluate("depth 4", params=dict(max_depth=4))
evaluate("lr .03 n1600", n_est=1600, lr=0.03)
evaluate("mcw 1", params=dict(min_child_weight=1))
evaluate("mcw 20", params=dict(min_child_weight=20))
evaluate("subsample .6 col .5", params=dict(subsample=0.6, colsample_bytree=0.5))
evaluate("reg_lambda 5", params=dict(reg_lambda=5.0))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

f = A.load_saved("feats_v3.parquet").drop(columns=["index"])
tt = A.train_targets()
df = tt.merge(f, on=["household_key","snapshot_day"], how="inner")
FEATS = [c for c in f.columns if c not in ("household_key","snapshot_day")]
BASE = dict(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, n_jobs=4,
        objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist")

def evaluate(name, params=None, blend_fn=None, obj="reg:quantileerror"):
    t0=time.time(); out={}
    for held in [403, 431]:
        td = [d for d in range(95,432,28) if d < held]
        tr = df[df.snapshot_day.isin(td)]
        p = dict(BASE); p["objective"]=obj
        if params: p.update(params)
        m = xgb.XGBRegressor(**p).fit(tr[FEATS], tr.future_spend_4w)
        pr = df[df.snapshot_day==held]
        pred = m.predict(pr[FEATS])
        if blend_fn is not None: pred = blend_fn(pred, pr)
        out[held] = np.abs(pred - pr.future_spend_4w.values).mean()
    print(f"{name:44s} 403:{out[403]:7.3f} 431:{out[431]:7.3f} avg:{np.mean(list(out.values())):7.3f} ({time.time()-t0:.0f}s)")
    return out

def blend_w(w):
    return lambda p, pr: (1-w)*p + w*pr.exp4w_blend.values

# combos
evaluate("blend .3 + depth4", params=dict(max_depth=4), blend_fn=blend_w(0.3))
evaluate("blend .3 + depth4 + mcw20", params=dict(max_depth=4, min_child_weight=20), blend_fn=blend_w(0.3))
evaluate("blend .35 depth4", params=dict(max_depth=4), blend_fn=blend_w(0.35))
# ensemble quantile + pseudohuber averaged, then blend
def ens_blend(w, wh=0.5):
    return lambda p, pr: (1-w)*((1-wh)*p + wh*pr.pred_huber) + w*pr.exp4w_blend.values
for held in [403, 431]:
    td = [d for d in range(95,432,28) if d < held]
    tr = df[df.snapshot_day.isin(td)]
    pr = df[df.snapshot_day==held]
    m1 = xgb.XGBRegressor(**{**BASE, "objective":"reg:quantileerror"}).fit(tr[FEATS], tr.future_spend_4w)
    m2 = xgb.XGBRegressor(**{**BASE, "objective":"reg:pseudohubererror"}).fit(tr[FEATS], tr.future_spend_4w)
    pr = pr.assign(pred_huber=m2.predict(pr[FEATS]))
    for wh in [0.3, 0.5]:
        for w in [0.2, 0.3]:
            pred = (1-w)*((1-wh)*m1.predict(pr[FEATS]) + wh*pr.pred_huber) + w*pr.exp4w_blend.values
            print(f"  held {held} ens huber_w{wh} blend{w}: {np.abs(pred-pr.future_spend_4w.values).mean():.3f}")


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

f = A.load_saved("feats_v3.parquet").drop(columns=["index"])
tt = A.train_targets()
df = tt.merge(f, on=["household_key","snapshot_day"], how="inner")
FEATS = [c for c in f.columns if c not in ("household_key","snapshot_day")]
BASE = dict(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, n_jobs=4,
        objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist")

td = [d for d in range(95,432,28) if d < 431]
tr = df[df.snapshot_day.isin(td)]
m = xgb.XGBRegressor(**BASE).fit(tr[FEATS], tr.future_spend_4w)
pr = df[df.snapshot_day==431].copy()
pr["pred"] = m.predict(pr[FEATS])
pr["err"] = (pr.pred - pr.future_spend_4w).abs()

print("=== error structure at 431 ===")
print("overall MAE:", round(pr.err.mean(),2))
print("\nby target bucket:")
pr["tb"] = pd.cut(pr.future_spend_4w, [-1,0.01,50,100,200,400,10000])
print(pr.groupby("tb").agg(n=("err","size"), mae=("err","mean"), med_pred=("pred","median")))
print("\nby days_since_last bucket:")
pr["dsl"] = pd.cut(pr.days_since_last, [-1,7,14,28,56,10000])
print(pr.groupby("dsl").agg(n=("err","size"), mae=("err","mean"), med_target=("future_spend_4w","median"), med_pred=("pred","median")))
print("\nby tenure bucket:")
pr["tn"] = pd.cut(pr.tenure, [-1,150,300,500,10000])
print(pr.groupby("tn").agg(n=("err","size"), mae=("err","mean")))
print("\nzero-target rows: n=%d, mean pred=%.1f" % ((pr.future_spend_4w==0).sum(), pr.loc[pr.future_spend_4w==0,"pred"].mean()))
print("zero-target MAE contribution: %.1f of total %.1f" % (pr.loc[pr.future_spend_4w==0,"err"].mean()* (pr.future_spend_4w==0).mean(), pr.err.mean()))
imp = pd.Series(m.feature_importances_, index=FEATS).sort_values(ascending=False)
print("\ntop-15 importance:"); print(imp.head(15).round(4))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

f = A.load_saved("feats_v3.parquet").drop(columns=["index"])
tt = A.train_targets()
df = tt.merge(f, on=["household_key","snapshot_day"], how="inner")
FEATS = [c for c in f.columns if c not in ("household_key","snapshot_day")]
BASE = dict(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, n_jobs=4,
        objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist")

td = [d for d in range(95,432,28) if d < 431]
tr = df[df.snapshot_day.isin(td)]
m = xgb.XGBRegressor(**BASE).fit(tr[FEATS], tr.future_spend_4w)
pr = df[df.snapshot_day==431].copy()
pr["pred"] = m.predict(pr[FEATS])

# churn definition: no purchase in trailing 28d at snapshot (days_since_last > 28)
pr["churn28"] = pr.days_since_last > 28
print("churn28 rows: %d (%.1f%%)" % (pr.churn28.sum(), 100*pr.churn28.mean()))
print("target==0 rate by churn28:")
print(pr.groupby("churn28").future_spend_4w.apply(lambda s: (s==0).mean()).round(3))
print("MAE by churn28:")
print(pr.groupby("churn28").apply(lambda g: (g.pred-g.future_spend_4w).abs().mean()).round(2))

# does model already use dsl? check partial dependence-ish: pred vs dsl among churn28
print("\nchurn28: pred distribution:", pr.loc[pr.churn28,"pred"].describe().round(1).to_dict())

# what about households active recently but low spend: how often do they go to 0?
pr["dsl_b"] = pd.cut(pr.days_since_last, [-1,7,14,21,28,42,56,10000])
print("\nzero-rate & mean target by dsl bucket:")
print(pr.groupby("dsl_b").agg(n=("pred","size"), zero_rate=("future_spend_4w", lambda s:(s==0).mean()), mean_t=("future_spend_4w","mean"), med_pred=("pred","median")).round(2))

# check wk_cv8 / gap features for churn signal
print("\ncorr of features with (future==0) among dsl<=28 rows:")
sub = pr[pr.days_since_last<=28]
for c in ["wk_cv8","gap_max_84","gap_mean_84","active_weeks8","trips_28","spend_28","exp4w_blend","wk_std8","basket_max_84"]:
    print(f"  {c:16s} {np.corrcoef(sub[c], (sub.future_spend_4w==0).astype(float))[0,1]:.3f}")


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

f = A.load_saved("feats_v3.parquet").drop(columns=["index"])
tt = A.train_targets()
df = tt.merge(f, on=["household_key","snapshot_day"], how="inner")
FEATS = [c for c in f.columns if c not in ("household_key","snapshot_day")]
BASE = dict(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, n_jobs=4,
        objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist")
td = [d for d in range(95,432,28) if d < 431]
m = xgb.XGBRegressor(**BASE).fit(df[df.snapshot_day.isin(td)][FEATS], df[df.snapshot_day.isin(td)].future_spend_4w)
pr = df[df.snapshot_day==431].copy()
pr["pred"] = m.predict(pr[FEATS])
pr["resid"] = pr.future_spend_4w - pr.pred

# candidate features from raw transactions at snapshot 431
v = A.snapshot(431)
tx = v.table("transactions")
hhs = pr.household_key.values
tx = tx[tx.household_key.isin(hhs)]

def build(day):
    t = tx[tx.day <= day]
    g = t.groupby("household_key")
    out = pd.DataFrame(index=g.size().index)
    # last trip spend & day
    bs = t.groupby(["household_key","basket_id"]).agg(sp=("sales_value","sum"), d=("day","max"))
    last = bs.groupby("household_key").agg(last_sp=("sp","last"), last_d=("d","max"))
    out["last_trip_sp"] = last.last_sp
    out["days_last_trip"] = day - last.last_d
    for w in [7,14]:
        s = t[t.day > day-w].groupby("household_key").sales_value.sum().rename(f"s{w}")
        out = out.join(s)
    out["accel"] = out.s7/(out.s14/2+1e-9)
    # max basket in 28d
    b28 = bs[bs.d > day-28]
    out["max_basket28"] = b28.groupby("household_key").sp.max()
    out["n_trips14"] = b28[b28.d > day-14].groupby("household_key").size()
    # gap regularity: std of trip days in 84d
    b84 = bs[bs.d > day-84].reset_index()
    out["gap_std_84"] = b84.groupby("household_key").d.apply(lambda s: s.sort_values().diff().std())
    # dept HHI 84d
    t84 = t[t.day > day-84].merge(v.table("products")[["product_id","department"]], on="product_id", how="left")
    ds = t84.groupby(["household_key","department"]).sales_value.sum().reset_index()
    tot = ds.groupby("household_key").sales_value.transform("sum")
    ds["sh"] = (ds.sales_value/tot)**2
    out["dept_hhi"] = ds.groupby("household_key").sh.sum()
    # private label share 84d
    t84b = t84.merge(v.table("products")[["product_id","brand"]], on="product_id", how="left", suffixes=("","_b"))
    sh_pl = t84b.assign(pl=(t84b.brand=="Private").astype(float)).groupby("household_key").apply(
        lambda g: (g.sales_value*g.pl).sum()/g.sales_value.sum())
    out["pl_share"] = sh_pl
    # quantity per trip 28d
    q = t[t.day > day-28].groupby("household_key").quantity.sum()
    out["qty28"] = q
    return out

cf = build(431)
pr2 = pr.set_index("household_key").join(cf)
sub = pr2[pr2.days_since_last<=28]
print("corr(resid, cand) | corr(cand, target) among dsl<=28 (n=%d):" % len(sub))
for c in cf.columns:
    r1 = np.corrcoef(sub[c].fillna(sub[c].median()), sub.resid)[0,1]
    r2 = np.corrcoef(sub[c].fillna(sub[c].median()), sub.future_spend_4w)[0,1]
    print(f"  {c:14s} resid {r1:+.3f}   target {r2:+.3f}")


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

f = A.load_saved("feats_v3.parquet").drop(columns=["index"])
tt = A.train_targets()
df = tt.merge(f, on=["household_key","snapshot_day"], how="inner")
FEATS = [c for c in f.columns if c not in ("household_key","snapshot_day")]

m = xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=4, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0, n_jobs=4,
    objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist")
m.fit(df[FEATS], df.future_spend_4w)

val = f[f.snapshot_day>=459].copy()
pred = m.predict(val[FEATS])
w = 0.3
val["prediction"] = (1-w)*pred + w*val.exp4w_blend.values
out = val[["household_key","snapshot_day","prediction"]].copy()
out["household_key"]=out.household_key.astype(int); out["snapshot_day"]=out.snapshot_day.astype(int)
print(out.shape, out.prediction.describe().round(1).to_dict())
p = A.save_table(out, "pred_e007.parquet")
print(p)
