import agent_api as A, pandas as pd, numpy as np
print(A.snapshot_days())
f3 = A.load_saved("feats_v3.parquet")
print("feats_v3:", f3.shape)
print(sorted(f3.columns))
oo = A.load_saved("oof_e008.parquet")
print("oof:", oo.shape, list(oo.columns))
print(oo.head(3))
tt = A.train_targets()
print("targets:", tt.shape)
print(tt.groupby("snapshot_day")[A.TARGET].agg(["count","mean","median"]))
p8 = A.load_saved("pred_e008.parquet")
print("pred8:", p8.shape, list(p8.columns))
print(p8.head(3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
oo = A.load_saved("oof_e008.parquet")
tt = A.train_targets()
m = tt.merge(oo, on=["household_key","snapshot_day"])
m["blend"] = 0.25*m.oof_sq + 0.25*m.oof_med + 0.5*m.oof_log
y = m.future_spend_4w.values
for c in ["oof_sq","oof_med","oof_log","blend"]:
    e = np.abs(m[c].values - y)
    print(c, "OOF MAE", round(np.mean(e),3))
print("zero targets share:", (y==0).mean())
for mask,name in [((y==0).values,"zero"), ((y>0).values,"nonzero"), ((y>200).values,">200"), ((y>500).values,">500")]:
    e = np.abs(m["blend"].values[mask] - y[mask])
    print(name, "n", mask.sum(), "MAE", round(np.mean(e),2), "share of total abs err", round(np.sum(e)/np.sum(np.abs(m.blend.values-y)),3))
# error by snapshot
for d,g in m.groupby("snapshot_day"):
    print(d, len(g), round(np.mean(np.abs(g.blend-g.future_spend_4w)),2), "mean y", round(g.future_spend_4w.mean(),1))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
oo = A.load_saved("oof_e008.parquet")
tt = A.train_targets()
m = tt.merge(oo, on=["household_key","snapshot_day"])
y = m.future_spend_4w.values
abs_err = np.abs(m.blend.values - y)
tot = abs_err.sum()
for mask,name in [(y==0,"zero"), (y>200,">200"), (y>500,">500"), (y>1000,">1000")]:
    e = abs_err[mask]
    print(name, "n", mask.sum(), "MAE", round(np.mean(e),2), "err share", round(e.sum()/tot,3))
# best possible blend weights on OOF
from itertools import product
best=(None,1e9)
for w in np.arange(0,1.01,0.05):
    for v in np.arange(0,1.01-w,0.05):
        p = w*m.oof_sq+v*m.oof_med+(1-w-v)*m.oof_log
        e = np.mean(np.abs(p-y))
        if e<best[1]: best=((w,v,1-w-v),e)
print("best OOF blend:", best)
# error by snapshot
for d,g in m.groupby("snapshot_day"):
    print(d, len(g), round(np.mean(np.abs(g.blend-g.future_spend_4w)),2), "mean y", round(g.future_spend_4w.mean(),1))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
oo = A.load_saved("oof_e008.parquet")
tt = A.train_targets()
m = tt.merge(oo, on=["household_key","snapshot_day"])
m["blend"] = 0.25*m["oof_sq"] + 0.25*m["oof_med"] + 0.5*m["oof_log"]
y = m["future_spend_4w"].values
abs_err = np.abs(m["blend"].values - y)
tot = abs_err.sum()
for mask,name in [(y==0,"zero"), (y>200,">200"), (y>500,">500"), (y>1000,">1000")]:
    e = abs_err[mask]
    print(name, "n", mask.sum(), "MAE", round(np.mean(e),2), "err share", round(e.sum()/tot,3))
best=(None,1e9)
for w in np.arange(0,1.01,0.05):
    for v in np.arange(0,1.01-w,0.05):
        p = w*m["oof_sq"]+v*m["oof_med"]+(1-w-v)*m["oof_log"]
        e = np.mean(np.abs(p-y))
        if e<best[1]: best=((w,v,1-w-v),e)
print("best OOF blend:", best)
for d,g in m.groupby("snapshot_day"):
    print(d, len(g), round(np.mean(np.abs(g["blend"]-g["future_spend_4w"])),2), "mean y", round(g["future_spend_4w"].mean(),1))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3[["household_key","snapshot_day","spend_28","spend_56","spend_84","active_28","days_since_last"]], on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
def mae(p): return round(np.mean(np.abs(np.asarray(p)-y)),3)
print("spend_28:", mae(m.spend_28))
print("spend_84:", mae(m.spend_84))
print("0.5*(spend28,med):", mae(0.5*m.spend_28+0.5*m.oof_med))
print("0.7*med+0.3*spend28:", mae(0.7*m.oof_med+0.3*m.spend_28))
print("med only:", mae(m.oof_med))
# conditional median of y given spend_28 bins
bins = pd.qcut(m.spend_28, 12, duplicates="drop")
print(m.groupby(bins, observed=True).agg(y_med=("future_spend_4w","median"), y_mean=("future_spend_4w","mean"), s28=("spend_28","median"), n=("future_spend_4w","size")))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3[["household_key","snapshot_day","spend_28","spend_84","active_28","days_since_last","baskets_28"]], on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
bins = pd.qcut(m.spend_28, 10, duplicates="drop")
g = m.groupby(bins, observed=True).agg(y_med=("future_spend_4w","median"), y_mean=("future_spend_4w","mean"),
    p_sq=("oof_sq","median"), p_med=("oof_med","median"), p_log=("oof_log","median"), n=("future_spend_4w","size"))
print(g)
# MAE within each decile for each model
for c in ["oof_sq","oof_med","oof_log"]:
    e = np.abs(m[c].values-y)
    print(c, [round(e[bins.values==b].mean(),1) for b in sorted(bins.unique(), key=str)])
# zero-spend-28 households: what do models predict?
z = m[m.spend_28==0]
print("spend28==0: n", len(z), "y mean", round(z.future_spend_4w.mean(),2), "y median", z.future_spend_4w.median(),
      "pred med median", z.oof_med.median(), "MAE med", round(np.abs(z.oof_med-z.future_spend_4w).mean(),2))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3[["household_key","snapshot_day","spend_28","spend_84","active_28","days_since_last","baskets_28","gap_mean_112","gap_std_112","active_84"]], on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
def mae(p): return round(np.mean(np.abs(np.asarray(p)-y)),3)
print("med:", mae(m.oof_med))
print("sq:", mae(m.oof_sq))
print("log:", mae(m.oof_log))
print("0.5med+0.5log:", mae(0.5*m.oof_med+0.5*m.oof_log))
print("0.5sq+0.5log:", mae(0.5*m.oof_sq+0.5*m.oof_log))
print("0.6sq+0.4log:", mae(0.6*m.oof_sq+0.4*m.oof_log))
print("0.33sq+0.33med+0.33log:", mae((m.oof_sq+m.oof_med+m.oof_log)/3))
print("0.4sq+0.2med+0.4log:", mae(0.4*m.oof_sq+0.2*m.oof_med+0.4*m.oof_log))
print("0.4sq+0.6log:", mae(0.4*m.oof_sq+0.6*m.oof_log))
print("0.3sq+0.7log:", mae(0.3*m.oof_sq+0.7*m.oof_log))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
pd.set_option("display.width", 200)
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3[["household_key","snapshot_day","spend_28","spend_84","active_28","days_since_last","baskets_28","gap_mean_112","gap_std_112","active_84"]], on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
bins = pd.qcut(m.spend_28, 10, duplicates="drop")
g = m.groupby(bins, observed=True).agg(y_med=("future_spend_4w","median"), y_mean=("future_spend_4w","mean"),
    p_sq=("oof_sq","median"), p_med=("oof_med","median"), p_log=("oof_log","median"), n=("future_spend_4w","size"))
print(g.to_string())
# high-spend households: bias
hi = m[m.spend_28>300]
print("\nspend28>300: n", len(hi), "y mean", round(hi.future_spend_4w.mean(),1), "y med", hi.future_spend_4w.median(),
      "| pred sq", round(hi.oof_sq.median(),1), "med", round(hi.oof_med.median(),1), "log", round(hi.oof_log.median(),1))
print("MAE hi: sq", round(np.abs(hi.oof_sq-hi.future_spend_4w).mean(),1), "med", round(np.abs(hi.oof_med-hi.future_spend_4w).mean(),1), "log", round(np.abs(hi.oof_log-hi.future_spend_4w).mean(),1))
# how much of total error from top decile of spend_28?
abs_err = np.abs(m.oof_med.values - y)
print("\nerr share by spend28 decile (med model):")
for b in sorted(bins.unique(), key=str):
    mask = (bins.values==b)
    print(str(b)[:25], round(abs_err[mask].sum()/abs_err.sum(),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
pd.set_option("display.width", 220)
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3[["household_key","snapshot_day","spend_28","spend_84","active_28","days_since_last","baskets_28","gap_mean_112","gap_std_112","active_84"]], on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
def mae(p): return round(np.mean(np.abs(np.asarray(p)-y)),3)
# Is sq better than med on high spend?
for lo,hi in [(0,50),(50,150),(150,300),(300,600),(600,10**9)]:
    s = m[(m.spend_28>=lo)&(m.spend_28<hi)]
    print(f"s28 [{lo},{hi}) n={len(s)}", 
          "sq", round(np.abs(s.oof_sq-s.future_spend_4w).mean(),1),
          "med", round(np.abs(s.oof_med-s.future_spend_4w).mean(),1),
          "log", round(np.abs(s.oof_log-s.future_spend_4w).mean(),1),
          "y_mean", round(s.future_spend_4w.mean(),1))
# overall best weighted blend using scipy
from scipy.optimize import minimize
def f(w):
    p = w[0]*m.oof_sq + w[1]*m.oof_med + w[2]*m.oof_log
    return np.mean(np.abs(p-y))
r = minimize(f, [0.33,0.34,0.33], method="Nelder-Mead")
print("best simplex blend:", np.round(r.x,3), round(r.fun,3))
# blend of sq and log only
def f2(w): return np.mean(np.abs((w*m.oof_sq+(1-w)*m.oof_log)-y))
r2 = minimize(f2, [0.5], method="Nelder-Mead")
print("sq+log blend:", round(r2.x[0],3), round(r2.fun,3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
pd.set_option("display.width", 220)
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3[["household_key","snapshot_day","spend_28","spend_84","active_28","days_since_last","baskets_28","gap_mean_112","gap_std_112","active_84"]], on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
def mae(p): return round(np.mean(np.abs(np.asarray(p)-y)),3)
# median-objective model bias by quantile
for q in [0.1,0.25,0.5,0.75,0.9]:
    print("q",q, "pred_q", round(np.quantile(m.oof_med,q),1), "y_q", round(np.quantile(y,q),1))
# what if we scale the median predictions slightly?
for s in [1.0,1.05,1.1,1.15,1.2,1.25]:
    print("scale",s, mae(m.oof_med*s))
# shift (MAE-optimal shift)
for sh in [0,2,4,6,8]:
    print("shift",sh, mae(m.oof_med+sh))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
pd.set_option("display.width", 220)
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3[["household_key","snapshot_day","spend_28","spend_84","active_28","days_since_last","baskets_28","gap_mean_112","gap_std_112","active_84"]], on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
def mae(p): return round(np.mean(np.abs(np.asarray(p)-y)),3)
# per-decile optimal scaling of med predictions (to see if heteroscedastic bias)
bins = pd.qcut(m.spend_28, 8, duplicates="drop")
for b in sorted(bins.unique(), key=str):
    s = m[bins.values==b]
    best=(None,1e9)
    for sc in np.arange(0.7,1.5,0.05):
        e = np.mean(np.abs(s.oof_med.values*sc - s.future_spend_4w.values))
        if e<best[1]: best=(round(sc,2),round(e,2))
    print(str(b)[:28], "n",len(s), "y_mean", round(s.future_spend_4w.mean(),1), "p_med", round(s.oof_med.median(),1), "best scale", best)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
pd.set_option("display.width", 220)
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3[["household_key","snapshot_day","spend_28","spend_84","active_28","days_since_last","baskets_28","gap_mean_112","gap_std_112","active_84","active_112","baskets_112"]], on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
def mae(p): return round(np.mean(np.abs(np.asarray(p)-y)),3)
# gap_mean_112: median gap between baskets over 112d
print("corr spend_28 vs y:", round(np.corrcoef(m.spend_28, y)[0,1],3))
# households with long gaps (infrequent): how do they behave?
for lo,hi in [(0,7),(7,14),(14,28),(28,60),(60,10**9)]:
    s = m[(m.gap_mean_112>=lo)&(m.gap_mean_112<hi)&(m.spend_28>0)]
    print(f"gap[{lo},{hi}) n={len(s)} y_med={s.future_spend_4w.median():.0f} y_mean={s.future_spend_4w.mean():.0f} s28_med={s.spend_28.median():.0f} p_med={s.oof_med.median():.0f} MAE_med={np.abs(s.oof_med-s.future_spend_4w).mean():.1f}")
# days_since_last effect
for lo,hi in [(0,7),(7,14),(14,28),(28,56),(56,10**9)]:
    s = m[(m.days_since_last>=lo)&(m.days_since_last<hi)]
    print(f"dsl[{lo},{hi}) n={len(s)} y_med={s.future_spend_4w.median():.0f} p_med={s.oof_med.median():.0f} MAE={np.abs(s.oof_med-s.future_spend_4w).mean():.1f}")


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
pd.set_option("display.width", 220)
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3, on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
def mae(p): return round(np.mean(np.abs(np.asarray(p)-y)),3)
# How well does the model predict ZERO vs nonzero (classification)?
z = (y==0)
from sklearn.metrics import roc_auc_score
print("AUC spend28 -> y>0:", round(roc_auc_score(1-z, m.spend_28),3))
print("AUC pred_med -> y>0:", round(roc_auc_score(1-z, m.oof_med),3))
print("AUC dsl -> y==0:", round(roc_auc_score(z, m.days_since_last),3))
# conditional P(y==0) by spend_28 bins
bins = pd.qcut(m.spend_28, 10, duplicates="drop")
print(m.groupby(bins, observed=True)["future_spend_4w"].agg([("p0", lambda s:(s==0).mean()), ("med", "median"), ("n","size")]))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
m = tt.merge(f3, on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
# check spend_28 == 0 rows: how many, and what's y distribution
z = m[m.spend_28==0]
print("n spend28==0:", len(z), "P(y==0):", round((z.future_spend_4w==0).mean(),3), "y>0 mean:", round(z[z.future_spend_4w>0].future_spend_4w.mean(),1))
# among spend28==0, does days_since_last matter?
for lo,hi in [(0,28),(28,56),(56,84),(84,10**9)]:
    s = z[(z.days_since_last>=lo)&(z.days_since_last<hi)]
    print(f"dsl[{lo},{hi}) n={len(s)} P(y==0)={round((s.future_spend_4w==0).mean(),2)} y>0 mean={round(s[s.future_spend_4w>0].future_spend_4w.mean(),1)}")


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb

f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
p8 = A.load_saved("pred_e008.parquet")
train_days = A.snapshot_days()["train"]; val_days = A.snapshot_days()["validation"]
feat_cols = [c for c in f3.columns if c not in ("household_key","snapshot_day")]
tr = tt.merge(f3, on=["household_key","snapshot_day"]).sort_values(["snapshot_day","household_key"]).reset_index(drop=True)
val = f3[f3.snapshot_day.isin(val_days)].sort_values(["snapshot_day","household_key"]).reset_index(drop=True)
print("train rows", len(tr), "val rows", len(val))

def fit_two_stage(X, y, seed):
    ypos = (y>0).astype(int).values if hasattr(y,"values") else (y>0).astype(int)
    clf = xgb.XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=6, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=4, random_state=seed, eval_metric="logloss")
    clf.fit(X, ypos)
    reg = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, n_estimators=1200, learning_rate=0.03,
        max_depth=7, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=4, random_state=seed)
    m = ypos.astype(bool)
    reg.fit(X[m], np.asarray(y)[m])
    return clf, reg

def pred_two_stage(clfs, regs, X):
    ps = np.mean([c.predict_proba(X)[:,1] for c in clfs], axis=0)
    vs = np.mean([np.clip(r.predict(X),0,None) for r in regs], axis=0)
    return ps*vs, ps, vs

# ---- OOF check on last-4 train snapshots (same split as oof_e008) ----
fit = tr[tr.snapshot_day < 347]; hold = tr[tr.snapshot_day >= 347].reset_index(drop=True)
clfs, regs = [], []
for seed in (7, 42):
    c, r = fit_two_stage(fit[feat_cols].values, fit["future_spend_4w"].values, seed)
    clfs.append(c); regs.append(r)
stageA, p, v = pred_two_stage(clfs, regs, hold[feat_cols].values)
y = hold["future_spend_4w"].values
h = hold[["household_key","snapshot_day"]].merge(oo, on=["household_key","snapshot_day"])
base_med = h["oof_med"].values; base_mix = 0.5*(h["oof_sq"].values+h["oof_med"].values)
def mae(pr): return round(float(np.mean(np.abs(np.asarray(pr)-y))),3)
print("OOF holdout MAE (n=%d):"%len(y))
print("  oof_med", mae(base_med), "| oof_sq", mae(h["oof_sq"].values), "| oof_log", mae(h["oof_log"].values))
cands = {
 "stageA": stageA,
 "stageB_hard0": np.where(p<0.5, 0.0, v),
 "blendA50_med": 0.5*stageA+0.5*base_med,
 "blendA30_med": 0.3*stageA+0.7*base_med,
 "blendA70_med": 0.7*stageA+0.3*base_med,
 "blendA50_mix": 0.5*stageA+0.5*base_mix,
}
for k,pr in cands.items(): print("  %-14s"%k, mae(pr))
best = min(cands, key=lambda k: mae(cands[k]))
print("BEST OOF variant:", best)

# ---- final: train on ALL train snapshots, predict validation ----
clfsF, regsF = [], []
for seed in (7, 42):
    c, r = fit_two_stage(tr[feat_cols].values, tr["future_spend_4w"].values, seed)
    clfsF.append(c); regsF.append(r)
stageA_v, p_v, v_v = pred_two_stage(clfsF, regsF, val[feat_cols].values)
pv8 = val[["household_key","snapshot_day"]].merge(p8, on=["household_key","snapshot_day"], how="left")["prediction"].values
print("val pred_e008 aligned NaNs:", int(np.isnan(pv8).sum()))
vmap = {
 "stageA": stageA_v,
 "stageB_hard0": np.where(p_v<0.5, 0.0, v_v),
 "blendA50_med": 0.5*stageA_v+0.5*pv8,
 "blendA30_med": 0.3*stageA_v+0.7*pv8,
 "blendA70_med": 0.7*stageA_v+0.3*pv8,
 "blendA50_mix": 0.5*stageA_v+0.5*pv8,
}
out = val[["household_key","snapshot_day"]].copy()
out["prediction"] = vmap[best]
print("saved variant:", best, "| val pred mean", round(out.prediction.mean(),2), "| rows", len(out))
path = A.save_table(out, "pred_e009.parquet")
print(path)
