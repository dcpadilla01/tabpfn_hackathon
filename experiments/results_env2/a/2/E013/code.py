import agent_api, pandas as pd, numpy as np
f3 = agent_api.load_saved("feats_v3.parquet")
print("feats_v3", f3.shape)
print(list(f3.columns))
oof = agent_api.load_saved("oof_e008.parquet")
print("\noof_e008", oof.shape, list(oof.columns))
print(oof.head(3))
p8 = agent_api.load_saved("pred_e008.parquet")
print("\npred_e008", p8.shape, list(p8.columns))
print(p8.head(3))
tt = agent_api.train_targets()
print("\ntrain_targets", tt.shape)
print(tt.future_spend_4w.describe())
print("\nsnapshot_days", agent_api.snapshot_days())


# ---- cell ----
import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e008.parquet")
tt = agent_api.train_targets()
df = oof.merge(tt, on=["household_key","snapshot_day"])
print(df.shape)
df["err_sq"] = df.oof_sq - df.future_spend_4w
df["err_med"] = df.oof_med - df.future_spend_4w
df["err_log"] = df.oof_log - df.future_spend_4w
df["eq"] = (df.oof_sq+df.oof_med+df.oof_log)/3
print("\nMAE by learner (OOF):")
for c in ["oof_sq","oof_med","oof_log","eq"]:
    print(c, round(np.abs(df[c]-df.future_spend_4w).mean(),3))
print("\nMean target & mean pred & mean err by snapshot day:")
g = df.groupby("snapshot_day").agg(t=("future_spend_4w","mean"), sq=("oof_sq","mean"), med=("oof_med","mean"), log=("oof_log","mean"))
g["err_eq"] = (g.sq+g.med+g.log)/3 - g.t
print(g.round(2))
# grid search blend weights on OOF for MAE
best=None
for a in np.arange(0,1.01,0.05):
    for b in np.arange(0,1.01-a+1e-9,0.05):
        c = 1-a-b
        p = a*df.oof_sq+b*df.oof_med+c*df.oof_log
        m = np.abs(p-df.future_spend_4w).mean()
        if best is None or m<best[0]: best=(m,a,b,c)
print("\nbest OOF blend weights (sq,med,log):", best)
# equal-weight OOF MAE for reference
print("equal:", np.abs(df.eq-df.future_spend_4w).mean())


# ---- cell ----
import agent_api, pandas as pd, numpy as np
f3 = agent_api.load_saved("feats_v3.parquet")
oof = agent_api.load_saved("oof_e008.parquet")
tt = agent_api.train_targets()
df = oof.merge(tt, on=["household_key","snapshot_day"]).merge(
    f3[["household_key","snapshot_day","spend_28","spend_56","spend_84","days_since_last","active_28","avg_basket_84","has_demo","week_of_year"]],
    on=["household_key","snapshot_day"])
y = df.future_spend_4w
print("OOF MAE: med", round(np.abs(df.oof_med-y).mean(),3))
# persistence baselines
for c in ["spend_28","spend_56","spend_84"]:
    print("persist", c, round(np.abs(df[c]-y).mean(),3))
print("persist blend 0.5*28+0.3*56+0.2*84:", round(np.abs((0.5*df.spend_28+0.3*df.spend_56+0.2*df.spend_84)-y).mean(),3))
# optimal global scale for med
for s in [0.95,1.0,1.05,1.1,1.15,1.2]:
    print("scale",s, round(np.abs(s*df.oof_med-y).mean(),3))
# error by segment
df["seg"] = pd.cut(df.spend_28, [-1,1,50,150,300,100000], labels=["0","low","mid","high","vhigh"])
print("\nMAE(oof_med) & mean err & n by spend_28 seg:")
print(df.groupby("seg", observed=True).apply(lambda g: pd.Series({
    "n":len(g), "mae":np.abs(g.oof_med-g.future_spend_4w).mean(), "bias":(g.oof_med-g.future_spend_4w).mean()}), include_groups=False).round(2))
df["seg2"] = pd.cut(df.days_since_last, [-1,7,14,28,56,10000], labels=["<=7","8-14","15-28","29-56",">56"])
print("\nby days_since_last:")
print(df.groupby("seg2", observed=True).apply(lambda g: pd.Series({
    "n":len(g), "mae":np.abs(g.oof_med-g.future_spend_4w).mean(), "bias":(g.oof_med-g.future_spend_4w).mean()}), include_groups=False).round(2))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e008.parquet")
tt = agent_api.train_targets()
df = oof.merge(tt, on=["household_key","snapshot_day"])
y = df.future_spend_4w.values
med = df.oof_med.values; sq = df.oof_sq.values
print("base MAE med", round(np.abs(med-y).mean(),3), " sq", round(np.abs(sq-y).mean(),3))
for off in [0,5,10,15,17,20,25,30]:
    print("offset",off, "med:", round(np.abs(med+off-y).mean(),3), " sq:", round(np.abs(sq+off-y).mean(),3))
# per-snapshot optimal offset for med
print("\nper-snapshot optimal offset (med):")
for d,g in df.groupby("snapshot_day"):
    yy=g.future_spend_4w.values; mm=g.oof_med.values
    offs=np.arange(0,60,1); maes=[np.abs(mm+o-yy).mean() for o in offs]
    i=int(np.argmin(maes))
    print(d, "n=",len(g), "opt_off=",offs[i], "mae=",round(maes[i],2), "base=",round(np.abs(mm-yy).mean(),2), "mean_bias=",round((mm-yy).mean(),2))
# offset by spend_28 segment (need feats)
f3 = agent_api.load_saved("feats_v3.parquet")
df2 = df.merge(f3[["household_key","snapshot_day","spend_28"]], on=["household_key","snapshot_day"])
print("\nper spend_28 bin optimal offset (med):")
bins=[-1,1,50,150,300,1e9]
df2["b"]=pd.cut(df2.spend_28,bins,labels=["0","low","mid","high","vhigh"])
for b,g in df2.groupby("b",observed=True):
    yy=g.future_spend_4w.values; mm=g.oof_med.values
    offs=np.arange(0,80,2); maes=[np.abs(mm+o-yy).mean() for o in offs]
    i=int(np.argmin(maes))
    print(b, "n=",len(g), "opt_off=",offs[i], "mae=",round(maes[i],2), "base=",round(np.abs(mm-yy).mean(),2))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
print("mean/median target by train snapshot day:")
g = tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median","count"])
print(g.round(2))
# weekly aggregate spend seasonality (capped view at 459)
v = agent_api.snapshot(459)
tr = v.transactions
wk = tr.groupby("week_no").sales_value.sum()
print("\nweekly total spend, first 30 weeks:"); print(wk.head(30).round(0).to_dict())
print("weekly total spend, weeks 50-80:"); print(wk.loc[50:80].round(0).to_dict())
print("\nn weeks:", wk.shape[0], "min/max week:", wk.index.min(), wk.index.max())


# ---- cell ----
import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e008.parquet")
tt = agent_api.train_targets()
df = oof.merge(tt, on=["household_key","snapshot_day"])
y = df.future_spend_4w.values; med = df.oof_med.values
# calibration: bin pred, compare pred vs median/mean target in bin
bins = np.quantile(med, np.arange(0,1.01,0.1))
df["pb"] = pd.cut(med, bins, include_lowest=True, duplicates="drop")
cal = df.groupby("pb", observed=True).apply(lambda g: pd.Series({
    "n":len(g), "pred_med":g.oof_med.median(), "y_med":g.future_spend_4w.median(),
    "y_mean":g.future_spend_4w.mean(), "mae":np.abs(g.oof_med-g.future_spend_4w).mean()}), include_groups=False)
print(cal.round(2))
# capping test
for cap in [400,600,800,1000,1200,1500,1e9]:
    print("cap",cap, round(np.abs(np.minimum(med,cap)-y).mean(),3))
print("y quantiles:", np.quantile(y,[0.9,0.95,0.99,0.999]).round(1), "max", y.max())
print("pred quantiles:", np.quantile(med,[0.9,0.95,0.99,0.999]).round(1), "max", med.max())


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time

f3 = agent_api.load_saved("feats_v3.parquet")
obj = [c for c in f3.columns if f3[c].dtype == object]
print("object cols:", obj)
print(f3[[c for c in f3.columns if c.startswith("dem")]].head(3))

LAGS = [84, 112, 168, 252, 308, 364]

def fn(view, snapshot_day):
    d = snapshot_day
    hh = view.households
    if hasattr(hh, "columns"):
        if "household_key" in list(hh.columns):
            keys = pd.Index(hh["household_key"].astype(int))
        else:
            keys = pd.Index(hh.index.astype(int))
    else:
        keys = pd.Index([int(k) for k in hh])
    if d == 95:
        print("hh type:", type(hh), "n_keys:", len(keys))
    tr = view.table("transactions")
    out = pd.DataFrame(index=keys)
    for L in LAGS:
        if d >= L:
            lo, hi = d + 1 - L, d + 28 - L   # same 4-week window, L days earlier
            m = tr[(tr.day >= lo) & (tr.day <= hi)]
            g = m.groupby("household_key").agg(sp=("sales_value", "sum"), bk=("basket_id", "nunique"))
            out[f"lag{L}_spend"] = g["sp"].reindex(keys).fillna(0.0)
            out[f"lag{L}_bask"] = g["bk"].reindex(keys).fillna(0.0)
        else:
            out[f"lag{L}_spend"] = np.nan
            out[f"lag{L}_bask"] = np.nan
    if d >= 391:  # 8-week window one year ago
        m = tr[(tr.day >= d - 390) & (tr.day <= d - 308)]
        g = m.groupby("household_key")["sales_value"].sum()
        out["lag364_8w"] = g.reindex(keys).fillna(0.0)
    else:
        out["lag364_8w"] = np.nan
    return out

t0 = time.time()
fn_out = agent_api.build_features(fn)
print("build time", round(time.time()-t0,1), "shape:", fn_out.shape)
fn_out = fn_out.reset_index(drop=True) if fn_out.index.name else fn_out
print(fn_out.head(3))
# coverage by snapshot
cov = fn_out.groupby("snapshot_day")[["lag84_spend","lag168_spend","lag252_spend","lag308_spend","lag364_spend","lag364_8w"]].apply(lambda g: g.notna().mean().round(2))
print(cov)
agent_api.save_table(fn_out, "feats_seasonal.parquet")
print("saved")


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time
f3 = agent_api.load_saved("feats_v3.parquet")
fs = agent_api.load_saved("feats_seasonal.parquet")
m = f3.merge(fs, on=["household_key","snapshot_day"], how="inner")
print("merged:", m.shape, "dtypes non-numeric:", sum(m.dtypes != "number"))
agent_api.save_table(m, "feats_v6.parquet")
tt = agent_api.train_targets()
tr = m[m.snapshot_day.isin(agent_api.snapshot_days()["train"])].merge(tt, on=["household_key","snapshot_day"])
print("train rows:", tr.shape, "val rows:", m[m.snapshot_day>=459].shape)

import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
FEATS = [c for c in m.columns if c not in ("household_key","snapshot_day")]
Xtr = tr[FEATS].values; ytr = tr.future_spend_4w.values
t0=time.time()
mdl = xgb.XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=7, min_child_weight=10,
                       subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                       objective="reg:quantileerror", quantile_alpha=0.5, n_jobs=8, random_state=0)
mdl.fit(Xtr, ytr)
print("xgb 800 fit time:", round(time.time()-t0,1))
t0=time.time()
h = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, loss="quantile", quantile=0.5, random_state=0)
h.fit(Xtr, ytr)
print("hgb 400 fit time:", round(time.time()-t0,1))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time
m = agent_api.load_saved("feats_v6.parquet")
sd = agent_api.snapshot_days()
tt = agent_api.train_targets()
tr = m[m.snapshot_day.isin(sd["train"])].merge(tt, on=["household_key","snapshot_day"])
va = m[m.snapshot_day.isin(sd["validation"])]
FEATS = [c for c in m.columns if c not in ("household_key","snapshot_day")]
print("n features:", len(FEATS))
Xtr, ytr = tr[FEATS].values, tr.future_spend_4w.values
Xva = va[FEATS].values
t0 = time.time()
preds = []
for seed in [0, 1]:
    mdl = xgb.XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=7, min_child_weight=10,
                           subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                           objective="reg:quantileerror", quantile_alpha=0.5, n_jobs=8, random_state=seed)
    mdl.fit(Xtr, ytr); preds.append(mdl.predict(Xva))
    print("seed", seed, "done", round(time.time()-t0,1))
p = np.mean(preds, axis=0)
out = va[["household_key","snapshot_day"]].copy(); out["prediction"] = p
agent_api.save_table(out, "pred_seasonal.parquet")
print("saved", out.shape)


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time, xgboost as xgb
m = agent_api.load_saved("feats_v6.parquet")
sd = agent_api.snapshot_days()
tt = agent_api.train_targets()
tr = m[m.snapshot_day.isin(sd["train"])].merge(tt, on=["household_key","snapshot_day"])
va = m[m.snapshot_day.isin(sd["validation"])]
FEATS = [c for c in m.columns if c not in ("household_key","snapshot_day")]
print("n features:", len(FEATS))
Xtr, ytr = tr[FEATS].values, tr.future_spend_4w.values
Xva = va[FEATS].values
t0 = time.time()
preds = []
for seed in [0, 1]:
    mdl = xgb.XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=7, min_child_weight=10,
                           subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                           objective="reg:quantileerror", quantile_alpha=0.5, n_jobs=8, random_state=seed)
    mdl.fit(Xtr, ytr); preds.append(mdl.predict(Xva))
    print("seed", seed, "done", round(time.time()-t0,1))
p = np.mean(preds, axis=0)
out = va[["household_key","snapshot_day"]].copy(); out["prediction"] = p
agent_api.save_table(out, "pred_seasonal.parquet")
print("saved", out.shape)
