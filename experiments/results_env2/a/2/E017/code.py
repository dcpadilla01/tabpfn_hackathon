
import agent_api as A
import pandas as pd

for name in ["oof_e008","oof_e016_cv","oof_harness","oof_pt","pred_e008","pred_e009","pred_e014","pred_e016","pred_log1p","pred_pt","pred_seasonal","past_targets","feats_v3","feats_seasonal"]:
    try:
        df = A.load_saved(name+".parquet")
        print("==", name, df.shape)
        print(df.head(3))
        print("cols:", list(df.columns)[:20])
    except Exception as e:
        print("==", name, "ERR", type(e).__name__, e)


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet")
print(oof.shape, oof.snapshot_day.unique())
print(oof.head())
comps = ["med_v3","med_all","hgbq_v3","hgbq_all"]
for c in comps:
    m = np.abs(oof[c]-oof["y"]).mean()
    print(c, "MAE", round(m,3))

# current E016 blend weights: 0.4 medXGB(v3) + ... need to recall; test some blends
def mae(p): return np.abs(p-oof["y"]).mean()
print("blend .5med_v3+.5hgbq_all:", mae(.5*oof.med_v3+.5*oof.hgbq_all))
print("blend .4med_v3+.3med_all+.3hgbq_all:", mae(.4*oof.med_v3+.3*oof.med_all+.3*oof.hgbq_all))
print("mean of 4:", mae(oof[comps].mean(axis=1)))

# per-snapshot bias
oof["resid"] = oof["y"] - oof["med_all"]
print(oof.groupby("snapshot_day").resid.agg(["mean","median","count"]))

# calibration by prediction bucket
oof["bucket"] = pd.qcut(oof["med_all"], 10, duplicates="drop")
print(oof.groupby("bucket", observed=True).agg(pred_med=("med_all","median"), y_med=("y","median"), y_mean=("y","mean"), n=("y","size")))


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression

oof = A.load_saved("oof_e016_cv.parquet")
oof8 = A.load_saved("oof_e008.parquet")[["household_key","snapshot_day","oof_sq","oof_med","oof_log"]]
oof = oof.merge(oof8, on=["household_key","snapshot_day"], how="left")
y = oof["y"].values
comps = ["med_v3","med_all","hgbq_v3","hgbq_all","oof_sq","oof_med","oof_log"]
X = oof[comps].fillna(0).values

def mae(p): return np.abs(p-y).mean()
print("best single hgbq_all:", mae(oof.hgbq_all))

# OLS stacking (non-negative, no intercept)
lr = LinearRegression(positive=True, fit_intercept=False).fit(X, y)
print("stack w:", dict(zip(comps, lr.coef_.round(3))))
print("stack MAE:", mae(lr.predict(X)))

# household aggregation: does averaging OOF preds per household reduce MAE?
for c in ["med_v3","hgbq_all","med_all"]:
    g = oof.groupby("household_key")[c].transform("mean")
    print("hh-mean of", c, "MAE:", round(mae(g),3))

# blend of household-mean and per-row
for c in ["hgbq_all","med_v3"]:
    g = oof.groupby("household_key")[c].transform("mean")
    for w in [0.2,0.3,0.4]:
        print(f"({w}hhmean+{1-w}{c}) {c}:", round(mae(w*g+(1-w)*oof[c]),3))

# top-bucket check: shrink/expansion factor on high preds
p = oof.hgbq_all.values
for f in [0.9,0.95,1.0,1.05,1.1]:
    mask = p>300
    q = p.copy(); q[mask]*=f
    print(f"scale top(>300) by {f}: MAE {mae(q):.3f} (n_top={mask.sum()})")


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet")
# reconstruct E016 blend on OOF: find weights matching MAE 61.651
y = oof["y"].values
def mae(p): return np.abs(p-y).mean()
cands = {
 "0.4med_v3+0.3med_all+0.3hgbq_all": .4*oof.med_v3+.3*oof.med_all+.3*oof.hgbq_all,
 "0.4med_v3+0.3hgbq_v3+0.3hgbq_all": .4*oof.med_v3+.3*oof.hgbq_v3+.3*oof.hgbq_all,
 "0.4med_v3+0.6hgbq_all": .4*oof.med_v3+.6*oof.hgbq_all,
 "0.5med_v3+0.5hgbq_all": .5*oof.med_v3+.5*oof.hgbq_all,
}
for k,v in cands.items(): print(k, round(mae(v),3))

# simulate household-mean shrinkage with train-only hhmean, evaluated on later snapshots
oof["e016"] = cands["0.4med_v3+0.3med_all+0.3hgbq_all"]
for cut in [319, 347]:
    tr = oof[oof.snapshot_day<=cut]
    hh = tr.groupby("household_key")["e016"].mean().rename("hhmean")
    ev = oof[oof.snapshot_day>cut].merge(hh, on="household_key", how="left")
    ev["hhmean"] = ev["hhmean"].fillna(ev["e016"])
    print(f"--- cut {cut}: base MAE {np.abs(ev.e016-ev.y).mean():.3f}, n_hh_missing {ev.hhmean.isna().sum()}")
    for w in [0.2,0.3,0.4,0.5,0.6,0.7]:
        p = w*ev.hhmean + (1-w)*ev.e016
        print(f"  w={w}: MAE {np.abs(p-ev.y).mean():.3f}")


# ---- cell ----

import agent_api as A
import pandas as pd
f3 = A.load_saved("feats_v3.parquet")
print(list(f3.columns))
fs = A.load_saved("feats_seasonal.parquet")
print(list(fs.columns))


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np

# Build household-level long-run profile features from snapshot(459) history,
# then attach to all snapshot rows by household_key.
import agent_api as A
v = A.snapshot()  # capped at day 459
tx = v.table("transactions")
print("tx shape", tx.shape, "max day", tx.day.max())

g = tx.groupby("household_key")
prof = pd.DataFrame({
    "hh_spend_mean": g.spend_all if False else g.apply(lambda d: d.sales_value.sum()),
})


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np

v = A.snapshot()
tx = v.table("transactions")
g = tx.groupby("household_key")
prof = pd.DataFrame({
    "hh_spend_sum": g.sales_value.sum(),
    "hh_spend_mean": g.sales_value.mean(),
    "hh_spend_std": g.sales_value.std(),
    "hh_baskets": g.basket_id.nunique(),
    "hh_days_span": g.day.max() - g.day.min() + 1,
    "hh_first_day": g.day.min(),
})
prof["hh_spend_rate"] = prof.hh_spend_sum / prof.hh_days_span
prof["hh_active_frac"] = prof.hh_baskets / prof.hh_days_span
prof = prof.reset_index()
print(prof.shape)
print(prof.describe().round(2).T)

# distribution of hh_spend_rate: is it stable over time? compare first-half vs second-half rate per hh
tx["half"] = (tx.day > 229).astype(int)
h1 = tx[tx.half==0].groupby("household_key").sales_value.sum()
h2 = tx[tx.half==1].groupby("household_key").sales_value.sum()
cmp = pd.concat([h1.rename("h1"), h2.rename("h2")], axis=1).dropna()
cmp["r1"] = cmp.h1/229; cmp["r2"] = cmp.h2/230
print("corr of rates:", cmp.r1.corr(cmp.r2).round(3))
print("mean r1,r2:", cmp.r1.mean().round(3), cmp.r2.mean().round(3))
print("r2/r1 quantiles:", (cmp.r2/cmp.r1).quantile([.1,.25,.5,.75,.9]).round(2).to_dict())


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np

# check views at arbitrary days and raw table columns at 459
v400 = A.snapshot(400)
tx = v400.table("transactions")
print("day400 tx max day:", tx.day.max(), "shape", tx.shape)
cr = v400.table("coupon_redemptions"); print("cr max day", cr.day.max())
v459 = A.snapshot()
c = v459.table("campaigns"); print(c.shape); print(c.head(3)); print("start_day max", c.start_day.max())
ct = v459.table("campaign_targets"); print("campaign_targets", ct.shape, ct.description.nunique())
dm = v459.table("display_mailer"); print("display_mailer", dm.shape, "week max", dm.week_no.max())
print(A.snapshot_days())


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

def prof_fn(view, snapshot_day):
    tx = view.table("transactions")
    g = tx.groupby("household_key")
    prof = pd.DataFrame({
        "prof_spend_sum": g.sales_value.sum(),
        "prof_baskets": g.basket_id.nunique(),
        "prof_first_day": g.day.min(),
        "prof_last_day": g.day.max(),
        "prof_ndays_active": g.day.nunique(),
        "prof_spend_mean_line": g.sales_value.mean(),
    })
    prof["prof_tenure"] = (snapshot_day - prof.prof_first_day + 1).clip(lower=1)
    prof["prof_spend_rate"] = prof.prof_spend_sum / prof.prof_tenure
    prof["prof_active_frac"] = prof.prof_baskets / prof.prof_tenure
    prof["prof_spend_per_active_day"] = prof.prof_spend_sum / prof.prof_ndays_active.clip(lower=1)
    # decayed spend rate (hl=112) over full history
    tx2 = tx[["household_key","day","sales_value"]].copy()
    tx2["w"] = np.exp(-np.log(2)*(snapshot_day-tx2.day)/112.0)
    d = tx2.groupby("household_key").agg(dec=("w","sum"), decsp=("sales_value", lambda s: 0))
    tx2["decsp"] = tx2.sales_value*tx2.w
    d = tx2.groupby("household_key").agg(dec=("w","sum"), decsp=("decsp","sum"))
    prof["prof_decay_rate"] = d.decsp/d.dec.clip(lower=1e-9)
    # share of spend in last 84d
    s84 = tx[tx.day>snapshot_day-84].groupby("household_key").sales_value.sum()
    prof["prof_share_last84"] = (s84/prof.prof_spend_sum).fillna(0)
    return prof.drop(columns=["prof_first_day","prof_last_day"])

feats = A.build_features(prof_fn)
print(feats.shape, feats.snapshot_day.nunique())
print(feats.head(3))
A.save_table(feats, "feats_prof.parquet")


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb

feats = A.load_saved("feats_prof.parquet")
f3 = A.load_saved("feats_v3.parquet")
fs = A.load_saved("feats_seasonal.parquet")
oof16 = A.load_saved("oof_e016_cv.parquet")

df = feats.merge(f3, on=["household_key","snapshot_day"], how="left", suffixes=("","_v3"))
df = df.merge(fs, on=["household_key","snapshot_day"], how="left")
df = df.merge(oof16[["household_key","snapshot_day","y"]], on=["household_key","snapshot_day"], how="left")
print(df.shape, "missing y:", df.y.isna().sum())

prof_cols = [c for c in feats.columns if c.startswith("prof_")]
Xcols = [c for c in f3.columns if c not in ("household_key","snapshot_day")] + \
        [c for c in fs.columns if c not in ("household_key","snapshot_day")] + prof_cols
X = df[Xcols].astype(float).copy()
y = df["y"].values
days = df["snapshot_day"].values

def run_xgb(Xtr, ytr, Xte, seed=7):
    m = xgb.XGBRegressor(n_estimators=900, learning_rate=0.05, max_depth=6, min_child_weight=8,
                         subsample=0.8, colsample_bytree=0.8, objective="reg:quantileerror",
                         quantile_alpha=0.5, tree_method="hist", random_state=seed, n_jobs=8)
    m.fit(Xtr, ytr)
    return m.predict(Xte)

def run_hgb(Xtr, ytr, Xte, seed=7):
    m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=400, learning_rate=0.06,
                                      max_leaf_nodes=31, random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict(Xte)

tr_mask = ~np.isnan(y)
val_mask = ~tr_mask
pred_med = np.zeros(len(df)); pred_hgb = np.zeros(len(df))
for d in [459,487,515,543]:
    m_te = days==d
    pred_med[m_te] = run_xgb(X[tr_mask], y[tr_mask], X[m_te])
    pred_hgb[m_te] = run_hgb(X[tr_mask], y[tr_mask], X[m_te])
def mae(p): return np.abs(p[val_mask]-y[val_mask]).mean()
print("medXGB+prof MAE:", round(mae(pred_med),3))
print("HGB+prof MAE:", round(mae(pred_hgb),3))
print("blend 0.5/0.5:", round(mae(.5*pred_med+.5*pred_hgb),3))
print("blend 0.4/0.6:", round(mae(.4*pred_med+.6*pred_hgb),3))

# save validation predictions of the best variant
out = df.loc[val_mask, ["household_key","snapshot_day"]].copy()
out["prediction"] = 0.5*pred_med[val_mask] + 0.5*pred_hgb[val_mask]
path = A.save_table(out, "pred_prof.parquet")
print("saved", path)


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

feats = A.load_saved("feats_prof.parquet")
f3 = A.load_saved("feats_v3.parquet")
fs = A.load_saved("feats_seasonal.parquet")
oof16 = A.load_saved("oof_e016_cv.parquet")
df = feats.merge(f3, on=["household_key","snapshot_day"], how="left", suffixes=("","_v3"))
df = df.merge(fs, on=["household_key","snapshot_day"], how="left")
df = df.merge(oof16[["household_key","snapshot_day","y"]], on=["household_key","snapshot_day"], how="left")
prof_cols = [c for c in feats.columns if c.startswith("prof_")]
Xcols = [c for c in f3.columns if c not in ("household_key","snapshot_day")] + \
        [c for c in fs.columns if c not in ("household_key","snapshot_day")] + prof_cols
X = df[Xcols].astype(float)
print("X nan cols:", X.columns[X.isna().all()].tolist())
print("X inf count:", np.isinf(X.values).sum())
inf_cols = Xcolumns = [c for c in Xcols if np.isinf(X[c].values).any()]
print("inf cols:", inf_cols)
for c in inf_cols:
    print(c, df[c].describe())


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

feats = A.load_saved("feats_prof.parquet")
f3 = A.load_saved("feats_v3.parquet")
fs = A.load_saved("feats_seasonal.parquet")
oof16 = A.load_saved("oof_e016_cv.parquet")
df = feats.merge(f3, on=["household_key","snapshot_day"], how="left", suffixes=("","_v3"))
df = df.merge(fs, on=["household_key","snapshot_day"], how="left")
df = df.merge(oof16[["household_key","snapshot_day","y"]], on=["household_key","snapshot_day","y"], how="left")
prof_cols = [c for c in feats.columns if c.startswith("prof_")]
Xcols = [c for c in f3.columns if c not in ("household_key","snapshot_day")] + \
        [c for c in fs.columns if c not in ("household_key","snapshot_day")] + prof_cols
X = df[Xcols].astype(float)
y = df["y"].values
days = df["snapshot_day"].values
print("y stats: nan", np.isnan(y).sum(), "min", np.nanmin(y), "max", np.nanmax(y))
print("rows per day:", df.groupby("snapshot_day").size().to_dict())
tr_mask = ~np.isnan(y); val_mask = np.isnan(y)
print("tr", tr_mask.sum(), "val", val_mask.sum())
print("val days:", np.unique(days[val_mask]))
print("tr days:", np.unique(days[tr_mask]))
