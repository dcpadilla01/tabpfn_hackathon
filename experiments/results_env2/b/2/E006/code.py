
import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days

df = load_saved("rfm_cadence_v1.parquet")
print("E003 table:", df.shape)
print(df.dtypes.value_counts())
tt = train_targets()
days = snapshot_days(); tr, va = days["train"], days["validation"]
m = df.merge(tt, on=["household_key","snapshot_day"], how="left")
print("unmatched:", m["future_spend_4w"].isna().sum())
y = m["future_spend_4w"].values
print("target mean/median/pct0:", np.nanmean(y).round(2), np.nanmedian(y), round(np.nanmean(y==0),3))
print("mean target by snapshot day:")
print(m.groupby("snapshot_day")["future_spend_4w"].mean().round(1))

cat_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("category")]
print("cat cols:", cat_cols)
num_cols = [c for c in df.columns if c not in cat_cols + ["household_key","snapshot_day"]]
print("num cols:", num_cols)

Xn = m[num_cols].copy()
trm = m.snapshot_day.isin(tr).values; vam = m.snapshot_day.isin(va).values
med = Xn[trm].median()
Xn = Xn.fillna(med)
Xd = pd.get_dummies(m[cat_cols].astype(str)) if cat_cols else pd.DataFrame(index=m.index)
X = pd.concat([Xn, Xd.astype(float)], axis=1).astype(float)
mu = X[trm].mean(); sd = X[trm].std().replace(0,1)
Xs = ((X-mu)/sd).fillna(0).values
Xtr, ytr = Xs[trm], y[trm]; Xva, yva = Xs[vam], y[vam]
print("ridge proxy (harness E003 MAE=61.131):")
for a in [1, 10, 100, 1000]:
    w = np.linalg.solve(Xtr.T@Xtr + a*np.eye(Xtr.shape[1]), Xtr.T@ytr)
    print(" alpha", a, "val MAE", round(np.mean(np.abs(Xva@w - yva)),3))
# naive predictors
for c in num_cols:
    if "spend" in c and ("28" in c or "w1" in c):
        print("naive", c, round(np.mean(np.abs(m.loc[vam,c].values - yva)),3))
cors = []
for j in range(Xtr.shape[1]):
    if Xtr[:,j].std() > 0:
        cors.append((X.columns[j], np.corrcoef(Xtr[:,j], ytr)[0,1]))
cors.sort(key=lambda t: -abs(t[1]))
print("top |corr| with target:")
for n,c in cors[:20]: print("  ", n, round(c,3))


# ---- cell ----

import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days

df = load_saved("rfm_cadence_v1.parquet")
tt = train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
days = snapshot_days(); tr_all = days["train"]
fit_days = [d for d in tr_all if d <= 347]
hold_days = [d for d in tr_all if d > 347]
print("fit days:", fit_days, "hold days:", hold_days, "rows:", len(m))

num_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
X = m[num_cols].astype(float)
fitm = m.snapshot_day.isin(fit_days).values
holdm = m.snapshot_day.isin(hold_days).values
y = m["future_spend_4w"].values.astype(float)

med = pd.Series(X[fitm].median(), index=num_cols)
X = X.fillna(med)
mu = X[fitm].mean(); sd = X[fitm].std().replace(0,1)
Xs = ((X-mu)/sd).values
Xf, yf = Xs[fitm], y[fitm]; Xh, yh = Xs[holdm], y[holdm]

def ridge(Xf, yf, Xh, yh, alphas=(3,10,30,100,300)):
    G = Xf.T@Xf; b = Xf.T@yf; n = Xf.shape[1]
    best = None
    for a in alphas:
        w = np.linalg.solve(G + a*np.eye(n), b)
        mae = np.mean(np.abs(Xh@w - yh))
        if best is None or mae < best[1]: best = (a, mae, w)
    return best

a, mae, w = ridge(Xf, yf, Xh, yh)
print(f"E003 proxy ridge: alpha={a} holdout MAE={mae:.3f}  (harness val MAE 61.131)")
print("naive spend_28d holdout MAE:", round(np.mean(np.abs(m.loc[holdm,'spend_28d'].values - yh)),3))
print("naive ew_spend_hl28 holdout MAE:", round(np.mean(np.abs(m.loc[holdm,'ew_spend_hl28'].values - yh)),3))
print("target std:", round(y.std(),2), "mean:", round(y.mean(),2))


# ---- cell ----

import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days

df = load_saved("rfm_cadence_v1.parquet")
m = df.merge(train_targets(), on=["household_key","snapshot_day"], how="inner")
tr_all = snapshot_days()["train"]
fitm = m.snapshot_day.isin([d for d in tr_all if d<=347]).values
holdm = ~fitm
num_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
y = m["future_spend_4w"].values.astype(float)

X = m[num_cols].astype(float)
print("NaN counts top:", X.isna().sum().sort_values(ascending=False).head(8).to_dict())
print("max |value| per col (top 8):")
print(X.abs().max().sort_values(ascending=False).head(8))

med = pd.Series(X[fitm].median(), index=num_cols)
Xf_raw = X[fitm].fillna(med); Xh_raw = X[holdm].fillna(med)
mu = Xf_raw.mean(); sd = Xf_raw.std().replace(0,1)
Xf = ((Xf_raw-mu)/sd).values; Xh = ((Xh_raw-mu)/sd).values
yf, yh = y[fitm], y[holdm]
print("any nan in Xf/Xh:", np.isnan(Xf).any(), np.isnan(Xh).any())

G = Xf.T@Xf; b = Xf.T@yf; n = Xf.shape[1]
for a in [1,3,10,30,100,300,1000,3000,1e4,1e5,1e6]:
    w = np.linalg.solve(G + a*np.eye(n), b)
    p = Xh@w
    print(f"alpha={a:>8.0f} holdMAE={np.mean(np.abs(p-yh)):8.3f} corr={np.corrcoef(p,yh)[0,1]:.3f} predmean={p.mean():7.1f}")
# single-feature ridge on spend_28d
j = num_cols.index("spend_28d")
w1 = np.linalg.solve(G[j,j]+np.array([1,10,100]), b[j])
print("1feat:", [(round(float(np.mean(np.abs(x*Xh[:,j]-yh))),2)) for x in w1])


# ---- cell ----

import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days

df = load_saved("rfm_cadence_v1.parquet")
m = df.merge(train_targets(), on=["household_key","snapshot_day"], how="inner")
tr_all = snapshot_days()["train"]
fitm = m.snapshot_day.isin([d for d in tr_all if d<=347]).values
holdm = ~fitm
num_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
y = m["future_spend_4w"].values.astype(float)

X = m[num_cols].astype(float)
med = pd.Series(X[fitm].median(), index=num_cols)
Xf_raw = X[fitm].fillna(med); Xh_raw = X[holdm].fillna(med)
mu = Xf_raw.mean(); sd = Xf_raw.std().replace(0,1)
Xf = ((Xf_raw-mu)/sd).values; Xh = ((Xh_raw-mu)/sd).values
yf, yh = y[fitm], y[holdm]
ym = yf.mean(); yf0 = yf - ym

G = Xf.T@Xf; b = Xf.T@yf0; n = Xf.shape[1]
best = None
for a in [0.3,1,3,10,30,100,300,1000,3000]:
    w = np.linalg.solve(G + a*np.eye(n), b)
    p = Xh@w + ym
    mae = np.mean(np.abs(p-yh))
    print(f"alpha={a:>6} holdMAE={mae:8.3f} corr={np.corrcoef(p,yh)[0,1]:.3f}")
    if best is None or mae < best[1]: best = (a,mae)
print("best:", best, "| harness E003 val MAE = 61.131")


# ---- cell ----

import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days

df = load_saved("rfm_cadence_v1.parquet")
m = df.merge(train_targets(), on=["household_key","snapshot_day"], how="inner")
tr_all = snapshot_days()["train"]
fitm = m.snapshot_day.isin([d for d in tr_all if d<=347]).values
holdm = ~fitm
y = m["future_spend_4w"].values.astype(float)
base_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]

def ridge_eval(Xf, Xh, yf, yh, alphas=(1,3,10,30,100,300,1000,3000,10000)):
    ym = yf.mean(); yf0 = yf - ym
    G = Xf.T@Xf; b = Xf.T@yf0; n = Xf.shape[1]
    best = (None, 1e18)
    for a in alphas:
        w = np.linalg.solve(G + a*np.eye(n), b)
        p = Xh@w + ym
        mae = np.mean(np.abs(p-yh))
        if mae < best[1]: best = (a, mae)
    return best

def prep(cols, transform=None):
    X = m[cols].astype(float)
    if transform == "log":
        X = np.log1p(X.clip(lower=0))
    med = pd.Series(X[fitm].median(), index=cols)
    Xf_raw = X[fitm].fillna(med); Xh_raw = X[holdm].fillna(med)
    mu = Xf_raw.mean(); sd = Xf_raw.std().replace(0,1)
    return ((Xf_raw-mu)/sd).values, ((Xh_raw-mu)/sd).values, y[fitm], y[holdm]

Xf, Xh, yf, yh = prep(base_cols)
print("raw E003:", ridge_eval(Xf, Xh, yf, yh))
Xf, Xh, yf, yh = prep(base_cols, "log")
print("log E003:", ridge_eval(Xf, Xh, yf, yh))

# log + a few log-ratios
X = m[base_cols].astype(float)
L = np.log1p(X.clip(lower=0))
extra = pd.DataFrame(index=m.index)
extra["lr_w1_w2"] = L["spend_w1"] - L["spend_w2"]
extra["lr_ew_usual"] = L["ew_spend_hl28"] - L["usual_4w"]
extra["lr_rec_long"] = L["spend_84d"] - L["spend_364d"]
extra["lr_w1_usual"] = L["spend_w1"] - L["usual_4w"]
extra["lr_w12_w34"] = L["spend_w1"].add(L["spend_w2"], fill_value=0) - L["spend_w3"].add(L["spend_w4"], fill_value=0)
X2 = pd.concat([L, extra], axis=1)
cols2 = list(X2.columns)
med = pd.Series(X2[fitm].median(), index=cols2)
Xf_raw = X2[fitm].fillna(med); Xh_raw = X2[holdm].fillna(med)
mu = Xf_raw.mean(); sd = Xf_raw.std().replace(0,1)
Xf = ((Xf_raw-mu)/sd).values; Xh = ((Xh_raw-mu)/sd).values
print("log+ratios:", ridge_eval(Xf, Xh, yf, yh))


# ---- cell ----

import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days

df = load_saved("rfm_cadence_v1.parquet")
m = df.merge(train_targets(), on=["household_key","snapshot_day"], how="inner")
tr_all = snapshot_days()["train"]
fitm = m.snapshot_day.isin([d for d in tr_all if d<=347]).values
holdm = ~fitm
y = m["future_spend_4w"].values.astype(float)
base_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]

def ridge_eval(Xf, Xh, yf, yh, alphas=(1,3,10,30,100,300,1000,3000,10000)):
    ym = yf.mean(); yf0 = yf - ym
    G = Xf.T@Xf; b = Xf.T@yf0; n = Xf.shape[1]
    best = (None, 1e18)
    for a in alphas:
        w = np.linalg.solve(G + a*np.eye(n), b)
        p = Xh@w + ym
        mae = np.mean(np.abs(p-yh))
        if mae < best[1]: best = (a, mae)
    return best

def run(cols_frame, label):
    cols = list(cols_frame.columns)
    med = pd.Series(cols_frame[fitm].median(), index=cols)
    Xf_raw = cols_frame[fitm].fillna(med); Xh_raw = cols_frame[holdm].fillna(med)
    mu = Xf_raw.mean(); sd = Xf_raw.std().replace(0,1)
    Xf = ((Xf_raw-mu)/sd).values; Xh = ((Xh_raw-mu)/sd).values
    a, mae = ridge_eval(Xf, Xh, y[fitm], y[holdm])
    print(f"{label:32s} alpha={a} holdMAE={mae:.3f}")
    return mae

B = m[base_cols].astype(float)
run(B, "E003 raw (reference)")

# 1) calendar features
cal = pd.DataFrame(index=m.index)
cal["sin1"] = np.sin(2*np.pi*m.snapshot_day/364); cal["cos1"] = np.cos(2*np.pi*m.snapshot_day/364)
cal["sin2"] = np.sin(4*np.pi*m.snapshot_day/364); cal["cos2"] = np.cos(4*np.pi*m.snapshot_day/364)
cal["t"] = m.snapshot_day/1000.0
run(pd.concat([B, cal], axis=1), "+ calendar sin/cos/t")

# 2) calendar x recent spend interactions
Xc = pd.concat([B, cal], axis=1)
inter = pd.DataFrame(index=m.index)
for k in ["spend_28d","spend_84d","ew_spend_hl28","usual_4w"]:
    for c in ["sin1","cos1","sin2","cos2"]:
        inter[f"{k}x{c}"] = B[k]*cal[c]
run(pd.concat([B, cal, inter], axis=1), "+ cal + cal x recent spend")

# 3) churn-risk interactions
ch = pd.DataFrame(index=m.index)
ch["recency_x_spend"] = B["days_since_last"]*B["spend_28d"]
ch["days_since_sq"] = B["days_since_last"]**2
ch["inactive28"] = (B["days_since_last"]>28).astype(float)*B["spend_84d"]
ch["inactive56"] = (B["days_since_last"]>56).astype(float)*B["spend_84d"]
ch["inactive84"] = (B["days_since_last"]>84).astype(float)*B["spend_364d"]
run(pd.concat([B, ch], axis=1), "+ churn interactions")

# 4) trend interactions
tr = pd.DataFrame(index=m.index)
tr["w1xw2"] = B["spend_w1"]*B["spend_w2"]
tr["w1xw3"] = B["spend_w1"]*B["spend_w3"]
tr["min_w1w2"] = B[["spend_w1","spend_w2"]].min(axis=1)
tr["max_w1w2"] = B[["spend_w1","spend_w2"]].max(axis=1)
tr["min_w1w2w3"] = B[["spend_w1","spend_w2","spend_w3"]].min(axis=1)
tr["med_w1w2w3"] = B[["spend_w1","spend_w2","spend_w3"]].median(axis=1)
run(pd.concat([B, tr], axis=1), "+ trend products/min/max")


# ---- cell ----

import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days, snapshot

df = load_saved("rfm_cadence_v1.parquet")
m = df.merge(train_targets(), on=["household_key","snapshot_day"], how="inner")
tr_all = snapshot_days()["train"]
fitm = m.snapshot_day.isin([d for d in tr_all if d<=347]).values
holdm = ~fitm
y = m["future_spend_4w"].values.astype(float)
base_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]

def ridge_eval(Xf, Xh, yf, yh, alphas=(1,3,10,30,100,300,1000,3000,10000)):
    ym = yf.mean(); yf0 = yf - ym
    G = Xf.T@Xf; b = Xf.T@yf0; n = Xf.shape[1]
    best = (None, 1e18)
    for a in alphas:
        w = np.linalg.solve(G + a*np.eye(n), b)
        p = Xh@w + ym
        mae = np.mean(np.abs(p-yh))
        if mae < best[1]: best = (a, mae)
    return best

def run(cols_frame, label):
    cols = list(cols_frame.columns)
    med = pd.Series(cols_frame[fitm].median(), index=cols)
    Xf_raw = cols_frame[fitm].fillna(med); Xh_raw = cols_frame[holdm].fillna(med)
    mu = Xf_raw.mean(); sd = Xf_raw.std().replace(0,1)
    Xf = ((Xf_raw-mu)/sd).values; Xh = ((Xh_raw-mu)/sd).values
    a, mae = ridge_eval(Xf, Xh, y[fitm], y[holdm])
    print(f"{label:34s} alpha={a} holdMAE={mae:.3f}")
    return mae

B = m[base_cols].astype(float)
run(B, "E003 raw (reference)")

# demographics
snap = snapshot()
demo = snap.demographics.copy()
dd = pd.get_dummies(demo.drop(columns=["household_key"]).astype(str), prefix_sep="=")
dd["household_key"] = demo["household_key"].values
Md = m[["household_key"]].merge(dd, on="household_key", how="left").drop(columns=["household_key"])
Md = Md.astype(float)
Md["has_demo"] = Md.notna().any(axis=1).astype(float)
Md = Md.fillna(0)
run(pd.concat([B, Md], axis=1), "+ demographics dummies")

# winsorize at train 99th pct
caps = B[fitm].quantile(0.99)
Bw = B.clip(upper=caps, axis=1)
run(Bw, "+ winsorized 99pct")

# raw ratios vs usual level
R = pd.DataFrame(index=m.index)
R["r28_usual"] = B["spend_28d"]/B["usual_4w"].replace(0,np.nan)
R["rew_usual"] = B["ew_spend_hl28"]/B["usual_4w"].replace(0,np.nan)
R["rw1_mean6"] = B["spend_w1"]/B["mean_w1_w6"].replace(0,np.nan)
R["r84_364"] = B["spend_84d"]/B["spend_364d"].replace(0,np.nan)
R["r28_364"] = B["spend_28d"]/B["spend_364d"].replace(0,np.nan)
R = R.replace([np.inf,-np.inf], np.nan).fillna(1.0).clip(0,10)
run(pd.concat([B, R], axis=1), "+ raw ratios vs usual")

# within-snapshot ranks of key levels
RK = pd.DataFrame(index=m.index)
for c in ["spend_28d","spend_84d","ew_spend_hl28","usual_4w","spend_364d","days_since_last","trips_28d"]:
    RK["rank_"+c] = B[c].rank(pct=True)
run(pd.concat([B, RK], axis=1), "+ within-snapshot ranks")

# trimmed set: drop late windows w7..w14 and dup spend_364
trim = [c for c in base_cols if c not in [f"spend_w{i}" for i in range(7,15)]+["spend_364"]]
run(B[trim], "trimmed (drop w7-w14, spend_364)")


# ---- cell ----

import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days, snapshot

df = load_saved("rfm_cadence_v1.parquet")
m = df.merge(train_targets(), on=["household_key","snapshot_day"], how="inner")
tr_all = snapshot_days()["train"]
fitm = m.snapshot_day.isin([d for d in tr_all if d<=347]).values
holdm = ~fitm
y = m["future_spend_4w"].values.astype(float)
base_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]

def ridge_eval(Xf, Xh, yf, yh, alphas=(1,3,10,30,100,300,1000,3000,10000)):
    ym = yf.mean(); yf0 = yf - ym
    G = Xf.T@Xf; b = Xf.T@yf0; n = Xf.shape[1]
    best = (None, 1e18)
    for a in alphas:
        w = np.linalg.solve(G + a*np.abs(np.diag(G)).mean()*np.eye(n), b) if False else np.linalg.solve(G + a*np.eye(n), b)
        p = Xh@w + ym
        mae = np.mean(np.abs(p-yh))
        if mae < best[1]: best = (a, mae)
    return best

def run(cols_frame, label):
    cols = list(cols_frame.columns)
    med = pd.Series(cols_frame[fitm].median(), index=cols)
    Xf_raw = cols_frame[fitm].fillna(med); Xh_raw = cols_frame[holdm].fillna(med)
    mu = Xf_raw.mean(); sd = Xf_raw.std().replace(0,1)
    Xf = ((Xf_raw-mu)/sd).values; Xh = ((Xh_raw-mu)/sd).values
    a, mae = ridge_eval(Xf, Xh, y[fitm], y[holdm])
    print(f"{label:36s} alpha={a} holdMAE={mae:.3f}")
    return mae

B = m[base_cols].astype(float)
run(B, "E003 raw (reference)")

# build untapped features from transactions history
def make_feats(view):
    t = view.table("transactions")
    g = t.groupby("household_key")
    f = pd.DataFrame(index=g.size().index)
    for w in (28, 56, 84):
        tw = t[t.day > view.day - w]
        gw = tw.groupby("household_key")
        f[f"units_{w}"] = gw["quantity"].sum()
        f[f"nprod_{w}"] = gw["product_id"].nunique()
        f[f"nbask_{w}"] = gw["basket_id"].nunique()
        f[f"nstore_{w}"] = gw["store_id"].nunique()
    f["max_basket_84"] = t[t.day > view.day-84].groupby("household_key")["sales_value"].max()
    f["med_basket_84"] = t[t.day > view.day-84].groupby(["household_key","basket_id"])["sales_value"].sum().groupby("household_key").median()
    f["q75_basket_84"] = t[t.day > view.day-84].groupby(["household_key","basket_id"])["sales_value"].sum().groupby("household_key").quantile(0.75)
    f["std_basket_84"] = t[t.day > view.day-84].groupby(["household_key","basket_id"])["sales_value"].sum().groupby("household_key").std()
    f["spend_per_unit_84"] = f["units_84"].rdiv if False else t[t.day > view.day-84].groupby("household_key")["sales_value"].sum() / f["units_84"].replace(0,np.nan)
    return f

from agent_api import build_features
F = build_features(make_feats)
M = m.merge(F, on=["household_key","snapshot_day"], how="left")
new_cols = [c for c in F.columns if c not in ("household_key","snapshot_day")]
print("new feats:", len(new_cols), "| NaN frac:", M[new_cols].isna().mean().round(2).to_dict())
run(pd.concat([B, M[new_cols].astype(float)], axis=1), "+ units/variety/basket-dist")
run(M[new_cols].astype(float), "units/variety/basket-dist alone")


# ---- cell ----

import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days, snapshot, build_features

df = load_saved("rfm_cadence_v1.parquet")
m = df.merge(train_targets(), on=["household_key","snapshot_day"], how="inner")
tr_all = snapshot_days()["train"]
fitm = m.snapshot_day.isin([d for d in tr_all if d<=347]).values
holdm = ~fitm
y = m["future_spend_4w"].values.astype(float)
base_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]

def ridge_eval(Xf, Xh, yf, yh, alphas=(1,3,10,30,100,300,1000,3000,10000)):
    ym = yf.mean(); yf0 = yf - ym
    G = Xf.T@Xf; b = Xf.T@yf0; n = Xf.shape[1]
    best = (None, 1e18)
    for a in alphas:
        w = np.linalg.solve(G + a*np.eye(n), b)
        p = Xh@w + ym
        mae = np.mean(np.abs(p-yh))
        if mae < best[1]: best = (a, mae)
    return best

def run(cols_frame, label):
    cols = list(cols_frame.columns)
    med = pd.Series(cols_frame[fitm].median(), index=cols)
    Xf_raw = cols_frame[fitm].fillna(med); Xh_raw = cols_frame[holdm].fillna(med)
    mu = Xf_raw.mean(); sd = Xf_raw.std().replace(0,1)
    Xf = ((Xf_raw-mu)/sd).values; Xh = ((Xh_raw-mu)/sd).values
    a, mae = ridge_eval(Xf, Xh, y[fitm], y[holdm])
    print(f"{label:36s} alpha={a} holdMAE={mae:.3f}")
    return mae

B = m[base_cols].astype(float)

def make_feats(view, snapshot_day):
    t = view.table("transactions")
    day = view.day
    g = t.groupby("household_key")
    f = pd.DataFrame(index=g.size().index)
    for w in (28, 56, 84):
        tw = t[t.day > day - w]
        gw = tw.groupby("household_key")
        f[f"units_{w}"] = gw["quantity"].sum()
        f[f"nprod_{w}"] = gw["product_id"].nunique()
        f[f"nstore_{w}"] = gw["store_id"].nunique()
    b84 = t[t.day > day-84].groupby(["household_key","basket_id"])["sales_value"].sum()
    f["max_basket_84"] = b84.groupby("household_key").max()
    f["med_basket_84"] = b84.groupby("household_key").median()
    f["q75_basket_84"] = b84.groupby("household_key").quantile(0.75)
    f["std_basket_84"] = b84.groupby("household_key").std()
    u84 = t[t.day > day-84].groupby("household_key")["quantity"].sum()
    f["spend_per_unit_84"] = t[t.day > day-84].groupby("household_key")["sales_value"].sum() / u84.replace(0,np.nan)
    return f

F = build_features(make_feats)
M = m.merge(F, on=["household_key","snapshot_day"], how="left")
new_cols = [c for c in F.columns if c not in ("household_key","snapshot_day")]
print("new feats:", len(new_cols))
run(pd.concat([B, M[new_cols].astype(float)], axis=1), "+ units/variety/basket-dist")
run(M[new_cols].astype(float), "units/variety/basket-dist alone")


# ---- cell ----

import numpy as np, pandas as pd
from agent_api import load_saved, snapshot, save_table

df = load_saved("rfm_cadence_v1.parquet")
snap = snapshot()
demo = snap.demographics.copy()
dd = pd.get_dummies(demo.drop(columns=["household_key"]).astype(str), prefix_sep="=")
dd.insert(0, "household_key", demo["household_key"].values)
out = df.merge(dd, on="household_key", how="left")
dcol = [c for c in dd.columns if c != "household_key"]
out["has_demo"] = out[dcol].notna().any(axis=1).astype(int)
out[dcol] = out[dcol].fillna(0).astype(int)
print(out.shape)
path = save_table(out, "demo_v1.parquet")
print(path)
