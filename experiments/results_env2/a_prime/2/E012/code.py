
import agent_api as api

for name in ["e011_discounts", "e010_lifecycle", "e005_decay_gapcv", "e004_temporal", "e001_recent_behavior"]:
    df = api.load_saved(name + ".parquet")
    print(name, df.shape)
    print(list(df.columns))
    print("---")

tt = api.train_targets()
y = tt["future_spend_4w"]
print("train rows:", len(tt), "zero share:", (y == 0).mean(), "mean:", y.mean(), "median:", y.median(), "p90:", y.quantile(.9))
print("calls used so far: see counter")


# ---- cell ----

import agent_api as api, numpy as np, pandas as pd

df = api.load_saved("e011_discounts.parquet")
tt = api.train_targets()
df = df.merge(tt, on=["household_key","snapshot_day"], how="left")
train = df[df.future_spend_4w.notna()].copy()
val   = df[df.future_spend_4w.isna()].copy()
print("train", train.shape, "val", val.shape)

feat = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
Xtr = train[feat].astype(float).fillna(0).values
ytr = train.future_spend_4w.values
Xva = val[feat].astype(float).fillna(0).values

def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0)+1e-9
    Z = (X-mu)/sd
    A = np.hstack([Z, np.ones((len(Z),1))])
    I = np.eye(A.shape[1]); I[-1,-1]=0
    w = np.linalg.solve(A.T@A + lam*I, A.T@y)
    return mu, sd, w

def mae(X, y, p):
    Z = (X-p[0])/p[1]
    pred = np.hstack([Z, np.ones((len(Z),1))]) @ p[2]
    return np.abs(pred-y).mean()

for lam in [1.0, 10.0, 100.0]:
    p = ridge_fit(Xtr, ytr, lam)
    print(f"lam={lam}: train MAE {mae(Xtr,ytr,p):.2f}  val MAE {mae(Xva,np.zeros(len(Xva)),p):.2f}")

# correlation of target with key features
for c in ["spend_28","ew_28","spend_84","lr_mean28","spend_7","ew_7","days_since_last","active_28","spend_364"]:
    print(c, np.corrcoef(train[c].fillna(0), ytr)[0,1].round(3))


# ---- cell ----

import agent_api as api, numpy as np, pandas as pd

df = api.load_saved("e011_discounts.parquet")
tt = api.train_targets()
df = df.merge(tt, on=["household_key","snapshot_day"], how="left")
train = df[df.future_spend_4w.notna()].copy()
val   = df[df.future_spend_4w.isna()].copy()

feat = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
def prep(d):
    out = {}
    for c in feat:
        s = d[c]
        if s.dtype == object:
            s = pd.Categorical(s.fillna("NA")).codes
        out[c] = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    return pd.DataFrame(out, index=d.index)

Xtr_df, Xva_df = prep(train), prep(val)
Xtr, ytr = Xtr_df.values, train.future_spend_4w.values
Xva = Xva_df.values

def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0)+1e-9
    Z = np.hstack([(X-mu)/sd, np.ones((len(X),1))])
    I = np.eye(Z.shape[1]); I[-1,-1]=0
    w = np.linalg.solve(Z.T@Z + lam*I, Z.T@y)
    return mu, sd, w

def mae(X, y, p):
    Z = np.hstack([(X-p[0])/p[1], np.ones((len(X),1))])
    return np.abs(Z@p[2]-y).mean()

for lam in [1.0, 10.0, 100.0]:
    p = ridge_fit(Xtr, ytr, lam)
    print(f"lam={lam}: train MAE {mae(Xtr,ytr,p):.2f}  val MAE {mae(Xva,np.zeros(len(Xva)),p):.2f}")

print("\ncorr with target:")
for c in feat:
    r = np.corrcoef(Xtr_df[c].values, ytr)[0,1]
    if abs(r) > 0.35:
        print(f"  {c}: {r:.3f}")


# ---- cell ----

import agent_api as api, numpy as np, pandas as pd

df = api.load_saved("e011_discounts.parquet")
tt = api.train_targets()
df = df.merge(tt, on=["household_key","snapshot_day"], how="left")
train = df[df.future_spend_4w.notna()].copy()
val   = df[df.future_spend_4w.isna()].copy()

feat = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
print(df[feat].dtypes.value_counts())

def prep(d):
    out = {}
    for c in feat:
        s = pd.Series(d[c].values if hasattr(d[c], "values") else d[c], index=d.index)
        if s.dtype == object or str(s.dtype) == "category":
            s = pd.Categorical(s.astype(str).fillna("NA")).codes
        out[c] = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    return pd.DataFrame(out, index=d.index)

Xtr_df, Xva_df = prep(train), prep(val)
Xtr, ytr = Xtr_df.values, train.future_spend_4w.values
Xva = Xva_df.values

def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0)+1e-9
    Z = np.hstack([(X-mu)/sd, np.ones((len(X),1))])
    I = np.eye(Z.shape[1]); I[-1,-1]=0
    w = np.linalg.solve(Z.T@Z + lam*I, Z.T@y)
    return mu, sd, w

def mae(X, y, p):
    Z = np.hstack([(X-p[0])/p[1], np.ones((len(X),1))])
    return np.abs(Z@p[2]-y).mean()

for lam in [1.0, 10.0, 100.0]:
    p = ridge_fit(Xtr, ytr, lam)
    print(f"lam={lam}: train MAE {mae(Xtr,ytr,p):.2f}  val MAE {mae(Xva,np.zeros(len(Xva)),p):.2f}")

print("\n|corr|>0.35 with target:")
for c in feat:
    r = np.corrcoef(Xtr_df[c].values, ytr)[0,1]
    if abs(r) > 0.35:
        print(f"  {c}: {r:.3f}")


# ---- cell ----

import agent_api as api, numpy as np, pandas as pd

df = api.load_saved("e011_discounts.parquet")
tt = api.train_targets()
df = df.merge(tt, on=["household_key","snapshot_day"], how="left")
train = df[df.future_spend_4w.notna()].copy()
val   = df[df.future_spend_4w.isna()].copy()

feat = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w")]

def prep(d):
    out = {}
    for c in feat:
        s = pd.Series(list(d[c]), index=d.index)
        if s.dtype == object or str(s.dtype) == "category":
            s = pd.Categorical(s.fillna("NA")).codes
        out[c] = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    return pd.DataFrame(out, index=d.index)

Xtr_df, Xva_df = prep(train), prep(val)
Xtr, ytr = Xtr_df.values, train.future_spend_4w.values
Xva = Xva_df.values

def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0)+1e-9
    Z = np.hstack([(X-mu)/sd, np.ones((len(X),1))])
    I = np.eye(Z.shape[1]); I[-1,-1]=0
    w = np.linalg.solve(Z.T@Z + lam*I, Z.T@y)
    return mu, sd, w

def mae(X, y, p):
    Z = np.hstack([(X-p[0])/p[1], np.ones((len(X),1))])
    return np.abs(Z@p[2]-y).mean()

for lam in [1.0, 10.0, 100.0]:
    p = ridge_fit(Xtr, ytr, lam)
    print(f"lam={lam}: train MAE {mae(Xtr,ytr,p):.2f}  val MAE {mae(Xva,np.zeros(len(Xva)),p):.2f}")

print("\n|corr|>0.35 with target:")
for c in feat:
    r = np.corrcoef(Xtr_df[c].values, ytr)[0,1]
    if abs(r) > 0.35:
        print(f"  {c}: {r:.3f}")


# ---- cell ----

import agent_api as api, numpy as np, pandas as pd

df = api.load_saved("e011_discounts.parquet")
tt = api.train_targets()
df = df.merge(tt, on=["household_key","snapshot_day"], how="left")
train = df[df.future_spend_4w.notna()].copy()
val   = df[df.future_spend_4w.isna()].copy()

feat = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w")]

def prep(d):
    out = {}
    for c in feat:
        s = pd.Series(list(d[c]), index=d.index)
        if s.dtype == object or str(s.dtype) == "category":
            s = pd.Series(pd.Categorical(s.fillna("NA")).codes, index=d.index)
        out[c] = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    return pd.DataFrame(out, index=d.index)

Xtr_df, Xva_df = prep(train), prep(val)
Xtr, ytr = Xtr_df.values, train.future_spend_4w.values
Xva = Xva_df.values

def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0)+1e-9
    Z = np.hstack([(X-mu)/sd, np.ones((len(X),1))])
    I = np.eye(Z.shape[1]); I[-1,-1]=0
    w = np.linalg.solve(Z.T@Z + lam*I, Z.T@y)
    return mu, sd, w

def mae(X, y, p):
    Z = np.hstack([(X-p[0])/p[1], np.ones((len(X),1))])
    return np.abs(Z@p[2]-y).mean()

for lam in [1.0, 10.0, 100.0]:
    p = ridge_fit(Xtr, ytr, lam)
    print(f"lam={lam}: train MAE {mae(Xtr,ytr,p):.2f}  val MAE {mae(Xva,np.zeros(len(Xva)),p):.2f}")

print("\n|corr|>0.35 with target:")
for c in feat:
    r = np.corrcoef(Xtr_df[c].values, ytr)[0,1]
    if abs(r) > 0.35:
        print(f"  {c}: {r:.3f}")


# ---- cell ----

import agent_api as api, numpy as np, pandas as pd

df = api.load_saved("e011_discounts.parquet")
tt = api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="left")
tr = m[m.future_spend_4w.notna()]; va = m[m.future_spend_4w.isna()]
hh_tr, hh_va = set(tr.household_key), set(va.household_key)
print("households: train", len(hh_tr), "val", len(hh_va), "val in train:", len(hh_va & hh_tr), f"({len(hh_va & hh_tr)/len(hh_va):.1%})")

# persistence: between-household share of target variance (train rows)
g = tr.groupby("household_key").future_spend_4w.agg(["mean","count","std"])
mu = tr.future_spend_4w.mean()
ssb = (g["count"]*(g["mean"]-mu)**2).sum()
sst = ((tr.future_spend_4w-mu)**2).sum()
print(f"between-household variance share: {ssb/sst:.3f}")
print("pooled std:", tr.future_spend_4w.std().round(1), " mean within-household std:", g['std'].mean().round(1))

# how well would a pure 'household mean from train history' predictor do on val?
# use household's mean target over TRAIN snapshots only, predict val rows
hm = tr.groupby("household_key").future_spend_4w.mean()
pred = va.household_key.map(hm).fillna(mu)
print("val MAE of household-mean-from-train predictor:", np.abs(pred - va.future_spend_4w.values if False else 0))  # placeholder
# careful: val targets unknown to us; can't compute. Instead check within-train consistency:
# predict each train row by mean of OTHER train snapshots of same household (leave-one-out)
tmp = tr.merge(tr.groupby("household_key").future_spend_4w.transform("sum").rename("s"), left_index=True, right_index=True)
tmp = tmp.merge(tr.groupby("household_key").future_spend_4w.transform("count").rename("n"), left_index=True, right_index=True)
loo = (tmp.s - tmp.future_spend_4w)/(tmp.n-1)
print("train LOO MAE of household-mean predictor:", np.abs(loo - tr.future_spend_4w).mean().round(2))
print("train MAE of global mean predictor:", np.abs(tr.future_spend_4w - mu).mean().round(2))


# ---- cell ----

import agent_api as api, numpy as np, pandas as pd

v = api.snapshot(459)
t = v.transactions
wk = t.groupby("week_no").sales_value.sum()
print("weeks range:", wk.index.min(), wk.index.max(), "n weeks:", len(wk))
# overall weekly mean per week index
m = wk.mean(); s = wk.std()
print("mean weekly total spend:", round(m), "std:", round(s))
big = wk[wk > m + 2*s]
print("\nSpike weeks (> mean+2sd):")
print(big.round(0))
print("\nTop 12 weeks by spend:")
print(wk.sort_values(ascending=False).head(12).round(0))
print("\nBottom 5 weeks:")
print(wk.sort_values().head(5).round(0))
# also transactions count per week
cnt = t.groupby("week_no").size()
print("\nTop 6 weeks by trip count:")
print(cnt.sort_values(ascending=False).head(6))
# what day range does week w cover: week_no=(day+8)//7 -> day in [7w-8, 7w-2]
print("\nweek->day mapping examples: week 66 -> days", 7*66-8, "to", 7*66-2)


# ---- cell ----

import agent_api as api, numpy as np, pandas as pd

df = api.load_saved("e011_discounts.parquet")
tt = api.train_targets().sort_values(["household_key","snapshot_day"])

# per-household aggregates over train snapshots
g = tt.groupby("household_key").future_spend_4w
agg = pd.DataFrame({"hh_sum": g.sum(), "hh_sumsq": (g.apply(lambda s: (s*s).sum())),
                    "hh_n": g.count()}).reset_index()
df = df.merge(agg, on="household_key", how="left")
df = df.merge(tt.rename(columns={"future_spend_4w":"tgt"})[["household_key","snapshot_day","tgt"]],
              on=["household_key","snapshot_day"], how="left")

is_tr = df.tgt.notna()
n = df.hh_n
# LOO mean/std over the household's train targets (val rows: full 13-target mean)
loo_mean = (df.hh_sum - df.tgt) / (n - 1)
loo_var = ((df.hh_sumsq - df.tgt**2) - (df.hh_sum - df.tgt)**2/(n-1)) / (n-2)
df["hh_mean_loo"] = np.where(is_tr & (n > 1), loo_mean, df.hh_sum/n)
df["hh_std_loo"]  = np.where(is_tr & (n > 2), np.sqrt(loo_var.clip(lower=0)), np.nan)
# causal (strictly earlier snapshots only); val rows -> full mean
tt["hh_mean_causal"] = (tt.groupby("household_key").future_spend_4w.cumsum() - tt.future_spend_4w) / \
                       (tt.groupby("household_key").future_spend_4w.cumcount()).replace(0, np.nan)
tt["hh_mean_last3"] = tt.groupby("household_key").future_spend_4w.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
df = df.merge(tt[["household_key","snapshot_day","hh_mean_causal","hh_mean_last3"]],
              on=["household_key","snapshot_day"], how="left")
df["hh_mean_causal"] = df.hh_mean_causal.fillna(df.hh_mean_loo)
df["hh_mean_last3"]  = df.hh_mean_last3.fillna(df.hh_mean_loo)

newcols = ["hh_mean_loo","hh_std_loo","hh_n","hh_mean_causal","hh_mean_last3"]
df = df.drop(columns=["tgt","hh_sum","hh_sumsq"])
print(df[newcols].describe().round(1))
print("rows:", len(df), "cols:", df.shape[1])
path = api.save_table(df, "e012_hh_target_enc")
print(path)
