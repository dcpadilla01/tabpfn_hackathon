for name in ["e017_v2.parquet","e018_timing_hazard.parquet","e019_everything.parquet","e019_full_merged.parquet"]:
    df = load_saved(name)
    print("=== ", name, df.shape)
    print(list(df.columns))
    print()

# ---- cell ----
for name in ["e017_disc_seasonal.parquet","e019_everything.parquet"]:
    df = load_saved(name)
    print("===", name, df.shape)
    print(list(df.columns))
    print()

# ---- cell ----
import numpy as np, pandas as pd
df = load_saved("e019_everything.parquet")   # E017 best table
tt = train_targets()
d = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("merged", d.shape)
y = d["future_spend_4w"].values.astype(float)
print("target mean/med", y.mean().round(2), np.median(y))
print("pcts", np.percentile(y,[10,25,50,75,90,95,99]).round(1))
print("zero share", (y==0).mean().round(3))
print("naive MAEs:")
for c in ["spend_l1","spend_l2","spend_l123_mean","spend_rate28","newm4","nwmean12","spend_ly4w","cohort_prior4w"]:
    if c in d: print("  ", c, np.abs(d[c].fillna(0).values - y).mean().round(2))
print("  global median", np.abs(np.median(y)-y).mean().round(2))
# per-snapshot target mean (drift check)
g = d.groupby("snapshot_day")["future_spend_4w"].agg(["mean","median","size"])
print(g.round(1))
# per-snapshot mean of key predictors
for c in ["spend_l1","spend_l13","spend_ly4w"]:
    if c in d: print(c, d.groupby("snapshot_day")[c].mean().round(1).values)

# ---- cell ----
import numpy as np, pandas as pd
e16 = load_saved("e018_timing_hazard.parquet")  # E016 table
e17 = load_saved("e019_everything.parquet")     # E017 table
c16, c17 = set(e16.columns), set(e17.columns)
print("in E016 not E017:", sorted(c16-c17))
print("in E017 not E016:", sorted(c17-c16))
print()
dm = snapshot().display_mailer
print(dm.shape); print(dm.head(8))
print(dm['display'].value_counts().head())
print(dm['mailer'].value_counts().head())

# ---- cell ----
import numpy as np, pandas as pd
tt = train_targets()
best = load_saved("e019_everything.parquet")
d = best.merge(tt, on=["household_key","snapshot_day"])
y = d["future_spend_4w"].values.astype(float)
# which predictors are best per-snapshot (like a one-feature model)?
# crude: correlation and MAE of prediction = snapshot-day mean of feature (i.e. feature as-is)
num = d.drop(columns=["household_key","snapshot_day","future_spend_4w"])
num = num.select_dtypes(include=[np.number])
res=[]
for c in num.columns:
    v = num[c].values
    m = ~np.isnan(v)
    if m.sum() < len(v)*0.5: continue
    r = np.corrcoef(v[m], y[m])[0,1]
    res.append((c, r))
res.sort(key=lambda t:-abs(t[1]))
print("Top |corr| with future_spend_4w:")
for c,r in res[:35]: print(f"  {c:28s} {r: .3f}")
print()
# correlation among the top spend-level features
top = ["spend_l1","spend_l2","spend_l3","spend_l123_mean","spend_rate28","momentum","spend_7d","spend_14d","nspend28","newm4","newm8","newm13","nwmean12","spend_ly4w","spend_l13","peer_recent28","cohort_prior4w","spend_total","nspend_ly"]
sub = d[top].copy()
print(sub.corrwith(d["future_spend_4w"]).round(3).to_string())
print()
print("corr spend_l1 vs others:")
print(sub.drop(columns=["spend_l1"]).corrwith(sub["spend_l1"]).round(3).to_string())

# ---- cell ----
import numpy as np, pandas as pd
v = snapshot()
camp = v.campaigns; ct = v.campaign_targets
print(camp.shape, ct.shape)
print(camp.head())
print(camp['description'].value_counts())
print("campaign day ranges: start", camp.start_day.min(), camp.start_day.max(), "end", camp.end_day.min(), camp.end_day.max())
print(ct['description'].value_counts())
print("n households targeted:", ct.household_key.nunique())
# how many campaigns overlap a hypothetical next-28d window at day 459?
for sd in [95, 459]:
    act = camp[(camp.start_day <= sd+28) & (camp.end_day >= sd+1)]
    t = ct[ct.campaign.isin(act.campaign)]
    print(f"snapshot {sd}: campaigns overlapping next28={len(act)}, targeted hh-campaign pairs={len(t)}, distinct hh={t.household_key.nunique()}")
# redemption stats
cr = v.coupon_redemptions
print("redemptions", cr.shape, "hh", cr.household_key.nunique(), "day max", cr.day.max())
print(cr.head(3))

# ---- cell ----
import numpy as np, pandas as pd
v = snapshot()
camp = v.campaigns; ct = v.campaign_targets
# For each snapshot day, campaigns overlapping next-28d window and their types
for sd in [95,123,151,179,207,235,263,291,319,347,375,403,431,459,487,515,543]:
    if sd > 459: continue
    act = camp[(camp.start_day <= sd+28) & (camp.end_day >= sd+1)]
    t = ct[ct.campaign.isin(act.campaign)]
    bytype = t.groupby('description')['household_key'].nunique().to_dict()
    print(sd, "camps:", sorted(act.campaign.tolist()), "types:", {k:int(x) for k,x in bytype.items()})

# ---- cell ----
import numpy as np, pandas as pd
v = snapshot()
camp = v.campaigns; ct = v.campaign_targets
# campaign 8 = TypeA 412-460; campaign 9 = TypeB 435-467. Who is targeted?
t8 = ct[ct.campaign==8].household_key.unique()
t9 = ct[ct.campaign==9].household_key.unique()
print("campaign8 targets:", len(t8), "campaign9 targets:", len(t9))
print("overlap:", len(set(t8)&set(t9)))
# do targeted households actually spend more?
tt = train_targets()
best = load_saved("e019_everything.parquet")
d = best.merge(tt, on=["household_key","snapshot_day"])
d["t8"] = d.household_key.isin(t8).astype(int)
d["t9"] = d.household_key.isin(t9).astype(int)
print(d.groupby("t8")["future_spend_4w"].agg(["mean","median","size"]))
print(d.groupby("t9")["future_spend_4w"].agg(["mean","median","size"]))
# TypeA targets across time - is targeting persistent per household?
tA = ct[ct.description=="TypeA"].groupby("household_key")["campaign"].nunique()
print("TypeA: hh with multiple targeted campaigns:", (tA>1).sum(), "of", len(tA))
tB = ct[ct.description=="TypeB"].groupby("household_key")["campaign"].nunique()
print("TypeB:", (tB>1).sum(), "of", len(tB))
tC = ct[ct.description=="TypeC"].groupby("household_key")["campaign"].nunique()
print("TypeC:", (tC>1).sum(), "of", len(tC))

# ---- cell ----
import numpy as np, pandas as pd
v = snapshot()
ct = v.campaign_targets; camp = v.campaigns
# how many distinct campaigns per household per type (proxy for "high-value target")
cnt = ct.groupby(["household_key","description"]).size().unstack(fill_value=0)
print(cnt.describe().round(2))
# cross-tab with spend level
tt = train_targets()
best = load_saved("e019_everything.parquet")
d = best.merge(tt, on=["household_key","snapshot_day"])
d = d.merge(cnt, on="household_key", how="left").fillna(0)
for t in ["TypeA","TypeB","TypeC"]:
    d[t+"_n"] = d[t]
    print(t, d.groupby(t+"_n")["future_spend_4w"].agg(["mean","size"]).round(1).T.to_dict())
# also check: does spend_l1 differ by TypeA count? (i.e. is it just selection on past spend?)
print("spend_l1 by TypeA_n:")
print(d.groupby("TypeA_n")["spend_l1"].mean().round(1).to_dict())
print("spend_l1 by TypeB_n:")
print(d.groupby("TypeB_n")["spend_l1"].mean().round(1).to_dict())

# ---- cell ----
import numpy as np, pandas as pd
v = snapshot()
ct = v.campaign_targets
cnt = ct.groupby(["household_key","description"]).size().unstack(fill_value=0).reset_index()
print(cnt.head(3))
tt = train_targets()
best = load_saved("e019_everything.parquet")
d = best.merge(tt, on=["household_key","snapshot_day"])
d = d.merge(cnt, on="household_key", how="left").fillna({c:0 for c in ["TypeA","TypeB","TypeC"]})
for t in ["TypeA","TypeB","TypeC"]:
    g = d.groupby(t)["future_spend_4w"].agg(["mean","size"]).round(1)
    print(t); print(g)
    print("  spend_l1:", d.groupby(t)["spend_l1"].mean().round(1).to_dict())
    print("  spend_l123_mean:", d.groupby(t)["spend_l123_mean"].mean().round(1).to_dict())

# ---- cell ----
import numpy as np, pandas as pd

# Build the TRUE union: E017 best table (e019_everything) + E016's 33 trip-timing features (e018_timing_hazard)
a = load_saved("e019_everything.parquet")
b = load_saved("e018_timing_hazard.parquet")
timing_cols = [c for c in b.columns if c not in a.columns and c not in ("index",)]
print("timing cols to add:", len(timing_cols), timing_cols)
m = a.merge(b[["household_key","snapshot_day"]+timing_cols], on=["household_key","snapshot_day"], how="inner")
print("merged:", m.shape)
m = m.drop(columns=[c for c in m.columns if c=="index"], errors="ignore")
print("dupes:", m.duplicated(["household_key","snapshot_day"]).sum())
p = save_table(m, "e018_union_full")
print("saved:", p)

# --- internal sanity: ridge LOSO on TRAIN snapshots only, compare e019 vs union ---
tt = train_targets()
def loso(tab, label):
    d = tab.merge(tt, on=["household_key","snapshot_day"])
    tr_days = sorted([s for s in d.snapshot_day.unique()])[:13]
    num = d.select_dtypes(include=[np.number])
    cats = [c for c in d.columns if d[c].dtype==object or str(d[c].dtype)=="category"]
    X = num.drop(columns=["future_spend_4w"], errors="ignore")
    for c in cats:
        oh = pd.get_dummies(d[c].astype("category"), prefix=c[:6], dummy_na=True)
        X = pd.concat([X, oh.astype(float)], axis=1)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = d["future_spend_4w"].values.astype(float)
    days = d["snapshot_day"].values
    mu, sd = X.mean().values, X.std().values+1e-9
    Xs = (X.values-mu)/sd
    errs=[]
    for s in tr_days:
        tr = days!=s; te = days==s
        Xt, yt = Xs[tr], y[tr]
        A = Xt.T@Xt + 3.0*np.eye(Xt.shape[1])
        w = np.linalg.solve(A, Xt.T@yt)
        pred = Xs[te]@w
        errs.append(np.abs(pred-y[te]).mean())
    print(f"{label}: LOSO-MAE {np.mean(errs):.3f}")
    return np.mean(errs)

loso(a, "e019_everything (E017 best)")
loso(m, "union (+33 timing)")

# ---- cell ----
import numpy as np, pandas as pd

tt = train_targets()
def loso(tab, label, alpha=50.0):
    d = tab.merge(tt, on=["household_key","snapshot_day"])
    num = d.select_dtypes(include=[np.number])
    cats = [c for c in d.columns if d[c].dtype==object or str(d[c].dtype)=="category"]
    X = num.drop(columns=["future_spend_4w"], errors="ignore")
    for c in cats:
        oh = pd.get_dummies(d[c].astype("category"), prefix=c[:6], dummy_na=True)
        X = pd.concat([X, oh.astype(float)], axis=1)
    # drop degenerate columns
    keep = X.columns[(X.nunique(dropna=True)>1).values]
    X = X[keep]
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = d["future_spend_4w"].values.astype(float)
    days = d["snapshot_day"].values
    mu, sd = X.mean().values, X.std().values+1e-9
    Xs = np.clip((X.values-mu)/sd, -5, 5)
    errs=[]
    for s in sorted(set(days))[:13]:
        tr = days!=s; te = days==s
        Xt, yt = Xs[tr], y[tr]
        A = Xt.T@Xt + alpha*np.eye(Xt.shape[1])
        w = np.linalg.solve(A, Xt.T@yt)
        pred = np.clip(Xs[te]@w, 0, 1500)
        errs.append(np.abs(pred-y[te]).mean())
    print(f"{label}: LOSO-MAE {np.mean(errs):.3f} (n_feat {X.shape[1]})")
    return np.mean(errs)

a = load_saved("e019_everything.parquet")
m = load_saved("e018_union_full.parquet")
loso(a, "e019_everything (E017 best)")
loso(m, "union (+31 timing)")

# ---- cell ----
import numpy as np, pandas as pd

tt = train_targets()
def loso(tab, label, alpha=50.0):
    d = tab.merge(tt, on=["household_key","snapshot_day"])
    num = d.select_dtypes(include=[np.number])
    cats = [c for c in d.columns if d[c].dtype==object or str(d[c].dtype)=="category"]
    X = num.drop(columns=["future_spend_4w"], errors="ignore")
    for c in cats:
        oh = pd.get_dummies(d[c].astype("category"), prefix=c[:6], dummy_na=True)
        X = pd.concat([X, oh.astype(float)], axis=1)
    keep = X.columns[(X.nunique(dropna=True)>1).values]
    X = X[keep]
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = d["future_spend_4w"].values.astype(float)
    days = d["snapshot_day"].values
    mu, sd = X.mean().values, X.std().values+1e-9
    Xs = np.clip((X.values-mu)/sd, -5, 5)
    errs=[]
    for s in sorted(set(days))[:13]:
        tr = days!=s; te = days==s
        Xt, yt = Xs[tr], y[tr]
        A = Xt.T@Xt + alpha*np.eye(Xt.shape[1])
        w = np.linalg.solve(A, Xt.T@yt)
        pred = np.clip(Xs[te]@w, 0, 1500)
        errs.append(np.abs(pred-y[te]).mean())
    return np.mean(errs)

a = load_saved("e019_everything.parquet")
m = load_saved("e018_union_full.parquet")
print("e019:", loso(a,"x"))
print("union:", loso(m,"x"))

# What if we add ONLY a few timing features that don't exist in e019?
b = load_saved("e018_timing_hazard.parquet")
cand = ["overdue_days","overdue_ratio","next_trip_eta","exp_trips28","gap_last5_mean","rhythm_break","burst7","dow_sd","m_spend28_mean","m_spend28_med","zero_frac_l13","zero_frac_l26","act_w4","act_w8","act_w13","trend14_42"]
cand = [c for c in cand if c not in a.columns]
small = a.merge(b[["household_key","snapshot_day"]+cand], on=["household_key","snapshot_day"])
print("small timing add:", loso(small,"x"))
# try even smaller: 6
cand6 = ["overdue_days","next_trip_eta","gap_last5_mean","burst7","m_spend28_mean","zero_frac_l13"]
cand6 = [c for c in cand6 if c not in a.columns]
small6 = a.merge(b[["household_key","snapshot_day"]+cand6], on=["household_key","snapshot_day"])
print("6 timing add:", loso(small6,"x"))