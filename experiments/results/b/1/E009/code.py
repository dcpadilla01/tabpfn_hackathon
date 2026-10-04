
import pandas as pd, numpy as np
import agent_api as A

tabs = {}
for name in ["e001_history", "e007_new", "e008_decomp2"]:
    df = A.load_saved(name + ".parquet")
    tabs[name] = df
    print("==", name, df.shape)
    print(list(df.columns))
    print()

tt = A.train_targets()
print("targets:", tt.shape)
print(tt["future_spend_4w"].describe())
print("zero frac:", (tt["future_spend_4w"] == 0).mean())
print()

e8, e1, e7 = tabs["e008_decomp2"], tabs["e001_history"], tabs["e007_new"]
c8 = set(e8.columns); c1 = set(e1.columns); c7 = set(e7.columns)
print("e001 cols not in e008:", sorted(c1 - c8))
print("e007 cols not in e008:", sorted(c7 - c8))
print()

m = tt.merge(e8, on=["household_key", "snapshot_day"], how="inner")
print("merged e8+target:", m.shape)
num = [c for c in e8.columns if c not in ("household_key", "snapshot_day") and pd.api.types.is_numeric_dtype(m[c])]
corr = m[num].corrwith(m["future_spend_4w"]).abs().sort_values(ascending=False)
print("top |corr| with target:")
print(corr.head(20))
print()

# naive predictor MAEs on train rows
def mae(p, y):
    return np.mean(np.abs(p - y))
y = m["future_spend_4w"].values
for c in num:
    if m[c].notna().sum() > 0.9 * len(m):
        pass
cands = [c for c in num if any(k in c.lower() for k in ["spend", "usual", "p_active", "exp", "pred"])]
for c in cands[:25]:
    v = m[c].values
    print(f"{c:35s} MAE={mae(v, y):8.3f}  corr={np.corrcoef(v, y)[0,1]:.3f}")


# ---- cell ----

import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
e8 = A.load_saved("e008_decomp2.parquet")
e1 = A.load_saved("e001_history.parquet")
m = tt.merge(e8, on=["household_key","snapshot_day"]).merge(
    e1[["household_key","snapshot_day","spend_7","spend_56","spend_84","spend_182","spend_365",
        "nb_28","nb_56","nb_all","avg_basket_28","spend_prev28","trend28","qty_28","nprod_28",
        "spend_ly28","weekly_rate_84","snapshot_day_index","week_of_year"]],
    on=["household_key","snapshot_day"])

# 1) drift of target and key features across snapshot days
print("target mean by snapshot_day:")
print(m.groupby("snapshot_day")["future_spend_4w"].agg(["mean","median","count"]).round(1))
print("\nspend_28 mean by snapshot (drift check):")
print(m.groupby("snapshot_day")["spend_28"].mean().round(1))
print("\np13 mean by snapshot:")
print(m.groupby("snapshot_day")["p13"].mean().round(3))

# 2) ridge fit on E008 features -> residual correlation with candidate new features
feats = [c for c in e8.columns if c not in ("household_key","snapshot_day")]
X = m[feats].copy()
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(X.median())
y = m["future_spend_4w"].values
Xm = X.values
Xm = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-9)
# simple ridge via normal equations
lam = 1.0
G = Xm.T @ Xm + lam*np.eye(Xm.shape[1])
b = np.linalg.solve(G, Xm.T @ y)
pred = Xm @ b
res = y - pred
print("\nridge-train MAE:", np.abs(res).mean().round(2))

cand = ["spend_7","spend_56","spend_84","spend_182","spend_365","nb_28","nb_56","nb_all",
        "avg_basket_28","spend_prev28","trend28","qty_28","nprod_28","spend_ly28","weekly_rate_84",
        "snapshot_day_index","week_of_year"]
for c in cand:
    v = pd.to_numeric(m[c], errors="coerce").fillna(0).values
    v = (v - v.mean())/(v.std()+1e-9)
    print(f"corr(resid, {c:18s}) = {np.corrcoef(res, v)[0,1]: .4f}")

# 3) zero-rate features
print("\nzero frac by snapshot:")
print(m.assign(z=(m.future_spend_4w==0)).groupby("snapshot_day")["z"].mean().round(3))


# ---- cell ----

import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
e8 = A.load_saved("e008_decomp2.parquet")
m = tt.merge(e8, on=["household_key","snapshot_day"])
feats = [c for c in e8.columns if c not in ("household_key","snapshot_day")]
X = m[feats].apply(pd.to_numeric, errors="coerce")
Xf = X.fillna(X.median()).values
y = m.future_spend_4w.values
Xs = (Xf - Xf.mean(0))/(Xf.std(0)+1e-9)
lam = 1.0
b = np.linalg.solve(Xs.T@Xs + lam*np.eye(Xs.shape[1]), Xs.T@y)
res = y - Xs@b
print("ridge train MAE:", np.abs(res).mean().round(2))

rc = {c: np.corrcoef(res, Xs[:,i])[0,1] for i,c in enumerate(feats)}
print("\nE008 features by |resid corr|:")
for c,v in sorted(rc.items(), key=lambda kv: -abs(kv[1])):
    print(f"  {c:12s} {v: .4f}")

print("\nsemantics: corr(usual13,p13)=", round(np.corrcoef(m.usual13.fillna(0), m.p13.fillna(0))[0,1],3),
      " corr(usual_med,p13)=", round(np.corrcoef(m.usual_med.fillna(0), m.p13.fillna(0))[0,1],3),
      " corr(e13,p13)=", round(np.corrcoef(m.e13.fillna(0), m.p13.fillna(0))[0,1],3))

# ---- macro weekly series ----
snap = A.snapshot(459)
tx = snap.transactions
wk = tx.day.values // 7
g = tx.assign(wk=wk).groupby("wk").agg(sales=("sales_value","sum"), hh=("household_key","nunique"))
macro_w = (g.sales/g.hh)
w = macro_w.index.values
pairs = [(a, a+52) for a in w if a+52 in macro_w.index]
va = np.array([macro_w[a] for a,_ in pairs]); vb = np.array([macro_w[b] for _,b in pairs])
print("\nmacro weekly per-hh spend: lag-52 corr =", round(np.corrcoef(va,vb)[0,1],3),
      " mean lvl ratio (yr2/yr1) =", round(vb.mean()/va.mean(),3))
print("macro_w by 13-week blocks (yr1 vs yr2):")
blk1 = macro_w[macro_w.index<52].groupby(macro_w[macro_w.index<52].index//13).mean()
blk2 = macro_w[(macro_w.index>=52)&(macro_w.index<104)].groupby((macro_w[(macro_w.index>=52)&(macro_w.index<104)].index-52)//13).mean()
print(" yr1:", blk1.round(1).values)
print(" yr2:", blk2.round(1).values)

# ---- candidate features on train snapshots ----
train_days = A.snapshot_days()["train"]
first_day = tx.groupby("household_key").day.min()
cands = {}
for s in train_days:
    t = tx[tx.day <= s]
    # windows
    wi = ((s - t.day - 1) // 28).clip(min=0)
    piv = t.assign(wi=wi).pivot_table(index="household_key", columns="wi", values="sales_value", aggfunc="sum")
    piv = piv.reindex(columns=range(26)).fillna(0.0)
    fd = first_day.reindex(piv.index).fillna(1)
    avail = np.array([ (s - 28*k) >= fd.values for k in range(26) ])  # window end >= first day
    W = piv.values
    def decay(hl, nwin):
        d = 0.5 ** (np.arange(nwin)/hl)
        wgt = d[None,:] * avail
        wsum = wgt.sum(1); wsum[wsum==0] = 1
        return (W*wgt).sum(1)/wsum, wsum
    p26,_ = decay(4,26); p13,_ = decay(2,13)
    # active-window usual (hl=2, 13 win)
    pos = (W>0) & avail
    wgt = (0.5**(np.arange(26)/2))[None,:] * pos
    wsum = wgt.sum(1); wsum[wsum==0]=1
    usual_act = (np.where(pos, W, 0)*wgt).sum(1)/wsum
    # ewma daily hl=28
    tt2 = t.sort_values("day").groupby("day").sales_value.sum()
    full = pd.Series(0.0, index=range(1, s+1)); full.loc[tt2.index] = tt2
    ew = full.ewm(halflife=28).mean().iloc[-1]*28
    ew = pd.Series(ew, index=piv.index).reindex(piv.index).fillna(0) if np.isscalar(ew) else ew
    # macro
    win = t[(t.day > s-28)]
    macro_t28 = win.sales_value.sum()/max(win.household_key.nunique(),1)
    flo, fhi = s+1-364, s+28-364
    tl = t[(t.day>=flo)&(t.day<=fhi)]
    macro_fly = tl.sales_value.sum()/max(tl.household_key.nunique(),1) if len(tl)>0 else np.nan
    df = pd.DataFrame({
        "tenure": s - fd, "p26": p26, "p13h2": p13, "usual_act": usual_act,
        "ewma28": ew if isinstance(ew, pd.Series) else pd.Series(ew, index=piv.index),
        "macro_t28": macro_t28, "macro_fly": macro_fly,
    })
    df["macro_ratio"] = df.macro_fly/df.macro_t28
    df["snapshot_day"] = s
    cands[s] = df.reset_index().rename(columns={"index":"household_key"})

C = pd.concat(cands.values(), ignore_index=True)
M = m[["household_key","snapshot_day","future_spend_4w","e13","e6","e3","e_med","b75","rvu","ly_ratio","dsl","usual13","p13"]].merge(C, on=["household_key","snapshot_day"])
r = pd.Series(res, index=m.index)
M = M.merge(m[["household_key","snapshot_day"]].assign(resid=res), on=["household_key","snapshot_day"])
M["consensus"] = M[["e13","e6","e3","e_med","b75"]].apply(lambda c:(c-c.mean())/ (c.std()+1e-9)).mean(axis=1)
M["e6_rvu"] = M.e6 * M.rvu.clip(0,3)
M["e6_ly"] = M.e6 * M.ly_ratio.clip(0,3)
M["log_dsl"] = np.log1p(M.dsl)
M["week_fut"] = ((M.snapshot_day+15)//7) % 52
M["day_idx"] = M.snapshot_day

print("\ncandidate |corr(resid)| screen (train rows):")
for c in ["tenure","p26","p13h2","usual_act","ewma28","macro_t28","macro_fly","macro_ratio",
          "consensus","e6_rvu","e6_ly","log_dsl","week_fut","day_idx"]:
    v = M[c].astype(float).values
    ok = np.isfinite(v)
    print(f"  {c:12s} {np.corrcoef(M.resid[ok], v[ok])[0,1]: .4f}")
print("\nplain MAE of usual_act as predictor:", np.abs(M.usual_act.fillna(0)-M.future_spend_4w).mean().round(2))


# ---- cell ----
import pandas as pd, numpy as np
import agent_api as A

snap = A.snapshot(459)
tx = snap.transactions
g = tx.assign(wk=tx.day//7).groupby("wk").agg(sales=("sales_value","sum"), hh=("household_key","nunique"))
macro_w = (g.sales/g.hh)
print("macro weekly per-hh spend, weeks 1..66:")
print(macro_w.round(1).to_string())

# ---- cell ----
import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
e8 = A.load_saved("e008_decomp2.parquet")
snap = A.snapshot(459)
tx = snap.transactions
first_day = tx.groupby("household_key").day.min()

g = tx.assign(wk=tx.day//7).groupby("wk").agg(sales=("sales_value","sum"), hh=("household_key","nunique"))
macro_w = (g.sales/g.hh)
macro_w = macro_w[macro_w.index <= 64]

train_days = A.snapshot_days()["train"]
cands = {}
for s in train_days:
    t = tx[tx.day <= s]
    wi = ((s - t.day) // 28).clip(lower=0)
    piv = t.assign(wi=wi).pivot_table(index="household_key", columns="wi", values="sales_value", aggfunc="sum")
    piv = piv.reindex(columns=range(26)).fillna(0.0)
    fd = first_day.reindex(piv.index)
    W = piv.values
    avail = np.array([ (s - 28*k - 27) >= fd.values for k in range(26) ])
    def decay(hl, nwin):
        d = 0.5 ** (np.arange(nwin)/hl)
        wgt = d[None,:] * avail
        wsum = wgt.sum(1); wsum[wsum==0] = 1
        return (W*wgt).sum(1)/wsum
    p26 = decay(4,26); p13h2 = decay(2,13)
    pos = (W>0) & avail
    wgt = (0.5**(np.arange(26)/2))[None,:] * pos
    wsum = wgt.sum(1); wsum[wsum==0]=1
    usual_act = (np.where(pos, W, 0)*wgt).sum(1)/wsum
    dd = t.groupby("day").sales_value.sum()
    full = pd.Series(0.0, index=range(max(1,s-181), s+1)); full.loc[dd.index] = dd
    ew28 = float(full.ewm(halflife=28).mean().iloc[-1]*28)
    win = t[t.day > s-28]
    macro_t28 = win.sales_value.sum()/max(win.household_key.nunique(),1)
    flo, fhi = s+1-364, s+28-364
    tl = t[(t.day>=flo)&(t.day<=fhi)]
    macro_fly = tl.sales_value.sum()/max(tl.household_key.nunique(),1) if len(tl) else np.nan
    mw_t = macro_w[macro_w.index <= s//7]
    macro_lvl = mw_t.tail(8).mean()
    macro_all = mw_t.mean()
    df = pd.DataFrame({"tenure": s - fd, "p26": p26, "p13h2": p13h2, "usual_act": usual_act,
                       "ewma28": ew28, "macro_t28": macro_t28, "macro_fly": macro_fly,
                       "macro_lvl": macro_lvl, "macro_all": macro_all})
    df["snapshot_day"] = s
    cands[s] = df.reset_index().rename(columns={"index":"household_key"})

C = pd.concat(cands.values(), ignore_index=True)
base = tt.merge(e8, on=["household_key","snapshot_day"])
M = base.merge(C, on=["household_key","snapshot_day"], how="left")
M["macro_rel"] = M.macro_lvl / M.macro_all
M["macro_fut_rel"] = M.macro_fly / M.macro_t28
M["day_idx"] = M.snapshot_day
M["e6_rvu"] = M.e6 * M.rvu.fillna(1).clip(lower=0, upper=3)
M["e6_ly"] = M.e6 * M.ly_ratio.fillna(1).clip(lower=0, upper=3)
M["e6_p26"] = M.e6 * M.p26.fillna(0)
M["consensus"] = M[["e13","e6","e3","e_med","b75"]].apply(lambda c: (c-c.mean())/(c.std()+1e-9)).mean(axis=1)
M["sqrt_e6"] = np.sqrt(M.e6.clip(lower=0))
M["e6_sq"] = (M.e6/100)**2

def loso(df, feats, hold_last=1, lam=10.0):
    days = sorted(df.snapshot_day.unique())
    ho = days[-hold_last:]; tr = [d for d in days if d not in ho]
    Xtr = df[df.snapshot_day.isin(tr)][feats].apply(pd.to_numeric, errors="coerce")
    ytr = df[df.snapshot_day.isin(tr)].future_spend_4w.values
    Xho = df[df.snapshot_day.isin(ho)][feats].apply(pd.to_numeric, errors="coerce")
    mu, sd = Xtr.mean(), Xtr.std()+1e-9
    Xtr_s = ((Xtr-mu)/sd).fillna(0).values; Xho_s = ((Xho-mu)/sd).fillna(0).values
    b = np.linalg.solve(Xtr_s.T@Xtr_s + lam*np.eye(len(feats)), Xtr_s.T@ytr)
    pred = Xho_s@b
    return np.abs(pred - df[df.snapshot_day.isin(ho)].future_spend_4w.values).mean()

E8 = [c for c in e8.columns if c not in ("household_key","snapshot_day")]
print("LOSO(1) E008 feats:", round(loso(M, E8),3), " LOSO(2):", round(loso(M, E8, hold_last=2),3))
newf = ["tenure","p26","p13h2","usual_act","ewma28","macro_t28","macro_fly","macro_lvl","macro_all",
        "macro_rel","macro_fut_rel","day_idx","e6_rvu","e6_ly","e6_p26","consensus","sqrt_e6","e6_sq"]
for f in newf:
    print(f"LOSO E008+{f:13s}: {loso(M, E8+[f]):.3f} / {loso(M, E8+[f], hold_last=2):.3f}")

# ---- cell ----
import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
e8 = A.load_saved("e008_decomp2.parquet")
snap = A.snapshot(459)
tx = snap.transactions
first_day = tx.groupby("household_key").day.min()

g = tx.assign(wk=tx.day//7).groupby("wk").agg(sales=("sales_value","sum"), hh=("household_key","nunique"))
macro_w = (g.sales/g.hh); macro_w = macro_w[macro_w.index <= 64]

train_days = A.snapshot_days()["train"]
cands = {}
for s in train_days:
    t = tx[tx.day <= s]
    wi = ((s - t.day) // 28).clip(lower=0)
    piv = t.assign(wi=wi).pivot_table(index="household_key", columns="wi", values="sales_value", aggfunc="sum")
    piv = piv.reindex(columns=range(26)).fillna(0.0)
    fd = first_day.reindex(piv.index)
    W = piv.values
    avail = np.array([ (s - 28*k - 27) >= fd.values for k in range(26) ])
    def decay(hl, nwin):
        d = 0.5 ** (np.arange(nwin)/hl)
        wgt = d[None,:] * avail[None,:nwin]
        wsum = wgt.sum(1); wsum[wsum==0] = 1
        return (W[:,:nwin]*wgt).sum(1)/wsum
    p26 = decay(4,26); p13h2 = decay(2,13)
    pos = (W>0) & avail
    wgt = (0.5**(np.arange(26)/2))[None,:] * pos
    wsum = wgt.sum(1); wsum[wsum==0]=1
    usual_act = (np.where(pos, W, 0)*wgt).sum(1)/wsum
    dd = t.groupby("day").sales_value.sum()
    full = pd.Series(0.0, index=range(max(1,s-181), s+1)); full.loc[dd.index] = dd
    ew28 = float(full.ewm(halflife=28).mean().iloc[-1]*28)
    win = t[t.day > s-28]
    macro_t28 = win.sales_value.sum()/max(win.household_key.nunique(),1)
    flo, fhi = s+1-364, s+28-364
    tl = t[(t.day>=flo)&(t.day<=fhi)]
    macro_fly = tl.sales_value.sum()/max(tl.household_key.nunique(),1) if len(tl) else np.nan
    mw_t = macro_w[macro_w.index <= s//7]
    macro_lvl = mw_t.tail(8).mean(); macro_all = mw_t.mean()
    df = pd.DataFrame({"tenure": s - fd, "p26": p26, "p13h2": p13h2, "usual_act": usual_act,
                       "ewma28": ew28, "macro_t28": macro_t28, "macro_fly": macro_fly,
                       "macro_lvl": macro_lvl, "macro_all": macro_all})
    df["snapshot_day"] = s
    cands[s] = df.reset_index().rename(columns={"index":"household_key"})

C = pd.concat(cands.values(), ignore_index=True)
base = tt.merge(e8, on=["household_key","snapshot_day"])
M = base.merge(C, on=["household_key","snapshot_day"], how="left")
M["macro_rel"] = M.macro_lvl / M.macro_all
M["macro_fut_rel"] = M.macro_fly / M.macro_t28
M["day_idx"] = M.snapshot_day
M["e6_rvu"] = M.e6 * M.rvu.fillna(1).clip(lower=0, upper=3)
M["e6_ly"] = M.e6 * M.ly_ratio.fillna(1).clip(lower=0, upper=3)
M["e6_p26"] = M.e6 * M.p26.fillna(0)
M["consensus"] = M[["e13","e6","e3","e_med","b75"]].apply(lambda c: (c-c.mean())/(c.std()+1e-9)).mean(axis=1)
M["sqrt_e6"] = np.sqrt(M.e6.clip(lower=0))
M["e6_sq"] = (M.e6/100)**2

def loso(df, feats, hold_last=1, lam=10.0):
    days = sorted(df.snapshot_day.unique())
    ho = days[-hold_last:]; tr = [d for d in days if d not in ho]
    Xtr = df[df.snapshot_day.isin(tr)][feats].apply(pd.to_numeric, errors="coerce")
    ytr = df[df.snapshot_day.isin(tr)].future_spend_4w.values
    Xho = df[df.snapshot_day.isin(ho)][feats].apply(pd.to_numeric, errors="coerce")
    mu, sd = Xtr.mean(), Xtr.std()+1e-9
    Xtr_s = ((Xtr-mu)/sd).fillna(0).values; Xho_s = ((Xho-mu)/sd).fillna(0).values
    b = np.linalg.solve(Xtr_s.T@Xtr_s + lam*np.eye(len(feats)), Xtr_s.T@ytr)
    pred = Xho_s@b
    return np.abs(pred - df[df.snapshot_day.isin(ho)].future_spend_4w.values).mean()

E8 = [c for c in e8.columns if c not in ("household_key","snapshot_day")]
print("LOSO(1) E008 feats:", round(loso(M, E8),3), " LOSO(2):", round(loso(M, E8, hold_last=2),3))
newf = ["tenure","p26","p13h2","usual_act","ewma28","macro_t28","macro_fly","macro_lvl","macro_all",
        "macro_rel","macro_fut_rel","day_idx","e6_rvu","e6_ly","e6_p26","consensus","sqrt_e6","e6_sq"]
for f in newf:
    print(f"LOSO E008+{f:13s}: {loso(M, E8+[f]):.3f} / {loso(M, E8+[f], hold_last=2):.3f}")

# ---- cell ----
import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
e8 = A.load_saved("e008_decomp2.parquet")
snap = A.snapshot(459)
tx = snap.transactions
first_day = tx.groupby("household_key").day.min()

g = tx.assign(wk=tx.day//7).groupby("wk").agg(sales=("sales_value","sum"), hh=("household_key","nunique"))
macro_w = (g.sales/g.hh); macro_w = macro_w[macro_w.index <= 64]

train_days = A.snapshot_days()["train"]
cands = {}
for s in train_days:
    t = tx[tx.day <= s]
    wi = ((s - t.day) // 28).clip(lower=0)
    piv = t.assign(wi=wi).pivot_table(index="household_key", columns="wi", values="sales_value", aggfunc="sum")
    piv = piv.reindex(columns=range(26)).fillna(0.0)
    fd = first_day.reindex(piv.index).values
    W = piv.values
    avail = np.array([ (s - 28*k - 27) >= fd for k in range(26) ])  # (26, n)
    def decay(hl, nwin):
        d = 0.5 ** (np.arange(nwin)/hl)
        wgt = d[:,None] * avail[:nwin]
        wsum = wgt.sum(0); wsum[wsum==0] = 1
        return (W[:,:nwin]*wgt).sum(0)/wsum
    p26 = decay(4,26); p13h2 = decay(2,13)
    pos = (W>0) & avail
    wgt = (0.5**(np.arange(26)/2))[:,None] * pos
    wsum = wgt.sum(0); wsum[wsum==0]=1
    usual_act = (np.where(pos, W, 0)*wgt).sum(0)/wsum
    dd = t.groupby("day").sales_value.sum()
    full = pd.Series(0.0, index=range(max(1,s-181), s+1)); full.loc[dd.index] = dd
    ew28 = float(full.ewm(halflife=28).mean().iloc[-1]*28)
    win = t[t.day > s-28]
    macro_t28 = win.sales_value.sum()/max(win.household_key.nunique(),1)
    flo, fhi = s+1-364, s+28-364
    tl = t[(t.day>=flo)&(t.day<=fhi)]
    macro_fly = tl.sales_value.sum()/max(tl.household_key.nunique(),1) if len(tl) else np.nan
    mw_t = macro_w[macro_w.index <= s//7]
    macro_lvl = mw_t.tail(8).mean(); macro_all = mw_t.mean()
    df = pd.DataFrame({"tenure": s - fd, "p26": p26, "p13h2": p13h2, "usual_act": usual_act,
                       "ewma28": ew28, "macro_t28": macro_t28, "macro_fly": macro_fly,
                       "macro_lvl": macro_lvl, "macro_all": macro_all})
    df["snapshot_day"] = s
    cands[s] = df.reset_index().rename(columns={"index":"household_key"})

C = pd.concat(cands.values(), ignore_index=True)
base = tt.merge(e8, on=["household_key","snapshot_day"])
M = base.merge(C, on=["household_key","snapshot_day"], how="left")
M["macro_rel"] = M.macro_lvl / M.macro_all
M["macro_fut_rel"] = M.macro_fly / M.macro_t28
M["day_idx"] = M.snapshot_day
M["e6_rvu"] = M.e6 * M.rvu.fillna(1).clip(lower=0, upper=3)
M["e6_ly"] = M.e6 * M.ly_ratio.fillna(1).clip(lower=0, upper=3)
M["e6_p26"] = M.e6 * M.p26.fillna(0)
M["consensus"] = M[["e13","e6","e3","e_med","b75"]].apply(lambda c: (c-c.mean())/(c.std()+1e-9)).mean(axis=1)
M["sqrt_e6"] = np.sqrt(M.e6.clip(lower=0))
M["e6_sq"] = (M.e6/100)**2

def loso(df, feats, hold_last=1, lam=10.0):
    days = sorted(df.snapshot_day.unique())
    ho = days[-hold_last:]; tr = [d for d in days if d not in ho]
    Xtr = df[df.snapshot_day.isin(tr)][feats].apply(pd.to_numeric, errors="coerce")
    ytr = df[df.snapshot_day.isin(tr)].future_spend_4w.values
    Xho = df[df.snapshot_day.isin(ho)][feats].apply(pd.to_numeric, errors="coerce")
    mu, sd = Xtr.mean(), Xtr.std()+1e-9
    Xtr_s = ((Xtr-mu)/sd).fillna(0).values; Xho_s = ((Xho-mu)/sd).fillna(0).values
    b = np.linalg.solve(Xtr_s.T@Xtr_s + lam*np.eye(len(feats)), Xtr_s.T@ytr)
    pred = Xho_s@b
    return np.abs(pred - df[df.snapshot_day.isin(ho)].future_spend_4w.values).mean()

E8 = [c for c in e8.columns if c not in ("household_key","snapshot_day")]
print("LOSO(1) E008 feats:", round(loso(M, E8),3), " LOSO(2):", round(loso(M, E8, hold_last=2),3))
newf = ["tenure","p26","p13h2","usual_act","ewma28","macro_t28","macro_fly","macro_lvl","macro_all",
        "macro_rel","macro_fut_rel","day_idx","e6_rvu","e6_ly","e6_p26","consensus","sqrt_e6","e6_sq"]
for f in newf:
    print(f"LOSO E008+{f:13s}: {loso(M, E8+[f]):.3f} / {loso(M, E8+[f], hold_last=2):.3f}")

# ---- cell ----
import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
e8 = A.load_saved("e008_decomp2.parquet")
snap = A.snapshot(459)
tx = snap.transactions
first_day = tx.groupby("household_key").day.min()

g = tx.assign(wk=tx.day//7).groupby("wk").agg(sales=("sales_value","sum"), hh=("household_key","nunique"))
macro_w = (g.sales/g.hh); macro_w = macro_w[macro_w.index <= 64]

train_days = A.snapshot_days()["train"]
cands = {}
for s in train_days:
    t = tx[tx.day <= s]
    wi = ((s - t.day) // 28).clip(lower=0)
    piv = t.assign(wi=wi).pivot_table(index="household_key", columns="wi", values="sales_value", aggfunc="sum")
    piv = piv.reindex(columns=range(26)).fillna(0.0)
    fd = first_day.reindex(piv.index).values
    W = piv.values  # (n, 26), col 0 = most recent window
    avail = np.array([ (s - 28*k - 27) >= fd for k in range(26) ])  # (26, n)
    availT = avail.T  # (n, 26)
    def decay(hl, nwin):
        d = 0.5 ** (np.arange(nwin)/hl)
        wgt = d[None,:] * availT[:,:nwin]   # (n, nwin)
        wsum = wgt.sum(1); wsum[wsum==0] = 1
        return (W[:,:nwin]*wgt).sum(1)/wsum
    p26 = decay(4,26); p13h2 = decay(2,13)
    pos = (W>0) & availT
    wgt = (0.5**(np.arange(26)/2))[None,:] * pos
    wsum = wgt.sum(1); wsum[wsum==0]=1
    usual_act = (np.where(pos, W, 0)*wgt).sum(1)/wsum
    dd = t.groupby("day").sales_value.sum()
    full = pd.Series(0.0, index=range(max(1,s-181), s+1)); full.loc[dd.index] = dd
    ew28 = float(full.ewm(halflife=28).mean().iloc[-1]*28)
    win = t[t.day > s-28]
    macro_t28 = win.sales_value.sum()/max(win.household_key.nunique(),1)
    flo, fhi = s+1-364, s+28-364
    tl = t[(t.day>=flo)&(t.day<=fhi)]
    macro_fly = tl.sales_value.sum()/max(tl.household_key.nunique(),1) if len(tl) else np.nan
    mw_t = macro_w[macro_w.index <= s//7]
    macro_lvl = mw_t.tail(8).mean(); macro_all = mw_t.mean()
    df = pd.DataFrame({"tenure": s - fd, "p26": p26, "p13h2": p13h2, "usual_act": usual_act,
                       "ewma28": ew28, "macro_t28": macro_t28, "macro_fly": macro_fly,
                       "macro_lvl": macro_lvl, "macro_all": macro_all})
    df["snapshot_day"] = s
    cands[s] = df.reset_index().rename(columns={"index":"household_key"})

C = pd.concat(cands.values(), ignore_index=True)
base = tt.merge(e8, on=["household_key","snapshot_day"])
M = base.merge(C, on=["household_key","snapshot_day"], how="left")
M["macro_rel"] = M.macro_lvl / M.macro_all
M["macro_fut_rel"] = M.macro_fly / M.macro_t28
M["day_idx"] = M.snapshot_day
M["e6_rvu"] = M.e6 * M.rvu.fillna(1).clip(lower=0, upper=3)
M["e6_ly"] = M.e6 * M.ly_ratio.fillna(1).clip(lower=0, upper=3)
M["e6_p26"] = M.e6 * M.p26.fillna(0)
M["consensus"] = M[["e13","e6","e3","e_med","b75"]].apply(lambda c: (c-c.mean())/(c.std()+1e-9)).mean(axis=1)
M["sqrt_e6"] = np.sqrt(M.e6.clip(lower=0))
M["e6_sq"] = (M.e6/100)**2

def loso(df, feats, hold_last=1, lam=10.0):
    days = sorted(df.snapshot_day.unique())
    ho = days[-hold_last:]; tr = [d for d in days if d not in ho]
    Xtr = df[df.snapshot_day.isin(tr)][feats].apply(pd.to_numeric, errors="coerce")
    ytr = df[df.snapshot_day.isin(tr)].future_spend_4w.values
    Xho = df[df.snapshot_day.isin(ho)][feats].apply(pd.to_numeric, errors="coerce")
    mu, sd = Xtr.mean(), Xtr.std()+1e-9
    Xtr_s = ((Xtr-mu)/sd).fillna(0).values; Xho_s = ((Xho-mu)/sd).fillna(0).values
    b = np.linalg.solve(Xtr_s.T@Xtr_s + lam*np.eye(len(feats)), Xtr_s.T@ytr)
    pred = Xho_s@b
    return np.abs(pred - df[df.snapshot_day.isin(ho)].future_spend_4w.values).mean()

E8 = [c for c in e8.columns if c not in ("household_key","snapshot_day")]
print("LOSO(1) E008 feats:", round(loso(M, E8),3), " LOSO(2):", round(loso(M, E8, hold_last=2),3))
newf = ["tenure","p26","p13h2","usual_act","ewma28","macro_t28","macro_fly","macro_lvl","macro_all",
        "macro_rel","macro_fut_rel","day_idx","e6_rvu","e6_ly","e6_p26","consensus","sqrt_e6","e6_sq"]
for f in newf:
    print(f"LOSO E008+{f:13s}: {loso(M, E8+[f]):.3f} / {loso(M, E8+[f], hold_last=2):.3f}")

# ---- cell ----
import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
e8 = A.load_saved("e008_decomp2.parquet")
snap = A.snapshot(459)
tx = snap.transactions
first_day = tx.groupby("household_key").day.min()

g = tx.assign(wk=tx.day//7).groupby("wk").agg(sales=("sales_value","sum"), hh=("household_key","nunique"))
macro_w = (g.sales/g.hh); macro_w = macro_w[macro_w.index <= 64]

train_days = A.snapshot_days()["train"]
cands = {}
for s in train_days:
    t = tx[tx.day <= s]
    wi = ((s - t.day) // 28).clip(lower=0)
    piv = t.assign(wi=wi).pivot_table(index="household_key", columns="wi", values="sales_value", aggfunc="sum")
    piv = piv.reindex(columns=range(26)).fillna(0.0)
    fd = first_day.reindex(piv.index).values
    W = piv.values  # (n, 26), col 0 = most recent window
    avail = np.array([ (s - 28*k - 27) >= fd for k in range(26) ]).T  # (n, 26)
    def decay(hl, nwin):
        d = 0.5 ** (np.arange(nwin)/hl)
        wgt = d[None,:] * avail[:,:nwin]
        wsum = wgt.sum(1); wsum[wsum==0] = 1
        return (W[:,:nwin]*wgt).sum(1)/wsum
    p26 = decay(4,26); p13h2 = decay(2,13)
    pos = (W>0) & avail
    wgt = (0.5**(np.arange(26)/2))[None,:] * pos
    wsum = wgt.sum(1); wsum[wsum==0]=1
    usual_act = (np.where(pos, W, 0)*wgt).sum(1)/wsum
    dd = t.groupby("day").sales_value.sum()
    dd = dd[dd.index >= max(1, s-181)]
    full = pd.Series(0.0, index=range(max(1,s-181), s+1)); full.loc[dd.index] = dd
    ew28 = float(full.ewm(halflife=28).mean().iloc[-1]*28)
    win = t[t.day > s-28]
    macro_t28 = win.sales_value.sum()/max(win.household_key.nunique(),1)
    flo, fhi = s+1-364, s+28-364
    tl = t[(t.day>=flo)&(t.day<=fhi)]
    macro_fly = tl.sales_value.sum()/max(tl.household_key.nunique(),1) if len(tl) else np.nan
    mw_t = macro_w[macro_w.index <= s//7]
    macro_lvl = mw_t.tail(8).mean(); macro_all = mw_t.mean()
    df = pd.DataFrame({"tenure": s - fd, "p26": p26, "p13h2": p13h2, "usual_act": usual_act,
                       "ewma28": ew28, "macro_t28": macro_t28, "macro_fly": macro_fly,
                       "macro_lvl": macro_lvl, "macro_all": macro_all})
    df["snapshot_day"] = s
    cands[s] = df.reset_index().rename(columns={"index":"household_key"})

C = pd.concat(cands.values(), ignore_index=True)
base = tt.merge(e8, on=["household_key","snapshot_day"])
M = base.merge(C, on=["household_key","snapshot_day"], how="left")
M["macro_rel"] = M.macro_lvl / M.macro_all
M["macro_fut_rel"] = M.macro_fly / M.macro_t28
M["day_idx"] = M.snapshot_day
M["e6_rvu"] = M.e6 * M.rvu.fillna(1).clip(lower=0, upper=3)
M["e6_ly"] = M.e6 * M.ly_ratio.fillna(1).clip(lower=0, upper=3)
M["e6_p26"] = M.e6 * M.p26.fillna(0)
M["consensus"] = M[["e13","e6","e3","e_med","b75"]].apply(lambda c: (c-c.mean())/(c.std()+1e-9)).mean(axis=1)
M["sqrt_e6"] = np.sqrt(M.e6.clip(lower=0))
M["e6_sq"] = (M.e6/100)**2

def loso(df, feats, hold_last=1, lam=10.0):
    days = sorted(df.snapshot_day.unique())
    ho = days[-hold_last:]; tr = [d for d in days if d not in ho]
    Xtr = df[df.snapshot_day.isin(tr)][feats].apply(pd.to_numeric, errors="coerce")
    ytr = df[df.snapshot_day.isin(tr)].future_spend_4w.values
    Xho = df[df.snapshot_day.isin(ho)][feats].apply(pd.to_numeric, errors="coerce")
    mu, sd = Xtr.mean(), Xtr.std()+1e-9
    Xtr_s = ((Xtr-mu)/sd).fillna(0).values; Xho_s = ((Xho-mu)/sd).fillna(0).values
    b = np.linalg.solve(Xtr_s.T@Xtr_s + lam*np.eye(len(feats)), Xtr_s.T@ytr)
    pred = Xho_s@b
    return np.abs(pred - df[df.snapshot_day.isin(ho)].future_spend_4w.values).mean()

E8 = [c for c in e8.columns if c not in ("household_key","snapshot_day")]
print("LOSO(1) E008 feats:", round(loso(M, E8),3), " LOSO(2):", round(loso(M, E8, hold_last=2),3))
newf = ["tenure","p26","p13h2","usual_act","ewma28","macro_t28","macro_fly","macro_lvl","macro_all",
        "macro_rel","macro_fut_rel","day_idx","e6_rvu","e6_ly","e6_p26","consensus","sqrt_e6","e6_sq"]
for f in newf:
    print(f"LOSO E008+{f:13s}: {loso(M, E8+[f]):.3f} / {loso(M, E8+[f], hold_last=2):.3f}")

# ---- cell ----
import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
e8 = A.load_saved("e008_decomp2.parquet")
e1 = A.load_saved("e001_history.parquet")
snap = A.snapshot(459)
tx = snap.transactions
first_day = tx.groupby("household_key").day.min()
g = tx.assign(wk=tx.day//7).groupby("wk").agg(sales=("sales_value","sum"), hh=("household_key","nunique"))
macro_w = (g.sales/g.hh); macro_w = macro_w[macro_w.index <= 64]
train_days = A.snapshot_days()["train"]

rows = {}
for s in train_days:
    t = tx[tx.day <= s]
    wi = ((s - t.day) // 28).clip(lower=0)
    piv = t.assign(wi=wi).pivot_table(index="household_key", columns="wi", values="sales_value", aggfunc="sum").reindex(columns=range(26)).fillna(0.0)
    fd = first_day.reindex(piv.index).values
    hh_idx = piv.index
    # year-ago future window: days s-363 .. s-336
    flo, fhi = s-363, s-336
    tw = t[(t.day >= flo) & (t.day <= fhi)].groupby("household_key").sales_value.sum()
    mfly = tw.reindex(hh_idx).fillna(0.0).values
    obs_lo = np.maximum(flo, fd); obs_hi = np.minimum(fhi, s)
    frac = np.clip((obs_hi - obs_lo + 1) / 28.0, 0, 1)  # observable fraction of the 28d window
    mfly_adj = np.where(frac > 0.25, mfly / np.maximum(frac, 0.25), np.nan)
    # macro drift
    mw_t = macro_w[macro_w.index <= s//7]
    macro_lvl = mw_t.tail(8).mean(); macro_all = mw_t.mean()
    # year-ago future weeks macro (seasonality) - only where available
    wf0, wf1 = (s+1)//7, (s+28)//7
    ya = macro_w.reindex(range(wf0-52, wf1-52+1)).dropna()
    mfs = (ya.mean()/macro_lvl) if len(ya) >= 2 else np.nan
    df = pd.DataFrame({"household_key": hh_idx, "mfly": mfly, "mfly_frac": frac,
                       "mfly_adj": mfly_adj, "macro_rel": macro_lvl/macro_all,
                       "macro_fut_seas": mfs, "tenure": s - fd})
    df["mfut_rel"] = df.mfly_adj / np.maximum(piv[0].values, 1.0)
    df["mfly_raw_rel"] = df.mfly / np.maximum(piv[0].values, 1.0)
    df["snapshot_day"] = s
    rows[s] = df
C = pd.concat(rows.values(), ignore_index=True)

base = tt.merge(e8, on=["household_key","snapshot_day"])
M = base.merge(C, on=["household_key","snapshot_day"], how="left")
M["h200"] = np.maximum(M.e6-200, 0); M["h400"] = np.maximum(M.e6-400, 0)
M["rank_e6"] = M.groupby("snapshot_day").e6.rank(pct=True)
M["rank_usual13"] = M.groupby("snapshot_day").usual13.rank(pct=True)
M["sqrt_e6"] = np.sqrt(M.e6.clip(lower=0))
E1raw = ["spend_7","spend_56","spend_84","spend_182","spend_365","nb_28","nb_56","nb_all","avg_basket_28",
         "days_since_last","spend_prev28","trend28","qty_28","nprod_28","spend_ly28","weekly_rate_84"]
M = M.merge(e1[["household_key","snapshot_day"]+E1raw], on=["household_key","snapshot_day"], how="left")

# correlations of new macro feats with E008 ly features
for a,b in [("mfly","ly_spend"),("mfly_adj","ly_spend"),("mfut_rel","ly_ratio"),("mfly_raw_rel","ly_ratio")]:
    ok = M[a].notna() & M[b].notna()
    print(f"corr({a},{b}) = {np.corrcoef(M.loc[ok,a], M.loc[ok,b])[0,1]:.3f}  (n={ok.sum()})")
print("mfly_adj NaN frac by snapshot:"); print(M.groupby("snapshot_day").mfly_adj.apply(lambda x: x.isna().mean().round(2)).to_string())

def loso(df, feats, hold_last=1, lam=10.0):
    days = sorted(df.snapshot_day.unique())
    ho = days[-hold_last:]; tr = [d for d in days if d not in ho]
    Xtr = df[df.snapshot_day.isin(tr)][feats].apply(pd.to_numeric, errors="coerce")
    ytr = df[df.snapshot_day.isin(tr)].future_spend_4w.values
    Xho = df[df.snapshot_day.isin(ho)][feats].apply(pd.to_numeric, errors="coerce")
    mu, sd = Xtr.mean(), Xtr.std()+1e-9
    Xtr_s = ((Xtr-mu)/sd).fillna(0).values; Xho_s = ((Xho-mu)/sd).fillna(0).values
    b = np.linalg.solve(Xtr_s.T@Xtr_s + lam*np.eye(len(feats)), Xtr_s.T@ytr)
    return np.abs(Xho_s@b - df[df.snapshot_day.isin(ho)].future_spend_4w.values).mean()

E8 = [c for c in e8.columns if c not in ("household_key","snapshot_day")]
b1 = loso(M, E8); b2 = loso(M, E8, hold_last=2)
print(f"\nBASE E008: {b1:.3f} / {b2:.3f}")
tests = {
 "mfly_raw": ["mfly"],
 "mfly_adj+frac": ["mfly_adj","mfly_frac"],
 "mfut_rel": ["mfut_rel"],
 "mfly_raw_rel": ["mfly_raw_rel"],
 "macro_rel": ["macro_rel"],
 "macro_fut_seas": ["macro_fut_seas"],
 "tenure": ["tenure"],
 "macro_grp": ["mfly_adj","mfly_frac","mfut_rel","macro_rel"],
 "macro_grp+ten": ["mfly_adj","mfly_frac","mfut_rel","macro_rel","tenure"],
 "hinges": ["h200","h400"],
 "sqrt": ["sqrt_e6"],
 "ranks": ["rank_e6","rank_usual13"],
 "w0w5": ["mfly"],  # placeholder replaced below
}
# raw windows w0..w5
M["w0"]=M.spend_28  # spend_28 == w0
# compute w1..w5 quickly from e1? not present; approximate via spend_56-spend_28 etc.
M["w1"] = M.spend_56 - M.spend_28
M["w2"] = M.spend_84 - M.spend_56
M["w3"] = M.spend_182 - M.spend_84  # 98d window, not exact; rough
tests["w12"] = ["w1","w2"]
tests["e1union"] = E1raw
tests["macro+hinges"] = ["mfly_adj","mfly_frac","mfut_rel","macro_rel","h200","h400"]
for k,v in tests.items():
    if k=="w0w5": continue
    print(f"{k:16s}: {loso(M, E8+v):.3f} / {loso(M, E8+v, hold_last=2):.3f}")

# ---- cell ----
import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
e8 = A.load_saved("e008_decomp2.parquet")
e7 = A.load_saved("e007_new.parquet")
e1 = A.load_saved("e001_history.parquet")
snap = A.snapshot(459)
tx = snap.transactions
first_day = tx.groupby("household_key").day.min()
g = tx.assign(wk=tx.day//7).groupby("wk").agg(sales=("sales_value","sum"), hh=("household_key","nunique"))
macro_w = (g.sales/g.hh); macro_w = macro_w[macro_w.index <= 64]
train_days = A.snapshot_days()["train"]

rows = {}
for s in train_days:
    t = tx[tx.day <= s]
    wi = ((s - t.day) // 28).clip(lower=0)
    piv = t.assign(wi=wi).pivot_table(index="household_key", columns="wi", values="sales_value", aggfunc="sum").reindex(columns=range(26)).fillna(0.0)
    fd = first_day.reindex(piv.index).values
    W = piv.values
    avail = np.array([ (s - 28*k - 27) >= fd for k in range(26) ]).T
    def dec_p(hl, nwin):
        d = 0.5 ** (np.arange(nwin)/hl)
        wgt = d[None,:] * avail[:,:nwin]
        wsum = wgt.sum(1); wsum[wsum==0]=1
        return (W[:,:nwin]*wgt).sum(1)/wsum
    def dec_u(hl, nwin):
        pos = (W[:,:nwin]>0) & avail[:,:nwin]
        d = 0.5 ** (np.arange(nwin)/hl)
        wgt = d[None,:] * pos
        wsum = wgt.sum(1); wsum[wsum==0]=1
        return (np.where(pos, W[:,:nwin], 0)*wgt).sum(1)/wsum
    mw_t = macro_w[macro_w.index <= (s-6)//7]  # full weeks only
    macro_lvl = mw_t.tail(8).mean(); macro_all = mw_t.mean()
    wf0 = (s+1)//7
    ya = macro_w.reindex(range(wf0-52, wf0-52+4)).dropna()
    mfs = (ya.mean()/macro_lvl) if len(ya)>=2 else np.nan
    flo, fhi = s-363, s-336
    tw = t[(t.day>=flo)&(t.day<=fhi)].groupby("household_key").sales_value.sum()
    mfly = tw.reindex(piv.index).fillna(0.0).values
    # weekly activity last 8 weeks
    t8 = t[t.day > s-56]
    nact8 = t8.groupby("household_key").day.nunique().reindex(piv.index).fillna(0).values
    t7 = t[t.day > s-7].groupby("household_key").sales_value.sum().reindex(piv.index).fillna(0).values
    df = pd.DataFrame({"household_key": piv.index,
        "p1_13": dec_p(1,13), "p26_8": dec_p(8,26), "u1_13": dec_u(1,13), "u26_8": dec_u(8,26),
        "usual_act": dec_u(2,26), "nact8": nact8, "spend_7d": t7,
        "mfly": mfly, "macro_rel": macro_lvl/macro_all, "macro_fut_seas": mfs, "tenure": s-fd})
    df["snapshot_day"] = s
    rows[s] = df
C = pd.concat(rows.values(), ignore_index=True)

base = tt.merge(e8, on=["household_key","snapshot_day"])
M = base.merge(C, on=["household_key","snapshot_day"], how="left")
E8 = [c for c in e8.columns if c not in ("household_key","snapshot_day")]
E7 = [c for c in e7.columns if c not in ("household_key","snapshot_day")]
E1 = [c for c in e1.columns if c not in ("household_key","snapshot_day")]
E1num = [c for c in E1 if pd.api.types.is_numeric_dtype(e1[c])]
# demographics one-hot for local screen
dem = e1[["household_key","snapshot_day","has_demographics"]].copy()
for c in ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc"]:
    d = pd.get_dummies(e1[c], prefix=c[:6])
    dem = pd.concat([dem, d], axis=1)
DEM = [c for c in dem.columns if c not in ("household_key","snapshot_day")]
M = M.merge(dem, on=["household_key","snapshot_day"], how="left")

def loso(df, feats, hold_last=1, lam=10.0):
    days = sorted(df.snapshot_day.unique())
    ho = days[-hold_last:]; tr = [d for d in days if d not in ho]
    Xtr = df[df.snapshot_day.isin(tr)][feats].apply(pd.to_numeric, errors="coerce")
    ytr = df[df.snapshot_day.isin(tr)].future_spend_4w.values
    Xho = df[df.snapshot_day.isin(ho)][feats].apply(pd.to_numeric, errors="coerce")
    mu, sd = Xtr.mean(), Xtr.std()+1e-9
    Xtr_s = ((Xtr-mu)/sd).fillna(0).values; Xho_s = ((Xho-mu)/sd).fillna(0).values
    b = np.linalg.solve(Xtr_s.T@Xtr_s + lam*np.eye(len(feats)), Xtr_s.T@ytr)
    return np.abs(Xho_s@b - df[df.snapshot_day.isin(ho)].future_spend_4w.values).mean()

print("SANITY (harness: E001=63.0, E007=62.8, E008=61.7):")
print("  E001:", round(loso(M, E1num),3), "/", round(loso(M, E1num,2),3))
print("  E007:", round(loso(M, E7),3), "/", round(loso(M, E7,2),3))
print("  E008:", round(loso(M, E8),3), "/", round(loso(M, E8,2),3))
print("\nCANDIDATES:")
tests = {
 "macro2": ["macro_rel","macro_fut_seas"],
 "macro2+mfly+ten": ["macro_rel","macro_fut_seas","mfly","tenure"],
 "macro2+mfly": ["macro_rel","macro_fut_seas","mfly"],
 "decay_variants": ["p1_13","p26_8","u1_13","u26_8","usual_act","nact8","spend_7d"],
 "demog": DEM,
 "macro2+decay": ["macro_rel","macro_fut_seas","p1_13","p26_8","u1_13","u26_8","usual_act","nact8","spend_7d"],
 "macro2+decay+mfly": ["macro_rel","macro_fut_seas","mfly","p1_13","p26_8","u1_13","u26_8","usual_act","nact8","spend_7d"],
}
for k,v in tests.items():
    print(f"  {k:18s}: {loso(M, E8+v):.3f} / {loso(M, E8+v,2):.3f}  (n={len(v)})")
print("\nmacro_rel by snapshot (train):"); print(M.groupby("snapshot_day").macro_rel.first().round(3).to_string())
print("macro_fut_seas by snapshot (train):"); print(M.groupby("snapshot_day").macro_fut_seas.first().round(3).to_string())

# ---- cell ----
import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
e8 = A.load_saved("e008_decomp2.parquet"); e7 = A.load_saved("e007_new.parquet"); e1 = A.load_saved("e001_history.parquet")
snap = A.snapshot(459); tx = snap.transactions
first_day = tx.groupby("household_key").day.min()
g = tx.assign(wk=tx.day//7).groupby("wk").agg(sales=("sales_value","sum"), hh=("household_key","nunique"))
macro_w = (g.sales/g.hh); macro_w = macro_w[macro_w.index <= 64]
train_days = A.snapshot_days()["train"]
rows = {}
for s in train_days:
    t = tx[tx.day <= s]
    wi = ((s - t.day) // 28).clip(lower=0)
    piv = t.assign(wi=wi).pivot_table(index="household_key", columns="wi", values="sales_value", aggfunc="sum").reindex(columns=range(26)).fillna(0.0)
    fd = first_day.reindex(piv.index).values
    mw_t = macro_w[macro_w.index <= (s-6)//7]
    macro_lvl = mw_t.tail(8).mean(); macro_all = mw_t.mean()
    wf0 = (s+1)//7
    ya = macro_w.reindex(range(wf0-52, wf0-52+4)).dropna()
    mfs = (ya.mean()/macro_lvl) if len(ya)>=2 else np.nan
    flo, fhi = s-363, s-336
    tw = t[(t.day>=flo)&(t.day<=fhi)].groupby("household_key").sales_value.sum()
    mfly = tw.reindex(piv.index).fillna(0.0).values
    df = pd.DataFrame({"household_key": piv.index, "mfly": mfly, "macro_rel": macro_lvl/macro_all,
                       "macro_fut_seas": mfs, "tenure": s-fd})
    df["snapshot_day"] = s
    rows[s] = df
C = pd.concat(rows.values(), ignore_index=True)
base = tt.merge(e8, on=["household_key","snapshot_day"])
M = base.merge(C, on=["household_key","snapshot_day"], how="left")
M = M.merge(e1[["household_key","snapshot_day","spend_7","spend_56","spend_84","spend_182","spend_365","spend_all","nb_28","nb_56","nb_all","avg_basket_28","days_since_last","spend_prev28","trend28","qty_28","nprod_28","spend_ly28","weekly_rate_84","snapshot_day_index","week_of_year","index","ly_avail","log_spend_28","log_spend_all","has_demographics"]], on=["household_key","snapshot_day"], how="left")
E8 = [c for c in e8.columns if c not in ("household_key","snapshot_day")]
E7 = [c for c in e7.columns if c not in ("household_key","snapshot_day")]
E1num = ["spend_7","spend_56","spend_84","spend_182","spend_365","spend_all","nb_28","nb_56","nb_all","avg_basket_28","days_since_last","spend_prev28","trend28","qty_28","nprod_28","spend_ly28","weekly_rate_84","snapshot_day_index","week_of_year","index","ly_avail","log_spend_28","log_spend_all","has_demographics"]

def loso(df, feats, hold_last=1, lam=10.0):
    days = sorted(df.snapshot_day.unique())
    ho = days[-hold_last:]; tr = [d for d in days if d not in ho]
    Xtr = df[df.snapshot_day.isin(tr)][feats].apply(pd.to_numeric, errors="coerce")
    ytr = df[df.snapshot_day.isin(tr)].future_spend_4w.values
    Xho = df[df.snapshot_day.isin(ho)][feats].apply(pd.to_numeric, errors="coerce")
    mu, sd = Xtr.mean(), Xtr.std()+1e-9
    Xtr_s = ((Xtr-mu)/sd).fillna(0).values; Xho_s = ((Xho-mu)/sd).fillna(0).values
    b = np.linalg.solve(Xtr_s.T@Xtr_s + lam*np.eye(len(feats)), Xtr_s.T@ytr)
    return np.abs(Xho_s@b - df[df.snapshot_day.isin(ho)].future_spend_4w.values).mean()

print("SANITY (harness: E001=63.0, E007=62.8, E008=61.7):")
print("  E001:", round(loso(M, E1num),3), "/", round(loso(M, E1num,2),3))
print("  E007:", round(loso(M, E7),3), "/", round(loso(M, E7,2),3))
print("  E008:", round(loso(M, E8),3), "/", round(loso(M, E8,2),3))
tests = {
 "macro2": ["macro_rel","macro_fut_seas"],
 "macro2+mfly": ["macro_rel","macro_fut_seas","mfly"],
 "macro2+mfly+ten": ["macro_rel","macro_fut_seas","mfly","tenure"],
 "macro2+ten": ["macro_rel","macro_fut_seas","tenure"],
}
for k,v in tests.items():
    print(f"  {k:16s}: {loso(M, E8+v):.3f} / {loso(M, E8+v,2):.3f}")
print("macro_rel by snap:", M.groupby("snapshot_day").macro_rel.first().round(3).values)
print("macro_fut_seas by snap:", M.groupby("snapshot_day").macro_fut_seas.first().round(3).values)

# ---- cell ----
import pandas as pd, numpy as np
import agent_api as A

def fn(view, s):
    tx = view.transactions
    h = view.households
    if isinstance(h, pd.DataFrame):
        hh = pd.Index(h["household_key"]) if "household_key" in h.columns else pd.Index(h.index)
    else:
        hh = pd.Index(h)
    # household spend in the year-ago analogue of the future window (days s-363..s-336)
    flo, fhi = s - 363, s - 336
    tw = tx[(tx.day >= flo) & (tx.day <= fhi)].groupby("household_key").sales_value.sum()
    mfly = tw.reindex(hh).fillna(0.0) if len(hh) else tw
    # macro weekly per-active-household spend, full weeks only (week w = days 7w..7w+6)
    wk = (tx.day // 7).astype(int)
    g = tx.assign(wk=wk).groupby("wk").agg(sales=("sales_value", "sum"), hh=("household_key", "nunique"))
    macro_w = g.sales / g.hh
    full = macro_w[macro_w.index <= (s - 6) // 7]
    macro_lvl = float(full.tail(8).mean()) if len(full) else np.nan
    macro_all = float(full.mean()) if len(full) else np.nan
    macro_rel = macro_lvl / macro_all if (macro_all and macro_all > 0) else 1.0
    # year-ago macro weeks matching the future window, relative to recent macro level
    w0, w1 = (s + 1) // 7, (s + 28) // 7
    ya = macro_w.reindex(range(w0 - 52, w1 - 52 + 1)).dropna()
    mfs = float(ya.mean() / macro_lvl) if (len(ya) >= 2 and macro_lvl and macro_lvl > 0) else 1.0
    return pd.DataFrame({"mfly": mfly, "macro_rel": macro_rel, "macro_fut_seas": mfs})

newf = A.build_features(fn)
print(newf.shape)
print(newf.groupby("snapshot_day")[["macro_rel", "macro_fut_seas"]].first().round(4).to_string())
print(newf.mfly.describe().round(2).to_string())

e8 = A.load_saved("e008_decomp2.parquet")
m = e8.merge(newf, on=["household_key", "snapshot_day"], how="left")
for c in ["macro_rel", "macro_fut_seas"]:
    m[c] = m.groupby("snapshot_day")[c].transform("first")
m["macro_rel"] = m["macro_rel"].fillna(1.0)
m["macro_fut_seas"] = m["macro_fut_seas"].fillna(1.0)
m["mfly"] = m["mfly"].fillna(0.0)
m["e6_mfs"] = m.e6 * (m.macro_fut_seas - 1.0)
m["usual13_mfs"] = m.usual13 * (m.macro_fut_seas - 1.0)
m["b75_mfs"] = m.b75 * (m.macro_fut_seas - 1.0)
print("merged:", m.shape, "| NaNs:", m.isna().sum().sum())
path = A.save_table(m, "e009_macro")
print("saved:", path)
print("new cols:", [c for c in m.columns if c not in e8.columns])