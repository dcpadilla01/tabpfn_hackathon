import agent_api, pandas as pd
base = agent_api.load_saved("e006_newblock.parquet")
print(base.shape)
cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
print(len(cols))
print(cols)


# ---- cell ----
import agent_api
t = agent_api.load_saved("e006_zero_inflation.parquet")
print(t.shape)
feat = [c for c in t.columns if c not in ("household_key","snapshot_day")]
print(len(feat))
print(feat)


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def fn(view, s):
    tx = view.table("transactions")
    hh = view.households
    tx = tx[tx.household_key.isin(hh)]
    # lag1: spend in [s-28, s-1]  (what the target at snapshot s-28 would have been)
    def win(lo, hi):
        w = tx[(tx.day >= lo) & (tx.day <= hi)]
        return w.groupby("household_key").sales_value.sum()
    lag1 = win(s-28, s-1)
    lag2 = win(s-56, s-29)
    lag3 = win(s-84, s-57)
    lag4 = win(s-112, s-85)
    ly   = win(s-28-364, s-1-364)   # same 4w window last year
    df = pd.DataFrame({"lag1": lag1, "lag2": lag2, "lag3": lag3, "lag4": lag4, "ly_lag": ly})
    df = df.reindex(hh).fillna(0.0)
    return df

feats = agent_api.build_features(fn)
tt = agent_api.train_targets()
m = tt.merge(feats, on=["household_key","snapshot_day"])
tr = m[m.snapshot_day <= 431]
for c in ["lag1","lag2","lag3","lag4","ly_lag"]:
    print(c, "corr:", round(np.corrcoef(tr[c], tr.future_spend_4w)[0,1],3),
          "mean:", round(tr[c].mean(),1))
print("target mean/std:", round(tr.future_spend_4w.mean(),1), round(tr.future_spend_4w.std(),1))
# how often is lag1 exactly the target?
print("lag1==target frac:", (tr.lag1==tr.future_spend_4w).mean())
# zero structure
print("target zero frac:", (tr.future_spend_4w==0).mean(), "lag1 zero frac:", (tr.lag1==0).mean())


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def fn(view, s):
    tx = view.table("transactions")
    hh = view.households
    tx = tx[tx.household_key.isin(hh)]
    g = tx.groupby(["household_key","day"]).sales_value.sum().reset_index()
    # daily spend series per household, then window sums for candidate definitions
    def wsum(lo, hi):
        w = g[(g.day >= lo) & (g.day <= hi)]
        return w.groupby("household_key").sales_value.sum()
    out = pd.DataFrame(index=hh)
    out["c_fwd_1_28"] = wsum(s+1, s+28)      # stated definition
    out["c_fwd_0_27"] = wsum(s, s+27)
    out["c_fwd_0_28"] = wsum(s, s+28)
    out["c_bwd_27_0"] = wsum(s-27, s)
    out["c_fwd_1_29"] = wsum(s+1, s+29)
    return out.fillna(0.0)

feats = agent_api.build_features(fn)
tt = agent_api.train_targets()
m = tt.merge(feats, on=["household_key","snapshot_day"])
for c in ["c_fwd_1_28","c_fwd_0_27","c_fwd_0_28","c_bwd_27_0","c_fwd_1_29"]:
    eq = (m[c] == m.future_spend_4w).mean()
    print(c, "exact-match frac:", round(eq,4), " corr:", round(np.corrcoef(m[c], m.future_spend_4w)[0,1],4))


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def fn(view, s):
    tx = view.table("transactions")
    hh = view.households
    tx = tx[tx.household_key.isin(hh)]
    g = tx.groupby(["household_key","day"]).sales_value.sum().reset_index()
    def wsum(lo, hi):
        w = g[(g.day >= lo) & (g.day <= hi)]
        return w.groupby("household_key").sales_value.sum()
    out = pd.DataFrame(index=hh)
    out["c_fwd_1_28"] = wsum(s+1, s+28)
    out["c_bwd_27_0"] = wsum(s-27, s)
    return out.fillna(0.0)

feats = agent_api.build_features(fn)
tt = agent_api.train_targets()
m = tt.merge(feats, on=["household_key","snapshot_day"])
for c in ["c_fwd_1_28","c_bwd_27_0"]:
    print(c, "NaNs:", m[c].isna().sum(), "unique:", m[c].nunique())
d = (m.c_fwd_1_28 - m.future_spend_4w).abs()
print("fwd_1_28: |diff|<0.01 frac:", (d<0.01).mean(), " max diff:", d.max())
print(m[["future_spend_4w","c_fwd_1_28","c_bwd_27_0"]].head(8).round(2))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

t = agent_api.load_saved("e006_zero_inflation.parquet")
tt = agent_api.train_targets()
m = tt.merge(t, on=["household_key","snapshot_day"])
feat = [c for c in t.columns if c not in ("household_key","snapshot_day")]
print(m[feat].dtypes.value_counts())
str_cols = [c for c in feat if m[c].dtype == object]
print("string cols:", str_cols)
if str_cols:
    for c in str_cols:
        print(c, m[c].unique()[:8])

# naive baselines on last train snapshot as pseudo-val
val = m[m.snapshot_day==431]
spend28 = val["spend_28"].values; y = val.future_spend_4w.values
print("\n--- pseudo-val snapshot 431, n=", len(val))
print("mean-target MAE:", round(np.abs(y - m[m.snapshot_day<431].future_spend_4w.mean()),2))
print("predict spend_28 MAE:", round(np.abs(y-spend28).mean(),2))
for k in [0.7,0.8,0.9,1.0]:
    print(f"predict {k}*spend_28 MAE:", round(np.abs(y-k*spend28).mean(),2))
# blend with spend_56
s56 = val["spend_56"].values
best=None
for a in np.linspace(0,1,21):
    p = a*spend28 + (1-a)*(s56-spend28)  # spend_56 minus spend_28 = previous block? approx
    mae = np.abs(y-p).mean()
    if best is None or mae<best[1]: best=(a,mae)
print("best blend spend28 vs (spend56-spend28):", best)


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

t = agent_api.load_saved("e006_zero_inflation.parquet")
tt = agent_api.train_targets()
m = tt.merge(t, on=["household_key","snapshot_day"])
feat = [c for c in t.columns if c not in ("household_key","snapshot_day")]

val = m[m.snapshot_day==431]
y = val.future_spend_4w.values
spend28 = val["spend_28"].values; s56 = val["spend_56"].values
print("--- pseudo-val snapshot 431, n=", len(val))
print("mean-target MAE:", round(float(np.abs(y - m[m.snapshot_day<431].future_spend_4w.mean())),2))
print("predict spend_28 MAE:", round(float(np.abs(y-spend28).mean()),2))
for k in [0.7,0.8,0.9,1.0,1.1]:
    print(f"predict {k}*spend_28 MAE:", round(float(np.abs(y-k*spend28).mean()),2))
best=None
for a in np.linspace(0,1,21):
    p = a*spend28 + (1-a)*(s56-spend28)
    mae = float(np.abs(y-p).mean())
    if best is None or mae<best[1]: best=(round(float(a),2),round(mae,2))
print("best blend spend28 vs (spend56-spend28):", best)

# distribution of target vs spend_28
print("\ntarget mean", round(float(y.mean()),1), "spend28 mean", round(float(spend28.mean()),1))
print("corr(target, spend28):", round(float(np.corrcoef(y,spend28)[0,1]),3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

t = agent_api.load_saved("e006_zero_inflation.parquet")
tt = agent_api.train_targets()
m = tt.merge(t, on=["household_key","snapshot_day"])
val = m[m.snapshot_day==431]
y = val.future_spend_4w.values
print("y shape", y.shape, "val shape", val.shape)
mu = m[m.snapshot_day<431].future_spend_4w.mean()
print("mu:", mu, type(mu))
d = np.abs(y - mu)
print("d shape", d.shape, "mae", d.mean())
print("MAE mean-target:", round(d.mean(),2))
spend28 = val["spend_28"].values
print("MAE spend28:", round(np.abs(y-spend28).mean(),2))
for k in [0.7,0.8,0.9,1.0,1.1]:
    print(f"MAE {k}*spend28:", round(np.abs(y-k*spend28).mean(),2))
s56 = val["spend_56"].values
res=[]
for a in np.linspace(0,1,21):
    p = a*spend28 + (1-a)*(s56-spend28)
    res.append((round(float(a),2), round(float(np.abs(y-p).mean()),2)))
print("blend sweep:", res)
print("corr(y, spend28):", round(float(np.corrcoef(y,spend28)[0,1]),3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

t = agent_api.load_saved("e006_zero_inflation.parquet")
tt = agent_api.train_targets()
m = tt.merge(t, on=["household_key","snapshot_day"])
feat = [c for c in t.columns if c not in ("household_key","snapshot_day")]
na = m[feat].isna().mean().sort_values(ascending=False)
print(na[na>0])


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

t = agent_api.load_saved("e006_zero_inflation.parquet")
tt = agent_api.train_targets()
m = tt.merge(t, on=["household_key","snapshot_day"])
val = m[m.snapshot_day==431]
y = val.future_spend_4w.values
s28 = val["spend_28"].fillna(0).values
s56 = val["spend_56"].fillna(0).values
s84 = val["spend_84"].fillna(0).values
print("MAE mean-target:", round(float(np.abs(y - m[m.snapshot_day<431].future_spend_4w.mean())),2))
print("MAE spend28:", round(float(np.abs(y-s28).mean()),2))
for k in [0.7,0.8,0.9,1.0,1.1]:
    print(f"MAE {k}*spend28:", round(float(np.abs(y-k*s28).mean()),2))
res=[]
for a in np.linspace(0,1,11):
    p = a*s28 + (1-a)*(s56-s28)
    res.append((round(float(a),2), round(float(np.abs(y-p).mean()),2)))
print("blend sweep (spend28 vs spend56-spend28):", res)

# how different is spend_28 from lag1 (trailing 28 days ending s-1)?
def fn(view, s):
    tx = view.table("transactions")
    hh = view.households
    tx = tx[tx.household_key.isin(hh)]
    def wsum(lo,hi):
        w = tx[(tx.day>=lo)&(tx.day<=hi)]
        return w.groupby("household_key").sales_value.sum()
    out = pd.DataFrame(index=hh)
    out["lag1"] = wsum(s-28, s-1)
    out["lag2"] = wsum(s-56, s-29)
    return out.reindex(hh).fillna(0.0)
feats = agent_api.build_features(fn)
m2 = tt.merge(feats, on=["household_key","snapshot_day"])
v2 = m2[m2.snapshot_day==431]
print("\nMAE lag1:", round(float(np.abs(v2.future_spend_4w.values - v2.lag1.values).mean()),2))
print("corr(lag1, spend_28):", round(float(np.corrcoef(v2.lag1, v2.spend_28.fillna(0))[0,1]),4))
print("mean |lag1-spend28|:", round(float((v2.lag1-v2.spend_28.fillna(0)).abs().mean()),2))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

t = agent_api.load_saved("e006_zero_inflation.parquet")
tt = agent_api.train_targets()
m = tt.merge(t, on=["household_key","snapshot_day"])
print("m shape:", m.shape, "cols sample:", m.columns[:6].tolist())
mu_obj = m[m.snapshot_day<431].future_spend_4w.mean()
print("mu type:", type(mu_obj), "shape:", np.shape(mu_obj))
print("snapshot_day dtype:", m.snapshot_day.dtype, "unique:", sorted(m.snapshot_day.unique())[:20])


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

t = agent_api.load_saved("e006_zero_inflation.parquet")
tt = agent_api.train_targets()
m = tt.merge(t, on=["household_key","snapshot_day"])

# pseudo-val: fit ridge on snapshots <=403, eval on 431
feat = [c for c in t.columns if c not in ("household_key","snapshot_day")]
X = pd.get_dummies(m[feat].fillna(-1), columns=["class2"], dummy_na=True)
X = X.astype(float)
mu, sd = X.mean(), X.std().replace(0,1)
Xz = (X-mu)/sd
tr_mask = m.snapshot_day <= 403
va_mask = m.snapshot_day == 431
Xtr, ytr = Xz[tr_mask].values, m.future_spend_4w[tr_mask].values
Xva, yva = Xz[va_mask].values, m.future_spend_4w[va_mask].values
I = np.eye(Xz.shape[1])
for lam in [1e2, 1e3, 1e4]:
    w = np.linalg.solve(Xtr.T@Xtr + lam*I, Xtr.T@ytr)
    p = Xva@w
    mae = float(np.abs(yva-p).mean())
    r2 = float(1 - ((yva-p)**2).sum()/((yva-yva.mean())**2).sum())
    print(f"ridge lam={lam}: pseudo-val MAE {mae:.2f} R2 {r2:.3f}")

# simple heuristics on 431
val = m[va_mask]
y = val.future_spend_4w.values
s28 = val["spend_28"].fillna(0).values
s56 = val["spend_56"].fillna(0).values
s84 = val["spend_84"].fillna(0).values
print("MAE spend28:", round(float(np.abs(y-s28).mean()),2))
for k in [0.8,0.9,1.0]:
    print(f"MAE {k}*spend28:", round(float(np.abs(y-k*s28).mean()),2))
best = min(((round(float(a),2), round(float(np.abs(y-(a*s28+(1-a)*(s56-s28))).mean()),2)) for a in np.linspace(0,1,11)), key=lambda z:z[1])
print("best blend spend28 vs (spend56-spend28):", best)
# residual analysis of ridge best model
lam=1e3
w = np.linalg.solve(Xtr.T@Xtr + lam*I, Xtr.T@ytr)
p = Xva@w
res = yva - p
print("\nresid mean:", round(float(res.mean()),2), "MAE:", round(float(np.abs(res).mean()),2))
# error by target bucket
for lo,hi in [(0,1),(1,50),(50,150),(150,300),(300,10**9)]:
    msk = (yva>=lo)&(yva<hi)
    if msk.sum()>0:
        print(f"y in [{lo},{hi}): n={msk.sum()}, MAE={float(np.abs(res[msk]).mean()):.1f}, mean_pred={float(p[msk].mean()):.1f}, mean_y={float(yva[msk].mean()):.1f}")


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def fn(view, s):
    tx = view.table("transactions")
    hh = view.households
    tx = tx[tx.household_key.isin(hh)]
    g = tx.groupby(["household_key","day"], as_index=False).sales_value.sum()
    def wsum(lo,hi):
        w = g[(g.day>=lo)&(g.day<=hi)]
        return w.groupby("household_key").sales_value.sum()
    # aligned 28-day blocks: lag1 = [s-27, s] == target at snapshot s-28
    df = pd.DataFrame({
        "lag1": wsum(s-27, s),
        "lag2": wsum(s-55, s-28),
        "lag3": wsum(s-83, s-56),
        "lag4": wsum(s-111, s-84),
        "ly_lag": wsum(s-391, s-364),
    }).reindex(hh).fillna(0.0)
    df["lag_trend"] = df.lag1 - df.lag2
    df["lag_ratio"] = df.lag1/(df.lag2+1.0)
    df["lag_mean4"] = df[["lag1","lag2","lag3","lag4"]].mean(axis=1)
    df["lag_std4"] = df[["lag1","lag2","lag3","lag4"]].std(axis=1).fillna(0.0)
    df["ly_ratio"] = df.lag1/(df.ly_lag+1.0)
    return df

feats = agent_api.build_features(fn)
print("feats:", feats.shape)
base = agent_api.load_saved("e006_zero_inflation.parquet")
print("base:", base.shape)
m = base.merge(feats, on=["household_key","snapshot_day"], how="left")
print("merged:", m.shape, "dup rows:", int(m.duplicated(["household_key","snapshot_day"]).sum()),
      "NaN lag1:", int(m.lag1.isna().sum()))
# quick sanity on last train snapshot
tt = agent_api.train_targets()
chk = tt[tt.snapshot_day==431].merge(m[["household_key","snapshot_day","lag1","lag2"]],
                                     on=["household_key","snapshot_day"])
print("corr(lag1,target@431):", round(float(np.corrcoef(chk.lag1, chk.future_spend_4w)[0,1]),3))
print("corr(lag2,target@431):", round(float(np.corrcoef(chk.lag2, chk.future_spend_4w)[0,1]),3))
path = agent_api.save_table(m, "e007_ar_lags.parquet")
print(path)
