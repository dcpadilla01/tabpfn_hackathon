import pandas as pd, numpy as np, agent_api
print("days:", agent_api.snapshot_days())
tt = agent_api.train_targets()
print("targets shape:", tt.shape)
print(tt.future_spend_4w.describe())
e6 = agent_api.load_saved("e006_newblock.parquet")
print("e6 shape:", e6.shape)
print("e6 cols:", sorted(e6.columns.tolist()))
v = agent_api.snapshot()
print("hh type:", type(v.households))
print("day:", v.day, "week:", v.week)
tx = v.table("transactions")
print("tx shape:", tx.shape, "max day:", tx.day.max())
print(tx.head(3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api
for name in ["e004_marketing_demo","e005_trend_season","e006_zero_inflation","e007_ar_lags","e008_log_transform","e009_full","e010_store_mix"]:
    t = agent_api.load_saved(name+".parquet")
    print(name, t.shape)
print()
print("e006_zero cols:", sorted(agent_api.load_saved("e006_zero_inflation.parquet").columns.tolist()))
print()
print("e005 cols:", sorted(agent_api.load_saved("e005_trend_season.parquet").columns.tolist()))


# ---- cell ----
import pandas as pd, numpy as np, agent_api
v = agent_api.snapshot()
tx = v.table("transactions")
print(tx[["quantity","sales_value","coupon_disc","coupon_match_disc","retail_disc","trans_time"]].describe().T)
print()
print("coupons:", v.table("coupons").shape, v.table("coupons").head(3).to_string())
print()
ct = v.table("campaign_targets")
print("campaign_targets:", ct.shape)
print(ct.description.value_counts())
print("n campaigns:", ct.campaign.nunique(), "hh targeted:", ct.household_key.nunique())
cr = v.table("coupon_redemptions")
print("redemptions:", cr.shape, "hh:", cr.household_key.nunique())


# ---- cell ----
import pandas as pd, numpy as np, agent_api
tt = agent_api.train_targets()
feats = agent_api.load_saved("e010_store_mix.parquet")
df = feats.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("merged:", df.shape)
ycol="future_spend_4w"
num = [c for c in df.columns if c not in ("household_key","snapshot_day",ycol) and pd.api.types.is_numeric_dtype(df[c])]
cor = df[num].corrwith(df[ycol]).sort_values()
print("TOP +corr:"); print(cor.tail(15).round(3).to_string())
print("TOP -corr:"); print(cor.head(8).round(3).to_string())
# target autocorrelation across snapshots (train only)
tt2 = tt.sort_values(["household_key","snapshot_day"])
tt2["lag_t"] = tt2.groupby("household_key")[ycol].shift(1)
print("\ncorr(target, target_lag1):", tt2[[ycol,"lag_t"]].corr().iloc[0,1].round(3))
# zero share
print("zero share:", (tt[ycol]==0).mean().round(3), " share<10:", (tt[ycol]<10).mean().round(3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
tt = agent_api.train_targets()
feats = agent_api.load_saved("e010_store_mix.parquet")
df = feats.merge(tt, on=["household_key","snapshot_day"], how="inner")
ycol="future_spend_4w"
cat = [c for c in df.columns if df[c].dtype.name in ("object","category","bool")]
num = [c for c in df.columns if c not in ("household_key","snapshot_day",ycol) and c not in cat]
X = pd.concat([df[num], pd.get_dummies(df[cat].astype(str), dummy_na=True)], axis=1).astype(float)
X = X.fillna(X.median()).clip(-1e6,1e6)
tr = df.snapshot_day<=431; va = df.snapshot_day>=459
def ridge_fit(Xtr,ytr,lam):
    mu,sd = Xtr.mean(), Xtr.std().replace(0,1)
    Z=(Xtr-mu)/sd; Z=np.c_[np.ones(len(Z)),Z.values]
    A=Z.T@Z+lam*np.eye(Z.shape[1]); A[0,0]-=lam
    w=np.linalg.solve(A,Z.T@ytr)
    return mu,sd,w
def ridge_pred(Xte,mu,sd,w):
    Z=((Xte-mu)/sd); Z=np.c_[np.ones(len(Z)),Z.values]
    return Z@w
for lam in [30,100,300,1000]:
    mu,sd,w = ridge_fit(X[tr], df.loc[tr,ycol].values, lam)
    p = np.clip(ridge_pred(X[va],mu,sd,w),0,None)
    print(f"lam={lam} raw-y MAE={np.abs(p-df.loc[va,ycol]).mean():.2f}")
# log target
mu,sd,w = ridge_fit(X[tr], np.log1p(df.loc[tr,ycol].values), 100)
p = np.clip(np.expm1(ridge_pred(X[va],mu,sd,w)),0,None)
print("log-y MAE:", np.abs(p-df.loc[va,ycol]).mean().round(2))
# sqrt target
mu,sd,w = ridge_fit(X[tr], np.sqrt(df.loc[tr,ycol].values), 100)
p = np.clip(ridge_pred(X[va],mu,sd,w)**2,0,None)
print("sqrt-y MAE:", np.abs(p-df.loc[va,ycol]).mean().round(2))
# baseline: predict lag1 target
m = df[df.snapshot_day>=459].merge(tt.rename(columns={"snapshot_day":"prev_day",ycol:"lag1"}),
        left_on=["household_key"], right_on=["household_key"])
m = m[m.prev_day==m.snapshot_day-28]
print("lag1-target MAE:", np.abs(m.lag1-m[ycol]).mean().round(2), "n:",len(m))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
tt = agent_api.train_targets()
feats = agent_api.load_saved("e010_store_mix.parquet")
df = feats.merge(tt, on=["household_key","snapshot_day"], how="inner")
ycol="future_spend_4w"
cat = [c for c in df.columns if df[c].dtype.name in ("object","category","bool")]
num = [c for c in df.columns if c not in ("household_key","snapshot_day",ycol) and c not in cat]
X = pd.concat([df[num], pd.get_dummies(df[cat].astype(str), dummy_na=True)], axis=1).astype(float)
X = X.replace([np.inf,-np.inf], np.nan).fillna(X.median())
X = X.loc[:, X.std()>0]
print("X:", X.shape)
tr = df.snapshot_day<=431; va = df.snapshot_day>=459
def fit(Xtr,ytr,lam):
    mu,sd = Xtr.mean(), Xtr.std().replace(0,1)
    Z=(Xtr-mu)/sd; Z=np.c_[np.ones(len(Z)),Z.values]
    A=Z.T@Z+lam*np.eye(Z.shape[1]); A[0,0]-=lam
    return mu,sd,np.linalg.solve(A,Z.T@ytr)
def pred(Xte,mu,sd,w):
    Z=((Xte-mu)/sd); return np.c_[np.ones(len(Z)),Z.values]@w
for lam in [30,100,300]:
    mu,sd,w = fit(X[tr], df.loc[tr,ycol].values, lam)
    p = np.clip(pred(X[va],mu,sd,w),0,None)
    print(f"lam={lam} raw MAE={np.abs(p-df.loc[va,ycol]).mean():.2f}")
mu,sd,w = fit(X[tr], np.log1p(df.loc[tr,ycol].values), 100)
p = np.clip(np.expm1(pred(X[va],mu,sd,w)),0,None)
print("log-y MAE:", round(float(np.abs(p-df.loc[va,ycol]).mean()),2))
mu,sd,w = fit(X[tr], np.sqrt(df.loc[tr,ycol].values), 100)
p = np.clip(pred(X[va],mu,sd,w),0,None)**2
print("sqrt-y MAE:", round(float(np.abs(p-df.loc[va,ycol]).mean()),2))
# per-snapshot error of sqrt model
mu,sd,w = fit(X[tr], np.sqrt(df.loc[tr,ycol].values), 100)
p = np.clip(pred(X[va],mu,sd,w),0,None)**2
tmp = pd.DataFrame({"d":df.loc[va,"snapshot_day"].values,"y":df.loc[va,ycol].values,"p":p})
print(tmp.groupby("d").apply(lambda g: pd.Series({"MAE":np.abs(g.p-g.y).mean(),"mean_y":g.y.mean(),"zero_share":(g.y==0).mean()}), include_groups=False).round(2).to_string())


# ---- cell ----
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
tt = agent_api.train_targets()
feats = agent_api.load_saved("e010_store_mix.parquet")
df = feats.merge(tt, on=["household_key","snapshot_day"], how="inner")
ycol="future_spend_4w"
cat = [c for c in df.columns if df[c].dtype.name in ("object","category","bool")]
num = [c for c in df.columns if c not in ("household_key","snapshot_day",ycol) and c not in cat]
X = pd.concat([df[num], pd.get_dummies(df[cat].astype(str), dummy_na=True)], axis=1).astype(float)
X = X.replace([np.inf,-np.inf], np.nan)
X = X.fillna(X.median())
X = X.loc[:, X.std()>0]
print("X:", X.shape, "nan left:", int(X.isna().sum().sum()))
tr = (df.snapshot_day<=431).values; va=(df.snapshot_day>=459).values
Xv = X.values
def fit(ytr, lam):
    Z = Xv[tr]
    mu = Z.mean(0); sd = Z.std(0); sd[sd==0]=1
    Z = (Z-mu)/sd
    Z = np.c_[np.ones(len(Z)), Z]
    A = Z.T@Z + lam*np.eye(Z.shape[1]); A[0,0] -= lam
    return mu, sd, np.linalg.solve(A, Z.T@ytr)
def pr(mu, sd, w):
    Z = (Xv[va]-mu)/sd
    Z = np.c_[np.ones(len(Z)), Z]
    return Z@w
y = df[ycol].values
for lam in [30,100,300]:
    mu,sd,w = fit(y[tr], lam)
    p = np.clip(pr(mu,sd,w),0,None)
    print("lam",lam,"raw MAE", round(float(np.abs(p-y[va]).mean()),2))
mu,sd,w = fit(np.log1p(y[tr]), 100)
p = np.clip(np.expm1(pr(mu,sd,w)),0,None)
print("log MAE", round(float(np.abs(p-y[va]).mean()),2))
mu,sd,w = fit(np.sqrt(y[tr]), 100)
p = np.clip(pr(mu,sd,w),0,None)**2
print("sqrt MAE", round(float(np.abs(p-y[va]).mean()),2))
tmp = pd.DataFrame({"d":df.loc[va,"snapshot_day"].values,"y":y[va],"p":p})
g = tmp.groupby("d").apply(lambda x: pd.Series({"MAE":np.abs(x.p-x.y).mean(),"mean_y":x.y.mean(),"zero":(x.y==0).mean()}), include_groups=False)
print(g.round(2).to_string())


# ---- cell ----
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
tt = agent_api.train_targets()
feats = agent_api.load_saved("e010_store_mix.parquet")
print(feats.dtypes.value_counts())
print(feats.snapshot_day.unique()[:20])
df = feats.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("merged:", df.shape, df.snapshot_day.dtype)
print(df.snapshot_day.unique())


# ---- cell ----
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
tt = agent_api.train_targets()
feats = agent_api.load_saved("e010_store_mix.parquet")
df = feats.merge(tt, on=["household_key","snapshot_day"], how="inner")
ycol="future_spend_4w"
cat = [c for c in df.columns if df[c].dtype.name in ("object","category","bool")]
num = [c for c in df.columns if c not in ("household_key","snapshot_day",ycol) and c not in cat]
X = pd.concat([df[num], pd.get_dummies(df[cat].astype(str), dummy_na=True)], axis=1).astype(float)
X = X.replace([np.inf,-np.inf], np.nan).fillna(X.median())
X = X.loc[:, X.std()>0]
Xv = X.values
tr = (df.snapshot_day<=403).values; va=(df.snapshot_day==431).values
y = df[ycol].values
def fit(ytr, lam):
    Z = Xv[tr]; mu=Z.mean(0); sd=Z.std(0); sd[sd==0]=1
    Z = np.c_[np.ones(len(Z)), (Z-mu)/sd]
    A = Z.T@Z + lam*np.eye(Z.shape[1]); A[0,0]-=lam
    return mu, sd, np.linalg.solve(A, Z.T@ytr)
def pr(mu,sd,w):
    Z = np.c_[np.ones(va.sum()), (Xv[va]-mu)/sd]
    return Z@w
for lam in [30,100,300]:
    mu,sd,w = fit(y[tr], lam)
    p = np.clip(pr(mu,sd,w),0,None)
    print("lam",lam,"raw MAE", round(float(np.abs(p-y[va]).mean()),2))
mu,sd,w = fit(np.log1p(y[tr]), 100); p=np.clip(np.expm1(pr(mu,sd,w)),0,None)
print("log MAE", round(float(np.abs(p-y[va]).mean()),2))
mu,sd,w = fit(np.sqrt(y[tr]), 100); p=np.clip(pr(mu,sd,w),0,None)**2
print("sqrt MAE", round(float(np.abs(p-y[va]).mean()),2))
tmp = pd.DataFrame({"y":y[va],"p":p})
print("mean_y", tmp.y.mean().round(1), "mean_p", tmp.p.mean().round(1))
print("MAE by y-quantile:")
q = pd.qcut(tmp.y, 5, duplicates="drop")
print(tmp.groupby(q, observed=True).apply(lambda g: pd.Series({"MAE":np.abs(g.p-g.y).mean(),"n":len(g),"mean_y":g.y.mean()}), include_groups=False).round(1).to_string())


# ---- cell ----
import pandas as pd, numpy as np, agent_api, warnings, re
warnings.filterwarnings("ignore")
tt = agent_api.train_targets()
fe = agent_api.load_saved("e006_zero_inflation.parquet")
df = fe.merge(tt, on=["household_key","snapshot_day"], how="inner")
ycol="future_spend_4w"
cat = [c for c in df.columns if df[c].dtype.name in ("object","category","bool")]
num = [c for c in df.columns if c not in ("household_key","snapshot_day",ycol) and c not in cat]
X = pd.concat([df[num], pd.get_dummies(df[cat].astype(str), dummy_na=True)], axis=1).astype(float)
X = X.replace([np.inf,-np.inf], np.nan).fillna(X.median())
X = X.loc[:, X.std()>0]
y = df[ycol].values
tr = (df.snapshot_day<=403).values; va=(df.snapshot_day==431).values
Xtr=X.values[tr]; ytr=y[tr]; Xva=X.values[va]; yva=y[va]
mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
Ztr=np.c_[np.ones(tr.sum()),(Xtr-mu)/sd]
lam=300; A=Ztr.T@Ztr+lam*np.eye(Ztr.shape[1]); A[0,0]-=lam
w=np.linalg.solve(A,Ztr.T@ytr)
def mae(Zv):
    return float(np.abs(np.clip(np.c_[np.ones(len(Zv)),Zv]@w,0,None)-yva).mean())
Zva=(Xva-mu)/sd
print("BASE MAE(431):", round(mae(Zva),2))
print("median-baseline MAE:", round(float(np.abs(np.median(ytr)-yva).mean()),2))
i28=list(X.columns).index("spend_28")
print("spend_28-only MAE:", round(mae(Zva*np.where(np.arange(len(X.columns))==i28,1,0)) if False else 0,2))
# proper single-feature model for reference
Z1=np.c_[np.ones(tr.sum()),(Xtr[:,i28]-mu[i28])/sd[i28]]
A1=Z1.T@Z1+lam*np.eye(2); A1[0,0]-=lam
w1=np.linalg.solve(A1,ytr)
p1=np.clip(np.c_[np.ones(va.sum()),(Xva[:,i28]-mu[i28])/sd[i28]]@w1,0,None)
print("spend_28-only MAE:", round(float(np.abs(p1-yva).mean()),2))
# block ablation (zero standardized cols = set to train mean)
blocks={}
for c in X.columns:
    if c.startswith("dept_"): b="dept"
    elif c.startswith("z_"): b="zeroint"
    elif c.startswith("weekend"): b="basket"
    elif re.search("camp|redemp|coupon|n_campaigns",c): b="campaign"
    elif re.search("age_code|^class|homeowner|kids_code|size_code|has_demo",c): b="demo"
    elif c.startswith(("week","trend","ly_")): b="season"
    elif re.search("spend|dec_|avg_weekly|weekly_|^r_",c): b="spend"
    elif re.search("trip|gap|zero|active|recency|days_since",c): b="recency_zero"
    elif re.search("basket|lines|unit_price|qty|national|n_products|n_depts|disc|evening",c): b="basket"
    else: b="misc"
    blocks.setdefault(b,[]).append(c)
res=[]
for b,cols in sorted(blocks.items()):
    Zv2=Zva.copy()
    for c in cols: Zv2[:,X.columns.get_loc(c)]=0.0
    res.append((b,len(cols),round(mae(Zv2),2)))
for b,n,m in res: print(f"drop {b:14s} n={n:3d} MAE={m}")
print()
coefs=pd.Series(np.abs(w[1:]),index=X.columns).sort_values(ascending=False)
print("top-15 |coef|:", coefs.head(15).round(1).to_dict())


# ---- cell ----
import pandas as pd, numpy as np, agent_api, warnings, re
warnings.filterwarnings("ignore")
tt = agent_api.train_targets()
fe = agent_api.load_saved("e006_zero_inflation.parquet")
df = fe.merge(tt, on=["household_key","snapshot_day"], how="inner")
ycol="future_spend_4w"
cat = [c for c in df.columns if df[c].dtype.name in ("object","category","bool")]
num = [c for c in df.columns if c not in ("household_key","snapshot_day",ycol) and c not in cat]
X = pd.concat([df[num], pd.get_dummies(df[cat].astype(str), dummy_na=True)], axis=1).astype(float)
X = X.replace([np.inf,-np.inf], np.nan).fillna(X.median())
X = X.loc[:, X.std()>0]
y = df[ycol].values
tr = (df.snapshot_day<=403).values; va=(df.snapshot_day==431).values
Xtr=X.values[tr]; ytr=y[tr]; Xva=X.values[va]; yva=y[va]
mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
Ztr=np.c_[np.ones(tr.sum()),(Xtr-mu)/sd]
lam=300; A=Ztr.T@Ztr+lam*np.eye(Ztr.shape[1]); A[0,0]-=lam
w=np.linalg.solve(A,Ztr.T@ytr)
def mae(Zv):
    return float(np.abs(np.clip(np.c_[np.ones(len(Zv)),Zv]@w,0,None)-yva).mean())
Zva=(Xva-mu)/sd
print("BASE MAE(431):", round(mae(Zva),2))
i28=list(X.columns).index("spend_28")
Z1=np.c_[np.ones(tr.sum()),(Xtr[:,i28]-mu[i28])/sd[i28]]
A1=Z1.T@Z1+lam*np.eye(2); A1[0,0]-=lam
w1=np.linalg.solve(A1,Z1.T@ytr)
p1=np.clip(np.c_[np.ones(va.sum()),(Xva[:,i28]-mu[i28])/sd[i28]]@w1,0,None)
print("spend_28-only MAE:", round(float(np.abs(p1-yva).mean()),2))
blocks={}
for c in X.columns:
    if c.startswith("dept_"): b="dept"
    elif c.startswith("z_"): b="zeroint"
    elif re.search("camp|redemp|coupon|n_campaigns",c): b="campaign"
    elif re.search("age_code|^class|homeowner|kids_code|size_code|has_demo",c): b="demo"
    elif c.startswith(("week","trend","ly_")): b="season"
    elif re.search("spend|dec_|avg_weekly|weekly_|^r_",c): b="spend"
    elif re.search("trip|gap|zero|active|recency|days_since",c): b="recency_zero"
    else: b="basket"
    blocks.setdefault(b,[]).append(c)
for b,cols in sorted(blocks.items()):
    Zv2=Zva.copy()
    for c in cols: Zv2[:,X.columns.get_loc(c)]=0.0
    print(f"drop {b:14s} n={len(cols):3d} MAE={round(mae(Zv2),2)}")
coefs=pd.Series(np.abs(w[1:]),index=X.columns).sort_values(ascending=False)
print("top-15 |coef|:", {k:round(v,1) for k,v in coefs.head(15).items()})


# ---- cell ----
import pandas as pd, agent_api
fe = agent_api.load_saved("e006_zero_inflation.parquet")
drop = ["lines_84","lines_per_trip","avg_basket_84","avg_basket_84_ts","unit_price_84",
        "qty_28","qty_84","national_share_84","n_products_84","n_depts_84",
        "evening_share_84","weekend_share_84","disc_share_84"]
drop = [c for c in drop if c in fe.columns]
out = fe.drop(columns=drop)
print("kept:", out.shape[1]-2, "dropped:", drop)
p = agent_api.save_table(out, "e011_pruned_basket.parquet")
print(p)
