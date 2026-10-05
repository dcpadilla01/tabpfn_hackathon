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
