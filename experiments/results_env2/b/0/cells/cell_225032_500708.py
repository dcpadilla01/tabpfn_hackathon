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
