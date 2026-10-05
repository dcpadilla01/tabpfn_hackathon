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
