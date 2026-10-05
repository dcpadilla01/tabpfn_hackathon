
import pandas as pd, numpy as np
base = load_saved("e008_level_shape.parquet")
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
y = df[TARGET].values.astype(float)
feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")
for i, c in enumerate(cat_cols):
    Xn = pd.concat([Xn, pd.get_dummies(X[c].astype("category"), prefix=f"cat{i}", dummy_na=True).astype(float)], axis=1)
Xn = Xn.fillna(Xn.median())
Xv = Xn.values.astype(np.float64); n = len(y)
sp28 = df["sp28"].values.astype(float)
med4 = df["z_med4w_hist"].values.astype(float)

idx = np.random.RandomState(0).permutation(n); K=5
def ridge_fit(Xtr,ytr,a=100):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-9
    A=np.hstack([(Xtr-mu)/sd,np.ones((len(Xtr),1))])
    G=A.T@A; p=G.shape[0]
    Gg=G.copy(); Gg[np.arange(p),np.arange(p)]+=a*np.r_[np.ones(p-1),0]
    return np.linalg.solve(Gg,A.T@ytr),mu,sd
def ridge_pred(tr,va,Xall):
    w,mu,sd=ridge_fit(Xall[tr],y[tr])
    return np.hstack([(Xall[va]-mu)/sd,np.ones((len(va),1))])@w
def cv_mae(pred_fn):
    s=0.0
    for k in range(K):
        va=idx[k::K]; tr=np.setdiff1d(idx,va)
        s+=np.abs(y[va]-pred_fn(tr,va)).sum()
    return round(s/n,2)

p0=np.zeros(n)
for k in range(K):
    va=idx[k::K]; tr=np.setdiff1d(idx,va); p0[va]=ridge_pred(tr,va,Xv)
print("oof ridge:", round(np.abs(y-p0).mean(),2))

bins=[0,0.01,25,50,100,200,400,1e9]
labs=np.asarray(pd.cut(sp28,bins,labels=False),dtype=float)
cal=np.zeros(n)
for k in range(K):
    va=idx[k::K]; tr=np.setdiff1d(idx,va)
    bm={b:y[tr][labs[tr]==b].mean() for b in np.unique(labs[tr])}
    gm=y[tr].mean()
    cal[va]=[bm.get(l,gm) for l in labs[va]]
print("cal alone:", round(np.abs(y-cal).mean(),2))
X2=np.hstack([Xv, cal[:,None]])
print("ridge + cal feature:", cv_mae(lambda tr,va: ridge_pred(tr,va,X2)))
for wgt in [0.3,0.5,0.7]:
    print(f"blend ridge+cal {wgt}:", round(np.abs(y-(wgt*p0+(1-wgt)*cal)).mean(),2))
for wgt in [0.3,0.4,0.5,0.6]:
    print(f"blend ridge+med4w {wgt}:", round(np.abs(y-(wgt*p0+(1-wgt)*med4)).mean(),2))
for a,b in [(0.45,0.2),(0.4,0.25),(0.5,0.15)]:
    print(f"ridge {a} med4w {b} cal {round(1-a-b,2)}:", round(np.abs(y-(a*p0+b*med4+(1-a-b)*cal)).mean(),2))
