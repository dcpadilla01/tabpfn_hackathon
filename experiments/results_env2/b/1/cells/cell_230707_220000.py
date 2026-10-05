
import pandas as pd, numpy as np, agent_api
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
sp28 = df.sp28.values.astype(float); med4 = df.z_med4w_hist.values.astype(float)
dsl = df.days_since_last.values.astype(float); gapmed = df.gap_med.values.astype(float)
act = sp28 > 0
dorm = dsl/np.maximum(gapmed,1.0)
print("dormancy nan frac:", np.isnan(dorm).mean())
actnan = act & ~np.isnan(dorm)
print("dormancy: y==0 mean %.2f | y>0 mean %.2f" % (np.nanmean(dorm[actnan & (y[actnan]==0)]), np.nanmean(dorm[actnan & (y[actnan]>0)])))
# AUROC-like separation for y==0
from numpy import argsort
def auroc(v, mask):
    vv = v[mask]; yy = (y[mask]==0).astype(int)
    ok = ~np.isnan(vv); vv=vv[ok]; yy=yy[ok]
    pos = vv[yy==1]; neg = vv[yy==0]
    if len(pos)==0 or len(neg)==0: return np.nan
    # rank-based
    allv = np.concatenate([pos,neg]); r = allv.argsort().argsort()+1
    rpos = r[:len(pos)].sum()
    return (rpos - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))
print("AUC(y==0) days_since_last:", round(auroc(dsl, act),3), "| dormancy:", round(auroc(dorm, act),3))
print("AUC(y==0) sp28:", round(auroc(sp28, act),3), "| trend_84:", round(auroc(np.asarray(df.trend_84,float), act),3))

# quick ridge on interaction-augmented design (offline, leak-free)
def add_inter(Xv, df):
    sp28v = df.sp28.values.astype(float); med4v = df.z_med4w_hist.values.astype(float)
    dslv = df.days_since_last.values.astype(float)
    inter = np.column_stack([
        sp28v*med4v/100.0, sp28v*dslv/100.0, med4v*dslv/100.0,
        sp28v*np.asarray(df.trend_84,float), med4v*np.asarray(df.trend_84,float),
        sp28v*np.asarray(df.wksp_std,float)/100.0,
        np.log1p(sp28v)*np.log1p(med4v),
        sp28v*np.asarray(df.nact84,float)/10.0, med4v*np.asarray(df.nact84,float)/10.0,
    ])
    return np.hstack([Xv, inter])
X2 = add_inter(Xv, df)
idx = np.random.RandomState(0).permutation(n); K=5
def cv_mae(Xall, a=100):
    s=0.0
    for k in range(K):
        va=idx[k::K]; tr=np.setdiff1d(idx,va)
        mu=Xall[tr].mean(0); sdv=Xall[tr].std(0)+1e-9
        A=np.hstack([(Xall[tr]-mu)/sdv,np.ones((len(tr),1))])
        G=A.T@A; p=G.shape[0]
        Gg=G.copy(); Gg[np.arange(p),np.arange(p)]+=a*np.r_[np.ones(p-1),0]
        w=np.linalg.solve(Gg,A.T@y[tr])
        pr=np.hstack([(Xall[va]-mu)/sdv,np.ones((len(va),1))])@w
        s+=np.abs(y[va]-pr).sum()
    return round(s/n,2)
print("ridge base:", cv_mae(Xv), "| with interactions:", cv_mae(X2))

# market-level calibration check: is there a snapshot-level level shift the model misses?
oof=np.zeros(n)
for k in range(K):
    va=idx[k::K]; tr=np.setdiff1d(idx,va)
    mu=Xv[tr].mean(0); sdv=Xv[tr].std(0)+1e-9
    A=np.hstack([(Xv[tr]-mu)/sdv,np.ones((len(tr),1))])
    G=A.T@A; p=G.shape[0]
    Gg=G.copy(); Gg[np.arange(p),np.arange(p)]+=100*np.r_[np.ones(p-1),0]
    w=np.linalg.solve(Gg,A.T@y[tr])
    oof[va]=np.hstack([(Xv[va]-mu)/sdv,np.ones((len(va),1))])@w
res = y-oof
print("\nresidual mean by snapshot:", {int(s): round(res[df.snapshot_day==s].mean(),1) for s in sorted(df.snapshot_day.unique())})
print("resid mean overall:", round(res.mean(),2), "| resid std:", round(res.std(),1))
