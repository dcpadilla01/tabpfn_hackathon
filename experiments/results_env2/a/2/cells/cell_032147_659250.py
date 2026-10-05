import agent_api as A
import pandas as pd, numpy as np
from sklearn.isotonic import IsotonicRegression

oof = A.load_saved("oof_e016_cv.parquet").drop(columns=["y"])
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
oof = oof.merge(tt, on=["household_key","snapshot_day"], how="left")
tw = A.load_saved("oof_tw.parquet")
oof = oof.merge(tw, on=["household_key","snapshot_day"], how="left")
days = sorted(oof.snapshot_day.unique())
L = ["med_v3","hgbq_v3","hgbq_all","oof_tw"]
X = oof[L].values; y = oof.y.values; D = oof.snapshot_day.values
W0 = np.array([0.4,0.3,0.3,0.0])
blend = X @ W0

def loso(fit_predict):
    maes=[]
    for d in days:
        tr = D != d; va = ~tr
        maes.append(np.mean(np.abs(fit_predict(X[tr],y[tr],D[tr],X[va],D[va]) - y[va])))
    return round(np.mean(maes),4)

# C) isotonic calibration of blend (fit on train folds, apply to val fold)
def fp_iso(Xtr,ytr,Dtr,Xva,Dva):
    b = Xtr@W0
    iso = IsotonicRegression(out_of_bounds="clip").fit(b, ytr)
    return iso.predict(Xva@W0)
print("C isotonic on blend LOSO:", loso(fp_iso))

# J) linear recalibration a*blend+b
def fp_lin(Xtr,ytr,Dtr,Xva,Dva):
    b = Xtr@W0
    A_ = np.vstack([b, np.ones_like(b)]).T
    coef, *_ = np.linalg.lstsq(A_, ytr, rcond=None)
    return Xva@W0*coef[0]+coef[1]
print("J linear recalib LOSO:", loso(fp_lin))

# K) CDF matching: map blend ranks to train-target quantiles
def fp_cdf(Xtr,ytr,Dtr,Xva,Dva):
    b = Xtr@W0
    qs = np.quantile(b, np.linspace(0.01,0.99,99))
    qy = np.quantile(ytr, np.linspace(0.01,0.99,99))
    return np.interp(Xva@W0, qs, qy)
print("K CDF-match LOSO:", loso(fp_cdf))

# H) median of learners
print("H median-of-4 LOSO:", loso(lambda a,b,c,d,e: np.median(np.vstack([d[:,0],d[:,1],d[:,2],d[:,3]]),axis=0)))

# I) rank averaging -> map back with train y quantiles
def fp_rank(Xtr,ytr,Dtr,Xva,Dva):
    r = np.mean([ (pd.Series(Xva[:,i]).rank(pct=True).values) for i in range(4)],axis=0)
    qy = np.quantile(ytr, np.linspace(0,1,101))
    return np.interp(r, np.linspace(0,1,101), qy)
print("I rank-avg LOSO:", loso(fp_rank))

# G) weights fit on recent snapshots only (last 5 before val fold), fallback to W0
def fp_recent(Xtr,ytr,Dtr,Xva,Dva,K=5):
    recent = np.sort(np.unique(Dtr))[-K:]
    m = np.isin(Dtr, recent)
    best=(1e9,W0)
    for w1 in np.arange(0,1.001,0.1):
        for w2 in np.arange(0,1.001,0.1):
            w4 = 1-w1-w2
            if w4<-1e-9: continue
            w=np.array([w1,w2,0.0,w4])
            mm=np.mean(np.abs(Xtr[m]@w-ytr[m]))
            if mm<best[0]: best=(mm,w)
    return Xva@best[1]
for K in [3,5,8]:
    print(f"G recent-{K} LOSO:", loso(lambda a,b,c,d,e,K=K: fp_recent(a,b,c,d,e,K)))

# naive reference: last-28d spend (spend_all is what window? check) -- just report blend MAE per fold
print("reference W0 LOSO:", loso(lambda a,b,c,d,e: e@W0))
