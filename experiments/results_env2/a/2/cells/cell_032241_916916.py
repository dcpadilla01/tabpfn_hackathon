import agent_api as A
import pandas as pd, numpy as np
from scipy.optimize import nnls

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
hh = oof.household_key.values

# household past realized targets (fully observed before day d): mean over tt rows with snap+28<=d
pt_mean = np.full(len(oof), np.nan)
tt_g = tt.groupby("household_key").apply(lambda g: g[["snapshot_day","y"]].values, include_groups=False)
for i in range(len(oof)):
    d = D[i]
    arr = tt_g.get(hh[i])
    if arr is None: continue
    m = arr[:,0] + 28 <= d
    if m.any(): pt_mean[i] = arr[m,1].mean()

# lagged OOF blend (same household, previous snapshot)
oof_s = oof.sort_values(["household_key","snapshot_day"])
oof_s["blend"] = oof_s[L].values @ W0
oof_s["lag_blend"] = oof_s.groupby("household_key")["blend"].shift(1)
oof_s["lag_y"] = oof_s.groupby("household_key")["y"].shift(1)
lag_blend = oof_s["lag_blend"].reindex(oof.index).values if oof.index.equals(oof_s.index) else oof_s["lag_blend"].values
# reindex safely by position after sorting back
oof_s = oof_s.sort_index()
lag_blend = oof_s["lag_blend"].values
lag_y = oof_s["lag_y"].values

def loso(fit_predict):
    maes=[]
    for d in days:
        tr = D != d; va = ~tr
        maes.append(np.mean(np.abs(fit_predict(tr, va) - y[va])))
    return round(np.mean(maes),4)

# N) blend with household past-target mean
for a in [0.05,0.1,0.2,0.3]:
    f = lambda tr,va,a=a: (1-a)*blend[va] + a*np.nan_to_num(pt_mean[va], nan=blend[va])
    print(f"N a={a} LOSO:", loso(f))

# O) blend with lagged OOF blend
for a in [0.05,0.1,0.2]:
    f = lambda tr,va,a=a: (1-a)*blend[va] + a*np.nan_to_num(lag_blend[va], nan=blend[va])
    print(f"O lagOOF a={a} LOSO:", loso(f))

# P) blend with lagged realized target
for a in [0.05,0.1,0.2]:
    f = lambda tr,va,a=a: (1-a)*blend[va] + a*np.nan_to_num(lag_y[va], nan=blend[va])
    print(f"P lagY a={a} LOSO:", loso(f))

# Q) NNLS weights LOSO
def fp_nnls(tr,va):
    w,_ = nnls(X[tr], y[tr])
    if w.sum()==0: w=W0
    else: w = w/w.sum()
    return X[va]@w
print("Q nnls LOSO:", loso(fp_nnls))

# R) segment weights: low/high blend halves
def fp_seg(tr,va):
    b_tr = blend[tr]
    med = np.median(b_tr)
    for seg in [0,1]:
        m = (b_tr<med) if seg==0 else (b_tr>=med)
        best=(1e9,W0)
        for w1 in np.arange(0,1.001,0.1):
            for w2 in np.arange(0,1.001,0.1):
                w4=1-w1-w2
                if w4<-1e-9: continue
                w=np.array([w1,w2,0.0,w4])
                mm=np.mean(np.abs(X[tr][m]@w-y[tr][m]))
                if mm<best[0]: best=(mm,w)
        if seg==0: wlo=best[1]
    b_va = blend[va]
    return np.where(b_va<med, X[va]@wlo, X[va]@best[1])
print("R segment weights LOSO:", loso(fp_seg))

# S) winsorize blend at caps
for cap in [600,800,1000,1200,1500]:
    f = lambda tr,va,cap=cap: np.minimum(blend[va], cap)
    print(f"S cap={cap} LOSO:", loso(f))
