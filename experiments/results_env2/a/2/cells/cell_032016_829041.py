import agent_api as A
import pandas as pd, numpy as np, itertools

oof = A.load_saved("oof_e016_cv.parquet").drop(columns=["y"])
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
oof = oof.merge(tt, on=["household_key","snapshot_day"], how="left")
tw = A.load_saved("oof_tw.parquet")
oof = oof.merge(tw, on=["household_key","snapshot_day"], how="left")
days = sorted(oof.snapshot_day.unique())
L = ["med_v3","hgbq_v3","hgbq_all","oof_tw"]  # drop med_all (worst, corr .997 w med_v3)
X = oof[L].values; y = oof.y.values

def loso_eval(fit_predict):
    """fit_predict(Xtr,ytr) -> predict(Xval); returns mean MAE over LOSO folds + per-fold"""
    maes=[]
    for d in days:
        tr = oof.snapshot_day.values != d; va = ~tr
        p = fit_predict(X[tr], y[tr], X[va])
        maes.append(np.mean(np.abs(p - y[va])))
    return np.mean(maes), maes

# 1) fixed E016 weights as reference (fit=identity)
def fp_fixed(w):
    def f(Xtr,ytr,Xva): return Xva @ np.array(w)
    return f
w0 = [0.4/0.65, 0.25/0.65, 0.35/0.65, 0.0]
m0,_ = loso_eval(fp_fixed(w0)); print("E016-style fixed weights LOSO MAE:", round(m0,4))

# 2) LOSO grid over weights incl oof_tw
def grid_search(step=0.1, allow_neg=False):
    best=(1e9,None)
    ws = np.arange(0,1.0001,step)
    for w1 in ws:
        for w2 in ws:
            for w3 in ws:
                w4 = 1-w1-w2-w3
                if w4 < -1e-9: continue
                for w5 in ([0.0] if False else [0.0,0.05,0.1,0.15,0.2]):
                    s = w1+w2+w3+w5
                    if s > 1.0001: continue
                    w = np.array([w1,w2,w3,w4*(1-w5) if s<=1 else 0, w5])
                    if abs(w.sum()-1)>1e-6: continue
                    mm,_ = loso_eval(fp_fixed(w))
                    if mm < best[0]: best=(mm,w)
    return best
# too slow with loso inside; use vectorized eval instead
def mae_w(w):
    p = X @ w
    return np.mean(np.abs(p-y))
best=(1e9,None)
for w1 in np.arange(0,1.001,0.1):
    for w2 in np.arange(0,1.001,0.1):
        for w5 in [0,0.05,0.1,0.15,0.2,0.3]:
            for w3 in np.arange(0,1.001,0.1):
                w4 = 1-w1-w2-w3-w5
                if w4 < -1e-9: continue
                w = np.array([w1,w2,w3,w4,w5])
                mm = mae_w(w)
                if mm < best[0]: best=(mm.round(4), w.round(2))
print("full-OOF best 5-learner weights:", best)

# 3) LOSO with per-fold grid on 5 learners (coarse, vectorized per fold)
def fp_grid(Xtr,ytr,Xva):
    best=(1e9,None)
    for w1 in np.arange(0,1.001,0.1):
        for w2 in np.arange(0,1.001,0.1):
            for w5 in [0,0.05,0.1,0.15,0.2]:
                for w3 in np.arange(0,1.001,0.1):
                    w4=1-w1-w2-w3-w5
                    if w4<-1e-9: continue
                    w=np.array([w1,w2,w3,w4,w5])
                    mm=np.mean(np.abs(Xtr@w-ytr))
                    if mm<best[0]: best=(mm,w)
    return Xva@best[1]
m3,f3 = loso_eval(fp_grid); print("LOSO per-fold grid (5 learners):", round(m3,4), [round(x,2) for x in f3])
