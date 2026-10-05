import agent_api as A
import pandas as pd, numpy as np, itertools

oof = A.load_saved("oof_e016_cv.parquet").drop(columns=["y"])
tt = A.train_targets().rename(columns={"future_spend_4w":"y"})
oof = oof.merge(tt, on=["household_key","snapshot_day"], how="left")
tw = A.load_saved("oof_tw.parquet")
oof = oof.merge(tw, on=["household_key","snapshot_day"], how="left")
days = sorted(oof.snapshot_day.unique())
L = ["med_v3","hgbq_v3","hgbq_all","oof_tw"]
X = oof[L].values; y = oof.y.values

def loso_eval(fit_predict):
    maes=[]
    for d in days:
        tr = oof.snapshot_day.values != d; va = ~tr
        p = fit_predict(X[tr], y[tr], X[va])
        maes.append(np.mean(np.abs(p - y[va])))
    return np.mean(maes), maes

# reference: E016 fixed weights [med_v3 .4, hgbq_v3 .3, hgbq_all .3, tw 0]
m0,_ = loso_eval(lambda a,b,c: c @ np.array([0.4,0.3,0.3,0.0]))
print("E016 weights LOSO MAE:", round(m0,4))
print("full-OOF MAE E016 weights:", round(np.mean(np.abs(X@np.array([0.4,0.3,0.3,0.0])-y)),4))

# full-OOF grid over 5... actually 4 learners incl tw
best=(1e9,None)
for w1 in np.arange(0,1.001,0.05):
    for w2 in np.arange(0,1.001,0.05):
        for w4 in [0,0.05,0.1,0.15,0.2,0.25,0.3]:
            w3 = 1-w1-w2-w4
            if w3 < -1e-9: continue
            w = np.array([w1,w2,w3,w4])
            mm = np.mean(np.abs(X@w-y))
            if mm<best[0]: best=(round(mm,4), w.round(2))
print("full-OOF best 4-learner weights:", best)

# LOSO per-fold grid
def fp_grid(Xtr,ytr,Xva):
    best=(1e9,None)
    for w1 in np.arange(0,1.001,0.05):
        for w2 in np.arange(0,1.001,0.05):
            for w4 in [0,0.05,0.1,0.15,0.2,0.25,0.3]:
                w3=1-w1-w2-w4
                if w3<-1e-9: continue
                w=np.array([w1,w2,w3,w4])
                mm=np.mean(np.abs(Xtr@w-ytr))
                if mm<best[0]: best=(mm,w)
    return Xva@best[1]
m3,f3 = loso_eval(fp_grid); print("LOSO per-fold grid:", round(m3,4), [round(x,2) for x in f3])
