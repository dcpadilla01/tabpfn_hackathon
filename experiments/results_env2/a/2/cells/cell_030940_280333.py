import agent_api as A, pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet").merge(A.load_saved("oof_tw.parquet"), on=["household_key","snapshot_day"], how="left")
oh = A.load_saved("oof_harness.parquet")
op = A.load_saved("oof_pt.parquet").rename(columns={"oof":"oof_pt"})
m = oof.merge(oh, on=["household_key","snapshot_day"], how="left").merge(op, on=["household_key","snapshot_day"], how="left")
L = ["med_v3","med_all","hgbq_v3","hgbq_all","oof_tw","oof_med","oof_sq","oof_med_w112","oof_med_w224","oof_pt"]
mm = m.dropna(subset=L)
print("rows:", len(mm), "snapshots:", sorted(mm.snapshot_day.unique()))
y = mm["y"].values; P = mm[L].values
K = len(L)
def mae_w(w): return np.abs(y - P @ w).mean()

rng = np.random.default_rng(5); bm, bw = 1e9, None
for it in range(60):
    W = rng.dirichlet(np.ones(K)*0.5, size=4000)
    maes = np.abs(y[:,None] - (P @ W.T)).mean(axis=0)
    b = maes.argmin()
    if maes[b] < bm: bm, bw = maes[b], W[b].copy()
def refine(y_, P_, w0, steps=(0.1,0.05,0.02,0.01,0.005)):
    w = w0.copy(); bm_ = np.abs(y_ - P_@w).mean()
    for step in steps:
        improved=True
        while improved:
            improved=False
            for i in range(len(w)):
                for j in range(len(w)):
                    if i==j or w[j]<step: continue
                    w2=w.copy(); w2[i]+=step; w2[j]-=step
                    m2=np.abs(y_-P_@w2).mean()
                    if m2<bm_: w,bm_=w2,m2; improved=True
    return w, bm_
wg, bmg = refine(y, P, bw)
print("FULL-data 10-learner optimal:", dict(zip(L, np.round(wg,3))), "OOF MAE", round(bmg,4))

# reference points on full data
for w,tag in [([0.4,0,0.3,0.3,0,0,0,0,0,0],"E016-assumed"), ([0.25]*4+[0,0,0,0,0,0],"equal4")]:
    print(tag, round(mae_w(np.array(w)),4))

# stability: fit on first-half vs second-half snapshots
d1 = mm[mm.snapshot_day <= 263]; d2 = mm[mm.snapshot_day > 263]
for tag, dd in [("early(151-263)",d1), ("late(291-431)",d2)]:
    yy, PP = dd["y"].values, dd[L].values
    rng2 = np.random.default_rng(9); bm2, bw2 = 1e9, None
    for it in range(30):
        W = rng2.dirichlet(np.ones(K)*0.5, size=3000)
        maes = np.abs(yy[:,None] - (PP @ W.T)).mean(axis=0)
        b = maes.argmin()
        if maes[b] < bm2: bm2, bw2 = maes[b], W[b].copy()
    w2_, m2_ = refine(yy, PP, bw2)
    print(tag, "opt:", dict(zip(L, np.round(w2_,2))), round(m2_,4))
    print("   full-data weights evaluated on", tag, ":", round(np.abs(yy - PP@wg).mean(),4))
