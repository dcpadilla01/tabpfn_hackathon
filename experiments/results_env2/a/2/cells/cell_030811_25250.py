import agent_api as A, pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet").merge(A.load_saved("oof_tw.parquet"), on=["household_key","snapshot_day"], how="left")
oh = A.load_saved("oof_harness.parquet")
op = A.load_saved("oof_pt.parquet").rename(columns={"oof":"oof_pt"})
oe = A.load_saved("oof_e008.parquet")
m2 = oof.merge(oh, on=["household_key","snapshot_day"], how="left").merge(op, on=["household_key","snapshot_day"], how="left").merge(oe[["household_key","snapshot_day","oof_log"]], on=["household_key","snapshot_day"], how="left")
L = ["med_v3","med_all","hgbq_v3","hgbq_all","oof_tw","oof_med","oof_sq","oof_log","oof_med_w112","oof_med_w224","oof_pt"]
mm = m2.dropna(subset=L)
y = mm["y"].values; P = mm[L].values
print("rows:", len(mm), "snapshots:", sorted(mm.snapshot_day.unique()))
for k,l in enumerate(L): print(f"{l:12s} MAE {np.abs(y-P[:,k]).mean():.4f}")
rng = np.random.default_rng(3); bm, bw = 1e9, None
K = len(L)
for it in range(40):
    W = rng.dirichlet(np.ones(K)*0.5, size=4000)
    maes = np.abs(y[:,None] - (P @ W.T)).mean(axis=0)
    b = maes.argmin()
    if maes[b] < bm: bm, bw = maes[b], W[b].copy()
def mae_w(w): return np.abs(y - P @ w).mean()
bw = bw.copy(); bm = mae_w(bw)
for step in [0.1,0.05,0.02,0.01,0.005]:
    improved=True
    while improved:
        improved=False
        for i in range(K):
            for j in range(K):
                if i==j or bw[j]<step: continue
                w2=bw.copy(); w2[i]+=step; w2[j]-=step
                if mae_w(w2)<bm: bw,bm=w2,mae_w(w2); improved=True
print("11-learner optimal:", dict(zip(L, np.round(bw,3))), "OOF MAE", round(bm,4))
# sanity: what would E016's assumed weights give on these same rows?
w0 = np.array([0.4,0.3,0.3,0.0,0,0,0,0,0,0,0])
print("E016-assumed weights on same rows:", round(mae_w(w0),4))
