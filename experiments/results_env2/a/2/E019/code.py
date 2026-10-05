import agent_api as A, pandas as pd, numpy as np
names = ["feats_all_e016","feats_e014","feats_prof","feats_seasonal","feats_v3","oof_e008","oof_e016_cv","oof_harness","oof_pt","past_targets",
         "pred_e001","pred_e002","pred_e003","pred_e004","pred_e005","pred_e006","pred_e007","pred_e008","pred_e009","pred_e010","pred_e010_blend",
         "pred_e014","pred_e016","pred_e018","pred_log1p","pred_prof","pred_pt","pred_seasonal"]
for n in names:
    try:
        df = A.load_saved(n+".parquet")
        print(n, df.shape, df.columns.tolist()[:12])
    except Exception as e:
        print(n, "ERR", type(e).__name__, str(e)[:80])
print(A.snapshot_days())
tt = A.train_targets()
print("train_targets", tt.shape, tt.columns.tolist())
print(tt.groupby("snapshot_day").future_spend_4w.agg(["count","mean","median"]).round(2))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet")
print("oof_e016_cv", oof.shape, oof.columns.tolist())
print(oof.groupby("snapshot_day").size())

y = oof["y"].values
L = ["med_v3","med_all","hgbq_v3","hgbq_all"]
P = oof[L].values
print("\nindividual OOF MAE / corr-with-y / pairwise corr:")
for k,l in enumerate(L):
    print(f"{l:9s} MAE {np.abs(y-P[:,k]).mean():.4f}  corr {np.corrcoef(y,P[:,k])[0,1]:.4f}")
C = np.corrcoef(P.T)
print("pairwise corr matrix:\n", np.round(C,3))

def mae(w, y, P):
    return np.abs(y - P @ w).mean()

# E016-style hand weights guesses
for w in [[0.4,0.3,0.3,0.0],[0.4,0.0,0.3,0.3],[0.34,0.33,0.33,0.0],[0.25,0.25,0.25,0.25]]:
    print("w",w,"OOF MAE", round(mae(np.array(w),y,P),4))

# random Dirichlet search over simplex
rng = np.random.default_rng(0)
W = rng.dirichlet(np.ones(4)*0.5, size=300000)
maes = np.abs(y[:,None] - (P @ W.T)).mean(axis=0)
best = maes.argmin()
print("\nbest random w:", np.round(W[best],3), "OOF MAE", round(maes[best],4))

# local refinement around best
bw = W[best].copy()
for step in [0.2,0.1,0.05,0.02,0.01,0.005]:
    improved = True
    while improved:
        improved = False
        for i in range(4):
            for j in range(4):
                if i==j or bw[j]<step: continue
                w2 = bw.copy(); w2[i]+=step; w2[j]-=step
                if mae(w2,y,P) < mae(bw,y,P):
                    bw = w2; improved=True
print("refined w:", np.round(bw,4), "OOF MAE", round(mae(bw,y,P),4))

# also check other OOF tables for extra learners
oh = A.load_saved("oof_harness.parquet"); op = A.load_saved("oof_pt.parquet"); oe = A.load_saved("oof_e008.parquet")
m = oof.merge(oh, on=["household_key","snapshot_day"], how="left").merge(op, on=["household_key","snapshot_day"], how="left", suffixes=("","_pt"))
print("\nmerged harness:", m[["oof_med","oof_sq","oof_med_w112","oof_med_w224","oof_pt"]].isna().mean().round(3).to_dict())
L2 = L + ["oof_med","oof_sq","oof_med_w112","oof_med_w224","oof_pt"]
P2 = m[L2].values; ok = ~np.isnan(P2).any(axis=1)
y2, P2 = y[ok], P2[ok]
print("rows with all:", ok.sum(), "of", len(y))
for k,l in enumerate(L2):
    print(f"{l:12s} MAE {np.abs(y2-P2[:,k]).mean():.4f}")
W2 = rng.dirichlet(np.ones(len(L2))*0.5, size=300000)
maes2 = np.abs(y2[:,None] - (P2 @ W2.T)).mean(axis=0)
b2 = maes2.argmin()
print("best random (9 learners):", dict(zip(L2, np.round(W2[b2],3))), "OOF MAE", round(maes2[b2],4))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
try:
    oof = A.load_saved("oof_e016_cv.parquet")
    print("ok", oof.shape, oof.columns.tolist())
    print(oof.head(3).to_string())
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:200])


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet")
y = oof["y"].values
L = ["med_v3","med_all","hgbq_v3","hgbq_all"]
P = oof[L].values
print("individual OOF MAE / corr:")
for k,l in enumerate(L):
    print(f"{l:9s} MAE {np.abs(y-P[:,k]).mean():.4f}  corr {np.corrcoef(y,P[:,k])[0,1]:.4f}")
print("pairwise corr:\n", np.round(np.corrcoef(P.T),3))

def mae_w(w): return np.abs(y - P @ w).mean()
for w in [[0.4,0.3,0.3,0.0],[0.4,0.0,0.3,0.3],[0.34,0.33,0.33,0.0],[0.25,0.25,0.25,0.25],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]:
    print("w",w,"OOF MAE", round(mae_w(np.array(w)),4))

# chunked random Dirichlet search
rng = np.random.default_rng(0)
best_mae, best_w = 1e9, None
for it in range(20):
    W = rng.dirichlet(np.ones(4)*0.5, size=5000)          # 5000x4
    PW = P @ W.T                                          # 26437 x 5000  (~1e8 ok)
    maes = np.abs(y[:,None] - PW).mean(axis=0)
    b = maes.argmin()
    if maes[b] < best_mae: best_mae, best_w = maes[b], W[b]
print("best random w:", np.round(best_w,4), "OOF MAE", round(best_mae,4))

# local refinement (coordinate exchange on simplex)
bw = best_w.copy(); bm = mae_w(bw)
for step in [0.2,0.1,0.05,0.02,0.01,0.005,0.002]:
    improved = True
    while improved:
        improved = False
        for i in range(4):
            for j in range(4):
                if i==j or bw[j]<step: continue
                w2 = bw.copy(); w2[i]+=step; w2[j]-=step
                m2 = mae_w(w2)
                if m2 < bm: bw, bm = w2, m2; improved=True
print("refined w:", np.round(bw,4), "OOF MAE", round(bm,4))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet")
y = oof["y"].values
L = ["med_v3","med_all","hgbq_v3","hgbq_all"]
P = oof[L].values
def mae_w(w): return np.abs(y - P @ np.array(w)).mean()
# candidate E016 weight vectors
cands = {"0.4/0.3/0.3/0(med_v3,med_all,hgbq_v3)": [0.4,0.3,0.3,0.0],
         "0.4/0/0.3/0.3": [0.4,0.0,0.3,0.3],
         "0.4/0.3/0/0.3": [0.4,0.3,0.0,0.3],
         "0.4/0/0.3/0.3 alt": [0.4,0.0,0.3,0.3]}
for k,w in cands.items(): print(k, round(mae_w(w),4))

# check oof_e008 provenance
oe = A.load_saved("oof_e008.parquet")
print("\noof_e008 snapshot days:", sorted(oe.snapshot_day.unique()), "n rows", len(oe))

# feature tables: is feats_all_e016 == v3 + seasonal?
v3 = A.load_saved("feats_v3.parquet"); seas = A.load_saved("feats_seasonal.parquet"); allf = A.load_saved("feats_all_e016.parquet")
key = ["household_key","snapshot_day"]
m = v3.merge(seas, on=key, how="outer", suffixes=("_v3","_s"))
print("\nv3 cols:", len(v3.columns)-2, "seasonal cols:", len(seas.columns)-2, "merged:", m.shape)
common = [c for c in allf.columns if c in m.columns]
print("all_e016 cols:", len(allf.columns)-2)
print("cols in all_e016 not in v3+seasonal:", [c for c in allf.columns if c not in v3.columns and c not in seas.columns and c not in key])
# verify equality on a few cols
chk = allf.merge(m, on=key, suffixes=("_a","_m"))
for c in ["spend_all","lag364_spend"]:
    if c+"_a" in chk.columns and c+"_m" in chk.columns:
        print(c, "max abs diff:", np.abs(chk[c+"_a"]-chk[c+"_m"]).max())
print("seasonal cols list:", [c for c in seas.columns if c not in key])


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, sklearn
print("xgb", xgb.__version__, "| sklearn", sklearn.__version__)
try:
    m = xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.4)
    print("tweedie OK")
except Exception as e: print("tweedie ERR", e)
try:
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5)
    print("quantileerror OK")
except Exception as e: print("quantileerror ERR", e)

# 9-learner chunked search on rows where all OOFs exist
oof = A.load_saved("oof_e016_cv.parquet")
oh = A.load_saved("oof_harness.parquet"); op = A.load_saved("oof_pt.parquet")
m = oof.merge(oh, on=["household_key","snapshot_day"], how="left").merge(op, on=["household_key","snapshot_day"], how="left")
L = ["med_v3","med_all","hgbq_v3","hgbq_all","oof_med","oof_sq","oof_med_w112","oof_med_w224","oof_pt"]
mm = m.dropna(subset=L)
print("rows with all 9 OOFs:", len(mm), "snapshot days:", sorted(mm.snapshot_day.unique()))
y = mm["y"].values; P = mm[L].values
for k,l in enumerate(L): print(f"{l:12s} MAE {np.abs(y-P[:,k]).mean():.4f}")
rng = np.random.default_rng(1)
best_mae, best_w = 1e9, None
for it in range(30):
    W = rng.dirichlet(np.ones(9)*0.5, size=4000)
    PW = P @ W.T
    maes = np.abs(y[:,None]-PW).mean(axis=0)
    b = maes.argmin()
    if maes[b] < best_mae: best_mae, best_w = maes[b], W[b].copy()
print("best random 9-learner w:", dict(zip(L, np.round(best_w,3))), "OOF MAE", round(best_mae,4))
# refine
def mae_w(w): return np.abs(y - P @ w).mean()
bw = best_w.copy(); bm = mae_w(bw)
for step in [0.1,0.05,0.02,0.01,0.005]:
    improved=True
    while improved:
        improved=False
        for i in range(9):
            for j in range(9):
                if i==j or bw[j]<step: continue
                w2=bw.copy(); w2[i]+=step; w2[j]-=step
                m2=mae_w(w2)
                if m2<bm: bw,bm=w2,m2; improved=True
print("refined:", dict(zip(L, np.round(bw,3))), "OOF MAE", round(bm,4))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet")
oh = A.load_saved("oof_harness.parquet"); op = A.load_saved("oof_pt.parquet").rename(columns={"oof":"oof_pt"})
m = oof.merge(oh, on=["household_key","snapshot_day"], how="left").merge(op, on=["household_key","snapshot_day"], how="left")
L = ["med_v3","med_all","hgbq_v3","hgbq_all","oof_med","oof_sq","oof_med_w112","oof_med_w224","oof_pt"]
mm = m.dropna(subset=L)
print("rows with all 9 OOFs:", len(mm), "snapshots:", sorted(mm.snapshot_day.unique()))
y = mm["y"].values; P = mm[L].values
for k,l in enumerate(L): print(f"{l:12s} MAE {np.abs(y-P[:,k]).mean():.4f}")
rng = np.random.default_rng(1)
best_mae, best_w = 1e9, None
for it in range(30):
    W = rng.dirichlet(np.ones(9)*0.5, size=4000)
    PW = P @ W.T
    maes = np.abs(y[:,None]-PW).mean(axis=0)
    b = maes.argmin()
    if maes[b] < best_mae: best_mae, best_w = maes[b], W[b].copy()
print("best random 9-learner w:", dict(zip(L, np.round(best_w,3))), "OOF MAE", round(best_mae,4))
def mae_w(w): return np.abs(y - P @ w).mean()
bw = best_w.copy(); bm = mae_w(bw)
for step in [0.1,0.05,0.02,0.01,0.005]:
    improved=True
    while improved:
        improved=False
        for i in range(9):
            for j in range(9):
                if i==j or bw[j]<step: continue
                w2=bw.copy(); w2[i]+=step; w2[j]-=step
                m2=mae_w(w2)
                if m2<bm: bw,bm=w2,m2; improved=True
print("refined:", dict(zip(L, np.round(bw,3))), "OOF MAE", round(bm,4))
# how much of that is just because these rows are the last 4 snapshots (easier)?
sub = oof[oof.snapshot_day.isin(sorted(mm.snapshot_day.unique()))]
print("4-learner OOF MAE on same rows:", np.abs(sub.y.values - sub[L[:4]].values @ np.array([0.3894,0,0.2355,0.3751])).mean())


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time

feats = A.load_saved("feats_all_e016.parquet")
tt = A.train_targets()
fcols = [c for c in feats.columns if c not in ("household_key","snapshot_day")]
tr = tt.merge(feats, on=["household_key","snapshot_day"], how="left")
X = tr[fcols].astype(np.float32).values
y = tr.future_spend_4w.values.astype(np.float32)
hh = tr.household_key.values

# household-grouped 5-fold CV for Tweedie learner
rng = np.random.default_rng(7)
u = np.unique(hh); fmap = dict(zip(u, rng.permutation(u) % 5))
fold = np.array([fmap[h] for h in hh])
t0 = time.time()
oof_tw = np.zeros(len(tr))
for k in range(5):
    m = xgb.XGBRegressor(n_estimators=900, learning_rate=0.035, max_depth=6, min_child_weight=10,
                         subsample=0.8, colsample_bytree=0.8, objective="reg:tweedie",
                         tweedie_variance_power=1.4, tree_method="hist", n_jobs=4, random_state=k)
    m.fit(X[fold != k], y[fold != k])
    oof_tw[fold == k] = m.predict(X[fold == k])
    print("fold", k, "done", round(time.time()-t0,1), "s")
print("tweedie OOF MAE:", round(np.abs(y - oof_tw).mean(), 4))

A.save_table(pd.DataFrame({"household_key": tr.household_key, "snapshot_day": tr.snapshot_day, "oof_tw": oof_tw}), "oof_tw")

# blend searches
oof = A.load_saved("oof_e016_cv.parquet").merge(
      A.load_saved("oof_tw.parquet"), on=["household_key","snapshot_day"], how="left")
L4 = ["med_v3","med_all","hgbq_v3","hgbq_all","oof_tw"]
P4 = oof[L4].values; y4 = oof["y"].values
print("tweedie corr with med_v3:", round(np.corrcoef(oof.oof_tw, oof.med_v3)[0,1], 4))
for k,l in enumerate(L4): print(f"{l:9s} MAE {np.abs(y4-P4[:,k]).mean():.4f}")
rng2 = np.random.default_rng(2); bm, bw = 1e9, None
for it in range(30):
    W = rng2.dirichlet(np.ones(5)*0.5, size=4000)
    maes = np.abs(y4[:,None] - (P4 @ W.T)).mean(axis=0)
    b = maes.argmin()
    if maes[b] < bm: bm, bw = maes[b], W[b].copy()
def mae_w(w): return np.abs(y4 - P4 @ w).mean()
bw = bw.copy(); bm = mae_w(bw)
for step in [0.1,0.05,0.02,0.01,0.005]:
    improved=True
    while improved:
        improved=False
        for i in range(5):
            for j in range(5):
                if i==j or bw[j]<step: continue
                w2=bw.copy(); w2[i]+=step; w2[j]-=step
                if mae_w(w2)<bm: bw,bm=w2,mae_w(w2); improved=True
print("5-learner optimal:", dict(zip(L4, np.round(bw,3))), "OOF MAE", round(bm,4))

# subset search incl. legacy OOFs
op = A.load_saved("oof_pt.parquet").rename(columns={"oof":"oof_pt"})
oe = A.load_saved("oof_e008.parquet")
oh = A.load_saved("oof_harness.parquet")
m2 = oof.merge(oh, on=["household_key","snapshot_day"], how="left").merge(op, on=["household_key","snapshot_day"], how="left").merge(oe[["household_key","snapshot_day","oof_log"]], on=["household_key","snapshot_day"], how="left")
L9 = L4 + ["oof_med","oof_sq","oof_log","oof_med_w112","oof_med_w224","oof_pt"]
mm = m2.dropna(subset=L9)
y9 = mm["y"].values; P9 = mm[L9].values
print("subset rows:", len(mm))
rng3 = np.random.default_rng(3); bm9, bw9 = 1e9, None
for it in range(30):
    W = rng3.dirichlet(np.ones(9)*0.5, size=4000)
    maes = np.abs(y9[:,None] - (P9 @ W.T)).mean(axis=0)
    b = maes.argmin()
    if maes[b] < bm9: bm9, bw9 = maes[b], W[b].copy()
def mae_w9(w): return np.abs(y9 - P9 @ w).mean()
bw9 = bw9.copy(); bm9 = mae_w9(bw9)
for step in [0.1,0.05,0.02,0.01,0.005]:
    improved=True
    while improved:
        improved=False
        for i in range(9):
            for j in range(9):
                if i==j or bw9[j]<step: continue
                w2=bw9.copy(); w2[i]+=step; w2[j]-=step
                if mae_w9(w2)<bm9: bw9,bm9=w2,mae_w9(w2); improved=True
print("9-learner optimal:", dict(zip(L9, np.round(bw9,3))), "OOF MAE", round(bm9,4))

# full-train tweedie -> validation predictions
val = feats[feats.snapshot_day.isin([459,487,515,543])]
Xv = val[fcols].astype(np.float32).values
mf = xgb.XGBRegressor(n_estimators=900, learning_rate=0.035, max_depth=6, min_child_weight=10,
                      subsample=0.8, colsample_bytree=0.8, objective="reg:tweedie",
                      tweedie_variance_power=1.4, tree_method="hist", n_jobs=4, random_state=11)
mf.fit(X, y)
ptw = mf.predict(Xv)
A.save_table(pd.DataFrame({"household_key": val.household_key.values, "snapshot_day": val.snapshot_day.values, "prediction": ptw}), "pred_tw")
print("pred_tw saved", ptw.shape, "mean", round(ptw.mean(),2))


# ---- cell ----
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


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
from scipy.optimize import minimize

oof = A.load_saved("oof_e016_cv.parquet").merge(A.load_saved("oof_tw.parquet"), on=["household_key","snapshot_day"], how="left")
oh = A.load_saved("oof_harness.parquet")
op = A.load_saved("oof_pt.parquet").rename(columns={"oof":"oof_pt"})
oe = A.load_saved("oof_e008.parquet")
m2 = oof.merge(oh, on=["household_key","snapshot_day"], how="left").merge(op, on=["household_key","snapshot_day"], how="left").merge(oe[["household_key","snapshot_day","oof_log"]], on=["household_key","snapshot_day"], how="left")
L = ["med_v3","med_all","hgbq_v3","hgbq_all","oof_tw","oof_med","oof_sq","oof_log","oof_med_w112","oof_med_w224","oof_pt"]
mm = m2.dropna(subset=L)
y = mm["y"].values; P = mm[L].values

# per-snapshot optimal weights (simplex, coordinate refinement)
def refine(y, P, w0, steps=(0.1,0.05,0.02,0.01,0.005)):
    K = len(w0); w = w0.copy(); bm = np.abs(y - P@w).mean()
    for step in steps:
        improved=True
        while improved:
            improved=False
            for i in range(K):
                for j in range(K):
                    if i==j or w[j]<step: continue
                    w2=w.copy(); w2[i]+=step; w2[j]-=step
                    m2=np.abs(y-P@w2).mean()
                    if m2<bm: w,bm=w2,m2; improved=True
    return w, bm

for d in sorted(mm.snapshot_day.unique()):
    s = mm[mm.snapshot_day==d]
    ys, Ps = s["y"].values, s[L].values
    rng = np.random.default_rng(int(d))
    bm, bw = 1e9, None
    for it in range(15):
        W = rng.dirichlet(np.ones(len(L))*0.5, size=3000)
        maes = np.abs(ys[:,None]-(Ps@W.T)).mean(axis=0)
        b = maes.argmin()
        if maes[b]<bm: bm,bw = maes[b],W[b].copy()
    w,bm = refine(ys,Ps,bw)
    print(d, "n",len(s), "MAE", round(bm,3), dict(zip(L,np.round(w,2))))

# global simplex optimal on the subset (again, for reference)
rng = np.random.default_rng(3); bm, bw = 1e9, None
for it in range(40):
    W = rng.dirichlet(np.ones(len(L))*0.5, size=4000)
    maes = np.abs(y[:,None]-(P@W.T)).mean(axis=0)
    b = maes.argmin()
    if maes[b]<bm: bm,bw = maes[b],W[b].copy()
wg,bmg = refine(y,P,bw)
print("\nglobal subset optimal:", dict(zip(L,np.round(wg,3))), round(bmg,4))

# LAD stacking with unconstrained (nonneg) weights via scipy
K = len(L)
def obj(w): return np.abs(y - P@w).mean()
res = minimize(obj, np.ones(K)/K, method="Nelder-Mead", options={"maxiter":4000,"xatol":1e-4,"fatol":1e-6})
wnn = np.clip(res.x, 0, None); wnn = wnn/wnn.sum()
print("unconstrained-NM (nonneg renorm):", dict(zip(L,np.round(wnn,3))), round(obj(wnn),4))
# allow negatives
res2 = minimize(obj, wnn, method="Nelder-Mead", options={"maxiter":6000,"xatol":1e-4,"fatol":1e-6})
print("neg-allowed NM:", dict(zip(L,np.round(res2.x,3))), round(obj(res2.x),4))


# ---- cell ----
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


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
from sklearn.ensemble import HistGradientBoostingRegressor
KEYS = ["household_key","snapshot_day"]

def norm(df):
    df = df.copy()
    df["household_key"] = df["household_key"].astype(str)
    df["snapshot_day"] = df["snapshot_day"].astype(int)
    return df

oof = norm(A.load_saved("oof_e016_cv.parquet"))
oh  = norm(A.load_saved("oof_harness.parquet"))
op  = norm(A.load_saved("oof_pt.parquet").rename(columns={"oof":"oof_pt"}))
tw  = norm(A.load_saved("oof_tw.parquet"))
for nm,d in [("oof",oof),("oh",oh),("op",op),("tw",tw)]:
    print(nm, d.shape, sorted(d.snapshot_day.unique())[:6], "nuniq_hh", d.household_key.nunique())
m = oof.merge(oh, on=KEYS, how="left").merge(op, on=KEYS, how="left").merge(tw[KEYS+["oof_tw"]], on=KEYS, how="left")
print("merged rows:", len(m))
L6 = ["med_v3","med_all","hgbq_v3","hgbq_all","oof_tw","oof_pt"]
print("nonnan per col:", m[L6].notna().mean().round(3).to_dict())

def fit_w(y, P, seed=0, nrand=40):
    K = P.shape[1]; rng = np.random.default_rng(seed); bm, bw = 1e9, None
    for it in range(nrand):
        W = rng.dirichlet(np.ones(K)*0.5, size=4000)
        maes = np.abs(y[:,None] - (P @ W.T)).mean(axis=0)
        b = maes.argmin()
        if maes[b] < bm: bm, bw = maes[b], W[b].copy()
    w = bw.copy(); bm = np.abs(y - P@w).mean()
    for step in (0.1,0.05,0.02,0.01,0.005):
        imp=True
        while imp:
            imp=False
            for i in range(K):
                for j in range(K):
                    if i==j or w[j]<step: continue
                    w2=w.copy(); w2[i]+=step; w2[j]-=step
                    m2=np.abs(y-P@w2).mean()
                    if m2<bm: w,bm=w2,m2; imp=True
    return w, bm

pt_ok = m["oof_pt"].notna().sum() > 26000
print("pt_ok:", pt_ok, int(m["oof_pt"].notna().sum()))
y_all = m["y"].values
if pt_ok:
    P_all = m[L6].values
    w_full, mae_full = fit_w(y_all, P_all, seed=1)
    print("C1 full-data 6-learner:", dict(zip(L6, np.round(w_full,3))), round(mae_full,4))
    sub = m[m.snapshot_day >= 347]
    w_sub, mae_sub = fit_w(sub["y"].values, sub[L6].values, seed=2)
    print("C2 late-sub 6-learner:", dict(zip(L6, np.round(w_sub,3))), round(mae_sub,4))
    w_avg = 0.5*w_full + 0.5*w_sub; w_avg = w_avg/w_avg.sum()
    print("C3 avg:", dict(zip(L6, np.round(w_avg,3))),
          "full", round(np.abs(y_all - P_all@w_avg).mean(),4),
          "sub", round(np.abs(sub["y"].values - sub[L6].values@w_avg).mean(),4))
cands = {}
if pt_ok:
    cands["C1_full6"] = (w_full, L6); cands["C2_sub6"] = (w_sub, L6); cands["C3_avg6"] = (w_avg, L6)
c4 = np.array([0.378,0.001,0.208,0.307,0.106,0.0]); cands["C4_full5+0pt"] = (c4, L6)
c5 = np.array([0.4,0.3,0.3,0.0,0.0,0.0]); cands["C5_e016assumed"] = (c5, L6)
print("\ncandidate eval (full / late-sub / mean):")
best_name, best_sc = None, 1e9
for nm,(w,LL) in cands.items():
    mf_ = np.abs(y_all - m[LL].values @ w).mean()
    ms_ = np.abs(sub["y"].values - sub[LL].values @ w).mean()
    sc = (mf_+ms_)/2
    print(f"{nm:16s} full {mf_:8.4f}  sub {ms_:8.4f}  mean {sc:8.4f}")
    if sc < best_sc: best_sc, best_name = sc, nm
w_use, L_use = cands[best_name]
print("CHOSEN:", best_name, dict(zip(L_use, np.round(w_use,4))))

# ---- retrain 4 base learners full-train, predict validation ----
t0=time.time()
fv3 = A.load_saved("feats_v3.parquet"); fall = A.load_saved("feats_all_e016.parquet")
tt = A.train_targets()
def mat(f):
    t = tt.merge(f, on=KEYS, how="left")
    cols = [c for c in f.columns if c not in KEYS]
    return t[cols].astype(np.float32).values, t
Xv3, tr3 = mat(fv3); Xal, tral = mat(fall)
ytr = tt.future_spend_4w.values.astype(np.float32)
val = fall[fall.snapshot_day.isin([459,487,515,543])]
valk = val[KEYS]
Xv3v = valk.merge(fv3, on=KEYS, how="left")[[c for c in fv3.columns if c not in KEYS]].astype(np.float32).values
Xalv = val[[c for c in fall.columns if c not in KEYS]].astype(np.float32).values
print("mats", Xv3.shape, Xal.shape, Xv3v.shape, Xalv.shape, round(time.time()-t0,1))

def xgbq(X, Xv, seed):
    mfit = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=10,
                            subsample=0.8, colsample_bytree=0.8, objective="reg:quantileerror",
                            quantile_alpha=0.5, tree_method="hist", n_jobs=4, random_state=seed)
    mfit.fit(X, ytr); return mfit.predict(Xv)
def hgbq(X, Xv):
    ps=[]
    for s in (1,2):
        h = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=400,
                                          learning_rate=0.06, random_state=s)
        h.fit(X, ytr); ps.append(h.predict(Xv))
    return np.mean(ps, axis=0)

pv = {}
pv["med_v3"]   = xgbq(Xv3, Xv3v, 21); print("med_v3 done", round(time.time()-t0,1))
pv["med_all"]  = xgbq(Xal, Xalv, 22); print("med_all done", round(time.time()-t0,1))
pv["hgbq_v3"]  = hgbq(Xv3, Xv3v);     print("hgbq_v3 done", round(time.time()-t0,1))
pv["hgbq_all"] = hgbq(Xal, Xalv);     print("hgbq_all done", round(time.time()-t0,1))
ptw = A.load_saved("pred_tw.parquet"); ppt = A.load_saved("pred_pt.parquet")
pv["oof_tw"] = valk.merge(ptw, on=KEYS, how="left")["prediction"].values
pv["oof_pt"] = valk.merge(ppt, on=KEYS, how="left")["prediction"].values
print("nan check:", {k: int(np.isnan(v).sum()) for k,v in pv.items()})

pred = np.zeros(len(valk))
for w_, l_ in zip(w_use, L_use): pred += w_ * pv[l_]
out = valk.copy(); out["prediction"] = pred
p = A.save_table(out, "pred_e019")
print("SAVED", p, out.shape, "pred mean", round(pred.mean(),2), "w:", dict(zip(L_use, np.round(w_use,4))))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
from sklearn.ensemble import HistGradientBoostingRegressor
KEYS = ["household_key","snapshot_day"]

oof = A.load_saved("oof_e016_cv.parquet").merge(A.load_saved("oof_tw.parquet"), on=KEYS, how="left")
op  = A.load_saved("oof_pt.parquet").rename(columns={"oof":"oof_pt"})
m   = oof.merge(op, on=KEYS, how="left")
L6  = ["med_v3","med_all","hgbq_v3","hgbq_all","oof_tw","oof_pt"]

def fit_w(y, P, seed, nrand=40):
    K = P.shape[1]; rng = np.random.default_rng(seed); bm, bw = 1e9, None
    for it in range(nrand):
        W = rng.dirichlet(np.ones(K)*0.5, size=4000)
        maes = np.abs(y[:,None] - (P @ W.T)).mean(axis=0)
        b = maes.argmin()
        if maes[b] < bm: bm, bw = maes[b], W[b].copy()
    w = bw.copy(); bm = np.abs(y - P@w).mean()
    for step in (0.1,0.05,0.02,0.01,0.005):
        imp=True
        while imp:
            imp=False
            for i in range(K):
                for j in range(K):
                    if i==j or w[j]<step: continue
                    w2=w.copy(); w2[i]+=step; w2[j]-=step
                    m2=np.abs(y-P@w2).mean()
                    if m2<bm: w,bm=w2,m2; imp=True
    return w, bm

sub = m[m.snapshot_day >= 347]
y_all, P5 = m["y"].values, m[L6[:5]].values
y_sub, P6 = sub["y"].values, sub[L6].values
w5, m5 = fit_w(y_all, P5, seed=11)
w6, m6 = fit_w(y_sub, P6, seed=12)
print("full 5-learner:", dict(zip(L6[:5], np.round(w5,3))), round(m5,4))
print("late 6-learner:", dict(zip(L6, np.round(w6,3))), round(m6,4))
w_avg = np.zeros(6); w_avg[:5] = w5; w_avg = 0.5*w_avg + 0.5*w6; w_avg = w_avg/w_avg.sum()
cands = {"C1_full5": (np.r_[w5,0.0],), "C2_late6": (w6,), "C3_avg": (w_avg,),
         "C4_hand": (np.array([0.35,0.05,0.25,0.25,0.05,0.05]),)}
best_name, best_sc = None, 1e9
for nm,(w,) in cands.items():
    f_ = np.abs(y_all - m[L6].values @ w).mean()
    s_ = np.abs(y_sub - P6 @ w).mean()
    sc = (f_+s_)/2
    print(f"{nm:9s} full {f_:8.4f} sub {s_:8.4f} mean {sc:8.4f}  w={dict(zip(L6,np.round(w,3)))}")
    if sc < best_sc: best_sc, best_name = sc, nm
w_use = cands[best_name][0]
print("CHOSEN", best_name)

# retrain 4 base learners on FULL train, predict validation
t0=time.time()
fv3 = A.load_saved("feats_v3.parquet"); fall = A.load_saved("feats_all_e016.parquet")
tt = A.train_targets()
def cols(f): return [c for c in f.columns if c not in KEYS]
tr3 = tt.merge(fv3, on=KEYS, how="left"); tral = tt.merge(fall, on=KEYS, how="left")
Xv3, Xal = tr3[cols(fv3)].astype(np.float32).values, tral[cols(fall)].astype(np.float32).values
ytr = tt.future_spend_4w.values.astype(np.float32)
val = fall[fall.snapshot_day.isin([459,487,515,543])]
valk = val[KEYS].reset_index(drop=True)
Xv3v = valk.merge(fv3, on=KEYS, how="left")[cols(fv3)].astype(np.float32).values
Xalv = val[cols(fall)].astype(np.float32).values
def xgbq(X, Xv, seed):
    mm_ = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=10,
                           subsample=0.8, colsample_bytree=0.8, objective="reg:quantileerror",
                           quantile_alpha=0.5, tree_method="hist", n_jobs=4, random_state=seed)
    mm_.fit(X, ytr); return mm_.predict(Xv)
def hgbq(X, Xv):
    return np.mean([HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=400,
                   learning_rate=0.06, random_state=s).fit(X, ytr).predict(Xv) for s in (1,2)], axis=0)
pv = {"med_v3": xgbq(Xv3, Xv3v, 21), "med_all": xgbq(Xal, Xalv, 22),
      "hgbq_v3": hgbq(Xv3, Xv3v), "hgbq_all": hgbq(Xal, Xalv)}
print("base learners done", round(time.time()-t0,1), "s")
pv["oof_tw"] = valk.merge(A.load_saved("pred_tw.parquet"), on=KEYS, how="left")["prediction"].values
pv["oof_pt"] = valk.merge(A.load_saved("pred_pt.parquet"), on=KEYS, how="left")["prediction"].values
print("nan:", {k:int(np.isnan(v).sum()) for k,v in pv.items()})
pred = np.zeros(len(valk))
for w_, l_ in zip(w_use, L6): pred += w_*pv[l_]
out = valk.copy(); out["prediction"] = pred
p = A.save_table(out, "pred_e019")
print("SAVED", p, out.shape, "mean", round(pred.mean(),2), "std", round(pred.std(),2), "w", dict(zip(L6,np.round(w_use,4))))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
from sklearn.ensemble import HistGradientBoostingRegressor
KEYS = ["household_key","snapshot_day"]

oof = A.load_saved("oof_e016_cv.parquet").merge(A.load_saved("oof_tw.parquet"), on=KEYS, how="left")
op  = A.load_saved("oof_pt.parquet").rename(columns={"oof":"oof_pt"})
m   = oof.merge(op, on=KEYS, how="left")
L6  = ["med_v3","med_all","hgbq_v3","hgbq_all","oof_tw","oof_pt"]

def fit_w(y, P, seed, nrand=40):
    K = P.shape[1]; rng = np.random.default_rng(seed); bm, bw = 1e9, None
    for it in range(nrand):
        W = rng.dirichlet(np.ones(K)*0.5, size=4000)
        maes = np.abs(y[:,None] - (P @ W.T)).mean(axis=0)
        b = maes.argmin()
        if maes[b] < bm: bm, bw = maes[b], W[b].copy()
    w = bw.copy(); bm = np.abs(y - P@w).mean()
    for step in (0.1,0.05,0.02,0.01,0.005):
        imp=True
        while imp:
            imp=False
            for i in range(K):
                for j in range(K):
                    if i==j or w[j]<step: continue
                    w2=w.copy(); w2[i]+=step; w2[j]-=step
                    m2=np.abs(y-P@w2).mean()
                    if m2<bm: w,bm=w2,m2; imp=True
    return w, bm

sub = m[m.snapshot_day >= 347]
w5, m5 = fit_w(m["y"].values, m[L6[:5]].values, seed=11)
w6, m6 = fit_w(sub["y"].values, sub[L6].values, seed=12)
w_avg = np.r_[w5, 0.0]; w_avg = 0.5*w_avg + 0.5*w6; w_avg = w_avg/w_avg.sum()
C1 = np.r_[w5, 0.0]                      # full-data 5-learner fit
C2 = w6                                  # late-subset 6-learner fit
C3 = w_avg                               # hedge
print("C1", dict(zip(L6,np.round(C1,3))))
print("C2", dict(zip(L6,np.round(C2,3))))
print("C3", dict(zip(L6,np.round(C3,3))))
# proper evals: full-data on 5 cols, sub on 6 cols
for nm,w in [("C1",C1),("C2",C2),("C3",C3)]:
    f_ = np.abs(m["y"].values - m[L6[:5]].values @ w[:5]).mean()
    s_ = np.abs(sub["y"].values - sub[L6].values @ w).mean()
    print(f"{nm}: full5 {f_:.4f}  late6 {s_:.4f}")

# retrain 4 base learners on FULL train -> validation predictions
t0=time.time()
fv3 = A.load_saved("feats_v3.parquet"); fall = A.load_saved("feats_all_e016.parquet")
tt = A.train_targets()
def cols(f): return [c for c in f.columns if c not in KEYS]
tr3 = tt.merge(fv3, on=KEYS, how="left"); tral = tt.merge(fall, on=KEYS, how="left")
Xv3, Xal = tr3[cols(fv3)].astype(np.float32).values, tral[cols(fall)].astype(np.float32).values
ytr = tt.future_spend_4w.values.astype(np.float32)
val = fall[fall.snapshot_day.isin([459,487,515,543])]
valk = val[KEYS].reset_index(drop=True)
Xv3v = valk.merge(fv3, on=KEYS, how="left")[cols(fv3)].astype(np.float32).values
Xalv = val[cols(fall)].astype(np.float32).values
def xgbq(X, Xv, seed):
    mm_ = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=10,
                           subsample=0.8, colsample_bytree=0.8, objective="reg:quantileerror",
                           quantile_alpha=0.5, tree_method="hist", n_jobs=4, random_state=seed)
    mm_.fit(X, ytr); return mm_.predict(Xv)
def hgbq(X, Xv):
    return np.mean([HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=400,
                   learning_rate=0.06, random_state=s).fit(X, ytr).predict(Xv) for s in (1,2)], axis=0)
pv = {"med_v3": xgbq(Xv3, Xv3v, 21), "med_all": xgbq(Xal, Xalv, 22),
      "hgbq_v3": hgbq(Xv3, Xv3v), "hgbq_all": hgbq(Xal, Xalv)}
print("base learners done", round(time.time()-t0,1), "s")
pv["oof_tw"] = valk.merge(A.load_saved("pred_tw.parquet"), on=KEYS, how="left")["prediction"].values
pv["oof_pt"] = valk.merge(A.load_saved("pred_pt.parquet"), on=KEYS, how="left")["prediction"].values
print("nan:", {k:int(np.isnan(v).sum()) for k,v in pv.items()})
Pv = np.column_stack([pv[l] for l in L6])
for nm, w in [("C1",C1),("C2",C2),("C3",C3)]:
    pred = Pv @ w
    out = valk.copy(); out["prediction"] = pred
    p = A.save_table(out, f"pred_e019_{nm}")
    print("SAVED", p, out.shape, "mean", round(pred.mean(),2), "std", round(pred.std(),2))
