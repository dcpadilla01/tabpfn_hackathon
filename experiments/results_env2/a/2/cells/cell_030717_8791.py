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
