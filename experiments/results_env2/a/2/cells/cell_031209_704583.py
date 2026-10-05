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
