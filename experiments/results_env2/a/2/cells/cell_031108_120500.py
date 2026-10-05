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
