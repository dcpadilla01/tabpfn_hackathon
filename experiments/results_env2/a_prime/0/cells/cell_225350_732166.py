import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
y = m["future_spend_4w"].values
feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
X = m[feats].copy()
obj = [c for c in feats if X[c].dtype == object]
for c in obj:
    X[c] = pd.factorize(X[c])[0]
X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(float)
print("X", X.shape, "obj cols:", obj)

def mae(p, t): return np.mean(np.abs(p - t))

# --- conditional structure ---
print("\n== y by recent-activity ==")
for col in ["spend_4w_recent","nbask_4w","spend_4w_lag1"]:
    if col in m.columns:
        g = (m[col].fillna(0) > 0)
        print(col, "| inactive n=%d mean=%.1f med=%.1f zero%%=%.1f || active n=%d mean=%.1f med=%.1f zero%%=%.1f" % (
            (~g).sum(), y[~g].mean(), np.median(y[~g]), (y[~g]==0).mean()*100,
            g.sum(), y[g].mean(), np.median(y[g]), (y[g]==0).mean()*100))

print("\n== simple predictor MAE on snapshot 431 rows (last train snapshot) ==")
t431 = m[m.snapshot_day==431]; y431 = t431.future_spend_4w.values
tr = m[m.snapshot_day<=403]
cands = {
 "global_median(<=403)": np.full(len(t431), np.median(tr.future_spend_4w)),
 "te_hh_mean": t431.te_hh_mean.values,
 "te_hh_shrunk": t431.te_hh_shrunk.values,
 "spend_112": t431.spend_112.values,
 "spend_4w_recent": t431.spend_4w_recent.values,
 "blend(te,spend112)": 0.5*(t431.te_hh_mean.values + t431.spend_112.values),
}
for k,v in cands.items(): print(f"{k:24s} {mae(v, y431):8.3f}")

print("\n== calendar: mean/median y by snapshot_day (train) ==")
cal = m.groupby("snapshot_day").future_spend_4w.agg(["mean","median","size"])
print(cal.round(1))

# --- local ridge: raw vs log1p, fit <=403, eval 431 ---
def ridge_eval(Xa, ya, mask_tr, mask_te, lam):
    mu = Xa[mask_tr].mean(0); sd = Xa[mask_tr].std(0)+1e-9
    A = (Xa[mask_tr]-mu)/sd; A = np.c_[np.ones(len(A)), A]
    b = (Xa[mask_te]-mu)/sd; b = np.c_[np.ones(len(b)), b]
    I = np.eye(A.shape[1]); I[0,0]=0
    beta = np.linalg.solve(A.T@A + lam*I, A.T@ya[mask_tr])
    return mae(b@beta, ya[mask_te])

trmask = (m.snapshot_day<=403).values; temask = (m.snapshot_day==431).values
Xs = np.sign(X)*np.log1p(np.abs(X))
print("\n== local ridge (fit<=403, eval@431) ==")
for lam in [1.0, 10.0, 100.0]:
    print(f"lam={lam:6.1f} raw {ridge_eval(X,y,trmask,temask,lam):8.3f}   logsigned {ridge_eval(Xs,y,trmask,temask,lam):8.3f}")

# duplicate columns count
Xdf = pd.DataFrame(X, columns=feats)
dup = Xdf.T.duplicated()
print("\nexact duplicate feature columns:", int(dup.sum()), feats[dup.values][:20] if dup.sum() else "")
