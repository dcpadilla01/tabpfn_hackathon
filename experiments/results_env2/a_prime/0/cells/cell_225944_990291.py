import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")

# outcome-history features from past targets (leak-free: shift(1) within household)
tr = m[["household_key","snapshot_day","future_spend_4w"]].sort_values(["household_key","snapshot_day"])
g = tr.groupby("household_key")["future_spend_4w"]
S = pd.DataFrame({f"lag{k}": g.shift(k) for k in range(1,14)})
W = {hl: 0.5**((np.arange(1,14)-1)/hl) for hl in [1,1.5,2,3,4,6]}
for hl,w in W.items():
    M = S[list(range(1,14))].notna().values.astype(float) * w[None,:]
    tr[f"oh_ewm{hl}"] = (S[list(range(1,14))].fillna(0).values * (M/ M.sum(1,keepdims=True))).sum(1)
tr["oh_last2m"] = g.apply(lambda s: s.shift(1).rolling(2).mean()).reset_index(level=0,drop=True)
tr["oh_last3m"] = g.apply(lambda s: s.shift(1).rolling(3).mean()).reset_index(level=0,drop=True)
tr["oh_mean"] = g.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0,drop=True)
tr["oh_median"] = g.apply(lambda s: s.shift(1).expanding().median()).reset_index(level=0,drop=True)
tr["oh_std"] = g.apply(lambda s: s.shift(1).expanding().std()).reset_index(level=0,drop=True)
tr["oh_min"] = g.apply(lambda s: s.shift(1).expanding().min()).reset_index(level=0,drop=True)
tr["oh_max"] = g.apply(lambda s: s.shift(1).expanding().max()).reset_index(level=0,drop=True)
tr["oh_slope"] = g.apply(lambda s: s.shift(1).diff()).reset_index(level=0,drop=True)
tr["oh_n"] = g.cumcount()
oh_cols = [c for c in tr.columns if c.startswith("oh_")]
m2 = m.merge(tr[["household_key","snapshot_day"]+oh_cols], on=["household_key","snapshot_day"], how="left")

feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
def prep(mm, cols):
    X = mm[cols].copy()
    for c in cols:
        if X[c].dtype == object: X[c] = pd.factorize(X[c])[0]
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(float)
def ridge_eval(cols, fit_max, eval_day, lam=10.0):
    mm = m2[m2.snapshot_day<=eval_day]
    trm = (mm.snapshot_day<=fit_max).values; tem = (mm.snapshot_day==eval_day).values
    Xa = prep(mm, cols); ya = mm.future_spend_4w.values
    mu, sd = Xa[trm].mean(0), Xa[trm].std(0)+1e-9
    A = np.c_[np.ones(trm.sum()), (Xa[trm]-mu)/sd]; B = np.c_[np.ones(tem.sum()), (Xa[tem]-mu)/sd]
    I = np.eye(A.shape[1]); I[0,0]=0
    beta = np.linalg.solve(A.T@A+lam*I, A.T@ya[trm])
    return np.mean(np.abs(B@beta - ya[tem]))
def mae(p,t): return np.mean(np.abs(np.asarray(p)-np.asarray(t)))

base = feats[:]
print("n base feats:", len(base), "n oh feats:", len(oh_cols))
for lam in [1.0, 10.0, 100.0]:
    print(f"lam={lam}: base@431 {ridge_eval(base,403,431,lam):.3f}  base+oh@431 {ridge_eval(base+oh_cols,403,431,lam):.3f}")
print("eval@403(fit<=375): base", round(ridge_eval(base,375,403),3), " base+oh", round(ridge_eval(base+oh_cols,375,403),3))
print("eval@375(fit<=347): base", round(ridge_eval(base,347,375),3), " base+oh", round(ridge_eval(base+oh_cols,347,375),3))

# LEAN table: te block + demo + calendar + core rfm + oh
lean = ["te_hh_mean","te_hh_n","te_hh_shrunk","te_bin","te_prior","te_size","te_home","te_kid",
        "spend_112","spend_56w","spend_28w","spend_12w","spend_4w_recent","nbask_4w","nbask_12w","nbask_112w",
        "nprod_112","nweek_active_112","days_since_last_basket","tenure_days","spend_per_active_week_112",
        "sum_retail_disc","sum_coupon_disc","has_demographics","snapshot_day","week_of_year"]
lean = [c for c in lean if c in df.columns]
print("\nlean cols found:", len(lean), [c for c in lean if c not in df.columns])
for lam in [1.0,10.0,100.0]:
    print(f"lam={lam}: lean@431 {ridge_eval(lean,403,431,lam):.3f}  lean+oh@431 {ridge_eval(lean+oh_cols,403,431,lam):.3f}")
print("eval@403: lean", round(ridge_eval(lean,375,403),3), " lean+oh", round(ridge_eval(lean+oh_cols,375,403),3))
