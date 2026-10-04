import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()

names = ["e001_txhist","e003_catmix","e004_mkt","e004_mkt_full","nf_p1","nf_candidates","nf_transforms","nf_seasonal"]
tabs = {nm: A.load_saved(nm+".parquet") for nm in names}

def prep(df):
    fcols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
    m = tt.merge(df, on=["household_key","snapshot_day"], how="left")
    X = m[fcols].copy()
    for c in fcols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf,-np.inf], np.nan)
    return X, fcols, m

def ridge_fit(X, y, lam=10.0):
    mu = X.mean(); sd = X.std().replace(0,1.0)
    Z = ((X-mu)/sd).fillna(0.0).values
    Z = np.hstack([Z, np.ones((len(Z),1))])
    A_ = Z.T@Z + lam*np.eye(Z.shape[1]); A_[-1,-1] -= lam
    w = np.linalg.solve(A_, Z.T@y.values)
    return mu, sd, w

def ridge_pred(X, mu, sd, w):
    Z = ((X-mu)/sd).fillna(0.0).values
    Z = np.hstack([Z, np.ones((len(Z),1))])
    return Z@w

TRAIN_DAYS = [95,123,151,179,207,235,263,291,319,347,375,403,431]
def internal_eval(df, hold=(403,431), lam=10.0, logt=False, name=""):
    X, fcols, m = prep(df)
    y = m.future_spend_4w
    tr_days = [d for d in TRAIN_DAYS if d not in hold]
    tr = m.snapshot_day.isin(tr_days)
    va = m.snapshot_day.isin(hold)
    if logt:
        mu,sd,w = ridge_fit(X[tr], np.log1p(y[tr]), lam)
        p = np.expm1(np.clip(ridge_pred(X[va],mu,sd,w),0,15))
    else:
        mu,sd,w = ridge_fit(X[tr], y[tr], lam)
        p = np.clip(ridge_pred(X[va],mu,sd,w),0,None)
    mae = float(np.abs(p - y[va]).mean())
    return mae, len(fcols)

print(f"{'table':16s} {'lin':>8s} {'log':>8s} nfeat")
for nm in names:
    df = tabs[nm]
    m1,_ = internal_eval(df, logt=False)
    m2,_ = internal_eval(df, logt=True)
    print(f"{nm:16s} {m1:8.3f} {m2:8.3f} {df.shape[1]-2}")

# combos
e1, e3, mkt = tabs["e001_txhist"], tabs["e003_catmix"], tabs["e004_mkt"]
nf = tabs["nf_p1"]
key = ["household_key","snapshot_day"]
def merge(a,b): return a.merge(b, on=key, how="inner", suffixes=("","_d"))
c_e3_nf = merge(e3, nf)
m1,_ = internal_eval(c_e3_nf); m2,_ = internal_eval(c_e3_nf, logt=True)
print(f"{'E003+nf_p1':16s} {m1:8.3f} {m2:8.3f} {c_e3_nf.shape[1]-2}")
c_all = merge(c_e3_nf, mkt)
m1,_ = internal_eval(c_all); m2,_ = internal_eval(c_all, logt=True)
print(f"{'E003+nf+mkt':16s} {m1:8.3f} {m2:8.3f} {c_all.shape[1]-2}")
