import numpy as np, pandas as pd, agent_api as A, warnings
warnings.filterwarnings("ignore")
aw = A.load_saved("aw_hist.parquet")
df = A.load_saved("e005_marketing.parquet")
m = A.train_targets().merge(df, on=["household_key","snapshot_day"], how="left").merge(aw, on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"].astype(float).values
tr_days = A.snapshot_days()["train"]; PV=431
is_pv = m["snapshot_day"]==PV
is_fit = m["snapshot_day"].isin([d for d in tr_days if d<PV])
yv = y[is_pv]
awc = [f"aw_{k}" for k in range(1,14)]
base_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
Xd = m[base_cols].copy()
for c in Xd.columns:
    if not np.issubdtype(Xd[c].dtype, np.number): Xd[c]=Xd[c].astype("category").cat.codes
Xd = Xd.replace([np.inf,-np.inf],np.nan)
Xb = Xd.values.astype(float)

def ridge_eval(X, alpha=3000):
    Xf,Xv=X[is_fit],X[is_pv]
    mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0)+1e-9
    Zf=np.nan_to_num(np.where(np.isnan(Xf),mu,Xf)/sd); Zv=np.nan_to_num(np.where(np.isnan(Xv),mu,Xv)/sd)
    G=Zf.T@Zf+alpha*np.eye(Zf.shape[1]); w=np.linalg.solve(G,Zf.T@y[is_fit])
    return float(np.abs(yv-Zv@w).mean())

print("base:", round(ridge_eval(Xb),2))
f_med8 = m[awc[:8]].median(1).fillna(0).values.reshape(-1,1)
f_mean8 = m[awc[:8]].mean(1).fillna(0).values.reshape(-1,1)
X1 = np.column_stack([Xb, f_med8, f_mean8])
print("base+aw8med/mean:", round(ridge_eval(X1),2))

# rank transform of all features (tree-like invariance) as EXTRA cols for top features only
def rank_col(v):
    r = pd.Series(v).rank(pct=True).values
    return (r-0.5).reshape(-1,1)
top = ["spend_112_mean4","spend_28w","spend_4w","spend_56w","spend_per_day_total","spend_112_max4","spend_112_min4","spend_4w_lag1","nbask_28w","spend_112_std"]
Xr = np.column_stack([X1] + [rank_col(m[c].fillna(0).values) for c in top])
print("base+aw+rank(top10):", round(ridge_eval(Xr),2))

# squared top feature (convexity)
s = m["spend_112_mean4"].fillna(0).values
Xsq = np.column_stack([X1, (s**2/1e4).reshape(-1,1), rank_col(s)])
print("base+aw+sq+rank(top1):", round(ridge_eval(Xsq),2))

# log1p copies of key spend cols
key = ["spend_4w","spend_8w","spend_12w","spend_28w","spend_56w","spend_112w","spend_total","nbask_4w","nbask_8w","nbask_28w","nbask_56w","nbask_112w","spend_per_day_total","spend_per_week_112","spend_112_mean4","spend_112_max4","spend_112_min4","spend_112_std","spend_4w_lag1","spend_4w_lag2","spend_4w_lag3"]
Xlog = np.column_stack([X1] + [np.log1p(np.clip(m[c].fillna(0).values,0,None)).reshape(-1,1) for c in key])
print("base+aw+log1p(21 keys):", round(ridge_eval(Xlog),2))

# combined best guess
Xall = np.column_stack([Xlog] + [rank_col(m[c].fillna(0).values) for c in top[:6]])
print("base+aw+log+rank6:", round(ridge_eval(Xall),2))
print("n cols:", Xall.shape[1])
