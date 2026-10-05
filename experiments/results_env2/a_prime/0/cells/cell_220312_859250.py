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

# drop day/week/snap_day/snap_week/index (snapshot-calendar cols) from base
drop = ["day","week","snap_day","snap_week","index"]
keep = [c for c in base_cols if c not in drop]
Xk = m[keep].copy()
for c in Xk.columns:
    if not np.issubdtype(Xk[c].dtype, np.number): Xk[c]=Xk[c].astype("category").cat.codes
Xk = Xk.replace([np.inf,-np.inf],np.nan).values.astype(float)
print("base w/o calendar cols:", round(ridge_eval(Xk),2))
# calendar only
Xc = m[drop].replace([np.inf,-np.inf],np.nan).values.astype(float)
print("calendar only:", round(ridge_eval(Xc),2))

f_med8 = m[awc[:8]].median(1).fillna(0).values.reshape(-1,1)
f_mean8 = m[awc[:8]].mean(1).fillna(0).values.reshape(-1,1)
X1 = np.column_stack([Xk, f_med8, f_mean8])
print("noCal+aw8:", round(ridge_eval(X1),2))
key = ["spend_4w","spend_8w","spend_12w","spend_28w","spend_56w","spend_112w","spend_total","nbask_4w","nbask_8w","nbask_28w","nbask_56w","nbask_112w","spend_per_day_total","spend_per_week_112","spend_112_mean4","spend_112_max4","spend_112_min4","spend_112_std","spend_4w_lag1","spend_4w_lag2","spend_4w_lag3"]
Xlog = np.column_stack([X1] + [np.log1p(np.clip(m[c].fillna(0).values,0,None)).reshape(-1,1) for c in key])
print("noCal+aw8+log1p:", round(ridge_eval(Xlog),2))
