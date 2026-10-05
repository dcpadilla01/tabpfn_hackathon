import numpy as np, pandas as pd, agent_api as A
aw = A.load_saved("aw_hist.parquet")
df = A.load_saved("e005_marketing.parquet")
tt = A.train_targets()
m = tt.merge(df, on=["household_key","snapshot_day"], how="left").merge(aw, on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"].astype(float).values
tr_days = A.snapshot_days()["train"]; PV=431
is_pv = m["snapshot_day"]==PV
is_fit = m["snapshot_day"].isin([d for d in tr_days if d<PV])
yv = y[is_pv]
awc = [f"aw_{k}" for k in range(1,14)]
A_ = m[awc]; n_avail = A_.notna().sum(1)

def ridge_eval(X, y, is_fit, is_pv, alpha=3000):
    Xf,Xv=X[is_fit],X[is_pv]
    mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0)+1e-9
    Zf=np.nan_to_num(np.where(np.isnan(Xf),mu,Xf)/sd); Zv=np.nan_to_num(np.where(np.isnan(Xv),mu,Xv)/sd)
    G=Zf.T@Zf+alpha*np.eye(Zf.shape[1]); w=np.linalg.solve(G,Zf.T@y[is_fit])
    return float(np.abs(y[is_pv]-Zv@w).mean())

base_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
Xb = m[base_cols].copy()
for c in Xb.columns:
    if not np.issubdtype(Xb[c].dtype, np.number): Xb[c]=Xb[c].astype("category").cat.codes
Xb = Xb.replace([np.inf,-np.inf],np.nan).values.astype(float)

print("base:", round(ridge_eval(Xb,y,is_fit,is_pv),2))
for Kk in [3,5,8,13]:
    sub = m[[f"aw_{k}" for k in range(1,Kk+1)]]
    f_med = sub.median(1).values.reshape(-1,1); f_mean = sub.mean(1).values.reshape(-1,1)
    print(f"K={Kk}: med {ridge_eval(np.column_stack([Xb,f_med]),y,is_fit,is_pv):.2f} | mean {ridge_eval(np.column_stack([Xb,f_mean]),y,is_fit,is_pv):.2f} | both {ridge_eval(np.column_stack([Xb,f_med,f_mean]),y,is_fit,is_pv):.2f}")
print("aw_med13 raw pv MAE:", round(float(np.abs(yv-np.nan_to_num(m[awc].median(1).values)[is_pv]).mean()),2))
print("aw_mean13 raw pv MAE:", round(float(np.abs(yv-np.nan_to_num(m[awc].mean(1).values)[is_pv]).mean()),2))
# weighted: recent lags weighted more
wts = np.array([1/k for k in range(1,14)])
f_wm = (m[awc]*wts).sum(1)/ (m[awc].notna()*wts).sum(1).values
print("aw_wmean raw pv MAE:", round(float(np.abs(yv-np.nan_to_num(f_wm.values)[is_pv]).mean()),2))
