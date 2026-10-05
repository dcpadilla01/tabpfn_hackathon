import numpy as np, pandas as pd, agent_api as A
def fn(view, snapshot_day):
    tx = view.table("transactions")
    hhs = view.households
    g = tx[tx.household_key.isin(set(hhs))].groupby(["household_key","day"])["sales_value"].sum().reset_index()
    R = pd.DataFrame(index=pd.Index(hhs, name="household_key"))
    for k in range(1,14):
        lo, hi = snapshot_day-28*k+1, snapshot_day-28*(k-1)
        w = g[(g.day>=lo)&(g.day<=hi)].groupby("household_key")["sales_value"].sum()
        R[f"aw_{k}"] = w.reindex(R.index).fillna(0.0)
    return R
aw = A.build_features(fn)
A.save_table(aw, "aw_hist")
m0 = A.train_targets().merge(A.load_saved("e005_marketing.parquet"), on=["household_key","snapshot_day"], how="left").merge(aw, on=["household_key","snapshot_day"], how="left")
y = m0["future_spend_4w"].astype(float).values
tr_days = A.snapshot_days()["train"]; PV=431
is_pv = m0["snapshot_day"]==PV
is_fit = m0["snapshot_day"].isin([d for d in tr_days if d<PV])
yv = y[is_pv]
awc = [f"aw_{k}" for k in range(1,14)]
print("aw_1==spend_4w_recent?", np.allclose(m0["aw_1"].fillna(0), m0["spend_4w_recent"].fillna(0)))
print("aw_2==spend_4w_lag1?", np.allclose(m0["aw_2"].fillna(0), m0["spend_4w_lag1"].fillna(0)))
print("aw_med13 raw pv MAE:", round(float(np.abs(yv-np.nan_to_num(m0[awc].median(1).values)[is_pv]).mean()),2))
print("aw_mean13 raw pv MAE:", round(float(np.abs(yv-np.nan_to_num(m0[awc].mean(1).values)[is_pv]).mean()),2))

def ridge_eval(X, y, is_fit, is_pv, alpha=3000):
    Xf,Xv=X[is_fit],X[is_pv]
    mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0)+1e-9
    Zf=np.nan_to_num(np.where(np.isnan(Xf),mu,Xf)/sd); Zv=np.nan_to_num(np.where(np.isnan(Xv),mu,Xv)/sd)
    G=Zf.T@Zf+alpha*np.eye(Zf.shape[1]); w=np.linalg.solve(G,Zf.T@y[is_fit])
    return float(np.abs(y[is_pv]-Zv@w).mean())
base_cols = [c for c in A.load_saved("e005_marketing.parquet").columns if c not in ("household_key","snapshot_day")]
Xb = m0[base_cols].copy()
for c in Xb.columns:
    if not np.issubdtype(Xb[c].dtype, np.number): Xb[c]=Xb[c].astype("category").cat.codes
Xb = Xb.replace([np.inf,-np.inf],np.nan).values.astype(float)
print("base:", round(ridge_eval(Xb,y,is_fit,is_pv),2))
for Kk in [3,5,8,13]:
    sub = m0[[f"aw_{k}" for k in range(1,Kk+1)]]
    f_med = sub.median(1).values.reshape(-1,1); f_mean = sub.mean(1).values.reshape(-1,1)
    print(f"K={Kk}: med {ridge_eval(np.column_stack([Xb,f_med]),y,is_fit,is_pv):.2f} | mean {ridge_eval(np.column_stack([Xb,f_mean]),y,is_fit,is_pv):.2f} | both {ridge_eval(np.column_stack([Xb,f_med,f_mean]),y,is_fit,is_pv):.2f}")
