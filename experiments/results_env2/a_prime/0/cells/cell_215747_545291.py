import numpy as np, pandas as pd, agent_api as A

df = A.load_saved("e005_marketing.parquet")
tt = A.train_targets()
m = tt.merge(df, on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"].astype(float).values
tr_days = A.snapshot_days()["train"]; PV=431
is_pv = m["snapshot_day"]==PV
is_fit = m["snapshot_day"].isin([d for d in tr_days if d<PV])

feat_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
X = m[feat_cols].copy()
for c in X.columns:
    if not np.issubdtype(X[c].dtype, np.number): X[c]=X[c].astype("category").cat.codes
X = X.replace([np.inf,-np.inf],np.nan).values.astype(float)

def ridge_pred(X,y,is_fit,is_pv,alpha=3000):
    Xf,Xv=X[is_fit],X[is_pv]
    mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0)+1e-9
    Zf=np.nan_to_num(np.where(np.isnan(Xf),mu,Xf)/sd); Zv=np.nan_to_num(np.where(np.isnan(Xv),mu,Xv)/sd)
    G=Zf.T@Zf+alpha*np.eye(Zf.shape[1]); w=np.linalg.solve(G,Zf.T@y[is_fit])
    return Zv@w, Zf@w

p_pv, p_fit = ridge_pred(X,y,is_fit,is_pv,3000)
yv = y[is_pv]
mae0 = np.abs(yv-p_pv).mean()
print("proxy ridge MAE:", round(mae0,2))

# error by predicted-spend decile
q = pd.qcut(p_fit, 10, duplicates="drop")
err_fit = pd.DataFrame({"q":q,"y":y[is_fit],"p":p_fit})
print("\nfit-set: mean |err| and mean bias by predicted decile:")
g = err_fit.groupby("q",observed=True).apply(lambda d: pd.Series({"mae":np.abs(d.y-d.p).mean(),"bias":(d.p-d.y).mean(),"n":len(d),"ybar":d.y.mean(),"pbar":d.p.mean()}))
print(g.round(1))

# calibration: fit isotonic-like piecewise linear via binning predicted -> mean y on fit set
bins = pd.qcut(p_fit, 50, duplicates="drop")
cal = pd.DataFrame({"b":bins,"y":y[is_fit],"p":p_fit}).groupby("b",observed=True).agg(pmid=("p","mean"),ybar=("y","mean"))
pc = np.interp(p_pv, cal["pmid"].values, cal["ybar"].values)
print("\nbin-calibrated MAE:", round(float(np.abs(yv-pc).mean()),2))

# blend with household median feature
med_hh = m[is_fit].groupby("household_key")["future_spend_4w"].median()
f_med = m["household_key"].map(med_hh).fillna(m[is_fit]["future_spend_4w"].median()).values
for wgt in [0,0.25,0.5,0.75,1.0]:
    pb = (1-wgt)*p_pv + wgt*f_med[is_pv]
    print(f"blend ridge/{wgt} hhmed: pv MAE {np.abs(yv-pb).mean():.2f}")

# simple power scaling of predictions: p^gamma style (p -> a*p + b*p^2?) try p*clip
for g in [0.8,0.9,1.0,1.1,1.2]:
    ps = np.sign(p_pv)*np.abs(p_pv)**g
    print(f"power {g}: MAE {np.abs(yv-ps).mean():.2f}")
