import numpy as np, pandas as pd, agent_api as A

df = A.load_saved("e005_marketing.parquet")
tt = A.train_targets()
m = tt.merge(df, on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"].astype(float).values

tr_days = A.snapshot_days()["train"]
print("train rows with NaN y:", int(np.isnan(y).sum()))
PV = 431
is_pv = m["snapshot_day"]==PV
is_fit = m["snapshot_day"].isin([d for d in tr_days if d<PV])
print("fit rows", int(is_fit.sum()), "pv rows", int(is_pv.sum()))

# oracle: household median from fit snaps -> predict pv
med = m[is_fit].groupby("household_key")["future_spend_4w"].median()
pred_or = m[is_pv]["household_key"].map(med).fillna(m[is_fit]["future_spend_4w"].median()).values
print("oracle hh-median pv MAE:", round(float(np.abs(m[is_pv]["future_spend_4w"].values-pred_or).mean()),2))

feat_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
X = m[feat_cols].copy()
for c in X.columns:
    if not np.issubdtype(X[c].dtype, np.number):
        X[c] = X[c].astype("category").cat.codes
X = X.replace([np.inf,-np.inf], np.nan).values.astype(float)

def ridge_eval(X, y, is_fit, is_pv, alpha=100.0, logy=False):
    Xf, Xv = X[is_fit], X[is_pv]
    mu = np.nanmean(Xf,0); sd = np.nanstd(Xf,0)+1e-9
    Zf = np.where(np.isnan(Xf),(Xf-mu),Xf)/sd; Zf=np.nan_to_num(Zf)
    Zv = np.where(np.isnan(Xv),(Xv-mu),Xv)/sd; Zv=np.nan_to_num(Zv)
    yt = np.log1p(y) if logy else y
    d=Zf.shape[1]; G=Zf.T@Zf+alpha*np.eye(d); w=np.linalg.solve(G, Zf.T@yt[is_fit])
    p = Zv@w
    if logy: p=np.expm1(np.clip(p,0,8))
    return float(np.abs(y[is_pv]-p).mean())

for a in [10,100,1000,3000]:
    print(f"ridge a={a}: pv MAE {ridge_eval(X,y,is_fit,is_pv,a):.2f}")
print("ridge logy a=1000:", round(ridge_eval(X,y,is_fit,is_pv,1000,logy=True),2))

# single feature spend_112_mean4 on pv
f = m["spend_112_mean4"].values
p = np.nan_to_num(f)
print("spend_112_mean4 raw pv MAE:", round(float(np.abs(y[is_pv]-p[is_pv]).mean()),2))
al=np.linspace(0,2,41); maes=[np.abs(y[is_pv]-a*p[is_pv]).mean() for a in al]
i=int(np.argmin(maes)); print("scaled best alpha",round(float(al[i]),2),"MAE",round(float(maes[i]),2))
