import numpy as np, pandas as pd, agent_api as A

df = A.load_saved("e005_marketing.parquet")
df4 = A.load_saved("e004_seasonal.parquet")
tt = A.train_targets()
m = tt.merge(df, on=["household_key","snapshot_day"], how="left")
m = m.merge(df4[["household_key","snapshot_day"]+[c for c in df4.columns if "lag" in c]], on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"].astype(float).values
tr_days = A.snapshot_days()["train"]; PV=431
is_pv = m["snapshot_day"]==PV
is_fit = m["snapshot_day"].isin([d for d in tr_days if d<PV])
yv = y[is_pv]

lag_cols = [c for c in m.columns if c.startswith("spend_4w_lag")]
print("lag cols:", lag_cols)
L = m[lag_cols].replace([np.inf,-np.inf],np.nan)
n_avail = L.notna().sum(1)
med_lags = L.median(1)          # median of household's own prior 4w spends
mean_lags = L.mean(1)
max_lags = L.max(1)
last_lag = L[lag_cols[0]]       # most recent lag (t-28..t]

def ridge_eval(X, y, is_fit, is_pv, alpha=3000):
    Xf,Xv=X[is_fit],X[is_pv]
    mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0)+1e-9
    Zf=np.nan_to_num(np.where(np.isnan(Xf),mu,Xf)/sd); Zv=np.nan_to_num(np.where(np.isnan(Xv),mu,Xv)/sd)
    G=Zf.T@Zf+alpha*np.eye(Zf.shape[1]); w=np.linalg.solve(G,Zf.T@y[is_fit])
    return float(np.abs(y[is_pv]-Zv@w).mean()), Zv@w

base_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
Xb = m[base_cols].copy()
for c in Xb.columns:
    if not np.issubdtype(Xb.columns.dtype if False else Xb[c].dtype, np.number): Xb[c]=Xb[c].astype("category").cat.codes
Xb = Xb.replace([np.inf,-np.inf],np.nan).values.astype(float)

def test(name, extra):
    X = np.column_stack([Xb, extra]) if extra is not None else Xb
    mae,_ = ridge_eval(X, y, is_fit, is_pv, 3000)
    print(f"{name}: pv MAE {mae:.2f}")

test("E005 base", None)
test("+med_lags", med_lags.values.reshape(-1,1))
test("+mean_lags", mean_lags.values.reshape(-1,1))
test("+med+mean+max+last", np.column_stack([med_lags,mean_lags,max_lags,last_lag]))
test("+med+last+n_avail", np.column_stack([med_lags,last_lag,n_avail]))

# log transforms of all spend-ish columns
spendish = [c for c in base_cols if ("spend" in c or c.startswith("sum_")) ]
Xlog = Xb.copy()
idx = [base_cols.index(c) for c in spendish]
Xlog[:, idx] = np.log1p(np.clip(Xlog[:, idx], 0, None))
test("+log(spend cols)", Xlog-Xb if False else np.zeros((len(Xlog),0)))  # replace base
def test2(name, X):
    mae,_ = ridge_eval(X, y, is_fit, is_pv, 3000)
    print(f"{name}: pv MAE {mae:.2f}")
test2("base with log-spend", Xlog)
Xlog2 = np.column_stack([Xlog, med_lags, last_lag])
test2("log-spend + med_lags + last_lag", Xlog2)

# ratios: recent vs baseline
r1 = (m["spend_4w"]/m["spend_28w"].replace(0,np.nan)).fillna(1).values.reshape(-1,1)
r2 = (m["spend_4w"]/m["spend_112w"].replace(0,np.nan)).fillna(1).values.reshape(-1,1)
test2("log-spend + med/last lags + ratios", np.column_stack([Xlog2, r1, r2]))
