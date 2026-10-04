import numpy as np, pandas as pd
base = load_saved('e005_full_plus_mix.parquet')
tt = train_targets()
m = tt.merge(base, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)
bcols = [c for c in base.columns if c not in ('household_key','snapshot_day')]

def prep(df, med=None):
    X = df.copy()
    if med is None: med = X.median()
    for c in X.columns:
        col = X[c]
        if col.dtype.kind not in 'ifbu':
            col = pd.Series(pd.factorize(col)[0], index=X.index).astype(float)
        X[c] = col.astype(float).replace([np.inf,-np.inf], np.nan).fillna(med[c])
    return X.values, med
Xb, med = prep(m[bcols])

tr = np.where(m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403]).values)[0]
va = np.where((m.snapshot_day==431).values)[0]
lam=100.0
mu, sd = Xb[tr].mean(0), Xb[tr].std(0)+1e-9
Z = (Xb-mu)/sd; Zt = np.hstack([Z, np.ones((len(Z),1))])
A = Zt[tr].T@Zt[tr] + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
w = np.linalg.solve(A, Zt[tr].T@y[tr])
p = Zt@w
err = y - p
mv = m.iloc[va].copy(); mv['pred']=p[va]; mv['err']=err[va]; mv['y']=y[va]
mv['abserr']=np.abs(err[va])
print("val MAE:", mv.abserr.mean().round(2))
# error by predicted level decile
mv['pdec'] = pd.qcut(mv.pred, 10, duplicates='drop')
print(mv.groupby('pdec').agg(n=('abserr','size'), mae=('abserr','mean'), bias=('err','mean'), y=('y','mean'), pred=('pred','mean')).round(1).to_string())
# zero-y households
z = mv[mv.y==0]
print("\nzero-y rows:", len(z), "mean pred:", z.pred.mean().round(1), "MAE contribution:", z.abserr.mean().round(1))
nz = mv[mv.y>0]
print("nonzero rows:", len(nz), "mean pred:", nz.pred.mean().round(1))
# high spenders
h = mv.nlargest(10, 'y')[['y','pred','err','spend_84','spend_28','x_exp4w','recency','zero_w12']]
print("\ntop-10 y rows:\n", h.round(1).to_string())
# error vs spend_84 decile
mv['sdec'] = pd.qcut(mv.spend_84, 10, duplicates='drop')
print("\nby spend_84 decile:")
print(mv.groupby('sdec').agg(n=('abserr','size'), mae=('abserr','mean'), bias=('err','mean'), y=('y','mean'), pred=('pred','mean')).round(1).to_string())
# what correlates with |err| or err among base features
rows=[]
for c in bcols:
    x = mv[c]
    if x.dtype.kind not in 'ifb': continue
    x = x.astype(float).replace([np.inf,-np.inf],np.nan).fillna(x.median())
    if x.std()==0: continue
    rows.append((c, float(np.corrcoef(x, mv.err)[0,1]), float(np.corrcoef(x, mv.abserr)[0,1])))
ec = pd.DataFrame(rows, columns=['f','r_err','r_abserr'])
print("\ncorr with err / |err| (top 15 by |r_err|):")
print(ec.reindex(ec.r_err.abs().sort_values(ascending=False).index).head(15).round(3).to_string())
print("\ncorr with |err| (top 15):")
print(ec.reindex(ec.r_abserr.abs().sort_values(ascending=False).index).head(15).round(3).to_string())