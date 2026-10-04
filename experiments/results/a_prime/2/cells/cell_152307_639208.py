import numpy as np, pandas as pd
t = agent_api.load_saved('e008_main_plus_marketing.parquet')
tt = agent_api.train_targets()
print('feat shape', t.shape)
print('cols:', sorted(t.columns))
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df['future_spend_4w'].astype(float)
print('train rows', len(df), 'zero share %.3f' % (y==0).mean())
print(y.describe(percentiles=[.25,.5,.75,.9,.95]))
print('by snapshot:')
print(df.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median']))
num = df.select_dtypes(include=[np.number,'bool']).drop(columns=['snapshot_day','future_spend_4w'], errors='ignore').astype(float)
sp = num.corrwith(y, method='spearman').sort_values()
print('TOP corr:'); print(sp.tail(22).round(3))
print('BOTTOM:'); print(sp.head(6).round(3))

cats = [c for c in df.columns if str(df[c].dtype) not in ('float64','int64','bool','float32','int32','uint8','int8') and c not in ('household_key',)]
print('categorical cols:', cats)
parts = [num.values]
for c in cats:
    parts.append(pd.get_dummies(df[c].astype('category'), dummy_na=True).values.astype(float))
X = np.hstack(parts)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
tr_mask = (df['snapshot_day'] <= 403).values
va_mask = (df['snapshot_day'] == 431).values
mu, sd = X[tr_mask].mean(0), X[tr_mask].std(0)+1e-9
Xs = (X-mu)/sd
Xs = np.hstack([Xs, np.ones((len(Xs),1))])
def ridge_fit(Xtr, ytr, lam):
    A = Xtr.T@Xtr + lam*np.eye(Xtr.shape[1]); A[-1,-1] -= lam
    return np.linalg.solve(A, Xtr.T@ytr)
yv = y.values
for lam in [1.0, 10.0, 100.0, 1000.0]:
    w = ridge_fit(Xs[tr_mask], yv[tr_mask], lam)
    pred = np.clip(Xs[va_mask]@w, 0, None)
    print('ridge lam=%g holdout431 MAE %.2f' % (lam, np.abs(pred-yv[va_mask]).mean()))
c28 = [c for c in df.columns if '28' in c and ('spend' in c or 'tlag' in c)]
print('28d spend-like cols:', c28)
for c in c28[:3]:
    print('baseline %s MAE on 431: %.2f' % (c, np.abs(np.clip(df[c][va_mask],0,None)-yv[va_mask]).mean()))
print('mean-train MAE on 431: %.2f' % np.abs(np.full(va_mask.sum(), yv[tr_mask].mean())-yv[va_mask]).mean())
