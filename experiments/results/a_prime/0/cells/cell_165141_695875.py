import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

t = A.load_saved('e013_stock.parquet')
tt = A.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df['future_spend_4w'].values
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].copy()
for c in feat_cols:
    if X[c].dtype == object or str(X[c].dtype)=='category':
        X[c] = X[c].astype('category').cat.codes.astype(float)
Xv = X.values.astype(float)
med = np.nanmedian(Xv, axis=0)
Xv = np.where(np.isnan(Xv), med, Xv)
mu, sd = Xv.mean(0), Xv.std(0)+1e-9
Xs = (Xv-mu)/sd

def fit(Xs, y, lam):
    n = len(Xs); Xd = np.hstack([np.ones((n,1)), Xs])
    M = Xd.T@Xd + lam*np.eye(Xd.shape[1]); M[0,0] -= lam
    return np.linalg.solve(M, Xd.T@y)
def pred(w, Xs):
    return np.hstack([np.ones((len(Xs),1)), Xs])@w

fit_mask = df.snapshot_day<=375; int_mask = df.snapshot_day==403
best = None
for lam in [1,3,10,30,100,300]:
    w = fit(Xs[fit_mask], y[fit_mask], lam)
    mae = np.abs(pred(w, Xs[int_mask]) - y[int_mask]).mean()
    if best is None or mae < best[1]: best = (lam, round(mae,3))
print('internal lambda/MAE@403:', best)
lam = best[0]
tr = df.snapshot_day<=403
w = fit(Xs[tr], y[tr], lam)
m431 = df.snapshot_day==431
p = pred(w, Xs[m431]); yy = y[m431]
print('pseudo-val MAE@431:', round(np.abs(p-yy).mean(),3), 'n=', m431.sum())
print('baseline spend_28:', round(np.abs(df.loc[m431,'spend_28'].values-yy).mean(),3),
      '| x_exp4w:', round(np.abs(df.loc[m431,'x_exp4w'].values-yy).mean(),3),
      '| zero:', round(np.abs(yy).mean(),3))
r = pd.DataFrame({'y':yy,'p':p}); r['ae']=abs(r.p-r.y); r['bias']=r.p-r.y
r['bin'] = pd.qcut(r.y, 10, duplicates='drop')
print(r.groupby('bin', observed=True).agg(n=('y','size'), ymean=('y','mean'), pmean=('p','mean'), mae=('ae','mean'), bias=('bias','mean')).round(1))
z = r.y==0
print('zero-true rows:', z.sum(), 'mean pred on them:', round(r.loc[z,'p'].mean(),2), 'MAE there:', round(r.loc[z,'ae'].mean(),2))
nz = ~z
print('nonzero rows MAE:', round(r.loc[nz,'ae'].mean(),2))
top = r.nlargest(15,'ae')
cols = ['spend_28','spend_84','spend_364','recency','active_28','rs_active_share_84','stk_w1_share','x_exp4w']
print(pd.concat([top, df.loc[m431, cols].loc[top.index]], axis=1).round(1))
