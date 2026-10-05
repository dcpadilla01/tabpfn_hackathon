import numpy as np, pandas as pd
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
df = T.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype(float)
def mae(p): return float(np.mean(np.abs(np.asarray(p)-y)))

# scale-adjusted candidates
cands = {
 'sp28': df.spend_28.values,
 '0.9*sp28': 0.9*df.spend_28.values,
 's84/3': df.spend_84.values/3,
 's182/6.5': df.spend_182.values/6.5,
 's365/13': df.spend_365.values/13,
 'avg28_all': df.avg28_all.values,
 'blend .5(s84/3)+.3s28+.2avg': 0.5*df.spend_84.values/3+0.3*df.spend_28.values+0.2*df.avg28_all.values,
 'blend .4(s84/3)+.3s28+.3avg': 0.4*df.spend_84.values/3+0.3*df.spend_28.values+0.3*df.avg28_all.values,
 'max(s28, s84/3)': np.maximum(df.spend_28.values, df.spend_84.values/3),
 'ewma_hl28': df.d_ewma_spend_hl28.values,
 '0.5*ewma28+0.5*avg28': 0.5*df.d_ewma_spend_hl28.values+0.5*df.avg28_all.values,
}
for k,v in cands.items(): print(f"{k:30s} MAE {mae(v):8.2f}")

feats = ['spend_28','spend_84','spend_182','spend_365','avg28_all','d_ewma_spend_hl28','d_ewma_spend_hl112',
         'trips_28','trips_84','recency','tenure','blk_1','d_wk_spend_mean_12w','basket_mean_84','zero6','decay_mean']
X = df[feats].fillna(0).values.astype(float)
X[:,1] = X[:,1]/3.0; X[:,2]=X[:,2]/6.5; X[:,3]=X[:,3]/13.0  # scale-adjust
X = np.hstack([X, np.ones((len(X),1))])
def ridge_fit(Xtr, ytr, alpha):
    mu, sd = Xtr[:,:-1].mean(0), Xtr[:,:-1].std(0); sd[sd==0]=1
    Z = (Xtr[:,:-1]-mu)/sd
    Z1 = np.hstack([Z, np.ones((len(Z),1))])
    A = Z1.T@Z1 + alpha*np.eye(Z1.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Z1.T@ytr)
    return mu, sd, w
def ridge_pred(Xte, m):
    mu, sd, w = m
    Z = (Xte[:,:-1]-mu)/sd
    return np.hstack([Z, np.ones((len(Z),1))])@w
d = df.snapshot_day.values.astype(int)
tr, te = d<=403, d==431
for alpha in [1,10,100,1000]:
    m = ridge_fit(X[tr], y[tr], alpha)
    print(f"ridge alpha={alpha}: holdout(431) MAE {mae(ridge_pred(X[te],m)):.2f}")
m = ridge_fit(X, y, 100)
print("ridge alpha=100 in-sample MAE:", mae(ridge_pred(X,m)))
# day-holdout for simple candidates too
for k in ['sp28','0.9*sp28','avg28_all','blend .5(s84/3)+.3s28+.2avg']:
    v = cands[k]
    print(f"{k:30s} holdout(431) MAE {mae(v[te]):.2f}")
