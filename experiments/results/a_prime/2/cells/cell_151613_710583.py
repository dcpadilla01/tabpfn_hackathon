import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

t = agent_api.load_saved('weekly_history.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df['future_spend_4w'].values
X = df.drop(columns=['household_key','snapshot_day','future_spend_4w'])
HOLD = [375,403,431]
tr = ~df['snapshot_day'].isin(HOLD).values
va = df['snapshot_day'].isin(HOLD).values
cat = [c for c in X.columns if X[c].dtype=='object' or str(X[c].dtype)=='category']
num = [c for c in X.columns if c not in cat]

def design(dfX):
    parts = [dfX[num].fillna(0).replace([np.inf,-np.inf],0).values.astype(float)]
    for c in cat:
        parts.append(pd.get_dummies(dfX[c].astype(str), prefix=c).values.astype(float))
    return np.hstack(parts)

def ridge_mae(A, lam=300):
    mu, sd = A[tr].mean(0), A[tr].std(0)+1e-9
    Xs = np.clip((A-mu)/sd, -10, 10)
    Xt = np.hstack([Xs, np.ones((len(Xs),1))])
    G = Xt[tr].T@Xt[tr] + lam*np.eye(Xt.shape[1]); G[-1,-1]-=lam
    w = np.linalg.solve(G, Xt[tr].T@y[tr])
    pv = Xt[va]@w
    return np.mean(np.abs(y[va]-pv)), pv

A_full = design(X)
base_mae, pv = ridge_mae(A_full)
print('BASE full design (train-only Gram):', round(base_mae,3), 'pv mean', round(pv.mean(),1))

# simple predictor baselines on holdout
for c in ['spend_84','spend_28','tlag_2','wmean26','tlag_mean','ts_p1']:
    print(f'  predict {c:10s}: MAE {np.mean(np.abs(y[va]-X[c].values[va])):.2f}')

# G1: log1p transforms of spend-like numeric cols
Xlog = X.copy()
spendlike = [c for c in num if c.startswith(('spend_','tlag_','ws_','wblk','ts_sw','ts_sum','lifetime','d28_','wmean','disc_'))]
for c in spendlike:
    Xlog['log_'+c] = np.log1p(X[c].fillna(0).clip(lower=0).values)
m,_ = ridge_mae(design(Xlog)); print('G1 +log1p spend-like:', round(m,3), 'nfeat', Xlog.shape[1]-3)

# G3: pairwise products of top-12 |corr|
Xn = X[num]
corv = {c: abs(np.corrcoef(np.nan_to_num(Xn[c].values), y)[0,1]) for c in Xn.columns if np.nanstd(Xn[c].values)>0}
top = sorted(list(corv), key=lambda c: -corv[c])[:12]
Xint = X.copy()
for i in range(len(top)):
    for j in range(i+1, len(top)):
        Xint[f'i_{top[i]}_{top[j]}'] = Xn[top[i]].fillna(0).values*Xn[top[j]].fillna(0).values
m,_ = ridge_mae(design(Xint)); print('G3 +top12 pairwise products:', round(m,3), 'nfeat', Xint.shape[1]-3)

# G2: snapshot-day one-hots (calendar dummies)
Xcal = X.copy()
for d in sorted(df['snapshot_day'].unique()):
    Xcal[f'cal_{int(d)}'] = (df['snapshot_day']==d).values.astype(float)
m,_ = ridge_mae(design(Xcal)); print('G2 +snapshot-day dummies:', round(m,3), 'nfeat', Xcal.shape[1]-3)