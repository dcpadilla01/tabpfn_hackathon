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

def design(dfX):
    cat = [c for c in dfX.columns if dfX[c].dtype=='object' or str(dfX[c].dtype)=='category']
    num = [c for c in dfX.columns if c not in cat]
    parts = [dfX[num].fillna(0).replace([np.inf,-np.inf],0).values.astype(float)]
    for c in cat:
        parts.append(pd.get_dummies(dfX[c].astype(str), prefix=c).values.astype(float))
    return np.hstack(parts)

def ridge_mae(Xdf, lam=300):
    A = design(Xdf)
    mu, sd = A[tr].mean(0), A[tr].std(0)+1e-9
    Xs = np.clip((A-mu)/sd, -10, 10)
    Xt = np.hstack([Xs, np.ones((len(Xs),1))])
    G = Xt[tr].T@Xt[tr] + lam*np.eye(Xt.shape[1]); G[-1,-1]-=lam
    w = np.linalg.solve(G, Xt[tr].T@y[tr])
    pv = Xt[va]@w
    return np.mean(np.abs(y[va]-pv)), pv

print('BASE:', round(ridge_mae(X)[0],3))

# G1: log1p of spend-like
Xlog = X.copy()
spendlike = [c for c in X.columns if X[c].dtype!=bool and c not in cat and c.startswith(('spend_','tlag_','ws_','wblk','ts_sw','ts_sum','lifetime','d28_','wmean','disc_'))]
for c in spendlike:
    Xlog['log_'+c] = np.log1p(X[c].fillna(0).clip(lower=0).values)
m,_ = ridge_mae(Xlog); print('G1 +log1p spend-like:', round(m,3), 'nfeat', Xlog.shape[1])

# G3: top-12 pairwise products
Xn = X.select_dtypes(include=[np.number])
corv = {c: abs(np.corrcoef(np.nan_to_num(Xn[c].values), y)[0,1]) for c in Xn.columns if np.nanstd(Xn[c].values)>0}
top = sorted(list(corv), key=lambda c: -corv[c])[:12]
Xint = X.copy()
for i in range(len(top)):
    for j in range(i+1, len(top)):
        Xint[f'i_{top[i]}_{top[j]}'] = Xn[top[i]].fillna(0).values*Xn[top[j]].fillna(0).values
m,_ = ridge_mae(Xint); print('G3 +top12 products:', round(m,3), 'nfeat', Xint.shape[1])

# G2: snapshot dummies
Xcal = X.copy()
for d in sorted(df['snapshot_day'].unique()):
    Xcal[f'cal_{int(d)}'] = (df['snapshot_day']==d).values.astype(float)
m,_ = ridge_mae(Xcal); print('G2 +snapshot dummies:', round(m,3), 'nfeat', Xcal.shape[1])

# G4: sqrt transforms of spend-like
Xsq = X.copy()
for c in spendlike:
    Xsq['sqrt_'+c] = np.sqrt(X[c].fillna(0).clip(lower=0).values)
m,_ = ridge_mae(Xsq); print('G4 +sqrt spend-like:', round(m,3), 'nfeat', Xsq.shape[1])

# G5: log target? can't change target. Check residual skew of base model
m, pv = ridge_mae(X)
resid = y[va]-pv
print('resid: mean', round(resid.mean(),2), 'skew', round(pd.Series(resid).skew(),2), 'q', np.percentile(resid,[5,25,50,75,95]).round(0))