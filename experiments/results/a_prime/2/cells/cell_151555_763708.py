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
print('bool cols:', [c for c in num if X[c].dtype==bool])

def ridge_mae(A, lam=300):
    mu, sd = A[tr].mean(0), A[tr].std(0)+1e-9
    Xs = np.clip((A-mu)/sd, -10, 10)
    Xt = np.hstack([Xs, np.ones((len(Xs),1))])
    M = Xt.T@Xt + lam*np.eye(Xt.shape[1]); M[-1,-1]-=lam
    w = np.linalg.solve(M, Xt[tr].T@y[tr])
    pv = Xt[va]@w
    return np.mean(np.abs(y[va]-pv)), pv

def design(dfX):
    parts = [dfX[num].fillna(0).replace([np.inf,-np.inf],0).values.astype(float)]
    for c in cat:
        d = pd.get_dummies(dfX[c].astype(str), prefix=c)
        parts.append(d.values.astype(float))
    return np.hstack(parts)

A_full = design(X)
A_num  = X[num].fillna(0).replace([np.inf,-np.inf],0).values.astype(float)
print('full+dummies :', round(ridge_mae(A_full)[0],3))
print('numeric only :', round(ridge_mae(A_num)[0],3))

# diagnose numeric-only: check pv vs y
mae, pv = ridge_mae(A_num)
print('corr(pv,y_val):', round(np.corrcoef(pv, y[va])[0,1],3), 'pv mean', round(pv.mean(),1), 'y mean', round(y[va].mean(),1))
print('pv describe:', np.percentile(pv,[5,25,50,75,95]).round(1))

# check per-column: drop 'index' col?
print('index col stats:', X['index'].describe().to_string() if 'index' in X.columns else 'none')

# maybe inf values present?
print('num cols with inf:', [c for c in num if np.isinf(X[c].fillna(0).values).any()][:10])
print('num cols all-NaN in train:', [c for c in num if pd.isna(X[c].values[tr]).all()][:10])

# try numeric-only but EXCLUDING bool and 'index'
num2 = [c for c in num if X[c].dtype!=bool and c!='index']
print('numeric excl bool/index:', round(ridge_mae(X[num2].fillna(0).replace([np.inf,-np.inf],0).values.astype(float))[0],3))