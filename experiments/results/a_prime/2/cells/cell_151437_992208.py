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

def ridge_mae(Xdf, lam=300):
    parts = [Xdf[num_].fillna(0).replace([np.inf,-np.inf],0).values.astype(float) for num_ in [ [c for c in Xdf.columns if Xdf[c].dtype!='object' and str(Xdf[c].dtype)!='category'] ]][0]
    # simpler: use provided numeric frame directly
    A = Xdf.fillna(0).replace([np.inf,-np.inf],0).values.astype(float)
    mu, sd = A[tr].mean(0), A[tr].std(0)+1e-9
    Xs = np.clip((A-mu)/sd, -10, 10)
    Xt = np.hstack([Xs, np.ones((len(Xs),1))])
    M = Xt.T@Xt; M = M + lam*np.eye(M.shape[0]); M[-1,-1]-=lam
    w = np.linalg.solve(M, Xt[tr].T@y[tr])
    pv = Xt[va]@w
    return np.mean(np.abs(y[va]-pv))

# numeric-only base (drop cats for speed of screening)
Xn = X[num].copy()
print('base numeric-only ridge MAE:', round(ridge_mae(Xn),3))

# simple predictor baselines on holdout
for c in ['spend_84','spend_28','tlag_2','wmean26','tlag_mean']:
    print(f'predict {c:10s}: MAE {np.mean(np.abs(y[va]-X[c].values[va])):.2f}')
print('blend 0.5*spend_84+0.5*tlag_2:', round(np.mean(np.abs(y[va]-0.5*X['spend_84'].values[va]-0.5*X['tlag_2'].values[va])),2))

# G1: log1p transforms of all numeric features
Xlog = Xn.copy()
for c in num:
    v = Xn[c].fillna(0).values
    if (v>0).mean() > 0.5 and v.min() >= 0:
        Xlog['log_'+c] = np.log1p(v)
print('G1 +log1p all:', round(ridge_mae(Xlog),3), 'nfeat', Xlog.shape[1])

# G1b: log1p only of spend-like cols
spendlike = [c for c in num if c.startswith(('spend_','tlag_','ws_','wblk','ts_sw','ts_sum','lifetime','d28_','wmean'))]
Xlog2 = Xn.copy()
for c in spendlike:
    Xlog2['log_'+c] = np.log1p(Xn[c].fillna(0).clip(lower=0).values)
print('G1b +log spend-like:', round(ridge_mae(Xlog2),3), 'nfeat', Xlog2.shape[1])

# G3: pairwise products of top-12 |corr| features
cor = pd.Series(Xn.columns, index=Xn.columns).to_dict()
corv = {c: abs(np.corrcoef(np.nan_to_num(Xn[c].values), y)[0,1]) if np.nanstd(Xn[c].values)>0 else 0 for c in Xn.columns}
top = sorted(Xn.columns, key=lambda c: -corv[c])[:12]
Xint = Xn.copy()
for i in range(len(top)):
    for j in range(i+1, len(top)):
        a, b = Xn[top[i]].fillna(0).values, Xn[top[j]].fillna(0).values
        Xint[f'i_{top[i]}_{top[j]}'] = a*b
print('G3 +top12 interactions:', round(ridge_mae(Xint),3), 'nfeat', Xint.shape[1])