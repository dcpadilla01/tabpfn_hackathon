import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

t = agent_api.load_saved('weekly_history.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df['future_spend_4w'].values
X = df.drop(columns=['household_key','snapshot_day','future_spend_4w'])
days = sorted(df['snapshot_day'].unique())
print('snapshots:', days)

cat = [c for c in X.columns if X[c].dtype=='object' or str(X[c].dtype)=='category']
num = [c for c in X.columns if c not in cat]
print('num:', len(num), 'cat:', len(cat))

def design(dfX):
    parts = [dfX[num].fillna(0).replace([np.inf,-np.inf],0).values.astype(float)]
    for c in cat:
        d = pd.get_dummies(dfX[c].astype(str), prefix=c)
        parts.append(d.values.astype(float))
    return np.hstack(parts)

Xa = design(X)
print('design shape:', Xa.shape)

def ridge_eval(holdout_days, lam):
    tr = ~df['snapshot_day'].isin(holdout_days).values
    va = df['snapshot_day'].isin(holdout_days).values
    mu, sd = Xa[tr].mean(0), Xa[tr].std(0)+1e-9
    Xs = np.clip((Xa-mu)/sd, -10, 10)
    Xt = np.hstack([Xs, np.ones((len(Xs),1))])
    A = Xt[tr].T@Xt[tr] + lam*np.eye(Xt.shape[1]); A[-1,-1] -= lam
    w = np.linalg.solve(A, Xt[tr].T@y[tr])
    pv = Xt[va]@w
    return np.mean(np.abs(y[va]-pv))

for hd in [[431],[403,431],[375,403,431]]:
    for lam in [300,100,30,10]:
        print(f'holdout {hd} lam={lam}: MAE {ridge_eval(hd,lam):.3f}')

# per-snapshot mean target (drift check)
print(df.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']).to_string())