import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

t = agent_api.load_saved('weekly_history.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df['future_spend_4w'].values
X = df.drop(columns=['household_key','snapshot_day','future_spend_4w'])

# simple predictor baselines on VALIDATION rows
val_days = agent_api.snapshot_days()['validation']
is_val = df['snapshot_day'].isin(val_days).values
print('rows:', len(df), 'val rows:', int(is_val.sum()))
print('MAE predict-0          :', np.mean(np.abs(y[is_val])))
print('MAE predict-median     :', np.mean(np.abs(y[is_val]-np.median(y[~is_val]))))
print('MAE predict-spend_84   :', np.mean(np.abs(y[is_val]-X['spend_84'].values[is_val])))
print('MAE predict-spend_28   :', np.mean(np.abs(y[is_val]-X['spend_28'].values[is_val])))
print('MAE predict-tlag_2     :', np.mean(np.abs(y[is_val]-X['tlag_2'].values[is_val])))
print('MAE predict-wmean26    :', np.mean(np.abs(y[is_val]-X['wmean26'].values[is_val])))
print('MAE predict-0.8*spend_84:', np.mean(np.abs(y[is_val]-0.8*X['spend_84'].values[is_val])))

# numpy ridge with one-hot categoricals: gauge linear-model headroom
cat = X.select_dtypes(include=['object','category','bool']).columns
num = X.select_dtypes(include=[np.number]).columns
print('num feats:', len(num), 'cat feats:', list(cat))

def design(dfX):
    parts = [dfX[num].fillna(0).replace([np.inf,-np.inf],0).values.astype(float)]
    for c in cat:
        d = pd.get_dummies(dfX[c].fillna('NA'), prefix=c, dummy_na=False)
        parts.append(d.values.astype(float))
    return np.hstack(parts)

Xa = design(X)
tr = ~is_val
mu, sd = Xa[tr].mean(0), Xa[tr].std(0)+1e-9
Xs = (Xa-mu)/sd
Xs = np.clip(Xs, -10, 10)
Xt = np.hstack([Xs, np.ones((len(Xs),1))])
def ridge_fit(Xt, y, lam):
    A = Xt.T@Xt + lam*np.eye(Xt.shape[1]); A[-1,-1] -= lam
    return np.linalg.solve(A, Xt.T@y)
for lam in [100, 30, 10, 3, 1]:
    w = ridge_fit(Xt[tr], y[tr], lam)
    pv = Xt[is_val]@w
    print(f'ridge lam={lam}: val MAE {np.mean(np.abs(y[is_val]-pv)):.3f}  R2 {1-np.sum((y[is_val]-pv)**2)/np.sum((y[is_val]-y[tr].mean())**2):.4f}')