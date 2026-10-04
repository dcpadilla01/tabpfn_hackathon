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

# W1: winsorize top-|corr| numeric features at 99.5th pct of train
Xn = X.select_dtypes(include=[np.number])
corv = {c: abs(np.corrcoef(np.nan_to_num(Xn[c].values), y)[0,1]) for c in Xn.columns if np.nanstd(Xn[c].values)>0}
top = sorted(list(corv), key=lambda c: -corv[c])[:40]
Xw = X.copy()
for c in top:
    hi = np.nanquantile(X[c].values[tr], 0.995)
    Xw[c] = X[c].clip(upper=hi)
m,_ = ridge_mae(Xw); print('W1 winsorize top40 @99.5pct:', round(m,3))

# W2: winsorize ALL numeric at 99.5 pct
Xw2 = X.copy()
for c in Xn.columns:
    if Xn[c].dtype!=bool:
        hi = np.nanquantile(X[c].values[tr], 0.995)
        Xw2[c] = X[c].clip(upper=hi)
m,_ = ridge_mae(Xw2); print('W2 winsorize all @99.5pct:', round(m,3))

# R1: momentum ratios
Xr = X.copy()
def ratio(a, b, name):
    den = X[b].fillna(0).values
    Xr[name] = np.where(den>1e-9, X[a].fillna(0).values/np.maximum(den,1e-9), np.nan)
ratio('spend_28','spend_84','r_28_84')
ratio('spend_28','spend_112','r_28_112')
ratio('tlag_2','tlag_4','r_t2_t4')
ratio('tlag_2','tlag_6','r_t2_t6')
ratio('trips_28','trips_84','r_trips')
ratio('ts_sum4','ts_sum8','r_s4_s8')
ratio('wblk_1','wblk_4','r_w1_w4')
Xr['r_28_84'] = Xr['r_28_84'].clip(0,3); Xr['r_28_112'] = Xr['r_28_112'].clip(0,3)
Xr['r_t2_t4'] = Xr['r_t2_t4'].clip(0,3); Xr['r_t2_t6'] = Xr['r_t2_t6'].clip(0,3)
Xr['r_trips'] = Xr['r_trips'].clip(0,3); Xr['r_s4_s8'] = Xr['r_s4_s8'].clip(0,3)
Xr['r_w1_w4'] = Xr['r_w1_w4'].clip(0,3)
m,_ = ridge_mae(Xr); print('R1 +momentum ratios:', round(m,3))

# P1: drop 31 near-zero-corr features
low = [c for c,s in corv.items() if s<0.01]
Xp = X.drop(columns=low)
m,_ = ridge_mae(Xp); print('P1 drop low-corr feats:', round(m,3), 'dropped', len(low))

# W1+R1 combined
Xwr = Xw.copy()
for name in ['r_28_84','r_28_112','r_t2_t4','r_t2_t6','r_trips','r_s4_s8','r_w1_w4']:
    Xwr[name] = Xr[name]
m,_ = ridge_mae(Xwr); print('W1+R1:', round(m,3))