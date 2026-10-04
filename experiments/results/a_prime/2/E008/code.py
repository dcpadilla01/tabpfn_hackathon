import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('weekly_history.parquet')
print('E007 table:', t.shape)
cols = list(t.columns)
for i in range(0, len(cols), 10):
    print('|', ', '.join(cols[i:i+10]))

tt = agent_api.train_targets()
print('\ntargets:', tt.shape, tt.columns.tolist())
print(tt['future_spend_4w'].describe())
print('zero frac:', float((tt['future_spend_4w']==0).mean()))

m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
num = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).select_dtypes(include=[np.number])
cor = num.corrwith(m['future_spend_4w']).abs().sort_values(ascending=False)
print('\ntop-30 |corr|:')
print(cor.head(30).to_string())
print('features with |corr|<0.01:', int((cor<0.01).sum()), '/', len(cor))

v = agent_api.snapshot()
print('\nview attrs:', [a for a in dir(v) if not a.startswith('_')])
hh = v.households
print('households type:', str(pd.Series([hh]).dtype), hh.shape if hasattr(hh,'shape') else '')
print('cols', list(hh.columns)[:6])
print('index name:', hh.index.name)
print(hh.head(3))
print('day', v.day, 'week', v.week)
tx = v.transactions
print('tx', tx.shape, list(tx.columns))
print('has products:', hasattr(v, 'products'))

# ---- cell ----
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

# ---- cell ----
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

# ---- cell ----
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

# ---- cell ----
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

# ---- cell ----
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

# ---- cell ----
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

# ---- cell ----
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
CAT = [c for c in X.columns if X[c].dtype=='object' or str(X[c].dtype)=='category']
NUM = [c for c in X.columns if c not in CAT]

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

spendlike = [c for c in NUM if c.startswith(('spend_','tlag_','ws_','wblk','ts_sw','ts_sum','lifetime','d28_','wmean','disc_'))]
Xlog = X.copy()
for c in spendlike:
    Xlog['log_'+c] = np.log1p(X[c].fillna(0).clip(lower=0).values)
m,_ = ridge_mae(Xlog); print('G1 +log1p spend-like:', round(m,3), 'nfeat', Xlog.shape[1])

Xn = X[NUM]
corv = {c: abs(np.corrcoef(np.nan_to_num(Xn[c].values), y)[0,1]) for c in Xn.columns if np.nanstd(Xn[c].values)>0}
top = sorted(list(corv), key=lambda c: -corv[c])[:12]
Xint = X.copy()
for i in range(len(top)):
    for j in range(i+1, len(top)):
        Xint[f'i_{top[i]}_{top[j]}'] = Xn[top[i]].fillna(0).values*Xn[top[j]].fillna(0).values
m,_ = ridge_mae(Xint); print('G3 +top12 products:', round(m,3), 'nfeat', Xint.shape[1])

Xcal = X.copy()
for d in sorted(df['snapshot_day'].unique()):
    Xcal[f'cal_{int(d)}'] = (df['snapshot_day']==d).values.astype(float)
m,_ = ridge_mae(Xcal); print('G2 +snapshot dummies:', round(m,3), 'nfeat', Xcal.shape[1])

Xsq = X.copy()
for c in spendlike:
    Xsq['sqrt_'+c] = np.sqrt(X[c].fillna(0).clip(lower=0).values)
m,_ = ridge_mae(Xsq); print('G4 +sqrt spend-like:', round(m,3), 'nfeat', Xsq.shape[1])

m, pv = ridge_mae(X)
resid = y[va]-pv
print('resid: mean', round(resid.mean(),2), 'skew', round(pd.Series(resid).skew(),2), 'q', np.percentile(resid,[5,25,50,75,95]).round(0))
print('corr(pv,y):', round(np.corrcoef(pv,y[va])[0,1],3))

# ---- cell ----
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

# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
v = agent_api.snapshot()
print('campaigns:', v.campaigns.shape); print(v.campaigns.head(3))
print('desc values:', v.campaigns['description'].value_counts().to_string())
print('campaign_targets:', v.campaign_targets.shape); print(v.campaign_targets.head(3))
print('coupon_redemptions:', v.coupon_redemptions.shape)
dm = v.display_mailer
print('display_mailer:', dm.shape)
print(dm.head(3))
print('display vals:', dm['display'].value_counts().head(8).to_string())
print('mailer vals:', dm['mailer'].value_counts().head(8).to_string())
tx = v.transactions
print('tx up to 459:', tx.shape, 'max day', tx['day'].max())
prod = v.products[['product_id','department']].drop_duplicates('product_id')
ds = tx.merge(prod, on='product_id', how='left').groupby('department')['sales_value'].sum().sort_values(ascending=False)
print('top depts:', list(ds.head(16).index))

# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
base = agent_api.load_saved('weekly_history.parquet')
mk = agent_api.load_saved('marketing_exposure.parquet')
print('base', base.shape, 'mkt', mk.shape)
mkt_cols = [c for c in mk.columns if c not in ('household_key','snapshot_day')]
print('mkt cols:', mkt_cols)
m = base.merge(mk, on=['household_key','snapshot_day'], how='left', suffixes=('','_mkt'))
m = m.loc[:, ~m.columns.duplicated()]
print('merged:', m.shape)
print('NaN rate in mkt cols:', round(float(m[mkt_cols].isna().mean().mean()),3))
p = agent_api.save_table(m, 'e008_main_plus_marketing')
print('saved:', p)