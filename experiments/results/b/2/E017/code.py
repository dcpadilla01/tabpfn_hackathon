import agent_api as A
import pandas as pd, numpy as np

for name in ['e014_gbm_oob.parquet', 'e014_stack.parquet', 'e013_denoise.parquet']:
    df = A.load_saved(name)
    print('==', name, df.shape)
    print(df.columns.tolist())
    print(df.head(3))
    print()


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

s = A.load_saved('e014_stack.parquet')
print(s.columns.tolist()[-8:])
print(s[['household_key','snapshot_day', s.columns[-1]]].head())
print(s[s.columns[-1]].describe())

g = A.load_saved('e014_gbm_oob.parquet')
print(g.columns.tolist()[-3:])
print(g['gbm_corr'].describe())


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

base = A.load_saved('e013_denoise.parquet')
stack = A.load_saved('e014_stack.parquet')
tt = A.train_targets()

df = stack.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('rows', len(df), 'snaps', sorted(df.snapshot_day.unique()))
y = df['future_spend_4w'].values.astype(float)
print('target mean/med', y.mean(), np.median(y))

feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feats].astype(float)
X = X.fillna(X.median())
# standardize
mu, sd = X.mean(), X.std().replace(0,1)
Xs = ((X-mu)/sd).values

days = sorted(df.snapshot_day.unique())
def cv_mae(Xmat, alpha=100.0):
    from numpy.linalg import solve
    maes=[]
    for d in days:
        tr = df.snapshot_day.values != d
        va = ~tr
        Xt, yt = Xmat[tr], y[tr]
        A_ = Xt.T@Xt + alpha*np.eye(Xt.shape[1])
        b = np.linalg.solve(A_, Xt.T@yt)
        pred = Xmat[va]@b
        maes.append(np.abs(pred-y[va]).mean())
    return np.mean(maes)

print('ridge CV MAE base(180):', cv_mae(Xs))
Xg = np.column_stack([Xs, df['gbm_pred'].values])
print('ridge CV MAE +gbm_pred:', cv_mae(Xg))
Xg2 = np.column_stack([Xs, df['gbm_pred'].values, np.log1p(df['gbm_pred'].values)])
print('ridge CV MAE +gbm_pred+log:', cv_mae(Xg2))
print('gbm-only CV MAE:', cv_mai if False else cv_mae(np.column_stack([np.log1p(df['gbm_pred'].values)])))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

stack = A.load_saved('e014_stack.parquet')
tt = A.train_targets()
df = stack.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df['future_spend_4w'].values.astype(float)
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w','gbm_corr')]
X = df[feats].astype(float)
X = X.fillna(X.median())
mu, sd = X.mean(), X.std().replace(0,1)
Xs = ((X-mu)/sd).values
days = sorted(df.snapshot_day.unique())

def cv_mae(Xmat, alpha=1000.0):
    maes=[]
    for d in days:
        tr = df.snapshot_day.values != d
        va = ~tr
        Xt, yt = Xmat[tr], y[tr]
        U,S,Vt = np.linalg.svd(Xt, full_matrices=False)
        Sinv = S/(S**2+alpha)
        b = Vt.T@(Sinv*(U.T@yt))
        pred = Xmat[va]@b
        maes.append(np.abs(pred-y[va]).mean())
    return np.mean(maes)

print('base180 ridge(1000):', round(cv_mae(Xs),3))
print('base180+gbm ridge(1000):', round(cv_mae(np.column_stack([Xs, df['gbm_pred'].values])),3))
# also check per-validation-snap performance with leave-one-snap-out on train only
print('base180+gbm+loggbm ridge(1000):', round(cv_mae(np.column_stack([Xs, df['gbm_pred'].values, np.log1p(df['gbm_pred'].values)])),3))
# scaled gbm variants
gp = df['gbm_pred'].values
print('base180+gbm/100 ridge(1000):', round(cv_mae(np.column_stack([Xs, gp/100])),3))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

stack = A.load_saved('e014_stack.parquet')
tt = A.train_targets()
df = stack.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df['future_spend_4w'].values.astype(float)
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w','gbm_corr')]
X = df[feats].astype(float)
X = X.fillna(X.median())
mu, sd = X.mean(), X.std().replace(0,1)
Xs = ((X-mu)/sd).values
gp = df['gbm_pred'].values
days = sorted(df.snapshot_day.unique())

def cv_mae(Xmat, alpha):
    maes=[]
    for d in days:
        tr = df.snapshot_day.values != d; va = ~tr
        Xt, yt = Xmat[tr], y[tr]
        U,S,Vt = np.linalg.svd(Xt, full_matrices=False)
        Sinv = S/(S**2+alpha)
        b = Vt.T@(Sinv*(U.T@yt))
        maes.append(np.abs(Xmat[va]@b - y[va]).mean())
    return np.mean(maes)

for alpha in [300, 1000, 3000, 10000]:
    print('alpha',alpha, 'base:', round(cv_mae(Xs,alpha),3), '+gbm:', round(cv_mae(np.column_stack([Xs, gp/100]),alpha),3))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

stack = A.load_saved('e014_stack.parquet')
tt = A.train_targets()
df = stack.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df['future_spend_4w'].values.astype(float)
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w','gbm_corr')]
X = df[feats].astype(float)
X = X.fillna(X.median())
mu, sd = X.mean(), X.std().replace(0,1)
Xs = ((X-mu)/sd).values
gp = df['gbm_pred'].values
days = sorted(df.snapshot_day.unique())

def cv_mae(Xmat, alpha):
    maes=[]
    for d in days:
        tr = df.snapshot_day.values != d; va = ~tr
        Xt, yt = Xmat[tr], y[tr]
        U,S,Vt = np.linalg.svd(Xt, full_matrices=False)
        Sinv = S/(S**2+alpha)
        b = Vt.T@(Sinv*(U.T@yt))
        maes.append(np.abs(Xmat[va]@b - y[va]).mean())
    return np.mean(maes)

# check gbm_pred scale on train vs val snaps
for d in [95, 431, 459, 487]:
    sub = stack[stack.snapshot_day==d]
    print('snap',d,'n',len(sub),'gbm_pred mean',round(sub['gbm_pred'].mean(),1),'median',round(sub['gbm_pred'].median(),1))
# does adding gbm_pred hurt the earliest train snaps? per-snap MAE at alpha=3000
Xg = np.column_stack([Xs, gp/100])
for d in days:
    tr = df.snapshot_day.values != d; va = df.snapshot_day.values==d
    Xt, yt = Xg[tr], y[tr]
    U,S,Vt = np.linalg.svd(Xt, full_matrices=False)
    Sinv = S/(S**2+alpha) if False else S/(S**2+3000)
    b = Vt.T@(Sinv*(U.T@yt))
    m = np.abs(Xg[va]@b - y[va]).mean()
    m0 = np.abs(Xs[va]@b - y[va]).mean() if False else 0
    print('snap',d,'MAE',round(m,1))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

stack = A.load_saved('e014_stack.parquet')
base = A.load_saved('e013_denoise.parquet')
# verify key alignment with e013
k1 = stack[['household_key','snapshot_day']].sort_values(['household_key','snapshot_day']).reset_index(drop=True)
k2 = base[['household_key','snapshot_day']].sort_values(['household_key','snapshot_day']).reset_index(drop=True)
print('keys identical:', k1.equals(k2), '| dup:', stack.duplicated(['household_key','snapshot_day']).sum(), '| gbm NaN:', stack['gbm_pred'].isna().sum())

tt = A.train_targets()
df = stack.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df['future_spend_4w'].values.astype(float)
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w','gbm_corr')]
X = df[feats].astype(float); X = X.fillna(X.median())
mu, sd = X.mean(), X.std().replace(0,1)
Xs = ((X-mu)/sd).values
gp = df['gbm_pred'].values
days = sorted(df.snapshot_day.unique())

def cv_mae(Xmat, alpha=3000):
    maes=[]
    for d in days:
        tr = df.snapshot_day.values != d; va = ~tr
        Xt, yt = Xmat[tr], y[tr]
        U,S,Vt = np.linalg.svd(Xt, full_matrices=False)
        Sinv = S/(S**2+alpha)
        b = Vt.T@(Sinv*(U.T@yt))
        maes.append(np.abs(Xmat[va]@b - y[va]).mean())
    return round(np.mean(maes),3)

print('A raw:', cv_mae(np.column_stack([Xs, gp])))
print('B /100:', cv_mae(np.column_stack([Xs, gp/100])))
print('C raw+log:', cv_mae(np.column_stack([Xs, gp, np.log1p(gp)])))
print('D /100+log/100:', cv_mae(np.column_stack([Xs, gp/100, np.log1p(gp)/100])))
print('E gbm only:', cv_mae(np.column_stack([np.log1p(gp)])))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

stack = A.load_saved('e014_stack.parquet')
df = stack.drop(columns=['gbm_corr'])
df['gbm_pred'] = df['gbm_pred'] / 100.0
df['gbm_log'] = np.log1p(df['gbm_pred']*100.0)/100.0
p = A.save_table(df, 'e017_stack_scaled.parquet')
print(p, df.shape)
print(df[['household_key','snapshot_day','gbm_pred','gbm_log']].head())

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

stack = A.load_saved('e014_stack.parquet')
print('cols tail:', stack.columns.tolist()[-3:])
df = stack.copy()
df['gbm_pred'] = df['gbm_pred'] / 100.0
df['gbm_log'] = np.log1p(df['gbm_pred']*100.0)/100.0
p = A.save_table(df, 'e017_stack_scaled.parquet')
print(p, df.shape)
print(df[['household_key','snapshot_day','gbm_pred','gbm_log']].head())