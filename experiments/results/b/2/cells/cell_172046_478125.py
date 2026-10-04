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