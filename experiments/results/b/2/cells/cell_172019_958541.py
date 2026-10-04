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