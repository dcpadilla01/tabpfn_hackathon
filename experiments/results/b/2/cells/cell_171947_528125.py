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
