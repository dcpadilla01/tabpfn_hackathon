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