import pandas as pd, numpy as np

e7 = load_saved('e007_log.parquet')
tt = train_targets()
m = tt.merge(e7, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values

X = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).astype(float)
# drop exact duplicates
X = X.T.drop_duplicates().T
print('after dedup:', X.shape)
med = X.median()
Xf = X.fillna(med).clip(-1e6,1e6).values
mu = Xf.mean(0); sd = Xf.std(0)+1e-9
Z = (Xf-mu)/sd
Zb = np.c_[np.ones(len(Z)), Z]
def ridge(Zb, y, lam):
    A = Zb.T@Zb + lam*np.eye(Zb.shape[1]); A[0,0]=0
    b = np.linalg.solve(A, Zb.T@y)
    p = Zb@b
    return np.mean(np.abs(p-y))
for lam in [1,10,50,100,300]:
    print(f'ridge lam={lam}: in-sample MAE={ridge(Zb,y,lam):.2f}')

# how much of MAE is from zero-target rows? per-target-bin MAE for a simple predictor spend_28*0.87
s28 = m['spend_28'].values
p = 0.87*s28
err = np.abs(p-y)
print('\nMAE by target bucket:')
for lo,hi in [(0,1),(1,50),(50,100),(100,200),(200,400),(400,1e9)]:
    msk = (y>=lo)&(y<hi)
    print(f'  y in [{lo},{hi}): n={msk.sum():6d}  MAE={err[msk].mean():7.2f}  mean|y|={y[msk].mean():7.1f}')
print('share of total MAE from y==0 rows:', (err[y==0].sum())/err.sum())
print('share of rows y==0:', (y==0).mean())