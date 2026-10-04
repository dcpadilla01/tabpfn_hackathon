import pandas as pd, numpy as np

e7 = load_saved('e007_log.parquet')
tt = train_targets()
m = tt.merge(e7, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values

# OLS with ridge on e007 features (in-sample proxy for the fixed linear model)
def ridge(X, y, lam=1.0):
    Xb = np.c_[np.ones(len(X)), X]
    A = Xb.T@Xb + lam*np.eye(Xb.shape[1]); A[0,0]=0
    return np.linalg.solve(A, Xb.T@y)

X = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).astype(float)
med = X.median()
Xf = X.fillna(med).clip(-1e6,1e6).values
Xm = np.c_[np.ones(len(Xf)), Xf]
# standardize
mu = Xm[:,1:].mean(0); sd = Xm[:,1:].std(0)+1e-9
Z = np.c_[np.ones(len(Xm)), (Xm[:,1:]-mu)/sd]
for lam in [1,10,50,100]:
    b = ridge(Z[:,1:], y, lam)
    p = Z@np.r_[0,b]
    print(f'ridge lam={lam}: in-sample MAE={np.mean(np.abs(p-y)):.2f}')

# also check: how many households are in train but not validation (new households)
hh_tr = set(m.household_key[m.snapshot_day<=431])
print('\ntrain rows', len(m), 'unique hh', m.household_key.nunique())

# target stats per household variance vs within
tt2 = tt.sort_values(['household_key','snapshot_day'])
print('within-hh std/overall std:')
print(tt2.groupby('household_key')['future_spend_4w'].std().mean(), tt2.future_spend_4w.std())