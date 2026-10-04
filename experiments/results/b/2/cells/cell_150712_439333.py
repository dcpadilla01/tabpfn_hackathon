import pandas as pd, numpy as np

e7 = load_saved('e007_log.parquet')
tt = train_targets()
m = tt.merge(e7, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values

X = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).astype(float)
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

# per-household target variability
tt2 = tt.sort_values(['household_key','snapshot_day'])
w = tt2.groupby('household_key')['future_spend_4w'].std()
print('\nmean within-hh std:', w.mean(), 'overall std:', tt2.future_spend_4w.std())
# correlation of household mean spend_28 with household mean target
hm = tt2.groupby('household_key')['future_spend_4w'].mean()
e5 = load_saved('e005_longrun.parquet')
s28 = e5.groupby('household_key')['spend_28'].mean()
j = pd.concat([hm, s28], axis=1).dropna()
print('corr(hh-mean target, hh-mean spend_28):', j.corr().iloc[0,1].round(3))