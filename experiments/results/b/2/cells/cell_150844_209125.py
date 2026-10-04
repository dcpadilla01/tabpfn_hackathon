import pandas as pd, numpy as np
tt = train_targets()
y = tt.future_spend_4w.values
gm = tt.groupby('household_key')['future_spend_4w']

e7 = load_saved('e007_log.parquet')
hh_feat = e7.groupby('household_key').mean(numeric_only=True)
hh_mean_t = gm.mean()
j = hh_feat.join(hh_mean_t.rename('y'), how='inner')
print('joined', j.shape)
X = j.drop(columns='y'); mu,sd = X.mean(), X.std()+1e-9
Z = np.c_[np.ones(len(j)), ((X-mu)/sd).values]
b = np.linalg.lstsq(Z, j.y.values, rcond=None)[0]
p = Z@b
print('in-sample MAE predicting hh-mean level:', np.mean(np.abs(p-j.y.values)).round(2))
print('corr(pred level, actual level):', np.corrcoef(p, j.y.values)[0,1].round(3))
m = tt.merge(e7, on=['household_key','snapshot_day'], how='left')
Xf = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).astype(float).fillna(hh_feat.median()).values
Xf = (Xf-mu.values)/sd.values
Xf = np.c_[np.ones(len(Xf)), Xf]
print('row-level MAE using level model:', np.mean(np.abs(Xf@b-y)).round(2))