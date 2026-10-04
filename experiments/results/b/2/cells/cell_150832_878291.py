import pandas as pd, numpy as np
tt = train_targets()
y = tt.future_spend_4w.values

# Best possible "household level" predictor: leave-one-snapshot-out hh mean (honest, no leakage)
gm = tt.groupby('household_key')['future_spend_4w']
hsum = gm.transform('sum'); hsize = gm.transform('size')
loo = (hsum - tt.future_spend_4w) / (hsize - 1)
print('LOO hh-mean MAE (upper bound on level-based):', np.mean(np.abs(loo.values-y)).round(2))

# Now the key question: can we estimate the household LEVEL well from features?
# Test: fit a model to predict the household's train-mean target from its mean E005/E007 features.
e7 = load_saved('e007_log.parquet')
hh_feat = e7.drop(columns=['household_key','snapshot_day']).astype(float).groupby('household_key').mean()
hh_mean_t = gm.mean()
j = hh_feat.join(hh_mean_t.rename('y'), how='inner')
print('joined', j.shape)
X = j.drop(columns='y'); mu,sd = X.mean(), X.std()+1e-9
Z = np.c_[np.ones(len(j)), ((X-mu)/sd).values]
b = np.linalg.lstsq(Z, j.y.values, rcond=None)[0]
p = Z@b
print('in-sample MAE predicting hh-mean level:', np.mean(np.abs(p-j.y.values)).round(2))
# corr of predicted level with actual level
print('corr(pred level, actual level):', np.corrcoef(p, j.y.values)[0,1].round(3))
# and what MAE would that give on row targets?
m = tt.merge(e7, on=['household_key','snapshot_day'], how='left')
Xf = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).astype(float).fillna(hh_feat.median()).values
Xf = (Xf-mu.values)/sd.values
Xf = np.c_[np.ones(len(Xf)), Xf]
prow = Xf@b
print('row-level MAE using level model:', np.mean(np.abs(prow-y)).round(2))