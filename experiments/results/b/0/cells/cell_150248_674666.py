import warnings; warnings.filterwarnings('ignore')
t = load_saved('e007_lagseq.parquet')
tt = train_targets()
m = tt.merge(t, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values
b2 = (0.4*m.spend28.fillna(0)+0.3*m.spend112.fillna(0)/4+0.3*m.lag_mean_5_8.fillna(0)).values
act = m.spend28.fillna(0)>0
# For actives: how much better can a 2-piece linear model do? quick ridge on a few feats
feats = ['spend28','spend56','spend112','spend364','lag_mean_1_4','lag_mean_5_8','trips28','nprod28','recency','iv_mean112','trend_112_364','lt_spend','tenure']
X = m[feats].fillna(0).values
Xa = X[act]; ya = y[act]; ba = b2[act]
# standardize
mu = Xa.mean(0); sd = Xa.std(0)+1e-9
Xs = (Xa-mu)/sd
from numpy.linalg import lstsq
# ridge via normal equations
lam=1.0
A = Xs.T@Xs + lam*np.eye(len(feats)); bvec = Xs.T@(ya-ba)
w = np.linalg.solve(A,bvec)
pred = ba + Xs@w
print('active MAE blend3', np.abs(ba-ya).mean().round(2), '-> ridge resid corr feats', np.abs(pred-ya).mean().round(2))
print(dict(zip(feats,w.round(1))))
# also check zero-inflation: P(y>0) model — proxy by spend28>0 already; among zero28, P(y>0)?
z = ~act
print('P(y>0 | zero28)', (y[z]>0).mean().round(3))
# among zero28 with y>0, magnitude
print('y | zero28 & y>0: median', np.median(y[z&(y>0)]).round(1), 'mean', y[z&(y>0)].mean().round(1))
# does recency predict return among zero28?
zz = m[z]
for c in ['recency','lag_zero_streak','iv_mean112','lt_spend','spend112']:
    a = zz.groupby(pd.qcut(zz[c], 5, duplicates='drop'), observed=True).future_spend_4w.apply(lambda s:(s>0).mean())
    print(c, a.round(2).values)