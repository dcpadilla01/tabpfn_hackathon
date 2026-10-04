import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e011_price.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
yv = df.future_spend_4w.values.astype(float)
tr = (df.snapshot_day<=403).values; va = (df.snapshot_day==431).values
x = df.spend28.values.astype(float)
m = tr & ~np.isnan(x)
b_, a_ = np.polyfit(x[m], yv[m], 1)
mv = va & ~np.isnan(x)
p = a_ + b_*x[mv]
print('univar spend28: corr', round(float(np.corrcoef(x[m], yv[m])[0,1]),3), 'slope', round(b_,3), 'MAE431', round(float(np.abs(p-yv[mv]).mean()),3))
# clipped slope (spend can't be huge)
b2, a2 = np.polyfit(np.minimum(x[m], 400), yv[m], 1)
p2 = a2 + b2*np.minimum(x[mv],400)
print('clipped400 univar MAE431', round(float(np.abs(p2-yv[mv]).mean()),3))
# binned means: 25 quantile bins of spend28 on train, predict bin mean
qs = np.quantile(x[m], np.linspace(0,1,26))
bins = np.digitize(x[mv], qs[1:-1])
bmean = np.array([yv[m][(np.digitize(x[m],qs[1:-1])==k)].mean() for k in range(25)])
pb = bmean[bins]
print('binned25 spend28 MAE431', round(float(np.abs(pb-yv[mv]).mean()),3))
# two-feature: spend28 + lag trend
x2 = df.lag_mean_1_4.values.astype(float)
mm = tr & ~np.isnan(x) & ~np.isnan(x2)
A = np.c_[np.ones(mm.sum()), x[mm], x2[mm]]
w = np.linalg.lstsq(A, yv[mm], rcond=None)[0]
mv2 = va & ~np.isnan(x) & ~np.isnan(x2)
p3 = w[0]+w[1]*x[mv2]+w[2]*x2[mv2]
print('spend28+lagmean14 MAE431', round(float(np.abs(p3-yv[mv2]).mean()),3))
# per-snapshot corr of spend28 with y
for sd in [95,207,291,375,403,431]:
    mm2 = df.snapshot_day==sd
    print('corr@',sd, round(float(np.corrcoef(x[mm2.values], yv[mm2.values])[0,1]),3))
# what does the best possible constant-mixture do: predict y = spend28 exactly? MAE of |y - spend28|
print('MAE if pred=spend28:', round(float(np.abs(x[mv]-yv[mv]).mean()),3))
print('MAE if pred=0.8*spend28:', round(float(np.abs(0.8*x[mv]-yv[mv]).mean()),3))
