import pandas as pd, numpy as np, agent_api as A

t = A.load_saved('e003_catmix.parquet')
tt = A.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'])
print('merged', df.shape)
y = df['future_spend_4w']

def mae(p): return float(np.mean(np.abs(p-y)))

print('median pred MAE:', mae(np.full(len(y), y.median())))
print('mean pred MAE:', mae(np.full(len(y), y.mean())))
print('spend_l1 MAE:', mae(df.spend_l1))
print('mean(l1..l3) MAE:', mae(df[['spend_l1','spend_l2','spend_l3']].mean(1)))
print('spend_l123_mean MAE:', mae(df.spend_l123_mean))
print('spend_rate28 MAE:', mae(df.spend_rate28))
print('spend_l13 MAE:', mae(df.spend_l13))
# shrunk versions
for w in [0.5,0.6,0.7,0.8]:
    print(f'shrink {w}: l1*w MAE:', mae(df.spend_l1*w))
# correlation structure
print('\ncorr with y:')
for c in ['spend_l1','spend_l2','spend_l3','spend_l123_mean','spend_l13','days_since_last','zero_recent','momentum','spend_rate28']:
    print(c, round(float(df[c].corr(y)),3), '| log-log', round(float(np.corrcoef(np.log1p(df[c].clip(lower=0)), np.log1p(y))[0,1]),3))
# zero structure
print('\nmean y by spend_l1 bucket:')
df['b'] = pd.cut(df.spend_l1, [-1,0,25,75,150,300,10000])
print(df.groupby('b', observed=True).agg(n=('y','size'), ymean=('y','mean'), l1mean=('spend_l1','mean'), ymed=('y','median')))
print('\nmean y by days_since_last:')
df['b2'] = pd.cut(df.days_since_last, [-1,3,7,14,21,28,56,10000])
print(df.groupby('b2', observed=True).agg(n=('y','size'), ymean=('y','mean'), pzero=('y', lambda s:(s==0).mean())))