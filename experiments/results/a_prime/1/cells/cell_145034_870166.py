import pandas as pd, numpy as np, agent_api as A

t = A.load_saved('e003_catmix.parquet')
tt = A.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'])
y = df['future_spend_4w'].values
def mae(p): return float(np.mean(np.abs(p-y)))

# hand predictors on train rows
p1 = df[['spend_l1','spend_l2','spend_l3']].mean(1).values
p2 = df.spend_rate28.values
print('mean(l1..l3):', round(mae(p1),2), ' rate28:', round(mae(p2),2))
for a in [0.3,0.5,0.7]:
    print(f'blend {a}*p1+{1-a}*p2:', round(mae(a*p1+(1-a)*p2),2))

# per-snapshot target stats (seasonality)
g = df.groupby('snapshot_day').agg(n=('future_spend_4w','size'), mean=('future_spend_4w','mean'), med=('future_spend_4w','median'), p1m=('spend_l1','mean'))
print('\nper snapshot:'); print(g.round(1))

# residual of naive predictor by snapshot
df['res'] = np.abs(df[['spend_l1','spend_l2','spend_l3']].mean(1).values - y)
print('\nnaive MAE per snapshot:'); print(df.groupby('snapshot_day')['res'].mean().round(1))

# zero structure: mean y when l1==0 vs >0
print('\nl1==0: n=', int((df.spend_l1==0).sum()), 'mean y=', round(float(y[df.spend_l1==0].mean()),1))
print('l1>0: mean y=', round(float(y[df.spend_l1>0].mean()),1))
print('l1==0 & l2==0 & l3==0: n=', int(((df.spend_l1==0)&(df.spend_l2==0)&(df.spend_l3==0)).sum()),
      'mean y=', round(float(y[(df.spend_l1==0)&(df.spend_l2==0)&(df.spend_l3==0)].mean()),1))