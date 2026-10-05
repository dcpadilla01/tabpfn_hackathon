
import pandas as pd, numpy as np, agent_api

feats = agent_api.load_saved('feats_v3.parquet')
t = agent_api.train_targets()
tr = feats.merge(t, on=['household_key','snapshot_day'], how='inner')
print("train rows:", tr.shape)

p6 = agent_api.load_saved('pred_e006.parquet')
p4 = agent_api.load_saved('pred_e004.parquet')
p5 = agent_api.load_saved('pred_e005.parquet')
for nm,p in [('e004',p4),('e005',p5),('e006',p6)]:
    print(nm, p.shape, list(p.columns))

# merge preds onto train rows (in-sample diagnostic)
m = tr[['household_key','snapshot_day','future_spend_4w','spend_28','spend_56','spend_84']].merge(
    p6.rename(columns={'prediction':'pred6'}), on=['household_key','snapshot_day'])
m = m.merge(p4.rename(columns={'prediction':'pred4'}), on=['household_key','snapshot_day'])
m = m.merge(p5.rename(columns={'prediction':'pred5'}), on=['household_key','snapshot_day'])
y = m['future_spend_4w']

def mae(a,b): return np.mean(np.abs(a-b))
print("\nIN-SAMPLE (train) diagnostics:")
print("MAE spend_28 naive:", mae(m['spend_28'], y))
print("MAE pred4:", mae(m['pred4'], y), "MAE pred5:", mae(m['pred5'], y), "MAE pred6:", mae(m['pred6'], y))
print("mean y:", y.mean(), "mean pred6:", m['pred6'].mean(), "median y:", y.median(), "median pred6:", m['pred6'].median())
print("corr(pred6,y):", np.corrcoef(m['pred6'], y)[0,1], " corr(spend28,y):", np.corrcoef(m['spend_28'], y)[0,1])

# bias by target decile
m['dec'] = pd.qcut(y, 10, duplicates='drop')
print(m.groupby('dec', observed=True).apply(lambda g: pd.Series({
    'n': len(g), 'y_mean': g['future_spend_4w'].mean(), 'pred6_mean': g['pred6'].mean(),
    'bias': (g['pred6']-g['future_spend_4w']).mean()}), include_groups=False))

# how often pred6==0 vs y==0
print("\nzero pred6 share:", (m['pred6']<=1e-9).mean(), " zero y share:", (y==0).mean())
print("MAE on y==0 rows:", mae(m.loc[y==0,'pred6'], 0), " MAE on y>0 rows:", mae(m.loc[y>0,'pred6'], y[y>0]))
