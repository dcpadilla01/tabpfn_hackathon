import pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
tt = agent_api.train_targets()
train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
val_days = [459,487,515,543]
FEAT = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
print('n_feat', len(FEAT))
# per-snapshot target stats
g = tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median',lambda s:(s==0).mean()])
g.columns=['mean','median','zerofrac']; print(g.round(1))
p5 = agent_api.load_saved('e005_preds.parquet')
gp = p5.groupby('snapshot_day').prediction.agg(['mean','median',lambda s:(s<=1).mean()])
gp.columns=['mean','median','nearzero']; print(gp.round(1))
print(xgb.__version__)