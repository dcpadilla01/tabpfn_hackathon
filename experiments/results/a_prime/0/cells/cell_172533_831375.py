import numpy as np, pandas as pd
t = agent_api.load_saved('e013_stock.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
days = agent_api.snapshot_days()
trm = df.snapshot_day.isin(days['train']).values
feat = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
num = df[feat].select_dtypes(include=[np.number,'bool']).columns.tolist()
cat = [c for c in feat if c not in num]
Xtr = df.loc[trm, num].astype(float).replace([np.inf,-np.inf], np.nan)
ytr = df.future_spend_4w[trm].values.astype(float)
Xf = Xtr.fillna(Xtr.median()).fillna(0)
var = Xf.var(); zv = var[var<=1e-12].index.tolist()
num2 = [c for c in num if c not in zv]
Xf = Xf[num2]
corry = {}
for c in num2:
    x = Xf[c].values
    r = np.corrcoef(x, ytr)[0,1]
    corry[c] = abs(r) if np.isfinite(r) else 0.0
order = sorted(num2, key=lambda c: -corry[c])
Cm = np.nan_to_num(np.corrcoef(Xf[order].values, rowvar=False))
kept=[]; dropped=[]
for i,c in enumerate(order):
    if any(abs(Cm[i,j])>0.95 for j in kept): dropped.append(c)
    else: kept.append(c)
print('kept %d | dropped %d | zero-var dropped %d'%(len(kept),len(dropped),len(zv)))
print('DROPPED:', dropped)
out = df[['household_key','snapshot_day'] + kept + cat].copy()
print('out shape', out.shape, '| dtypes', out.dtypes.value_counts().to_dict())
path = agent_api.save_table(out, 'pruned_v1.parquet')
print('saved:', path)
