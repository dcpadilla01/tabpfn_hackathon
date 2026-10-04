
import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved('e012_basket_shape.parquet')
tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])
ycol = agent_api.TARGET
df = base.set_index(['household_key','snapshot_day']).join(tt[ycol])
num = df.drop(columns=[ycol]).select_dtypes(include=[np.number]).columns.tolist()
X = df[num].astype(float); X = X.fillna(X.median())
y = df[ycol].astype(float)
itr = X.index.get_level_values(1).isin([d for d in agent_api.snapshot_days()['train'] if d != 431])
iva = X.index.get_level_values(1).isin([431])
mu, sd = X[itr].mean(), X[itr].std().replace(0,1)
Xs = (X-mu)/sd
Xtr = np.c_[np.ones(int(itr.sum())), Xs[itr].values]; Xva = np.c_[np.ones(int(iva.sum())), Xs[iva].values]
w = np.linalg.solve(Xtr.T@Xtr + 3000*np.eye(Xtr.shape[1]), Xtr.T@y[itr].values)
p = np.clip(Xva@w, 0, None)
res = y[iva].values - p
d431 = df[iva].copy(); d431['pred'] = p; d431['res'] = res; d431['y'] = y[iva].values
print('431 MAE', round(float(np.abs(res).mean()),2))
# residual by target bucket
d431['ybucket'] = pd.cut(d431['y'], [-1,0.01,50,100,200,400,800,10000])
print(d431.groupby('ybucket', observed=True).apply(lambda g: pd.Series({'n':len(g),'y':g['y'].mean(),'pred':g['pred'].mean(),'mae':g['res'].abs().mean(),'bias':g['res'].mean()})).round(1).to_string())
# by has_demo / size
print(d431.groupby('has_demo').apply(lambda g: pd.Series({'n':len(g),'mae':g['res'].abs().mean(),'bias':g['res'].mean()})).round(1).to_string())
print(d431.groupby('size_ord', dropna=False).apply(lambda g: pd.Series({'n':len(g),'mae':g['res'].abs().mean(),'bias':g['res'].mean()})).round(1).to_string())
# drift: target mean by snapshot day (train)
g = tt.groupby('snapshot_day')[ycol].agg(['mean','median','count'])
print(g.round(1).to_string())
# also mean of spend_28 by snapshot day (proxy for level drift)
s = base.groupby('snapshot_day')[['spend_28','wk_avg_8','fwd28_mean']].mean().round(1)
print(s.to_string())
