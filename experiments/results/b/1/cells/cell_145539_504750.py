import pandas as pd, numpy as np, agent_api
pd.set_option('display.width', 250)
t = agent_api.load_saved('e001_history.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('shape', df.shape)
print('cols:', df.columns.tolist())
y = df['future_spend_4w'].astype(float)
print('\ntarget describe:'); print(y.describe())
print('zero frac:', round(float((y==0).mean()),3))
g = df.groupby('snapshot_day')['future_spend_4w']
print('\nper-snapshot target:'); print(pd.DataFrame({'mean':g.mean().round(1),'median':g.median(),'zero':g.apply(lambda v:(v==0).mean()).round(3)}))
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
rows=[]
for c in feats:
    x = pd.to_numeric(df[c], errors='coerce')
    rows.append((c, round(x.corr(y),3), round(x.notna().mean(),3)))
cs = pd.DataFrame(rows, columns=['feat','corr_y','notna']).sort_values('corr_y', key=lambda s: s.abs(), ascending=False)
print('\nfeature corr with target:'); print(cs.to_string())
print('\nnaive single-feature MAE (train):')
res=[]
for c in feats:
    x = pd.to_numeric(df[c], errors='coerce')
    if x.notna().mean()>0.9:
        p = x.fillna(0).clip(lower=0)
        res.append((c, round(float((p-y).abs().mean()),2)))
print(sorted(res, key=lambda r: r[1])[:12])
