import pandas as pd, numpy as np, agent_api
key=['household_key','snapshot_day']
e13=agent_api.load_saved('e013_union.parquet')
e1=agent_api.load_saved('e001_history.parquet')
cand=agent_api.load_saved('cand_new.parquet')
for nm,df in [('e13',e13),('e1',e1),('cand(E012)',cand)]:
    print(nm, df.shape)
print('E013 features:', sorted([c for c in e13.columns if c not in key]))
print()
print('E001 features:', sorted([c for c in e1.columns if c not in key]))
print()
print('E012 cand features:', sorted([c for c in cand.columns if c not in key]))
ov=(set(e13.columns)&set(e1.columns))-set(key)
print('overlap e13/e1:', sorted(ov))
m=e13.merge(e1,on=key,how='inner',suffixes=('','_dup'))
m=m.drop(columns=[c for c in m.columns if c.endswith('_dup')])
print('merged', m.shape)
tt=agent_api.train_targets()
tr=m.merge(tt,on=key,how='inner')
print('train rows', tr.shape, 'val rows', len(m)-len(tr))
y=tr[agent_api.TARGET].astype(float)
print(y.describe())
feats=[c for c in m.columns if c not in key]
res=[]
for c in feats:
    s=tr[c]
    if not pd.api.types.is_numeric_dtype(s):
        s=pd.Series(pd.factorize(s)[0],index=tr.index).astype(float)
    v=s.corr(y,method='spearman')
    res.append((c, 0.0 if pd.isna(v) else v))
res.sort(key=lambda t:-abs(t[1]))
print('--- top 30 |spearman| ---')
for c,v in res[:30]: print(f'{v: .3f}  {c}')
print('--- bottom 10 ---')
for c,v in res[-10:]: print(f'{v: .3f}  {c}')
agent_api.save_table(m,'e014_base.parquet')
print('saved e014_base')
