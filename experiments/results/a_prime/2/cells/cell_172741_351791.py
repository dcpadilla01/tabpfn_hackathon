import agent_api, pandas as pd, numpy as np, re

v = agent_api.snapshot(459)
tx = v.transactions.merge(v.products[['product_id','department']], on='product_id', how='left')
tx['department'] = tx['department'].astype(str)
tx.loc[tx.department.isin(['nan','None','']), 'department'] = 'UNK'
ds = tx.groupby('department')['sales_value'].sum().sort_values(ascending=False)
top = list(ds.head(24).index)

def sanitize(s): return re.sub(r'[^A-Za-z0-9]+','_',str(s))[:30]

def fn(view, snapshot_day):
    s = snapshot_day
    keys = np.asarray(view.households)
    tx = view.transactions.merge(view.products[['product_id','department']], on='product_id', how='left')
    tx['department'] = tx['department'].astype(str)
    tx.loc[tx.department.isin(['nan','None','']), 'department'] = 'UNK'
    w = tx[(tx.day > s-364) & (tx.day <= s-336)]
    res = pd.DataFrame(index=keys)
    if len(w):
        piv = w.pivot_table(index='household_key', columns='department', values='sales_value', aggfunc='sum')
        piv = piv.reindex(columns=top).fillna(0.0).reindex(keys).fillna(0.0)
        tot = w.groupby('household_key')['sales_value'].sum().reindex(keys).fillna(0.0).values
    else:
        piv = pd.DataFrame(0.0, index=keys, columns=top)
        tot = np.zeros(len(keys))
    for d in top:
        res['dt13_'+sanitize(d)] = piv[d].values
    for d in top:
        res['ds13_'+sanitize(d)] = piv[d].values/(tot+1.0)
    return res

new = agent_api.build_features(fn)
base = agent_api.load_saved('e009_ewma_longlags.parquet')
m = base.merge(new, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
tt = agent_api.train_targets()
df = m.merge(tt, on=['household_key','snapshot_day'])
tr = df[df.snapshot_day <= 347]; iv = df[df.snapshot_day >= 375]
def ev(cols, alpha=100.0):
    A = tr[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    B = iv[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    mu=A.mean(); sd=A.std().replace(0,1)
    A=((A-mu)/sd).fillna(0).values; B=((B-mu)/sd).fillna(0).values
    ytr=tr.future_spend_4w.values; yiv=iv.future_spend_4w.values
    w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(B@w-yiv).mean()
bcols=[c for c in base.columns if c not in ('household_key','snapshot_day')]
acols=[c for c in m.columns if c not in ('household_key','snapshot_day')]
print('proxy base:', ev(bcols))
print('proxy +dept-season:', ev(acols))
newcols=[c for c in m.columns if c.startswith(('dt13_','ds13_'))]
cm = df[newcols].corrwith(df.future_spend_4w).abs().sort_values(ascending=False)
print(cm.head(10))
p = agent_api.save_table(m, 'e017_dept_season.parquet')
print('saved', p)