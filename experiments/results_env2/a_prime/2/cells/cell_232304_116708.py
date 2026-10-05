import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
t = A.load_saved('e011_discounts.parquet')
tt = A.train_targets()
print('E011', t.shape, '| targets', tt.shape)
cols = list(t.columns)
print('COLS:', cols)
m = tt.merge(t, on=['household_key','snapshot_day'], how='inner')
y = m['future_spend_4w'].astype(float).values
print('merged', m.shape, '| zero-frac %.3f' % (y==0).mean())
print(pd.Series(y).describe(percentiles=[.25,.5,.75,.9,.95,.99]).round(1))
print(m.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median']).round(1).T)
num_cols = [c for c in cols if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(m[c])]
str_cols = [c for c in cols if c not in ('household_key','snapshot_day') and c not in num_cols]
print('num', len(num_cols), '| str', str_cols)
rows=[]
for c in num_cols:
    v = m[c].astype(float); vn = v.fillna(v.median()).values
    if vn.std()==0: continue
    rows.append((c, float(np.corrcoef(vn,y)[0,1]), float(np.abs(vn-y).mean())))
d = pd.DataFrame(rows, columns=['feat','corr','naiveMAE'])
d = d.reindex(d.corr.abs().sort_values(ascending=False).index)
print(d.head(28).round(3).to_string())