import agent_api as A, pandas as pd, numpy as np
v = A.snapshot(459)
print('day', v.day, 'week', v.week)
hh = v.households
print(type(hh)); print(hh.shape); print(hh.head(3))
tx = v.table('transactions')
print('tx shape', tx.shape, 'max day', tx.day.max())
dm = v.table('display_mailer')
print('dm shape', dm.shape); print(dm.head(3)); print(dm.display.value_counts().head()); print(dm.mailer.value_counts().head())
dem = v.table('demographics')
print('dem shape', dem.shape)
for c in ['classification_1','classification_3','classification_4','classification_5','classification_2','homeowner_desc','kid_category_desc']:
    print(c, sorted(dem[c].unique())[:15])
cp = v.table('campaign_targets')
print('ct shape', cp.shape, cp.description.unique())
pr = v.table('products')
print('products shape', pr.shape, pr.columns.tolist())
print(pr.brand.unique()[:10])
