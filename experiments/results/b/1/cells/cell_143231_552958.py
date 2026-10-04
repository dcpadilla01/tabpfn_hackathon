import agent_api as A
v = A.snapshot(459)
p = v.products
print(p.department.value_counts().head(20))
print(p.brand.value_counts(dropna=False))
t = v.transactions
print('tx rows', len(t), 'hh', t.household_key.nunique())
print(t.household_key.head().tolist())
# check keys format
print(A.KEYS)
print(A.TARGET)
# quick: top dept spend share overall
m = t.merge(p[['product_id','department','brand']], on='product_id', how='left')
g = m.groupby('department').sales_value.sum().sort_values(ascending=False)
print((g/g.sum()).head(12))
print('brand na share', m.brand.isna().mean())