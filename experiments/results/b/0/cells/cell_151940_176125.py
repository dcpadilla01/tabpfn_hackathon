v = snapshot()
p = v.products
print('products shape:', p.shape)
print(p['brand'].value_counts(dropna=False).head())
print()
print(p['department'].value_counts().head(15))
print()
dm = v.display_mailer
print('display_mailer shape:', dm.shape, 'weeks:', dm.week_no.min(), dm.week_no.max())
print(dm['display'].value_counts(dropna=False))
print(dm['mailer'].value_counts(dropna=False))
print('n products in dm:', dm.product_id.nunique(), 'n stores:', dm.store_id.nunique())
