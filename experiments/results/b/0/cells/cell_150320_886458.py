import warnings; warnings.filterwarnings('ignore')
v = snapshot()
dm = v.display_mailer
tx = v.transactions
# join recent tx (last 28d) with display/mailer for that product/store/week
last28 = tx[tx.day >= 459-27]
print('last28 rows', len(last28))
dm_small = dm[['product_id','store_id','week_no','display','mailer']]
j = last28.merge(dm_small, on=['product_id','store_id','week_no'], how='left')
print('match rate', j.display.notna().mean().round(3))
print('display dist among matched', j.display.value_counts(dropna=False).head(8))
print('mailer dist among matched', j.mailer.value_counts(dropna=False).head(8))
# per-household exposure
j['has_disp'] = (j.display.notna() & (j.display!=0)).astype(int)
j['has_mail'] = (j.mailer.notna() & (j.mailer!='0')).astype(int)
g = j.groupby('household_key').agg(disp_rows=('has_disp','sum'), mail_rows=('has_mail','sum'), rows=('has_disp','size'))
print(g.describe().T.round(2))
# spend on displayed products vs not, last 28d
print('spend share on displayed rows', (j.loc[j.has_disp==1,'sales_value'].sum()/j.sales_value.sum()).round(3))
print('spend share on mailer rows', (j.loc[j.has_mail==1,'sales_value'].sum()/j.sales_value.sum()).round(3))