import warnings; warnings.filterwarnings('ignore')
v = snapshot()
dm = v.display_mailer
print(dm.shape, dm.columns.tolist())
print(dm.head())
print(dm.display.value_counts())
print(dm.mailer.value_counts())
print('weeks', dm.week_no.min(), dm.week_no.max())
print('n products', dm.product_id.nunique(), 'n stores', dm.store_id.nunique())
# how many households' recent purchases have display/mailer info?
tx = v.transactions
print('tx max day', tx.day.max(), 'shape', tx.shape)
print(tx[['quantity','sales_value','coupon_disc','coupon_match_disc','retail_disc','trans_time']].describe().T.round(2))
# check zero-variance dsp cols in full table
t = load_saved('e007_lagseq.parquet')
d = [c for c in t.columns if c.startswith('dsp_')]
vv = t[d].var()
print('zero-var dsp cols:', [c for c in d if vv[c]==0])