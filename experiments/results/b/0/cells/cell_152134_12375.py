v = snapshot(431)
t = v.transactions
day = v.day
first = t.groupby('household_key').day.min()
hh = set(first[first <= day-84].index)
print('eligible hh at 431:', len(hh))
t = t[t.household_key.isin(hh)]

def agg(lo, hi):
    w = t[(t.day > day-lo) & (t.day <= day-hi)]
    return w.groupby('household_key').agg(spend=('sales_value','sum'), qty=('quantity','sum'), ntrip=('basket_id','nunique'))

a28 = agg(28,0); a_p = agg(56,28); a112 = agg(112,0)
f = a28.join(a_p, rsuffix='_p', how='outer').join(a112, rsuffix='_112').fillna(0)
f['ap28'] = f.spend/f.qty.clip(lower=1)
f['ap_p'] = f.spend_p/f.qty_p.clip(lower=1)
f['price_trend'] = f.ap28/f.ap_p.clip(lower=0.5)
f['qty_trend'] = f.qty/f.qty_p.clip(lower=1)
w28 = t[t.day > day-28]
gq = w28.groupby(['household_key','basket_id']).quantity.sum().reset_index()
mx = gq.groupby('household_key').quantity.max()
f['maxtrip_qty28'] = mx
f['share_qty_toptrip'] = mx/f.qty.clip(lower=1)

tt = train_targets()
t431 = tt[tt.snapshot_day==431].set_index('household_key').future_spend_4w
f = f.join(t431.rename('y'), how='inner')
print('n hh with target:', len(f))
for c in ['ap28','price_trend','qty_trend','maxtrip_qty28','share_qty_toptrip','qty','qty_p']:
    print(c, 'spearman:', round(f[c].corr(f.y.rank(), method='spearman'),3))

df0 = load_saved('e010_decay.parquet')
d431 = df0[df0.snapshot_day==431].set_index('household_key').join(t431.rename('y'), how='inner')
for c in ['rwspend84','spend_per_day28','lag_mean_1_4','dec_spend14','spend28','spend112']:
    print('log1p(%s)'%c, 'spearman:', round(np.log1p(d431[c]).corr(d431.y.rank(), method='spearman'),3))