df = load_saved('e010_decay.parquet')
print('e010 shape:', df.shape)
print(list(df.columns))
print()
print('snapshot days:', snapshot_days())
print('KEYS:', KEYS, 'TARGET:', TARGET)


# ---- cell ----
df = load_saved('e010_decay.parquet')
tt = train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('train rows:', len(m), 'val rows:', len(df)-len(m))
y = m['future_spend_4w']
print('target: mean %.1f median %.1f p90 %.1f p99 %.1f max %.1f, zero share %.3f' % (y.mean(), y.median(), y.quantile(.9), y.quantile(.99), y.max(), (y==0).mean()))

# simple predictor baselines on train rows
for c in ['lag_spend_1','spend28','lag_mean_1_4','dec_spend14','rwspend84']:
    pred = m[c].fillna(0).clip(lower=0)
    print('MAE %s: %.2f' % (c, (pred-y).abs().mean()))
print('MAE median: %.2f' % (y.median()-y).abs().mean())
print('MAE blend lag1*0.6+lag2*0.25+lag3*0.15: %.2f' % ((0.6*m.lag_spend_1+0.25*m.lag_spend_2+0.15*m.lag_spend_3).fillna(0).sub(y).abs().mean()))

# correlation of features with target
num = df.select_dtypes(include=[np.number]).columns.tolist()
num = [c for c in num if c not in ('snapshot_day',)]
cor = m[num].corrwith(y).sort_values()
print('\nmost negative corr:'); print(cor.head(8).round(3))
print('\nmost positive corr:'); print(cor.tail(15).round(3))

# MAE by target bucket for the best single predictor
pred = (0.6*m.lag_spend_1+0.25*m.lag_spend_2+0.15*m.lag_spend_3).fillna(0)
b = pd.qcut(y, 5, duplicates='drop')
print('\nMAE by target quintile (blend pred):')
print(m.groupby(b, observed=True).apply(lambda g: pd.Series({'n':len(g),'y_med':g.future_spend_4w.median(),'mae':(g.future_spend_4w-pred[g.index]).abs().mean()})))


# ---- cell ----
df = load_saved('e010_decay.parquet')
tt = train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = m['future_spend_4w']

num = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ('snapshot_day',) and m[c].std()>0]
cor = m[num].corrwith(y).sort_values()
print('TOP positive:'); print(cor.tail(20).round(3))
print('\nTOP negative:'); print(cor.head(15).round(3))

# rank correlation (spearman) may reveal monotone tail predictors
sp = m[num].corrwith(y.rank(), method='spearman').sort_values()
print('\nSpearman top:'); print(sp.tail(15).round(3))
print('\nSpearman bottom:'); print(sp.head(10).round(3))


# ---- cell ----
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


# ---- cell ----
v = snapshot(431)
t = v.transactions
hh = set(v.households)
t = t[t.household_key.isin(hh)]
day = v.day

def agg(lo, hi):  # window (day-lo, day-hi]
    w = t[(t.day > day-lo) & (t.day <= day-hi)]
    g = w.groupby('household_key').agg(spend=('sales_value','sum'), qty=('quantity','sum'), ntrip=('basket_id','nunique'))
    return g

a28 = agg(28,0); a_p = agg(56,28); a112 = agg(112,0); a364 = agg(364,0)

f = a28.join(a_p, rsuffix='_p', how='outer').join(a112, rsuffix='_112').join(a364, rsuffix='_364').fillna(0)
f['ap28'] = f.spend/f.qty.clip(lower=1)
f['ap_p'] = f.spend_p/f.qty_p.clip(lower=1)
f['price_trend'] = f.ap28/f.ap_p.clip(lower=0.5)
f['qty_trend'] = f.qty/f.qty_p.clip(lower=1)
f['stockp'] = f.qty/(f.qty_56 if 'qty_56' in f else f.qty_p).clip(lower=1)  # placeholder
# max single-trip qty in 28d
w28 = t[(t.day > day-28)]
gq = w28.groupby(['household_key','basket_id']).quantity.sum().reset_index()
mx = gq.groupby('household_key').quantity.max()
f['maxtrip_qty28'] = mx
f['share_qty_toptrip'] = mx/f.qty.clip(lower=1)

tt = train_targets()
t431 = tt[tt.snapshot_day==431].set_index('household_key').future_spend_4w
f = f.join(t431.rename('y'), how='inner')
print('n hh at 431:', len(f))
for c in ['ap28','price_trend','qty_trend','maxtrip_qty28','share_qty_toptrip','qty','qty_p']:
    print(c, 'spearman:', round(f[c].corr(f.y.rank(), method='spearman'),3))

# log transforms of existing top features
df0 = load_saved('e010_decay.parquet')
d431 = df0[df0.snapshot_day==431].set_index('household_key')
d431 = d431.join(t431.rename('y'), how='inner')
for c in ['rwspend84','spend_per_day28','lag_mean_1_4','dec_spend14','spend28','spend112']:
    print('log1p(%s)'%c, 'spearman:', round(np.log1p(d431[c]).corr(d431.y.rank(), method='spearman'),3))


# ---- cell ----
v = snapshot()
print('day:', v.day, 'week:', v.week, 'n hh:', len(v.households))
t = v.transactions
print('tx day max:', t.day.max())
hh = set(v.households)
t = t[t.household_key.isin(hh)]
day = v.day

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
t459 = tt[tt.snapshot_day==459].set_index('household_key').future_spend_4w
f = f.join(t459.rename('y'), how='inner')
print('n hh:', len(f))
for c in ['ap28','price_trend','qty_trend','maxtrip_qty28','share_qty_toptrip','qty','qty_p']:
    print(c, 'spearman:', round(f[c].corr(f.y.rank(), method='spearman'),3))

df0 = load_saved('e010_decay.parquet')
d459 = df0[df0.snapshot_day==459].set_index('household_key').join(t459.rename('y'), how='inner')
for c in ['rwspend84','spend_per_day28','lag_mean_1_4','dec_spend14','spend28','spend112']:
    print('log1p(%s)'%c, 'spearman:', round(np.log1p(d459[c]).corr(d459.y.rank(), method='spearman'),3))

# ---- cell ----
v = snapshot()
print('day:', v.day, 'week:', v.week)
print('n hh:', len(v.households))
t = v.transactions
print('tx rows:', len(t), 'day max:', t.day.max())

# ---- cell ----
v = snapshot()
print('day:', v.day, 'week:', v.week)
t = v.transactions
print(type(t))
print('tx rows:', len(t), 'day max:', t.day.max())
print('n hh:', v.households.nunique())

# ---- cell ----
v = snapshot(431)
print('day:', v.day)
print('households:', type(v.households), (v.households if v.households is None else len(v.households)))
t = v.transactions
print('tx rows:', len(t), 'day max:', t.day.max())
hh = set(v.households)
t = t[t.household_key.isin(hh)]
day = v.day

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
print('n hh:', len(f))
for c in ['ap28','price_trend','qty_trend','maxtrip_qty28','share_qty_toptrip','qty','qty_p']:
    print(c, 'spearman:', round(f[c].corr(f.y.rank(), method='spearman'),3))

df0 = load_saved('e010_decay.parquet')
d431 = df0[df0.snapshot_day==431].set_index('household_key').join(t431.rename('y'), how='inner')
for c in ['rwspend84','spend_per_day28','lag_mean_1_4','dec_spend14','spend28','spend112']:
    print('log1p(%s)'%c, 'spearman:', round(np.log1p(d431[c]).corr(d431.y.rank(), method='spearman'),3))

# ---- cell ----
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

# ---- cell ----
def new_feats(view, snapshot_day):
    day = snapshot_day
    t = view.transactions
    hh = list(view.households)
    t = t[t.household_key.isin(hh)]
    idx = pd.Index(hh, name='household_key')

    def win(lo, hi):
        return t[(t.day > day - lo) & (t.day <= day - hi)]

    def agg(lo, hi):
        w = win(lo, hi)
        return w.groupby('household_key').agg(spend=('sales_value','sum'), qty=('quantity','sum'), ntrip=('basket_id','nunique'))

    a = {k: agg(*v) for k, v in {'7':(7,0),'28':(28,0),'56':(56,0),'112':(112,0),'364':(364,0),
                                 'p28':(56,28),'p112':(224,112),'p364':(728,364)}.items()}
    f = pd.DataFrame(index=idx)
    for k in a:
        a[k] = a[k].reindex(idx).fillna(0)
        f['spend_'+k] = a[k].spend; f['qty_'+k] = a[k].qty
    # unit price = spend per unit (qty clipped at 1)
    for k in ['7','28','56','112','364','p28','p112','p364']:
        f['ap_'+k] = f['spend_'+k] / f['qty_'+k].clip(lower=1)
    # price trends
    f['price_trend_28'] = f['ap_28'] / f['ap_p28'].clip(lower=0.25)
    f['price_trend_112'] = f['ap_112'] / f['ap_p112'].clip(lower=0.25)
    f['price_trend_364'] = f['ap_364'] / f['ap_p364'].clip(lower=0.25)
    # quantity dynamics
    f['qty_trend_28'] = f['qty_28'] / f['qty_p28'].clip(lower=1)
    f['qty_trend_112'] = f['qty_112'] / f['qty_p112'].clip(lower=1)
    f['upt28'] = f['qty_28'] / a['28'].ntrip.clip(lower=1)          # units per trip 28d
    f['upt112'] = f['qty_112'] / a['112'].ntrip.clip(lower=1)
    # max single-trip quantity in 28d
    w28 = win(28,0)
    gq = w28.groupby(['household_key','basket_id']).quantity.sum()
    mx = gq.groupby('household_key').max().reindex(idx).fillna(0)
    f['maxtrip_qty28'] = mx
    f['share_qty_toptrip'] = mx / f['qty_28'].clip(lower=1)
    # momentum ratios
    f['mom_28_112'] = f['spend_28'] / f['spend_112'].clip(lower=1)
    f['mom_112_364'] = f['spend_112'] / f['spend_364'].clip(lower=1)
    return f.reset_index()

bt = build_features(new_feats)
print('built:', bt.shape)
e10 = load_saved('e010_decay.parquet')
print('e010:', e10.shape)
newcols = [c for c in bt.columns if c not in ('household_key','snapshot_day')]
comb = e10.merge(bt[['household_key','snapshot_day']+newcols], on=['household_key','snapshot_day'], how='left')
print('combined:', comb.shape)
print('new cols:', newcols)
print('nan share in new cols:', comb[newcols].isna().mean().round(3).to_dict())
p = save_table(comb, 'e011_price.parquet')
print(p)