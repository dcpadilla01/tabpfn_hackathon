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