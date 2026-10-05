t = load_saved('e011_table.parquet')
print('e011_table', t.shape)
print('has keys:', 'household_key' in t.columns, 'snapshot_day' in t.columns)
cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
print('n_feat', len(cols))
import re
groups = {}
for c in cols:
    g = re.split(r'(?=[A-Z])|_', c)[0]
    groups.setdefault(g, []).append(c)
for g in sorted(groups):
    cs = groups[g]
    print(g, len(cs), cs[:6])
print()
print('snapshot_days:', snapshot_days())


# ---- cell ----
tt = train_targets()
y = tt.future_spend_4w
print('n', len(tt), 'mean %.1f median %.1f zero%% %.3f' % (y.mean(), y.median(), (y==0).mean()))
print('quantiles', y.quantile([.5,.75,.9,.95,.99]).to_dict())
v = snapshot()
tr = v.transactions
print('transactions up to 459:', tr.shape)
tr2 = tr.assign(dow=tr.day % 7)
print(tr2.groupby('dow').sales_value.agg(['mean','count']))
print('trans_time quantiles', tr.trans_time.quantile([.1,.5,.9]).to_dict())
d = tr[['sales_value','coupon_disc','coupon_match_disc','retail_disc','quantity']]
print(d.describe().loc[['mean','50%','max']])


# ---- cell ----
for name in ['selfcal_v1','churn_vol_v1','mkt_v1','comp_v1','ewma_block_v1','rfm_traj_v1','lvl_v1']:
    t = load_saved(name + '.parquet')
    cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
    print(name, t.shape, 'feat:', cols[:30])
    print()


# ---- cell ----
v = snapshot()
cr = v.coupon_redemptions
print('coupon_redemptions', cr.shape, cr.columns.tolist())
print(cr.head())
print('n households redeeming:', cr.household_key.nunique())
print('redemptions per day quantiles:', cr.groupby('day').size().quantile([.5,.9]).to_dict())
tr = v.transactions
# promo dependence quick check
import numpy as np
tr['promo'] = (tr.coupon_disc>0)|(tr.coupon_match_disc>0)|(tr.retail_disc>0)
print('share of line items with any promo:', tr.promo.mean())
print('share of spend with promo:', (tr.sales_value[tr.promo]).sum()/tr.sales_value.sum())


# ---- cell ----
t = load_saved('e011_table.parquet')
cols = sorted(c for c in t.columns if c not in ('household_key','snapshot_day'))
print('\n'.join(cols))


# ---- cell ----
m = load_saved('mkt_v1.parquet')
mc = [c for c in m.columns if c not in ('household_key','snapshot_day')]
base = set(load_saved('churn_vol_v1.parquet').columns)
print('mkt-only feats:', [c for c in mc if c not in base])
cv = load_saved('churn_vol_v1.parquet')
bc = [c for c in cv.columns if c.startswith('b_')]
print('b_* feats:', bc)


# ---- cell ----
def timing_fn(view, snapshot_day):
    hh = view.households
    if hasattr(hh, 'columns') and 'household_key' in getattr(hh, 'columns', []):
        idx = pd.Index(hh.household_key)
    else:
        idx = pd.Index(hh)
    tr = view.table('transactions')
    tr = tr[tr.day <= snapshot_day]
    tr = tr[tr.day > snapshot_day - 112]
    out = pd.DataFrame(index=idx)
    if len(tr) == 0:
        return out
    tr = tr.assign(hour=(tr.trans_time // 100).clip(0, 23).astype(float),
                   dow=(tr.day % 7).astype(int))
    # basket-level style
    bs = tr.groupby(['household_key', 'basket_id']).agg(
        sp=('sales_value', 'sum'), hour=('hour', 'first'), dow=('dow', 'first')).reset_index()
    g = bs.groupby('household_key')
    out['t_hour_mean'] = g.hour.mean()
    out['t_hour_std'] = g.hour.std()
    # spend shares by time of day
    sp_tot = tr.groupby('household_key').sales_value.sum().rename('tot')
    m = tr[(tr.hour >= 7) & (tr.hour <= 11)].groupby('household_key').sales_value.sum()
    e = tr[(tr.hour >= 17) & (tr.hour <= 22)].groupby('household_key').sales_value.sum()
    out['t_morning_sp_share'] = (m / sp_tot)
    out['t_evening_sp_share'] = (e / sp_tot)
    # day-of-week concentration of spend
    dow_sp = tr.groupby(['household_key', 'dow']).sales_value.sum()
    share = dow_sp / dow_sp.groupby('household_key').transform('sum')
    out['t_dow_hhi'] = (share ** 2).groupby('household_key').sum()
    out['t_n_dow'] = tr[tr.sales_value > 0].groupby('household_key').dow.nunique()
    # store breadth in window
    out['t_n_stores'] = tr.groupby('household_key').store_id.nunique()
    st = tr.groupby(['household_key', 'store_id']).sales_value.sum().reset_index()
    top = st.sort_values('sales_value', ascending=False).groupby('household_key').head(1).set_index('household_key')
    out['t_store_share'] = top.sales_value / sp_tot
    return out.reindex(idx)

bt = build_features(timing_fn)
print(bt.shape)
print(bt.head(3))
print(bt.isna().mean().round(3).to_dict())
save_table(bt, 'timing_v1')
print('saved timing_v1')


# ---- cell ----
l = load_saved('lvl_v1.parquet')
print(l.shape, l.columns.tolist())
sub = l[l.snapshot_day == 95]
hh0 = sub.household_key.iloc[0]
row = sub[sub.household_key == hh0].iloc[0]
v95 = snapshot(95)
tr = v95.transactions
trh = tr[(tr.household_key == hh0) & (tr.day > 95 - 364) & (tr.day <= 95)]
print('hh', hh0)
print('recomputed nprod_364', trh.product_id.nunique(), '| saved', row.nprod_364)
print('recomputed nbask_364', trh.basket_id.nunique(), '| saved', row.nbask_364)
st = trh.groupby('store_id').sales_value.sum()
print('recomputed top-store share %.4f' % (st.max() / st.sum()), '| saved store_share %.4f' % row.store_share)
print('recomputed sales/qty %.4f' % (trh.sales_value.sum() / trh.quantity.sum()), '| saved unit_price_364 %.4f' % row.unit_price_364)
# also check a validation-snapshot row for plausibility (no leak check possible directly, but verify shape/coverage)
print(l.groupby('snapshot_day').size())
print('NaN frac:', l.isna().mean().round(3).to_dict())


# ---- cell ----
e011 = load_saved('e011_table.parquet')
lvl = load_saved('lvl_v1.parquet')
tim = load_saved('timing_v1.parquet')
m = e011.merge(lvl.drop(columns=['household_key','snapshot_day']), left_index=True, right_index=True, how='left')
m = m.merge(tim.drop(columns=['household_key','snapshot_day']), left_index=True, right_index=True, how='left')
print(m.shape)
print('keys ok:', m.household_key.notna().all(), m.snapshot_day.notna().all())
print('n_feat:', len([c for c in m.columns if c not in ('household_key','snapshot_day')]))
# verify identical to e011 on parent cols
sub = m.head(50)
e = e011.head(50)
pc = [c for c in e011.columns]
print('parent cols preserved:', m[pc].equals(e011[pc]))
save_table(m, 'e014_table')
print('saved e014_table')
