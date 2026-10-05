
import agent_api as api, pandas as pd, numpy as np, time

def make_features(view, snapshot_day):
    day = snapshot_day
    hh = pd.Index(view.households, name='household_key')
    tx = view.table('transactions')
    tx = tx[tx.day <= day]
    g = tx.groupby('household_key')
    base = pd.DataFrame(index=hh)
    base['spend_total'] = g.sales_value.sum()
    base['tenure'] = day - g.day.min()
    base['nbask_total'] = g.basket_id.nunique()
    base['nprod_total'] = g.product_id.nunique()
    base['nstore_total'] = g.store_id.nunique()
    base['spend_per_basket_total'] = base.spend_total / base.nbask_total.clip(lower=1)
    base['spend_per_day_total'] = base.spend_total / base.tenure.clip(lower=1)
    base['avg_price_line'] = base.spend_total / g.size()
    base['avg_units_line'] = g.quantity.sum() / g.size()
    base['neg_spend_share_total'] = tx.assign(neg=tx.sales_value < 0).groupby('household_key').neg.mean()
    for c in ['coupon_disc', 'coupon_match_disc', 'retail_disc']:
        base['sum_' + c] = g[c].sum()
    base['disc_share_total'] = (base.sum_coupon_disc + base.sum_coupon_match_disc + base.sum_retail_disc) / base.spend_total.clip(lower=1)

    w = tx[tx.day >= day - 111]
    gw = w.groupby('household_key')
    base['spend_112'] = gw.sales_value.sum()
    base['nbask_112'] = gw.basket_id.nunique()
    base['spend_per_basket_112'] = base.spend_112 / base.nbask_112.clip(lower=1)
    base['nprod_112'] = gw.product_id.nunique()
    base['nstore_112'] = gw.store_id.nunique()
    base['nweek_active_112'] = gw.week_no.nunique()
    base['week_active_share_112'] = base.nweek_active_112 / 16
    base['spend_per_week_112'] = base.spend_112 / 16
    base['spend_per_active_week_112'] = base.spend_112 / base.nweek_active_112.clip(lower=1)
    base['avg_price_line_112'] = base.spend_112 / gw.size()
    base['avg_units_line_112'] = gw.quantity.sum() / gw.size()
    dsum = w.groupby('household_key')[['coupon_disc', 'coupon_match_disc', 'retail_disc']].sum().sum(axis=1)
    base['disc_share_112'] = dsum / base.spend_112.clip(lower=1)
    base['neg_spend_share_112'] = w.assign(neg=w.sales_value < 0).groupby('household_key').neg.mean()
    base['mean_trans_time_112'] = gw.trans_time.mean()

    w2 = w.copy()
    w2['widx'] = (day - w2.day) // 28
    s4 = w2.groupby(['household_key', 'widx']).sales_value.sum().unstack(fill_value=0.0)
    for k in range(4):
        if k not in s4.columns:
            s4[k] = 0.0
    s4 = s4[range(4)]
    base['spend_4w_recent'] = s4[0]
    base['spend_4w_lag1'] = s4[1]
    base['spend_4w_lag2'] = s4[2]
    base['spend_4w_lag3'] = s4[3]
    base['spend_112_std'] = s4.std(axis=1)
    base['spend_112_mean4'] = s4.mean(axis=1)
    base['spend_112_cv'] = base.spend_112_std / base.spend_112_mean4.clip(lower=1e-6)
    base['spend_112_min4'] = s4.min(axis=1)
    base['spend_112_max4'] = s4.max(axis=1)
    base['spend_112_range'] = base.spend_112_max4 - base.spend_112_min4

    def gapstats(days):
        d = np.sort(np.asarray(days))
        if len(d) < 2:
            return (np.nan, np.nan, np.nan)
        gg = np.diff(d)
        return (gg.mean(), gg.max(), gg.std())
    gs = w.groupby('household_key').day.unique().apply(gapstats)
    base['gap_mean_112'] = gs.apply(lambda t: t[0])
    base['gap_max_112'] = gs.apply(lambda t: t[1])
    base['gap_std_112'] = gs.apply(lambda t: t[2])

    pr = view.table('products')
    wd = w.merge(pr[['product_id', 'department', 'brand']], on='product_id', how='left')
    wd['brand'] = wd['brand'].astype(str)
    share = pd.crosstab(wd.household_key, wd.department, wd.sales_value, aggfunc='sum', normalize='index')
    share.columns = ['dsh_' + str(c) for c in share.columns]
    brand = pd.crosstab(wd.household_key, wd.brand, wd.sales_value, aggfunc='sum', normalize='index')
    brand.columns = ['bsh_' + str(c) for c in brand.columns]

    out = base.join([share, brand])
    out['snap_day'] = day
    out['snap_week'] = view.week
    out = out.reindex(hh)
    num_cols = [c for c in out.columns if out[c].dtype.kind in 'fiub']
    for c in out.columns:
        if c not in num_cols:
            out[c] = out[c].astype('object').fillna('0')
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

t0 = time.time()
df = api.build_features(make_features)
print('elapsed', round(time.time()-t0,1), 'shape', df.shape)
p = api.save_table(df, 'e002_composition')
print(p)
