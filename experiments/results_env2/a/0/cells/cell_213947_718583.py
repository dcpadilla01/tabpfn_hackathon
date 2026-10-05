import numpy as np, pandas as pd, time

def feats(view, day):
    t = view.table('transactions')
    idx = view.households
    t = t[['household_key','basket_id','day','sales_value','quantity','product_id','store_id',
           'coupon_disc','retail_disc','coupon_match_disc','week_no']]
    out = pd.DataFrame(index=idx)
    g = t.groupby('household_key')
    out['last_day'] = g['day'].max()
    out['first_day'] = g['day'].min()
    def wsum(lo, hi=None):
        m = t['day'] > day - lo
        if hi is not None: m &= t['day'] <= day - hi
        return t[m].groupby('household_key')['sales_value'].sum()
    out['spend_7']   = wsum(7)
    out['spend_28']  = wsum(28)
    out['spend_56']  = wsum(56)
    out['spend_84']  = wsum(84)
    out['spend_168'] = wsum(168)
    out['spend_all'] = g['sales_value'].sum()
    out['spend_prev28'] = wsum(56, 28)
    out['spend_prev56'] = wsum(112, 56)
    def wtrips(lo, hi=None):
        m = t['day'] > day - lo
        if hi is not None: m &= t['day'] <= day - hi
        return t[m].groupby('household_key')['basket_id'].nunique()
    out['trips_28'] = wtrips(28)
    out['trips_84'] = wtrips(84)
    out['trips_prev28'] = wtrips(56, 28)
    # basket-level stats last 84d
    b = t[t['day'] > day-84].groupby(['household_key','basket_id']).agg(
        d=('day','first'), s=('sales_value','sum'), n=('product_id','count'))
    bg = b.groupby(level=0)
    out['basket_mean_84'] = bg['s'].mean()
    out['basket_std_84'] = bg['s'].std()
    out['basket_max_84'] = bg['s'].max()
    out['items_per_basket_84'] = bg['n'].mean()
    out['median_gap_84'] = bg['d'].apply(lambda x: np.median(np.diff(np.sort(x.values))) if len(x) > 1 else np.nan)
    out['active_weeks_84'] = b.reset_index().groupby('household_key')['d'].apply(
        lambda x: x.apply(lambda d: (day - d)//7).nunique())
    out['n_products_84'] = t[t['day'] > day-84].groupby('household_key')['product_id'].nunique()
    out['n_stores_84'] = t[t['day'] > day-84].groupby('household_key')['store_id'].nunique()
    out['coupon_disc_28'] = t[t['day'] > day-28].groupby('household_key')['coupon_disc'].sum().abs()
    out['retail_disc_84'] = t[t['day'] > day-84].groupby('household_key')['retail_disc'].sum().abs()
    out['qty_28'] = t[t['day'] > day-28].groupby('household_key')['quantity'].sum()
    # derived
    out['recency'] = day - out['last_day']
    out['tenure'] = day - out['first_day']
    out['avg_basket_28'] = out['spend_28'] / out['trips_28'].clip(lower=1)
    out['weekly_rate_84'] = out['spend_84'] / 12.0
    out['ratio_28_prev'] = out['spend_28'] / (out['spend_prev28'] + 5.0)
    out['trend_28_84'] = out['spend_28'] / (out['spend_84']/3.0 + 5.0)
    out['spend_28_log'] = np.log1p(out['spend_28'])
    out['spend_84_log'] = np.log1p(out['spend_84'])
    out = out.drop(columns=['last_day','first_day'])
    return out

t0 = time.time()
F = agent_api.build_features(feats)
print('build', time.time()-t0, F.shape)
print(F.columns.tolist())
print(F.isna().mean().sort_values(ascending=False).head(8))
agent_api.save_table(F, 'feats_v1')
