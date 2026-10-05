
import agent_api, pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')

def make_features(view, snapshot_day):
    t = view.table('transactions')
    t = t[t.day <= snapshot_day]
    end = snapshot_day
    w = {}
    def agg(hh):
        g = t[t.household_key==hh]
        return g
    # vectorized per-window sums
    for name, lo in [('spend_7',7),('spend_14',14),('spend_28',28),('spend_56',56),('spend_84',84),('spend_180',180),('spend_365',365)]:
        s = t[(t.day>end-lo)&(t.day<=end)].groupby('household_key')['sales_value'].sum()
        w[name] = s
    for name, lo in [('spend_28_prior',56),('spend_84_prior',168)]:
        s = t[(t.day>end-lo)&(t.day<=end-lo//2)].groupby('household_key')['sales_value'].sum()
        w[name] = s
    b = t[(t.day>end-84)&(t.day<=end)].groupby('household_key').agg(
        baskets_84=('basket_id','nunique'), days_since_last=('day','max'),
        days_since_first=('day','min'), n_products_84=('product_id','nunique'),
        n_stores_84=('store_id','nunique'))
    b['days_since_last'] = end - b['days_since_last']
    b['days_since_first'] = end - b['days_since_first']
    b['baskets_28'] = t[(t.day>end-28)&(t.day<=end)].groupby('household_key')['basket_id'].nunique()
    b['avg_basket_84'] = b['baskets_84'].where(b['baskets_84']>0, np.nan)
    b['avg_basket_84'] = w['spend_84']/b['baskets_84'].replace(0,np.nan)
    b['trips_per_wk_84'] = b['baskets_84']/12.0
    b['spend_28_ratio'] = w['spend_28']/w['spend_28_prior'].replace(0,np.nan)
    out = pd.DataFrame(w).join(b)
    out['spend_trend'] = (w['spend_28']-w['spend_28_prior'])/28.0
    out['active_28'] = (out['baskets_28']>0).astype(float)
    out = out.reindex(view.households).fillna({'spend_7':0.0,'spend_14':0.0,'spend_28':0.0,'spend_28_prior':0.0})
    return out

t0 = time.time()
F = agent_api.build_features(make_features)
print('build_features took %.1fs, shape' % (time.time()-t0), F.shape)
print(F.head())
