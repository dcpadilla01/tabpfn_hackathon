
import agent_api as api, pandas as pd, numpy as np, time

def probe(view, snapshot_day):
    print('day', view.day, 'week', view.week, 'type hh', type(view.households))
    try:
        print('hh sample', list(view.households)[:3])
    except Exception as e:
        print('hh err', e)
    tx = view.table('transactions')
    print('tx shape', tx.shape, 'cols ok')
    pr = view.table('products')
    print('products', pr.shape, 'depts', pr.department.nunique())
    cr = view.table('coupon_redemptions')
    print('coupon_red', cr.shape)
    cp = view.table('coupons')
    print('coupons', cp.shape)
    tg = view.table('campaign_targets')
    print('targets', tg.shape)
    return pd.DataFrame(index=list(view.households))

t0=time.time()
df = api.build_features(probe)
print('elapsed', round(time.time()-t0,1), 'rows', df.shape)
