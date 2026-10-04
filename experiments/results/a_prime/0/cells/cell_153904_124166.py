import numpy as np, pandas as pd

def probe(view, day):
    diag = {}
    diag['day'] = day
    diag['view_day'] = view.day
    diag['view_week'] = view.week
    diag['n_hh'] = len(view.households)
    diag['tx_max_day'] = view.transactions.day.max()
    diag['demo_rows'] = view.demographics.shape[0]
    diag['camp_rows'] = view.campaigns.shape[0]
    diag['ct_rows'] = view.campaign_targets.shape[0]
    diag['red_rows'] = view.coupon_redemptions.shape[0]
    diag['dm_rows'] = view.display_mailer.shape[0]
    diag['dm_max_week'] = view.display_mailer.week_no.max()
    try:
        t = agent_api.train_targets()
        diag['tt'] = 'OK %d' % t.shape[0]
    except Exception as e:
        diag['tt'] = 'ERR ' + repr(e)[:80]
    try:
        d = agent_api.load_saved('mkt_v2.parquet')
        diag['ls'] = 'OK %d' % d.shape[0]
    except Exception as e:
        diag['ls'] = 'ERR ' + repr(e)[:80]
    try:
        h = agent_api.history(list(view.households)[:2])
        diag['hist'] = 'OK %d' % h.shape[0]
    except Exception as e:
        diag['hist'] = 'ERR ' + repr(e)[:80]
    return pd.DataFrame(diag, index=list(view.households)[:1])

df = agent_api.build_features(probe)
print(df.drop_duplicates(subset=['snapshot_day']).drop(columns=['household_key']).T)
