import numpy as np, pandas as pd

def probe(view, day):
    out = []
    out.append(('day', day, view.day, view.week))
    out.append(('n_hh', len(view.households), None, None))
    try:
        t = agent_api.train_targets()
        out.append(('train_targets', 'OK ' + str(t.shape), None, None))
    except Exception as e:
        out.append(('train_targets', 'ERR ' + repr(e)[:90], None, None))
    try:
        d = agent_api.load_saved('mkt_v2.parquet')
        out.append(('load_saved', 'OK ' + str(d.shape), None, None))
    except Exception as e:
        out.append(('load_saved', 'ERR ' + repr(e)[:90], None, None))
    try:
        h = agent_api.history(list(view.households)[:2], as_of_day=None)
        out.append(('history', 'OK ' + str(h.shape), None, None))
    except Exception as e:
        out.append(('history', 'ERR ' + repr(e)[:90], None, None))
    for row in out:
        print(row[0], '->', row[1])
    print('tx day max', view.transactions.day.max(), 'min', view.transactions.day.min())
    print('demo rows', view.demographics.shape)
    print('camp rows', view.campaigns.shape, 'targets', view.campaign_targets.shape)
    print('redemptions', view.coupon_redemptions.shape)
    print('display_mailer', view.display_mailer.shape)
    return pd.DataFrame({'x': [1.0]}, index=list(view.households)[:1])

df = agent_api.build_features(probe)
print('built', df.shape)
