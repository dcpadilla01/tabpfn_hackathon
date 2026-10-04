import numpy as np, pandas as pd

def make(view, day):
    tr = view.transactions
    hh = view.households
    def agg(w, suffix):
        t = tr[tr.day > day - w]
        g = t.groupby('household_key').agg(
            spend=('sales_value','sum'),
            trips=('basket_id','nunique'),
            lines=('sales_value','size'),
            qty=('quantity','sum'),
            disc=('retail_disc','sum'),
        )
        g.columns = [f'{c}_{suffix}' for c in g.columns]
        return g
    f = agg(28,'w28').join(agg(56,'w56'), how='outer').join(agg(84,'w84'), how='outer').join(agg(112,'w112'), how='outer')
    # days since last purchase
    last = tr.groupby('household_key').day.max()
    f['days_since_last'] = day - last
    # basket value
    b = tr[tr.day>day-56].groupby(['household_key','basket_id']).sales_value.sum()
    f['basket_mean_w56'] = b.groupby('household_key').mean()
    f['basket_max_w56'] = b.groupby('household_key').max()
    # distinct stores
    f['stores_w84'] = tr[tr.day>day-84].groupby('household_key').store_id.nunique()
    # spend trend: recent 28 / previous 28
    prev = tr[(tr.day>day-56)&(tr.day<=day-28)].groupby('household_key').sales_value.sum()
    f['spend_prev28'] = prev
    f = f.reindex(hh)
    f = f.fillna({'spend_w28':0.0,'spend_w56':0.0,'spend_w84':0.0,'spend_w112':0.0,
                  'trips_w28':0,'trips_w56':0,'trips_w84':0,'trips_w112':0,
                  'lines_w28':0,'lines_w56':0,'lines_w84':0,'lines_w112':0,
                  'qty_w28':0.0,'qty_w56':0.0,'qty_w84':0.0,'qty_w112':0.0,
                  'disc_w28':0.0,'disc_w56':0.0,'disc_w84':0.0,'disc_w112':0.0,
                  'spend_prev28':0.0,'basket_mean_w56':0.0,'basket_max_w56':0.0,'stores_w84':0})
    f['days_since_last'] = f['days_since_last'].fillna(999)
    f['spend_ratio_28_56'] = f.spend_w28 / (f.spend_w56+1e-6)
    return f

tab = agent_api.build_features(make)
print(tab.shape, tab.snapshot_day.nunique())
p = agent_api.save_table(tab, 'recency_agg')
print(p)
