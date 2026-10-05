import pandas as pd, numpy as np

t = agent_api.load_saved('e003_full.parquet')
print('e003 shape', t.shape)
print('days', sorted(t.snapshot_day.unique().tolist()))
print('last cols', list(t.columns)[-8:])
tt = agent_api.train_targets()
d = tt['future_spend_4w']
print('target mean/med/std/zeros', round(float(d.mean()),2), round(float(d.median()),2), round(float(d.std()),2), round(float((d==0).mean()),3))

BASE = 'e003_full.parquet'

def compute_new(tx, hh, day):
    tx = tx[['household_key','basket_id','day','sales_value','coupon_disc','retail_disc','coupon_match_disc']]
    f = pd.DataFrame(index=hh)
    def agg(mask):
        sub = tx[mask]
        g = sub.groupby('household_key')
        return g.sales_value.sum(), g.basket_id.nunique()
    # lag-aligned 4-week windows + same-window seasonality
    for name, w1, w2 in [('lag2',56,28),('lag3',84,56),('seas1y',392,364),('seas2y',756,728)]:
        s, tr = agg((tx.day > day-w1) & (tx.day <= day-w2))
        f['n7_sp_'+name] = s.reindex(hh).fillna(0.0)
        f['n7_tr_'+name] = tr.reindex(hh).fillna(0.0)
    # logs of key trailing spends
    for name, w in [('28',28),('84',84),('364',364)]:
        s, _ = agg((tx.day > day-w) & (tx.day <= day))
        f['n7_log_sp'+name] = np.log1p(s.reindex(hh).fillna(0.0))
    g = tx.groupby('household_key').day
    f['n7_tenure'] = (day - g.min()).reindex(hh)
    f['n7_recency'] = (day - g.max()).reindex(hh)
    f['n7_had_hist_1y'] = (f['n7_tenure'] >= 364).astype(float)
    # promotion engagement
    for name, w in [('28',28),('84',84)]:
        sub = tx[(tx.day > day-w) & (tx.day <= day)]
        gg = sub.groupby('household_key')
        tot = {}
        for col, tag in [('coupon_disc','cd'),('retail_disc','rd'),('coupon_match_disc','md')]:
            v = gg[col].sum().reindex(hh).fillna(0.0)
            f['n7_'+tag+name] = v
            tot[tag] = v
        sp = gg.sales_value.sum().reindex(hh).fillna(0.0)
        disc = tot['cd']+tot['rd']+tot['md']
        f['n7_disc_share'+name] = (disc/sp.clip(lower=1.0)).clip(0,3)
    return f

def fn(view, day):
    base = agent_api.load_saved(BASE)
    base = base[base['snapshot_day']==day].drop(columns=['snapshot_day']).set_index('household_key')
    hh = pd.Index(view.households, name='household_key')
    tx = view.table('transactions')
    newf = compute_new(tx, hh, day)
    return base.join(newf, how='left')

out = agent_api.build_features(fn)
print('built', out.shape)
print('new cols present:', [c for c in out.columns if c.startswith('n7_')])
p = agent_api.save_table(out, 'e007_temporal.parquet')
print('saved', p)


# ---- cell ----
import pandas as pd, numpy as np

base_all = agent_api.load_saved('e003_full.parquet')
print('base', base_all.shape)

def compute_new(tx, hh, day):
    tx = tx[['household_key','basket_id','day','sales_value','coupon_disc','retail_disc','coupon_match_disc']]
    f = pd.DataFrame(index=hh)
    def agg(mask):
        sub = tx[mask]
        g = sub.groupby('household_key')
        return g.sales_value.sum(), g.basket_id.nunique()
    for name, w1, w2 in [('lag2',56,28),('lag3',84,56),('seas1y',392,364),('seas2y',756,728)]:
        s, tr = agg((tx.day > day-w1) & (tx.day <= day-w2))
        f['n7_sp_'+name] = s.reindex(hh).fillna(0.0)
        f['n7_tr_'+name] = tr.reindex(hh).fillna(0.0)
    for name, w in [('28',28),('84',84),('364',364)]:
        s, _ = agg((tx.day > day-w) & (tx.day <= day))
        f['n7_log_sp'+name] = np.log1p(s.reindex(hh).fillna(0.0))
    g = tx.groupby('household_key').day
    f['n7_tenure'] = (day - g.min()).reindex(hh)
    f['n7_recency'] = (day - g.max()).reindex(hh)
    f['n7_had_hist_1y'] = (f['n7_tenure'] >= 364).astype(float)
    for name, w in [('28',28),('84',84)]:
        sub = tx[(tx.day > day-w) & (tx.day <= day)]
        gg = sub.groupby('household_key')
        tot = {}
        for col, tag in [('coupon_disc','cd'),('retail_disc','rd'),('coupon_match_disc','md')]:
            v = gg[col].sum().reindex(hh).fillna(0.0)
            f['n7_'+tag+name] = v
            tot[tag] = v
        sp = gg.sales_value.sum().reindex(hh).fillna(0.0)
        disc = tot['cd']+tot['rd']+tot['md']
        f['n7_disc_share'+name] = (disc/sp.clip(lower=1.0)).clip(0,3)
    return f

def fn(view, day, _base=base_all):
    base = _base[_base['snapshot_day']==day].drop(columns=['snapshot_day']).set_index('household_key')
    hh = pd.Index(view.households, name='household_key')
    tx = view.table('transactions')
    newf = compute_new(tx, hh, day)
    return base.join(newf, how='left')

out = agent_api.build_features(fn)
print('built', out.shape)
print('new cols present:', [c for c in out.columns if c.startswith('n7_')])
p = agent_api.save_table(out, 'e007_temporal.parquet')
print('saved', p)
