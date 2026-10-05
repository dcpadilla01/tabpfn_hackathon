e6 = agent_api.load_saved('e006_dynamics.parquet')
e9 = agent_api.load_saved('e009_momentum.parquet')
print('E006', e6.shape)
print(sorted([c for c in e6.columns if c not in ('household_key','snapshot_day')]))
print()
print('E009 extra cols:', sorted([c for c in e9.columns if c not in e6.columns]))


# ---- cell ----
import numpy as np, pandas as pd

base = agent_api.load_saved('e006_dynamics.parquet')
print('base', base.shape)

def fn(view, snapshot_day):
    day = snapshot_day
    tx = view.table('transactions')
    tx = tx[tx.day <= day]
    hh = view.households.index
    f = pd.DataFrame(index=hh)

    def win(w):
        t = tx[(tx.day > day - w) & (tx.day <= day)]
        return (t.groupby('household_key').sales_value.sum(),
                t.groupby('household_key').basket_id.nunique())
    s7,_  = win(7)
    s28,tr28 = win(28); s56,tr56 = win(56); s84,tr84 = win(84)
    s182,tr182 = win(182); s365,tr365 = win(365)

    g = tx.groupby('household_key').day
    rec = (day - g.max()).reindex(hh)
    ten = (day - g.min()).reindex(hh)

    age = (day - tx.day).clip(lower=0).astype(float)
    for hl, nm in [(14,'14'),(28,'28'),(112,'112')]:
        w = np.power(0.5, age/hl)
        f['x_ewma'+nm] = (tx.sales_value*w).groupby(tx.household_key).sum().reindex(hh)

    blocks = {}
    for k in range(1,7):
        t = tx[(tx.day > day-28*k) & (tx.day <= day-28*(k-1))]
        blocks[k] = t.groupby('household_key').sales_value.sum()
    b = pd.DataFrame(blocks).reindex(hh).fillna(0.0)
    bstd = b.std(axis=1)

    t84 = tx[(tx.day > day-84) & (tx.day <= day)].sort_values(['household_key','day'])
    prev = t84.groupby('household_key').day.shift()
    gap = (t84.day - prev).groupby(t84.household_key).mean().reindex(hh)

    s28=s28.reindex(hh).fillna(0); s56=s56.reindex(hh).fillna(0)
    s84=s84.reindex(hh).fillna(0); s182=s182.reindex(hh).fillna(0)
    s365=s365.reindex(hh).fillna(0); s7=s7.reindex(hh).fillna(0)
    tr28=tr28.reindex(hh).fillna(0); tr84=tr84.reindex(hh).fillna(0)
    tr182=tr182.reindex(hh).fillna(0)
    gap=gap.fillna(999.0)

    f['x_lg_spend_7']  = np.log1p(s7)
    f['x_lg_spend_28'] = np.log1p(s28)
    f['x_lg_spend_56'] = np.log1p(s56)
    f['x_lg_spend_84'] = np.log1p(s84)
    f['x_lg_spend_182']= np.log1p(s182)
    f['x_lg_spend_365']= np.log1p(s365)
    f['x_lg_trips_28'] = np.log1p(tr28)
    f['x_lg_trips_84'] = np.log1p(tr84)
    f['x_lg_trips_182']= np.log1p(tr182)
    f['x_lg_recency']  = np.log1p(rec.fillna(999))
    f['x_lg_tenure']   = np.log1p(ten.fillna(0))
    f['x_lg_gap']      = np.log1p(gap.clip(upper=999))
    f['x_lg_ewma28']   = np.log1p(f['x_ewma28'].fillna(0))
    f['x_inact28']     = (s28 <= 0)
    f['x_inact56']     = (s56 <= 0)
    f['x_returning']   = (s28 <= 0) & (s84 > 0)
    f['x_lowtrips28']  = (tr28 <= 1)
    f['x_block_cv']    = bstd / (s182/6.5 + 1.0)
    f['x_ewma14_112']  = f['x_ewma14'] / (f['x_ewma112'] + 1.0)
    f['x_rec_gap']     = rec.fillna(999) / (gap + 1.0)
    f['x_lg_s84_x_ten']= f['x_lg_spend_84'] * f['x_lg_tenure']
    f['x_lg_ew28_x_ten']= f['x_lg_ewma28'] * f['x_lg_tenure']
    return f

feats = agent_api.build_features(fn)
print('feats', feats.shape)
m = base.merge(feats.drop(columns=['index'], errors='ignore'), on=agent_api.KEYS, how='inner')
print('merged', m.shape)
assert len(m) == len(base) == len(feats)
path = agent_api.save_table(m, 'e010_logstate.parquet')
print(path)


# ---- cell ----
import numpy as np, pandas as pd

base = agent_api.load_saved('e006_dynamics.parquet')

def fn(view, snapshot_day):
    day = snapshot_day
    tx = view.table('transactions')
    tx = tx[tx.day <= day]
    hh = view.households
    f = pd.DataFrame(index=hh)

    def win(w):
        t = tx[(tx.day > day - w) & (tx.day <= day)]
        return (t.groupby('household_key').sales_value.sum(),
                t.groupby('household_key').basket_id.nunique())
    s7,_  = win(7)
    s28,tr28 = win(28); s56,tr56 = win(56); s84,tr84 = win(84)
    s182,tr182 = win(182); s365,tr365 = win(365)

    g = tx.groupby('household_key').day
    rec = (day - g.max()).reindex(hh)
    ten = (day - g.min()).reindex(hh)

    age = (day - tx.day).clip(lower=0).astype(float)
    for hl, nm in [(14,'14'),(28,'28'),(112,'112')]:
        w = np.power(0.5, age/hl)
        f['x_ewma'+nm] = (tx.sales_value*w).groupby(tx.household_key).sum().reindex(hh)

    blocks = {}
    for k in range(1,7):
        t = tx[(tx.day > day-28*k) & (tx.day <= day-28*(k-1))]
        blocks[k] = t.groupby('household_key').sales_value.sum()
    b = pd.DataFrame(blocks).reindex(hh).fillna(0.0)
    bstd = b.std(axis=1)

    t84 = tx[(tx.day > day-84) & (tx.day <= day)].sort_values(['household_key','day'])
    prev = t84.groupby('household_key').day.shift()
    gap = (t84.day - prev).groupby(t84.household_key).mean().reindex(hh)

    s28=s28.reindex(hh).fillna(0); s56=s56.reindex(hh).fillna(0)
    s84=s84.reindex(hh).fillna(0); s182=s182.reindex(hh).fillna(0)
    s365=s365.reindex(hh).fillna(0); s7=s7.reindex(hh).fillna(0)
    tr28=tr28.reindex(hh).fillna(0); tr84=tr84.reindex(hh).fillna(0)
    tr182=tr182.reindex(hh).fillna(0)
    gap=gap.fillna(999.0)

    f['x_lg_spend_7']  = np.log1p(s7)
    f['x_lg_spend_28'] = np.log1p(s28)
    f['x_lg_spend_56'] = np.log1p(s56)
    f['x_lg_spend_84'] = np.log1p(s84)
    f['x_lg_spend_182']= np.log1p(s182)
    f['x_lg_spend_365']= np.log1p(s365)
    f['x_lg_trips_28'] = np.log1p(tr28)
    f['x_lg_trips_84'] = np.log1p(tr84)
    f['x_lg_trips_182']= np.log1p(tr182)
    f['x_lg_recency']  = np.log1p(rec.fillna(999))
    f['x_lg_tenure']   = np.log1p(ten.fillna(0))
    f['x_lg_gap']      = np.log1p(gap.clip(upper=999))
    f['x_lg_ewma28']   = np.log1p(f['x_ewma28'].fillna(0))
    f['x_inact28']     = (s28 <= 0)
    f['x_inact56']     = (s56 <= 0)
    f['x_returning']   = (s28 <= 0) & (s84 > 0)
    f['x_lowtrips28']  = (tr28 <= 1)
    f['x_block_cv']    = bstd / (s182/6.5 + 1.0)
    f['x_ewma14_112']  = f['x_ewma14'] / (f['x_ewma112'] + 1.0)
    f['x_rec_gap']     = rec.fillna(999) / (gap + 1.0)
    f['x_lg_s84_x_ten']= f['x_lg_spend_84'] * f['x_lg_tenure']
    f['x_lg_ew28_x_ten']= f['x_lg_ewma28'] * f['x_lg_tenure']
    return f

feats = agent_api.build_features(fn)
print('feats', feats.shape)
m = base.merge(feats.drop(columns=['index'], errors='ignore'), on=agent_api.KEYS, how='inner')
print('merged', m.shape)
assert len(m) == len(base) == len(feats)
path = agent_api.save_table(m, 'e010_logstate.parquet')
print(path)
