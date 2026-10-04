import pandas as pd, numpy as np
api = agent_api
print(api.snapshot_days())
tx = api.snapshot().transactions
print('tx shape', tx.shape, 'day range', tx.day.min(), tx.day.max())
print(tx[['sales_value','quantity']].describe())
print('n households', tx.household_key.nunique())
bf = api.baseline_features()
print('baseline', bf.shape, bf.columns.tolist())
print(bf.head(3))
tt = api.train_targets()
print('targets', tt.shape); print(tt.head())
v = api.snapshot(95)
print('households type', type(v.households), 'n', len(v.households))
print('view day/week', v.day, v.week)
print('has demographics?', hasattr(v, 'demographics'))


# ---- cell ----
v = agent_api.snapshot(95)
print([a for a in dir(v) if not a.startswith('_')])
print('households:', v.households)
print('day', v.day, 'week', v.week)
h = agent_api.history(1)
print(h.shape, h.columns.tolist())
print(h.head(3))
print(agent_api.KEYS, agent_api.TARGET)


# ---- cell ----
def probe(view, day):
    print('inside build_features: day', day, 'households type', type(view.households))
    import numpy as np
    hh = view.households
    print('n hh', len(hh) if hh is not None else None)
    return pd.DataFrame({'x': [1.0]}, index=pd.Index(hh[:3] if hh is not None else [], name='household_key'))

df = agent_api.build_features(probe)
print(df.head())


# ---- cell ----
def probe(view, day):
    import numpy as np
    hh = view.households
    print('type', type(hh), 'n', len(hh), 'first', hh[:5] if hasattr(hh,'__getitem__') else None)
    return pd.DataFrame({'x': np.ones(len(hh))}, index=pd.Index(hh, name='household_key'))

df = agent_api.build_features(probe)
print(df.shape, df.snapshot_day.unique())
print(df.head())


# ---- cell ----
import pandas as pd, numpy as np

def fn(view, day):
    tx = view.transactions
    hh = pd.Index(view.households, name='household_key')
    g = tx.groupby('household_key')
    out = pd.DataFrame(index=hh)
    # window sums
    for w in [7, 28, 56, 84, 182, 365]:
        s = tx[tx.day > day - w].groupby('household_key').sales_value.sum()
        out[f'spend_{w}'] = s.reindex(hh).fillna(0.0)
    out['spend_all'] = g.sales_value.sum().reindex(hh).fillna(0.0)
    # baskets
    nb = tx[tx.day > day - 28].groupby('household_key').basket_id.nunique()
    out['nb_28'] = nb.reindex(hh).fillna(0)
    nb56 = tx[tx.day > day - 56].groupby('household_key').basket_id.nunique()
    out['nb_56'] = nb56.reindex(hh).fillna(0)
    out['nb_all'] = g.basket_id.nunique().reindex(hh).fillna(0)
    out['avg_basket_28'] = out['spend_28'] / out['nb_28'].replace(0, np.nan)
    last = g.day.max().reindex(hh)
    out['days_since_last'] = (day - last).astype(float)
    # previous 28d window and trend
    prev = tx[(tx.day > day - 56) & (tx.day <= day - 28)].groupby('household_key').sales_value.sum()
    out['spend_prev28'] = prev.reindex(hh).fillna(0.0)
    out['trend28'] = out['spend_28'] / (out['spend_prev28'] + 1.0)
    # items and products
    out['qty_28'] = tx[tx.day > day - 28].groupby('household_key').quantity.sum().reindex(hh).fillna(0.0)
    out['nprod_28'] = tx[tx.day > day - 28].groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    # last-year same window
    ly = tx[(tx.day > day - 364) & (tx.day <= day - 336)].groupby('household_key').sales_value.sum()
    out['spend_ly28'] = ly.reindex(hh).fillna(0.0)
    out['ly_avail'] = (day > 364).astype(float)
    out['log_spend_28'] = np.log1p(out['spend_28'])
    out['log_spend_all'] = np.log1p(out['spend_all'])
    out['weekly_rate_84'] = out['spend_84'] / 12.0
    return out

df = agent_api.build_features(fn)
bf = agent_api.baseline_features()
m = bf.merge(df.reset_index(), on=agent_api.KEYS, how='left')
print(m.shape)
p = agent_api.save_table(m, 'e001_history')
print(p)
print(m[['spend_28','spend_all','nb_28','days_since_last']].describe())


# ---- cell ----
import pandas as pd, numpy as np

def fn(view, day):
    tx = view.transactions
    hh = pd.Index(view.households, name='household_key')
    out = pd.DataFrame(index=hh)
    for w in [7, 28, 56, 84, 182, 365]:
        s = tx[tx.day > day - w].groupby('household_key').sales_value.sum()
        out[f'spend_{w}'] = s.reindex(hh).fillna(0.0)
    out['spend_all'] = tx.groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
    nb = tx[tx.day > day - 28].groupby('household_key').basket_id.nunique()
    out['nb_28'] = nb.reindex(hh).fillna(0)
    nb56 = tx[tx.day > day - 56].groupby('household_key').basket_id.nunique()
    out['nb_56'] = nb56.reindex(hh).fillna(0)
    out['nb_all'] = tx.groupby('household_key').basket_id.nunique().reindex(hh).fillna(0)
    out['avg_basket_28'] = out['spend_28'] / out['nb_28'].replace(0, np.nan)
    last = tx.groupby('household_key').day.max().reindex(hh)
    out['days_since_last'] = (day - last).astype(float)
    prev = tx[(tx.day > day - 56) & (tx.day <= day - 28)].groupby('household_key').sales_value.sum()
    out['spend_prev28'] = prev.reindex(hh).fillna(0.0)
    out['trend28'] = out['spend_28'] / (out['spend_prev28'] + 1.0)
    out['qty_28'] = tx[tx.day > day - 28].groupby('household_key').quantity.sum().reindex(hh).fillna(0.0)
    out['nprod_28'] = tx[tx.day > day - 28].groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    ly = tx[(tx.day > day - 364) & (tx.day <= day - 336)].groupby('household_key').sales_value.sum()
    out['spend_ly28'] = ly.reindex(hh).fillna(0.0)
    out['ly_avail'] = float(day > 364)
    out['log_spend_28'] = np.log1p(out['spend_28'])
    out['log_spend_all'] = np.log1p(out['spend_all'])
    out['weekly_rate_84'] = out['spend_84'] / 12.0
    return out

df = agent_api.build_features(fn)
bf = agent_api.baseline_features()
m = bf.merge(df.reset_index(), on=agent_api.KEYS, how='left')
print(m.shape)
p = agent_api.save_table(m, 'e001_history')
print(p)
print(m[['spend_28','spend_all','nb_28','days_since_last']].describe().round(2))
