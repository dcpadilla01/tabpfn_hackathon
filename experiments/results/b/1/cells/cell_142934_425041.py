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
