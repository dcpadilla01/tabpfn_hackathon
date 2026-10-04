import numpy as np, pandas as pd
import agent_api
from agent_api import build_features, baseline_features, save_table

def get_index(hh):
    if hasattr(hh, 'columns'):
        hh = hh['household_key'].values
    return pd.Index(pd.unique(pd.Series(list(hh))), name='household_key')

def make_features(view, snapshot_day):
    idx = get_index(view.households)
    tx = view.transactions
    try:
        prods = view.products
    except AttributeError:
        prods = agent_api.snapshot(snapshot_day).products
    tx = tx.merge(prods[['product_id','department']], on='product_id', how='left')
    tx['disc'] = tx[['coupon_disc','coupon_match_disc','retail_disc']].fillna(0).sum(axis=1)
    feats = pd.DataFrame(index=idx)

    # E001-style rolling recency aggregates
    for W in (28, 56, 84, 112):
        w = tx[tx['day'] > snapshot_day - W]
        g = w.groupby('household_key')
        feats[f'spend_{W}'] = g['sales_value'].sum().reindex(idx).fillna(0.0)
        feats[f'trips_{W}'] = g['basket_id'].nunique().reindex(idx).fillna(0.0)
        feats[f'lines_{W}'] = g.size().reindex(idx).fillna(0.0)
        feats[f'qty_{W}']   = g['quantity'].sum().reindex(idx).fillna(0.0)
        feats[f'disc_{W}']  = g['disc'].sum().reindex(idx).fillna(0.0)
    feats['days_since_last'] = (snapshot_day - tx.groupby('household_key')['day'].max()).reindex(idx).fillna(180.0)
    feats['spend_ratio_28_56'] = feats['spend_28'] / feats['spend_56'].replace(0.0, np.nan)
    feats['avg_basket_28'] = feats['spend_28'] / feats['trips_28'].replace(0.0, np.nan)

    # NEW: department mix of recent spend
    d28 = (tx[tx['day'] > snapshot_day - 28]
           .groupby(['household_key','department'])['sales_value'].sum()
           .unstack(fill_value=0.0).reindex(idx).fillna(0.0))
    d28.columns = ['d28_' + str(c) for c in d28.columns]
    d112 = (tx[tx['day'] > snapshot_day - 112]
            .groupby(['household_key','department'])['sales_value'].sum()
            .unstack(fill_value=0.0).reindex(idx).fillna(0.0))
    tot112 = d112.sum(axis=1).replace(0.0, np.nan)
    sh112 = d112.div(tot112, axis=0)
    sh112.columns = ['sh112_' + str(c) for c in sh112.columns]
    feats = pd.concat([feats, d28, sh112], axis=1)

    # keep E000 baseline (demographics + calendar)
    base = baseline_features()
    if 'household_key' in base.columns:
        base = base.set_index('household_key')
    base = base.drop(columns=['snapshot_day'], errors='ignore')
    feats = feats.join(base, how='left')
    return feats

out = build_features(make_features)
print('shape:', out.shape)
print('n feat cols:', out.shape[1] - 2)
print(list(out.columns)[:40])
path = save_table(out, 'dept_mix_recency.parquet')
print('saved:', path)


# ---- cell ----
import pandas as pd, numpy as np
import agent_api
v = agent_api.snapshot(95)
hh = v.households
print(type(hh))
print(hh if isinstance(hh, pd.Series) else hh.head())
print('len', len(hh), 'unique', pd.unique(pd.Series(list(hh))).size if not hasattr(hh,'columns') else pd.unique(hh['household_key']).size)
base = agent_api.baseline_features()
print('base shape', base.shape, 'cols', list(base.columns)[:12])
print('base index name:', base.index.name, 'dup idx:', base.index.duplicated().sum() if base.index.name else 'n/a')


# ---- cell ----
import agent_api
v = agent_api.snapshot(95)
attrs = [a for a in dir(v) if not a.startswith('_')]
print(attrs)
print('day:', v.day, 'week:', v.week)
print(agent_api.KEYS, agent_api.TARGET)


# ---- cell ----
import pandas as pd
import agent_api

def dbg(view, snapshot_day):
    hh = view.households
    print('day', snapshot_day, 'type', type(hh))
    if hh is not None:
        print('len', len(hh))
    # return minimal frame indexed by household_key
    if isinstance(hh, pd.DataFrame):
        idx = pd.Index(hh['household_key'].values, name='household_key')
    else:
        idx = pd.Index(list(hh), name='household_key')
    return pd.DataFrame({'x': 1.0}, index=idx)

out = agent_api.build_features(dbg)
print('OK', out.shape)
print(out.head())


# ---- cell ----
import numpy as np, pandas as pd
import agent_api
from agent_api import build_features, baseline_features, save_table

def make_features(view, snapshot_day):
    hh = view.households
    idx = pd.Index(hh if not isinstance(hh, pd.DataFrame) else hh['household_key'].values,
                   name='household_key')
    tx = view.transactions.merge(view.products[['product_id','department']], on='product_id', how='left')
    tx['disc'] = tx[['coupon_disc','coupon_match_disc','retail_disc']].fillna(0).sum(axis=1)
    feats = pd.DataFrame(index=idx)

    for W in (28, 56, 84, 112):
        w = tx[tx['day'] > snapshot_day - W]
        g = w.groupby('household_key')
        feats[f'spend_{W}'] = g['sales_value'].sum().reindex(idx).fillna(0.0)
        feats[f'trips_{W}'] = g['basket_id'].nunique().reindex(idx).fillna(0.0)
        feats[f'lines_{W}'] = g.size().reindex(idx).fillna(0.0)
        feats[f'qty_{W}']   = g['quantity'].sum().reindex(idx).fillna(0.0)
        feats[f'disc_{W}']  = g['disc'].sum().reindex(idx).fillna(0.0)
    feats['days_since_last'] = (snapshot_day - tx.groupby('household_key')['day'].max()).reindex(idx).fillna(180.0)
    feats['spend_ratio_28_56'] = feats['spend_28'] / feats['spend_56'].replace(0.0, np.nan)
    feats['avg_basket_28'] = feats['spend_28'] / feats['trips_28'].replace(0.0, np.nan)

    # NEW: department mix of recent spend
    d28 = (tx[tx['day'] > snapshot_day - 28]
           .groupby(['household_key','department'])['sales_value'].sum()
           .unstack(fill_value=0.0).reindex(idx).fillna(0.0))
    d28.columns = ['d28_' + str(c) for c in d28.columns]
    d112 = (tx[tx['day'] > snapshot_day - 112]
            .groupby(['household_key','department'])['sales_value'].sum()
            .unstack(fill_value=0.0).reindex(idx).fillna(0.0))
    sh112 = d112.div(d112.sum(axis=1).replace(0.0, np.nan), axis=0)
    sh112.columns = ['sh112_' + str(c) for c in sh112.columns]
    feats = pd.concat([feats, d28, sh112], axis=1)

    base = baseline_features()
    base = base[base['snapshot_day'] == snapshot_day].set_index('household_key')
    base = base.drop(columns=['snapshot_day'], errors='ignore')
    feats = feats.join(base, how='left')
    return feats

out = build_features(make_features)
print('shape:', out.shape, 'feat cols:', out.shape[1]-2)
path = save_table(out, 'dept_mix_recency.parquet')
print('saved:', path)
