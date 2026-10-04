import numpy as np
import pandas as pd
import agent_api

EPS = 1e-6
WINDOWS = (7, 28, 56, 112, 364)

def make_features(view, D):
    hh = view.households
    tx = view.transactions
    idx = pd.Index(hh)
    feats = pd.DataFrame(index=idx)

    for w in WINDOWS:
        sub = tx[tx.day > D - w]
        g = sub.groupby('household_key')
        feats[f'spend{w}'] = g['sales_value'].sum().reindex(idx).fillna(0.0)
        feats[f'trips{w}'] = g['basket_id'].nunique().reindex(idx).fillna(0.0)
        feats[f'actdays{w}'] = g['day'].nunique().reindex(idx).fillna(0.0)
        feats[f'qty{w}'] = g['quantity'].sum().reindex(idx).fillna(0.0)
        feats[f'nprod{w}'] = g['product_id'].nunique().reindex(idx).fillna(0.0)
        feats[f'nstore{w}'] = g['store_id'].nunique().reindex(idx).fillna(0.0)

    g = tx.groupby('household_key')
    feats['recency'] = (D - g['day'].max()).reindex(idx).astype(float)
    feats['tenure'] = (D - g['day'].min()).reindex(idx).astype(float)
    feats['lt_spend'] = g['sales_value'].sum().reindex(idx).fillna(0.0)
    feats['lt_trips'] = g['basket_id'].nunique().reindex(idx).fillna(0.0)

    feats['trend_7_28'] = feats['spend7'] / (feats['spend28'] + EPS)
    feats['trend_28_56'] = feats['spend28'] / (feats['spend56'] + EPS)
    feats['trend_56_112'] = feats['spend56'] / (feats['spend112'] + EPS)
    feats['trend_112_364'] = feats['spend112'] / (feats['spend364'] + EPS)
    feats['avg_basket28'] = feats['spend28'] / (feats['trips28'] + EPS)
    feats['spend_per_day28'] = feats['spend28'] / 28.0
    feats['trips_per_day28'] = feats['trips28'] / 28.0

    yoy = tx[(tx.day > D - 364) & (tx.day <= D - 336)]
    feats['spend_yoy28'] = yoy.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0)

    sub = tx[tx.day > D - 112]
    g2 = sub.groupby('household_key')
    for col in ('coupon_disc', 'retail_disc', 'coupon_match_disc'):
        feats[col + '112'] = g2[col].sum().reindex(idx).fillna(0.0)
    disc = feats['coupon_disc112'].abs() + feats['retail_disc112'].abs() + feats['coupon_match_disc112'].abs()
    feats['disc_share112'] = disc / (feats['spend112'] + EPS)

    feats['mean_hour112'] = (sub['trans_time'] // 100.0).groupby(sub['household_key']).mean().reindex(idx)
    feats['mean_dow112'] = (sub['day'] % 7).groupby(sub['household_key']).mean().reindex(idx)

    prods = view.products
    m = sub.merge(prods[['product_id', 'department']].drop_duplicates('product_id'), on='product_id', how='left')
    feats['ndept112'] = m.groupby('household_key')['department'].nunique().reindex(idx).fillna(0.0)

    tgt = view.campaign_targets
    feats['n_camp_targeted'] = tgt.groupby('household_key').size().reindex(idx).fillna(0.0)
    red = view.coupon_redemptions
    feats['redemptions112'] = red[red.day > D - 112].groupby('household_key').size().reindex(idx).fillna(0.0)
    feats['redemptions_lt'] = red.groupby('household_key').size().reindex(idx).fillna(0.0)

    feats['snapshot_day'] = float(D)
    feats['week'] = float(view.week)
    return feats

table = agent_api.build_features(make_features)
print('shape', table.shape, 'n_features', table.shape[1] - 2)
path = agent_api.save_table(table, 'rich_behavioral')
print('SAVED', path)

tt = agent_api.train_targets()
tr = table.merge(tt, on=['household_key', 'snapshot_day'], how='inner')
print('merged train rows:', len(tr))
num = tr.select_dtypes(include=[np.number]).drop(columns=['snapshot_day'])
corr = num.corrwith(tr['future_spend_4w'])
print(corr.reindex(corr.abs().sort_values(ascending=False).index).head(25))
