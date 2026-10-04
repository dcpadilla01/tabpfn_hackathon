import agent_api as A
import pandas as pd, numpy as np

def make_micro(view, sd):
    hh = view.households
    if hasattr(hh, 'columns'):
        keys = pd.Index(hh['household_key'].unique()) if 'household_key' in hh.columns else pd.Index(hh.index)
    else:
        keys = pd.Index(hh)
    idx = keys
    tx = view.transactions
    tx = tx[tx.day > sd - 380]
    out = pd.DataFrame(index=idx)

    def agg(mask, col, how):
        t = tx[mask]
        s = t.groupby('household_key')[col].sum() if how=='sum' else t.groupby('household_key')[col].nunique()
        return s.reindex(idx).fillna(0.0)

    spends, trips = {}, {}
    for k in range(1,13):
        hi = sd - 28*(k-1); lo = sd - 28*k
        m = (tx.day>lo)&(tx.day<=hi)
        spends[k] = agg(m,'sales_value','sum')
        if k<=6: trips[k] = agg(m,'basket_id','nunique')
    for k in range(1,7):
        out[f'mspend_l{k}'] = spends[k]; out[f'mtrips_l{k}'] = trips[k]
    out['spend_7d']  = agg((tx.day>sd-7)&(tx.day<=sd),'sales_value','sum')
    out['spend_14d'] = agg((tx.day>sd-14)&(tx.day<=sd),'sales_value','sum')
    out['trips_7d']  = agg((tx.day>sd-7)&(tx.day<=sd),'basket_id','nunique')
    out['trips_14d'] = agg((tx.day>sd-14)&(tx.day<=sd),'basket_id','nunique')
    out['n_zero_w6']  = sum((spends[k]==0).astype(float) for k in range(1,7))
    out['n_zero_w12'] = sum((spends[k]==0).astype(float) for k in range(1,13))
    S = pd.concat([spends[k] for k in range(1,7)], axis=1)
    mu = S.mean(1); med = S.median(1)
    out['spend_cv_l6'] = (S.std(1)/(mu+1e-6)).where(mu>0, 0.0)
    out['spend_max_over_med_l6'] = (S.max(1)/(med+1e-6)).where(med>0, 3.0)
    # inter-trip gaps, last 168d
    t2 = tx[['household_key','day']].drop_duplicates().sort_values(['household_key','day'])
    t2['gap'] = t2.groupby('household_key').day.diff()
    rec = t2[t2.day > sd-168]
    gs = rec.groupby('household_key').gap.agg(['mean','std','max'])
    out['gap_mean_l6'] = gs['mean'].reindex(idx)
    out['gap_std_l6']  = gs['std'].reindex(idx)
    out['gap_max_l6']  = gs['max'].reindex(idx)
    out['n_gap21_l6']  = (rec.gap>=21).groupby(rec.household_key).sum().reindex(idx).fillna(0.0)
    out['dsl'] = sd - tx.groupby('household_key').day.max().reindex(idx)
    # 84d basket micro-structure
    t84 = tx[tx.day > sd-84]
    b84 = t84.groupby(['household_key','basket_id']).sales_value.sum().rename('v').reset_index()
    bday = t84.groupby('basket_id').day.max()
    bmed = b84.groupby('household_key').v.median(); bmean = b84.groupby('household_key').v.mean()
    bmax = b84.groupby('household_key').v.max(); bstd = b84.groupby('household_key').v.std()
    out['basket_med_84'] = bmed.reindex(idx).fillna(0.0)
    out['basket_max_84'] = bmax.reindex(idx).fillna(0.0)
    out['basket_cv_84'] = (bstd/(bmean+1e-6)).reindex(idx).fillna(0.0)
    out['basket_max_over_med_84'] = (bmax/(bmed+1e-6)).reindex(idx)
    bb = b84.copy(); bb['day'] = bb.basket_id.map(bday); bb['med'] = bb.household_key.map(bmed)
    big = bb[bb.v > 2*bb.med]
    out['n_stockup_84'] = big.groupby('household_key').size().reindex(idx).fillna(0.0)
    out['dsl_stockup'] = (sd - big.groupby('household_key').day.max().reindex(idx)).fillna(999.0)
    sp84 = t84.groupby('household_key').sales_value.sum().reindex(idx).fillna(0.0)
    out['stockup_share_84'] = big.groupby('household_key').v.sum().reindex(idx).fillna(0.0)/(sp84+1e-6)
    trips84 = t84.groupby('household_key').basket_id.nunique().reindex(idx).fillna(0.0)
    units84 = t84.groupby('household_key').quantity.sum().reindex(idx).fillna(0.0)
    out['units_84'] = units84
    out['unit_price_84'] = (sp84/(units84.abs()+1e-6)).clip(-50,200)
    out['units_per_trip_84'] = units84.abs()/(trips84+1e-6)
    t84b = t84[['household_key','basket_id','day']].drop_duplicates()
    t84b = t84b.assign(we=(t84b.day%7).isin([5,6]))
    out['weekend_share_84'] = t84b.groupby('household_key').we.mean().reindex(idx)
    out['stores_84'] = t84.groupby('household_key').store_id.nunique().reindex(idx).fillna(0.0)
    out['morning_share_84'] = (t84.trans_time<1200).groupby(t84.household_key).mean().reindex(idx)
    return out.astype(float)

micro = A.build_features(make_micro)
print(micro.shape)
print(micro.head(3).T)
A.save_table(micro, 'micro.parquet')
print('saved')
