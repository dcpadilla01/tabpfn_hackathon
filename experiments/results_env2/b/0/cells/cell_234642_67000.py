import agent_api, pandas as pd, numpy as np

def fn(view, s):
    tx = view.table('transactions')
    hh = pd.Index(view.households, name='household_key')
    first = tx.groupby('household_key')['day'].min().reindex(hh)
    tenure = (s - first).astype(float)
    out = pd.DataFrame(index=hh)
    out['tenure_cap364'] = tenure.clip(upper=364)
    eff = tenure.clip(lower=1.0)
    for W in (112, 182, 364):
        w = tx[tx.day > s - W]
        sp = w.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
        out[f'rate_{W}'] = sp / eff.clip(upper=W) * 7.0   # weekly-equivalent spend rate
    tr = tx[tx.day > s - 364].groupby('household_key')['basket_id'].nunique().reindex(hh).fillna(0.0)
    out['trips_rate_364'] = tr / (eff.clip(upper=364) / 7.0)
    age = (s - tx['day']).astype(float)
    dec = (tx['sales_value'] * (0.5 ** (age / 112.0))).groupby(tx['household_key']).sum().reindex(hh).fillna(0.0)
    normw = (112.0 / np.log(2.0)) * (1.0 - 0.5 ** (tenure / 112.0))
    out['dec112_rate'] = dec / normw.clip(lower=1e-6) * 7.0
    return out

newf = agent_api.build_features(fn).reset_index()
print("new feats:", newf.shape, newf.columns.tolist())

t = agent_api.load_saved('e013_stationary.parquet')
drops = ['week_sin','week_cos','trend','redemp_84','homeowner_code','age_code','class3_code','has_demo',
         'z_dec_224','spend_364','trips_364','dec_112','z_zero_block_share','r_84_364']
t2 = t.drop(columns=drops).merge(newf, on=['household_key','snapshot_day'], how='inner')
print("merged:", t2.shape, "dups:", t2[['household_key','snapshot_day']].duplicated().sum())

tt = agent_api.train_targets()
tr = t2.merge(tt, on=['household_key','snapshot_day']).query('snapshot_day <= 431')
for c in ['rate_364','rate_182','rate_112','dec112_rate','trips_rate_364','tenure_cap364']:
    ok = tr[c].notna()
    print(c, "corr:", round(np.corrcoef(tr.loc[ok,c], tr.loc[ok,'future_spend_4w'])[0,1],3))

g = t2.groupby('snapshot_day')[['rate_364','dec112_rate','tenure_cap364']].mean().round(3)
print(g.T)

path = agent_api.save_table(t2, 'e015_norm_windows')
print(path)
