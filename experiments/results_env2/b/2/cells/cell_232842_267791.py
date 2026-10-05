
import agent_api, pandas as pd, numpy as np

def fn(view, snapshot_day):
    day = snapshot_day
    tx = view.table('transactions')
    hh_raw = view.households
    if isinstance(hh_raw, pd.DataFrame):
        hh = pd.Index(hh_raw.index, name='household_key')
    else:
        hh = pd.Index(hh_raw, name='household_key')
    out = pd.DataFrame(index=hh)
    # trailing window spend/trips
    for w in [28,56,84,112,168,364]:
        sub = tx[tx.day > day - w]
        out[f'spend_{w}d'] = sub.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
        out[f'trips_{w}d'] = sub.groupby('household_key').size().reindex(hh).fillna(0.0)
    out['avg_basket_112'] = out['spend_112d'] / out['trips_112d'].replace(0, np.nan)
    # EWMA daily spend
    daily = tx.groupby(['household_key','day'], as_index=False)['sales_value'].sum()
    for hl, name in [(28,'ew28'), (112,'ew112')]:
        wgt = 0.5 ** ((day - daily['day']) / hl)
        num = (daily['sales_value'] * wgt).groupby(daily['household_key']).sum()
        den = wgt.groupby(daily['household_key']).sum()
        out[name] = (num/den).reindex(hh)
    # self-calibration: spend in aligned 28d windows k=1..12 back
    first = tx.groupby('household_key')['day'].min().reindex(hh)
    fk = []
    for k in range(1,13):
        end = day - 28*(k-1); start = day - 28*k + 1
        s = tx[(tx.day>=start)&(tx.day<=end)].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
        s[end < first] = np.nan
        fk.append(s.rename(k))
    fkdf = pd.concat(fk, axis=1)
    out['sc_f_mean'] = fkdf.mean(axis=1)
    out['ya_spend'] = tx[(tx.day>=day-391)&(tx.day<=day-364)].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
    out['wk_spend_mean'] = out['spend_364d']/52.0
    # log transforms
    for c in ['spend_28d','spend_56d','spend_84d','spend_168d','spend_364d','trips_28d',
              'avg_basket_112','ew28','ew112','sc_f_mean','ya_spend','wk_spend_mean']:
        out['lg_'+c] = np.log1p(out[c].clip(lower=0))
    # spend x demographic-segment interactions
    demo = view.table('demographics').set_index('household_key')
    s28 = out['spend_28d']
    for c in ['classification_1','classification_4','homeowner_desc','kid_category_desc']:
        dums = pd.get_dummies(demo[c], prefix=c).astype(float).reindex(hh)
        for col in dums.columns:
            out['ix_'+col] = s28 * dums[col].fillna(0.0).values
    return out

newb = agent_api.build_features(fn)
print("new block:", newb.shape)
print([c for c in newb.columns if not c in ('household_key','snapshot_day')][:8], "... n_feat:", newb.shape[1]-2)
print(newb[[c for c in newb.columns if c.startswith('lg_')][:3]].describe().loc[['mean','std']])
print("NaN frac total:", newb.isna().mean().mean())

base = agent_api.load_saved('e011_table.parquet')
m = base.merge(newb, on=['household_key','snapshot_day'], how='inner')
print("merged:", m.shape)
assert m.shape[0] == base.shape[0] and m.isna().all(axis=1).sum() == 0
p = agent_api.save_table(m, 'e013_table.parquet')
print("saved:", p)
