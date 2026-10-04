import agent_api, pandas as pd, numpy as np

BASE = agent_api.load_saved('e009_ewma_longlags.parquet')

def fn(view, s):
    tx = view.transactions
    p = view.products
    t = tx[(tx['day'] > s-182) & (tx['day'] <= s)][['household_key','day','product_id','sales_value']].copy()
    hh_all = BASE.loc[BASE['snapshot_day']==s, 'household_key']
    if len(t) == 0:
        return pd.DataFrame(index=hh_all)
    t = t.merge(p[['product_id','commodity_desc']], on='product_id', how='left')
    t['commodity_desc'] = t['commodity_desc'].astype('object').fillna('UNK')
    # recency weights for EWMA variant
    t['w'] = 0.5 ** ((s - t['day'])/56.0)
    t['sw'] = t['sales_value']*t['w']

    daily = t.groupby(['household_key','commodity_desc','day'], as_index=False)['sales_value'].sum()
    daily = daily.sort_values(['household_key','commodity_desc','day'])
    g = daily.groupby(['household_key','commodity_desc'])
    occ = g.agg(first_d=('day','min'), last_d=('day','max'), n=('day','size'), tot=('sales_value','sum')).reset_index()
    occ = occ[occ['n']>=2].copy()
    occ['gap'] = (occ['last_d']-occ['first_d'])/(occ['n']-1)
    occ['mean_spend'] = occ['tot']/occ['n']

    # EWMA-weighted spend per commodity
    ew = t.groupby(['household_key','commodity_desc']).agg(sw=('sw','sum'), w=('w','sum')).reset_index()
    ew['ew_spend'] = ew['sw']/ew['w'].replace(0, np.nan)
    ew = ew[['household_key','commodity_desc','ew_spend']]

    res = None
    for clip, tag in [(30,'c30'),(60,'c60'),(90,'c90')]:
        o = occ.copy()
        o['gap_c'] = o['gap'].clip(1, clip)
        o['next'] = o['last_d'] + o['gap_c']
        o['due'] = (o['next'] <= s+28).astype(float)
        o['contrib'] = o['due']*o['mean_spend']
        o['pressure'] = np.minimum(1.0, (s - o['last_d'])/o['gap_c'])
        agg = o.groupby('household_key').agg(**{f'cyc_pred_{tag}':('contrib','sum'), f'cyc_due_{tag}':('due','sum'), f'cyc_press_{tag}':('pressure','sum')}).reset_index()
        if tag == 'c60':
            act = o.groupby('household_key')['commodity_desc'].size().rename('cyc_active').reset_index()
            agg = agg.merge(act, on='household_key', how='outer')
            # EWMA-spend projection using c60 due flags
            o2 = o.merge(ew, on=['household_key','commodity_desc'], how='left')
            o2['contrib_ew'] = o2['due']*o2['ew_spend'].fillna(o2['mean_spend'])
            ewp = o2.groupby('household_key')['contrib_ew'].sum().rename('cyc_pred_ew').reset_index()
            agg = agg.merge(ewp, on='household_key', how='outer')
        res = agg if res is None else res.merge(agg, on='household_key', how='outer')
    res = res.set_index('household_key').reindex(hh_all)
    res['cyc_press_ratio'] = res['cyc_press_c60']/(res['cyc_active']+1.0)
    res['cyc_pred_per_active'] = res['cyc_pred_c60']/(res['cyc_active']+1.0)
    return res.fillna(0.0)

feats = agent_api.build_features(fn)
print("feats shape", feats.shape)
newcols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
print("new cols", newcols)

full = BASE.merge(feats, on=['household_key','snapshot_day'], how='left')
print("full shape", full.shape, "| NaN new feats:", int(full[newcols].isna().sum().sum()))
full[newcols] = full[newcols].fillna(0.0)
path = agent_api.save_table(full, 'e018_cycle_projection.parquet')
print("saved", path)

# sanity: correlations at s=431
tt = agent_api.train_targets()
m = full[full['snapshot_day']==431].merge(tt[tt['snapshot_day']==431], on=['household_key','snapshot_day'])
for c in newcols:
    print(c, round(m[c].corr(m['future_spend_4w']), 4))
