import pandas as pd, numpy as np, agent_api

def fn(view, s):
    hh = np.asarray(view.households)
    tx = view.table('transactions')
    tx = tx[tx['household_key'].isin(hh)]
    out = pd.DataFrame(index=pd.Index(hh, name='household_key'))
    g = tx.groupby('household_key')
    last = g['day'].max().reindex(hh)
    out['dsl'] = (s - last).astype(float).values
    for name,(lo,hi) in {'w2':(s-13,s-7),'w3':(s-20,s-14),'w4':(s-27,s-21)}.items():
        v = tx[(tx.day>=lo)&(tx.day<=hi)].groupby('household_key')['sales_value'].sum()
        out[name] = v.reindex(hh).fillna(0.0).values
    sp = {}
    for k in range(3):
        lo, hi = s-27-28*k, s-28*k
        v = tx[(tx.day>=lo)&(tx.day<=hi)].groupby('household_key')['sales_value'].sum()
        sp[k] = v.reindex(hh).fillna(0.0).values
    out['zeros_3win'] = ((sp[0]==0).astype(float)+(sp[1]==0).astype(float)+(sp[2]==0).astype(float)).values
    s28 = sp[0]
    trips = tx[['household_key','day']].drop_duplicates().sort_values(['household_key','day'])
    trips['gap'] = trips.groupby('household_key')['day'].diff()
    def gapstats(daymin):
        t = trips[trips.day>=daymin]
        return (t.groupby('household_key')['gap'].median(),
                t.groupby('household_key')['gap'].mean(),
                t.groupby('household_key')['gap'].count())
    med364, mean364, n364 = gapstats(s-363)
    medAll, meanAll, nAll = gapstats(-1)
    med364 = med364.where(n364>=2, medAll); mean364 = mean364.where(n364>=2, meanAll)
    out['med_gap'] = med364.reindex(hh).values
    out['mean_gap'] = mean364.reindex(hh).values
    dsl = out['dsl'].values; mg = out['med_gap'].values; mgA = out['mean_gap'].values
    out['dsl_over_medgap'] = np.where(np.isfinite(mg)&(mg>0), dsl/np.maximum(mg,1e-9), np.nan)
    out['dsl_over_meangap'] = np.where(np.isfinite(mgA)&(mgA>0), dsl/np.maximum(mgA,1e-9), np.nan)
    out['dsl_gt_medgap'] = (dsl > mg)
    w = tx[tx.day>=s-363]
    u = w.groupby('household_key')['quantity'].sum().reindex(hh).fillna(0.0).values
    sv = w.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0).values
    out['units_364'] = u
    out['avg_unit_price_364'] = np.where(u>0, sv/np.maximum(u,1e-9), np.nan)
    out['s28_x_active21'] = s28 * (dsl<=21)
    ang = 2*np.pi*(s+14)/364.0
    out['ann_sin_fw'] = np.sin(ang); out['ann_cos_fw'] = np.cos(ang)
    return out

feat = agent_api.build_features(fn)
print('feat rows', feat.shape)
base = agent_api.load_saved('e015_best_pseudo.parquet').drop(columns=['s28_x_churnrisk'])
merged = base.merge(feat, on=['household_key','snapshot_day'], how='inner')
print('merged', merged.shape, 'nan cols:', merged.isna().all().sum())
tt = agent_api.train_targets()
m = merged.merge(tt, on=['household_key','snapshot_day'])
newc = ['dsl','w2','w3','w4','zeros_3win','med_gap','mean_gap','dsl_over_medgap','dsl_over_meangap','dsl_gt_medgap','units_364','avg_unit_price_364','s28_x_active21','ann_sin_fw','ann_cos_fw']
print(m[newc].corrwith(m['future_spend_4w']).round(3).to_string())
path = agent_api.save_table(merged, 'e016_churn_gapratio.parquet')
print(path)
