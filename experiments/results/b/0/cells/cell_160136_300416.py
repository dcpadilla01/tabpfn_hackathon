import agent_api as A, pandas as pd, numpy as np

def macro_fn(view, day):
    t = view.transactions
    d = t.groupby('day')['sales_value'].sum()
    hh = t.groupby('day')['household_key'].nunique()
    first = t.groupby('household_key')['day'].min()
    idx = first.index[first <= day-84]
    def wsum(a,b):
        i = d.index[(d.index>=a)&(d.index<=b)]
        return float(d.reindex(i).sum()) if len(i) else np.nan
    def whh(a,b):
        i = hh.index[(hh.index>=a)&(hh.index<=b)]
        return float(hh.reindex(i).sum()) if len(i) else np.nan
    r = {}
    r['macro_spend28'] = wsum(day-27, day)
    r['macro_spend_p28'] = wsum(day-55, day-28)
    r['macro_spend28_ly'] = wsum(day-391, day-364) if day >= 392 else np.nan
    r['macro_spend112'] = wsum(day-111, day)
    r['macro_spend112_ly'] = wsum(day-475, day-364) if day >= 476 else np.nan
    r['macro_hh28'] = whh(day-27, day)
    r['macro_spend_per_hh28'] = r['macro_spend28']/r['macro_hh28'] if r['macro_hh28'] else np.nan
    ly_hh = whh(day-391, day-364) if day>=392 else np.nan
    r['macro_spend_per_hh28_ly'] = r['macro_spend28_ly']/ly_hh if ly_hh else np.nan
    r['macro_ratio_ly'] = r['macro_spend28']/r['macro_spend28_ly'] if day>=392 else np.nan
    r['macro_growth'] = r['macro_spend28']/r['macro_spend_p28'] if r['macro_spend_p28'] else np.nan
    r['macro_ratio112_ly'] = r['macro_spend112']/r['macro_spend112_ly'] if day>=476 else np.nan
    wk = t.groupby('week_no')['sales_value'].sum()
    wcur = (day+8)//7
    l12 = wk.reindex(range(wcur-11, wcur+1)).values.astype(float)
    r['macro_wk_ratio_last3'] = np.nanmean(l12[-3:])/np.nanmean(l12[:-3]) if np.nanmean(l12[:-3]) else np.nan
    return pd.DataFrame({k: r[k] for k in r}, index=idx).astype(float)

mac = A.build_features(macro_fn)
print('macro table:', mac.shape, mac.columns.tolist())
ps = mac.drop_duplicates('snapshot_day').set_index('snapshot_day')
print(ps[['macro_spend28','macro_spend_per_hh28','macro_ratio_ly','macro_growth','macro_wk_ratio_last3']].round(3).to_string())
A.save_table(mac, 'macro.parquet')
print('saved macro.parquet; e014_dm.parquet already saved earlier')
