import agent_api as api
import pandas as pd, numpy as np

def feats(view, s):
    hh = pd.Index(view.households, name='household_key')
    tx = view.transactions
    out = {}
    # population seasonal uplift: avg weekly total spend at same week_no one year earlier
    if len(tx)>0:
        wk = ((tx.day+8)//7).astype(int)
        wtot = tx.assign(w=wk).groupby('w').sales_value.sum()
        wmax = (s+8)//7
        nxt = sorted(set(range(wmax+1, wmax+5)))
        vals=[]
        for w in nxt:
            cands = [w-52, w-104]
            cands = [c for c in cands if c>=1 and c in wtot.index]
            vals.append(wtot[cands].mean() if cands else np.nan)
        overall = wtot[wmax-4:wmax+1].mean() if wmax>=5 else wtot.mean()
        out['nseas_uplift'] = np.nansum(vals)/ (overall if overall>0 else np.nan)
        out['nseas_uplift_mean'] = np.nanmean(vals)/(overall if overall>0 else np.nan)
    else:
        out['nseas_uplift']=np.nan; out['nseas_uplift_mean']=np.nan
    # per-household same-weeks-last-year spend (l13) and ratio to recent
    if len(tx)>0:
        m = tx[(tx.day>s-364)&(tx.day<=s-336)]
        out['nspend_ly'] = m.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
        m28 = tx[(tx.day>s-28)&(tx.day<=s)]
        sp28 = m28.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
        out['nratio_ly'] = sp28/(out['nspend_ly']/13.0 + 5)
    else:
        out['nspend_ly']=0.0; out['nratio_ly']=np.nan
    df = pd.DataFrame(out).reindex(hh)
    return df

tab = api.build_features(feats)
print(tab.shape, tab.columns.tolist())
print(tab[['nseas_uplift','nseas_uplift_mean']].describe().round(3))
print(tab.groupby('snapshot_day')[['nseas_uplift','nseas_uplift_mean']].mean().round(3))
path = api.save_table(tab, 'nf_seasonal.parquet')
print(path)