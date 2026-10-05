
import pandas as pd, numpy as np, time
import agent_api as A

def feat(view, sd):
    hh = view.households
    if callable(hh): hh = hh()
    hh = pd.Index(np.asarray(hh).ravel(), name='household_key')
    tx = view.table('transactions')
    tx = tx[tx.household_key.isin(set(hh))]
    out = pd.DataFrame(index=hh)
    base = tx.groupby('household_key').day
    out['tenure'] = (sd - base.min()).reindex(hh)
    out['days_since_last'] = (sd - base.max()).reindex(hh)

    bs = tx.groupby(['household_key','basket_id']).agg(sp=('sales_value','sum'), d0=('day','min')).reset_index()

    for w in [28,56,84,168,364,728]:
        t = tx[tx.day > sd-w]
        g = t.groupby('household_key')
        a = g.agg(sp=('sales_value','sum'), trips=('basket_id','nunique'),
                  prods=('product_id','nunique'), stores=('store_id','nunique'), qty=('quantity','sum'))
        for c in a.columns: out[f'{c}{w}'] = a[c].reindex(hh).fillna(0)
        bw = bs[bs.d0 > sd-w]
        gb = bw.groupby('household_key').sp
        out[f'avgbs{w}'] = gb.mean().reindex(hh)
        out[f'maxbs{w}'] = gb.max().reindex(hh)
        out[f'nact{w}'] = bw.groupby('household_key').d0.nunique().reindex(hh).fillna(0)

    for w in [84,364]:
        t = tx[tx.day > sd-w]
        g = t.groupby('household_key')
        out[f'cdisc{w}'] = g.coupon_disc.sum().reindex(hh).fillna(0)
        out[f'rdisc{w}'] = g.retail_disc.sum().reindex(hh).fillna(0)

    out['trend_28_56'] = (out.sp28+1)/(out.sp56+1)
    out['trend_84'] = (out.sp28+1)/(out.sp84/3+1)
    out['sp28_rate'] = out.sp28/28
    out['sp84_rate'] = out.sp84/84
    out['sp364_rate'] = out.sp364/364
    t = tx[(tx.day > sd-392) & (tx.day <= sd-364)]
    out['sp_lag1y'] = t.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)

    d = view.demographics.set_index('household_key').reindex(hh)
    for c in d.columns:
        out['d_'+c] = d[c]
    out['has_demo'] = d['classification_1'].notna().astype(int)
    out['snapshot_day'] = sd
    out['week_mod52'] = ((sd+8)//7) % 52
    if sd == 95:
        print('shape', out.shape, 'nan frac', out.isna().mean().mean())
    return out

t0=time.time()
df = A.build_features(feat)
print('build time', time.time()-t0, df.shape)
print(df.head(3).T.head(30))
p = A.save_table(df, 'e001_history')
print(p)
