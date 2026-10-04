
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

DEPTS = ['GROCERY','DRUG GM','PRODUCE','COSMETICS','NUTRITION','MEAT','MEAT-PCKGD','DELI','PASTRY','FLORAL','SEAFOOD-PCKGD','MISC. TRANS.','SPIRITS','SEAFOOD']
W = 84

def mix(view, sd):
    hh = view.households
    tx = view.transactions
    prod = view.products[['product_id','department','brand']]
    t = tx[tx.day > sd - W].merge(prod, on='product_id', how='left')
    out = pd.DataFrame(index=hh)
    if len(t)==0: return out
    g = t.groupby('household_key')
    sp = g.sales_value.sum().rename('sp'); den = sp.where(sp>0)
    dep = t.groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0)
    for d in DEPTS:
        out['p_'+d] = (dep[d] if d in dep.columns else pd.Series(0.0,index=dep.index))/den
    oth = dep[[c for c in dep.columns if c not in DEPTS]].sum(axis=1)
    out['p_other'] = oth/den
    br = t.groupby(['household_key','brand']).sales_value.sum().unstack(fill_value=0.0)
    out['p_private'] = (br['Private'] if 'Private' in br.columns else pd.Series(0.0,index=br.index))/den
    qty = g.quantity.sum().replace(0, np.nan)
    out['unit_price'] = sp/qty
    out['n_prod84'] = g.product_id.nunique()
    bk = t.drop_duplicates('basket_id').copy()
    bk['dow'] = bk.day % 7
    cnt = bk.groupby(['household_key','dow']).size().unstack(fill_value=0.0)
    tot = cnt.sum(axis=1).replace(0, np.nan)
    p = cnt.div(tot, axis=0).replace(0, np.nan)
    out['dow_entropy'] = (-(p*np.log(p)).sum(axis=1)/np.log(7)).fillna(0)
    out['modal_dow'] = (cnt.max(axis=1)/tot).fillna(0)
    w_sd = (sd+8)//7
    t2 = t.assign(w=(t.day+8)//7)
    ws = t2.groupby(['household_key','w']).sales_value.sum().unstack(fill_value=0.0)
    cols = [c for c in ws.columns if w_sd-11 <= c <= w_sd]
    if cols:
        m = ws[cols].values
        out['zero_w12'] = pd.Series((m==0).sum(axis=1), index=ws.index)
        out['wk_cv'] = pd.Series(m.std(axis=1)/(m.mean(axis=1)+1e-9), index=ws.index)
    return out

def fn(view, sd):
    return mix(view, sd)

df = agent_api.build_features(fn)
p = agent_api.save_table(df, 'mix_v1.parquet')
print(p, df.shape)
print(df.isna().mean().round(3).to_string())
