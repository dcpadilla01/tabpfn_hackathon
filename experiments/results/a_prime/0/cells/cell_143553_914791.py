
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
    oth = dep[[c for c in dep.columns if c not in DEPTS]].sum(axis=1) if len([c for c in dep.columns if c not in DEPTS]) else pd.Series(0.0,index=dep.index)
    out['p_other'] = oth/den
    br = t.groupby(['household_key','brand']).sales_value.sum().unstack(fill_value=0.0)
    out['p_private'] = (br['Private'] if 'Private' in br.columns else pd.Series(0.0,index=br.index))/den
    qty = g.quantity.sum().replace(0, np.nan)
    out['unit_price'] = sp/qty
    out['n_prod84'] = g.product_id.nunique()
    tt = (t.trans_time//100)*60 + t.trans_time%100
    out['tt_mean'] = tt.groupby(t.household_key).mean()
    bk = t.drop_duplicates('basket_id').copy()
    bk['minute'] = (bk.trans_time//100)*60 + bk.trans_time%100
    out['frac_evening'] = bk.assign(e=bk.minute>=1020).groupby('household_key').e.mean()
    w_sd = (sd+8)//7
    t2 = t.assign(w=(t.day+8)//7)
    ws = t2.groupby(['household_key','w']).sales_value.sum().unstack(fill_value=0.0)
    cols = [c for c in ws.columns if w_sd-11 <= c <= w_sd]
    if cols:
        m = ws[cols].values
        out['wk_std'] = pd.Series(m.std(axis=1), index=ws.index)
        out['wk_cv'] = pd.Series(m.std(axis=1)/(m.mean(axis=1)+1e-9), index=ws.index)
        out['zero_w12'] = pd.Series((m==0).sum(axis=1), index=ws.index)
    days = t.groupby('household_key').day.apply(lambda s: np.sort(s.unique()))
    out['max_gap'] = days.apply(lambda a: (np.diff(a).max() if len(a)>1 else W))
    bk['dow'] = bk.day % 7
    cnt = bk.groupby(['household_key','dow']).size().unstack(fill_value=0.0)
    tot = cnt.sum(axis=1).replace(0, np.nan)
    p = cnt.div(tot, axis=0).replace(0, np.nan)
    out['dow_entropy'] = (-(p*np.log(p)).sum(axis=1)/np.log(7)).fillna(0)
    out['modal_dow'] = (cnt.max(axis=1)/tot).fillna(0)
    return out

tt = agent_api.train_targets()
h2 = agent_api.load_saved('hist_v2.parquet')[['household_key','snapshot_day','spend_84']]
feats=[]
for sd in agent_api.snapshot_days()['train']:
    f = mix(agent_api.snapshot(sd), sd)
    f['snapshot_day']=sd
    feats.append(f.reset_index())
F = pd.concat(feats)
F = tt.merge(F, on=['household_key','snapshot_day'], how='left').merge(h2, on=['household_key','snapshot_day'], how='left')
y = np.log1p(F.future_spend_4w)
base = np.log1p(F.spend_84.fillna(0))
res = y - base*0  # placeholder
# residualize y on log spend_84
A = np.column_stack([np.ones(len(F)), base])
b,_,_,_ = np.linalg.lstsq(A, y, rcond=None)
res = y - A@b
num = F.select_dtypes(include=[np.number]).drop(columns=['snapshot_day','future_spend_4w'])
cor_raw, cor_res = {}, {}
for c in num.columns:
    x = num[c]
    ok = x.notna() & y.notna()
    if ok.sum() < 500: continue
    cor_raw[c] = np.corrcoef(x[ok], y[ok])[0,1]
    ok2 = ok & res.notna() & base.notna()
    if ok2.sum() > 500:
        xb = np.column_stack([np.ones(ok2.sum()), base[ok2]])
        bb,_,_,_ = np.linalg.lstsq(xb, x[ok2].fillna(x[ok2].median()), rcond=None)
        xr = x[ok2] - xb@bb
        cor_res[c] = np.corrcoef(xr, res[ok2])[0,1]
R = pd.DataFrame({'raw':pd.Series(cor_raw),'resid_on_spend84':pd.Series(cor_res)})
R['max_abs'] = R.abs().max(axis=1)
print(R.reindex(R.max_abs.sort_values(ascending=False).index).round(3).to_string())
