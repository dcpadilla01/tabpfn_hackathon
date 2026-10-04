
import agent_api, pandas as pd, numpy as np
print(agent_api.snapshot_days())
tt = agent_api.train_targets()
t = tt.future_spend_4w
print('train rows:', len(tt))
print('target mean %.2f std %.2f median %.2f zero-frac %.3f' % (t.mean(), t.std(), t.median(), (t==0).mean()))
K = agent_api.KEYS
h2 = agent_api.load_saved('hist_v2.parquet'); m2 = agent_api.load_saved('mkt_v2.parquet')
print('hist_v2', h2.shape, 'mkt_v2', m2.shape)
print('hist_v2 cols:', h2.columns.tolist())
print('mkt_v2 extra cols:', [c for c in m2.columns if c not in h2.columns])
df = tt.merge(h2, on=K, how='left')
extra = [c for c in m2.columns if c not in df.columns and c not in K]
if extra: df = df.merge(m2[extra+K], on=K, how='left')
num = df.select_dtypes(include=[np.number])
cor = num.corr()['future_spend_4w'].drop('future_spend_4w')
cor = cor.reindex(cor.abs().sort_values(ascending=False).index)
print(cor.head(45).round(3).to_string())


# ---- cell ----

import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tx = v.transactions
prod = v.products
print('brand values:', prod.brand.value_counts(dropna=False).head().to_dict())
# target seasonality
tt = agent_api.train_targets()
print(tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median','count']).round(1).to_string())
# per-household 4-week block autocorrelation (same-season lag 13 blocks)
tx = tx[['household_key','day','sales_value']]
def block_spend(lo, hi):
    m = tx[(tx.day>=lo)&(tx.day<=hi)]
    return m.groupby('household_key').sales_value.sum()
s = agent_api.snapshot_days()
blocks = {}
for sd in s['train']:
    blocks[sd] = block_spend(sd+1, sd+28)
B = pd.DataFrame(blocks)
# correlate block sd with block sd-364 and sd-28
for lag, pairs in [(28,[(sd, sd-28) for sd in s['train'] if sd-28 in B.columns]),
                   (364,[(sd, sd-364) for sd in s['train'] if sd-364 in B.columns])]:
    cs = []
    for a,b in pairs:
        x = B[a]; y = B[b]
        ok = x.notna()&y.notna()
        if ok.sum()>50: cs.append(np.corrcoef(np.log1p(x[ok]), np.log1p(y[ok]))[0,1])
    print('lag',lag,'mean log-corr %.3f (n pairs %d)'%(np.mean(cs), len(cs)))


# ---- cell ----

import agent_api, pandas as pd, numpy as np
tx = agent_api.snapshot(459).transactions[['household_key','day','sales_value']]
def bs(lo,hi):
    m = tx[(tx.day>=lo)&(tx.day<=hi)]
    return m.groupby('household_key').sales_value.sum()
s = agent_api.snapshot_days()['train']
# history blocks: last block before snapshot (sd-27..sd), same season one year back (sd-391..sd-364), one block back (sd-55..sd-28)
rows=[]
for sd in s:
    cur = bs(sd-27, sd); yr = bs(sd-391, sd-364); prev = bs(sd-55, sd-28)
    d = pd.DataFrame({'cur':cur,'yr':yr,'prev':prev}).dropna()
    r_yr = np.corrcoef(np.log1p(d.cur), np.log1p(d.yr))[0,1]
    r_prev = np.corrcoef(np.log1p(d.cur), np.log1p(d.prev))[0,1]
    rows.append((sd, r_yr, r_prev, len(d)))
print(pd.DataFrame(rows, columns=['sd','r_lag364','r_lag28','n']).round(3).to_string())
# how many households have data 364 days back at each snapshot
hh_first = tx.groupby('household_key').day.min()
for sd in [95,123,207,347,431,459,487,515,543]:
    print(sd, 'eligible:', (hh_first<=sd-84).sum(), 'with 364d hist:', (hh_first<=sd-364).sum())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tx = agent_api.snapshot(459).transactions[['household_key','day','sales_value']]
tt = agent_api.train_targets()
s = agent_api.snapshot_days()['train']
def bs(lo,hi):
    m = tx[(tx.day>=lo)&(tx.day<=hi)]
    return m.groupby('household_key').sales_value.sum()
# build small feature frame for train rows with lag364 available
rows=[]
for sd in s:
    cur = bs(sd-27, sd).rename('cur')      # recent 4w (proxy for x_spend_4)
    yr  = bs(sd-391, sd-364).rename('yr')  # same season last year
    d = pd.concat([cur, yr], axis=1)
    d['sd']=sd
    rows.append(d)
F = pd.concat(rows)
F = tt.merge(F.reset_index().rename(columns={'index':'household_key'}), on=['household_key','snapshot_day'], how='inner')
F = F.dropna(subset=['cur','yr'])
print('rows with lag364:', len(F))
X = np.column_stack([np.ones(len(F)), np.log1p(F.cur), np.log1p(F.yr)])
y = np.log1p(F.future_spend_4w)
b,res,rank,sv = np.linalg.lstsq(X, y, rcond=None)
yhat = X@b
print('coef [const, log cur, log yr]:', b.round(3))
print('R2 log-space with yr: %.3f  without: %.3f' % (
    1-((y-yhat)**2).mean()/((y-y.mean())**2).mean(),
    1-((y-np.linalg.lstsq(X[:,:2],y,rcond=None)[0]@X[:,:2])**2).mean()/((y-y.mean())**2).mean()))
# partial corr of yr given cur
rc = np.corrcoef(np.log1p(F.cur), y)[0,1]; ry=np.corrcoef(np.log1p(F.yr), y)[0,1]; rcc=np.corrcoef(np.log1p(F.cur), np.log1p(F.yr))[0,1]
print('partial corr yr|cur: %.3f' % ((ry-rc*rcc)/np.sqrt((1-rc**2)*(1-rcc**2))))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tx = agent_api.snapshot(459).transactions[['household_key','day','sales_value']]
tt = agent_api.train_targets()
s = agent_api.snapshot_days()['train']
def bs(lo,hi):
    m = tx[(tx.day>=lo)&(tx.day<=hi)]
    return m.groupby('household_key').sales_value.sum()
rows=[]
for sd in s:
    cur = bs(sd-27, sd).rename('cur')
    yr  = bs(sd-391, sd-364).rename('yr')
    d = pd.concat([cur, yr], axis=1)
    d['snapshot_day']=sd
    rows.append(d.reset_index())
F = pd.concat(rows)
F = tt.merge(F, on=['household_key','snapshot_day'], how='inner').dropna(subset=['cur','yr'])
print('rows with lag364:', len(F))
X = np.column_stack([np.ones(len(F)), np.log1p(F.cur), np.log1p(F.yr)])
y = np.log1p(F.future_spend_4w)
b,_,_,_ = np.linalg.lstsq(X, y, rcond=None)
yhat = X@b
b2,_,_,_ = np.linalg.lstsq(X[:,:2], y, rcond=None)
yhat2 = X[:,:2]@b2
print('coef [const, log cur, log yr]:', b.round(3))
print('R2 log with yr: %.3f  without: %.3f' % (1-((y-yhat)**2).mean()/((y-y.mean())**2).mean(), 1-((y-yhat2)**2).mean()/((y-y.mean())**2).mean()))
rc = np.corrcoef(np.log1p(F.cur), y)[0,1]; ry=np.corrcoef(np.log1p(F.yr), y)[0,1]; rcc=np.corrcoef(np.log1p(F.cur), np.log1p(F.yr))[0,1]
print('partial corr yr|cur: %.3f' % ((ry-rc*rcc)/np.sqrt((1-rc**2)*(1-rcc**2))))


# ---- cell ----

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


# ---- cell ----

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

feats=[]
for sd in agent_api.snapshot_days()['train'] + agent_api.snapshot_days()['validation']:
    f = mix(agent_api.snapshot(sd), sd)
    f['snapshot_day']=sd
    feats.append(f.reset_index())
F = pd.concat(feats)
F = F.rename(columns={'index':'household_key'})
p = agent_api.save_table(F, 'mix_v1.parquet')
print(p)
print(F.shape, F.columns.tolist()[:8])


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
def fn(view, sd):
    m = agent_api.load_saved('mkt_v2.parquet')
    sub = m[m.snapshot_day==sd].set_index('household_key')
    sub = sub.reindex(view.households)
    return sub[['m_spend28']]
df = agent_api.build_features(fn)
print(df.shape, df.head(3))
print('missing:', df.m_spend28.isna().mean())


# ---- cell ----

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
    oth = oth if hasattr(oth,'index') else pd.Series(0.0,index=dep.index)
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

feats=[]
def fn(view, sd):
    feats.append(mix(view, sd).assign(snapshot_day=sd).reset_index())
    return pd.DataFrame(index=view.households)  # placeholder

agent_api.build_features(fn)
F = pd.concat(feats).rename(columns={'index':'household_key'})
p = agent_api.save_table(F, 'mix_v1.parquet')
print(p, F.shape)
print(F.isna().mean().round(3).to_string())


# ---- cell ----

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
