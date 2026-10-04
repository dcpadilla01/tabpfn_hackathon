import agent_api as A
base = A.load_saved('e009_ewma_longlags.parquet')
print(base.shape)
cols = list(base.columns)
print("KEYS:", [c for c in cols if c in ('household_key','snapshot_day')])
for i in range(0, len(cols), 12):
    print(i, "|".join(cols[i:i+12]))

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
v = A.snapshot(459)
t = v.transactions
p = v.products
print("tx rows:", len(t), "hh:", t.household_key.nunique(), "day range:", t.day.min(), t.day.max())
print("products rows:", len(p), "unique pid:", p.product_id.nunique())
print("coverage:", t.product_id.isin(p.product_id).mean())
print("n departments:", p.department.nunique(), "n commodities:", p.commodity_desc.nunique(), "n manufacturers:", p.manufacturer.nunique())
m = t.merge(p[['product_id','department','commodity_desc','manufacturer','brand']], on='product_id', how='left')
sp = m.groupby('commodity_desc').sales_value.sum().sort_values(ascending=False)
print(sp.head(25))
print("share top20 commodities:", sp.head(20).sum()/sp.sum())
sp2 = m.groupby('manufacturer').sales_value.sum().sort_values(ascending=False)
print(sp2.head(10))
print("manufacturer NaN:", m.manufacturer.isna().mean())
tt = A.train_targets()
print(tt.future_spend_4w.describe())

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

FRESH = {'PRODUCE','MEAT','MEAT-PCKGD','MEAT-WHSE','PORK','SEAFOOD','SEAFOOD-PCKGD','DELI',
         'DAIRY DELI','GRO BAKERY','PASTRY','SALAD BAR','FLORAL','PROD-WHS SALES'}

def fn(view, snapshot_day):
    sd = snapshot_day
    idx = pd.Index(view.households, name='household_key')
    p = view.products[['product_id','department','commodity_desc','manufacturer','brand']]
    t = view.transactions.merge(p, on='product_id', how='left')
    d = t['day'].values
    def W(lo, hi):
        return t[(d > sd - hi) & (d <= sd - lo)]
    w28, q28 = W(0,28), W(28,56)
    w56, q56 = W(0,56), W(56,112)
    w84, w182 = W(0,84), W(0,182)
    prior84 = W(28,112)
    out = pd.DataFrame(index=idx)
    eps = 1e-9

    def tot(df):
        return df.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0)

    def repeat(w, pw, col):
        s = w.groupby(['household_key', col])['sales_value'].sum().reset_index()
        sp = pw.groupby(['household_key', col])['sales_value'].sum().reset_index()[['household_key', col]]
        m = s.merge(sp, on=['household_key', col], how='inner')
        rep = m.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0)
        return rep / (tot(w) + eps)

    def topk_share(w, k, col):
        s = w.groupby(['household_key', col])['sales_value'].sum().reset_index()
        s = s.sort_values(['household_key', 'sales_value'], ascending=[True, False])
        tk = s.groupby('household_key', sort=False).head(k)
        return tk.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0) / (tot(w) + eps)

    def entropy(w, col):
        s = w.groupby(['household_key', col])['sales_value'].sum()
        tt = s.groupby(level=0).transform('sum')
        pp = s / (tt + eps)
        return (-(pp * np.log(pp + eps))).groupby(level=0).sum().reindex(idx)

    def nuniq(w, col):
        return w.groupby('household_key')[col].nunique().reindex(idx).fillna(0.0)

    def share_mask(w, mask):
        sp = w.loc[mask].groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0)
        return sp / (tot(w) + eps)

    wine = set(c for c in t['commodity_desc'].dropna().unique() if 'WINE' in str(c).upper())
    ALC = {'BEERS/ALES','SPIRITS'} | wine

    # repeat-purchase loyalty
    out['rep_com_28'] = repeat(w28, q28, 'commodity_desc')
    out['rep_com_56'] = repeat(w56, q56, 'commodity_desc')
    out['rep_prod_28'] = repeat(w28, q28, 'product_id')
    f28, fq28 = w28[w28.department.isin(FRESH)], q28[q28.department.isin(FRESH)]
    out['rep_fresh_28'] = repeat(f28, fq28, 'commodity_desc')
    # concentration
    out['top1_com_28'] = topk_share(w28, 1, 'commodity_desc')
    out['top5_com_28'] = topk_share(w28, 5, 'commodity_desc')
    out['top10_com_28'] = topk_share(w28, 10, 'commodity_desc')
    out['top3_com_56'] = topk_share(w56, 3, 'commodity_desc')
    out['top1_man_84'] = topk_share(w84, 1, 'manufacturer')
    out['top5_man_84'] = topk_share(w84, 5, 'manufacturer')
    out['top1_man_28'] = topk_share(w28, 1, 'manufacturer')
    # breadth / entropy
    out['n_com_28'] = nuniq(w28, 'commodity_desc')
    out['n_com_84'] = nuniq(w84, 'commodity_desc')
    out['n_com_182'] = nuniq(w182, 'commodity_desc')
    out['n_prod_28'] = nuniq(w28, 'product_id')
    out['n_prod_182'] = nuniq(w182, 'product_id')
    out['n_man_84'] = nuniq(w84, 'manufacturer')
    out['n_man_182'] = nuniq(w182, 'manufacturer')
    out['ent_com_28'] = entropy(w28, 'commodity_desc')
    out['ent_com_84'] = entropy(w84, 'commodity_desc')
    out['ent_man_84'] = entropy(w84, 'manufacturer')
    out['n_com_ratio'] = out['n_com_28'] * 3.0 / (out['n_com_84'] + eps)
    # mix: fresh vs packaged, vice, special
    out['fresh_share_28'] = share_mask(w28, w28.department.isin(FRESH))
    out['fresh_share_84'] = share_mask(w84, w84.department.isin(FRESH))
    out['alcohol_28'] = share_mask(w28, w28.commodity_desc.isin(ALC))
    out['alcohol_84'] = share_mask(w84, w84.commodity_desc.isin(ALC))
    out['tobacco_28'] = share_mask(w28, w28.commodity_desc.eq('CIGARETTES'))
    out['tobacco_84'] = share_mask(w84, w84.commodity_desc.eq('CIGARETTES'))
    out['couponmisc_28'] = share_mask(w28, w28.commodity_desc.eq('COUPON/MISC ITEMS'))
    out['gas_28'] = share_mask(w28, w28.department.eq('KIOSK-GAS'))
    out['private_182'] = share_mask(w182, w182.brand.eq('Private'))
    # brand loyalty: share of 28d spend on the HH's dominant 84d brand
    s = w84.groupby(['household_key','brand'])['sales_value'].sum().reset_index()
    s = s.sort_values(['household_key','sales_value'], ascending=[True, False])
    topb = s.groupby('household_key', sort=False).head(1)[['household_key','brand']]
    mb = w28.merge(topb, on=['household_key','brand'], how='inner')
    out['brand_loy_28'] = mb.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0) / (tot(w28) + eps)
    # staple momentum: top-3 commodities of w28 vs their spend in prior 28d
    s28 = w28.groupby(['household_key','commodity_desc'])['sales_value'].sum().reset_index()
    s28 = s28.sort_values(['household_key','sales_value'], ascending=[True, False])
    top3 = s28.groupby('household_key', sort=False).head(3)[['household_key','commodity_desc']]
    sq = q28.groupby(['household_key','commodity_desc'])['sales_value'].sum().reset_index().rename(columns={'sales_value':'s_prev'})
    mm = top3.merge(sq, on=['household_key','commodity_desc'], how='left').fillna({'s_prev':0.0})
    cur = mm.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0)
    prv = mm.groupby('household_key')['s_prev'].sum().reindex(idx).fillna(0.0)
    out['mom_com'] = cur / (prv + 5.0)
    # product stickiness: share of 28d spend on HH's top-20 products of 182d
    s2 = w182.groupby(['household_key','product_id'])['sales_value'].sum().reset_index()
    s2 = s2.sort_values(['household_key','sales_value'], ascending=[True, False])
    top20 = s2.groupby('household_key', sort=False).head(20)[['household_key','product_id']]
    mp = w28.merge(top20, on=['household_key','product_id'], how='inner')
    out['prod_stick_28'] = mp.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0) / (tot(w28) + eps)
    # newness: share of 28d spend on commodities NOT bought in prior 84d
    pk = prior84[['household_key','commodity_desc']].drop_duplicates().assign(_k=1)
    s28k = w28.groupby(['household_key','commodity_desc'])['sales_value'].sum().reset_index().assign(_k=1)
    mg = s28k.merge(pk, on=['household_key','commodity_desc','_k'], how='left', indicator=True)
    newsp = mg.loc[mg['_merge']=='left_only'].groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0)
    out['new_com_28'] = newsp / (tot(w28) + eps)
    return out

print("building...")
feat = A.build_features(fn)
print(feat.shape)
base = A.load_saved('e009_ewma_longlags.parquet')
m = base.merge(feat.reset_index(), on=['household_key','snapshot_day'], how='left')
print("merged:", m.shape)
assert len(m) == len(base) == 36426
tt = A.train_targets()
tr = m.merge(tt, on=['household_key','snapshot_day'], how='inner')
newcols = [c for c in feat.reset_index().columns if c not in ('household_key','snapshot_day')]
cor = tr[newcols].corrwith(tr['future_spend_4w']).sort_values(key=np.abs, ascending=False)
print(cor.round(3))
path = A.save_table(m, 'product_habits.parquet')
print(path)

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

def fn(view, snapshot_day):
    t = view.transactions
    info = [list(t.columns), type(t)]
    p = view.products[['product_id','department','commodity_desc','manufacturer','brand']]
    m = t.merge(p, on='product_id', how='left')
    info.append(list(m.columns))
    print(info)
    return pd.DataFrame(index=pd.Index(view.households, name='household_key'))

f = A.build_features(fn)
print(f.shape)

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

def fn(view, snapshot_day):
    t = view.transactions
    p = view.products[['product_id','department','commodity_desc','manufacturer','brand']]
    m = t.merge(p, on='product_id', how='left')
    idx = pd.Index(view.households, name='household_key')
    return pd.DataFrame({
        'tcols': ' | '.join(t.columns),
        'mcols': ' | '.join(m.columns),
        'n_tx': len(t), 'n_m': len(m),
    }, index=[idx[0]])

f = A.build_features(fn)
print(f.iloc[0,0]); print(f.iloc[0,1]); print(f.iloc[0,2], f.iloc[0,3])

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

def fn(view, snapshot_day):
    sd = snapshot_day
    t = view.transactions
    p = view.products[['product_id','department','commodity_desc','manufacturer','brand']]
    m = t.merge(p, on='product_id', how='left')
    d = m['day'].values
    w = m[(d > sd-28) & (d <= sd)]
    idx = pd.Index(view.households, name='household_key')
    res = {
        'wcols': ' | '.join(w.columns),
        'wshape': str(w.shape),
        'gsum': str(w.groupby('household_key')['sales_value'].sum().sum()),
    }
    # also try filtering transactions first then merging
    t2 = t[(t.day > sd-28) & (t.day <= sd)]
    m2 = t2.merge(p, on='product_id', how='left')
    res['m2cols'] = ' | '.join(m2.columns)[:200]
    res['gsum2'] = str(m2.groupby('household_key')['sales_value'].sum().sum())
    return pd.DataFrame(res, index=[idx[0]])

f = A.build_features(fn)
for c in f.columns: print(c, '=>', f.iloc[0][c])

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

FRESH = {'PRODUCE','MEAT','MEAT-PCKGD','MEAT-WHSE','PORK','SEAFOOD','SEAFOOD-PCKGD','DELI',
         'DAIRY DELI','GRO BAKERY','PASTRY','SALAD BAR','FLORAL','PROD-WHS SALES'}

def fn(view, snapshot_day):
    sd = snapshot_day
    idx = pd.Index(view.households, name='household_key')
    p = view.products[['product_id','department','commodity_desc','manufacturer','brand']]
    t = view.transactions.merge(p, on='product_id', how='left')
    d = t['day'].values
    def W(lo, hi):
        return t[(d > sd - hi) & (d <= sd - lo)]
    w28, q28 = W(0,28), W(28,56)
    w56, q56 = W(0,56), W(56,112)
    w84, w182 = W(0,84), W(0,182)
    prior84 = W(28,112)
    out = pd.DataFrame(index=idx)
    eps = 1e-9

    def tot(df):
        return df.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0)

    def repeat(w, pw, col):
        s = w.groupby(['household_key', col])['sales_value'].sum().reset_index()
        sp = pw.groupby(['household_key', col])['sales_value'].sum().reset_index()[['household_key', col]]
        m = s.merge(sp, on=['household_key', col], how='inner')
        rep = m.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0)
        return rep / (tot(w) + eps)

    def topk_share(w, k, col):
        s = w.groupby(['household_key', col])['sales_value'].sum().reset_index()
        s = s.sort_values(['household_key', 'sales_value'], ascending=[True, False])
        tk = s.groupby('household_key', sort=False).head(k)
        return tk.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0) / (tot(w) + eps)

    def entropy(w, col):
        s = w.groupby(['household_key', col])['sales_value'].sum()
        tt = s.groupby(level=0).transform('sum')
        pp = s / (tt + eps)
        return (-(pp * np.log(pp + eps))).groupby(level=0).sum().reindex(idx)

    def nuniq(w, col):
        return w.groupby('household_key')[col].nunique().reindex(idx).fillna(0.0)

    def share_mask(w, mask):
        sp = w.loc[mask].groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0)
        return sp / (tot(w) + eps)

    wine = set(c for c in t['commodity_desc'].dropna().unique() if 'WINE' in str(c).upper())
    ALC = {'BEERS/ALES','SPIRITS'} | wine

    out['rep_com_28'] = repeat(w28, q28, 'commodity_desc')
    out['rep_com_56'] = repeat(w56, q56, 'commodity_desc')
    out['rep_prod_28'] = repeat(w28, q28, 'product_id')
    f28, fq28 = w28[w28.department.isin(FRESH)], q28[q28.department.isin(FRESH)]
    out['rep_fresh_28'] = repeat(f28, fq28, 'commodity_desc')
    out['top1_com_28'] = topk_share(w28, 1, 'commodity_desc')
    out['top5_com_28'] = topk_share(w28, 5, 'commodity_desc')
    out['top10_com_28'] = topk_share(w28, 10, 'commodity_desc')
    out['top3_com_56'] = topk_share(w56, 3, 'commodity_desc')
    out['top1_man_84'] = topk_share(w84, 1, 'manufacturer')
    out['top5_man_84'] = topk_share(w84, 5, 'manufacturer')
    out['top1_man_28'] = topk_share(w28, 1, 'manufacturer')
    out['n_com_28'] = nuniq(w28, 'commodity_desc')
    out['n_com_84'] = nuniq(w84, 'commodity_desc')
    out['n_com_182'] = nuniq(w182, 'commodity_desc')
    out['n_prod_28'] = nuniq(w28, 'product_id')
    out['n_prod_182'] = nuniq(w182, 'product_id')
    out['n_man_84'] = nuniq(w84, 'manufacturer')
    out['n_man_182'] = nuniq(w182, 'manufacturer')
    out['ent_com_28'] = entropy(w28, 'commodity_desc')
    out['ent_com_84'] = entropy(w84, 'commodity_desc')
    out['ent_man_84'] = entropy(w84, 'manufacturer')
    out['n_com_ratio'] = out['n_com_28'] * 3.0 / (out['n_com_84'] + eps)
    out['fresh_share_28'] = share_mask(w28, w28.department.isin(FRESH))
    out['fresh_share_84'] = share_mask(w84, w84.department.isin(FRESH))
    out['alcohol_28'] = share_mask(w28, w28.commodity_desc.isin(ALC))
    out['alcohol_84'] = share_mask(w84, w84.commodity_desc.isin(ALC))
    out['tobacco_28'] = share_mask(w28, w28.commodity_desc.eq('CIGARETTES'))
    out['tobacco_84'] = share_mask(w84, w84.commodity_desc.eq('CIGARETTES'))
    out['couponmisc_28'] = share_mask(w28, w28.commodity_desc.eq('COUPON/MISC ITEMS'))
    out['gas_28'] = share_mask(w28, w28.department.eq('KIOSK-GAS'))
    out['private_182'] = share_mask(w182, w182.brand.eq('Private'))
    s = w84.groupby(['household_key','brand'])['sales_value'].sum().reset_index()
    s = s.sort_values(['household_key','sales_value'], ascending=[True, False])
    topb = s.groupby('household_key', sort=False).head(1)[['household_key','brand']]
    mb = w28.merge(topb, on=['household_key','brand'], how='inner')
    out['brand_loy_28'] = mb.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0) / (tot(w28) + eps)
    s28 = w28.groupby(['household_key','commodity_desc'])['sales_value'].sum().reset_index()
    s28 = s28.sort_values(['household_key','sales_value'], ascending=[True, False])
    top3 = s28.groupby('household_key', sort=False).head(3)
    sq = q28.groupby(['household_key','commodity_desc'])['sales_value'].sum().reset_index().rename(columns={'sales_value':'s_prev'})
    mm = top3.merge(sq, on=['household_key','commodity_desc'], how='left').fillna({'s_prev':0.0})
    cur = mm.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0)
    prv = mm.groupby('household_key')['s_prev'].sum().reindex(idx).fillna(0.0)
    out['mom_com'] = cur / (prv + 5.0)
    s2 = w182.groupby(['household_key','product_id'])['sales_value'].sum().reset_index()
    s2 = s2.sort_values(['household_key','sales_value'], ascending=[True, False])
    top20 = s2.groupby('household_key', sort=False).head(20)[['household_key','product_id']]
    mp = w28.merge(top20, on=['household_key','product_id'], how='inner')
    out['prod_stick_28'] = mp.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0) / (tot(w28) + eps)
    pk = prior84[['household_key','commodity_desc']].drop_duplicates().assign(_k=1)
    s28k = w28.groupby(['household_key','commodity_desc'])['sales_value'].sum().reset_index().assign(_k=1)
    mg = s28k.merge(pk, on=['household_key','commodity_desc','_k'], how='left', indicator=True)
    newsp = mg.loc[mg['_merge']=='left_only'].groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0)
    out['new_com_28'] = newsp / (tot(w28) + eps)
    return out

print("building...")
feat = A.build_features(fn)
print(feat.shape)
base = A.load_saved('e009_ewma_longlags.parquet')
m = base.merge(feat.reset_index(), on=['household_key','snapshot_day'], how='left')
print("merged:", m.shape)
assert len(m) == len(base) == 36426
tt = A.train_targets()
tr = m.merge(tt, on=['household_key','snapshot_day'], how='inner')
newcols = [c for c in feat.reset_index().columns if c not in ('household_key','snapshot_day')]
cor = tr[newcols].corrwith(tr['future_spend_4w']).sort_values(key=np.abs, ascending=False)
print(cor.round(3))
path = A.save_table(m, 'product_habits.parquet')
print(path)

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

FRESH = {'PRODUCE','MEAT','MEAT-PCKGD','MEAT-WHSE','PORK','SEAFOOD','SEAFOOD-PCKGD','DELI',
         'DAIRY DELI','GRO BAKERY','PASTRY','SALAD BAR','FLORAL','PROD-WHS SALES'}

def fn(view, snapshot_day):
    sd = snapshot_day
    idx = pd.Index(view.households, name='household_key')
    p = view.products[['product_id','department','commodity_desc','manufacturer','brand']]
    t = view.transactions.merge(p, on='product_id', how='left')
    d = t['day'].values
    def W(lo, hi):
        return t[(d > sd - hi) & (d <= sd - lo)]
    w28, q28 = W(0,28), W(28,56)
    w56, q56 = W(0,56), W(56,112)
    w84, w182 = W(0,84), W(0,182)
    prior84 = W(28,112)
    out = pd.DataFrame(index=idx)
    eps = 1e-9
    def tot(df):
        return df.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0)
    def repeat(w, pw, col):
        s = w.groupby(['household_key', col])['sales_value'].sum().reset_index()
        sp = pw.groupby(['household_key', col])['sales_value'].sum().reset_index()[['household_key', col]]
        m = s.merge(sp, on=['household_key', col], how='inner')
        rep = m.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0)
        return rep / (tot(w) + eps)
    def topk_share(w, k, col):
        s = w.groupby(['household_key', col])['sales_value'].sum().reset_index()
        s = s.sort_values(['household_key', 'sales_value'], ascending=[True, False])
        tk = s.groupby('household_key', sort=False).head(k)
        return tk.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0) / (tot(w) + eps)
    def entropy(w, col):
        s = w.groupby(['household_key', col])['sales_value'].sum()
        tt = s.groupby(level=0).transform('sum')
        pp = s / (tt + eps)
        return (-(pp * np.log(pp + eps))).groupby(level=0).sum().reindex(idx)
    def nuniq(w, col):
        return w.groupby('household_key')[col].nunique().reindex(idx).fillna(0.0)
    def share_mask(w, mask):
        sp = w.loc[mask].groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0)
        return sp / (tot(w) + eps)
    wine = set(c for c in t['commodity_desc'].dropna().unique() if 'WINE' in str(c).upper())
    ALC = {'BEERS/ALES','SPIRITS'} | wine
    out['rep_com_28'] = repeat(w28, q28, 'commodity_desc')
    out['rep_com_56'] = repeat(w56, q56, 'commodity_desc')
    out['rep_prod_28'] = repeat(w28, q28, 'product_id')
    f28, fq28 = w28[w28.department.isin(FRESH)], q28[q28.department.isin(FRESH)]
    out['rep_fresh_28'] = repeat(f28, fq28, 'commodity_desc')
    out['top1_com_28'] = topk_share(w28, 1, 'commodity_desc')
    out['top5_com_28'] = topk_share(w28, 5, 'commodity_desc')
    out['top10_com_28'] = topk_share(w28, 10, 'commodity_desc')
    out['top3_com_56'] = topk_share(w56, 3, 'commodity_desc')
    out['top1_man_84'] = topk_share(w84, 1, 'manufacturer')
    out['top5_man_84'] = topk_share(w84, 5, 'manufacturer')
    out['top1_man_28'] = topk_share(w28, 1, 'manufacturer')
    out['n_com_28'] = nuniq(w28, 'commodity_desc')
    out['n_com_84'] = nuniq(w84, 'commodity_desc')
    out['n_com_182'] = nuniq(w182, 'commodity_desc')
    out['n_prod_28'] = nuniq(w28, 'product_id')
    out['n_prod_182'] = nuniq(w182, 'product_id')
    out['n_man_84'] = nuniq(w84, 'manufacturer')
    out['n_man_182'] = nuniq(w182, 'manufacturer')
    out['ent_com_28'] = entropy(w28, 'commodity_desc')
    out['ent_com_84'] = entropy(w84, 'commodity_desc')
    out['ent_man_84'] = entropy(w84, 'manufacturer')
    out['n_com_ratio'] = out['n_com_28'] * 3.0 / (out['n_com_84'] + eps)
    out['fresh_share_28'] = share_mask(w28, w28.department.isin(FRESH))
    out['fresh_share_84'] = share_mask(w84, w84.department.isin(FRESH))
    out['alcohol_28'] = share_mask(w28, w28.commodity_desc.isin(ALC))
    out['alcohol_84'] = share_mask(w84, w84.commodity_desc.isin(ALC))
    out['tobacco_28'] = share_mask(w28, w28.commodity_desc.eq('CIGARETTES'))
    out['tobacco_84'] = share_mask(w84, w84.commodity_desc.eq('CIGARETTES'))
    out['couponmisc_28'] = share_mask(w28, w28.commodity_desc.eq('COUPON/MISC ITEMS'))
    out['gas_28'] = share_mask(w28, w28.department.eq('KIOSK-GAS'))
    out['private_182'] = share_mask(w182, w182.brand.eq('Private'))
    s = w84.groupby(['household_key','brand'])['sales_value'].sum().reset_index()
    s = s.sort_values(['household_key','sales_value'], ascending=[True, False])
    topb = s.groupby('household_key', sort=False).head(1)[['household_key','brand']]
    mb = w28.merge(topb, on=['household_key','brand'], how='inner')
    out['brand_loy_28'] = mb.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0) / (tot(w28) + eps)
    s28 = w28.groupby(['household_key','commodity_desc'])['sales_value'].sum().reset_index()
    s28 = s28.sort_values(['household_key','sales_value'], ascending=[True, False])
    top3 = s28.groupby('household_key', sort=False).head(3)
    sq = q28.groupby(['household_key','commodity_desc'])['sales_value'].sum().reset_index().rename(columns={'sales_value':'s_prev'})
    mm = top3.merge(sq, on=['household_key','commodity_desc'], how='left').fillna({'s_prev':0.0})
    cur = mm.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0)
    prv = mm.groupby('household_key')['s_prev'].sum().reindex(idx).fillna(0.0)
    out['mom_com'] = cur / (prv + 5.0)
    s2 = w182.groupby(['household_key','product_id'])['sales_value'].sum().reset_index()
    s2 = s2.sort_values(['household_key','sales_value'], ascending=[True, False])
    top20 = s2.groupby('household_key', sort=False).head(20)[['household_key','product_id']]
    mp = w28.merge(top20, on=['household_key','product_id'], how='inner')
    out['prod_stick_28'] = mp.groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0) / (tot(w28) + eps)
    pk = prior84[['household_key','commodity_desc']].drop_duplicates().assign(_k=1)
    s28k = w28.groupby(['household_key','commodity_desc'])['sales_value'].sum().reset_index().assign(_k=1)
    mg = s28k.merge(pk, on=['household_key','commodity_desc','_k'], how='left', indicator=True)
    newsp = mg.loc[mg['_merge']=='left_only'].groupby('household_key')['sales_value'].sum().reindex(idx).fillna(0.0)
    out['new_com_28'] = newsp / (tot(w28) + eps)
    return out

print("building...")
feat = A.build_features(fn)
print(feat.shape)
base = A.load_saved('e009_ewma_longlags.parquet')
m = base.merge(feat.reset_index(), on=['household_key','snapshot_day'], how='left')
print("merged:", m.shape)
assert len(m) == len(base) == 36426
tt = A.train_targets()
tr = m.merge(tt, on=['household_key','snapshot_day'], how='inner')
newcols = [c for c in feat.columns if c not in ('household_key','snapshot_day')]
cor = tr[newcols].corrwith(tr['future_spend_4w']).sort_values(key=np.abs, ascending=False)
print(cor.round(3))
path = A.save_table(m, 'product_habits.parquet')
print(path)