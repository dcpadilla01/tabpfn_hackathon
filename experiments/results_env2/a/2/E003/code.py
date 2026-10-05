import agent_api as api, pandas as pd, numpy as np, xgboost as xgb
print('xgb version', xgb.__version__)
OLD = api.load_saved('feats_v2.parquet')
print('OLD shape', OLD.shape)
print('dup keys', OLD.duplicated(['household_key','snapshot_day']).sum())
print('OLD cols:', list(OLD.columns))
tt = api.train_targets()
print('targets', tt.shape, 'hh', tt.household_key.nunique(), 'days', sorted(tt.snapshot_day.unique()))
print(tt.future_spend_4w.describe())
v = api.snapshot()
tx = v.table('transactions'); pr = v.table('products')
print('tx', tx.shape, 'day range', tx.day.min(), tx.day.max())
print('brand vals:'); print(pr.brand.value_counts(dropna=False))
print('dept nunique', pr.department.nunique()); print(pr.department.value_counts().head(12))
print('hh needing row at 459:', len(v.households))
print('snapshot_days', api.snapshot_days())
print(tx[['sales_value','quantity']].describe())


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np
v = api.snapshot()
print(type(v.households), v.households if v.households is None else len(v.households))
print([a for a in dir(v) if not a.startswith('_')])
print('day', v.day, 'week', v.week)
tt = api.train_targets()
print('n hh per day:'); print(tt.groupby('snapshot_day').household_key.nunique())
# how many val rows expected?
OLD = api.load_saved('feats_v2.parquet')
print(OLD.groupby('snapshot_day').size())
print(OLD[OLD.snapshot_day>=459].shape)


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np, xgboost as xgb, time

WINS=[7,14,28,56,84,112,224]
TOP_DEPTS=['GROCERY','DRUG GM','PRODUCE','COSMETICS','NUTRITION','MEAT','MEAT-PCKGD','DELI','PASTRY','FLORAL']

def build(v, sd):
    hh = pd.Index(list(v.households), name='household_key')
    t = v.table('transactions')
    t = t[t.household_key.isin(hh)]
    t = t[t.day <= sd]
    prod = v.table('products')[['product_id','department','brand']]
    X = pd.DataFrame(index=hh)
    if len(t)==0:
        for c in ['spend_all']: X[c]=0.0
        return X
    # basket level
    b = t.groupby(['household_key','basket_id'], as_index=False).agg(
        day=('day','max'), spend=('sales_value','sum'), items=('sales_value','size'),
        qty=('quantity','sum'), store=('store_id','first'), tt=('trans_time','first'))
    def agg_win(w, suffix, extra=False):
        m = b[b.day > sd-w] if w else b
        g = m.groupby('household_key')
        X['spend_'+suffix] = g.spend.sum().reindex(hh).fillna(0.0)
        X['baskets_'+suffix] = g.size().reindex(hh).fillna(0).astype(float)
        if extra:
            tm = t[t.day > sd-w]
            gp = tm.groupby('household_key')['product_id']
            X['prods_'+suffix] = gp.nunique().reindex(hh).fillna(0)
            X['stores_'+suffix] = tm.groupby('household_key')['store_id'].nunique().reindex(hh).fillna(0)
            X['qty_'+suffix] = tm.groupby('household_key')['quantity'].sum().reindex(hh).fillna(0)
            X['retail_disc_'+suffix] = tm.groupby('household_key')['retail_disc'].sum().reindex(hh).fillna(0)
            X['coupon_disc_'+suffix] = tm.groupby('household_key')['coupon_disc'].sum().reindex(hh).fillna(0)
            X['spend_max_basket_'+suffix] = g.spend.max().reindex(hh).fillna(0.0)
    agg_win(0,'all',extra=True)
    for w in WINS: agg_win(w, str(w), extra=(w in (28,84)))
    last = b.groupby('household_key')['day'].max().reindex(hh)
    first = t.groupby('household_key')['day'].min().reindex(hh)
    X['days_since_last'] = (sd-last).clip(lower=0)
    X['days_since_first'] = (sd-first).clip(lower=0)
    # recency-weighted spend (half-life 28d) over 112d
    m = b[b.day > sd-112].copy()
    m['w'] = 0.5**((sd-m.day)/28.0)
    X['spend_decay_112'] = (m.spend*m.w).groupby(m.household_key).sum().reindex(hh).fillna(0.0)
    # trends
    X['trend_28_56'] = (X.spend_28/(X.spend_56/2+1.0)).clip(0,10)
    X['trend_56_112'] = (X.spend_56/2/(X.spend_112/4+1.0)).clip(0,10)
    X['trend_84_224'] = (X.spend_84/3/(X.spend_224/8+1.0)).clip(0,10)
    X['share_recent_all'] = X.spend_28/(X.spend_all+1.0)
    # basket composition 84d
    b84 = b[b.day > sd-84]
    g84 = b84.groupby('household_key')
    X['avg_basket_84'] = g84.spend.mean().reindex(hh).fillna(0.0)
    X['basket_std_84'] = g84.spend.std().reindex(hh).fillna(0.0)
    X['basket_min_84'] = g84.spend.min().reindex(hh).fillna(0.0)
    X['items_per_basket_84'] = g84.items.mean().reindex(hh).fillna(0.0)
    X['qty_per_basket_84'] = g84.qty.mean().reindex(hh).fillna(0.0)
    X['baskets_per_day_84'] = X.baskets_84/84.0
    X['avg_basket_28'] = b[b.day>sd-28].groupby('household_key').spend.mean().reindex(hh).fillna(0.0)
    for w in (28,84,112): X['active_'+str(w)] = (X['baskets_'+str(w)]>0).astype(float)
    # gaps
    b112 = b[b.day > sd-112].sort_values(['household_key','day'])
    gaps = b112.groupby('household_key')['day'].diff()
    X['gap_mean_112'] = gaps.groupby(b112.household_key).mean().reindex(hh).fillna(60.0)
    X['gap_std_112'] = gaps.groupby(b112.household_key).std().reindex(hh).fillna(0.0)
    # time of day
    hr = (b84.tt//100).where(b84.tt>0)
    X['hour_mean_84'] = hr.groupby(b84.household_key).mean().reindex(hh).fillna(12.0)
    X['hour_std_84'] = hr.groupby(b84.household_key).std().reindex(hh).fillna(0.0)
    X['evening_share_84'] = (hr>=17).groupby(b84.household_key).mean().reindex(hh).fillna(0.0)
    # dept / brand mix
    t84 = t[t.day > sd-84][['household_key','product_id','sales_value']].merge(prod, on='product_id', how='left')
    t84['sales_value'] = t84.sales_value.fillna(0.0)
    t84['department'] = t84.department.fillna('OTHER')
    ds = t84.groupby(['household_key','department'])['sales_value'].sum().unstack(fill_value=0.0)
    tot = ds.sum(axis=1).reindex(hh).fillna(0.0)+1e-6
    for d in TOP_DEPTS:
        X['dept_share_'+d[:6].replace(' ','')] = (ds[d] if d in ds.columns else 0.0).reindex(hh).fillna(0.0)/tot
    known = [d for d in TOP_DEPTS if d in ds.columns]
    X['dept_other_share_84'] = ((tot-1e-6 - ds[known].sum(axis=1).reindex(hh).fillna(0.0)).clip(lower=0))/tot
    X['n_depts_84'] = t84.groupby('household_key')['department'].nunique().reindex(hh).fillna(0)
    bs = t84.assign(priv=(t84.brand=='Private').astype(float)*t84.sales_value).groupby('household_key')['priv'].sum().reindex(hh).fillna(0.0)
    X['private_share_84'] = bs/tot
    st = b84.groupby(['household_key','store']).size().unstack(fill_value=0.0)
    X['top_store_share_84'] = (st.max(axis=1) if len(st) else 0.0)
    X['top_store_share_84'] = X['top_store_share_84'].reindex(hh).fillna(0.0)/(X.baskets_84+1e-6)
    X['spend_rate_all_28'] = X.spend_all/(X.days_since_first+1.0)*28.0
    # campaigns
    cs = v.table('campaigns'); ct = v.table('campaign_targets')
    cdesc = dict(zip(cs.campaign, cs.description.astype(str))) if len(cs) else {}
    ct = ct[ct.household_key.isin(hh)]
    ct2 = ct.assign(desc=ct.campaign.map(cdesc).fillna(''))
    X['n_campaigns'] = ct2.groupby('household_key')['campaign'].nunique().reindex(hh).fillna(0)
    for ty in ['TypeA','TypeB','TypeC']:
        X['ct_'+ty] = ct2[ct2.desc.str.contains(ty)].groupby('household_key').size().reindex(hh).fillna(0)
    r = v.table('coupon_redemptions'); r = r[r.household_key.isin(hh) & (r.day<=sd)]
    X['redemp_all'] = r.groupby('household_key').size().reindex(hh).fillna(0)
    X['redemp_84'] = r[r.day>sd-84].groupby('household_key').size().reindex(hh).fillna(0)
    X['redemp_28'] = r[r.day>sd-28].groupby('household_key').size().reindex(hh).fillna(0)
    rlast = r.groupby('household_key')['day'].max().reindex(hh)
    X['days_since_redemp'] = (sd-rlast).fillna(999).clip(lower=0)
    # display exposure
    dm = v.table('display_mailer'); wk = (sd+8)//7
    dm = dm[(dm.week_no<=wk)&(dm.week_no>wk-4)]
    disp_p = set(dm.product_id.unique()) if len(dm) else set()
    t28 = t[t.day>sd-28]
    X['spend28_on_disp'] = t28[t28.product_id.isin(disp_p)].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
    X['share28_on_disp'] = X.spend28_on_disp/(X.spend_28+1e-6)
    # demographics
    d = v.table('demographics')
    d = d.set_index('household_key').reindex(hh)
    for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
        X['dem_'+c] = d[c].astype('category').cat.codes.values
    X['has_demo'] = d['classification_1'].notna().astype(float).values
    X['week_of_year'] = float(wk % 52)
    return X.astype(float)

t0=time.time()
feats = api.build_features(build)
print('build time', round(time.time()-t0,1), 'shape', feats.shape)
print('n features', feats.shape[1]-2)
print('NaNs total', int(feats.isna().sum().sum()))
print('inf', int(np.isinf(feats.drop(columns=['household_key','snapshot_day']).to_numpy()).sum()))
p = api.save_table(feats, 'feats_v3.parquet'); print('saved', p)


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np, xgboost as xgb, time

WINS=[7,14,28,56,84,112,224]
TOP_DEPTS=['GROCERY','DRUG GM','PRODUCE','COSMETICS','NUTRITION','MEAT','MEAT-PCKGD','DELI','PASTRY','FLORAL']

def build(v, sd):
    hh = pd.Index(list(v.households), name='household_key')
    t = v.table('transactions')
    t = t[t.household_key.isin(hh)]
    t = t[t.day <= sd]
    prod = v.table('products')[['product_id','department','brand']]
    X = pd.DataFrame(index=hh)
    if len(t)==0:
        for c in ['spend_all']: X[c]=0.0
        return X
    b = t.groupby(['household_key','basket_id'], as_index=False).agg(
        day=('day','max'), spend=('sales_value','sum'), items=('sales_value','size'),
        qty=('quantity','sum'), store=('store_id','first'), tt=('trans_time','first'))
    def agg_win(w, suffix, extra=False):
        m = b[b.day > sd-w] if w else b
        g = m.groupby('household_key')
        X['spend_'+suffix] = g.spend.sum().reindex(hh).fillna(0.0)
        X['baskets_'+suffix] = g.size().reindex(hh).fillna(0).astype(float)
        if extra:
            tm = t[t.day > sd-w]
            X['prods_'+suffix] = tm.groupby('household_key')['product_id'].nunique().reindex(hh).fillna(0)
            X['stores_'+suffix] = tm.groupby('household_key')['store_id'].nunique().reindex(hh).fillna(0)
            X['qty_'+suffix] = tm.groupby('household_key')['quantity'].sum().reindex(hh).fillna(0)
            X['retail_disc_'+suffix] = tm.groupby('household_key')['retail_disc'].sum().reindex(hh).fillna(0)
            X['coupon_disc_'+suffix] = tm.groupby('household_key')['coupon_disc'].sum().reindex(hh).fillna(0)
            X['spend_max_basket_'+suffix] = g.spend.max().reindex(hh).fillna(0.0)
    agg_win(0,'all',extra=True)
    for w in WINS: agg_win(w, str(w), extra=(w in (28,84)))
    last = b.groupby('household_key')['day'].max().reindex(hh)
    first = t.groupby('household_key')['day'].min().reindex(hh)
    X['days_since_last'] = (sd-last).clip(lower=0)
    X['days_since_first'] = (sd-first).clip(lower=0)
    m = b[b.day > sd-112].copy()
    m['w'] = 0.5**((sd-m.day)/28.0)
    X['spend_decay_112'] = (m.spend*m.w).groupby(m.household_key).sum().reindex(hh).fillna(0.0)
    X['trend_28_56'] = (X.spend_28/(X.spend_56/2+1.0)).clip(0,10)
    X['trend_56_112'] = (X.spend_56/2/(X.spend_112/4+1.0)).clip(0,10)
    X['trend_84_224'] = (X.spend_84/3/(X.spend_224/8+1.0)).clip(0,10)
    X['share_recent_all'] = X.spend_28/(X.spend_all+1.0)
    b84 = b[b.day > sd-84]
    g84 = b84.groupby('household_key')
    X['avg_basket_84'] = g84.spend.mean().reindex(hh).fillna(0.0)
    X['basket_std_84'] = g84.spend.std().reindex(hh).fillna(0.0)
    X['basket_min_84'] = g84.spend.min().reindex(hh).fillna(0.0)
    X['items_per_basket_84'] = g84.items.mean().reindex(hh).fillna(0.0)
    X['qty_per_basket_84'] = g84.qty.mean().reindex(hh).fillna(0.0)
    X['baskets_per_day_84'] = X.baskets_84/84.0
    X['avg_basket_28'] = b[b.day>sd-28].groupby('household_key').spend.mean().reindex(hh).fillna(0.0)
    for w in (28,84,112): X['active_'+str(w)] = (X['baskets_'+str(w)]>0).astype(float)
    b112 = b[b.day > sd-112].sort_values(['household_key','day'])
    gaps = b112.groupby('household_key')['day'].diff()
    X['gap_mean_112'] = gaps.groupby(b112.household_key).mean().reindex(hh).fillna(60.0)
    X['gap_std_112'] = gaps.groupby(b112.household_key).std().reindex(hh).fillna(0.0)
    hr = (b84.tt//100).where(b84.tt>0)
    X['hour_mean_84'] = hr.groupby(b84.household_key).mean().reindex(hh).fillna(12.0)
    X['hour_std_84'] = hr.groupby(b84.household_key).std().reindex(hh).fillna(0.0)
    X['evening_share_84'] = (hr>=17).groupby(b84.household_key).mean().reindex(hh).fillna(0.0)
    t84 = t[t.day > sd-84][['household_key','product_id','sales_value']].merge(prod, on='product_id', how='left')
    t84['sales_value'] = t84.sales_value.fillna(0.0).astype(float)
    t84['department'] = t84['department'].astype(object).fillna('OTHER')
    ds = t84.groupby(['household_key','department'])['sales_value'].sum().unstack(fill_value=0.0)
    tot = ds.sum(axis=1).reindex(hh).fillna(0.0)+1e-6
    for d in TOP_DEPTS:
        X['dept_share_'+d[:6].replace(' ','')] = (ds[d] if d in ds.columns else 0.0).reindex(hh).fillna(0.0)/tot
    known = [d for d in TOP_DEPTS if d in ds.columns]
    X['dept_other_share_84'] = ((tot-1e-6 - ds[known].sum(axis=1).reindex(hh).fillna(0.0)).clip(lower=0))/tot
    X['n_depts_84'] = t84.groupby('household_key')['department'].nunique().reindex(hh).fillna(0)
    bs = t84.assign(priv=(t84.brand.astype(object)=='Private').astype(float)*t84.sales_value).groupby('household_key')['priv'].sum().reindex(hh).fillna(0.0)
    X['private_share_84'] = bs/tot
    st = b84.groupby(['household_key','store']).size().unstack(fill_value=0.0)
    X['top_store_share_84'] = (st.max(axis=1) if len(st) else 0.0)
    X['top_store_share_84'] = X['top_store_share_84'].reindex(hh).fillna(0.0)/(X.baskets_84+1e-6)
    X['spend_rate_all_28'] = X.spend_all/(X.days_since_first+1.0)*28.0
    cs = v.table('campaigns'); ct = v.table('campaign_targets')
    cdesc = dict(zip(cs.campaign, cs.description.astype(str))) if len(cs) else {}
    ct = ct[ct.household_key.isin(hh)]
    ct2 = ct.assign(desc=ct.campaign.map(cdesc).fillna(''))
    X['n_campaigns'] = ct2.groupby('household_key')['campaign'].nunique().reindex(hh).fillna(0)
    for ty in ['TypeA','TypeB','TypeC']:
        X['ct_'+ty] = ct2[ct2.desc.str.contains(ty)].groupby('household_key').size().reindex(hh).fillna(0)
    r = v.table('coupon_redemptions'); r = r[r.household_key.isin(hh) & (r.day<=sd)]
    X['redemp_all'] = r.groupby('household_key').size().reindex(hh).fillna(0)
    X['redemp_84'] = r[r.day>sd-84].groupby('household_key').size().reindex(hh).fillna(0)
    X['redemp_28'] = r[r.day>sd-28].groupby('household_key').size().reindex(hh).fillna(0)
    rlast = r.groupby('household_key')['day'].max().reindex(hh)
    X['days_since_redemp'] = (sd-rlast).fillna(999).clip(lower=0)
    dm = v.table('display_mailer'); wk = (sd+8)//7
    dm = dm[(dm.week_no<=wk)&(dm.week_no>wk-4)]
    disp_p = set(dm.product_id.unique()) if len(dm) else set()
    t28 = t[t.day>sd-28]
    X['spend28_on_disp'] = t28[t28.product_id.isin(disp_p)].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
    X['share28_on_disp'] = X.spend28_on_disp/(X.spend_28+1e-6)
    d = v.table('demographics')
    d = d.set_index('household_key').reindex(hh)
    for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
        X['dem_'+c] = d[c].astype('category').cat.codes.values
    X['has_demo'] = d['classification_1'].notna().astype(float).values
    X['week_of_year'] = float(wk % 52)
    return X.astype(float)

t0=time.time()
feats = api.build_features(build)
print('build time', round(time.time()-t0,1), 'shape', feats.shape)
print('n features', feats.shape[1]-2)
print('NaNs total', int(feats.isna().sum().sum()))
num = feats.drop(columns=['household_key','snapshot_day']).to_numpy()
print('inf', int(np.isinf(num).sum()))
p = api.save_table(feats, 'feats_v3.parquet'); print('saved', p)


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np, xgboost as xgb, time

WINS=[7,14,28,56,84,112,224]
TOP_DEPTS=['GROCERY','DRUG GM','PRODUCE','COSMETICS','NUTRITION','MEAT','MEAT-PCKGD','DELI','PASTRY','FLORAL']

def build(v, sd):
    hh = pd.Index(list(v.households), name='household_key')
    t = v.table('transactions')
    t = t[t.household_key.isin(hh)]
    t = t[t.day <= sd]
    prod = v.table('products')[['product_id','department','brand']]
    X = pd.DataFrame(index=hh)
    if len(t)==0:
        for c in ['spend_all']: X[c]=0.0
        return X
    b = t.groupby(['household_key','basket_id'], as_index=False).agg(
        day=('day','max'), spend=('sales_value','sum'), items=('sales_value','size'),
        qty=('quantity','sum'), store=('store_id','first'), tt=('trans_time','first'))
    def agg_win(w, suffix, extra=False):
        m = b[b.day > sd-w] if w else b
        g = m.groupby('household_key')
        X['spend_'+suffix] = g.spend.sum().reindex(hh).fillna(0.0)
        X['baskets_'+suffix] = g.size().reindex(hh).fillna(0).astype(float)
        if extra:
            tm = t[t.day > sd-w]
            X['prods_'+suffix] = tm.groupby('household_key')['product_id'].nunique().reindex(hh).fillna(0)
            X['stores_'+suffix] = tm.groupby('household_key')['store_id'].nunique().reindex(hh).fillna(0)
            X['qty_'+suffix] = tm.groupby('household_key')['quantity'].sum().reindex(hh).fillna(0)
            X['retail_disc_'+suffix] = tm.groupby('household_key')['retail_disc'].sum().reindex(hh).fillna(0)
            X['coupon_disc_'+suffix] = tm.groupby('household_key')['coupon_disc'].sum().reindex(hh).fillna(0)
            X['spend_max_basket_'+suffix] = g.spend.max().reindex(hh).fillna(0.0)
    agg_win(0,'all',extra=True)
    for w in WINS: agg_win(w, str(w), extra=(w in (28,84)))
    last = b.groupby('household_key')['day'].max().reindex(hh)
    first = t.groupby('household_key')['day'].min().reindex(hh)
    X['days_since_last'] = (sd-last).clip(lower=0)
    X['days_since_first'] = (sd-first).clip(lower=0)
    m = b[b.day > sd-112].copy()
    m['w'] = 0.5**((sd-m.day)/28.0)
    X['spend_decay_112'] = (m.spend*m.w).groupby(m.household_key).sum().reindex(hh).fillna(0.0)
    X['trend_28_56'] = (X.spend_28/(X.spend_56/2+1.0)).clip(0,10)
    X['trend_56_112'] = (X.spend_56/2/(X.spend_112/4+1.0)).clip(0,10)
    X['trend_84_224'] = (X.spend_84/3/(X.spend_224/8+1.0)).clip(0,10)
    X['share_recent_all'] = X.spend_28/(X.spend_all+1.0)
    b84 = b[b.day > sd-84]
    g84 = b84.groupby('household_key')
    X['avg_basket_84'] = g84.spend.mean().reindex(hh).fillna(0.0)
    X['basket_std_84'] = g84.spend.std().reindex(hh).fillna(0.0)
    X['basket_min_84'] = g84.spend.min().reindex(hh).fillna(0.0)
    X['items_per_basket_84'] = g84.items.mean().reindex(hh).fillna(0.0)
    X['qty_per_basket_84'] = g84.qty.mean().reindex(hh).fillna(0.0)
    X['baskets_per_day_84'] = X.baskets_84/84.0
    X['avg_basket_28'] = b[b.day>sd-28].groupby('household_key').spend.mean().reindex(hh).fillna(0.0)
    for w in (28,84,112): X['active_'+str(w)] = (X['baskets_'+str(w)]>0).astype(float)
    b112 = b[b.day > sd-112].sort_values(['household_key','day'])
    gaps = b112.groupby('household_key')['day'].diff()
    X['gap_mean_112'] = gaps.groupby(b112.household_key).mean().reindex(hh).fillna(60.0)
    X['gap_std_112'] = gaps.groupby(b112.household_key).std().reindex(hh).fillna(0.0)
    hr = (b84.tt//100).where(b84.tt>0)
    X['hour_mean_84'] = hr.groupby(b84.household_key).mean().reindex(hh).fillna(12.0)
    X['hour_std_84'] = hr.groupby(b84.household_key).std().reindex(hh).fillna(0.0)
    X['evening_share_84'] = (hr>=17).groupby(b84.household_key).mean().reindex(hh).fillna(0.0)
    t84 = t[t.day > sd-84][['household_key','product_id','sales_value']].merge(prod, on='product_id', how='left')
    t84['sales_value'] = t84.sales_value.fillna(0.0).astype(float)
    t84['department'] = t84['department'].astype(object).fillna('OTHER')
    ds = t84.groupby(['household_key','department'])['sales_value'].sum().unstack(fill_value=0.0)
    tot = ds.sum(axis=1).reindex(hh).fillna(0.0)+1e-6
    for d in TOP_DEPTS:
        X['dept_share_'+d[:6].replace(' ','')] = (ds[d] if d in ds.columns else 0.0).reindex(hh).fillna(0.0)/tot
    known = [d for d in TOP_DEPTS if d in ds.columns]
    X['dept_other_share_84'] = ((tot-1e-6 - ds[known].sum(axis=1).reindex(hh).fillna(0.0)).clip(lower=0))/tot
    X['n_depts_84'] = t84.groupby('household_key')['department'].nunique().reindex(hh).fillna(0)
    bs = t84.assign(priv=(t84.brand.astype(object)=='Private').astype(float)*t84.sales_value).groupby('household_key')['priv'].sum().reindex(hh).fillna(0.0)
    X['private_share_84'] = bs/tot
    st = b84.groupby(['household_key','store']).size().unstack(fill_value=0.0)
    X['top_store_share_84'] = (st.max(axis=1) if len(st) else 0.0)
    X['top_store_share_84'] = X['top_store_share_84'].reindex(hh).fillna(0.0)/(X.baskets_84+1e-6)
    X['spend_rate_all_28'] = X.spend_all/(X.days_since_first+1.0)*28.0
    cs = v.table('campaigns'); ct = v.table('campaign_targets')
    cdesc = dict(zip(cs.campaign, cs.description.astype(str))) if len(cs) else {}
    ct = ct[ct.household_key.isin(hh)]
    ct2 = ct.assign(desc=ct.campaign.map(cdesc).fillna('').astype(str))
    X['n_campaigns'] = ct2.groupby('household_key')['campaign'].nunique().reindex(hh).fillna(0)
    for ty in ['TypeA','TypeB','TypeC']:
        X['ct_'+ty] = ct2[ct2.desc.str.contains(ty)].groupby('household_key').size().reindex(hh).fillna(0)
    r = v.table('coupon_redemptions'); r = r[r.household_key.isin(hh) & (r.day<=sd)]
    X['redemp_all'] = r.groupby('household_key').size().reindex(hh).fillna(0)
    X['redemp_84'] = r[r.day>sd-84].groupby('household_key').size().reindex(hh).fillna(0)
    X['redemp_28'] = r[r.day>sd-28].groupby('household_key').size().reindex(hh).fillna(0)
    rlast = r.groupby('household_key')['day'].max().reindex(hh)
    X['days_since_redemp'] = (sd-rlast).fillna(999).clip(lower=0)
    dm = v.table('display_mailer'); wk = (sd+8)//7
    dm = dm[(dm.week_no<=wk)&(dm.week_no>wk-4)]
    disp_p = set(dm.product_id.unique()) if len(dm) else set()
    t28 = t[t.day>sd-28]
    X['spend28_on_disp'] = t28[t28.product_id.isin(disp_p)].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
    X['share28_on_disp'] = X.spend28_on_disp/(X.spend_28+1e-6)
    d = v.table('demographics')
    d = d.set_index('household_key').reindex(hh)
    for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
        X['dem_'+c] = d[c].astype('category').cat.codes.values
    X['has_demo'] = d['classification_1'].notna().astype(float).values
    X['week_of_year'] = float(wk % 52)
    return X.astype(float)

t0=time.time()
feats = api.build_features(build)
print('build time', round(time.time()-t0,1), 'shape', feats.shape)
print('n features', feats.shape[1]-2)
print('NaNs total', int(feats.isna().sum().sum()))
num = feats.drop(columns=['household_key','snapshot_day']).to_numpy()
print('inf', int(np.isinf(num).sum()))
p = api.save_table(feats, 'feats_v3.parquet'); print('saved', p)


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np, xgboost as xgb

feats = api.load_saved('feats_v3.parquet')
tt = api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = df[df.future_spend_4w.notna()]
va = df[df.snapshot_day>=459]
feats_cols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
Xtr, ytr = tr[feats_cols].to_numpy(np.float32), tr.future_spend_4w.to_numpy(np.float32)
Xva = va[feats_cols].to_numpy(np.float32)
print('train', Xtr.shape, 'val', Xva.shape)

model = xgb.XGBRegressor(
    n_estimators=1600, learning_rate=0.03, max_depth=8, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, reg_alpha=0.5,
    objective='reg:absoluteerror', tree_method='hist', n_jobs=8, random_state=0)
model.fit(Xtr, ytr, verbose=False)
pv = model.predict(Xva)
imp = pd.Series(model.feature_importances_, index=feats_cols).sort_values(ascending=False)
print(imp.head(20).round(4).to_string())

pred = va[['household_key','snapshot_day']].copy()
pred['prediction'] = pv
p = api.save_table(pred, 'pred_e003.parquet')
print('saved', p, len(pred))
