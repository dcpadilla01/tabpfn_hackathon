import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
import xgboost as xgb

tt = A.train_targets()
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count'])
print(g.round(1))
print('overall mean/median/std:', round(tt.future_spend_4w.mean(),2), tt.future_spend_4w.median(), round(tt.future_spend_4w.std(),2))
print('zero frac:', round((tt.future_spend_4w==0).mean(),4))

oof = A.load_saved('oof_e008.parquet')
print('oof cols:', oof.columns.tolist()); print(oof.head(3))

f3 = A.load_saved('feats_v3.parquet')
print('feats_v3 shape:', f3.shape)
tr = f3.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in f3.columns if c not in ('household_key','snapshot_day')]
X = tr[feats].astype(float); X = X.fillna(X.median()); y = tr.future_spend_4w
t0=time.time()
m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.06, max_depth=7, min_child_weight=10,
                     subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=8)
m.fit(X, y)
imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
print(imp.head(30).round(4).to_string())
print('fit time', round(time.time()-t0,1))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

tt = A.train_targets()
oof = A.load_saved('oof_e008.parquet')
d = oof.merge(tt, on=['household_key','snapshot_day'])
print('oof rows', len(d), 'snapshots:', sorted(d.snapshot_day.unique()))
for c in ['oof_sq','oof_med','oof_log']:
    d[f'mae_{c}'] = (d[c]-d.future_spend_4w).abs()
g = d.groupby('snapshot_day').agg(n=('future_spend_4w','size'),
    y_mean=('future_spend_4w','mean'), y_med=('future_spend_4w','median'),
    mae_sq=('mae_oof_sq','mean'), mae_med=('mae_oof_med','mean'), mae_log=('mae_oof_log','mean'))
print(g.round(2))
# optimal per-snapshot multiplicative calibration on OOF (minimize MAE)
for c in ['oof_sq','oof_med','oof_log']:
    facs = {}
    for s, gr in d.groupby('snapshot_day'):
        f = gr.future_spend_4w.values; p = gr[c].values
        fs = np.linspace(0.7,1.3,121)
        maes = [np.abs(f-p*k).mean() for k in fs]
        facs[s] = fs[int(np.argmin(maes))]
    print(c, {k: round(v,3) for k,v in sorted(facs.items())})


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, itertools

tt = A.train_targets()
oof = A.load_saved('oof_e008.parquet').merge(tt, on=['household_key','snapshot_day'])
y = oof.future_spend_4w.values
P = {c: oof[c].values for c in ['oof_sq','oof_med','oof_log']}
print('E006-style blend MAE:', np.abs(0.5*P['oof_sq']+0.5*P['oof_med']-y).mean())
print('med only:', np.abs(P['oof_med']-y).mean())

# grid-search blend weights (step .05, sum=1)
best=(1e9,None)
for a in np.arange(0,1.01,0.05):
    for b in np.arange(0,1.01-a,0.05):
        c = 1-a-b
        p = a*P['oof_sq']+b*P['oof_med']+c*P['oof_log']
        m = np.abs(p-y).mean()
        if m<best[0]: best=(m,(round(a,2),round(b,2),round(c,2)))
print('best 3-blend:', best)

# clip test on med and on best blend
for cap in [150,200,250,300,400,500,1e9]:
    p = np.clip(P['oof_med'],0,cap)
    print('med clip',cap, round(np.abs(p-y).mean(),3))

# per-household shrink: blend pred with household's OOF median spend? need household history; use per-household mean of y? not available at pred time legitimately... skip
# check feats_v3 cols
f3 = A.load_saved('feats_v3.parquet')
print([c for c in f3.columns if 'snap' in c or 'week' in c or 'day' in c])


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
f3 = A.load_saved('feats_v3.parquet')
cols = [c for c in f3.columns if c not in ('household_key','snapshot_day')]
print(len(cols))
for c in cols: print(c)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
v = A.snapshot(459)
print('day', v.day, 'week', v.week)
hh = v.households
print(type(hh)); print(hh.shape); print(hh.head(3))
tx = v.table('transactions')
print('tx shape', tx.shape, 'max day', tx.day.max())
dm = v.table('display_mailer')
print('dm shape', dm.shape); print(dm.head(3)); print(dm.display.value_counts().head()); print(dm.mailer.value_counts().head())
dem = v.table('demographics')
print('dem shape', dem.shape)
for c in ['classification_1','classification_3','classification_4','classification_5','classification_2','homeowner_desc','kid_category_desc']:
    print(c, sorted(dem[c].unique())[:15])
cp = v.table('campaign_targets')
print('ct shape', cp.shape, cp.description.unique())
pr = v.table('products')
print('products shape', pr.shape, pr.columns.tolist())
print(pr.brand.unique()[:10])


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

def probe(view, snapshot_day):
    hh = view.households
    print('day', view.day, 'week', view.week, 'hh type', type(hh), 'n', 0 if hh is None else len(hh))
    if hh is not None: print(hh.head(3))
    tx = view.table('transactions')
    print('tx', tx.shape, 'maxday', tx.day.max())
    dm = view.table('display_mailer')
    print('dm', dm.shape)
    if snapshot_day==95:
        print(dm.display.value_counts().head()); print(dm.mailer.value_counts().head())
    pr = view.table('products')
    print('prod cols', pr.columns.tolist(), 'brand vals', pr.brand.unique()[:8])
    dem = view.table('demographics')
    print('dem', dem.shape)
    for c in ['classification_1','classification_3','classification_4','classification_5']:
        print(c, sorted(dem[c].unique())[:14])
    return hh.iloc[:2].to_frame('k') if hh is not None else pd.DataFrame()

out = A.build_features(probe)
print('out cols', out.columns.tolist(), out.shape)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

def probe(view, snapshot_day):
    hh = view.households
    print('day', view.day, 'week', view.week, 'n_hh', len(hh))
    tx = view.table('transactions')
    print('tx', tx.shape, 'maxday', int(tx.day.max()))
    dm = view.table('display_mailer')
    print('dm', dm.shape, 'weeks', dm.week_no.min(), dm.week_no.max())
    if snapshot_day==95:
        print(dm.display.value_counts()); print(dm.mailer.value_counts())
    pr = view.table('products')
    print('prod cols', pr.columns.tolist(), 'brand vals', list(pr.brand.unique()[:8]))
    dem = view.table('demographics')
    print('dem', dem.shape)
    for c in ['classification_1','classification_3','classification_4','classification_5']:
        print(c, sorted(dem[c].unique())[:14])
    return pd.DataFrame(index=hh[:2])

out = A.build_features(probe)
print('out', out.shape)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

def build(view, snapshot_day):
    hh = pd.Index(view.households)
    tx = view.table('transactions').copy()
    tx = tx[tx.household_key.isin(hh)]
    tx['sp'] = tx.sales_value.astype(float)
    tx['wk'] = ((tx.day + 8) // 7).astype(int)
    tx['hd'] = tx.household_key.astype('category').cat.codes
    pr = view.table('products')
    dept = pr.set_index('product_id').department.astype('category')
    tx['dept'] = pd.Categorical(tx.product_id.map(dept)).codes
    brand = pr.set_index('product_id').brand
    tx['priv'] = (tx.product_id.map(brand) == 'Private').astype(float)
    tx['store'] = tx.store_id.astype('category').cat.codes
    tx['manu'] = tx.product_id.map(pr.set_index('product_id').manufacturer).astype('category').cat.codes
    tx['prod'] = tx.product_id.astype('category').cat.codes
    tx['day'] = tx.day.astype(int)
    tx.sort_values(['hd','day'], inplace=True)
    dem = view.table('demographics')
    dmi = dem.set_index('household_key')

    def smask(w): return tx.sp > 0
    def agg(g, w):
        sp = g.sp; b = g.basket_id
        return pd.Series({
            'spend_%d'%w: sp.sum(), 'baskets_%d'%w: b.nunique(),
            'prods_%d'%w: g.prod.nunique(), 'stores_%d'%w: g.store.nunique(),
            'qty_%d'%w: g.quantity.sum(), 'retail_disc_%d'%w: -g.retail_disc.sum(),
            'coupon_disc_%d'%w: -(g.coupon_disc.sum()+g.coupon_match_disc.sum()),
            'spend_max_basket_%d'%w: g.groupby('basket_id').sp.sum().max() if len(g) else 0.0})
    def block(g, days_since_last, days_since_first):
        o = {}
        for w in [7,14,28,56,84,112,224]:
            g2 = g[g.day > snapshot_day-w]
            if len(g2): o.update(agg(g2, w))
            else: o.update({'spend_%d'%w:0.0,'baskets_%d'%w:0,'prods_%d'%w:0,'stores_%d'%w:0,'qty_%d'%w:0.0,'retail_disc_%d'%w:0.0,'coupon_disc_%d'%w:0.0,'spend_max_basket_%d'%w:0.0})
        o['spend_all']=g.sp.sum(); o['baskets_all']=g.basket_id.nunique(); o['prods_all']=g.prod.nunique()
        o['stores_all']=g.store.nunique(); o['qty_all']=g.quantity.sum()
        o['retail_disc_all']=-g.retail_disc.sum(); o['coupon_disc_all']=-(g.coupon_disc.sum()+g.coupon_match_disc.sum())
        o['spend_max_basket_all']=g.groupby('basket_id').sp.sum().max()
        o['days_since_last']=days_since_last; o['days_since_first']=days_since_first
        s7,s14,s28,s56,s84,s112,s224 = [o['spend_%d'%w] for w in [7,14,28,56,84,112,224]]
        b7,b14,b28,b56,b84,b112,b224 = [o['baskets_%d'%w] for w in [7,14,28,56,84,112,224]]
        o['spend_decay_112'] = (s7*0.5**(0/28)+ (s28-s7)*0.5**(7/28) + (s56-s28)*0.5**(28/28) + (s84-s56)*0.5**(56/28) + (s112-s84)*0.5**(84/28))/0.5**(0/28)
        o['trend_28_56']= s28/max(s56-s28,1e-6); o['trend_56_112']= s56/max(s112-s56,1e-6); o['trend_84_224']= s84/max(s224-s84,1e-6)
        o['share_recent_all']= s28/max(o['spend_all'],1e-6)
        o['avg_basket_84']= s84/max(b84,1e-6); o['avg_basket_28']= s28/max(b28,1e-6)
        o['basket_std_84']= g[g.day>snapshot_day-84].groupby('basket_id').sp.sum().std() if b84>0 else 0.0
        o['basket_min_84']= g[g.day>snapshot_day-84].groupby('basket_id').sp.sum().min() if b84>0 else 0.0
        g84 = g[g.day>snapshot_day-84]
        o['items_per_basket_84']= g84.groupby('basket_id').size().mean() if b84>0 else 0.0
        o['qty_per_basket_84']= g84.groupby('basket_id').quantity.sum().mean() if b84>0 else 0.0
        o['baskets_per_day_84']= b84/84.0
        o['active_28']= int(b28>0); o['active_84']= int(b84>0); o['active_112']= int(b112>0)
        gaps = g.day.diff().dropna()
        gaps = gaps[gaps>0]
        o['gap_mean_112']= gaps[g.day>snapshot_day-112].mean() if len(gaps)>0 else 0.0
        o['gap_std_112']= gaps[g.day>snapshot_day-112].std() if len(gaps)>0 else 0.0
        h = g.trans_time
        o['hour_mean_84']= (h//100).mean(); o['hour_std_84']= (h//100).std(); o['evening_share_84']= ((h//100)>=17).mean()
        for d in ['GROCER','DRUGG','PRODUC','COSMET','NUTRIT','MEAT','MEAT-P','DELI','PASTRY','FLORAL']:
            o['dept_share_%s'%d]= g84[g84.dept==d].sp.sum()/max(s84,1e-6)
        o['dept_other_share_84']= 1.0 - sum(o['dept_share_%s'%d] for d in ['GROCER','DRUGG','PRODUC','COSMET','NUTRIT','MEAT','MEAT-P','DELI','PASTRY','FLORAL'])
        o['n_depts_84']= g84.dept.nunique()
        o['private_share_84']= g84.priv.mean() if len(g84)>0 else 0.0
        o['top_store_share_84']= (g84.groupby('store').sp.sum().max()/s84) if s84>0 else 0.0
        o['spend_rate_all_28']= o['spend_all']/max(days_since_first,1)
        # NEW v5 features
        o['spend_per_trip_28']= s28/max(b28,1e-6); o['spend_per_trip_84']= s84/max(b84,1e-6)
        o['freq_ratio_28_84']= (b28/28.0)/max(b84/84.0,1e-6); o['freq_ratio_7_28']= (b7/7.0)/max(b28/28.0,1e-6)
        o['gap_last_vs_mean']= days_since_last/max(o['gap_mean_112'],1e-6)
        o['gap_z']= (days_since_last-o['gap_mean_112'])/max(o['gap_std_112'],1e-6)
        o['n_prod_84']= g84.prod.nunique(); o['n_manu_84']= g84.manu.nunique()
        o['store_switch_84']= g84.store.nunique(); o['top_store_spend_84']= g84.groupby('store').sp.sum().max()
        o['disc_share_84']= (o['retail_disc_84']+o['coupon_disc_84'])/max(s84,1e-6)
        o['disc_share_28']= (o['retail_disc_28']+o['coupon_disc_28'])/max(s28,1e-6)
        o['avg_unit_price_84']= s84/max(g84.quantity.sum(),1e-6)
        o['avg_unit_price_28']= s28/max(g[g.day>snapshot_day-28].quantity.sum(),1e-6)
        # weekly series (last 12w)
        wkmax = (snapshot_day+8)//7
        ws = g.groupby('wk').sp.sum()
        bs = g.groupby('wk').basket_id.nunique()
        w12 = [wkmax-i for i in range(12)]
        arr = np.array([ws.get(w,0.0) for w in w12]); arrb = np.array([bs.get(w,0) for w in w12])
        o['wk_mean_12']= arr.mean(); o['wk_std_12']= arr.std(); o['wk_max_12']= arr.max(); o['wk_min_12']= arr.min()
        o['wk_cv_12']= arr.std()/max(arr.mean(),1e-6); o['wk_zero_12']= (arr==0).sum()
        o['wk_lag1']= arr[0]; o['wk_lag2']= arr[1]; o['wk_lag3']= arr[2]; o['wk_lag4']= arr[3]
        o['wk_lag1_over_mean']= arr[0]/max(arr.mean(),1e-6)
        o['wk_active_12']= (arr>0).sum(); o['wkb_mean_12']= arrb.mean(); o['wkb_max_12']= arrb.max()
        # same-week-last-year
        o['spend_ly_4w']= ws[(ws.index>=wkmax-53)&(ws.index<=wkmax-50)].sum()
        o['spend_ly_8w']= ws[(ws.index.index if False else ws.index>=wkmax-56)&(ws.index<=wkmax-53)].sum()
        o['spend_ly_12w']= ws[(ws.index>=wkmax-60)&(ws.index<=wkmax-57)].sum()
        o['ratio_ly']= o['spend_ly_4w']/max(o['spend_ly_8w']+o['spend_ly_12w'],1e-6)
        o['active_ly']= int(o['spend_ly_4w']>0)
        # display exposure (last 4w)
        dm = view.table('display_mailer')
        dm4 = dm[(dm.week_no>wkmax-4)&(dm.week_no<=wkmax)]
        if len(dm4):
            disp4 = dm4[dm4.display!=0].drop_duplicates(['product_id','week_no'])
            mail4 = dm4[dm4.mailer!=0].drop_duplicates(['product_id','week_no'])
            disp_p = set(disp4.product_id.unique()); mail_p = set(mail4.product_id.unique())
            tx4 = tx[tx.day>snapshot_day-28]
            o['spend28_on_disp']= tx4[tx4.product_id.isin(disp_p)].sp.sum()
            o['spend28_on_mail']= tx4[tx4.product_id.isin(mail_p)].sp.sum()
            o['share28_on_disp']= o['spend28_on_disp']/max(s28,1e-6)
            o['share28_on_mail']= o['spend28_on_mail']/max(s28,1e-6)
        else:
            o['spend28_on_disp']=0.0; o['spend28_on_mail']=0.0; o['share28_on_disp']=0.0; o['share28_on_mail']=0.0
        # interactions
        o['ix_spend_x_freq']= s84*o['baskets_per_day_84']; o['ix_decay_x_trend']= o['spend_decay_112']*o['trend_28_56']
        o['ix_decay_x_freq']= o['spend_decay_112']*o['freq_ratio_28_84']
        o['log_spend_84']= np.log1p(s84); o['log_spend_28']= np.log1p(s28)
        o['sqrt_spend_84']= np.sqrt(s84); o['sqrt_spend_28']= np.sqrt(s28)
        return pd.Series(o)

    rows = {}
    for k in hh:
        g = tx[tx.hd == k] if False else tx.xs(k, level='hd') if False else tx[tx.hd == k]
        rows[k] = block(g, snapshot_day - (g.day.max() if len(g) else 0), snapshot_day - (g.day.min() if len(g) else 0))
    F = pd.DataFrame(rows).T
    # demographics
    for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
        F['dem_'+c]= F.index.map(dmi[c]).astype('category').cat.codes
    F['has_demo']= F.index.isin(dmi.index).astype(int)
    F['week_of_year']= ((snapshot_day+8)//7)%52
    F['snapshot_day']= snapshot_day
    F.index.name='household_key'
    return F.astype(float)

t0=time.time()
out = A.build_features(build)
print('shape', out.shape, 'time', round(time.time()-t0,1))
print(out.head(3).T.head(30))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')

def build(view, snapshot_day):
    hh = pd.Index(view.households)
    tx = view.table('transactions').copy()
    tx = tx[tx.household_key.isin(hh)]
    tx['sp'] = tx.sales_value.astype(float)
    tx['wk'] = ((tx.day + 8) // 7).astype(int)
    tx['hd'] = tx.household_key.astype('category').cat.codes
    pr = view.table('products')
    dept = pr.set_index('product_id').department.astype('category')
    tx['dept'] = pd.Categorical(tx.product_id.map(dept)).codes
    brand = pr.set_index('product_id').brand
    tx['priv'] = (tx.product_id.map(brand) == 'Private').astype(float)
    tx['store'] = tx.store_id.astype('category').cat.codes
    tx['manu'] = tx.product_id.map(pr.set_index('product_id').manufacturer).astype('category').cat.codes
    tx['prod'] = tx.product_id.astype('category').cat.codes
    tx['day'] = tx.day.astype(int)
    tx.sort_values(['hd','day'], inplace=True)
    dem = view.table('demographics')
    dmi = dem.set_index('household_key')

    def agg(g, w):
        sp = g.sp
        return pd.Series({
            'spend_%d'%w: sp.sum(), 'baskets_%d'%w: g.basket_id.nunique(),
            'prods_%d'%w: g.prod.nunique(), 'stores_%d'%w: g.store.nunique(),
            'qty_%d'%w: g.quantity.sum(), 'retail_disc_%d'%w: -g.retail_disc.sum(),
            'coupon_disc_%d'%w: -(g.coupon_disc.sum()+g.coupon_match_disc.sum()),
            'spend_max_basket_%d'%w: g.groupby('basket_id').sp.sum().max() if len(g) else 0.0})
    Z = {'spend_%d'%w:0.0,'baskets_%d'%w:0,'prods_%d'%w:0,'stores_%d'%w:0,'qty_%d'%w:0.0,'retail_disc_%d'%w:0.0,'coupon_disc_%d'%w:0.0,'spend_max_basket_%d'%w:0.0}
    def block(g, days_since_last, days_since_first):
        o = {}
        for w in [7,14,28,56,84,112,224]:
            g2 = g[g.day > snapshot_day-w]
            o.update(agg(g2, w) if len(g2) else Z)
        o['spend_all']=g.sp.sum(); o['baskets_all']=g.basket_id.nunique(); o['prods_all']=g.prod.nunique()
        o['stores_all']=g.store.nunique(); o['qty_all']=g.quantity.sum()
        o['retail_disc_all']=-g.retail_disc.sum(); o['coupon_disc_all']=-(g.coupon_disc.sum()+g.coupon_match_disc.sum())
        o['spend_max_basket_all']=g.groupby('basket_id').sp.sum().max()
        o['days_since_last']=days_since_last; o['days_since_first']=days_since_first
        s7,s14,s28,s56,s84,s112,s224 = [o['spend_%d'%w] for w in [7,14,28,56,84,112,224]]
        b7,b14,b28,b56,b84,b112,b224 = [o['baskets_%d'%w] for w in [7,14,28,56,84,112,224]]
        o['spend_decay_112'] = s7*0.5**(0/28) + (s28-s7)*0.5**(7/28) + (s56-s28)*0.5**(28/28) + (s84-s56)*0.5**(56/28) + (s112-s84)*0.5**(84/28)
        o['trend_28_56']= s28/max(s56-s28,1e-6); o['trend_56_112']= s56/max(s112-s56,1e-6); o['trend_84_224']= s84/max(s224-s84,1e-6)
        o['share_recent_all']= s28/max(o['spend_all'],1e-6)
        o['avg_basket_84']= s84/max(b84,1e-6); o['avg_basket_28']= s28/max(b28,1e-6)
        g84 = g[g.day>snapshot_day-84]
        o['basket_std_84']= g84.groupby('basket_id').sp.sum().std() if b84>0 else 0.0
        o['basket_min_84']= g84.groupby('basket_id').sp.sum().min() if b84>0 else 0.0
        o['items_per_basket_84']= g84.groupby('basket_id').size().mean() if b84>0 else 0.0
        o['qty_per_basket_84']= g84.groupby('basket_id').quantity.sum().mean() if b84>0 else 0.0
        o['baskets_per_day_84']= b84/84.0
        o['active_28']= int(b28>0); o['active_84']= int(b84>0); o['active_112']= int(b112>0)
        gaps = g.day.diff().dropna(); gaps = gaps[gaps>0]
        g112 = gaps[g.day[gaps.index]>snapshot_day-112] if len(gaps) else gaps
        o['gap_mean_112']= g112.mean() if len(g112)>0 else 0.0
        o['gap_std_112']= g112.std() if len(g112)>0 else 0.0
        h = g.trans_time
        o['hour_mean_84']= (h//100).mean(); o['hour_std_84']= (h//100).std(); o['evening_share_84']= ((h//100)>=17).mean()
        for d in ['GROCER','DRUGG','PRODUC','COSMET','NUTRIT','MEAT','MEAT-P','DELI','PASTRY','FLORAL']:
            o['dept_share_%s'%d]= g84[g84.dept==d].sp.sum()/max(s84,1e-6)
        o['dept_other_share_84']= 1.0 - sum(o['dept_share_%s'%d] for d in ['GROCER','DRUGG','PRODUC','COSMET','NUTRIT','MEAT','MEAT-P','DELI','PASTRY','FLORAL'])
        o['n_depts_84']= g84.dept.nunique()
        o['private_share_84']= g84.priv.mean() if len(g84)>0 else 0.0
        o['top_store_share_84']= (g84.groupby('store').sp.sum().max()/s84) if s84>0 else 0.0
        o['spend_rate_all_28']= o['spend_all']/max(days_since_first,1)
        o['spend_per_trip_28']= s28/max(b28,1e-6); o['spend_per_trip_84']= s84/max(b84,1e-6)
        o['freq_ratio_28_84']= (b28/28.0)/max(b84/84.0,1e-6); o['freq_ratio_7_28']= (b7/7.0)/max(b28/28.0,1e-6)
        o['gap_last_vs_mean']= days_since_last/max(o['gap_mean_112'],1e-6)
        o['gap_z']= (days_since_last-o['gap_mean_112'])/max(o['gap_std_112'],1e-6)
        o['n_prod_84']= g84.prod.nunique(); o['n_manu_84']= g84.manu.nunique()
        o['store_switch_84']= g84.store.nunique(); o['top_store_spend_84']= g84.groupby('store').sp.sum().max()
        o['disc_share_84']= (o['retail_disc_84']+o['coupon_disc_84'])/max(s84,1e-6)
        o['disc_share_28']= (o['retail_disc_28']+o['coupon_disc_28'])/max(s28,1e-6)
        o['avg_unit_price_84']= s84/max(g84.quantity.sum(),1e-6)
        o['avg_unit_price_28']= s28/max(g[g.day>snapshot_day-28].quantity.sum(),1e-6)
        wkmax = (snapshot_day+8)//7
        ws = g.groupby('wk').sp.sum(); bs = g.groupby('wk').basket_id.nunique()
        w12 = [wkmax-i for i in range(12)]
        arr = np.array([ws.get(w,0.0) for w in w12]); arrb = np.array([bs.get(w,0) for w in w12])
        o['wk_mean_12']= arr.mean(); o['wk_std_12']= arr.std(); o['wk_max_12']= arr.max(); o['wk_min_12']= arr.min()
        o['wk_cv_12']= arr.std()/max(arr.mean(),1e-6); o['wk_zero_12']= (arr==0).sum()
        o['wk_lag1']= arr[0]; o['wk_lag2']= arr[1]; o['wk_lag3']= arr[2]; o['wk_lag4']= arr[3]
        o['wk_lag1_over_mean']= arr[0]/max(arr.mean(),1e-6)
        o['wk_active_12']= (arr>0).sum(); o['wkb_mean_12']= arrb.mean(); o['wkb_max_12']= arrb.max()
        o['spend_ly_4w']= ws[(ws.index>=wkmax-53)&(ws.index<=wkmax-50)].sum()
        o['spend_ly_8w']= ws[(ws.index>=wkmax-56)&(ws.index<=wkmax-53)].sum()
        o['spend_ly_12w']= ws[(ws.index>=wkmax-60)&(ws.index<=wkmax-57)].sum()
        o['ratio_ly']= o['spend_ly_4w']/max(o['spend_ly_8w']+o['spend_ly_12w'],1e-6)
        o['active_ly']= int(o['spend_ly_4w']>0)
        dm = view.table('display_mailer')
        dm4 = dm[(dm.week_no>wkmax-4)&(dm.week_no<=wkmax)]
        if len(dm4):
            disp_p = set(dm4[dm4.display!=0].product_id.unique()); mail_p = set(dm4[dm4.mailer!=0].product_id.unique())
            tx4 = tx[tx.day>snapshot_day-28]
            o['spend28_on_disp']= tx4[tx4.product_id.isin(disp_p)].sp.sum()
            o['spend28_on_mail']= tx4[tx4.product_id.isin(mail_p)].sp.sum()
            o['share28_on_disp']= o['spend28_on_disp']/max(s28,1e-6)
            o['share28_on_mail']= o['spend28_on_mail']/max(s28,1e-6)
        else:
            o['spend28_on_disp']=0.0; o['spend28_on_mail']=0.0; o['share28_on_disp']=0.0; o['share28_on_mail']=0.0
        o['ix_spend_x_freq']= s84*o['baskets_per_day_84']; o['ix_decay_x_trend']= o['spend_decay_112']*o['trend_28_56']
        o['ix_decay_x_freq']= o['spend_decay_112']*o['freq_ratio_28_84']
        o['log_spend_84']= np.log1p(s84); o['log_spend_28']= np.log1p(s28)
        o['sqrt_spend_84']= np.sqrt(s84); o['sqrt_spend_28']= np.sqrt(s28)
        return pd.Series(o)

    rows = {}
    for k in hh:
        g = tx[tx.hd == k]
        rows[k] = block(g, snapshot_day - (g.day.max() if len(g) else 0), snapshot_day - (g.day.min() if len(g) else 0))
    F = pd.DataFrame(rows).T
    for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
        F['dem_'+c]= F.index.map(dmi[c]).astype('category').cat.codes
    F['has_demo']= F.index.isin(dmi.index).astype(int)
    F['week_of_year']= ((snapshot_day+8)//7)%52
    F['snapshot_day']= snapshot_day
    F.index.name='household_key'
    return F.astype(float)

t0=time.time()
out = A.build_features(build)
print('shape', out.shape, 'time', round(time.time()-t0,1))
p = A.save_table(out, 'feats_v5.parquet')
print('saved', p)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')

def build(view, snapshot_day):
    hh = pd.Index(view.households)
    tx = view.table('transactions').copy()
    tx = tx[tx.household_key.isin(hh)]
    tx['sp'] = tx.sales_value.astype(float)
    tx['wk'] = ((tx.day + 8) // 7).astype(int)
    tx['hd'] = tx.household_key.astype('category').cat.codes
    pr = view.table('products')
    dept = pr.set_index('product_id').department.astype('category')
    tx['dept'] = pd.Categorical(tx.product_id.map(dept)).codes
    brand = pr.set_index('product_id').brand
    tx['priv'] = (tx.product_id.map(brand) == 'Private').astype(float)
    tx['store'] = tx.store_id.astype('category').cat.codes
    tx['manu'] = tx.product_id.map(pr.set_index('product_id').manufacturer).astype('category').cat.codes
    tx['prod'] = tx.product_id.astype('category').cat.codes
    tx['day'] = tx.day.astype(int)
    tx.sort_values(['hd','day'], inplace=True)
    dem = view.table('demographics')
    dmi = dem.set_index('household_key')
    Zs = {}
    for w in [7,14,28,56,84,112,224]:
        Zs[w] = {'spend_%d'%w:0.0,'baskets_%d'%w:0,'prods_%d'%w:0,'stores_%d'%w:0,'qty_%d'%w:0.0,
                 'retail_disc_%d'%w:0.0,'coupon_disc_%d'%w:0.0,'spend_max_basket_%d'%w:0.0}

    def agg(g, w):
        return pd.Series({
            'spend_%d'%w: g.sp.sum(), 'baskets_%d'%w: g.basket_id.nunique(),
            'prods_%d'%w: g.prod.nunique(), 'stores_%d'%w: g.store.nunique(),
            'qty_%d'%w: g.quantity.sum(), 'retail_disc_%d'%w: -g.retail_disc.sum(),
            'coupon_disc_%d'%w: -(g.coupon_disc.sum()+g.coupon_match_disc.sum()),
            'spend_max_basket_%d'%w: g.groupby('basket_id').sp.sum().max() if len(g) else 0.0})

    def block(g, days_since_last, days_since_first):
        o = {}
        for w in [7,14,28,56,84,112,224]:
            g2 = g[g.day > snapshot_day-w]
            o.update(agg(g2, w) if len(g2) else Zs[w])
        o['spend_all']=g.sp.sum(); o['baskets_all']=g.basket_id.nunique(); o['prods_all']=g.prod.nunique()
        o['stores_all']=g.store.nunique(); o['qty_all']=g.quantity.sum()
        o['retail_disc_all']=-g.retail_disc.sum(); o['coupon_disc_all']=-(g.coupon_disc.sum()+g.coupon_match_disc.sum())
        o['spend_max_basket_all']=g.groupby('basket_id').sp.sum().max()
        o['days_since_last']=days_since_last; o['days_since_first']=days_since_first
        s7,s14,s28,s56,s84,s112,s224 = [o['spend_%d'%w] for w in [7,14,28,56,84,112,224]]
        b7,b14,b28,b56,b84,b112,b224 = [o['baskets_%d'%w] for w in [7,14,28,56,84,112,224]]
        o['spend_decay_112'] = s7*0.5**(0/28) + (s28-s7)*0.5**(7/28) + (s56-s28)*0.5**(28/28) + (s84-s56)*0.5**(56/28) + (s112-s84)*0.5**(84/28)
        o['trend_28_56']= s28/max(s56-s28,1e-6); o['trend_56_112']= s56/max(s112-s56,1e-6); o['trend_84_224']= s84/max(s224-s84,1e-6)
        o['share_recent_all']= s28/max(o['spend_all'],1e-6)
        o['avg_basket_84']= s84/max(b84,1e-6); o['avg_basket_28']= s28/max(b28,1e-6)
        g84 = g[g.day>snapshot_day-84]
        o['basket_std_84']= g84.groupby('basket_id').sp.sum().std() if b84>0 else 0.0
        o['basket_min_84']= g84.groupby('basket_id').sp.sum().min() if b84>0 else 0.0
        o['items_per_basket_84']= g84.groupby('basket_id').size().mean() if b84>0 else 0.0
        o['qty_per_basket_84']= g84.groupby('basket_id').quantity.sum().mean() if b84>0 else 0.0
        o['baskets_per_day_84']= b84/84.0
        o['active_28']= int(b28>0); o['active_84']= int(b84>0); o['active_112']= int(b112>0)
        gaps = g.day.diff().dropna(); gaps = gaps[gaps>0]
        g112 = gaps[g.day[gaps.index]>snapshot_day-112] if len(gaps) else gaps
        o['gap_mean_112']= g112.mean() if len(g112)>0 else 0.0
        o['gap_std_112']= g112.std() if len(g112)>0 else 0.0
        h = g.trans_time
        o['hour_mean_84']= (h//100).mean(); o['hour_std_84']= (h//100).std(); o['evening_share_84']= ((h//100)>=17).mean()
        for d in ['GROCER','DRUGG','PRODUC','COSMET','NUTRIT','MEAT','MEAT-P','DELI','PASTRY','FLORAL']:
            o['dept_share_%s'%d]= g84[g84.dept==d].sp.sum()/max(s84,1e-6)
        o['dept_other_share_84']= 1.0 - sum(o['dept_share_%s'%d] for d in ['GROCER','DRUGG','PRODUC','COSMET','NUTRIT','MEAT','MEAT-P','DELI','PASTRY','FLORAL'])
        o['n_depts_84']= g84.dept.nunique()
        o['private_share_84']= g84.priv.mean() if len(g84)>0 else 0.0
        o['top_store_share_84']= (g84.groupby('store').sp.sum().max()/s84) if s84>0 else 0.0
        o['spend_rate_all_28']= o['spend_all']/max(days_since_first,1)
        o['spend_per_trip_28']= s28/max(b28,1e-6); o['spend_per_trip_84']= s84/max(b84,1e-6)
        o['freq_ratio_28_84']= (b28/28.0)/max(b84/84.0,1e-6); o['freq_ratio_7_28']= (b7/7.0)/max(b28/28.0,1e-6)
        o['gap_last_vs_mean']= days_since_last/max(o['gap_mean_112'],1e-6)
        o['gap_z']= (days_since_last-o['gap_mean_112'])/max(o['gap_std_112'],1e-6)
        o['n_prod_84']= g84.prod.nunique(); o['n_manu_84']= g84.manu.nunique()
        o['store_switch_84']= g84.store.nunique(); o['top_store_spend_84']= g84.groupby('store').sp.sum().max()
        o['disc_share_84']= (o['retail_disc_84']+o['coupon_disc_84'])/max(s84,1e-6)
        o['disc_share_28']= (o['retail_disc_28']+o['coupon_disc_28'])/max(s28,1e-6)
        o['avg_unit_price_84']= s84/max(g84.quantity.sum(),1e-6)
        o['avg_unit_price_28']= s28/max(g[g.day>snapshot_day-28].quantity.sum(),1e-6)
        wkmax = (snapshot_day+8)//7
        ws = g.groupby('wk').sp.sum(); bs = g.groupby('wk').basket_id.nunique()
        w12 = [wkmax-i for i in range(12)]
        arr = np.array([ws.get(w,0.0) for w in w12]); arrb = np.array([bs.get(w,0) for w in w12])
        o['wk_mean_12']= arr.mean(); o['wk_std_12']= arr.std(); o['wk_max_12']= arr.max(); o['wk_min_12']= arr.min()
        o['wk_cv_12']= arr.std()/max(arr.mean(),1e-6); o['wk_zero_12']= (arr==0).sum()
        o['wk_lag1']= arr[0]; o['wk_lag2']= arr[1]; o['wk_lag3']= arr[2]; o['wk_lag4']= arr[3]
        o['wk_lag1_over_mean']= arr[0]/max(arr.mean(),1e-6)
        o['wk_active_12']= (arr>0).sum(); o['wkb_mean_12']= arrb.mean(); o['wkb_max_12']= arrb.max()
        o['spend_ly_4w']= ws[(ws.index>=wkmax-53)&(ws.index<=wkmax-50)].sum()
        o['spend_ly_8w']= ws[(ws.index>=wkmax-56)&(ws.index<=wkmax-53)].sum()
        o['spend_ly_12w']= ws[(ws.index>=wkmax-60)&(ws.index<=wkmax-57)].sum()
        o['ratio_ly']= o['spend_ly_4w']/max(o['spend_ly_8w']+o['spend_ly_12w'],1e-6)
        o['active_ly']= int(o['spend_ly_4w']>0)
        dm = view.table('display_mailer')
        dm4 = dm[(dm.week_no>wkmax-4)&(dm.week_no<=wkmax)]
        if len(dm4):
            disp_p = set(dm4[dm4.display!=0].product_id.unique()); mail_p = set(dm4[dm4.mailer!=0].product_id.unique())
            tx4 = tx[tx.day>snapshot_day-28]
            o['spend28_on_disp']= tx4[tx4.product_id.isin(disp_p)].sp.sum()
            o['spend28_on_mail']= tx4[tx4.product_id.isin(mail_p)].sp.sum()
            o['share28_on_disp']= o['spend28_on_disp']/max(s28,1e-6)
            o['share28_on_mail']= o['spend28_on_mail']/max(s28,1e-6)
        else:
            o['spend28_on_disp']=0.0; o['spend28_on_mail']=0.0; o['share28_on_disp']=0.0; o['share28_on_mail']=0.0
        o['ix_spend_x_freq']= s84*o['baskets_per_day_84']; o['ix_decay_x_trend']= o['spend_decay_112']*o['trend_28_56']
        o['ix_decay_x_freq']= o['spend_decay_112']*o['freq_ratio_28_84']
        o['log_spend_84']= np.log1p(s84); o['log_spend_28']= np.log1p(s28)
        o['sqrt_spend_84']= np.sqrt(s84); o['sqrt_spend_28']= np.sqrt(s28)
        return pd.Series(o)

    rows = {}
    for k in hh:
        g = tx[tx.hd == k]
        rows[k] = block(g, snapshot_day - (g.day.max() if len(g) else 0), snapshot_day - (g.day.min() if len(g) else 0))
    F = pd.DataFrame(rows).T
    for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
        F['dem_'+c]= F.index.map(dmi[c]).astype('category').cat.codes
    F['has_demo']= F.index.isin(dmi.index).astype(int)
    F['week_of_year']= ((snapshot_day+8)//7)%52
    F['snapshot_day']= snapshot_day
    F.index.name='household_key'
    return F.astype(float)

t0=time.time()
out = A.build_features(build)
print('shape', out.shape, 'time', round(time.time()-t0,1))
p = A.save_table(out, 'feats_v5.parquet')
print('saved', p)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

def build(view, snapshot_day):
    hh = pd.Index(view.households)
    tx = view.table('transactions').copy()
    tx = tx[tx.household_key.isin(hh)]
    tx['sp'] = tx.sales_value.astype(float)
    tx['wk'] = ((tx.day + 8) // 7).astype(int)
    tx['hour'] = tx.trans_time.astype(float)//100
    pr = view.table('products')
    pinfo = pr.set_index('product_id')
    tx['dept'] = tx.product_id.map(pinfo.department)
    tx['priv'] = (tx.product_id.map(pinfo.brand) == 'Private').astype(float)
    tx['store'] = tx.store_id.astype('category').cat.codes
    tx['manu'] = tx.product_id.map(pinfo.manufacturer).astype('category').cat.codes
    tx['prod'] = tx.product_id.astype('category').cat.codes
    tx['day'] = tx.day.astype(int)
    tx.sort_values(['household_key','day'], inplace=True)
    dem = view.table('demographics'); dmi = dem.set_index('household_key')
    wkmax = (snapshot_day+8)//7
    dm = view.table('display_mailer')
    dm4 = dm[(dm.week_no>wkmax-4)&(dm.week_no<=wkmax)]
    disp_p = set(dm4[dm4.display!=0].product_id.unique()); mail_p = set(dm4[dm4.mailer!=0].product_id.unique())
    Zs = {w: {'spend_%d'%w:0.0,'baskets_%d'%w:0,'prods_%d'%w:0,'stores_%d'%w:0,'qty_%d'%w:0.0,
              'retail_disc_%d'%w:0.0,'coupon_disc_%d'%w:0.0,'spend_max_basket_%d'%w:0.0} for w in [7,14,28,56,84,112,224]}
    DEPTS = ['GROCER','DRUGG','PRODUC','COSMET','NUTRIT','MEAT','MEAT-P','DELI','PASTRY','FLORAL']

    def block(g, dsl, dsf):
        o = {}
        for w in [7,14,28,56,84,112,224]:
            g2 = g[g.day > snapshot_day-w]
            if len(g2):
                o['spend_%d'%w]=g2.sp.sum(); o['baskets_%d'%w]=g2.basket_id.nunique()
                o['prods_%d'%w]=g2['prod'].nunique(); o['stores_%d'%w]=g2.store.nunique()
                o['qty_%d'%w]=g2.quantity.sum(); o['retail_disc_%d'%w]=-g2.retail_disc.sum()
                o['coupon_disc_%d'%w]=-(g2.coupon_disc.sum()+g2.coupon_match_disc.sum())
                o['spend_max_basket_%d'%w]=g2.groupby('basket_id').sp.sum().max()
            else: o.update(Zs[w])
        o['spend_all']=g.sp.sum(); o['baskets_all']=g.basket_id.nunique(); o['prods_all']=g['prod'].nunique()
        o['stores_all']=g.store.nunique(); o['qty_all']=g.quantity.sum()
        o['retail_disc_all']=-g.retail_disc.sum(); o['coupon_disc_all']=-(g.coupon_disc.sum()+g.coupon_match_disc.sum())
        o['spend_max_basket_all']=g.groupby('basket_id').sp.sum().max()
        o['days_since_last']=dsl; o['days_since_first']=dsf
        s7,s14,s28,s56,s84,s112,s224 = [o['spend_%d'%w] for w in [7,14,28,56,84,112,224]]
        b7,b14,b28,b56,b84,b112,b224 = [o['baskets_%d'%w] for w in [7,14,28,56,84,112,224]]
        o['spend_decay_112'] = s7 + (s28-s7)*0.5**(7/28) + (s56-s28)*0.5**(28/28) + (s84-s56)*0.5**(56/28) + (s112-s84)*0.5**(84/28)
        o['trend_28_56']= s28/max(s56-s28,1e-6); o['trend_56_112']= s56/max(s112-s56,1e-6); o['trend_84_224']= s84/max(s224-s84,1e-6)
        o['share_recent_all']= s28/max(o['spend_all'],1e-6)
        o['avg_basket_84']= s84/max(b84,1e-6); o['avg_basket_28']= s28/max(b28,1e-6)
        g84 = g[g.day>snapshot_day-84]; g28 = g[g.day>snapshot_day-28]
        o['basket_std_84']= g84.groupby('basket_id').sp.sum().std() if b84>1 else 0.0
        o['basket_min_84']= g84.groupby('basket_id').sp.sum().min() if b84>0 else 0.0
        o['items_per_basket_84']= g84.groupby('basket_id').size().mean() if b84>0 else 0.0
        o['qty_per_basket_84']= g84.groupby('basket_id').quantity.sum().mean() if b84>0 else 0.0
        o['baskets_per_day_84']= b84/84.0
        o['active_28']= int(b28>0); o['active_84']= int(b84>0); o['active_112']= int(b112>0)
        d = g.day.values; gaps = np.diff(d); gaps = gaps[gaps>0]
        aft = d[1:][np.diff(d)>0] if len(d)>1 else np.array([])
        gp = gaps[aft>snapshot_day-112] if len(gaps) else gaps
        o['gap_mean_112']= gp.mean() if len(gp)>0 else 0.0
        o['gap_std_112']= gp.std() if len(gp)>1 else 0.0
        h = g.hour
        o['hour_mean_84']= h.mean() if len(h)>0 else 0.0; o['hour_std_84']= h.std() if len(h)>1 else 0.0
        o['evening_share_84']= (h>=17).mean() if len(h)>0 else 0.0
        for dd in DEPTS:
            o['dept_share_%s'%dd]= g84[g84.dept==dd].sp.sum()/max(s84,1e-6)
        o['dept_other_share_84']= 1.0 - sum(o['dept_share_%s'%dd] for dd in DEPTS)
        o['n_depts_84']= g84.dept.nunique()
        o['private_share_84']= g84.priv.mean() if len(g84)>0 else 0.0
        o['top_store_share_84']= (g84.groupby('store').sp.sum().max()/s84) if s84>0 else 0.0
        o['spend_rate_all_28']= o['spend_all']/max(dsf,1)
        o['spend_per_trip_28']= s28/max(b28,1e-6); o['spend_per_trip_84']= s84/max(b84,1e-6)
        o['freq_ratio_28_84']= (b28/28.0)/max(b84/84.0,1e-6); o['freq_ratio_7_28']= (b7/7.0)/max(b28/28.0,1e-6)
        o['gap_last_vs_mean']= dsl/max(o['gap_mean_112'],1e-6)
        o['gap_z']= (dsl-o['gap_mean_112'])/max(o['gap_std_112'],1e-6)
        o['n_prod_84']= g84['prod'].nunique(); o['n_manu_84']= g84.manu.nunique()
        o['store_switch_84']= g84.store.nunique(); o['top_store_spend_84']= g84.groupby('store').sp.sum().max()
        o['disc_share_84']= (o['retail_disc_84']+o['coupon_disc_84'])/max(s84,1e-6)
        o['disc_share_28']= (o['retail_disc_28']+o['coupon_disc_28'])/max(s28,1e-6)
        o['avg_unit_price_84']= s84/max(g84.quantity.sum(),1e-6)
        o['avg_unit_price_28']= s28/max(g28.quantity.sum(),1e-6)
        ws = g.groupby('wk').sp.sum(); bs = g.groupby('wk').basket_id.nunique()
        w12 = [wkmax-i for i in range(12)]
        arr = np.array([ws.get(w,0.0) for w in w12]); arrb = np.array([bs.get(w,0) for w in w12])
        o['wk_mean_12']= arr.mean(); o['wk_std_12']= arr.std(); o['wk_max_12']= arr.max(); o['wk_min_12']= arr.min()
        o['wk_cv_12']= arr.std()/max(arr.mean(),1e-6); o['wk_zero_12']= float((arr==0).sum())
        o['wk_lag1']= arr[0]; o['wk_lag2']= arr[1]; o['wk_lag3']= arr[2]; o['wk_lag4']= arr[3]
        o['wk_lag1_over_mean']= arr[0]/max(arr.mean(),1e-6)
        o['wk_active_12']= float((arr>0).sum()); o['wkb_mean_12']= arrb.mean(); o['wkb_max_12']= arrb.max()
        o['spend_ly_4w']= ws[(ws.index>=wkmax-53)&(ws.index<=wkmax-50)].sum()
        o['spend_ly_8w']= ws[(ws.index>=wkmax-56)&(ws.index<=wkmax-53)].sum()
        o['spend_ly_12w']= ws[(ws.index>=wkmax-60)&(ws.index<=wkmax-57)].sum()
        o['ratio_ly']= o['spend_ly_4w']/max(o['spend_ly_8w']+o['spend_ly_12w'],1e-6)
        o['active_ly']= float(o['spend_ly_4w']>0)
        o['spend28_on_disp']= g28[g28.product_id.isin(disp_p)].sp.sum()
        o['spend28_on_mail']= g28[g28.product_id.isin(mail_p)].sp.sum()
        o['share28_on_disp']= o['spend28_on_disp']/max(s28,1e-6)
        o['share28_on_mail']= o['spend28_on_mail']/max(s28,1e-6)
        o['ix_spend_x_freq']= s84*o['baskets_per_day_84']; o['ix_decay_x_trend']= o['spend_decay_112']*o['trend_28_56']
        o['ix_decay_x_freq']= o['spend_decay_112']*o['freq_ratio_28_84']
        o['log_spend_84']= np.log1p(s84); o['log_spend_28']= np.log1p(s28)
        o['sqrt_spend_84']= np.sqrt(s84); o['sqrt_spend_28']= np.sqrt(s28)
        return pd.Series(o)

    rows = {}
    for k, g in tx.groupby('household_key'):
        rows[k] = block(g, snapshot_day - int(g.day.max()), snapshot_day - int(g.day.min()))
    for k in hh:
        if k not in rows: rows[k] = block(tx.iloc[0:0], 999, 999)
    F = pd.DataFrame(rows).T
    for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
        F['dem_'+c]= F.index.map(dmi[c]).astype('category').cat.codes
    F['has_demo']= F.index.isin(dmi.index).astype(float)
    F['week_of_year']= float(((snapshot_day+8)//7)%52)
    F.index.name='household_key'
    return F.astype(float)

t0=time.time()
try:
    out = A.build_features(build)
    print('built', out.shape, round(time.time()-t0,1),'s'); src='v5'
    A.save_table(out, 'feats_v5.parquet')
except Exception as e:
    print('BUILD FAILED:', type(e).__name__, e); src='v3'
    out = A.load_saved('feats_v3.parquet')

tt = A.train_targets()
tr = out.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats = [c for c in out.columns if c not in ('household_key','snapshot_day')]
Xtr = tr[feats].astype(float); med = Xtr.median(); Xtr = Xtr.fillna(med); y = tr.future_spend_4w.values
val = out[out.snapshot_day.isin(A.snapshot_days()['validation'])].copy()
Xv = val[feats].astype(float).fillna(med)
print('train', Xtr.shape, 'val', Xv.shape, 'src', src)
P = np.zeros(len(Xv))
t1=time.time()
m1=[]
for sd in [7,17]:
    m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=0.5,
        tree_method='hist', n_jobs=8, random_state=sd); m.fit(Xtr,y); m1.append(m); P += 0.2*m.predict(Xv)
m2 = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.02, max_depth=7, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=8, random_state=7); m2.fit(Xtr,y); P += 0.4*m2.predict(Xv)
for sd in [7,17]:
    h = HistGradientBoostingRegressor(loss='quantile', quantile=0.5, max_iter=400, learning_rate=0.06,
        max_leaf_nodes=31, l2_regularization=1.0, random_state=sd); h.fit(Xtr,y); P += 0.1*h.predict(Xv)
print('train time', round(time.time()-t1,1))
val['prediction']=P
p = A.save_table(val[['household_key','snapshot_day','prediction']], 'pred_e010.parquet')
print('saved', p, 'pred mean', round(P.mean(),2), 'frac<=0', round((P<=0).mean(),4))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

def build(view, snapshot_day):
    hh = pd.Index(view.households)
    tx = view.table('transactions').copy()
    tx = tx[tx.household_key.isin(hh)]
    tx['sp'] = tx.sales_value.astype(float)
    tx['wk'] = ((tx.day + 8) // 7).astype(int)
    tx['hour'] = tx.trans_time.astype(float)//100
    pr = view.table('products')
    pinfo = pr.set_index('product_id')
    tx['dept'] = tx.product_id.map(pinfo.department)
    tx['priv'] = (tx.product_id.map(pinfo.brand) == 'Private').astype(float)
    tx['store'] = pd.Categorical(tx.store_id).codes
    tx['manu'] = pd.Categorical(tx.product_id.map(pinfo.manufacturer)).codes
    tx['prod'] = pd.Categorical(tx.product_id).codes
    tx['day'] = tx.day.astype(int)
    tx.sort_values(['household_key','day'], inplace=True)
    dem = view.table('demographics'); dmi = dem.set_index('household_key')
    wkmax = (snapshot_day+8)//7
    dm = view.table('display_mailer')
    dm4 = dm[(dm.week_no>wkmax-4)&(dm.week_no<=wkmax)]
    disp_p = set(dm4[dm4.display!=0].product_id.unique()); mail_p = set(dm4[dm4.mailer!=0].product_id.unique())
    Zs = {w: {'spend_%d'%w:0.0,'baskets_%d'%w:0,'prods_%d'%w:0,'stores_%d'%w:0,'qty_%d'%w:0.0,
              'retail_disc_%d'%w:0.0,'coupon_disc_%d'%w:0.0,'spend_max_basket_%d'%w:0.0} for w in [7,14,28,56,84,112,224]}
    DEPTS = ['GROCER','DRUGG','PRODUC','COSMET','NUTRIT','MEAT','MEAT-P','DELI','PASTRY','FLORAL']

    def block(g, dsl, dsf):
        o = {}
        for w in [7,14,28,56,84,112,224]:
            g2 = g[g.day > snapshot_day-w]
            if len(g2):
                o['spend_%d'%w]=g2.sp.sum(); o['baskets_%d'%w]=g2.basket_id.nunique()
                o['prods_%d'%w]=g2['prod'].nunique(); o['stores_%d'%w]=g2.store.nunique()
                o['qty_%d'%w]=g2.quantity.sum(); o['retail_disc_%d'%w]=-g2.retail_disc.sum()
                o['coupon_disc_%d'%w]=-(g2.coupon_disc.sum()+g2.coupon_match_disc.sum())
                o['spend_max_basket_%d'%w]=g2.groupby('basket_id').sp.sum().max()
            else: o.update(Zs[w])
        o['spend_all']=g.sp.sum(); o['baskets_all']=g.basket_id.nunique(); o['prods_all']=g['prod'].nunique()
        o['stores_all']=g.store.nunique(); o['qty_all']=g.quantity.sum()
        o['retail_disc_all']=-g.retail_disc.sum(); o['coupon_disc_all']=-(g.coupon_disc.sum()+g.coupon_match_disc.sum())
        o['spend_max_basket_all']= g.groupby('basket_id').sp.sum().max() if len(g)>0 else 0.0
        o['days_since_last']=dsl; o['days_since_first']=dsf
        s7,s14,s28,s56,s84,s112,s224 = [o['spend_%d'%w] for w in [7,14,28,56,84,112,224]]
        b7,b14,b28,b56,b84,b112,b224 = [o['baskets_%d'%w] for w in [7,14,28,56,84,112,224]]
        o['spend_decay_112'] = s7 + (s28-s7)*0.5**(7/28) + (s56-s28)*0.5**(28/28) + (s84-s56)*0.5**(56/28) + (s112-s84)*0.5**(84/28)
        o['trend_28_56']= s28/max(s56-s28,1e-6); o['trend_56_112']= s56/max(s112-s56,1e-6); o['trend_84_224']= s84/max(s224-s84,1e-6)
        o['share_recent_all']= s28/max(o['spend_all'],1e-6)
        o['avg_basket_84']= s84/max(b84,1e-6); o['avg_basket_28']= s28/max(b28,1e-6)
        g84 = g[g.day>snapshot_day-84]; g28 = g[g.day>snapshot_day-28]
        o['basket_std_84']= g84.groupby('basket_id').sp.sum().std() if b84>1 else 0.0
        o['basket_min_84']= g84.groupby('basket_id').sp.sum().min() if b84>0 else 0.0
        o['items_per_basket_84']= g84.groupby('basket_id').size().mean() if b84>0 else 0.0
        o['qty_per_basket_84']= g84.groupby('basket_id').quantity.sum().mean() if b84>0 else 0.0
        o['baskets_per_day_84']= b84/84.0
        o['active_28']= float(b28>0); o['active_84']= float(b84>0); o['active_112']= float(b112>0)
        d = g.day.values; gaps = np.diff(d); gaps = gaps[gaps>0]
        aft = d[1:][np.diff(d)>0] if len(d)>1 else np.array([])
        gp = gaps[aft>snapshot_day-112] if len(gaps) else gaps
        o['gap_mean_112']= gp.mean() if len(gp)>0 else 0.0
        o['gap_std_112']= gp.std() if len(gp)>1 else 0.0
        h = g.hour
        o['hour_mean_84']= h.mean() if len(h)>0 else 0.0; o['hour_std_84']= h.std() if len(h)>1 else 0.0
        o['evening_share_84']= (h>=17).mean() if len(h)>0 else 0.0
        for dd in DEPTS:
            o['dept_share_%s'%dd]= g84[g84.dept==dd].sp.sum()/max(s84,1e-6)
        o['dept_other_share_84']= 1.0 - sum(o['dept_share_%s'%dd] for dd in DEPTS)
        o['n_depts_84']= g84.dept.nunique()
        o['private_share_84']= g84.priv.mean() if len(g84)>0 else 0.0
        o['top_store_share_84']= (g84.groupby('store').sp.sum().max()/s84) if s84>0 else 0.0
        o['spend_rate_all_28']= o['spend_all']/max(dsf,1)
        o['spend_per_trip_28']= s28/max(b28,1e-6); o['spend_per_trip_84']= s84/max(b84,1e-6)
        o['freq_ratio_28_84']= (b28/28.0)/max(b84/84.0,1e-6); o['freq_ratio_7_28']= (b7/7.0)/max(b28/28.0,1e-6)
        o['gap_last_vs_mean']= dsl/max(o['gap_mean_112'],1e-6)
        o['gap_z']= (dsl-o['gap_mean_112'])/max(o['gap_std_112'],1e-6)
        o['n_prod_84']= g84['prod'].nunique(); o['n_manu_84']= g84.manu.nunique()
        o['store_switch_84']= g84.store.nunique(); o['top_store_spend_84']= g84.groupby('store').sp.sum().max() if len(g84)>0 else 0.0
        o['disc_share_84']= (o['retail_disc_84']+o['coupon_disc_84'])/max(s84,1e-6)
        o['disc_share_28']= (o['retail_disc_28']+o['coupon_disc_28'])/max(s28,1e-6)
        o['avg_unit_price_84']= s84/max(g84.quantity.sum(),1e-6)
        o['avg_unit_price_28']= s28/max(g28.quantity.sum(),1e-6)
        ws = g.groupby('wk').sp.sum(); bs = g.groupby('wk').basket_id.nunique()
        arr = np.array([ws.get(wkmax-i,0.0) for i in range(12)]); arrb = np.array([bs.get(wkmax-i,0) for i in range(12)])
        o['wk_mean_12']= arr.mean(); o['wk_std_12']= arr.std(); o['wk_max_12']= arr.max(); o['wk_min_12']= arr.min()
        o['wk_cv_12']= arr.std()/max(arr.mean(),1e-6); o['wk_zero_12']= float((arr==0).sum())
        o['wk_lag1']= arr[0]; o['wk_lag2']= arr[1]; o['wk_lag3']= arr[2]; o['wk_lag4']= arr[3]
        o['wk_lag1_over_mean']= arr[0]/max(arr.mean(),1e-6)
        o['wk_active_12']= float((arr>0).sum()); o['wkb_mean_12']= arrb.mean(); o['wkb_max_12']= arrb.max()
        o['spend_ly_4w']= ws[(ws.index>=wkmax-53)&(ws.index<=wkmax-50)].sum()
        o['spend_ly_8w']= ws[(ws.index>=wkmax-56)&(ws.index<=wkmax-53)].sum()
        o['spend_ly_12w']= ws[(ws.index>=wkmax-60)&(ws.index<=wkmax-57)].sum()
        o['ratio_ly']= o['spend_ly_4w']/max(o['spend_ly_8w']+o['spend_ly_12w'],1e-6)
        o['active_ly']= float(o['spend_ly_4w']>0)
        o['spend28_on_disp']= g28[g28.product_id.isin(disp_p)].sp.sum()
        o['spend28_on_mail']= g28[g28.product_id.isin(mail_p)].sp.sum()
        o['share28_on_disp']= o['spend28_on_disp']/max(s28,1e-6)
        o['share28_on_mail']= o['spend28_on_mail']/max(s28,1e-6)
        o['ix_spend_x_freq']= s84*o['baskets_per_day_84']; o['ix_decay_x_trend']= o['spend_decay_112']*o['trend_28_56']
        o['ix_decay_x_freq']= o['spend_decay_112']*o['freq_ratio_28_84']
        o['log_spend_84']= np.log1p(s84); o['log_spend_28']= np.log1p(s28)
        o['sqrt_spend_84']= np.sqrt(s84); o['sqrt_spend_28']= np.sqrt(s28)
        return pd.Series(o)

    rows = {}
    for k, g in tx.groupby('household_key'):
        rows[k] = block(g, snapshot_day - int(g.day.max()), snapshot_day - int(g.day.min()))
    for k in hh:
        if k not in rows: rows[k] = block(tx.iloc[0:0], 999, 999)
    F = pd.DataFrame(rows).T
    for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
        s = pd.Series(pd.Categorical(F.index.map(dmi[c])).codes, index=F.index).astype(float)
        s[s<0]=np.nan
        F['dem_'+c]= s
    F['has_demo']= F.index.isin(dmi.index).astype(float)
    F['week_of_year']= float(((snapshot_day+8)//7)%52)
    F.index.name='household_key'
    return F.astype(float)

t0=time.time()
try:
    out = A.build_features(build)
    print('built', out.shape, round(time.time()-t0,1),'s'); 
    A.save_table(out, 'feats_v5.parquet')
except Exception as e:
    print('BUILD FAILED:', type(e).__name__, e)
    out = A.load_saved('feats_v3.parquet')

tt = A.train_targets()
tr = out.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats = [c for c in out.columns if c not in ('household_key','snapshot_day')]
Xtr = tr[feats].astype(float); med = Xtr.median(); Xtr = Xtr.fillna(med); y = tr.future_spend_4w.values
val = out[out.snapshot_day.isin(A.snapshot_days()['validation'])].copy()
Xv = val[feats].astype(float).fillna(med).replace([np.inf,-np.inf],0)
Xtr = Xtr.replace([np.inf,-np.inf],0)
print('train', Xtr.shape, 'val', Xv.shape, 'nfeat', len(feats))
P = np.zeros(len(Xv))
t1=time.time()
for sd in [7,17]:
    m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=0.5,
        tree_method='hist', n_jobs=8, random_state=sd); m.fit(Xtr,y); P += 0.2*m.predict(Xv)
m2 = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.02, max_depth=7, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=8, random_state=7); m2.fit(Xtr,y); P += 0.4*m2.predict(Xv)
for sd in [7,17]:
    h = HistGradientBoostingRegressor(loss='quantile', quantile=0.5, max_iter=400, learning_rate=0.06,
        max_leaf_nodes=31, l2_regularization=1.0, random_state=sd); h.fit(Xtr,y); P += 0.1*h.predict(Xv)
print('train time', round(time.time()-t1,1))
val['prediction']=P
print('pred mean', round(P.mean(),2))
p = A.save_table(val[['household_key','snapshot_day','prediction']], 'pred_e010.parquet')
print('saved', p)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

def load(name):
    df = A.load_saved(name)
    df.columns = [str(c) for c in df.columns]
    cand = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    pcol = 'prediction' if 'prediction' in df.columns else cand[-1]
    return df[['household_key','snapshot_day',pcol]].rename(columns={pcol:'prediction'})

p8 = load('pred_e008.parquet'); p9 = load('pred_e009.parquet'); p10 = load('pred_e010.parquet')
print(len(p8), len(p9), len(p10))
m = p8.merge(p9, on=['household_key','snapshot_day'], suffixes=('_8','_9')).merge(p10, on=['household_key','snapshot_day'])
print('merged', m.shape, 'nan', m.isna().sum().sum())
m['prediction'] = 0.4*m['prediction_8'] + 0.3*m['prediction_9'] + 0.3*m['prediction']
print(m['prediction'].describe().round(2))
path = A.save_table(m[['household_key','snapshot_day','prediction']], 'pred_e010_blend.parquet')
print(path)
