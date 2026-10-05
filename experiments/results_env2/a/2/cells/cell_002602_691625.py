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
