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
