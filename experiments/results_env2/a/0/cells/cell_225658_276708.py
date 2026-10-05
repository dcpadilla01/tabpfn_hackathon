import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def feats_v4(view, snapshot_day):
    T = view.table('transactions')
    P = view.table('products')
    hh = view.households
    day = snapshot_day
    out = hh.copy()
    # ---- trailing window spends (same as v3 core) ----
    for w in [7,14,28,56,84,168]:
        out[f'spend_{w}'] = T[T.day > day-w].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['spend_all'] = T.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    # cycle-aligned 4w lags
    for k in [1,2,3]:
        lo, hi = day-28*k-27, day-28*k
        out[f'lag{k}_spend'] = T[(T.day>=lo)&(T.day<=hi)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['exp4w_blend'] = 0.5*out.lag1_spend + 0.3*out.lag2_spend + 0.2*out.lag3_spend
    out['trend28'] = out.lag1_spend/(out.lag2_spend+5)
    out['trend84'] = out.spend_28/(out.spend_84/3+5)
    # weekly pattern last 8 weeks
    for k in range(8):
        out[f'wk{k}'] = T[(T.day>day-7*(k+1))&(T.day<=day-7*k)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['active_weeks8'] = (out[[f'wk{k}' for k in range(8)]]>0).sum(1)
    out['wk_mean8'] = out[[f'wk{k}' for k in range(8)]].mean(1)
    out['wk_std8'] = out[[f'wk{k}' for k in range(8)]].std(1).fillna(0)
    out['wk_cv8'] = out.wk_std8/(out.wk_mean8+1)
    # recency / tenure
    out['tenure'] = day - T.groupby('household_key').day.min().reindex(hh)
    out['days_since_last'] = day - T.groupby('household_key').day.max().reindex(hh)
    # trips / baskets
    b = T.groupby(['household_key','basket_id']).agg(sales=('sales_value','sum'), d=('day','max'))
    b = b.sort_values(['household_key','d'])
    b84 = b[b.d > day-84]
    out['trips_84'] = b84.groupby('household_key').size().reindex(hh).fillna(0)
    out['trips_28'] = b[b.d>day-28].groupby('household_key').size().reindex(hh).fillna(0)
    out['basket_mean_84'] = b84.groupby('household_key').sales.mean().reindex(hh)
    out['basket_max_84'] = b84.groupby('household_key').sales.max().reindex(hh)
    out['basket_std_84'] = b84.groupby('household_key').sales.std().reindex(hh)
    out['spend_per_trip28'] = out.spend_28/(out.trips_28+1e-9)
    # gaps
    b['prev_d'] = b.groupby('household_key').d.shift(1)
    b['gap'] = b.d - b.prev_d
    g84 = b[(b.d>day-84) & b.gap.notna()]
    out['gap_mean_84'] = g84.groupby('household_key').gap.mean().reindex(hh)
    out['gap_max_84'] = g84.groupby('household_key').gap.max().reindex(hh)
    # diversity
    out['nprod_84'] = T[T.day>day-84].groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    out['nstore_84'] = T[T.day>day-84].groupby('household_key').store_id.nunique().reindex(hh).fillna(0)
    out['n_days_active_84'] = T[T.day>day-84].groupby('household_key').day.nunique().reindex(hh).fillna(0)
    # discounts
    out['coupon_disc_84'] = T[T.day>day-84].groupby('household_key').coupon_disc.sum().reindex(hh).fillna(0)
    out['retail_disc_84'] = T[T.day>day-84].groupby('household_key').retail_disc.sum().reindex(hh).fillna(0)
    out['coupon_match_disc_84'] = T[T.day>day-84].groupby('household_key').coupon_match_disc.sum().reindex(hh).fillna(0)
    out['disc_share_84'] = (out.coupon_disc_84+out.retail_disc_84+out.coupon_match_disc_84)/(out.spend_84+1e-9)
    # time of day
    T84 = T[T.day>day-84].copy()
    T84['morn'] = (T84.trans_time<1200).astype(float)
    out['morn_share_84'] = T84.groupby('household_key').morn.mean().reindex(hh)
    # exp blends all-history
    w_all = 0.97**((day-T.day).clip(lower=0))
    out['exp4w_all'] = (T.sales_value*w_all).groupby(T.household_key).sum().reindex(hh).fillna(0)
    T168 = T[T.day>day-168]
    w168 = 0.97**((day-T168.day).clip(lower=0))
    out['exp4w_168'] = (T168.sales_value*w168).groupby(T168.household_key).sum().reindex(hh).fillna(0)
    out['exp4w_84'] = out.spend_84
    out['weekly_rate_all'] = out.spend_all/(out.tenure+1)
    # ================= NEW v4: product-mix / loyalty / stock-up =================
    M = T84.merge(P[['product_id','commodity_desc','brand']], on='product_id', how='left')
    M['commodity_desc'] = M.commodity_desc.fillna('UNK')
    M['brand'] = M.brand.fillna('UNK')
    sp = M.groupby(['household_key','commodity_desc']).sales_value.sum().reset_index()
    tot = sp.groupby('household_key').sales_value.transform('sum')
    sp['share'] = sp.sales_value/tot
    out['hhi_84'] = sp.assign(s2=sp.share**2).groupby('household_key').s2.sum().reindex(hh)
    out['n_commodities_84'] = sp.groupby('household_key').size().reindex(hh).fillna(0)
    out['top_comm_share_84'] = sp.groupby('household_key').share.max().reindex(hh)
    # repeat rate: commodities bought in (84,168] before snapshot also bought in trailing 84
    T168m = T[(T.day>day-168)&(T.day<=day-84)].merge(P[['product_id','commodity_desc']], on='product_id', how='left')
    prev = set(zip(T168m.household_key, T168m.commodity_desc.fillna('UNK')))
    sp['is_rep'] = [(h,c) in prev for h,c in zip(sp.household_key, sp.commodity_desc)]
    out['repeat_rate_84'] = sp.groupby('household_key').apply(lambda d: np.average(d.is_rep, weights=d.sales_value)).reindex(hh)
    # private-brand share
    out['private_share_84'] = (M.assign(pb=(M.brand=='Private').astype(float))
                                .groupby('household_key').apply(lambda d: np.average(d.pb, weights=d.sales_value)).reindex(hh))
    # grocery dept share
    out['grocery_share_84'] = M.assign(g=(M.commodity_desc=='UNK')*0 + (P.set_index('product_id').department.reindex(M.product_id).values=='GROCERY').astype(float)).groupby('household_key').apply(lambda d: np.average(d.g, weights=d.sales_value)).reindex(hh)
    # stock-up: max basket / median basket; frac of baskets with sales > 2.5x prev
    b84r = b84.reset_index().copy()
    b84r['prev_sales'] = b.groupby('household_key').sales.shift(1).reindex(b84.index)
    out['stockup_max_84'] = (b84r.sales/(b84.groupby('household_key').sales.median().reindex(b84r.household_key).values+1e-9)).groupby(b84r.household_key).max().reindex(hh)
    # store loyalty
    st = M.groupby(['household_key','store_id']).sales_value.sum().reset_index()
    out['store_loyalty_84'] = st.groupby('household_key').sales_value.max()/st.groupby('household_key').sales_value.sum().reindex(hh)
    # weekend share
    T84['is_wkend'] = ((T84.day % 7)==5) | ((T84.day % 7)==6)
    out['wkend_share_84'] = T84.groupby('household_key').apply(lambda d: np.average(d.is_wkend.astype(float), weights=d.sales_value)).reindex(hh)
    # demographics
    D = view.table('demographics')
    if D is not None and len(D):
        D = D.set_index('household_key')
        for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
            out[c] = D[c].reindex(hh)
        out['has_demo'] = out.index.isin(D.index).astype(float)
    out['snap_day'] = day
    out['snap_week'] = (day+8)//7
    out['snap_cycle_pos'] = day % 28
    return out

f4 = agent_api.build_features(feats_v4)
agent_api.save_table(f4, 'feats_v4.parquet')
print('saved', f4.shape)
print(f4.columns.tolist())
