import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def wmean_by(df, key, val, w):
    ww = w.clip(lower=0)
    num = (val*ww).groupby(df[key]).sum()
    den = ww.groupby(df[key]).sum()+1e-9
    return num/den

def feats_v4(view, snapshot_day):
    T = view.table('transactions')
    P = view.table('products')
    hh = list(view.households)
    day = snapshot_day
    out = pd.DataFrame(index=hh)
    for w in [7,14,28,56,84,168]:
        out[f'spend_{w}'] = T[T.day > day-w].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['spend_all'] = T.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    for k in [1,2,3]:
        lo, hi = day-28*k-27, day-28*k
        out[f'lag{k}_spend'] = T[(T.day>=lo)&(T.day<=hi)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['exp4w_blend'] = 0.5*out.lag1_spend + 0.3*out.lag2_spend + 0.2*out.lag3_spend
    out['trend28'] = out.lag1_spend/(out.lag2_spend+5)
    out['trend84'] = out.spend_28/(out.spend_84/3+5)
    for k in range(8):
        out[f'wk{k}'] = T[(T.day>day-7*(k+1))&(T.day<=day-7*k)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['active_weeks8'] = (out[[f'wk{k}' for k in range(8)]]>0).sum(1)
    out['wk_mean8'] = out[[f'wk{k}' for k in range(8)]].mean(1)
    out['wk_std8'] = out[[f'wk{k}' for k in range(8)]].std(1).fillna(0)
    out['wk_cv8'] = out.wk_std8/(out.wk_mean8+1)
    out['tenure'] = day - T.groupby('household_key').day.min().reindex(hh)
    out['days_since_last'] = day - T.groupby('household_key').day.max().reindex(hh)
    b = T.groupby(['household_key','basket_id']).agg(sales=('sales_value','sum'), d=('day','max'))
    b = b.sort_values(['household_key','d'])
    b84 = b[b.d > day-84]
    out['trips_84'] = b84.groupby('household_key').size().reindex(hh).fillna(0)
    out['trips_28'] = b[b.d>day-28].groupby('household_key').size().reindex(hh).fillna(0)
    out['basket_mean_84'] = b84.groupby('household_key').sales.mean().reindex(hh)
    out['basket_max_84'] = b84.groupby('household_key').sales.max().reindex(hh)
    out['basket_std_84'] = b84.groupby('household_key').sales.std().reindex(hh)
    out['spend_per_trip28'] = out.spend_28/(out.trips_28+1e-9)
    b['prev_d'] = b.groupby('household_key').d.shift(1)
    b['gap'] = b.d - b.prev_d
    g84 = b[(b.d>day-84) & b.gap.notna()]
    out['gap_mean_84'] = g84.groupby('household_key').gap.mean().reindex(hh)
    out['gap_max_84'] = g84.groupby('household_key').gap.max().reindex(hh)
    out['nprod_84'] = T[T.day>day-84].groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    out['nstore_84'] = T[T.day>day-84].groupby('household_key').store_id.nunique().reindex(hh).fillna(0)
    out['n_days_active_84'] = T[T.day>day-84].groupby('household_key').day.nunique().reindex(hh).fillna(0)
    out['coupon_disc_84'] = T[T.day>day-84].groupby('household_key').coupon_disc.sum().reindex(hh).fillna(0)
    out['retail_disc_84'] = T[T.day>day-84].groupby('household_key').retail_disc.sum().reindex(hh).fillna(0)
    out['coupon_match_disc_84'] = T[T.day>day-84].groupby('household_key').coupon_match_disc.sum().reindex(hh).fillna(0)
    out['disc_share_84'] = (out.coupon_disc_84+out.retail_disc_84+out.coupon_match_disc_84)/(out.spend_84+1e-9)
    T84 = T[T.day>day-84].copy()
    T84['morn'] = (T84.trans_time<1200).astype(float)
    out['morn_share_84'] = T84.groupby('household_key').morn.mean().reindex(hh)
    w_all = 0.97**((day-T.day).clip(lower=0))
    out['exp4w_all'] = (T.sales_value*w_all).groupby(T.household_key).sum().reindex(hh).fillna(0)
    T168 = T[T.day>day-168]
    w168 = 0.97**((day-T168.day).clip(lower=0))
    out['exp4w_168'] = (T168.sales_value*w168).groupby(T168.household_key).sum().reindex(hh).fillna(0)
    out['exp4w_84'] = out.spend_84
    out['weekly_rate_all'] = out.spend_all/(out.tenure+1)
    # ==== NEW v4 ====
    M = T84.merge(P[['product_id','commodity_desc','brand','department']], on='product_id', how='left')
    M['commodity_desc'] = M.commodity_desc.astype(object).where(M.commodity_desc.notna(), 'UNK')
    M['brand'] = M.brand.astype(object).where(M.brand.notna(), 'UNK')
    M['is_grocery'] = (M.department=='GROCERY').astype(float)
    sp = M.groupby(['household_key','commodity_desc']).sales_value.sum().reset_index()
    tot = sp.groupby('household_key').sales_value.transform('sum')
    sp['share'] = sp.sales_value/tot
    out['hhi_84'] = sp.assign(s2=sp.share**2).groupby('household_key').s2.sum().reindex(hh)
    out['n_commodities_84'] = sp.groupby('household_key').size().reindex(hh).fillna(0)
    out['top_comm_share_84'] = sp.groupby('household_key').share.max().reindex(hh)
    T168m = T[(T.day>day-168)&(T.day<=day-84)].merge(P[['product_id','commodity_desc']], on='product_id', how='left')
    T168m['commodity_desc'] = T168m.commodity_desc.astype(object).where(T168m.commodity_desc.notna(), 'UNK')
    prev = set(zip(T168m.household_key, T168m.commodity_desc))
    sp['is_rep'] = [(h,c) in prev for h,c in zip(sp.household_key, sp.commodity_desc)]
    out['repeat_rate_84'] = wmean_by(sp.reset_index(), 'household_key', sp.is_rep.values, sp.sales_value.values).reindex(hh)
    out['private_share_84'] = wmean_by(M, 'household_key', (M.brand=='Private').astype(float), M.sales_value).reindex(hh)
    out['grocery_share_84'] = wmean_by(M, 'household_key', M.is_grocery, M.sales_value).reindex(hh)
    med_b = b84.groupby('household_key').sales.median()
    r = (b84.sales/(b84.index.get_level_values(0).map(med_b)+1e-9))
    out['stockup_max_84'] = r.groupby(b84.index.get_level_values(0)).max().reindex(hh)
    st = M.groupby(['household_key','store_id']).sales_value.sum().reset_index()
    out['store_loyalty_84'] = (st.groupby('household_key').sales_value.max()/st.groupby('household_key').sales_value.sum()).reindex(hh)
    T84['is_wkend'] = (((T84.day % 7)==5) | ((T84.day % 7)==6)).astype(float)
    out['wkend_share_84'] = wmean_by(T84.reset_index(), 'household_key', T84.is_wkend.values, T84.sales_value.values).reindex(hh)
    D = view.table('demographics')
    if D is not None and len(D):
        D = D.set_index('household_key')
        for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
            out[c] = D[c].reindex(hh)
        out['has_demo'] = pd.Series(out.index.isin(D.index).astype(float), index=hh)
    out['snap_day'] = day; out['snap_week'] = (day+8)//7; out['snap_cycle_pos'] = day % 28
    return out

f4 = agent_api.build_features(feats_v4)
agent_api.save_table(f4, 'feats_v4.parquet')
print('saved', f4.shape)
new = [c for c in f4.columns if c not in agent_api.load_saved('feats_v3.parquet').columns]
print('new cols:', new)
print(f4[new].describe().T.round(3))
