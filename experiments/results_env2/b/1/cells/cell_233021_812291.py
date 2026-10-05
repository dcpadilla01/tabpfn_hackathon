import agent_api, numpy as np, pandas as pd, time
t0 = time.time()

def build_group(view, snapshot_day):
    D = int(snapshot_day)
    hh = view.households
    tx = view.table('transactions')
    out = pd.DataFrame(index=hh)

    prod = view.table('products')
    dep = prod[['product_id','department']].drop_duplicates('product_id')
    dep = dep.assign(department=dep.department.astype(str))
    tx = tx.merge(dep, on='product_id', how='left')
    tx['department'] = tx['department'].astype(object).where(tx['department'].notna(), 'UNK')

    m28 = tx.day >= D-27
    m84 = tx.day >= D-83

    # A: dept momentum
    s84 = tx[m84].groupby('department').sales_value.sum()
    topd = list(s84.sort_values(ascending=False).head(10).index)
    t28h = tx[m28].groupby(['household_key','department']).sales_value.sum()
    t84h = tx[m84].groupby(['household_key','department']).sales_value.sum()
    sh28 = t28h.unstack('department').reindex(columns=topd)
    sh84 = t84h.unstack('department').reindex(columns=topd)
    sh28 = sh28.div(sh28.sum(1)+1e-9, axis=0).reindex(hh)
    sh84 = sh84.div(sh84.sum(1)+1e-9, axis=0).reindex(hh)
    mom = (sh28.fillna(0) - sh84.fillna(0))
    for d in topd:
        out['dm_mom_'+d] = mom[d].values
    for d in topd[:5]:
        out['dm_sh28_'+d] = sh28[d].fillna(0).values

    # C: habit / repeat
    t28 = tx[m28]
    tprev = tx[(tx.day >= D-111) & (tx.day < D-27)]
    pairs = tprev[['household_key','product_id']].drop_duplicates()
    rep = t28.merge(pairs, on=['household_key','product_id'], how='left', indicator=True)
    rep28 = rep[rep['_merge']=='both'].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    sp28h = t28.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['hb_repeat_share_28'] = (rep28/(sp28h+1e-9)).values
    n28 = t28.groupby('household_key').product_id.nunique().reindex(hh)
    nrep = rep[rep['_merge']=='both'].groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    out['hb_rep_prod_frac_28'] = (nrep/(n28+1e-9)).values
    d84 = t84h.unstack('department').reindex(hh).fillna(0)
    p_ = d84.div(d84.sum(1)+1e-9, axis=0).clip(lower=1e-9)
    out['hb_dept_entropy_84'] = (-(p_*np.log(p_)).sum(1)).values
    gp84 = tx[m84].groupby(['household_key','product_id']).sales_value.sum()
    mx = gp84.groupby('household_key').max().reindex(hh)
    tot = gp84.groupby('household_key').sum().reindex(hh)
    out['hb_top1prod_share_84'] = (mx/(tot+1e-9)).values
    told = tx[tx.day >= D-363]
    oldp = told[['household_key','product_id']].drop_duplicates()
    newr = t28.merge(oldp, on=['household_key','product_id'], how='left', indicator=True)
    newsp = newr[newr['_merge']=='left_only'].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['hb_new_prod_share_28'] = (newsp/(sp28h+1e-9)).values

    # D: stock-up basket shape
    b = tx.groupby(['household_key','basket_id']).agg(bspend=('sales_value','sum'), bday=('day','first'), bt=('trans_time','mean'))
    b84 = b[b.bday >= D-83]
    g = b84.groupby('household_key')
    bmax = g.bspend.max().reindex(hh)
    bmed = g.bspend.median().reindex(hh)
    bsum = g.bspend.sum().reindex(hh)
    out['su_max_over_med_84'] = (bmax/(bmed+1e-9)).values
    bidx = g.bspend.idxmax()
    bdaymap = b84.bday
    out['su_days_since_maxb_84'] = (D - bidx.map(bdaymap)).reindex(hh).fillna(999).values
    out['su_top1basket_share_84'] = (bmax/(bsum+1e-9)).values
    bb = b84.assign(bb=b84.bspend > 2*(bmed+1e-9))
    out['su_bigbasket_cnt_84'] = bb.groupby('household_key').bb.sum().reindex(hh).fillna(0).values
    out['su_stockup_share_84'] = bb.assign(sv=lambda x: x.bspend*x.bb.astype(float)).groupby('household_key').sv.sum().reindex(hh).fillna(0).values/(bsum+1e-9)
    q28 = t28.groupby('household_key').quantity.sum().reindex(hh)
    tr28 = t28.groupby('household_key').basket_id.nunique().reindex(hh)
    q84 = tx[m84].groupby('household_key').quantity.sum().reindex(hh)
    tr84 = tx[m84].groupby('household_key').basket_id.nunique().reindex(hh)
    out['su_qtyrate_28_84'] = ((q28/(tr28+1e-9))/((q84/(tr84+1e-9))+1e-9)).replace([np.inf,-np.inf],np.nan).values

    # J: time of day
    am = b84[b84.bt < 1200].groupby('household_key').bspend.sum().reindex(hh).fillna(0)
    ev = b84[b84.bt >= 1800].groupby('household_key').bspend.sum().reindex(hh).fillna(0)
    out['td_am_share_84'] = (am/(bsum+1e-9)).values
    out['td_eve_share_84'] = (ev/(bsum+1e-9)).values
    out['td_avg_time_84'] = g.bt.mean().reindex(hh).values

    # B: global level/trend/seasonal (scalars)
    tx2 = tx.assign(wk=(tx.day+8)//7)
    gw = tx2.groupby('wk').sales_value.sum()
    ga = tx2.groupby('wk').household_key.nunique()
    gwk = (gw/(ga+1e-9))
    cw = (D+8)//7
    def wmean(a, b_):
        sel = gwk.loc[a:b_]
        return float(sel.mean()) if len(sel) else np.nan
    g_level = wmean(max(1,cw-8), cw-1)
    g_prev = wmean(max(1,cw-16), cw-9)
    g_trend = np.nan
    if g_prev is not None and not np.isnan(g_prev):
        g_trend = g_level/(g_prev+1e-9)
    w0 = cw+1
    seas = np.nan
    seas_rel = 1.0
    if w0-52 >= 1:
        seas = wmean(w0-52, w0-49)
        if not np.isnan(seas) and not np.isnan(g_level):
            seas_rel = seas/(g_level+1e-9)
    n = len(hh)
    out['g_level'] = np.full(n, g_level if not np.isnan(g_level) else 0.0)
    out['g_trend'] = np.full(n, g_trend if not np.isnan(g_trend) else 1.0)
    out['g_season_rel'] = np.full(n, seas_rel)
    out['g_season_abs'] = np.full(n, seas)
    return out

df = agent_api.build_features(build_group)
print('built', df.shape, 'time', round(time.time()-t0))

e8 = agent_api.load_saved('e008_level_shape.parquet')
groups = {
 'candA_dm': ['dm_'], 'candB_g': ['g_'], 'candC_hb': ['hb_'], 'candD_su': ['su_'], 'candE_td': ['td_'],
}
for name, prefs in groups.items():
    cols = ['household_key','snapshot_day'] + [c for c in df.columns if any(c.startswith(p) for p in prefs)]
    sub = df[cols]
    mg = e8.merge(sub, on=['household_key','snapshot_day'], how='left')
    assert len(mg) == len(e8)
    p = agent_api.save_table(mg, name)
    print(name, mg.shape, 'nan%', round(mg[cols[2:]].isna().mean().mean()*100,1))
print('done', round(time.time()-t0), 's')
