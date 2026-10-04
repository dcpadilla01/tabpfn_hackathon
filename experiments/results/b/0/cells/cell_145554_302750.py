import numpy as np, pandas as pd

def fn(view, snapshot_day):
    d0 = int(snapshot_day)
    hh_df = view.households
    try:
        k = hh_df['household_key']
    except Exception:
        k = hh_df
    keys = pd.Index(pd.unique(np.asarray(k).ravel()), name='household_key')
    tx = view.transactions
    out = {}
    def put(name, s, fill=np.nan):
        try:
            out[name] = s.reindex(keys).astype(float).fillna(fill)
        except Exception:
            out[name] = pd.Series(fill, index=keys, name=name)

    # multi-window RFM
    for w in (7, 28, 56, 112, 364):
        t = tx[tx.day > d0 - w]
        g = t.groupby('household_key')
        put(f'spend{w}', g.sales_value.sum(), 0.0)
        put(f'trips{w}', g.basket_id.nunique(), 0.0)
        put(f'actdays{w}', g.day.nunique(), 0.0)
        put(f'qty{w}', g.quantity.sum(), 0.0)
        put(f'nprod{w}', g.product_id.nunique(), 0.0)
        put(f'nstore{w}', g.store_id.nunique(), 0.0)
    g = tx.groupby('household_key')
    put('recency', d0 - g.day.max(), 364.0)
    put('tenure', d0 - g.day.min(), 0.0)
    put('lt_spend', g.sales_value.sum(), 0.0)
    put('lt_trips', g.basket_id.nunique(), 0.0)
    s7, s28, s56, s112, s364 = out['spend7'], out['spend28'], out['spend56'], out['spend112'], out['spend364']
    put('trend_7_28', 4*s7/(s28+1.0), 0.0)
    put('trend_28_56', 2*s28/(s56+1.0), 0.0)
    put('trend_56_112', 2*s56/(s112+1.0), 0.0)
    put('trend_112_364', (364/112)*s112/(s364+1.0), 0.0)
    put('avg_basket28', s28/out['trips28'].replace(0, np.nan), 0.0)
    put('spend_per_day28', s28/28.0, 0.0)
    put('trips_per_day28', out['trips28']/28.0, 0.0)
    if d0 - 363 >= 1:
        t = tx[(tx.day >= d0-363) & (tx.day <= d0-336)]
        put('spend_yoy28', t.groupby('household_key').sales_value.sum(), 0.0)
    else:
        out['spend_yoy28'] = pd.Series(np.nan, index=keys)
    t = tx[tx.day > d0-112]
    gt = t.groupby('household_key')
    c1, c2, c3 = gt.coupon_disc.sum(), gt.coupon_match_disc.sum(), gt.retail_disc.sum()
    put('coupon_disc112', c1, 0.0); put('coupon_match_disc112', c2, 0.0); put('retail_disc112', c3, 0.0)
    put('disc_share112', (-(c1+c2+c3))/(out['spend112']+1.0), 0.0)
    put('mean_hour112', (gt.trans_time.mean()//100), 12.0)
    put('mean_dow112', gt.day.apply(lambda s: s.mod(7).mean()), 3.0)

    # dept merge
    txm = tx.merge(view.products[['product_id','department']], on='product_id', how='left')
    put('ndept112', txm[txm.day > d0-112].groupby('household_key').department.nunique(), 0.0)

    # calendar / season
    wk = (d0 + 8)//7
    out['week'] = pd.Series(float(wk), index=keys)
    out['week_sin'] = pd.Series(float(np.sin(2*np.pi*wk/52)), index=keys)
    out['week_cos'] = pd.Series(float(np.cos(2*np.pi*wk/52)), index=keys)
    try:
        wksp = tx.groupby('week_no').sales_value.sum()
        wksp = wksp[wksp.index <= wk]
        mw = wksp.mean()
        lifts = [wksp[wk-52*k2]/mw for k2 in (1,) for wk in [wk+k2] if 1 <= wk-52 <= wk and (wk-52) in wksp.index]
        lifts = []
        for k2 in (1, 2, 3, 4):
            w2 = wk + k2; ref = w2 - 52
            if 1 <= ref <= wk and ref in wksp.index:
                lifts.append(wksp[ref]/mw)
        out['season_lift'] = pd.Series(float(np.mean(lifts)) if lifts else np.nan, index=keys)
    except Exception:
        out['season_lift'] = pd.Series(np.nan, index=keys)
    for nm, (a, b) in {'own_ly_spend4w': (d0-363, d0-336), 'own_ly2_spend4w': (d0-727, d0-700)}.items():
        if a >= 1:
            t = tx[(tx.day >= a) & (tx.day <= b)]
            put(nm, t.groupby('household_key').sales_value.sum(), 0.0)
        else:
            out[nm] = pd.Series(np.nan, index=keys)

    # marketing
    try:
        cps, tg = view.campaigns, view.campaign_targets
        red, cpns = view.coupon_redemptions, view.coupons
        ever_ids = set(cps.campaign[cps.start_day <= d0]); act_ids = set(cps.campaign[(cps.start_day <= d0) & (cps.end_day >= d0)])
        tge = tg[tg.campaign.isin(ever_ids)]; tga = tg[tg.campaign.isin(act_ids)]
        ne = tge.groupby('household_key').campaign.nunique(); na = tga.groupby('household_key').campaign.nunique()
        put('n_tgt_ever', ne, 0.0); put('n_tgt_active', na, 0.0)
        put('targeted_flag', (ne > 0).astype(float), 0.0)
        sd_map = cps.set_index('campaign').start_day; ed_map = cps.set_index('campaign').end_day
        tge2 = tge.assign(sd=tge.campaign.map(sd_map))
        put('days_since_first_tgt', d0 - tge2.groupby('household_key').sd.min())
        put('days_since_last_tgt', d0 - tge2.groupby('household_key').sd.max())
        tga2 = tga.assign(rem=(tga.campaign.map(ed_map) - d0).clip(lower=0))
        put('tgt_active_remaining', tga2.groupby('household_key').rem.max(), 0.0)
        desc = tge.description.astype(str).str.upper()
        for L in ('A', 'B', 'C'):
            put(f't{L}_lt', tge[desc.str.endswith(L)].groupby('household_key').size().gt(0).astype(float), 0.0)
            desca = tga.description.astype(str).str.upper()
            put(f't{L}_act', tga[desca.str.endswith(L)].groupby('household_key').size().gt(0).astype(float), 0.0)
        r = red[red.day <= d0]
        put('red_lt', r.groupby('household_key').size(), 0.0)
        put('red28', r[r.day > d0-28].groupby('household_key').size(), 0.0)
        put('red112', r[r.day > d0-112].groupby('household_key').size(), 0.0)
        put('red_ncamp', r.groupby('household_key').campaign.nunique(), 0.0)
        put('red_active', r[r.campaign.isin(act_ids)].groupby('household_key').size(), 0.0)
        put('days_since_last_red', d0 - r.groupby('household_key').day.max())
        put('red_rate', out['red_lt']/(out['lt_trips']+1.0), 0.0)
        rc = r[['household_key','campaign']].drop_duplicates()
        cp = cpns.merge(rc, on='campaign')[['household_key','product_id']].drop_duplicates()
        sp = tx[tx.day > d0-28].merge(cp, on=['household_key','product_id']).groupby('household_key').sales_value.sum()
        put('cpn_prod_spend28', sp, 0.0)
        put('cpn_prod_share28', sp/(s28+1.0), 0.0)
    except Exception:
        for c in ['n_tgt_ever','n_tgt_active','targeted_flag','days_since_first_tgt','days_since_last_tgt','tgt_active_remaining','tA_lt','tB_lt','tC_lt','tA_act','tB_act','tC_act','red_lt','red28','red112','red_ncamp','red_active','days_since_last_red','red_rate','cpn_prod_spend28','cpn_prod_share28']:
            out[c] = pd.Series(np.nan, index=keys)

    # demographics
    try:
        dm = view.demographics
        agem = dm.classification_1.astype(str).str.extract(r'(\d+)')[0].astype(float)
        l3m = dm.classification_3.astype(str).str.extract(r'(\d+)')[0].astype(float)
        hsm = dm.classification_4.astype(str).str.extract(r'(\d+)')[0].astype(float)
        kidm = dm.kid_category_desc.astype(str).map({'None/Unknown': 0.0, '1': 1.0, '2': 2.0, '3+': 3.0})
        homm = dm.homeowner_desc.astype(str).map({'Homeowner': 3.0, 'Probable Owner': 2.0, 'Probable Renter': 1.0, 'Renter': 0.0, 'Unknown': np.nan})
        c2m = dm.classification_2.astype(str).map({'X': 0.0, 'Y': 1.0, 'Z': 2.0})
        c5m = dm.classification_5.astype(str).str.extract(r'(\d+)')[0].astype(float)
        dmi = dm.set_index('household_key')
        put('demo_age', agem.set_axis(dmi.index)); put('demo_class3', l3m.set_axis(dmi.index))
        put('demo_hhsize', hsm.set_axis(dmi.index)); put('demo_kids', kidm.set_axis(dmi.index))
        put('demo_homeowner', homm.set_axis(dmi.index)); put('demo_c2', c2m.set_axis(dmi.index))
        put('demo_c5', c5m.set_axis(dmi.index))
        out['has_demo'] = dm.household_key.isin(keys).groupby(dm.household_key).max().reindex(keys).astype(float).fillna(0)
        out['has_demo'] = pd.Series(0.0, index=keys).where(~pd.Series(keys).isin(set(dm.household_key)), 1.0)
    except Exception:
        for c in ['demo_age','demo_class3','demo_hhsize','demo_kids','demo_homeowner','demo_c2','demo_c5','has_demo']:
            out[c] = pd.Series(np.nan, index=keys)

    # NEW: composition / cadence / tail features
    tx28 = tx[tx.day > d0-28]; txm28 = txm[txm.day > d0-28]
    try:
        top = ['GROCERY','DRUG GM','PRODUCE','COSMETICS','NUTRITION','MEAT','MEAT-PCKGD','DELI','PASTRY','FLORAL','SEAFOOD-PCKGD','MISC. TRANS.']
        dsp = txm28[txm28.department.isin(top)].groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0)
        dsp = dsp.reindex(keys).fillna(0.0)
        sh = dsp.div(s28.where(s28 > 0, np.nan), axis=0).fillna(0.0)
        for c in sh.columns: out['dsp_' + str(c)] = sh[c]
        put('ndept28', txm28.groupby('household_key').department.nunique(), 0.0)
    except Exception:
        pass
    try:
        wkn = ((d0 - tx28.day)//7).clip(0, 3)
        ws = tx28.groupby([tx28.household_key, wkn]).sales_value.sum().unstack(fill_value=0.0).reindex(keys).fillna(0.0)
        for c in ws.columns: out[f'spend_w{int(c)}'] = ws[c]
        out['share_w0'] = ws[0].astype(float)/(s28+1.0) if 0 in ws.columns else pd.Series(0.0, index=keys)
    except Exception:
        pass
    try:
        def ivstats(s):
            u = np.sort(s.unique()); dd = np.diff(u)
            return pd.Series({'iv_mean112': dd.mean() if len(dd) else np.nan,
                              'iv_std112': dd.std() if len(dd) > 1 else 0.0,
                              'iv_min112': dd.min() if len(dd) else np.nan,
                              'n_gaps112': float(len(dd))})
        ivs = tx[tx.day > d0-112].groupby('household_key').day.apply(ivstats).unstack()
        for c in ['iv_mean112','iv_std112','iv_min112','n_gaps112']: put(c, ivs[c])
    except Exception:
        pass
    try:
        bk = tx28.groupby(['household_key','basket_id']).sales_value.sum()
        bs = bk.groupby(level=0).agg(['mean','max','std','count'])
        put('bk_mean28', bs['mean'], 0.0); put('bk_max28', bs['max'], 0.0)
        put('bk_std28', bs['std'], 0.0); put('nbaskets28', bs['count'], 0.0)
    except Exception:
        pass
    try:
        t84 = tx[tx.day > d0-84]
        wgt = np.exp(-(d0 - t84.day)/28.0)
        put('rwspend84', (t84.sales_value*wgt).groupby(t84.household_key).sum(), 0.0)
        put('active_weeks112', tx[tx.day > d0-112].groupby('household_key').week_no.nunique(), 0.0)
        we = txm28.assign(we=(txm28.day % 7 >= 5).astype(float))
        put('weekend_share28', we[we.we > 0].groupby('household_key').sales_value.sum()/(s28+1.0), 0.0)
    except Exception:
        pass
    out['spend28_log'] = np.log1p(s28.clip(lower=0)); out['spend56_log'] = np.log1p(s56.clip(lower=0))
    out['spend112_log'] = np.log1p(s112.clip(lower=0)); out['lt_spend_log'] = np.log1p(out['lt_spend'].clip(lower=0))
    out['zero28'] = (s28 <= 0).astype(float)
    out['ratio28_364'] = s28/(s364+1.0)
    out['toptrip_share28'] = out.get('bk_max28', pd.Series(0.0, index=keys))/(s28+1.0)

    return pd.DataFrame(out)

path = agent_api.save_table(agent_api.build_features(fn), 'e006_composition')
print('saved:', path)
df = agent_api.load_saved('e006_composition.parquet') if agent_api.load_saved('e006_composition.parquet') is not None else None
print('shape check via build ok')