import pandas as pd, numpy as np

E3 = agent_api.load_saved('rfm_cadence_v1.parquet')

def mkt_fn(view, snapshot_day):
    snap = snapshot_day
    hh = pd.Index(view.households, name='household_key')
    tx = view.table('transactions')
    ct = view.table('campaign_targets')
    cps = view.table('campaigns')
    cr = view.table('coupon_redemptions')
    dm = view.table('display_mailer')
    feat = pd.DataFrame(index=hh)

    # ---- campaign targeting ----
    cps = cps[cps.start_day <= snap]
    ctm = ct.merge(cps[['campaign','start_day','end_day']], on='campaign', how='inner')
    g = ctm.groupby('household_key')
    feat['n_tgt_total'] = g.size().reindex(hh).fillna(0)
    feat['n_tgt_84d']  = ctm[ctm.start_day > snap-84].groupby('household_key').size().reindex(hh).fillna(0)
    feat['n_tgt_168d'] = ctm[ctm.start_day > snap-168].groupby('household_key').size().reindex(hh).fillna(0)
    for t in ['TypeA','TypeB','TypeC']:
        feat['tgt_'+t] = ctm[ctm.description==t].groupby('household_key').size().reindex(hh).fillna(0)
    feat['tgt_active_now'] = ctm[(ctm.start_day<=snap)&(ctm.end_day>=snap)].groupby('household_key').size().reindex(hh).fillna(0)
    feat['days_since_tgt'] = (snap - g['start_day'].max()).reindex(hh).fillna(730).clip(lower=0)

    # ---- coupon redemptions ----
    r = cr[cr.day <= snap]
    rg = r.groupby('household_key')
    feat['n_red_total']  = rg.size().reindex(hh).fillna(0)
    feat['n_red_84d']    = r[r.day > snap-84].groupby('household_key').size().reindex(hh).fillna(0)
    feat['n_red_168d']   = r[r.day > snap-168].groupby('household_key').size().reindex(hh).fillna(0)
    feat['days_since_red'] = (snap - rg['day'].max()).reindex(hh).fillna(730).clip(lower=0)
    feat['n_red_campaigns'] = rg['campaign'].nunique().reindex(hh).fillna(0)

    # ---- display / mailer exposure over last 8 weeks ----
    wk = (snap + 8)//7
    dmw = dm[(dm.week_no > wk-8) & (dm.week_no <= wk)]
    txw = tx[tx.day > snap-56].copy()
    txw['week_no'] = (txw.day + 8)//7
    txw = txw[txw.week_no > wk-8]
    m = txw.merge(dmw, on=['product_id','store_id','week_no'], how='left')
    disp = m['display'].notna() & (m['display'] != '0')
    mail = m['mailer'].notna() & (m['mailer'] != '0')
    tot = m.groupby('household_key')['sales_value'].sum()
    def share(mask):
        s = m[mask].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        return pd.Series(np.where(tot.reindex(hh).values > 0, s.values/np.maximum(tot.reindex(hh).values,1e-9), np.nan), index=hh)
    feat['disp_spend_share8'] = share(disp)
    feat['mail_spend_share8'] = share(mail)
    n_it = m.groupby('household_key').size().reindex(hh).fillna(0)
    feat['disp_item_share8'] = pd.Series(np.where(n_it.values>0, m[disp].groupby('household_key').size().reindex(hh).fillna(0).values/np.maximum(n_it.values,1), np.nan), index=hh)
    feat['n_disp_products8'] = m[disp].groupby('household_key')['product_id'].nunique().reindex(hh).fillna(0)
    # last 4 weeks only
    dm4 = dm[(dm.week_no > wk-4) & (dm.week_no <= wk)]
    m4 = txw[txw.week_no > wk-4].merge(dm4, on=['product_id','store_id','week_no'], how='left')
    d4 = m4['display'].notna() & (m4['display'] != '0')
    tot4 = m4.groupby('household_key')['sales_value'].sum()
    s4 = m4[d4].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
    feat['disp_spend_share4'] = pd.Series(np.where(tot4.reindex(hh).values>0, s4.values/np.maximum(tot4.reindex(hh).values,1e-9), np.nan), index=hh)

    # ---- promo / price sensitivity from tx (last 84d) ----
    t84 = tx[tx.day > snap-84]
    sp = t84.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
    for col, name in [('coupon_disc','sh_coupon'),('retail_disc','sh_retaildisc'),('coupon_match_disc','sh_match')]:
        s = t84[t84[col] != 0].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        feat[name+'_spend84'] = pd.Series(np.where(sp.values>0, s.values/np.maximum(sp.values,1e-9), np.nan), index=hh)
    return feat

# --- test on two snapshots with fake households from E3 ---
for day in [151, 459]:
    v = agent_api.snapshot(day)
    class FakeV: pass
    fv = FakeV(); fv.table=v.table; fv.day=v.day; fv.week=v.week
    fv.households = E3[E3.snapshot_day==day].household_key.values
    f = mkt_fn(fv, day)
    print(day, f.shape, 'NaN frac %.3f' % f.isna().mean().mean())
    print(f.describe().loc[['mean','50%']].round(3).to_string())