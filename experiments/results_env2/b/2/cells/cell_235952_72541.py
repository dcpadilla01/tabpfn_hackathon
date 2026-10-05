import agent_api, numpy as np, pandas as pd

def make_blocks(view, snapshot_day):
    day = view.day
    tx = view.table('transactions')
    red = view.table('coupon_redemptions')
    dm = view.table('display_mailer')
    hh = pd.Index(view.households, name='household_key')
    def win(df, w): return df[(df['day'] > day - w) & (df['day'] <= day)]
    out = pd.DataFrame(index=hh)

    for w, tag in [(84,'84'), (364,'364')]:
        t = win(tx, w)
        g = t.groupby('household_key')
        spend = g['sales_value'].sum()
        rd = -g['retail_disc'].sum()
        cd = -(g['coupon_disc'].sum() + g['coupon_match_disc'].sum())
        out[f'deal_share_{tag}'] = rd / spend.replace(0, np.nan)
        out[f'coup_share_{tag}'] = cd / spend.replace(0, np.nan)
        out[f'deal_line_frac_{tag}'] = g.apply(lambda x: (x['retail_disc'] < 0).mean())
        deep = t.assign(dl=(t['retail_disc'] <= -0.5) * t['sales_value']).groupby('household_key')['dl'].sum()
        out[f'deep_deal_share_{tag}'] = deep / spend.replace(0, np.nan)
        out[f'coup_trips_{tag}'] = t[t['coupon_disc'] < 0].groupby('household_key')['basket_id'].nunique().reindex(hh).fillna(0)
        out[f'redemp_{tag}'] = win(red, w).groupby('household_key').size().reindex(hh).fillna(0)
    out['deal_trend'] = out['deal_share_84'] / out['deal_share_364'].replace(0, np.nan)

    t364 = win(tx, 364)
    def gapstats(s):
        s = np.sort(s.unique())
        if len(s) < 2:
            return {'h_gap_mean': np.nan,'h_gap_med': np.nan,'h_gap_std': np.nan,
                    'h_gap_max': np.nan,'h_gap_p90': np.nan,'h_n_gaps': 0,
                    'h_gap_last3': np.nan,'h_gap_ratio': np.nan}
        d = np.diff(s)
        l3 = d[-3:] if len(d) >= 3 else d
        return {'h_gap_mean': d.mean(),'h_gap_med': np.median(d),'h_gap_std': d.std(),
                'h_gap_max': d.max(),'h_gap_p90': np.percentile(d,90),'h_n_gaps': len(d),
                'h_gap_last3': l3.mean(),'h_gap_ratio': (l3.mean()/np.median(d)) if np.median(d)>0 else np.nan}
    gs = t364.groupby('household_key')['day'].apply(gapstats)
    if isinstance(gs, pd.Series): gs = gs.unstack()
    for c in gs.columns: out[c] = gs[c]
    out['h_days_since'] = (day - t364.groupby('household_key')['day'].max()).reindex(hh)
    out['h_hazard'] = out['h_days_since'] / out['h_gap_med'].replace(0, np.nan)
    def stretch(s, cap):
        s = np.sort(s.unique())
        if len(s) == 0: return cap
        mask = np.isin(np.arange(day-cap+1, day+1), s)
        best = cur = 0
        for mch in mask:
            cur = 0 if mch else cur+1
            best = max(best, cur)
        return best
    out['h_zero_stretch_112'] = win(tx,112).groupby('household_key')['day'].apply(lambda s: stretch(s,112)).reindex(hh).fillna(112)
    for w in [7,14]:
        out[f'h_trips_{w}'] = win(tx,w).groupby('household_key')['basket_id'].nunique().reindex(hh).fillna(0)
    out['h_active_frac_84'] = (t364[t364['day']>day-84].groupby('household_key')['day'].nunique()/84).reindex(hh).fillna(0)

    tdisp = win(tx, 364)
    weeks = set(range((day-364+8)//7, (day+8)//7 + 1))
    dms = dm[dm['week_no'].isin(weeks)][['product_id','store_id','week_no','display','mailer']].copy()
    dms['mailer'] = dms['mailer'].astype(str).replace('nan','0')
    m = tdisp.merge(dms, on=['product_id','store_id','week_no'], how='left')
    m['disp'] = (m['display'].fillna(0) > 0)
    m['mail'] = (m['mailer'].fillna('0') != '0')
    m['mailAD'] = m['mailer'].isin(['A','D'])
    for w, tag in [(84,'84'), (364,'364')]:
        mm = m[m['day'] > day - w]
        g = mm.groupby('household_key')
        sp = g['sales_value'].sum()
        out[f'disp_share_{tag}'] = (mm[mm['disp']].groupby('household_key')['sales_value'].sum() / sp.replace(0,np.nan))
        out[f'mail_share_{tag}'] = (mm[mm['mail']].groupby('household_key')['sales_value'].sum() / sp.replace(0,np.nan))
        out[f'mailAD_share_{tag}'] = (mm[mm['mailAD']].groupby('household_key')['sales_value'].sum() / sp.replace(0,np.nan))
        out[f'disp_line_frac_{tag}'] = g['disp'].mean()
    out['disp_trend'] = out['disp_share_84'] / out['disp_share_364'].replace(0,np.nan)
    out.index.name = 'household_key'
    return out

res = agent_api.build_features(make_blocks)
print(res.shape)
deal_cols = [c for c in res.columns if c.startswith(('deal_','coup_','deep_','redemp'))]
haz_cols = [c for c in res.columns if c.startswith(('h_','gap'))]
disp_cols = [c for c in res.columns if c.startswith(('disp_','mail'))]
print(len(deal_cols), deal_cols)
print(len(haz_cols), haz_cols)
print(len(disp_cols), disp_cols)
p1 = agent_api.save_table(res[['household_key','snapshot_day']+deal_cols], 'deal_v1')
p2 = agent_api.save_table(res[['household_key','snapshot_day']+haz_cols], 'hazard_v1')
p3 = agent_api.save_table(res[['household_key','snapshot_day']+disp_cols], 'display_v1')
print(p1, p2, p3)
