import numpy as np, pandas as pd

def extra_features(view, sd):
    tx = view.table('transactions')
    hh = view.households
    tx = tx[tx['household_key'].isin(hh)]
    out = pd.DataFrame(index=hh)

    b = tx.groupby(['household_key','basket_id']).agg(day=('day','first'), sv=('sales_value','sum'))
    b = b.reset_index()
    g = b.groupby('household_key')

    def win(days):
        m = b['day'] > sd - days
        return b[m]

    # --- momentum ratios (recent vs prior) ---
    def wsum(days, off=0):
        m = (b['day'] > sd - off - days) & (b['day'] <= sd - off)
        return b[m].groupby('household_key')['sv'].sum()
    s28 = wsum(28); s28p = wsum(28, 28); s56p = wsum(56, 28)
    out['mom_28_vs_prev28'] = s28 / (s28p + 1)
    out['mom_28_vs_prev56'] = s28 / (s56p + 1)
    t28 = win(28).groupby('household_key').size()
    t28p = b[(b['day']>sd-56)&(b['day']<=sd-28)].groupby('household_key').size()
    out['trips_mom_28'] = t28 / (t28p + 1)

    # --- weekly trend slope over 8 weeks (log spend on OLS slope) ---
    wk = b[(b['day']>sd-56)].copy()
    wk['w'] = ((sd - wk['day']) // 7)
    wsums = wk.groupby(['household_key','w'])['sv'].sum().unstack(fill_value=0.0)
    wsums = wsums.reindex(columns=range(8), fill_value=0.0)
    x = np.arange(8, dtype=float); xc = x - x.mean(); denom = (xc**2).sum()
    out['wk_slope'] = ((wsums - wsums.mean(axis=1).values[:,None]) * xc).sum(axis=1) / denom
    out['wk_slope_log'] = np.log1p(wsums).sub(np.log1p(wsums).mean(axis=1), axis=0).pipe(lambda d: (d*xc).sum(axis=1)/denom)

    # --- dormancy / gaps ---
    days_sorted = b.groupby('household_key')['day'].apply(lambda s: np.sort(s.values))
    def gap_stats(s):
        if len(s) < 2: return (np.nan, np.nan, np.nan)
        d = np.diff(s)
        return (d.mean(), d.std(), d.max())
    gs = days_sorted.apply(gap_stats)
    out['gap_mean'] = gs.map(lambda t: t[0]); out['gap_std'] = gs.map(lambda t: t[1]); out['gap_max'] = gs.map(lambda t: t[2])
    out['gap_cv'] = out['gap_std'] / (out['gap_mean'] + 1)
    last = b.groupby('household_key')['day'].max()
    out['idle_frac_84'] = 1 - b[(b['day']>sd-84)].groupby('household_key')['day'].nunique() / 84.0
    out['days_since_2nd_last'] = sd - days_sorted.map(lambda s: s[-2] if len(s)>=2 else -1)
    out['days_since_2nd_last'] = out['days_since_2nd_last'].replace(np.inf, np.nan)

    # --- interarrival CV on recent 84d trips ---
    r84 = b[(b['day']>sd-84)].groupby('household_key')['day'].apply(lambda s: np.sort(s.values))
    def icv(s):
        if len(s) < 3: return np.nan
        d = np.diff(s); m = d.mean()
        return d.std()/m if m>0 else np.nan
    out['trip_reg_84'] = r84.apply(icv)

    # --- discount engagement ---
    tx2 = tx.copy()
    tx2['disc'] = -(tx2['coupon_disc'] + tx2['coupon_match_disc'] + tx2['retail_disc']).clip(lower=0)
    tx2['disc_pos'] = tx2['disc'] > 0
    for d in (28, 84):
        w = tx2[tx2['day'] > sd - d]
        agg = w.groupby('household_key').agg(dsum=('disc','sum'), sv=('sales_value','sum'), dshare=('disc_pos','mean'))
        out[f'disc_amt_{d}'] = agg['dsum']; out[f'disc_ratio_{d}'] = agg['dsum']/(agg['sv']+1); out[f'disc_line_share_{d}'] = agg['dshare']
    # coupon redemption recency
    cr = view.table('coupon_redemptions')
    if cr is not None and len(cr):
        cr = cr[cr['household_key'].isin(hh)]
        out['days_since_coupon'] = sd - cr.groupby('household_key')['day'].max()
    else:
        out['days_since_coupon'] = np.nan

    # --- timing concentration: share of spend in top-2 weekdays (84d) ---
    w84 = b[b['day'] > sd-84].copy()
    w84['dow'] = w84['day'] % 7
    dow = w84.groupby(['household_key','dow'])['sv'].sum().unstack(fill_value=0.0)
    if dow.shape[1] > 0:
        srt = np.sort(dow.values, axis=1)[:, -2:].sum(axis=1)
        tot = dow.sum(axis=1).replace(0, np.nan)
        out['dow_top2_share_84'] = (srt/tot)
    # trans_time spread (84d)
    tt = tx2[tx2['day']>sd-84].groupby('household_key')['trans_time'].std()
    out['time_std_84'] = tt

    # --- quantity-vs-spend: unit price level ---
    q84 = tx2[tx2['day']>sd-84].groupby('household_key').agg(q=('quantity','sum'), sv=('sales_value','sum'), lines=('sales_value','size'))
    out['unit_price_84'] = q84['sv']/(q84['q']+1)
    out['lines_84'] = q84['lines']
    out['sv_per_line_84'] = q84['sv']/(q84['lines']+1)

    # --- forward-looking: campaigns active in NEXT 4 weeks (known start days) ---
    cp = view.table('campaigns')
    if cp is not None and len(cp):
        nxt = cp[(cp['start_day'] > sd) & (cp['start_day'] <= sd+28)]
        out['n_camp_start_next28'] = nxt.groupby('campaign').size().sum() and len(nxt)
        tgt = view.table('campaign_targets')
        tgt = tgt[tgt['household_key'].isin(hh)]
        m = tgt.merge(nxt[['campaign']], on='campaign', how='inner')
        out['n_target_next28'] = m.groupby('household_key').size()
    return out

base = agent_api.load_saved('e006_dynamics.parquet')
feats = agent_api.build_features(extra_features)
print('feats', feats.shape)
m = base.merge(feats.drop(columns=[c for c in feats.columns if c in ('household_key','snapshot_day') and c in base.columns]), on=['household_key','snapshot_day'], how='left')
print('merged', m.shape, 'nan cols', m.isna().mean().gt(0.9).sum())
path = agent_api.save_table(m, 'e009_momentum.parquet')
print(path)
