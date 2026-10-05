import agent_api, numpy as np, pandas as pd, re
t = agent_api.load_saved('e011_table.parquet')
print('e011 shape', t.shape)
cols = list(t.columns)
print('n cols', len(cols))
groups = {}
for c in cols:
    pre = re.split(r'[_0-9]', c)[0]
    groups.setdefault(pre, []).append(c)
for k in sorted(groups):
    print(k, len(groups[k]), groups[k][:10])
tt = agent_api.train_targets()
print('targets', tt.shape)
m = tt.merge(t, on=['household_key','snapshot_day'], how='left')
print('merged', m.shape, 'rows with NA:', int(m.isna().any(axis=1).sum()))
y = m['future_spend_4w']
print(y.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).round(2))
print('zero share', round(float((y==0).mean()),4))
feat_cols = [c for c in cols if c not in ('household_key','snapshot_day')]
num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(m[c])]
print('numeric', len(num_cols), 'nonnumeric', [c for c in feat_cols if c not in num_cols][:20])
cor = m[num_cols].apply(lambda s: s.corr(y)).dropna()
order = cor.abs().sort_values(ascending=False).index
print('--- top corr with target ---')
print(cor[order[:35]].round(3))
print('--- weakest ---')
print(cor[order[-12:]].round(3))
print(m.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median']).round(1))


# ---- cell ----
import agent_api, numpy as np, pandas as pd
for name in ['timing_v1','lvl_v1','churn_vol_v1','selfcal_v1']:
    try:
        t = agent_api.load_saved(name+'.parquet')
        print('==',name, t.shape)
        print(list(t.columns)[:40])
    except Exception as e:
        print(name, 'ERR', e)


# ---- cell ----
import agent_api, numpy as np, pandas as pd
t = agent_api.load_saved('e011_table.parquet')
tt = agent_api.train_targets()
m = tt.merge(t, on=['household_key','snapshot_day'], how='left')
print('rows', len(m), 'any NA rows', int(m.drop(columns=['future_spend_4w']).isna().any(axis=1).sum()))
na_frac = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).isna().mean().sort_values(ascending=False)
print(na_frac.head(15).round(3))
# offline ridge benchmark
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
va_days = [459,487,515,543]
X = m[feat_cols].astype(float).copy()
y = m['future_spend_4w'].values
is_va = m['snapshot_day'].isin(va_days).values
mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
Xs = ((X-mu)/sd).fillna(0).values
Xtr, ytr = Xs[~is_va], y[~is_va]
Xva, yva = Xs[is_va], y[is_va]
def ridge_fit(Xt, yt, lam):
    A = Xt.T@Xt + lam*np.eye(Xt.shape[1])
    return np.linalg.solve(A, Xt.T@yt)
for lam in [1,10,100,1000]:
    w = ridge_fit(Xtr,ytr,lam)
    pred = Xva@w
    print('lam',lam,'val MAE', round(float(np.abs(pred-yva).mean()),3))


# ---- cell ----
import agent_api, numpy as np, pandas as pd

tx0 = agent_api.snapshot().table('transactions')
print(tx0[['coupon_disc','coupon_match_disc','retail_disc','sales_value']].describe().round(2))
dm0 = agent_api.snapshot().table('display_mailer')
print(dm0.shape); print(dm0.head(3))
print('display vals', dm0['display'].value_counts(dropna=False).head(8).to_dict())
print('mailer vals', dm0['mailer'].value_counts(dropna=False).head(8).to_dict())


# ---- cell ----
import agent_api, numpy as np, pandas as pd

def make_blocks(view, snapshot_day):
    day = view.day
    tx = view.table('transactions')
    red = view.table('coupon_redemptions')
    dm = view.table('display_mailer')
    hh = pd.Index(view.households, name='household_key')

    def win(df, w): return df[(df['day'] > day - w) & (df['day'] <= day)]

    out = pd.DataFrame(index=hh)

    # ---------- deal / coupon block ----------
    for w, tag in [(84,'84'), (364,'364')]:
        t = win(tx, w)
        g = t.groupby('household_key')
        spend = g['sales_value'].sum()
        rd = -g['retail_disc'].sum()
        cd = -(g['coupon_disc'].sum() + g['coupon_match_disc'].sum())
        deal_share = (rd / spend.replace(0, np.nan))
        coup_share = (cd / spend.replace(0, np.nan))
        line_frac = g.apply(lambda x: (x['retail_disc'] < 0).mean())
        deep = t.assign(dl=(t['retail_disc'] <= -0.5) * t['sales_value']).groupby('household_key')['dl'].sum()
        deep_share = deep / spend.replace(0, np.nan)
        coup_trips = t[t['coupon_disc'] < 0].groupby('household_key')['basket_id'].nunique()
        r = win(red, w).groupby('household_key').size()
        out[f'deal_share_{tag}'] = deal_share
        out[f'coup_share_{tag}'] = coup_share
        out[f'deal_line_frac_{tag}'] = line_frac
        out[f'deep_deal_share_{tag}'] = deep_share
        out[f'coup_trips_{tag}'] = coup_trips.reindex(hh).fillna(0)
        out[f'redemp_{tag}'] = r.reindex(hh).fillna(0)
    out['deal_trend'] = out['deal_share_84'] / out['deal_share_364'].replace(0, np.nan)

    # ---------- hazard / gap block ----------
    t364 = win(tx, 364)
    trip_days = t364.groupby('household_key')['day'].apply(lambda s: np.sort(s.unique()))
    def gapstats(s):
        if len(s) < 2: return pd.Series({'h_gap_mean': np.nan,'h_gap_med': np.nan,'h_gap_std': np.nan,
                                         'h_gap_max': np.nan,'h_gap_p90': np.nan,'h_n_gaps': 0,
                                         'h_gap_last3': np.nan,'h_gap_ratio': np.nan})
        d = np.diff(s)
        l3 = d[-3:] if len(d) >= 3 else d
        return pd.Series({'h_gap_mean': d.mean(),'h_gap_med': np.median(d),'h_gap_std': d.std(),
                          'h_gap_max': d.max(),'h_gap_p90': np.percentile(d,90),'h_n_gaps': len(d),
                          'h_gap_last3': l3.mean(),'h_gap_ratio': (l3.mean()/np.median(d)) if np.median(d)>0 else np.nan})
    gs = trip_days.apply(gapstats).unstack()
    for c in gs.columns: out[c] = gs[c]
    last_trip = t364.groupby('household_key')['day'].max()
    out['h_days_since'] = (day - last_trip).reindex(hh)
    out['h_hazard'] = out['h_days_since'] / out['h_gap_med'].replace(0, np.nan)
    t112 = win(tx, 112)
    def stretch(s, cap):
        s = np.sort(s.unique())
        if len(s) == 0: return cap
        full = np.arange(day-cap+1, day+1)
        mask = np.isin(full, s)
        # longest run of False
        best = cur = 0
        for mch in mask:
            cur = 0 if mch else cur+1
            best = max(best, cur)
        return best
    out['h_zero_stretch_112'] = t112.groupby('household_key')['day'].apply(lambda s: stretch(s,112)).reindex(hh).fillna(112)
    for w in [7,14]:
        tw = win(tx, w).groupby('household_key')['basket_id'].nunique()
        out[f'h_trips_{w}'] = tw.reindex(hh).fillna(0)
    out['h_active_frac_84'] = out['h_trips_14'].fillna(0)*0 + (t364[(t364['day']>day-84)].groupby('household_key')['day'].nunique()/84).reindex(hh).fillna(0)

    # ---------- display / mailer block ----------
    tdisp = win(tx, 364)
    weeks = set(range((day-364+8)//7, (day+8)//7 + 1))
    dms = dm[dm['week_no'].isin(weeks)][['product_id','store_id','week_no','display','mailer']]
    m = tdisp.merge(dms, on=['product_id','store_id','week_no'], how='left')
    m['disp'] = (m['display'].fillna(0) > 0)
    m['mail'] = (m['mailer'].fillna('0') != '0')
    m['mailAD'] = m['mailer'].fillna('0').isin(['A','D'])
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


# ---- cell ----
import agent_api, numpy as np, pandas as pd

def make_blocks(view, snapshot_day):
    day = view.day
    tx = view.table('transactions')
    red = view.table('coupon_redemptions')
    dm = view.table('display_mailer')
    hh = pd.Index(view.households, name='household_key')
    def win(df, w): return df[(df['day'] > day - w) & (df['day'] <= day)]
    out = pd.DataFrame(index=hh)

    # ---------- deal / coupon block ----------
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

    # ---------- hazard / gap block ----------
    t364 = win(tx, 364)
    def gapstats(s):
        s = np.sort(s.unique())
        if len(s) < 2:
            return pd.Series({'h_gap_mean': np.nan,'h_gap_med': np.nan,'h_gap_std': np.nan,
                              'h_gap_max': np.nan,'h_gap_p90': np.nan,'h_n_gaps': 0,
                              'h_gap_last3': np.nan,'h_gap_ratio': np.nan})
        d = np.diff(s)
        l3 = d[-3:] if len(d) >= 3 else d
        return pd.Series({'h_gap_mean': d.mean(),'h_gap_med': np.median(d),'h_gap_std': d.std(),
                          'h_gap_max': d.max(),'h_gap_p90': np.percentile(d,90),'h_n_gaps': len(d),
                          'h_gap_last3': l3.mean(),'h_gap_ratio': (l3.mean()/np.median(d)) if np.median(d)>0 else np.nan})
    gs = t364.groupby('household_key')['day'].apply(gapstats)
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

    # ---------- display / mailer block ----------
    tdisp = win(tx, 364)
    weeks = set(range((day-364+8)//7, (day+8)//7 + 1))
    dms = dm[dm['week_no'].isin(weeks)][['product_id','store_id','week_no','display','mailer']]
    m = tdisp.merge(dms, on=['product_id','store_id','week_no'], how='left')
    m['disp'] = (m['display'].fillna(0) > 0)
    m['mail'] = (m['mailer'].fillna('0') != '0')
    m['mailAD'] = m['mailer'].fillna('0').isin(['A','D'])
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


# ---- cell ----
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
    if isinstance(gs, pd.Series):
        gs = gs.unstack()
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
    dms = dm[dm['week_no'].isin(weeks)][['product_id','store_id','week_no','display','mailer']]
    m = tdisp.merge(dms, on=['product_id','store_id','week_no'], how='left')
    m['disp'] = (m['display'].fillna(0) > 0)
    m['mail'] = (m['mailer'].fillna('0') != '0')
    m['mailAD'] = m['mailer'].fillna('0').isin(['A','D'])
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


# ---- cell ----
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


# ---- cell ----
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
    dms['mailer'] = dms['mailer'].astype(str)
    dms['display'] = pd.to_numeric(dms['display'], errors='coerce')
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


# ---- cell ----
import agent_api, numpy as np, pandas as pd

base = agent_api.load_saved('e011_table.parquet')
tt = agent_api.train_targets()
blocks = {n: agent_api.load_saved(n+'.parquet') for n in ['deal_v1','hazard_v1','display_v1']}

def prep(df):
    return df.set_index(['household_key','snapshot_day'])

b = prep(base).join(prep(tt.set_index(['household_key','snapshot_day'])), how='inner')
inner_va = [375,403,431]
inner_tr = [95,123,151,179,207,235,263,291,319,347]

def eval_cols(feat_cols):
    X = b[feat_cols].astype(float)
    y = b['future_spend_4w'].values
    is_va = b.index.get_level_values('snapshot_day').isin(inner_va)
    mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
    Xs = ((X-mu)/sd).fillna(0).values
    Xtr, ytr = Xs[~is_va], y[~is_va]
    Xva, yva = Xs[is_va], y[is_va]
    w = np.linalg.solve(Xtr.T@Xtr + 50*np.eye(Xtr.shape[1]), Xtr.T@ytr)
    return float(np.abs(Xva@w - yva).mean())

base_cols = [c for c in base.columns if c not in ('household_key','snapshot_day')]
print('base MAE:', round(eval_cols(base_cols),3), len(base_cols))
for name, blk in blocks.items():
    bc = [c for c in blk.columns if c not in ('household_key','snapshot_day')]
    bb = b.join(prep(blk), how='left')
    X = bb[base_cols+bc].astype(float)
    y = bb['future_spend_4w'].values
    is_va = bb.index.get_level_values('snapshot_day').isin(inner_va)
    mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
    Xs = ((X-mu)/sd).fillna(0).values
    w = np.linalg.solve(Xs[~is_va].T@Xs[~is_va] + 50*np.eye(X.shape[1]), Xs[~is_va].T@y[~is_va])
    mae = float(np.abs(Xs[is_va]@w - y[is_va]).mean())
    print(f'{name}: MAE {mae:.3f} (+{len(bc)} cols)')

allb = b
for blk in blocks.values():
    allb = allb.join(prep(blk), how='left')
allcols = [c for c in allb.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = allb[allcols].astype(float)
y = allb['future_spend_4w'].values
is_va = allb.index.get_level_values('snapshot_day').isin(inner_va)
mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
Xs = ((X-mu)/sd).fillna(0).values
w = np.linalg.solve(Xs[~is_va].T@Xs[~is_va] + 50*np.eye(X.shape[1]), Xs[~is_va].T@y[~is_va])
print('ALL blocks:', round(float(np.abs(Xs[is_va]@w - y[is_va]).mean()),3), len(allcols))


# ---- cell ----
import agent_api, numpy as np, pandas as pd

def prep(df):
    df = df.copy()
    if 'household_key' not in df.columns:
        df = df.reset_index()
    return df.set_index(['household_key','snapshot_day'])

base = prep(agent_api.load_saved('e011_table.parquet'))
tt = agent_api.train_targets()
blocks = {n: prep(agent_api.load_saved(n+'.parquet')) for n in ['deal_v1','hazard_v1','display_v1']}
b = base.join(tt.set_index(['household_key','snapshot_day'])['future_spend_4w'], how='inner')
print('b', b.shape)
inner_va = [375,403,431]

def eval_cols(feat_cols, lam=50):
    X = b[feat_cols].astype(float)
    y = b['future_spend_4w'].values
    is_va = b.index.get_level_values('snapshot_day').isin(inner_va)
    mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
    Xs = ((X-mu)/sd).fillna(0).values
    Xtr, ytr = Xs[~is_va], y[~is_va]
    w = np.linalg.solve(Xtr.T@Xtr + lam*np.eye(Xtr.shape[1]), Xtr.T@ytr)
    return float(np.abs(Xva@w - y[is_va]).mean()) if False else float(np.abs(Xs[is_va]@w - y[is_va]).mean())

base_cols = [c for c in base.columns]
print('base MAE:', round(eval_cols(base_cols),3), len(base_cols))
for name, blk in blocks.items():
    bc = [c for c in blk.columns]
    bb = b.join(blk, how='left')
    X = bb[base_cols+bc].astype(float)
    y = bb['future_spend_4w'].values
    is_va = bb.index.get_level_values('snapshot_day').isin(inner_va)
    mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
    Xs = ((X-mu)/sd).fillna(0).values
    w = np.linalg.solve(Xs[~is_va].T@Xs[~is_va] + 50*np.eye(X.shape[1]), Xs[~is_va].T@y[~is_va])
    print(f'{name}: MAE {float(np.abs(Xs[is_va]@w - y[is_va]).mean()):.3f} (+{len(bc)})')

allb = b
for blk in blocks.values(): allb = allb.join(blk, how='left')
allcols = [c for c in allb.columns if c != 'future_spend_4w']
X = allb[allcols].astype(float)
y = allb['future_spend_4w'].values
is_va = allb.index.get_level_values('snapshot_day').isin(inner_va)
mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
Xs = ((X-mu)/sd).fillna(0).values
w = np.linalg.solve(Xs[~is_va].T@Xs[~is_va] + 50*np.eye(X.shape[1]), Xs[~is_va].T@y[~is_va])
print('ALL:', round(float(np.abs(Xs[is_va]@w - y[is_va]).mean()),3), len(allcols))


# ---- cell ----
import agent_api, numpy as np, pandas as pd

def prep(df):
    df = df.copy()
    if 'household_key' not in df.columns: df = df.reset_index()
    return df.set_index(['household_key','snapshot_day'])

base = prep(agent_api.load_saved('e011_table.parquet'))
tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])['future_spend_4w']
b = base.join(tt.rename('y'), how='inner')
print('b', b.shape, 'y nan', int(b['y'].isna().sum()))
print(b['y'].describe().round(1))
inner_va = [375,403,431]
is_va = b.index.get_level_values('snapshot_day').isin(inner_va)
print('n va', is_va.sum(), 'n tr', (~is_va).sum())
y = b['y'].values
print('mean-y train', y[~is_va].mean().round(1), 'val', y[is_va].mean().round(1))
print('MAE mean-pred:', round(float(np.abs(y[is_va]-y[~is_va].mean()).mean()),2))
for c in ['spend_84d','ew_spend_hl28','sc_f_ew_hl2','spend_28d']:
    mae = float(np.abs(b.loc[is_va, c].values - y[is_va]).mean())
    print('MAE', c, round(mae,2))
# ridge with fewer features, check lam
cols = [c for c in base.columns]
X = b[cols].astype(float)
mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
Xs = ((X-mu)/sd).fillna(0).values
for lam in [1,10,50,200,1000]:
    w = np.linalg.solve(Xs[~is_va].T@Xs[~is_va] + lam*np.eye(len(cols)), Xs[~is_va].T@y[~is_va])
    print('lam',lam,'inner MAE', round(float(np.abs(Xs[is_va]@w - y[is_va]).mean()),3))


# ---- cell ----
import agent_api, numpy as np, pandas as pd

def prep(df):
    df = df.copy()
    if 'household_key' not in df.columns: df = df.reset_index()
    return df.set_index(['household_key','snapshot_day'])

base = prep(agent_api.load_saved('e011_table.parquet'))
tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])['future_spend_4w']
b = base.join(tt.rename('y'), how='inner')
blocks = {n: prep(agent_api.load_saved(n+'.parquet')) for n in ['deal_v1','hazard_v1','display_v1']}
inner_va = [375,403,431]

def fit_eval(X, y, is_va, lam=200, clip=8.0):
    mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
    Xs = ((X-mu)/sd).clip(-clip, clip).fillna(0).values
    w = np.linalg.solve(Xs[~is_va].T@Xs[~is_va] + lam*np.eye(X.shape[1]), Xs[~is_va].T@y[~is_va])
    return float(np.abs(Xs[is_va]@w - y[is_va]).mean())

y = b['y'].values
is_va = b.index.get_level_values('snapshot_day').isin(inner_va).values
print('mean-pred MAE:', round(float(np.abs(y[is_va]-y[~is_va].mean()).mean()),2))
print('sanity sc_f_ew_hl2 ridge:', round(fit_eval(b[['sc_f_ew_hl2']].astype(float), y, is_va),2))
base_cols = [c for c in base.columns]
print('base ridge:', round(fit_eval(b[base_cols].astype(float), y, is_va),3))
for name, blk in blocks.items():
    bc = list(blk.columns)
    bb = b.join(blk, how='left')
    mae = fit_eval(bb[base_cols+bc].astype(float), y, is_va)
    print(f'{name}: {mae:.3f} (+{len(bc)})')
allb = b
for blk in blocks.values(): allb = allb.join(blk, how='left')
allcols = [c for c in allb.columns if c != 'y']
print('ALL:', round(fit_eval(allb[allcols].astype(float), y, is_va),3))


# ---- cell ----
import agent_api, numpy as np, pandas as pd

def prep(df):
    df = df.copy()
    if 'household_key' not in df.columns: df = df.reset_index()
    return df.set_index(['household_key','snapshot_day'])

base = prep(agent_api.load_saved('e011_table.parquet'))
tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])['future_spend_4w']
b = base.join(tt.rename('y'), how='inner')
blocks = {n: prep(agent_api.load_saved(n+'.parquet')) for n in ['deal_v1','hazard_v1','display_v1']}
inner_va = [375,403,431]

def fit_eval(X, y, is_va, lam=200, clip=8.0):
    mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
    Xs = ((X-mu)/sd).clip(-clip, clip).fillna(0).values
    w = np.linalg.solve(Xs[~is_va].T@Xs[~is_va] + lam*np.eye(X.shape[1]), Xs[~is_va].T@y[~is_va])
    return float(np.abs(Xs[is_va]@w - y[is_va]).mean())

y = b['y'].values
is_va = b.index.get_level_values('snapshot_day').isin(inner_va).values
print('mean-pred MAE:', round(float(np.abs(y[is_va]-y[~is_va].mean()).mean()),2))
print('sanity sc_f_ew_hl2 ridge:', round(fit_eval(b[['sc_f_ew_hl2']].astype(float), y, is_va),2))
base_cols = [c for c in base.columns]
print('base ridge:', round(fit_eval(b[base_cols].astype(float), y, is_va),3))
for name, blk in blocks.items():
    bc = list(blk.columns)
    bb = b.join(blk, how='left')
    mae = fit_eval(bb[base_cols+bc].astype(float), y, is_va)
    print(f'{name}: {mae:.3f} (+{len(bc)})')
allb = b
for blk in blocks.values(): allb = allb.join(blk, how='left')
allcols = [c for c in allb.columns if c != 'y']
print('ALL:', round(fit_eval(allb[allcols].astype(float), y, is_va),3))


# ---- cell ----
import agent_api, numpy as np, pandas as pd

def prep(df):
    df = df.copy()
    if 'household_key' not in df.columns: df = df.reset_index()
    return df.set_index(['household_key','snapshot_day'])

base = prep(agent_api.load_saved('e011_table.parquet'))
tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])['future_spend_4w']
b = base.join(tt.rename('y'), how='inner')
blocks = {n: prep(agent_api.load_saved(n+'.parquet')) for n in ['deal_v1','hazard_v1','display_v1']}
inner_va = [375,403,431]

def fit_eval(X, y, is_va, lam=200, clip=8.0):
    mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
    Xs = ((X-mu)/sd).clip(-clip, clip).fillna(0).values
    w = np.linalg.solve(Xs[~is_va].T@Xs[~is_va] + lam*np.eye(X.shape[1]), Xs[~is_va].T@y[~is_va])
    return float(np.abs(Xs[is_va]@w - y[is_va]).mean())

y = b['y'].values
is_va = np.isin(b.index.get_level_values('snapshot_day'), inner_va)
print('mean-pred MAE:', round(float(np.abs(y[is_va]-y[~is_va].mean()).mean()),2))
print('sanity sc_f_ew_hl2 ridge:', round(fit_eval(b[['sc_f_ew_hl2']].astype(float), y, is_va),2))
base_cols = [c for c in base.columns]
print('base ridge:', round(fit_eval(b[base_cols].astype(float), y, is_va),3))
for name, blk in blocks.items():
    bc = list(blk.columns)
    bb = b.join(blk, how='left')
    print(f'{name}: {fit_eval(bb[base_cols+bc].astype(float), y, is_va):.3f} (+{len(bc)})')
allb = b
for blk in blocks.values(): allb = allb.join(blk, how='left')
allcols = [c for c in allb.columns if c != 'y']
print('ALL:', round(fit_eval(allb[allcols].astype(float), y, is_va),3))
out = allb.reset_index()[['household_key','snapshot_day']+allcols]
p = agent_api.save_table(out, 'e017_union_v1')
print('saved', p, out.shape)
