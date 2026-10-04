import pandas as pd, numpy as np
import agent_api as A

def build(view, sd):
    tx = view.transactions
    need = pd.Index(list(view.households), name='household_key')
    g = tx.groupby(['household_key','day'], as_index=False).agg(spend=('sales_value','sum'))
    rows = {}
    # macro (snapshot-level) features
    hh28 = g[g.day > sd-28].groupby('household_key')['spend'].sum()
    act7 = g[g.day > sd-7]['household_key'].nunique(); act14 = g[g.day > sd-14]['household_key'].nunique()
    n_hh = g['household_key'].nunique()
    macro = dict(m_act7=act7/max(n_hh,1), m_act14=act14/max(n_hh,1),
                 m_spend28_mean=float(hh28.mean()), m_spend28_med=float(hh28.median()))
    for hh, sub in g.groupby('household_key', sort=False):
        days = sub['day'].to_numpy(float); sp = sub['spend'].to_numpy(float)
        n = len(days); last = days[-1]; since = sd - last
        gaps = np.diff(days) if n > 1 else np.array([np.nan])
        gm84 = np.nanmean(gaps[days[1:] > sd-84]) if np.any(days[1:] > sd-84) else np.nan
        d = {}
        d['overdue_days'] = since - gm84 if not np.isnan(gm84) else np.nan
        d['overdue_ratio'] = since/(gm84+1) if not np.isnan(gm84) else np.nan
        d['gap_med_full'] = np.nanmedian(gaps); d['gap_p90_full'] = np.nanpercentile(gaps,90)
        d['over_p90'] = since/(d['gap_p90_full']+1)
        d['gap_cv_full'] = np.nanstd(gaps)/(np.nanmean(gaps)+1e-9)
        d['gap21_frac_full'] = float(np.nanmean(gaps > 21)) if n > 2 else np.nan
        gl = gaps[-5:]; gl = gl[~np.isnan(gl)]
        d['gap_last5_mean'] = gl.mean() if len(gl) else np.nan
        d['gap_last5_std'] = gl.std() if len(gl) > 1 else np.nan
        # ewm of gaps halflife 5 (trip-index decay approx by sequence)
        s = np.nan
        for x in gaps:
            s = x if np.isnan(s) else 0.5*s + 0.5*x
        d['gap_ewm5'] = s
        d['rhythm_break'] = gaps[-1]/(s+1) if n > 1 else np.nan
        d['next_trip_eta'] = s - since if not np.isnan(s) else np.nan
        d['exp_trips28'] = 28/(np.nanmean(gaps)+1) if n > 1 else np.nan
        # zero 28d-windows (aligned backwards)
        z13 = 0; z26 = 0; nk26 = 0
        for k in range(1, 27):
            lo, hi = sd-28*k, sd-28*(k-1)
            m = (days > lo) & (days <= hi)
            z = 1.0 if sp[m].sum() == 0 else 0.0
            if k <= 13: z13 += z
            if days[0] <= lo:  # window fully inside tenure
                z26 += z; nk26 += 1
        d['n_zero_l13'] = z13; d['zero_frac_l13'] = z13/13.0
        d['zero_frac_l26'] = z26/max(nk26,1) if nk26 else np.nan
        # active weeks
        for w, lab in ((28,'w4'),(56,'w8'),(84,'w13')):
            dw = days[days > sd-w]
            d['act_'+lab] = len(np.unique((dw+8)//7)) if len(dw) else 0
        # day-decay ewm spend
        def ewm_d(h, vals=None):
            s = 0.0; prev = None
            for t, v in zip(days, sp if vals is None else vals):
                s = v if prev is None else s*0.5**((t-prev)/h) + v
                prev = t
            return s*0.5**((sd-prev)/h) if prev is not None else np.nan
        e4 = ewm_d(4); e13 = ewm_d(13)
        d['ewm4_d'] = e4; d['ewm13_d'] = e13; d['ewm_ratio'] = e4/(e13+1)
        d['n_ewm14'] = ewm_d(14, vals=np.ones(n))
        # burst / trend
        s7 = sp[days > sd-7].sum(); s14 = sp[days > sd-14].sum(); s84 = sp[days > sd-84].sum()
        m = (days > sd-42) & (days <= sd-14)
        d['burst7'] = s7/(s84+1); d['trend14_42'] = s14/(sp[m].sum()+1)
        d['dow_last'] = last % 7; d['dow_sd'] = sd % 7
        rows[hh] = d
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.name = 'household_key'
    for k, v in macro.items(): df[k] = v
    df['dow_sd'] = sd % 7
    df = df.reindex(need)
    return df

tt = A.train_targets()
new = A.build_features(build)
print('built', new.shape)
newcols = [c for c in new.columns if c not in ('household_key','snapshot_day')]
m = new.reset_index().merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day.isin(A.snapshot_days()['train'])]
print('NaN rates (train):')
print(new[newcols].isna().mean().round(3).to_string())
cor = tr[newcols].corrwith(tr.future_spend_4w).round(3)
print('corr with target (train):'); print(cor.to_string())
base = A.load_saved('e016_smoothed.parquet')
full = base.merge(new.reset_index(), on=['household_key','snapshot_day'], how='inner')
print('full', full.shape, 'rows match base:', len(full)==len(base))
path = A.save_table(full, 'e018_timing_hazard')
print(path)