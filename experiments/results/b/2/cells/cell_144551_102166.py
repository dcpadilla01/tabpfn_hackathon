import numpy as np, pandas as pd
import agent_api

base = agent_api.load_saved('e004_temporal.parquet')
print('base', base.shape)
print(sorted(base.columns))

def fn(view, snapshot_day):
    sd = int(snapshot_day)
    hh = pd.Index(list(view.households), name='household_key')
    out = pd.DataFrame(index=hh)
    t = view.transactions
    if t is None or len(t) == 0:
        return out
    g = t.groupby('household_key')
    first_day = g['day'].min(); last_day = g['day'].max()
    life_total = g['sales_value'].sum()
    out['x_life_total'] = life_total.reindex(hh)
    out['x_life_tenure'] = (sd - first_day + 1).reindex(hh)
    out['x_life_active_days'] = g['day'].nunique().reindex(hh)
    out['x_life_active_weeks'] = g['week_no'].nunique().reindex(hh)
    out['x_life_rate_wk'] = out['x_life_total'] / (out['x_life_tenure'] / 7.0)
    out['x_life_nbaskets'] = g['basket_id'].nunique().reindex(hh)
    wk = t.groupby(['household_key', 'week_no'])['sales_value'].sum()
    ws = wk.groupby(level=0).agg(['mean', 'std'])
    out['x_life_cv'] = (ws['std'] / ws['mean']).reindex(hh)

    def win(lo, hi):
        m = (t.day > lo) & (t.day <= hi)
        return t.loc[m].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)

    s28 = win(sd - 28, sd); prev28 = win(sd - 56, sd - 28)
    ya4 = win(sd - 364, sd - 336); ya8 = win(sd - 364, sd - 308)
    out['x_s28'] = s28; out['x_prev28'] = prev28
    out['x_trend_ratio'] = s28 / (prev28 + 1.0)
    out['x_ya4w'] = ya4; out['x_ya8w'] = ya8
    out['x_ya_ratio'] = s28 / (ya4 + 1.0)

    d = t[['household_key', 'day']].drop_duplicates().sort_values(['household_key', 'day'])
    gap = d.groupby('household_key')['day'].diff()
    d2 = pd.DataFrame({'k': d['household_key'].values, 'gap': gap.values})
    gs = d2.groupby('k')['gap'].agg(['mean', 'median', 'max', 'std']).reindex(hh)
    out['x_gap_mean'] = gs['mean']; out['x_gap_med'] = gs['median']
    out['x_gap_max'] = gs['max']; out['x_gap_std'] = gs['std']
    out['x_recency'] = (sd - last_day).reindex(hh)

    bk = t.groupby(['household_key', 'basket_id']).agg(v=('sales_value', 'sum'), n=('sales_value', 'size'), q=('quantity', 'sum'))
    bs = bk.groupby(level=0).agg(bm=('v', 'mean'), bmed=('v', 'median'), bmax=('v', 'max'), bl=('n', 'mean'), bq=('q', 'mean')).reindex(hh)
    out['x_b_mean_val'] = bs['bm']; out['x_b_med_val'] = bs['bmed']; out['x_b_max_val'] = bs['bmax']
    out['x_b_mean_lines'] = bs['bl']; out['x_b_mean_qty'] = bs['bq']

    tt = t.loc[t.day > sd - 112]
    hour = tt['trans_time'].fillna(0) // 100
    tmp = pd.DataFrame({'k': tt['household_key'].values, 'hour': hour.values, 'mor': (hour.values < 12).astype(float)})
    out['x_hour_mean'] = tmp.groupby('k')['hour'].mean().reindex(hh)
    out['x_morning_share'] = tmp.groupby('k')['mor'].mean().reindex(hh)

    t84 = t.loc[t.day > sd - 84]
    sv84 = t84.groupby('household_key')['sales_value'].sum()
    out['x_stores_84'] = t84.groupby('household_key')['store_id'].nunique().reindex(hh)
    st_top = t84.groupby(['household_key', 'store_id'])['sales_value'].sum().groupby(level=0).max().reindex(hh)
    out['x_topstore_share_84'] = st_top / sv84.where(sv84 > 0)
    disc = (t84['coupon_disc'].fillna(0) + t84['coupon_match_disc'].fillna(0) + t84['retail_disc'].fillna(0))
    dsum = disc.groupby(t84['household_key']).sum().reindex(hh)
    out['x_disc_share_84'] = (dsum / (sv84.reindex(hh) + 1.0)).clip(0, 5)

    w0 = (sd + 8) // 7
    last12 = t[t.week_no > w0 - 12]
    out['x_active_weeks_12w'] = last12.groupby('household_key')['week_no'].nunique().reindex(hh)

    cr = view.coupon_redemptions
    if cr is not None and len(cr) > 0:
        out['x_red_total'] = cr.groupby('household_key').size().reindex(hh)
        out['x_red_84'] = cr[cr.day > sd - 84].groupby('household_key').size().reindex(hh)
    else:
        out['x_red_total'] = 0.0; out['x_red_84'] = 0.0
    return out

feats = agent_api.build_features(fn)
print('feats', feats.shape)
overlap = (set(base.columns) & set(feats.columns)) - {'household_key', 'snapshot_day'}
print('overlap', overlap)
if overlap:
    feats = feats.rename(columns={c: c + '_n' for c in overlap})
merged = base.merge(feats, on=['household_key', 'snapshot_day'], how='inner')
print('merged', merged.shape)

tgt = agent_api.train_targets()
dg = merged.merge(tgt, on=['household_key', 'snapshot_day'], how='inner')
print('diag rows', len(dg))
xcols = [c for c in merged.columns if c.startswith('x_')]
cors = []
for c in xcols:
    s = dg[c]
    if s.notna().sum() > 50 and s.nunique() > 2:
        cors.append((c, round(s.corr(dg['future_spend_4w'], method='spearman'), 3), int(s.notna().sum())))
cors.sort(key=lambda r: -abs(r[1]))
print('top |spearman| with target:')
for r in cors:
    print(r)

path = agent_api.save_table(merged, 'e005_longrun.parquet')
print('SAVED', path)