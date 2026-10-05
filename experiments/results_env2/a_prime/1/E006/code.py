
import numpy as np, pandas as pd

base = load_saved('e004_seasonal.parquet')
print('base shape:', base.shape)
print('base cols:', list(base.columns)[:40], '...total', base.shape[1])
tt = train_targets()
print(tt['future_spend_4w'].describe())

def make_dyn(view, sd):
    tx = view.table("transactions")
    hh = pd.Index(view.households)
    tx = tx[tx.household_key.isin(hh)].copy()
    feats = pd.DataFrame(index=hh)
    age = sd - tx['day']
    # recency-decayed spend levels
    for hl in (14, 28, 56, 112):
        w = np.power(0.5, age / hl)
        feats['ewma_spend_hl%d' % hl] = (tx['sales_value'] * w).groupby(tx['household_key']).sum().reindex(hh, fill_value=0.0)
    # very short windows
    for win in (7, 14):
        sub = tx[tx['day'] > sd - win]
        feats['spend_%dd' % win] = sub.groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
        feats['trips_%dd' % win] = sub.groupby('household_key')['basket_id'].nunique().reindex(hh, fill_value=0.0)
    # weekly spend stats over last 12 weeks
    tx['week'] = tx['day'] // 7
    wk_now = sd // 7
    w12 = tx[tx['week'] >= wk_now - 11]
    ws = w12.groupby(['household_key', 'week'])['sales_value'].sum()
    agg = ws.groupby('household_key').agg(['mean', 'std', 'count'])
    feats['wk_spend_mean_12w'] = agg['mean'].reindex(hh, fill_value=0.0)
    feats['wk_spend_std_12w'] = agg['std'].reindex(hh).fillna(0.0)
    feats['wk_active_12w'] = agg['count'].reindex(hh, fill_value=0.0)
    # OLS slope of weekly spend, 12w and 24w (vectorized)
    for label, wdf, k in (('12w', w12, 11), ('24w', tx[tx['week'] >= wk_now - 23], 23)):
        trel = wdf['week'] - (wk_now - k); hhv = wdf['household_key']; y = wdf['sales_value']
        nn = trel.groupby(hhv).count(); st = trel.groupby(hhv).sum()
        sy = y.groupby(hhv).sum(); sty = (trel * y).groupby(hhv).sum(); st2 = (trel * trel).groupby(hhv).sum()
        den = (nn * st2 - st * st)
        slope = ((nn * sty - st * sy) / den.replace(0, np.nan)).fillna(0.0)
        feats['wk_spend_slope_' + label] = slope.reindex(hh, fill_value=0.0)
    # trailing zero-spend week streak (last 10 weeks)
    piv = tx.groupby(['household_key', 'week'])['sales_value'].sum().unstack(fill_value=0.0)
    cols = [wk_now - 9 + i for i in range(10)]
    sub = piv.reindex(columns=cols, fill_value=0.0).reindex(hh, fill_value=0.0)
    z = (sub.values == 0).astype(float)
    run = np.zeros(len(sub))
    for j in range(z.shape[1]):
        run = (run + 1.0) * z[:, j]
    feats['zero_wk_streak_10w'] = pd.Series(run, index=sub.index).reindex(hh, fill_value=0.0)
    # inter-purchase gap stats (84d)
    tx84 = tx[tx['day'] > sd - 84]
    def gap_stats(days):
        d = np.sort(pd.unique(days))
        if len(d) < 2:
            return pd.Series({'gap_mean_84d': np.nan, 'gap_std_84d': np.nan, 'gap_max_84d': np.nan})
        gp = np.diff(d).astype(float)
        return pd.Series({'gap_mean_84d': gp.mean(), 'gap_std_84d': gp.std(), 'gap_max_84d': gp.max()})
    gs = tx84.groupby('household_key')['day'].unique().apply(gap_stats)
    for c in ('gap_mean_84d', 'gap_std_84d', 'gap_max_84d'):
        feats[c] = gs[c].reindex(hh) if (len(gs) and c in getattr(gs, 'columns', [])) else np.nan
    # deal-seeking
    tx56 = tx[tx['day'] > sd - 56]
    sp56 = tx56.groupby('household_key')['sales_value'].sum()
    for col in ('retail_disc', 'coupon_disc', 'coupon_match_disc'):
        dsum = tx56[tx56[col] != 0].groupby('household_key')['sales_value'].sum()
        feats['dealshare_%s_56d' % col] = (dsum / sp56.replace(0, np.nan)).fillna(0.0).reindex(hh, fill_value=0.0)
    tx28 = tx[tx['day'] > sd - 28]
    d28 = tx28.groupby('household_key')[['retail_disc', 'coupon_disc', 'coupon_match_disc']].sum().sum(axis=1)
    feats['disc_amt_28d'] = d28.reindex(hh, fill_value=0.0)
    # day-of-week spend shares (182d)
    tx182 = tx[tx['day'] > sd - 182].copy()
    tx182['dow'] = tx182['day'] % 7
    dsp = tx182.groupby(['household_key', 'dow'])['sales_value'].sum().unstack(fill_value=0.0)
    dsp = dsp.reindex(columns=range(7), fill_value=0.0).reindex(hh, fill_value=0.0)
    tot = dsp.sum(axis=1).replace(0, np.nan)
    for d in range(7):
        feats['dow%d_share_182d' % d] = (dsp[d] / tot).fillna(0.0)
    # shopping time-of-day habits
    hr = (tx56['trans_time'] // 100)
    feats['shop_hour_mean_56d'] = hr.groupby(tx56['household_key']).mean().reindex(hh)
    feats['shop_hour_std_56d'] = hr.groupby(tx56['household_key']).std().reindex(hh)
    # variety / units / unit price
    feats['n_products_28d'] = tx28.groupby('household_key')['product_id'].nunique().reindex(hh, fill_value=0)
    feats['units_28d'] = tx28.groupby('household_key')['quantity'].sum().reindex(hh, fill_value=0.0)
    q56 = tx56.groupby('household_key')[['sales_value', 'quantity']].sum()
    feats['unit_price_56d'] = (q56['sales_value'] / q56['quantity'].replace(0, np.nan)).reindex(hh)
    # 28d-block level stats (all history)
    blk = tx.groupby(['household_key', tx['day'] // 28])['sales_value'].sum().unstack(fill_value=0.0)
    cb = sd // 28
    blk = blk.reindex(columns=range(cb + 1), fill_value=0.0).reindex(hh, fill_value=0.0)
    feats['block_max'] = blk.max(axis=1)
    feats['block_std'] = blk.std(axis=1).fillna(0.0)
    return feats

dyn = build_features(make_dyn)
print('dyn shape:', dyn.shape)
dyn.columns = ['d_' + c for c in dyn.columns]
m = base.merge(dyn, on=['household_key', 'snapshot_day'], how='inner')
print('merged shape:', m.shape, 'dupcols:', m.columns.duplicated().any())
assert len(m) == len(base)
print(m.filter(like='d_').describe().T[['mean', 'std', 'min', 'max']].round(2).to_string())
p = save_table(m, 'e006_dynamics')
print('saved:', p)


# ---- cell ----

import numpy as np, pandas as pd

base = load_saved('e004_seasonal.parquet')

def make_dyn(view, sd):
    tx = view.table("transactions")
    hh = pd.Index(view.households)
    tx = tx[tx.household_key.isin(hh)].copy()
    feats = pd.DataFrame(index=hh)
    age = sd - tx['day']
    for hl in (14, 28, 56, 112):
        w = np.power(0.5, age / hl)
        feats['ewma_spend_hl%d' % hl] = (tx['sales_value'] * w).groupby(tx['household_key']).sum().reindex(hh, fill_value=0.0)
    for win in (7, 14):
        sub = tx[tx['day'] > sd - win]
        feats['spend_%dd' % win] = sub.groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
        feats['trips_%dd' % win] = sub.groupby('household_key')['basket_id'].nunique().reindex(hh, fill_value=0.0)
    tx['week'] = tx['day'] // 7
    wk_now = sd // 7
    w12 = tx[tx['week'] >= wk_now - 11]
    ws = w12.groupby(['household_key', 'week'])['sales_value'].sum()
    agg = ws.groupby('household_key').agg(['mean', 'std', 'count'])
    feats['wk_spend_mean_12w'] = agg['mean'].reindex(hh, fill_value=0.0)
    feats['wk_spend_std_12w'] = agg['std'].reindex(hh).fillna(0.0)
    feats['wk_active_12w'] = agg['count'].reindex(hh, fill_value=0.0)
    for label, wdf, k in (('12w', w12, 11), ('24w', tx[tx['week'] >= wk_now - 23], 23)):
        trel = wdf['week'] - (wk_now - k); hhv = wdf['household_key']; y = wdf['sales_value']
        nn = trel.groupby(hhv).count(); st = trel.groupby(hhv).sum()
        sy = y.groupby(hhv).sum(); sty = (trel * y).groupby(hhv).sum(); st2 = (trel * trel).groupby(hhv).sum()
        den = (nn * st2 - st * st).replace(0, np.nan)
        slope = ((nn * sty - st * sy) / den).fillna(0.0)
        feats['wk_spend_slope_' + label] = slope.reindex(hh, fill_value=0.0)
    piv = tx.groupby(['household_key', 'week'])['sales_value'].sum().unstack(fill_value=0.0)
    cols = [wk_now - 9 + i for i in range(10)]
    sub = piv.reindex(columns=cols, fill_value=0.0).reindex(hh, fill_value=0.0)
    z = (sub.values == 0).astype(float)
    run = np.zeros(len(sub))
    for j in range(z.shape[1]):
        run = (run + 1.0) * z[:, j]
    feats['zero_wk_streak_10w'] = pd.Series(run, index=sub.index).reindex(hh, fill_value=0.0)
    tx84 = tx[tx['day'] > sd - 84]
    def gap_stats(days):
        d = np.sort(pd.unique(days))
        if len(d) < 2:
            return pd.Series({'gap_mean_84d': np.nan, 'gap_std_84d': np.nan, 'gap_max_84d': np.nan})
        gp = np.diff(d).astype(float)
        return pd.Series({'gap_mean_84d': gp.mean(), 'gap_std_84d': gp.std(), 'gap_max_84d': gp.max()})
    gs = tx84.groupby('household_key')['day'].unique().apply(gap_stats)
    for c in ('gap_mean_84d', 'gap_std_84d', 'gap_max_84d'):
        feats[c] = gs[c].reindex(hh) if (len(gs) and c in getattr(gs, 'columns', [])) else np.nan
    tx56 = tx[tx['day'] > sd - 56]
    sp56 = tx56.groupby('household_key')['sales_value'].sum()
    for col in ('retail_disc', 'coupon_disc', 'coupon_match_disc'):
        dsum = tx56[tx56[col] != 0].groupby('household_key')['sales_value'].sum()
        feats['dealshare_%s_56d' % col] = (dsum / sp56.replace(0, np.nan)).fillna(0.0).reindex(hh, fill_value=0.0)
    tx28 = tx[tx['day'] > sd - 28]
    d28 = tx28.groupby('household_key')[['retail_disc', 'coupon_disc', 'coupon_match_disc']].sum().sum(axis=1)
    feats['disc_amt_28d'] = d28.reindex(hh, fill_value=0.0)
    tx182 = tx[tx['day'] > sd - 182].copy()
    tx182['dow'] = tx182['day'] % 7
    dsp = tx182.groupby(['household_key', 'dow'])['sales_value'].sum().unstack(fill_value=0.0)
    dsp = dsp.reindex(columns=range(7), fill_value=0.0).reindex(hh, fill_value=0.0)
    tot = dsp.sum(axis=1).replace(0, np.nan)
    for d in range(7):
        feats['dow%d_share_182d' % d] = (dsp[d] / tot).fillna(0.0)
    hr = tx56['trans_time'] // 100
    feats['shop_hour_mean_56d'] = hr.groupby(tx56['household_key']).mean().reindex(hh)
    feats['shop_hour_std_56d'] = hr.groupby(tx56['household_key']).std().reindex(hh)
    feats['n_products_28d'] = tx28.groupby('household_key')['product_id'].nunique().reindex(hh, fill_value=0)
    feats['units_28d'] = tx28.groupby('household_key')['quantity'].sum().reindex(hh, fill_value=0.0)
    q56 = tx56.groupby('household_key')[['sales_value', 'quantity']].sum()
    feats['unit_price_56d'] = (q56['sales_value'] / q56['quantity'].replace(0, np.nan)).reindex(hh)
    blk = tx.groupby(['household_key', tx['day'] // 28])['sales_value'].sum().unstack(fill_value=0.0)
    cb = sd // 28
    blk = blk.reindex(columns=range(cb + 1), fill_value=0.0).reindex(hh, fill_value=0.0)
    feats['block_max'] = blk.max(axis=1)
    feats['block_std'] = blk.std(axis=1).fillna(0.0)
    return feats

dyn = build_features(make_dyn)
dyn = dyn.reset_index().rename(columns={'index': 'household_key'})
dyn.columns = ['d_' + c if c not in ('household_key', 'snapshot_day') else c for c in dyn.columns]
m = base.merge(dyn, on=['household_key', 'snapshot_day'], how='inner')
print('merged shape:', m.shape, 'dups:', m.columns.duplicated().any())
assert len(m) == len(base)
print(m.filter(like='d_').describe().T[['mean', 'std', 'min', 'max']].round(2).to_string())
p = save_table(m, 'e006_dynamics')
print('saved:', p)


# ---- cell ----

import pandas as pd
def probe(view, sd):
    f = pd.DataFrame({'x': 1.0}, index=pd.Index(view.households, name='household_key'))
    return f
out = build_features(probe)
print(type(out), out.shape)
print(out.columns.tolist())
print(out.head(3))
print('index name:', out.index.name)


# ---- cell ----

import numpy as np, pandas as pd

base = load_saved('e004_seasonal.parquet')

def make_dyn(view, sd):
    tx = view.table("transactions")
    hh = pd.Index(view.households)
    tx = tx[tx.household_key.isin(hh)].copy()
    feats = pd.DataFrame(index=hh)
    age = sd - tx['day']
    for hl in (14, 28, 56, 112):
        w = np.power(0.5, age / hl)
        feats['ewma_spend_hl%d' % hl] = (tx['sales_value'] * w).groupby(tx['household_key']).sum().reindex(hh, fill_value=0.0)
    for win in (7, 14):
        sub = tx[tx['day'] > sd - win]
        feats['spend_%dd' % win] = sub.groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
        feats['trips_%dd' % win] = sub.groupby('household_key')['basket_id'].nunique().reindex(hh, fill_value=0.0)
    tx['week'] = tx['day'] // 7
    wk_now = sd // 7
    w12 = tx[tx['week'] >= wk_now - 11]
    ws = w12.groupby(['household_key', 'week'])['sales_value'].sum()
    agg = ws.groupby('household_key').agg(['mean', 'std', 'count'])
    feats['wk_spend_mean_12w'] = agg['mean'].reindex(hh, fill_value=0.0)
    feats['wk_spend_std_12w'] = agg['std'].reindex(hh).fillna(0.0)
    feats['wk_active_12w'] = agg['count'].reindex(hh, fill_value=0.0)
    for label, wdf, k in (('12w', w12, 11), ('24w', tx[tx['week'] >= wk_now - 23], 23)):
        trel = wdf['week'] - (wk_now - k); hhv = wdf['household_key']; y = wdf['sales_value']
        nn = trel.groupby(hhv).count(); st = trel.groupby(hhv).sum()
        sy = y.groupby(hhv).sum(); sty = (trel * y).groupby(hhv).sum(); st2 = (trel * trel).groupby(hhv).sum()
        den = (nn * st2 - st * st).replace(0, np.nan)
        slope = ((nn * sty - st * sy) / den).fillna(0.0)
        feats['wk_spend_slope_' + label] = slope.reindex(hh, fill_value=0.0)
    piv = tx.groupby(['household_key', 'week'])['sales_value'].sum().unstack(fill_value=0.0)
    cols = [wk_now - 9 + i for i in range(10)]
    sub = piv.reindex(columns=cols, fill_value=0.0).reindex(hh, fill_value=0.0)
    z = (sub.values == 0).astype(float)
    run = np.zeros(len(sub))
    for j in range(z.shape[1]):
        run = (run + 1.0) * z[:, j]
    feats['zero_wk_streak_10w'] = pd.Series(run, index=sub.index).reindex(hh, fill_value=0.0)
    tx84 = tx[tx['day'] > sd - 84]
    def gap_stats(days):
        d = np.sort(pd.unique(days))
        if len(d) < 2:
            return pd.Series({'gap_mean_84d': np.nan, 'gap_std_84d': np.nan, 'gap_max_84d': np.nan})
        gp = np.diff(d).astype(float)
        return pd.Series({'gap_mean_84d': gp.mean(), 'gap_std_84d': gp.std(), 'gap_max_84d': gp.max()})
    gs = tx84.groupby('household_key')['day'].unique().apply(gap_stats)
    for c in ('gap_mean_84d', 'gap_std_84d', 'gap_max_84d'):
        feats[c] = gs[c].reindex(hh) if (len(gs) and c in getattr(gs, 'columns', [])) else np.nan
    tx56 = tx[tx['day'] > sd - 56]
    sp56 = tx56.groupby('household_key')['sales_value'].sum()
    for col in ('retail_disc', 'coupon_disc', 'coupon_match_disc'):
        dsum = tx56[tx56[col] != 0].groupby('household_key')['sales_value'].sum()
        feats['dealshare_%s_56d' % col] = (dsum / sp56.replace(0, np.nan)).fillna(0.0).reindex(hh, fill_value=0.0)
    tx28 = tx[tx['day'] > sd - 28]
    d28 = tx28.groupby('household_key')[['retail_disc', 'coupon_disc', 'coupon_match_disc']].sum().sum(axis=1)
    feats['disc_amt_28d'] = d28.reindex(hh, fill_value=0.0)
    tx182 = tx[tx['day'] > sd - 182].copy()
    tx182['dow'] = tx182['day'] % 7
    dsp = tx182.groupby(['household_key', 'dow'])['sales_value'].sum().unstack(fill_value=0.0)
    dsp = dsp.reindex(columns=range(7), fill_value=0.0).reindex(hh, fill_value=0.0)
    tot = dsp.sum(axis=1).replace(0, np.nan)
    for d in range(7):
        feats['dow%d_share_182d' % d] = (dsp[d] / tot).fillna(0.0)
    hr = tx56['trans_time'] // 100
    feats['shop_hour_mean_56d'] = hr.groupby(tx56['household_key']).mean().reindex(hh)
    feats['shop_hour_std_56d'] = hr.groupby(tx56['household_key']).std().reindex(hh)
    feats['n_products_28d'] = tx28.groupby('household_key')['product_id'].nunique().reindex(hh, fill_value=0)
    feats['units_28d'] = tx28.groupby('household_key')['quantity'].sum().reindex(hh, fill_value=0.0)
    q56 = tx56.groupby('household_key')[['sales_value', 'quantity']].sum()
    feats['unit_price_56d'] = (q56['sales_value'] / q56['quantity'].replace(0, np.nan)).reindex(hh)
    blk = tx.groupby(['household_key', tx['day'] // 28])['sales_value'].sum().unstack(fill_value=0.0)
    cb = sd // 28
    blk = blk.reindex(columns=range(cb + 1), fill_value=0.0).reindex(hh, fill_value=0.0)
    feats['block_max'] = blk.max(axis=1)
    feats['block_std'] = blk.std(axis=1).fillna(0.0)
    return feats

dyn = build_features(make_dyn)
dyn.columns = ['d_' + c if c not in ('household_key', 'snapshot_day') else c for c in dyn.columns]
m = base.merge(dyn, on=['household_key', 'snapshot_day'], how='inner')
print('merged shape:', m.shape, 'dups:', m.columns.duplicated().any())
assert len(m) == len(base)
print(m.filter(like='d_').describe().T[['mean', 'std', 'min', 'max']].round(2).to_string())
p = save_table(m, 'e006_dynamics')
print('saved:', p)
