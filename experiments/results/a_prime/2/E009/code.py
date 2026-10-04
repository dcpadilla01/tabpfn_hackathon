import numpy as np, pandas as pd
t = agent_api.load_saved('e008_main_plus_marketing.parquet')
tt = agent_api.train_targets()
print('feat shape', t.shape)
print('cols:', sorted(t.columns))
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df['future_spend_4w'].astype(float)
print('train rows', len(df), 'zero share %.3f' % (y==0).mean())
print(y.describe(percentiles=[.25,.5,.75,.9,.95]))
print('by snapshot:')
print(df.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median']))
num = df.select_dtypes(include=[np.number,'bool']).drop(columns=['snapshot_day','future_spend_4w'], errors='ignore').astype(float)
sp = num.corrwith(y, method='spearman').sort_values()
print('TOP corr:'); print(sp.tail(22).round(3))
print('BOTTOM:'); print(sp.head(6).round(3))

cats = [c for c in df.columns if str(df[c].dtype) not in ('float64','int64','bool','float32','int32','uint8','int8') and c not in ('household_key',)]
print('categorical cols:', cats)
parts = [num.values]
for c in cats:
    parts.append(pd.get_dummies(df[c].astype('category'), dummy_na=True).values.astype(float))
X = np.hstack(parts)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
tr_mask = (df['snapshot_day'] <= 403).values
va_mask = (df['snapshot_day'] == 431).values
mu, sd = X[tr_mask].mean(0), X[tr_mask].std(0)+1e-9
Xs = (X-mu)/sd
Xs = np.hstack([Xs, np.ones((len(Xs),1))])
def ridge_fit(Xtr, ytr, lam):
    A = Xtr.T@Xtr + lam*np.eye(Xtr.shape[1]); A[-1,-1] -= lam
    return np.linalg.solve(A, Xtr.T@ytr)
yv = y.values
for lam in [1.0, 10.0, 100.0, 1000.0]:
    w = ridge_fit(Xs[tr_mask], yv[tr_mask], lam)
    pred = np.clip(Xs[va_mask]@w, 0, None)
    print('ridge lam=%g holdout431 MAE %.2f' % (lam, np.abs(pred-yv[va_mask]).mean()))
c28 = [c for c in df.columns if '28' in c and ('spend' in c or 'tlag' in c)]
print('28d spend-like cols:', c28)
for c in c28[:3]:
    print('baseline %s MAE on 431: %.2f' % (c, np.abs(np.clip(df[c][va_mask],0,None)-yv[va_mask]).mean()))
print('mean-train MAE on 431: %.2f' % np.abs(np.full(va_mask.sum(), yv[tr_mask].mean())-yv[va_mask]).mean())


# ---- cell ----
import numpy as np, pandas as pd
t = agent_api.load_saved('e008_main_plus_marketing.parquet')
print(t[['days_since_last','tenure_days','snapshot_day']].dtypes)
print(t['days_since_last'].head(8).tolist(), t['tenure_days'].head(5).tolist())
ts_cols = [c for c in t.columns if c.startswith('ts_') or c.startswith('wl_') or c.startswith('wt_')]
print('temporal cols:', ts_cols)
print(t[ts_cols[:12]].describe().T.round(2))
print('index col?', t['index'].head(3).tolist() if 'index' in t.columns else None)
# check numeric-ness of days_since_last
dsl = pd.to_numeric(t['days_since_last'], errors='coerce')
print('dsl nan share', dsl.isna().mean(), 'range', dsl.min(), dsl.max())

# ---- cell ----
import numpy as np, pandas as pd

T8 = agent_api.load_saved('e008_main_plus_marketing.parquet')
prod = agent_api.snapshot().products
brand_map = prod.set_index('product_id')['brand']
dept_map = prod.set_index('product_id')['department']

def fn(view, snapshot_day, T8=T8, brand_map=brand_map, dept_map=dept_map):
    sd = snapshot_day
    base = T8[T8['snapshot_day'] == sd].set_index('household_key')
    hh = base.index
    tx = view.transactions
    out = pd.DataFrame(index=hh)

    # ---- weekly spend matrix, last 26 complete weeks ----
    W = (sd + 8) // 7
    gp = tx.groupby(['household_key', 'week_no'])['sales_value'].sum()
    mat = gp.unstack('week_no')
    weeks = [W - k for k in range(1, 27)]
    sw = mat.reindex(index=hh, columns=weeks)
    swv = np.nan_to_num(sw.values, nan=0.0)
    for h in (2, 4, 8):
        w = 0.5 ** (np.arange(1, 27) / h)
        out[f'ewma_{h}'] = (swv * w).sum(1) / w.sum()
    out['sw_sum26'] = swv.sum(1)
    out['sw_std26'] = swv.std(1)
    out['sw_max26'] = swv.max(1)
    out['sw_zero26'] = (swv == 0).sum(1)
    out['sw_cv26'] = swv.std(1) / (swv.mean(1) + 1e-6)
    out['sw_ratio_13_26'] = swv[:, :13].sum(1) / (swv[:, 13:].sum(1) + 1.0)

    # ---- long day-window aggregates ----
    def wagg(mask):
        m = tx.loc[mask]
        if len(m) == 0:
            return pd.DataFrame({'sp': 0.0, 'tr': 0.0, 'ln': 0.0}, index=hh)
        g = m.groupby('household_key').agg(sp=('sales_value', 'sum'),
                                           tr=('basket_id', 'nunique'),
                                           ln=('sales_value', 'size'))
        return g.reindex(hh).fillna(0.0)
    for lo, hi, sfx in [(sd-364, sd, '364'), (sd-182, sd, '182'), (sd-91, sd, '91')]:
        g = wagg((tx.day > lo) & (tx.day <= hi))
        out[f'spend_{sfx}'] = g['sp']; out[f'trips_{sfx}'] = g['tr']; out[f'lines_{sfx}'] = g['ln']

    # ---- target-aligned lags 14..18 (NaN where household too young) ----
    first_day = tx.groupby('household_key')['day'].min().reindex(hh)
    for k in range(14, 19):
        hi = sd - 28*(k-1); lo = sd - 28*k
        g = wagg((tx.day > lo) & (tx.day <= hi))
        v = g['sp'].where(first_day <= lo + 1)
        out[f'tlag_{k}'] = v
    out['yoy_spend364'] = wagg((tx.day > sd-392) & (tx.day <= sd-364))['sp']
    out['yoy_ratio'] = base['spend_28'] / (out['yoy_spend364'] + 1.0)

    # ---- day-of-week rhythm (84d spend shares) ----
    tx84 = tx[(tx.day > sd-84) & (tx.day <= sd)]
    g = tx84.assign(dow=tx84['day'] % 7).groupby(['household_key', 'dow'])['sales_value'].sum().unstack('dow')
    g = g.reindex(index=hh, columns=range(7)).fillna(0.0)
    tot = g.sum(1)
    for c in range(7):
        out[f'dow{c}_share84'] = g[c] / (tot + 1e-6)
    out['weekend_share84'] = (g[5] + g[6]) / (tot + 1e-6)

    # ---- time-of-day (84d, spend-weighted) ----
    if len(tx84):
        tt = tx84['trans_time'].astype(float); sp = tx84['sales_value'].astype(float)
        tmp = pd.DataFrame({'hh': tx84['household_key'].values,
                            'early': (tt.values < 1000) * sp.values,
                            'late': (tt.values >= 1800) * sp.values,
                            'sp': sp.values, 't': tt.values})
        g2 = tmp.groupby('hh').agg(early=('early', 'sum'), late=('late', 'sum'),
                                   sp=('sp', 'sum'), tmean=('t', 'mean'))
        out['early_share84'] = g2['early'] / (g2['sp'] + 1e-6)
        out['late_share84'] = g2['late'] / (g2['sp'] + 1e-6)
        out['tmean84'] = g2['tmean']
    else:
        out['early_share84'] = 0.0; out['late_share84'] = 0.0; out['tmean84'] = np.nan

    # ---- brand / department diversity recent ----
    tx28 = tx[(tx.day > sd-28) & (tx.day <= sd)]
    if len(tx28):
        br = tx28['product_id'].map(brand_map)
        priv = (br == 'Private').astype(float) * tx28['sales_value'].astype(float)
        g3 = pd.DataFrame({'hh': tx28['household_key'].values, 'priv': priv.values,
                           'sp': tx28['sales_value'].values}).groupby('hh').sum()
        out['private_share_28'] = g3['priv'] / (g3['sp'] + 1e-6)
        out['dept_div_28'] = tx28.assign(dep=tx28['product_id'].map(dept_map)) \
                                 .groupby('household_key')['dep'].nunique()
        tx84b = tx84.assign(dep=tx84['product_id'].map(dept_map))
        out['dept_div_84'] = tx84b.groupby('household_key')['dep'].nunique()
    else:
        out['private_share_28'] = np.nan; out['dept_div_28'] = 0; out['dept_div_84'] = 0
    out = out.reindex(hh)

    # ---- decay interactions & ratios ----
    dsl = base['days_since_last'].astype(float)
    out['rdecay'] = base['spend_28'] * np.exp(-dsl / 21.0)
    out['ewma4_active'] = out['ewma_4'] * (1 - base['ts_zero8'] / 8.0)
    out['sp28_ewma4'] = base['spend_28'] / (out['ewma_4'] + 1.0)
    out['sp28_over_84'] = base['spend_28'] / (base['spend_84'] + 1.0)
    tr28 = tx28.groupby('household_key')['basket_id'].nunique().reindex(hh).fillna(0.0)
    out['trips_28n'] = tr28
    out['spt_28'] = base['spend_28'] / (tr28 + 1.0)
    out['aup_28'] = base['spend_28'] / (base['qty_28'] + 1.0)
    out['disc_share_28'] = base['disc_28'] / (base['spend_28'] + 1.0)

    # ---- flags & target-window seasonality ----
    out['is_zero_28'] = (base['spend_28'] <= 0).astype(int)
    out['is_zero_56'] = (base['spend_56'] <= 0).astype(int)
    out['is_low_28'] = (base['spend_28'] < 20).astype(int)
    ph = 2 * np.pi * (((sd + 14) % 364) / 364.0)
    out['twsin'] = np.sin(ph); out['twcos'] = np.cos(ph)

    # ---- log1p versions of key skewed features ----
    LOGC = ['spend_7','spend_14','spend_28','spend_56','spend_84','spend_112','spend_182','spend_364',
            'lines_28','lines_84','lines_112','qty_28','qty_84','disc_28','disc_84','lifetime_spend',
            'ts_sum4','ts_sum8','ts_sum16','ts_max8','ts_std8','wl_mean13','distinct_products_84',
            'distinct_com_84','tlag_mean','tlag_max','tlag_std','basket_mean_84','basket_max_84',
            'ewma_2','ewma_4','ewma_8','sw_sum26','sw_max26','sw_std26','days_since_last']
    for c in LOGC:
        x = base[c] if c in base.columns else out[c]
        x = x.astype(float)
        out['lg_' + c] = np.sign(x) * np.log1p(np.abs(x))

    feat = base.join(out)
    if sd == 95:
        print('snap95 base rows', len(base), 'new feats', out.shape[1], 'total cols', feat.shape[1])
    return feat

df = agent_api.build_features(fn)
print('built', df.shape)
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key', 'snapshot_day'])
y = d['future_spend_4w'].astype(float)
newc = [c for c in df.columns if c not in set(T8.columns)]
print('new feats:', len(newc))
sp = d[newc].astype(float).corrwith(y, method='spearman').sort_values()
print('TOP:'); print(sp.tail(15).round(3))
print('BOTTOM:'); print(sp.head(8).round(3))
path = agent_api.save_table(df, 'e009_ewma_longlags.parquet')
print(path)
