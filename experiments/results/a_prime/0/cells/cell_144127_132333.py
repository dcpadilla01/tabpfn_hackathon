import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def temporal_fn(view, s):
    hh = pd.Index(view.households)
    tx = view.transactions
    hv = tx['household_key'].values
    day = tx['day'].values
    sv = tx['sales_value'].values
    bq = tx['basket_id'].values
    age = s - day
    first = tx.groupby('household_key')['day'].min()
    ten = (s - first).astype(float)
    blk = age // 28
    P = pd.DataFrame({'h': hv, 'b': blk, 'v': sv}).groupby(['h','b'])['v'].sum().unstack(fill_value=0.0)
    NB = pd.DataFrame({'h': hv, 'b': blk, 'v': bq}).drop_duplicates().groupby(['h','b']).size().unstack(fill_value=0)
    wk = age // 7
    W = pd.DataFrame({'h': hv, 'w': wk, 'v': sv}).groupby(['h','w'])['v'].sum().unstack(fill_value=0.0)
    idx = P.index
    obs = (ten.reindex(idx).values + 1.0)
    out = pd.DataFrame(index=idx)
    for k in range(1, 14):
        col = P[k-1] if (k-1) in P.columns else pd.Series(0.0, index=idx)
        raw = col.values.astype(float); win = 28.0*k
        full = obs >= win
        val = np.where(full, raw, np.where(obs > 0, raw/np.maximum(obs/win, 1e-9), np.nan))
        out['p%d' % k] = val
    NBr = NB.reindex(idx).fillna(0)
    for k in range(1, 14):
        col = NBr[k-1] if (k-1) in NBr.columns else pd.Series(0.0, index=idx)
        raw = col.values.astype(float); win = 28.0*k
        full = obs >= win
        val = np.where(full, raw, np.where(obs > 0, raw/np.maximum(obs/win, 1e-9), np.nan))
        out['nb%d' % k] = val
    for k in range(1, 9):
        col = W[k-1] if (k-1) in W.columns else pd.Series(0.0, index=idx)
        out['w%d' % k] = col.values.astype(float)
    Pc = ['p%d' % k for k in range(1, 14)]
    Pv = out[Pc].values
    M = ~np.isnan(Pv); K = np.arange(1, 14)[None, :]
    n = M.sum(1)
    sy = np.where(M, Pv, 0).sum(1); mean = sy/np.maximum(n, 1)
    sx = (K*M).sum(1); sxx = (K*K*M).sum(1)
    sxy = np.where(M, Pv*K, 0).sum(1)
    den = n*sxx - sx*sx
    out['b_mean'] = mean
    out['b_std'] = np.sqrt(np.where(M, (Pv-mean[:, None])**2, 0).sum(1)/np.maximum(n, 1))
    out['b_max'] = np.where(M, Pv, -1e18).max(1)
    out['b_cv'] = out['b_std']/(mean+5)
    out['zero_blocks'] = ((Pv < 2) & M).sum(1).astype(float)
    out['slope'] = np.where(den > 1e-9, (n*sxy - sx*sy)/np.maximum(den, 1e-9), np.nan)
    M6 = M & (K >= 8)
    n6 = M6.sum(1); sy6 = np.where(M6, Pv, 0).sum(1)
    sx6 = (K*M6).sum(1); sxx6 = (K*K*M6).sum(1); sxy6 = np.where(M6, Pv*K, 0).sum(1)
    den6 = n6*sxx6 - sx6*sx6
    out['slope6'] = np.where((den6 > 1e-9) & (n6 >= 3), (n6*sxy6 - sx6*sy6)/np.maximum(den6, 1e-9), np.nan)
    out['has_full_yr'] = (obs >= 364).astype(float)
    out['n_blocks_full'] = np.minimum(13, np.floor(obs/28)).astype(float)
    p1 = out['p1'].values; p13 = out['p13'].values
    out['seas_ratio'] = (p1+5)/(p13+5)
    out['yoy_diff'] = p1 - p13
    out['p1_div_bmean'] = p1/(mean+5)
    Wv = out[['w%d' % k for k in range(1, 9)]].values
    wm = Wv.mean(1); ws = Wv.std(1)
    out['w_cv'] = ws/(wm+5)
    Kw = np.arange(1, 9)[None, :]
    mx = Kw.mean()
    sxyw = (Wv*Kw).sum(1)
    out['w_slope'] = (sxyw - 8*wm*mx)/(sww := (Wv*Kw*Kw).sum(1) - 8*mx*mx + 1e-9)
    NBc = ['nb%d' % k for k in range(1, 14)]
    NBv = out[NBc].values
    Mn = ~np.isnan(NBv)
    out['nb_mean'] = np.where(Mn, NBv, 0).sum(1)/np.maximum(Mn.sum(1), 1)
    out['nb_std'] = np.sqrt(np.where(Mn, (NBv-out['nb_mean'].values[:, None])**2, 0).sum(1)/np.maximum(Mn.sum(1), 1))
    out['nb_max'] = np.where(Mn, NBv, -1e18).max(1)
    return out.reindex(hh)

temporal = A.build_features(temporal_fn)
print('temporal built', temporal.shape)
mkt = A.load_saved('mkt_v2.parquet')
merged = mkt.merge(temporal, on=['household_key', 'snapshot_day'], how='left')
print('merged', merged.shape)
path = A.save_table(merged, 'e006_temporal')
print('saved', path)

tt = A.train_targets()
newf = [c for c in temporal.columns if c not in ('household_key', 'snapshot_day')]
d = merged.merge(tt, on=['household_key', 'snapshot_day'], how='left')
is_tr = d['future_spend_4w'].notna().values
y = d['future_spend_4w'].values
print('new-feature corr with target (train rows):')
tr = d[is_tr]
cors = {}
for c in newf:
    v = pd.to_numeric(tr[c], errors='coerce')
    cors[c] = v.corr(tr['future_spend_4w'])
for c, v in sorted(cors.items(), key=lambda kv: -abs(kv[1] if kv[1] == kv[1] else 0))[:25]:
    print('  %-16s %.3f' % (c, v))
print('NaN rate max among new feats: %.3f' % d[newf].isna().mean().max())

def ridge_screen(df, feats, name):
    dd = df.merge(tt, on=['household_key', 'snapshot_day'], how='left')
    itr = dd['future_spend_4w'].notna().values
    X = dd[feats].apply(pd.to_numeric, errors='coerce').values.astype(float)
    yy = dd['future_spend_4w'].values
    med = np.nanmedian(X[itr], 0)
    X = np.where(np.isnan(X), med, X)
    Xtr, ytr = X[itr], yy[itr]; Xva, yva = X[~itr], yy[~itr]
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-9] = 1
    Ztr = np.c_[(Xtr-mu)/sd, np.ones(len(Xtr))]; Zva = np.c_[(Xva-mu)/sd, np.ones(len(Xva))]
    res = []
    for tname, ty in [('raw', ytr), ('log', np.log1p(ytr))]:
        for a in [3, 30, 300, 3000]:
            Am = Ztr.T@Ztr + a*np.eye(Ztr.shape[1]); Am[-1, -1] -= a
            w = np.linalg.solve(Am, Ztr.T@ty)
            pv = Zva@w
            if tname == 'log': pv = np.expm1(np.clip(pv, 0, 12))
            res.append((tname, a, np.abs(pv-yva).mean()))
    best = min(res, key=lambda r: r[2])
    print('%-28s best: %s a=%d MAE %.3f | all: %s' % (name, best[0], best[1], best[2],
          ' '.join('%s%d:%.1f' % r for r in res)))
    return best[2]

mktf = [c for c in mkt.columns if c not in ('household_key', 'snapshot_day')]
print('--- offline ridge screens (val MAE) ---')
ridge_screen(mkt, mktf, 'mkt_v2 (E003)')
ridge_screen(merged, mktf + newf, 'mkt_v2 + temporal')
ridge_screen(temporal.reset_index(), newf, 'temporal only')
med = np.median(y[is_tr]); print('predict train median: %.3f' % np.abs(med - y[~is_tr]).mean())
s28 = np.nan_to_num(d['spend_28'].values.astype(float))
print('predict spend_28:     %.3f' % np.abs(s28[~is_tr] - y[~is_tr]).mean())