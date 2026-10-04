import agent_api as A
import pandas as pd, numpy as np

for name in ['mkt_v2','mix_v1','hist_v2']:
    t = A.load_saved(name + '.parquet')
    print(name, t.shape)
    print(list(t.columns))
    print()

tt = A.train_targets()
y = tt.future_spend_4w
print('n rows', len(tt))
print(y.describe())
print('zero share', (y == 0).mean())
print(y.quantile([.5, .75, .9, .95, .99]))
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean', 'median', 'count']))


# ---- cell ----
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

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np
tt = A.train_targets()
print(tt.dtypes)
print(tt.head())
m = A.load_saved('e006_temporal.parquet')
print(m.dtypes.head(3))
print(m[['household_key','snapshot_day']].head())
print('tt keys sample', tt['household_key'].head().tolist(), m['household_key'].head().tolist())
print('overlap', m.set_index(['household_key','snapshot_day']).index.isin(tt.set_index(['household_key','snapshot_day']).index).sum(), 'of', len(tt))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

tt = A.train_targets()
mkt = A.load_saved('mkt_v2.parquet')
merged = A.load_saved('e006_temporal.parquet')
temporal_cols = [c for c in merged.columns if c not in mkt.columns]
newf = [c for c in temporal_cols if c not in ('household_key','snapshot_day')]
mktf = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]

def screen(df, feats, name, log_opts=(False, True)):
    dd = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    itr = dd['snapshot_day'].values <= 403
    X = dd[feats].apply(pd.to_numeric, errors='coerce').values.astype(float)
    yy = dd['future_spend_4w'].values
    med = np.nanmedian(X[itr], 0)
    X = np.where(np.isnan(X), med, X)
    Xtr, ytr = X[itr], yy[itr]; Xva, yva = X[~itr], yy[~itr]
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-9] = 1
    Ztr = np.c_[(Xtr-mu)/sd, np.ones(len(Xtr))]; Zva = np.c_[(Xva-mu)/sd, np.ones(len(Xva))]
    res = []
    for use_log in log_opts:
        ty = np.log1p(ytr) if use_log else ytr
        for a in [3, 30, 300, 3000]:
            Am = Ztr.T@Ztr + a*np.eye(Ztr.shape[1]); Am[-1,-1] -= a
            w = np.linalg.solve(Am, Ztr.T@ty)
            pv = Zva@w
            if use_log: pv = np.expm1(np.clip(pv, 0, 12))
            res.append(('log' if use_log else 'raw', a, np.abs(pv-yva).mean()))
    best = min(res, key=lambda r: r[2])
    print('%-26s best %s a=%-4d MAE %.3f | %s' % (name, best[0], best[1], best[2],
          ' '.join('%s%d:%.1f' % r for r in res)))
    return best[2]

print('inner split: fit <=403, eval day 431 (n_val=%d)' % ((tt.snapshot_day == 431).sum()))
med_all = tt[tt.snapshot_day <= 403].future_spend_4w.median()
v = tt[tt.snapshot_day == 431]
print('%-26s MAE %.3f' % ('const median', np.abs(med_all - v.future_spend_4w).mean()))
mm = mkt.merge(tt, on=['household_key','snapshot_day'])
vm = mm[mm.snapshot_day == 431]
print('%-26s MAE %.3f' % ('spend_28 naive', np.abs(np.nan_to_num(vm.spend_28.values) - vm.future_spend_4w.values).mean()))
screen(mkt, mktf, 'mkt_v2 (E003)')
screen(merged, mktf + newf, 'mkt_v2 + temporal')
screen(merged.reset_index(), newf, 'temporal only')

# which temporal feats help most: forward screen on top of mkt
base = screen(mkt, mktf, 'mkt_v2 base (repeat)', log_opts=(True,))
order = [c for c in newf]
cur = list(mktf)
gains = []
for c in order:
    m = screen(mkt, cur + [c], 'mkt+'+c, log_opts=(True,))
    gains.append((m - base, c))
gains.sort()
print('\nbest single additions (MAE delta, negative=good):')
for g, c in gains[:12]: print('  %-16s %+.3f' % (c, g))
print('worst:')
for g, c in gains[-5:]: print('  %-16s %+.3f' % (c, g))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
mkt = A.load_saved('mkt_v2.parquet')
merged = A.load_saved('e006_temporal.parquet')
newf = [c for c in merged.columns if c not in mkt.columns and c not in ('household_key','snapshot_day')]
sub = merged[newf].apply(pd.to_numeric, errors='coerce')
print('infs per col (top):')
inf_counts = np.isinf(sub.values).sum(0)
print({c: int(n) for c, n in zip(newf, inf_counts) if n > 0})
print('\nmax abs per col:')
mx = sub.abs().max()
print(mx.sort_values(ascending=False).head(15))
print('\n99.9% quantile abs:')
q = sub.abs().quantile(0.999)
print(q.sort_values(ascending=False).head(15))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
merged = A.load_saved('e006_temporal.parquet')
K = np.arange(1, 9)[None, :]
Wv = merged[['w%d' % k for k in range(1, 9)]].values.astype(float)
wm = Wv.mean(1, keepdims=True)
merged['w_slope'] = ((Wv - wm) * (K - K.mean())).sum(1) / 42.0
print('w_slope range', merged['w_slope'].abs().max())
path = A.save_table(merged, 'e006_temporal')
print('saved', path)

tt = A.train_targets()
mkt = A.load_saved('mkt_v2.parquet')
newf = [c for c in merged.columns if c not in mkt.columns and c not in ('household_key','snapshot_day')]
mktf = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]

def screen(df, feats, name, alphas=(3, 30, 300, 3000)):
    dd = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    itr = dd['snapshot_day'].values <= 403
    X = dd[feats].apply(pd.to_numeric, errors='coerce').values.astype(float)
    yy = dd['future_spend_4w'].values
    med = np.nanmedian(X[itr], 0)
    X = np.where(np.isnan(X), med, X)
    Xtr, ytr = X[itr], yy[itr]; Xva, yva = X[~itr], yy[~itr]
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-9] = 1
    Ztr = np.c_[(Xtr-mu)/sd, np.ones(len(Xtr))]; Zva = np.c_[(Xva-mu)/sd, np.ones(len(Xva))]
    res = []
    for use_log in (False, True):
        ty = np.log1p(ytr) if use_log else ytr
        for a in alphas:
            Am = Ztr.T@Ztr + a*np.eye(Ztr.shape[1]); Am[-1,-1] -= a
            w = np.linalg.solve(Am, Ztr.T@ty)
            pv = Zva@w
            if use_log: pv = np.expm1(np.clip(pv, 0, 12))
            res.append(('log' if use_log else 'raw', a, np.abs(pv-yva).mean()))
    best = min(res, key=lambda r: r[2])
    print('%-24s best %s a=%-4d MAE %.3f | %s' % (name, best[0], best[1], best[2],
          ' '.join('%s%d:%.1f' % r for r in res)))
    return best[2]

print('--- screens ---')
b1 = screen(mkt, mktf, 'mkt_v2 (E003)')
b2 = screen(merged, mktf + newf, 'mkt_v2 + temporal')
b3 = screen(merged, newf, 'temporal only')
# forward additions on top of mkt_v2
base = b1
gains = []
for c in newf:
    g = screen(mkt, mktf + [c], 'mkt+'+c)
    gains.append((g - base, c))
gains.sort()
print('\nsingle-feature deltas (neg = helps):')
for g, c in gains[:12]: print('  %-16s %+.3f' % (c, g))
print('  ...')
for g, c in gains[-4:]: print('  %-16s %+.3f' % (c, g))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
merged = A.load_saved('e006_temporal.parquet')
mkt = A.load_saved('mkt_v2.parquet')
tt = A.train_targets()
newf = [c for c in merged.columns if c not in mkt.columns and c not in ('household_key','snapshot_day')]
mktf = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]

def screen(df, feats):
    dd = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    itr = dd['snapshot_day'].values <= 403
    X = dd[feats].apply(pd.to_numeric, errors='coerce').values.astype(float)
    yy = dd['future_spend_4w'].values
    med = np.nanmedian(X[itr], 0)
    X = np.where(np.isnan(X), med, X)
    Xtr, ytr = X[itr], yy[itr]; Xva, yva = X[~itr], yy[~itr]
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-9] = 1
    Ztr = np.c_[(Xtr-mu)/sd, np.ones(len(Xtr))]; Zva = np.c_[(Xva-mu)/sd, np.ones(len(Xva))]
    res = []
    for use_log in (False, True):
        ty = np.log1p(ytr) if use_log else ytr
        for a in (3, 30, 300, 3000):
            Am = Ztr.T@Ztr + a*np.eye(Ztr.shape[1]); Am[-1,-1] -= a
            w = np.linalg.solve(Am, Ztr.T@ty)
            pv = Zva@w
            if use_log: pv = np.expm1(np.clip(pv, 0, 12))
            res.append(('log' if use_log else 'raw', a, np.abs(pv-yva).mean()))
    return min(res, key=lambda r: r[2])

base = screen(merged, mktf)[2]
print('base mkt_v2 on merged: %.3f' % base)
gains = []
for c in newf:
    g = screen(merged, mktf + [c])[2]
    gains.append((g - base, c))
gains.sort()
print('single-feature deltas (neg = helps):')
for g, c in gains[:15]: print('  %-16s %+.3f' % (c, g))
print('  ...')
for g, c in gains[-4:]: print('  %-16s %+.3f' % (c, g))
# greedy forward: add top features one by one
cur = list(mktf)
pool = [c for _, c in gains if _ < 0.05]
hist = []
for c in pool:
    cur.append(c)
    h = screen(merged, cur)
    hist.append((h[2], c, h[0], h[1]))
    print('after +%s: %.3f (%s a=%s)' % (c, h[2], h[0], h[1]))
best = min(hist, key=lambda x: x[0])
print('best greedy: %.3f with %d temporal feats added' % (best[0], hist.index(best)+1))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
merged = A.load_saved('e006_temporal.parquet')
mkt = A.load_saved('mkt_v2.parquet')
tt = A.train_targets()
newf = [c for c in merged.columns if c not in mkt.columns and c not in ('household_key','snapshot_day')]
mktf = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]

def screen(df, feats):
    dd = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    itr = dd['snapshot_day'].values <= 403
    X = dd[feats].apply(pd.to_numeric, errors='coerce').values.astype(float)
    yy = dd['future_spend_4w'].values
    med = np.nanmedian(X[itr], 0)
    X = np.where(np.isnan(X), med, X)
    Xtr, ytr = X[itr], yy[itr]; Xva, yva = X[~itr], yy[~itr]
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-9] = 1
    Ztr = np.c_[(Xtr-mu)/sd, np.ones(len(Xtr))]; Zva = np.c_[(Xva-mu)/sd, np.ones(len(Xva))]
    res = []
    for use_log in (False, True):
        ty = np.log1p(ytr) if use_log else ytr
        for a in (3, 30, 300, 3000):
            Am = Ztr.T@Ztr + a*np.eye(Ztr.shape[1]); Am[-1,-1] -= a
            w = np.linalg.solve(Am, Ztr.T@ty)
            pv = Zva@w
            if use_log: pv = np.expm1(np.clip(pv, 0, 12))
            res.append(('log' if use_log else 'raw', a, np.abs(pv-yva).mean()))
    return min(res, key=lambda r: r[2])

print('full fixed block:', screen(merged, mktf + newf))
curated = ['slope','p12','yoy_diff','seas_ratio','nb12','p5','w5','w2','p6','w_slope','b_mean','nb3','p11','nb11','p3','w8','nb5','w7','nb4','p1','n_blocks_full','w3','nb_std','nb_max','w_cv','nb2','p2','nb1','w1','p1_div_bmean']
print('curated 30:', screen(merged, mktf + curated))
print('mkt base:', screen(merged, mktf))

# ---- cell ----
import agent_api as A
import pandas as pd
merged = A.load_saved('e006_temporal.parquet')
mkt = A.load_saved('mkt_v2.parquet')
curated = ['slope','p12','yoy_diff','seas_ratio','nb12','p5','w5','w2','p6','w_slope','b_mean','nb3','p11','nb11','p3','w8','nb5','w7','nb4','p1','n_blocks_full','w3','nb_std','nb_max','w_cv','nb2','p2','nb1','w1','p1_div_bmean']
keep = ['household_key','snapshot_day'] + [c for c in mkt.columns if c not in ('household_key','snapshot_day')] + curated
out = merged[keep]
print(out.shape)
p = A.save_table(out, 'e006_curated')
print(p)