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