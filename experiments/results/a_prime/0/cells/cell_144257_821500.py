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