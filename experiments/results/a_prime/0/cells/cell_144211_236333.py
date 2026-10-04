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