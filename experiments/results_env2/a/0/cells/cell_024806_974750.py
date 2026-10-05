
import agent_api as A, numpy as np, pandas as pd

oof = A.load_saved('oof_e013.parquet')
p = oof.oof.values; y = oof.y.values
print('OOF MAE base: %.4f' % np.abs(p-y).mean())

def calib_mae(p, y, nbins, lam, folds=5, seed=0):
    # piecewise conditional-median calibration + shrinkage toward identity, honest via CV
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(p)); f = np.array_split(idx, folds)
    preds = np.empty(len(p))
    qs = np.quantile(p, np.linspace(0,1,nbins+1))
    for te in f:
        tr = np.setdiff1d(idx, te, assume_unique=False)
        pt, yt = p[tr], y[tr]
        bins = np.clip(np.searchsorted(qs[1:-1], pt, side='right'), 0, nbins-1)
        med = np.array([np.median(yt[bins==b]) if (bins==b).sum()>30 else np.nan for b in range(nbins)])
        # fill empty bins by interpolation from bin centers
        bc = np.array([(qs[b]+qs[b+1])/2 for b in range(nbins)])
        ok = ~np.isnan(med)
        med = np.interp(bc, bc[ok], med[ok])
        bt = np.clip(np.searchsorted(qs[1:-1], p[te], side='right'), 0, nbins-1)
        g = med[bt]
        preds[te] = lam*p[te] + (1-lam)*g
    return np.abs(preds-y).mean(), preds

best = None
for nbins in [6,10,15,25]:
    for lam in [0.0,0.2,0.4,0.6,0.8]:
        m,_ = calib_mae(p,y,nbins,lam)
        if best is None or m < best[0]: best = (m,nbins,lam)
        print('nbins %2d lam %.1f -> CV MAE %.4f' % (nbins,lam,m))
print('BEST:', best)

# global shift only, for reference
mshift = np.median(y-p)
print('shift-only (add median residual): CV-free MAE %.4f' % np.abs(p+mshift-y).mean())
# 2D check: does adding spend_28 to calibration help?
feats = A.load_saved('feats_v4.parquet')
o2 = oof.merge(feats[['household_key','snapshot_day','spend_28','exp4w_blend']], on=['household_key','snapshot_day'])
print(o2.shape)
