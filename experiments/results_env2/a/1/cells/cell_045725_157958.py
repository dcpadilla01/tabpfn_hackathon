import pandas as pd, numpy as np
from scipy.optimize import minimize

tt = train_targets().rename(columns={'future_spend_4w':'y'})
ap = load_saved('e016_allpreds.parquet')
held = load_saved('e016_held.parquet').rename(columns={'future_spend_4w':'y'})
h = held[['household_key','snapshot_day','sq0','sq1','sq2','y']].merge(
    ap[['household_key','snapshot_day','pq','pl','pa']], on=['household_key','snapshot_day'], how='inner')
print('holdout merged:', h.shape)
mem = ['pq','pl','pa','sq0','sq1','sq2']
P = h[mem].values; y = h['y'].values
for i,c in enumerate(mem):
    print(' %-4s MAE %.3f' % (c, np.abs(P[:,i]-y).mean()))

def mae_w(w, P, y):
    return np.abs(P @ w - y).mean()

# pairwise blends
print('\npair blends (0.5/0.5):')
for i in range(len(mem)):
    for j in range(i+1, len(mem)):
        m = np.abs(0.5*P[:,i]+0.5*P[:,j]-y).mean()
        if m < 62.0: print('  %s+%s: %.3f' % (mem[i],mem[j],m))

# weight fit with day cross-validation
d403 = (h.snapshot_day==403).values; d431 = ~d403
def fit_w(mask):
    Pm, ym = P[mask], y[mask]
    f = lambda z: mae_w(np.exp(z)/np.exp(z).sum(), Pm, ym)
    r = minimize(f, np.zeros(len(mem)), method='Nelder-Mead',
                 options={'maxiter':4000,'xatol':1e-4,'fatol':1e-5})
    return np.exp(r.x)/np.exp(r.x).sum()
w403 = fit_w(d403); w431 = fit_w(d431); wall = fit_w(np.ones(len(h),bool))
print('\nweights fit@403 :', dict(zip(mem, w403.round(3))), ' -> MAE@431 %.3f' % mae_w(w403, P[d431], y[d431]))
print('weights fit@431 :', dict(zip(mem, w431.round(3))), ' -> MAE@403 %.3f' % mae_w(w431, P[d403], y[d403]))
print('weights fit@all :', dict(zip(mem, wall.round(3))), ' -> MAE@all %.3f (in-sample)' % mae_w(wall, P, y))
print('equal weights    : MAE@all %.3f' % mae_w(np.ones(6)/6, P, y))
print('pq alone         : MAE@all %.3f' % np.abs(h.pq-y).mean())

# pq vs pq+sq0 weight curve
print('\npq/sq0 mix curve:')
for w in [0,.1,.2,.3,.4,.5,.6,.7,.8,1.0]:
    p = w*h.pq + (1-w)*h.sq0
    print('  w_pq=%.1f  MAE %.3f' % (w, np.abs(p-y).mean()))

# affine cal transfer for pq and best blend
for name, p in [('pq', h.pq), ('pq+sq0 .5', .5*h.pq+.5*h.sq0)]:
    a403 = np.polyfit(p[d403], y[d403], 1)
    p431 = a403[0]*p[d431]+a403[1]
    print('%s cal@403->431: %.3f (uncal %.3f)' % (name, np.abs(p431-y[d431]).mean(), np.abs(p[d431]-y[d431]).mean()))
