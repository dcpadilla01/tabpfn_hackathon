import agent_api, numpy as np, pandas as pd
from itertools import product

t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys = ['household_key','snapshot_day']
fc = [c for c in t.columns if c not in keys + ['index']]
tt = agent_api.train_targets()
m = t.merge(tt, on=keys, how='inner')
y = m['future_spend_4w'].astype(float).values
folds = m['snapshot_day'].values.astype(int)
trm = folds <= 375; vam = (folds==403)|(folds==431)
def mae(p): return float(np.abs(np.clip(p,0,None)[vam] - y[vam]).mean())

g = lambda c: m[c].astype(float).values
tlags = ['tlag_%d'%i for i in range(2,14)]
W = np.column_stack([g('spend_28')]+[g(c) for c in tlags])  # 13 windows: t..t-12
med13 = np.nanmedian(W, axis=1); mean13 = np.nanmean(W, axis=1)
ewma4 = g('ewma_4'); ewma2 = g('ewma_2'); ewma8 = g('ewma_8')
active = (W > 0).mean(1)
sp91 = g('spend_91'); sp84=g('spend_84'); sp112=g('spend_112')

print('decay-weighted window avg (13 windows):')
for dec in [1.0,0.9,0.8,0.7,0.6,0.5,0.4]:
    w = dec**np.arange(13); w/=w.sum()
    print('  decay%.1f -> %.2f' % (dec, mae(np.nansum(W*w,axis=1))))
print('trimmed mean (drop min/max of 13):')
Ws = np.sort(W, axis=1)
print('  %.2f' % mae(Ws[:,1:-1].mean(1)))
print('mean of top-6 windows: %.2f' % mae(Ws[:,-6:].mean(1)))
print('median of nonzero windows: %.2f' % mae(np.nanmedian(np.where(W>0,W,np.nan),axis=1)))

# LS blends on raw domain (fit on train only)
cands = {'e4':ewma4,'med':med13,'am':active*mean13,'m13':mean13,'e2':ewma2,'e8':ewma8}
names = list(cands)
Xall = np.column_stack([cands[n] for n in names] + [np.ones(len(y))])
best=None
for r in [1,2,3]:
    for combo in product(range(len(names)), repeat=r):
        if len(set(combo))<r: continue
        cols=[cands[names[i]] for i in combo]+[np.ones(len(y))]
        Xm=np.column_stack(cols)
        beta,*_=np.linalg.lstsq(Xm[trm], y[trm], rcond=None)
        mm=mae(Xm@beta)
        if best is None or mm<best[1]: best=(tuple(names[i] for i in combo), mm)
print('\nbest LS blend (raw):', best)
# sqrt domain LS
sq = lambda v: np.sqrt(np.clip(v,0,None))
best2=None
for r in [1,2,3]:
    for combo in product(range(len(names)), repeat=r):
        if len(set(combo))<r: continue
        Xm=np.column_stack([sq(cands[names[i]]) for i in combo]+[np.ones(len(y))])
        beta,*_=np.linalg.lstsq(Xm[trm], y[trm], rcond=None)
        mm=mae(Xm@beta)
        if best2 is None or mm<best2[1]: best2=(tuple(names[i] for i in combo), mm)
print('best LS blend (sqrt):', best2)

# fixed simple blends
print('\nfixed blends:')
for a,b,c in [(1,0,0),(0,1,0),(0,0,1),(.5,.5,0),(.5,0,.5),(0,.5,.5),(.34,.33,.33),(.6,.2,.2),(.4,.4,.2),(.2,.6,.2),(.2,.2,.6)]:
    print('  e4*%.2f+med*%.2f+am*%.2f -> %.2f' % (a,b,c, mae(a*ewma4+b*med13+c*active*mean13)))
# concavity on median13
for pw in [0.8,0.9,1.0,1.1,1.2]:
    print('  med13 pow%.1f -> %.2f' % (pw, mae(np.sign(med13)*np.abs(med13)**pw)))
for cap in [300,500,800]:
    print('  med13 cap%d -> %.2f' % (cap, mae(np.minimum(med13,cap))))