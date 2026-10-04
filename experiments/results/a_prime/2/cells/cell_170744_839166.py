import agent_api, numpy as np, pandas as pd

t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys = ['household_key','snapshot_day']
fc = [c for c in t.columns if c not in keys + ['index']]
tt = agent_api.train_targets()
m = t.merge(tt, on=keys, how='inner')
y = m['future_spend_4w'].astype(float).values
folds = m['snapshot_day'].values.astype(int)
print('mean y by day:', {int(d): round(float(y[folds==d].mean()),1) for d in sorted(set(folds))})

g = lambda c: m[c].astype(float).values
spend_28 = g('spend_28')
# find tlag cols
tlags = sorted([c for c in fc if c.startswith('tlag_') and c[5:].isdigit()], key=lambda c:int(c[5:]))
print('tlag cols:', tlags[:14])
W = np.column_stack([g(c) for c in ['spend_28']+tlags[:12]])  # windows t..t-11 (28d each)
med13 = np.nanmedian(W, axis=1)
mean13 = np.nanmean(W, axis=1)
ewma4 = g('ewma_4'); ewma8 = g('ewma_8'); ewma2 = g('ewma_2')
spend_91 = g('spend_91'); spend_182 = g('spend_182'); spend_364 = g('spend_364')
active = (W > 0).mean(1)

trm = folds <= 375; vam = (folds==403)|(folds==431)
def mae(p, mask=vam): return float(np.abs(np.clip(p,0,None)[mask] - y[mask]).mean())

print('\nstandalone predictor MAE on val {403,431}:')
cands = {'spend_28':spend_28,'ewma_2':ewma2,'ewma_4':ewma4,'ewma_8':ewma8,
         'mean13':mean13,'median13':med13,'spend_91':spend_91,'spend_182':spend_182,
         'spend_364':spend_364,'active*mean13':active*mean13,'active*ewma4':active*ewma4}
for k,v in cands.items(): print('  %-16s %.2f' % (k, mae(v)))
print('  mean-baseline   %.2f' % mae(np.full(len(y), y[trm].mean())))

# decay-weighted window average, w_i = decay^i
best=None
for dec in [1.0,0.9,0.8,0.7,0.6,0.5]:
    w = dec**np.arange(12); w/=w.sum()
    p = np.nansum(W*w,axis=1)
    print('  decay%.1f winavg  %.2f' % (dec, mae(p)))
    if best is None or mae(p)<best[1]: best=(dec,mae(p))
# blend grid: a*ewma4 + b*median13 + c*active*mean13 + d
A = np.column_stack([ewma4, med13, active*mean13, np.ones(len(y))])
from itertools import product
bestb=None
for a,b,c in product([0,.25,.5,.75,1],[0,.25,.5,.75,1],[0,.5,1]):
    if a+b+c>1.5: continue
    p = a*ewma4 + b*med13 + c*active*mean13 + (1-min(1,a+b+c))*y[trm].mean()*0  # no const
    mm = mae(p)
    if bestb is None or mm<bestb[1]: bestb=(a,b,c,mm)
print('  best blend a,b,c:', bestb)

# winsorization / concavity of best simple predictor
p = ewma4
for cap in [400,600,800,1200,2000]:
    print('  ewma4 capped@%d %.2f' % (cap, mae(np.minimum(p,cap))))
for pw in [0.5,0.7,0.85,1.0,1.15,1.3]:
    s = np.sign(p)*np.abs(p)**pw
    print('  ewma4 pow%.2f %.2f' % (pw, mae(s)))
# sqrt-domain blend fit on train via least squares (mean fit) then MAE eval
for cols,name in [([ewma4,med13],'e4+med'),([ewma4,med13,active*mean13],'e4+med+am'),([np.sqrt(ewma4),np.sqrt(med13)],'sqrt e4+med')]:
    Xm = np.column_stack(cols+[np.ones(len(y))])
    beta,*_ = np.linalg.lstsq(Xm[trm], y[trm], rcond=None)
    print('  LS %s: %.2f' % (name, mae(Xm@beta)))