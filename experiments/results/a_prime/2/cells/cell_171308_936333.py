import agent_api, numpy as np, pandas as pd

t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys = ['household_key','snapshot_day']
fc = [c for c in t.columns if c not in keys + ['index']]
tt = agent_api.train_targets()
m = t.merge(tt, on=keys, how='inner')
y = m['future_spend_4w'].astype(float).values
folds = m['snapshot_day'].values.astype(int)

# verify tlag_k alignment vs weekly lags ws_1..ws_26 (ws_j = spend in week j back)
ws = np.column_stack([m['ws_%d'%j].astype(float).values for j in range(1,27)])
g = lambda c: m[c].astype(float).values
tl2 = g('tlag_2'); tl3 = g('tlag_3'); tl13 = g('tlag_13')
for combo,name in [((4,5,6,7),'ws5-8'),((5,6,7,8),'ws6-9'),((3,4,5,6),'ws4-7')]:
    s = ws[:,[j-1 for j in combo]].sum(1)
    print('corr(tlag_2, %s) = %.4f' % (name, np.corrcoef(tl2, s)[0,1]))
for combo,name in [((5,6,7,8),'ws6-9'),((6,7,8,9),'ws7-10')]:
    s = ws[:,[j-1 for j in combo]].sum(1)
    print('corr(tlag_3, %s) = %.4f' % (name, np.corrcoef(tl3, s)[0,1]))
s13 = ws[:, [12,13,14,15]].sum(1)  # weeks 13-16 back
print('corr(tlag_13, ws13-16) = %.4f' % np.corrcoef(tl13, s13)[0,1])
print('means: spend_28 %.1f tlag_2 %.1f tlag_3 %.1f tlag_13 %.1f' % (g('spend_28').mean(), tl2.mean(), tl3.mean(), tl13.mean()))

# build new summary predictors from existing as-of-safe columns
tlags = ['tlag_%d'%i for i in range(2,14)]  # windows t-1 .. t-12 (plus spend_28 = t)
W = np.column_stack([g('spend_28')]+[g(c) for c in tlags])  # 13 windows: most recent first
W = np.nan_to_num(W)
dec_avg = None
for dec,w in [(0.65, None)]:
    w = dec**np.arange(13); w/=w.sum()
    dec_avg = W@w
med13 = np.median(W, axis=1)
Ws = np.sort(W, axis=1)
trim = Ws[:,1:-1].mean(1)
active = (W>0).mean(1)
am = active*W.mean(1)
nzmed = np.where((W>0).any(1), np.median(np.where(W>0,W,np.nan),axis=1), 0)
# LS trend slope over 13 windows
tt_ = np.arange(13, dtype=float); tt_ = (tt_-tt_.mean())/tt_.std()
slope = (W*tt_).sum(1)
cv13 = W.std(1)/np.maximum(W.mean(1),1)
new = pd.DataFrame({'household_key':m['household_key'],'snapshot_day':m['snapshot_day'],
    'dec_avg65':dec_avg,'med13':med13,'trim13':trim,'act_mean13':am,'nzmed13':np.nan_to_num(nzmed),
    'slope13':slope,'cv13':cv13,'max13':W.max(1),'min13':W.min(1)})
print('\nnew feats corr with y:', {c: round(float(np.corrcoef(new[c],y)[0,1]),3) for c in new.columns[2:]})

trm = folds<=375; vam=(folds==403)|(folds==431)
def mae(p): return float(np.abs(np.clip(p,0,None)[vam]-y[vam]).mean())
print('proxy MAE dec_avg65 %.2f | med13 %.2f | trim %.2f | am %.2f' % (mae(dec_avg), mae(med13), mae(trim), mae(am)))
print('blend 0.5*dec+0.5*med %.2f | 0.7*dec+0.3*med %.2f' % (mae(0.5*dec_avg+0.5*med13), mae(0.7*dec_avg+0.3*med13)))
print('blend dec+am %.2f' % mae(0.5*dec_avg+0.5*am))