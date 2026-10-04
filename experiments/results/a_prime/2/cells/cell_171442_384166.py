import agent_api, numpy as np, pandas as pd

t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys = ['household_key','snapshot_day']
Wcols = ['spend_28'] + ['tlag_%d'%i for i in range(2,14)]  # 13 windows: t..t-12
W = np.nan_to_num(t[Wcols].astype(float).values)
ten = t['tenure_days'].astype(float).values
avail = (ten[:,None] >= 28*np.arange(1,14)[None,:])

feats = {}
for d in [0.5,0.6,0.65,0.7,0.8]:
    w = d**np.arange(13); w/=w.sum()
    feats['dec%d'%(d*100)] = W@w
    wv = np.where(avail, w[None,:], 0.0); wv = wv/np.maximum(wv.sum(1,keepdims=True),1e-9)
    feats['dec%d_avail'%(d*100)] = (W*wv).sum(1)
w = 0.65**np.arange(13); w/=w.sum(); dec65 = W@w
feats['lg_dec65'] = np.log1p(dec65)
feats['med13'] = np.median(W,axis=1)
Ws = np.sort(W,axis=1); feats['trim13'] = Ws[:,1:-1].mean(1)
tt_ = (np.arange(13)-6)/np.sqrt(143.0); feats['slope13'] = (W*tt_).sum(1)
feats['cv13'] = W.std(1)/np.maximum(W.mean(1),1.0)
feats['max13'] = W.max(1)
zs = (W==0); feats['zero_streak'] = (np.cumprod(zs,axis=1)).sum(1)
old = W[:,5:13].mean(1); feats['momentum'] = dec65/np.maximum(old,1.0)
feats['act13'] = (W>0).mean(1)
feats['mean13_avail'] = np.where(avail.any(1), (W*np.where(avail,1,0)).sum(1)/np.maximum(avail.sum(1),1), 0.0)
newdf = pd.DataFrame(np.column_stack([t[keys[0]].values, t[keys[1]].values]+[feats[k] for k in feats]), columns=keys+list(feats))
for c in newdf.columns[2:]: newdf[c]=newdf[c].astype(float)
print('new feats:', len(feats))

ttg = agent_api.train_targets()
mn = t.merge(ttg, on=keys).merge(newdf, on=keys)
y = mn['future_spend_4w'].astype(float).values
folds = mn['snapshot_day'].values.astype(int)
trm = folds<=375; vam=(folds==403)|(folds==431)
def mae(p): return float(np.abs(np.clip(p,0,None)[vam]-y[vam]).mean())
for k in feats: print('  %-14s %.2f' % (k, mae(mn[k].values)))

snap = agent_api.snapshot()
tr = snap.transactions
wk = ((tr['day']+8)//7).astype(int)
g = tr.groupby(wk)['sales_value'].sum()
w2 = g.reindex(range(1,103)).fillna(0)
print('\nretailer weekly spend: wk5-19 %.0f wk31-45 %.0f wk57-71 %.0f' % (w2[5:20].mean(), w2[31:46].mean(), w2[57:72].mean()))
r = w2/w2.shift(52)
print('yoy ratio wk57-70:', r[57:71].round(2).tolist())
print('yoy ratio wk66-78:', r[66:79].round(2).tolist())