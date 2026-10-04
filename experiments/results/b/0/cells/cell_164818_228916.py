import agent_api, pandas as pd, numpy as np
print('start', flush=True)
df = agent_api.load_saved('e011_price.parquet').merge(agent_api.train_targets(), on=['household_key','snapshot_day'])
tr = (df.snapshot_day<=403).values
NB = 32
feats = ['spend28','trips28','recency']
B = np.zeros((len(df), len(feats)), dtype=np.uint8)
for j,c in enumerate(feats):
    x = df[c].values.astype(float)
    e = np.unique(np.quantile(x[tr], np.linspace(0,1,NB)))
    b = np.searchsorted(e[1:-1], x, side='right').astype(np.int8)
    B[:,j] = np.where(np.isnan(x), NB-1, np.clip(b,0,NB-1))
print('binned', flush=True)
rows = np.where(tr)[0]
r = df.future_spend_4w.values.astype(float); r = np.clip(r,0,300); r = r - r[tr].mean()
ci = np.arange(len(feats))
sub = B[np.ix_(rows, ci)]
print('sub', sub.shape, sub.dtype, flush=True)
idx = sub.astype(np.int32) + NB*np.arange(len(ci))[None,:]
print('idx done', idx.shape, idx.max(), flush=True)
w = np.repeat(r[rows][:,None], len(ci), axis=1).ravel()
print('w done', flush=True)
H = np.bincount(idx.ravel(), minlength=NB*len(ci)).reshape(len(ci), NB).astype(float)
print('H done', flush=True)
G = np.bincount(idx.ravel(), weights=w, minlength=NB*len(ci)).reshape(len(ci), NB)
print('G done', H.sum(), G.sum(), flush=True)
