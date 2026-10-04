import agent_api, pandas as pd, numpy as np
df = agent_api.load_saved('e011_price.parquet').merge(agent_api.train_targets(), on=['household_key','snapshot_day'])
tr = (df.snapshot_day<=403).values; va = (df.snapshot_day==431).values
NB = 32; NANB = NB-1
feats = ['spend28','trips28','recency','spend56','spend112','spend364','tenure','lt_spend']
B = np.zeros((len(df), len(feats)), dtype=np.uint8)
for j,c in enumerate(feats):
    x = df[c].values.astype(float)
    e = np.unique(np.quantile(x[tr], np.linspace(0,1,NB)))
    b = np.searchsorted(e[1:-1], x, side='right').astype(np.int8)
    B[:,j] = np.where(np.isnan(x), NANB, np.clip(b,0,NANB))
y = df.future_spend_4w.values.astype(float)
yw = np.clip(y, 0, np.quantile(y[tr], 0.97))
r = yw - yw[tr].mean()
rows = np.where(tr)[0]
ci = np.arange(len(feats))
sub = B[np.ix_(rows, ci)]
idx = sub.astype(np.int32) + NB*np.arange(len(ci))[None,:]
w = np.repeat(r[rows][:,None], len(ci), axis=1).ravel()
H = np.bincount(idx.ravel(), minlength=NB*len(ci)).reshape(len(ci), NB).astype(float)
G = np.bincount(idx.ravel(), weights=w, minlength=NB*len(ci)).reshape(len(ci), NB)
cg = np.cumsum(G, axis=1)[:,:-1]; ch = np.cumsum(H, axis=1)[:,:-1]
Gt = G.sum(1)[:,None]; Ht = H.sum(1)[:,None]
gain = cg**2/(ch+5.0) + (Gt-cg)**2/(Ht-ch+5.0) - Gt**2/(Ht+5.0)
gain[ch<60] = -1; gain[Ht-ch<60] = -1
fj, bj = np.unravel_index(np.argmax(gain), gain.shape)
print('root split: feat', feats[fj], 'bin', bj, 'gain', gain[fj,bj])
f = fj
bcol = B[rows, f]
L = rows[bcol <= bj]; Rr = rows[bcol > bj]
print('split sizes', len(L), len(Rr))
print('leaf vals', r[L].mean(), r[Rr].mean())
va_idx = np.where(va)[0]
pred = np.where(B[va_idx, f] <= bj, r[L].mean(), r[Rr].mean()) + yw[tr].mean()
print('1-split stump MAE431:', round(float(np.abs(pred - y[va_idx]).mean()),3))
