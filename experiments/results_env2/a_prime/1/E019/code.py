import agent_api as api
import pandas as pd, numpy as np

e15 = api.load_saved('e015_stack.parquet')
print('e15 shape:', e15.shape)
scols = [c for c in e15.columns if 'stack' in c.lower()]
print('stack cols:', scols)
print(e15[scols].describe())

tt = api.train_targets()
print('\ntrain_targets:', tt.shape)
print(tt.future_spend_4w.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]))
print('zero share:', (tt.future_spend_4w==0).mean())

base = api.baseline_features()
print('\nbaseline rows:', base.shape)
k15 = set(map(tuple, e15[['household_key','snapshot_day']].values))
kb = set(map(tuple, base[['household_key','snapshot_day']].values))
print('e15 missing rows vs baseline:', len(kb-k15), 'extra:', len(k15-kb))

# candidate columns for peer median base + recency
cands = [c for c in e15.columns if ('ew' in c.lower()) or ('spend_28' in c) or ('sp28' in c) or ('recen' in c.lower())]
print('\ncandidate cols:', cands[:40])
print('\ndtypes:', e15.dtypes.value_counts().to_dict())
print('snapshot days present:', sorted(e15.snapshot_day.unique()))


# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

e15 = api.load_saved('e015_stack.parquet')
print(list(e15.columns))


# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

e15 = api.load_saved('e015_stack.parquet')
tt = api.train_targets()
df = e15.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged:', df.shape)

y = df.future_spend_4w.values
# OOF stack MAE on train rows (stack is LOSO, so this estimates val performance)
mae_stack = np.abs(df.stack_ridge - y).mean()
mae_sp28 = np.abs(df.spend_28 - y).mean()
mae_ew28 = np.abs(df.d_ewma_spend_hl28 - y).mean()
print(f'MAE stack_ridge (train, OOF): {mae_stack:.3f}')
print(f'MAE spend_28: {mae_sp28:.3f}   MAE ewma_hl28: {mae_ew28:.3f}')
for w in [0.3,0.5,0.7]:
    b = w*df.stack_ridge + (1-w)*df.spend_28
    print(f'blend stack+spend28 w={w}: {np.abs(b-y).mean():.3f}')

# error by target level
df['abs_err'] = np.abs(df.stack_ridge - y)
df['ybin'] = pd.qcut(y.replace(0, np.nan), q=8, duplicates='drop')
print(df.groupby('ybin', observed=True).agg(n=('abs_err','size'), mae=('abs_err','mean'), med_pred=('stack_ridge','median'), med_y=('future_spend_4w','median')))

# zero rows: how much MAE comes from them
z = df.future_spend_4w==0
print('\nzero rows:', z.mean(), 'MAE on zero rows:', df.loc[z,'abs_err'].mean(), '-> contributes', z.mean()*df.loc[z,'abs_err'].mean())
print('MAE on nonzero rows:', df.loc[~z,'abs_err'].mean())
print('mean pred on zero rows:', df.loc[z,'stack_ridge'].mean())

# per snapshot
print('\nper-snapshot MAE:')
print(df.groupby('snapshot_day').agg(n=('abs_err','size'), mae=('abs_err','mean'), mean_y=('future_spend_4w','mean'), mean_pred=('stack_ridge','mean')))


# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

e15 = api.load_saved('e015_stack.parquet')
tt = api.train_targets()
df = e15.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values
df['abs_err'] = np.abs(df.stack_ridge - y)

# error by target level
tmp = df[y>0].copy()
tmp['ybin'] = pd.qcut(tmp.future_spend_4w, q=8, duplicates='drop')
print(tmp.groupby('ybin', observed=True).agg(n=('abs_err','size'), mae=('abs_err','mean'), med_pred=('stack_ridge','median'), med_y=('future_spend_4w','median')).round(1))

z = y==0
print('\nzero rows:', round(z.mean(),3), 'MAE on zero rows:', round(df.loc[z,'abs_err'].mean(),2), 'contrib:', round(z.mean()*df.loc[z,'abs_err'].mean(),2))
print('MAE nonzero:', round(df.loc[~z,'abs_err'].mean(),2))
print('mean pred on zero rows:', round(df.loc[z,'stack_ridge'].mean(),2))

print('\nper-snapshot:')
print(df.groupby('snapshot_day').agg(n=('abs_err','size'), mae=('abs_err','mean'), mean_y=('future_spend_4w','mean'), mean_pred=('stack_ridge','mean')).round(1))

# simple peer-median check on train rows: nearest neighbors in (log spend_28, log trips_28) space
from collections import defaultdict
feat = np.column_stack([np.log1p(df.spend_28.values), np.log1p(df.trips_28.values), np.log1p(df.recency.values)])
# scale
mu, sd = feat.mean(0), feat.std(0)+1e-9
F = (feat-mu)/sd
Y = y
# subsample reference for speed
rng = np.random.RandomState(0)
idx_ref = rng.choice(len(F), 8000, replace=False)
Fr, Yr = F[idx_ref], Y[idx_ref]
# for a sample of queries, find 50 NN and take median y
k=50
mae_nn = []; mae_base=[]
qs = rng.choice(len(F), 4000, replace=False)
from numpy.linalg import norm
for q in qs:
    d = ((Fr - F[q])**2).sum(1)
    nn = np.argpartition(d, k)[:k]
    mae_nn.append(abs(np.median(Yr[nn]) - Y[q]))
    mae_base.append(abs(df.stack_ridge.values[q] - Y[q]))
print('\npeer-median kNN(50) MAE (sample):', round(np.mean(mae_nn),2), ' vs stack:', round(np.mean(mae_base),2))


# ---- cell ----
import agent_api as api
import numpy as np, pandas as pd, time

def peer_feats(view, d, ks=(25,50,100)):
    tx = view.table('transactions')
    hh = pd.Index(view.households)
    g = tx.groupby('household_key')
    # current window [d-27, d], previous [d-55, d-28]
    cur = tx[(tx.day>=d-27)&(tx.day<=d)]
    prv = tx[(tx.day>=d-55)&(tx.day<=d-28)]
    sp_cur = cur.groupby('household_key').sales_value.sum()
    tr_cur = cur.groupby('household_key').basket_id.nunique()
    sp_prv = prv.groupby('household_key').sales_value.sum()
    tr_prv = prv.groupby('household_key').basket_id.nunique()
    first = g.day.min(); last = g.day.max()
    idx = first.index
    rec = (d - last).clip(lower=0).reindex(idx).fillna(999).values.astype(float)
    ten = (d - first + 1).values.astype(float)
    X = np.column_stack([
        np.log1p(sp_cur.reindex(idx).fillna(0).values),
        np.log1p(tr_cur.reindex(idx).fillna(0).values),
        np.log1p(rec), np.log1p(ten),
        np.log1p(sp_prv.reindex(idx).fillna(0).values)])
    realized = sp_cur.reindex(idx).fillna(0).values  # for refs: spend d-27..d
    ref_mask = (first.values <= d-112) & (first.index.isin(idx))
    ref_idx = np.where(first.values <= d-112)[0]
    q_idx = idx.get_indexer(hh)
    Xr = X[ref_idx]; yr = realized[ref_idx]
    mu, sd = Xr.mean(0), Xr.std(0)+1e-9
    Zr = (Xr-mu)/sd; Zq = (X[q_idx]-mu)/sd
    out = {}
    nq = Zq.shape[0]
    d2 = ((Zq[:,None,:]-Zr[None,:,:])**2).sum(-1)  # may be big: nq~2500 x nr~2500 fine
    for k in ks:
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        med = np.median(yr[nn], axis=1)
        out[f'knn_med{k}'] = med
        if k==50:
            out['knn_mean50'] = yr[nn].mean(1)
            out['knn_medlog50'] = np.expm1(np.median(np.log1p(yr[nn]), axis=1))
    # drift ratio: median realized/sp_prv over refs
    spp = sp_prv.values[ref_idx]
    ratio = yr[spp>0]/spp[spp>0]
    out['drift_r'] = np.full(nq, np.median(ratio))
    return pd.DataFrame(out, index=hh)

t0=time.time()
v = api.snapshot(as_of_day=431)
pf = peer_feats(v, 431)
print('time:', round(time.time()-t0,1),'s'); print(pf.describe().round(2))

# quick quality check vs train targets at d=431
tt = api.train_targets()
m = pf.join(tt[tt.snapshot_day==431].set_index('household_key').future_spend_4w, how='inner')
y = m.future_spend_4w
for c in pf.columns:
    print(c, 'MAE:', round(np.abs(m[c]-y).mean(),2))
b = 0.5*m.knn_med50+0.5*api.load_saved('e015_stack.parquet').set_index(['household_key','snapshot_day']).loc[(slice(None),431),:]['stack_ridge'].values if False else None
e15 = api.load_saved('e015_stack.parquet')
s431 = e15[e15.snapshot_day==431].set_index('household_key').stack_ridge
mm = pf.join(s431, how='inner').join(tt[tt.snapshot_day==431].set_index('household_key').future_spend_4w, how='inner')
yy = mm.future_spend_4w
print('stack MAE 431:', round(np.abs(mm.stack_ridge-yy).mean(),2))
for w in [0.3,0.5,0.7]:
    print(f'blend w={w}:', round(np.abs(w*mm.stack_ridge+(1-w)*mm.knn_med50-yy).mean(),2))


# ---- cell ----
import agent_api as api
import numpy as np, pandas as pd, time

def peer_feats(view, d, ks=(25,50,100)):
    tx = view.table('transactions')
    hh = pd.Index(np.asarray(view.households))
    cur = tx[(tx.day>=d-27)&(tx.day<=d)]
    prv = tx[(tx.day>=d-55)&(tx.day<=d-28)]
    agg = cur.groupby('household_key').agg(sp=('sales_value','sum'), tr=('basket_id','nunique'))
    agp = prv.groupby('household_key').agg(spp=('sales_value','sum'))
    first = tx.groupby('household_key').day.min()
    last = tx.groupby('household_key').day.max()
    idx = first.index
    rec = np.clip(d - last.reindex(idx).values, 0, None)
    ten = (d - first.values + 1).astype(float)
    sp_c = agg.sp.reindex(idx).fillna(0).values
    tr_c = agg.tr.reindex(idx).fillna(0).values
    sp_p = agp.spp.reindex(idx).fillna(0).values
    X = np.column_stack([np.log1p(sp_c), np.log1p(tr_c), np.log1p(rec), np.log1p(ten), np.log1p(sp_p)])
    ref_idx = np.where(first.values <= d-112)[0]
    q_idx = idx.get_indexer(hh)
    Xr, yr = X[ref_idx], sp_c[ref_idx]
    mu, sd = Xr.mean(0), Xr.std(0)+1e-9
    Zr = (Xr-mu)/sd; Zq = (X[q_idx]-mu)/sd
    d2 = ((Zq[:,None,:]-Zr[None,:,:])**2).sum(-1)
    out = {}
    for k in ks:
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        out[f'knn_med{k}'] = np.median(yr[nn], axis=1)
        if k==50:
            out['knn_mean50'] = yr[nn].mean(1)
            out['knn_medlog50'] = np.expm1(np.median(np.log1p(yr[nn]), axis=1))
    return pd.DataFrame(out, index=hh)

t0=time.time()
v = api.snapshot(as_of_day=431)
pf = peer_feats(v, 431)
print('time:', round(time.time()-t0,1),'s'); print(pf.describe().round(2))

tt = api.train_targets()
e15 = api.load_saved('e015_stack.parquet')
s431 = e15[e15.snapshot_day==431].set_index('household_key')[['stack_ridge']]
mm = pf.join(s431, how='inner').join(tt[tt.snapshot_day==431].set_index('household_key').future_spend_4w, how='inner')
yy = mm.future_spend_4w
print('n:', len(mm), 'stack MAE 431:', round(np.abs(mm.stack_ridge-yy).mean(),2))
for c in pf.columns:
    print(c, 'MAE:', round(np.abs(mm[c]-yy).mean(),2))
for w in [0.3,0.5,0.7]:
    print(f'blend stack w={w}:', round(np.abs(w*mm.stack_ridge+(1-w)*mm.knn_med50-yy).mean(),2))


# ---- cell ----
import agent_api as api
v = api.snapshot(as_of_day=431)
print(type(v.households), repr(v.households)[:200])
try:
    hh = v.households()
    print('callable ->', type(hh), len(hh), hh[:5])
except Exception as e:
    print('not callable:', e)
print('view.day:', v.day, 'view.week:', v.week)
tx = v.table('transactions')
print('tx:', tx.shape)


# ---- cell ----
import agent_api as api
import numpy as np, pandas as pd, time

base = api.baseline_features()
HH = {int(d): pd.Index(np.asarray(base[base.snapshot_day==d].household_key.unique())) for d in sorted(base.snapshot_day.unique())}

def peer_feats(view, d, ks=(25,50,100)):
    tx = view.table('transactions')
    hh = HH[d]
    cur = tx[(tx.day>=d-27)&(tx.day<=d)]
    prv = tx[(tx.day>=d-55)&(tx.day<=d-28)]
    agg = cur.groupby('household_key').agg(sp=('sales_value','sum'), tr=('basket_id','nunique'))
    agp = prv.groupby('household_key').agg(spp=('sales_value','sum'))
    first = tx.groupby('household_key').day.min()
    last = tx.groupby('household_key').day.max()
    idx = first.index
    rec = np.clip(d - last.reindex(idx).values, 0, None)
    ten = (d - first.values + 1).astype(float)
    sp_c = agg.sp.reindex(idx).fillna(0).values
    tr_c = agg.tr.reindex(idx).fillna(0).values
    sp_p = agp.spp.reindex(idx).fillna(0).values
    X = np.column_stack([np.log1p(sp_c), np.log1p(tr_c), np.log1p(rec), np.log1p(ten), np.log1p(sp_p)])
    ref_idx = np.where(first.values <= d-112)[0]
    q_idx = idx.get_indexer(hh)
    Xr, yr = X[ref_idx], sp_c[ref_idx]
    mu, sd = Xr.mean(0), Xr.std(0)+1e-9
    Zr = (Xr-mu)/sd; Zq = (X[q_idx]-mu)/sd
    d2 = ((Zq[:,None,:]-Zr[None,:,:])**2).sum(-1)
    out = {}
    for k in ks:
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        out[f'knn_med{k}'] = np.median(yr[nn], axis=1)
        if k==50:
            out['knn_mean50'] = yr[nn].mean(1)
            out['knn_medlog50'] = np.expm1(np.median(np.log1p(yr[nn]), axis=1))
    return pd.DataFrame(out, index=hh)

t0=time.time()
v = api.snapshot(as_of_day=431)
pf = peer_feats(v, 431)
print('time:', round(time.time()-t0,1),'s'); print(pf.describe().round(2))

tt = api.train_targets()
e15 = api.load_saved('e015_stack.parquet')
s431 = e15[e15.snapshot_day==431].set_index('household_key')[['stack_ridge']]
mm = pf.join(s431, how='inner').join(tt[tt.snapshot_day==431].set_index('household_key').future_spend_4w, how='inner')
yy = mm.future_spend_4w
print('n:', len(mm), 'stack MAE 431:', round(np.abs(mm.stack_ridge-yy).mean(),2))
for c in pf.columns:
    print(c, 'MAE:', round(np.abs(mm[c]-yy).mean(),2))
for w in [0.3,0.5,0.7]:
    print(f'blend stack w={w}:', round(np.abs(w*mm.stack_ridge+(1-w)*mm.knn_med50-yy).mean(),2))


# ---- cell ----
import agent_api as api
import numpy as np, pandas as pd, time

base = api.baseline_features()
HH = {int(d): pd.Index(np.asarray(base[base.snapshot_day==d].household_key.unique())) for d in sorted(base.snapshot_day.unique())}

def peer_feats(view, d):
    tx = view.table('transactions')
    hh = HH[d]
    cur = tx[(tx.day>=d-27)&(tx.day<=d)]
    prv = tx[(tx.day>=d-55)&(tx.day<=d-28)]
    agg = cur.groupby('household_key').agg(sp=('sales_value','sum'), tr=('basket_id','nunique'))
    agp = prv.groupby('household_key').agg(spp=('sales_value','sum'))
    g = tx.groupby('household_key').day
    first, last = g.min(), g.max()
    idx = first.index
    rec = np.clip(d - last.reindex(idx).values, 0, None)
    ten = (d - first.values + 1).astype(float)
    sp_c = agg.sp.reindex(idx).fillna(0).values
    tr_c = agg.tr.reindex(idx).fillna(0).values
    sp_p = agp.spp.reindex(idx).fillna(0).values
    X = np.column_stack([np.log1p(sp_c), np.log1p(tr_c), np.log1p(rec), np.log1p(ten), np.log1p(sp_p)])
    # ref pool: tenure >= 84d, relax if too small
    for thr in (84, 56, 28, 0):
        ref_idx = np.where(first.values <= d-thr)[0]
        if len(ref_idx) >= 300: break
    q_idx = idx.get_indexer(hh)
    miss = q_idx < 0
    if miss.any(): q_idx[miss] = 0
    Xr, yr = X[ref_idx], sp_c[ref_idx]
    mu, sd = Xr.mean(0), Xr.std(0)+1e-9
    Zr = (Xr-mu)/sd; Zq = (X[q_idx]-mu)/sd
    d2 = ((Zq[:,None,:]-Zr[None,:,:])**2).sum(-1)
    out = {}
    for k in (25, 50):
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        out[f'knn_med{k}'] = np.median(yr[nn], axis=1)
        if k==50:
            out['knn_mean50'] = yr[nn].mean(1)
            out['knn_medlog50'] = np.expm1(np.median(np.log1p(yr[nn]), axis=1))
    df = pd.DataFrame(out, index=hh)
    if miss.any(): df.loc[hh[miss]] = np.nan
    return df

for d in (95, 431):
    t0=time.time()
    v = api.snapshot(as_of_day=d)
    pf = peer_feats(v, d)
    tt = api.train_targets()
    e15 = api.load_saved('e015_stack.parquet')
    sd_ = e15[e15.snapshot_day==d].set_index('household_key')[['stack_ridge']]
    mm = pf.join(sd_, how='inner').join(tt[tt.snapshot_day==d].set_index('household_key').future_spend_4w, how='inner')
    yy = mm.future_spend_4w
    print(f'd={d} n={len(mm)} refs_ok time={time.time()-t0:.1f}s  stack MAE: {np.abs(mm.stack_ridge-yy).mean():.2f}')
    for c in pf.columns:
        print('  ', c, round(np.abs(mm[c]-yy).mean(),2))
    for w in (0.4,0.5,0.6):
        print(f'   blend w={w}:', round(np.abs(w*mm.stack_ridge+(1-w)*mm.knn_med25-yy).mean(),2))


# ---- cell ----
import agent_api as api
import numpy as np, pandas as pd, time

base = api.baseline_features()
HH = {int(d): pd.Index(np.asarray(base[base.snapshot_day==d].household_key.unique())) for d in sorted(base.snapshot_day.unique())}

def fn(view, snapshot_day):
    d = snapshot_day
    tx = view.table('transactions')
    hh = HH[d]
    cur = tx[(tx.day>=d-27)&(tx.day<=d)]
    prv = tx[(tx.day>=d-55)&(tx.day<=d-28)]
    agg = cur.groupby('household_key').agg(sp=('sales_value','sum'), tr=('basket_id','nunique'))
    agp = prv.groupby('household_key').agg(spp=('sales_value','sum'))
    g = tx.groupby('household_key').day
    first, last = g.min(), g.max()
    idx = first.index
    rec = np.clip(d - last.reindex(idx).values, 0, None)
    ten = (d - first.values + 1).astype(float)
    sp_c = agg.sp.reindex(idx).fillna(0).values
    tr_c = agg.tr.reindex(idx).fillna(0).values
    sp_p = agp.spp.reindex(idx).fillna(0).values
    X = np.column_stack([np.log1p(sp_c), np.log1p(tr_c), np.log1p(rec), np.log1p(ten), np.log1p(sp_p)])
    ref_idx = np.where(first.values <= d-84)[0]
    pos_of_hh = {k: p for p, k in enumerate(idx[ref_idx])}
    q_idx = idx.get_indexer(hh)
    Xr, yr = X[ref_idx], sp_c[ref_idx]
    mu, sd = Xr.mean(0), Xr.std(0)+1e-9
    Zr = (Xr-mu)/sd; Zq = (X[q_idx]-mu)/sd
    d2 = ((Zq[:,None,:]-Zr[None,:,:])**2).sum(-1)
    # exclude self from reference pool
    for i, k in enumerate(np.asarray(hh)):
        p = pos_of_hh.get(k)
        if p is not None: d2[i, p] = np.inf
    out = {}
    for k in (25, 50):
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        out[f'knn_med{k}'] = np.median(yr[nn], axis=1)
        if k==50:
            out['knn_mean50'] = yr[nn].mean(1)
            out['knn_medlog50'] = np.expm1(np.median(np.log1p(yr[nn]), axis=1))
    return pd.DataFrame(out, index=hh)

t0 = time.time()
tab = api.build_features(fn)
print('built:', tab.shape, 'time:', round(time.time()-t0,1))
print(tab.groupby('snapshot_day').knn_med50.mean().round(1))
p = api.save_table(tab, 'e019_peer')
print('saved:', p)


# ---- cell ----
import agent_api as api
import pandas as pd

e15 = api.load_saved('e015_stack.parquet')
peer = api.load_saved('e019_peer.parquet')
m = e15.merge(peer, on=['household_key','snapshot_day'], how='inner')
print(m.shape, list(peer.columns))
p = api.save_table(m, 'e019_merged')
print('saved:', p)
