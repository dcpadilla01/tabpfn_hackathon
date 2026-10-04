import agent_api, pandas as pd, numpy as np, datetime

t = agent_api.load_saved('e013_denoise.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
Xdf = df[feat_cols]
M = np.empty((len(df), len(feat_cols)), dtype=np.float64)
for j,c in enumerate(feat_cols):
    col = Xdf[c].values.astype(np.float64)
    col[~np.isfinite(col)] = np.nan
    M[:,j] = col
train_mask = df['future_spend_4w'].notna().values
y = df['future_spend_4w'].values.astype(np.float64)
hh = df['household_key'].values
uq = np.unique(hh); rng0 = np.random.RandomState(7); rng0.shuffle(uq)
hh_hold = set(uq[:int(len(uq)*0.3)])
ho = np.array([h in hh_hold for h in hh]) & train_mask
tr = train_mask & ~ho
med = np.nanmedian(M[train_mask], axis=0); med = np.where(np.isfinite(med), med, 0.0)
M = np.where(np.isnan(M), med[None,:], M)
q99 = np.quantile(np.abs(M[train_mask]), 0.999, axis=0)
M = np.clip(M, -q99[None,:], q99[None,:])
mu = M[train_mask].mean(0); sd = M[train_mask].std(0); sd[sd<1e-9]=1
Z = (M-mu)/sd

NB = 33
bins = np.empty((len(df), len(feat_cols)), dtype=np.uint8)
for j in range(len(feat_cols)):
    col = M[:,j]
    v = col[train_mask]
    edges = np.unique(np.quantile(v, np.linspace(0,1,NB)[1:-1]))
    cc = np.where(np.isnan(col), np.inf, col)
    b = np.minimum(np.searchsorted(edges, cc, side='right'), NB-2)
    b[np.isnan(col)] = NB-1
    bins[:,j] = b.astype(np.uint8)
print('bins done', flush=True)

def build_tree(btr, grad, resid, idx, fids, depth, min_leaf, rl):
    nodes = []
    def add(sidx, d):
        G = grad[sidx].sum(); H = float(len(sidx))
        nid = len(nodes)
        nodes.append({'leaf':True,'val':float(np.median(resid[sidx])),'f':-1,'thr':-1,'left':-1,'right':-1})
        if d >= depth or len(sidx) < 2*min_leaf:
            return nid
        bs = btr[np.ix_(sidx, fids)]
        gs = grad[sidx]
        best = None
        for kk in range(bs.shape[1]):
            b = bs[:,kk]
            hg = np.bincount(b, weights=gs, minlength=NB).cumsum()[:-1]
            hc = np.bincount(b, minlength=NB).cumsum()[:-1]
            nl = hc; nr = len(sidx)-hc
            ok = (nl>=min_leaf)&(nr>=min_leaf)
            if not ok.any(): continue
            GR = G-hg; HR = H-hc
            gain = hg*hg/(nl+rl) + GR*GR/(HR+rl) - G*G/(H+rl)
            gain[~ok] = -1e18
            jj = int(np.argmax(gain))
            if gain[jj] > 1e-6 and (best is None or gain[jj] > best[0]):
                best = (gain[jj], kk, jj)
        if best is None: return nid
        _, kk, thr = best
        mask = bs[:,kk] <= thr
        li = sidx[mask]; ri = sidx[~mask]
        if len(li)==0 or len(ri)==0: return nid
        nodes[nid].update({'leaf':False,'f':int(fids[kk]),'thr':int(thr)})
        nodes[nid]['left'] = add(li, d+1); nodes[nid]['right'] = add(ri, d+1)
        return nid
    return add(idx, 0), nodes

def tree_predict(nodes, brows):
    assign = np.zeros(brows.shape[0], dtype=np.int32)
    vals = np.array([nd['val'] for nd in nodes])
    for nid, nd in enumerate(nodes):
        if nd['leaf']: continue
        at = np.where(assign==nid)[0]
        if len(at)==0: continue
        gl = brows[at, nd['f']] <= nd['thr']
        assign[at[gl]] = nd['left']; assign[at[~gl]] = nd['right']
    return vals[assign]

# OLS on all train rows
A = np.hstack([Z[train_mask], np.ones((train_mask.sum(),1))])
coef,*_ = np.linalg.lstsq(A, y[train_mask], rcond=None)
B = np.hstack([Z, np.ones((len(df),1))])
ols_all = B@coef
resid0 = y[train_mask] - ols_all[train_mask]
print('OLS(all-train) holdout MAE %.3f' % np.abs(ols_all[ho]-y[ho]).mean(), flush=True)

# boosting with OOB accumulation
NT = 600; SEEDS = (1,2); lr=0.05; dep=4; ml=100; rl=5.0; rs=0.75; cs=0.5
n_all = len(df); n_tr = int(train_mask.sum())
full_sum = np.zeros(n_all); oob_sum = np.zeros(n_all); oob_cnt = np.zeros(n_all)
t0 = datetime.datetime.now()
btr_full = bins[train_mask]
for s in SEEDS:
    rng = np.random.RandomState(s)
    cur = resid0.copy()
    for it in range(NT):
        grad = -np.sign(cur)
        rows = rng.rand(n_tr) < rs
        ridx = np.where(rows)[0]
        fids = np.where(rng.rand(len(feat_cols)) < cs)[0]
        if len(fids) < 10: fids = np.arange(len(feat_cols))
        root, nodes = build_tree(btr_full, grad, cur, ridx, fids, dep, ml, rl)
        pred_all = tree_predict(nodes, bins)  # all 36k rows
        full_sum += lr*pred_all
        oobm = ~rows
        oob_sum[np.where(oobm)[0]] += lr*pred_all[train_mask][oobm] if False else 0  # placeholder
        # correct OOB accumulation: indices in full array
        tr_idx = np.where(train_mask)[0]
        oob_full_pos = tr_idx[oobm]
        oob_sum[oob_full_pos] += lr*pred_all[oob_full_pos]
        oob_cnt[oob_full_pos] += 1
        cur[ridx] -= lr*pred_all[train_mask][ridx]
dt = (datetime.datetime.now()-t0).total_seconds()
print('boost done %.0fs' % dt, flush=True)
corr_full = full_sum/len(SEEDS)
corr_oob = oob_sum/np.maximum(oob_cnt,1)
print('oob_cnt min/med:', oob_cnt.min(), np.median(oob_cnt))
print('corr_full[train] std %.1f | corr_oob[train] std %.1f' % (corr_full[train_mask].std(), corr_oob[train_mask].std()))
print('corr_full[val] std %.1f' % corr_full[~train_mask].std())

# direct stacked on holdout (full-model corr)
p_direct = np.clip(ols_all + corr_full, 0, 2500)
print('direct stacked holdout MAE %.3f' % np.abs(p_direct[ho]-y[ho]).mean())

# SIMULATE evaluator: OLS on [Z, corr_oob] over train rows -> apply to holdout with corr_full
Xtr = np.hstack([Z[tr], corr_oob[tr][:,None]])
Xho = np.hstack([Z[ho], corr_full[ho][:,None]])
c,*_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
print('SIM evaluator-OLS with OOB corr: MAE %.3f' % np.abs(Xho@c - y[ho]).mean())
# what weight does corr get?
w = c[-1]; print('corr weight w=%.3f' % w)
# also ridge-ish check: with corr only (no raw features)
c2,*_ = np.linalg.lstsq(corr_oob[tr][:,None], y[tr]-0, rcond=None)
print('corr-only coef %.3f' % c2[0])

# save final table: train rows get OOB corr, val rows get full corr
out = t.copy()
out['gbm_corr'] = np.where(train_mask, corr_oob, corr_full)
print('gbm_corr quantiles:', np.quantile(out['gbm_corr'], [0,.01,.5,.99,1]).round(1))
agent_api.save_table(out, 'e014_gbm_oob.parquet')
print('saved e014_gbm_oob.parquet', out.shape)