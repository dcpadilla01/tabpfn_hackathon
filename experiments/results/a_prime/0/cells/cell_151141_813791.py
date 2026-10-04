import numpy as np, pandas as pd
tt = agent_api.train_targets()
mkt = agent_api.load_saved("mkt_v2.parquet")
feat=[c for c in mkt.columns if c not in ("household_key","snapshot_day")]

def dist_block(view, s):
    t = view.transactions
    hh = view.households
    out = pd.DataFrame(index=hh)
    first = t.groupby("household_key").day.min()
    ten = (s - first).astype(float)
    nb = np.floor(ten/28).astype(int).clip(lower=0, upper=13)
    # aligned prior blocks: k=1..13 window (s-28k, s-28k+28]
    cols = {}
    for k in range(1, 14):
        lo, hi = s-28*k, s-28*k+28
        cols[k] = t[(t.day > lo) & (t.day <= hi)].groupby("household_key").sales_value.sum()
    M = pd.DataFrame(cols)
    avail = {}
    for h in M.index:
        n = int(nb.get(h, 0))
        avail[h] = M.loc[h, [k for k in range(1,14) if k <= n]].values.astype(float) if n>0 else np.array([])
    def stat(fn):
        return pd.Series({h: fn(v) for h, v in avail.items()})
    out["d_nblk"] = pd.Series({h: len(v) for h,v in avail.items()})
    out["d_med"] = stat(lambda v: np.median(v) if len(v) else np.nan)
    out["d_mean"] = stat(lambda v: np.mean(v) if len(v) else np.nan)
    out["d_std"] = stat(lambda v: np.std(v) if len(v) else np.nan)
    out["d_iqr"] = stat(lambda v: (np.percentile(v,75)-np.percentile(v,25)) if len(v) else np.nan)
    out["d_zerofrac"] = stat(lambda v: np.mean(v<=1.0) if len(v) else np.nan)
    out["d_max"] = stat(lambda v: np.max(v) if len(v) else np.nan)
    out["d_min"] = stat(lambda v: np.min(v) if len(v) else np.nan)
    s28 = t[t.day>s-28].groupby("household_key").sales_value.sum()
    out["d_cur_over_med"] = s28/out["d_med"]
    out["d_last_zero"] = (s28<=1.0).astype(float)
    return out

D = agent_api.build_features(dist_block)
print("dist built:", D.shape, list(D.columns))
dl = [c for c in D.columns if c.startswith("d_")]
Dfull = mkt.merge(D.reset_index(), on=["household_key","snapshot_day"], how="inner")
print("null rates:", {c: round(Dfull[c].isna().mean(),3) for c in dl})
agent_api.save_table(Dfull, "e008_dist.parquet")

def prep_cols(df, cols, tr, va):
    Xtr=df.loc[tr, cols].astype(float).values; ytr=df.loc[tr].future_spend_4w.values
    Xva=df.loc[va, cols].astype(float).values; yva=df.loc[va].future_spend_4w.values
    med=np.nanmedian(Xtr,0); med=np.where(np.isnan(med),0,med)
    Xtr=np.where(np.isnan(Xtr),med,Xtr); Xva=np.where(np.isnan(Xva),med,Xva)
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
    return (Xtr-mu)/sd, ytr, (Xva-mu)/sd, yva

def ridge2(df, cols, alphas=(10,100,300,1000,3000,10000), tr_snaps=[95,123,151,179,207,235,263,291,319,347], va_snaps=[375,403,431]):
    tr=df.snapshot_day.isin(tr_snaps); va=df.snapshot_day.isin(va_snaps)
    cols=[c for c in cols if c in df.columns]
    Xtr,ytr,Xva,yva = prep_cols(df, cols, tr, va)
    Xtr=np.c_[np.ones(len(Xtr)),Xtr]; Xva=np.c_[np.ones(len(Xva)),Xva]
    best=(1e9,None)
    for a in alphas:
        A=Xtr.T@Xtr+a*np.eye(Xtr.shape[1]); A[-1,-1]-=a
        w=np.linalg.solve(A,Xtr.T@ytr); p=np.clip(Xva@w,0,None)
        m=np.mean(np.abs(p-yva))
        if m<best[0]: best=(m,a)
    return best

# calibration vs harness: harness order E003(63.32) < E006(63.38) < E005(63.41) < E007(63.44)
rob = agent_api.load_saved("e007_robust.parquet"); tmp = agent_api.load_saved("e006_temporal.parquet")
mix = agent_api.load_saved("mix_v1.parquet")
rf=[c for c in rob.columns if c not in ("household_key","snapshot_day") and c not in feat]
tf=[c for c in tmp.columns if c not in ("household_key","snapshot_day") and c not in feat]
mf=[c for c in mix.columns if c not in ("household_key","snapshot_day")]
ttm = tt.merge(mkt, on=["household_key","snapshot_day"], how="inner")
print("CALIB E003:", ridge2(ttm, feat))
print("CALIB E005(E003+mix):", ridge2(ttm.merge(mix,on=["household_key","snapshot_day"],how="left"), feat+mf))
print("CALIB E006(+temporal):", ridge2(ttm.merge(tmp,on=["household_key","snapshot_day"],how="left"), feat+tf))
print("CALIB E007(+robust):", ridge2(ttm.merge(rob,on=["household_key","snapshot_day"],how="left"), feat+rf))
Dm = ttm.merge(D.reset_index(), on=["household_key","snapshot_day"], how="left")
print("RIDGE E003+dist:", ridge2(Dm, feat+dl))

# tiny hand-rolled GBM proxy (depth-3, squared loss)
def gbm_eval(df, cols, rounds=80, lr=0.15, max_depth=3, min_leaf=60):
    try:
        tr=df.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347]); va=df.snapshot_day.isin([375,403,431])
        cols=[c for c in cols if c in df.columns]
        Xtr,ytr,Xva,yva = prep_cols(df, cols, tr, va)
        n,d = Xtr.shape
        pred = np.full(n, ytr.mean()); pva = np.full(len(Xva), ytr.mean())
        rng = np.random.RandomState(0)
        for r in range(rounds):
            g = ytr - pred
            trees = []
            idx_nodes = [np.arange(n)]; pv_nodes=[np.arange(len(Xva))]
            # grow tree level by level
            node_assign = np.zeros(n, dtype=int); node_assign_va = np.zeros(len(Xva), dtype=int)
            leaves = {0: np.arange(n)}; leaves_va = {0: np.arange(len(Xva))}
            leaf_val = {}
            for depth in range(max_depth):
                newleaves={}; newleaves_va={}; newleaf_val={}
                for nid, idx in leaves.items():
                    Xn, gn = Xtr[idx], g[idx]
                    if len(idx) < 2*min_leaf:
                        newleaves[nid]=idx; newleaves_va[nid]=leaves_va[nid]; newleaf_val[nid]=gn.mean(); continue
                    best=( -1e18, None, None)
                    for j in range(d):
                        xs = Xn[:,j]
                        o = np.argsort(xs, kind="stable"); xs_s=xs[o]; gs=gn[o]
                        cs=np.cumsum(gs); tot=cs[-1]; cnt=len(idx)
                        cl=cs; cr=tot-cs; nl=np.arange(1,cnt+1); nr=cnt-nl
                        valid=(xs_s[1:]>xs_s[:-1]) & (nl>=min_leaf) & (nr>=min_leaf)
                        if not valid.any(): continue
                        gain = cl[:-1]**2/(nl[:-1]+1e-9) + cr[:-1]**2/(nr[:-1]+1e-9) - tot**2/(cnt+1e-9)
                        gain = np.where(valid, gain, -1e18)
                        k = int(np.argmax(gain))
                        if gain[k] > best[0]: best=(gain[k], j, (xs_s[k]+xs_s[k+1])/2)
                    if best[1] is None:
                        newleaves[nid]=idx; newleaves_va[nid]=leaves_va[nid]; newleaf_val[nid]=gn.mean(); continue
                    j, thr = best[1], best[2]
                    msk = Xn[:,j] <= thr
                    li = idx[msk]; ri = idx[~msk]
                    lva = leaves_va[nid][Xva[leaves_va[nid],j] <= thr]; rva = leaves_va[nid][Xva[leaves_va[nid],j] > thr]
                    newleaves[2*nid+1]=li; newleaves[2*nid+2]=ri
                    newleaves_va[2*nid+1]=lva; newleaves_va[2*nid+2]=rva
                    newleaf_val[2*nid+1]=g[li].mean(); newleaf_val[2*nid+2]=g[ri].mean()
                leaves, leaves_va, leaf_val = newleaves, newleaves_va, newleaf_val
            upd = np.zeros(n); upd_va = np.zeros(len(Xva))
            for nid, idx in leaves.items():
                upd[idx] = leaf_val[nid]
                upd_va[leaves_va[nid]] = leaf_val[nid]
            pred += lr*upd; pva += lr*upd_va
        pva = np.clip(pva, 0, None)
        return round(float(np.mean(np.abs(pva-yva))),3)
    except Exception as e:
        return f"GBM proxy failed: {e}"
print("GBM E003:", gbm_eval(ttm, feat))
print("GBM E003+dist:", gbm_eval(Dm, feat+dl))