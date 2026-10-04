import agent_api as A
import pandas as pd, numpy as np

names = ["e015_base","e013_union","e009_macro","e006_seq_gaps","cand_new","e001_history","e002_marketing",
         "e003_dept_mix","e004_long_hist","e005_seasonal_peer","e007_new","e008_decomp2","e010_composite",
         "e011_display","e012_full","e014_base"]
tabs = {}
for nm in names:
    df = A.load_saved(nm + ".parquet")
    tabs[nm] = df
    print(f"{nm}: shape={df.shape}")

base = tabs["e015_base"]
bcols = set(base.columns)
print("\n--- extra columns vs e015_base ---")
for nm in names:
    extra = [c for c in tabs[nm].columns if c not in bcols]
    print(f"{nm}: n_extra={len(extra)}")
    if 0 < len(extra) <= 30:
        print("   ", extra)

bl = A.baseline_features()
print("\nbaseline_features cols:", list(bl.columns))
print("baseline shape:", bl.shape)

tt = A.train_targets()
print("\ntrain_targets:", tt.shape)
print(tt.future_spend_4w.describe())
print("\nshare zeros:", (tt.future_spend_4w==0).mean())
print(tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median","count"]))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

base = A.load_saved("e015_base.parquet")
print("e015_base cols:")
for c in base.columns: print("  ", c)

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np
for nm in ["cand_new","e006_seq_gaps","e012_full","e004_long_hist","e007_new"]:
    df = A.load_saved(nm + ".parquet")
    print(f"--- {nm} ({df.shape[1]-3} feats) ---")
    print([c for c in df.columns if c not in ("household_key","snapshot_day")])

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()

def design(df, num_cols, cat_cols, tr_mask):
    Xn = df[num_cols].astype(float).copy()
    med = Xn[tr_mask].median()
    Xn = Xn.fillna(med).fillna(0.0)
    mu = Xn[tr_mask].mean(); sd = Xn[tr_mask].std().replace(0,1)
    Xn = (Xn - mu) / sd
    mats = [Xn.values]
    for c in cat_cols:
        d = pd.get_dummies(df[c].astype(str), prefix=c, dummy_na=True)
        d = d.reindex(columns=cat_levels[c], fill_value=0)
        mats.append(d.values.astype(float))
    return np.hstack(mats)

def fit_ridge(X, y, alpha):
    A_ = X.T @ X + alpha * np.eye(X.shape[1])
    b = X.T @ y
    return np.linalg.solve(A_, b)

def local_eval(df, alpha_grid=(0.3,1,3,10,30,100), log_target=False):
    M = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    num_cols = [c for c in M.columns if c not in ("household_key","snapshot_day","future_spend_4w")
                and (M[c].dtype.kind in "ifb")]
    cat_cols = [c for c in M.columns if c not in ("household_key","snapshot_day","future_spend_4w")
                and M[c].dtype.kind not in "ifb" and M[c].nunique() <= 25]
    global cat_levels
    cat_levels = {}
    for c in cat_cols:
        cat_levels[c] = list(pd.get_dummies(M[c].astype(str), dummy_na=True).columns)
    inner_tr = M.snapshot_day <= 347
    inner_va = M.snapshot_day == 375
    tr = M.snapshot_day <= 375
    va = M.snapshot_day >= 403
    Xi = design(M, num_cols, cat_cols, inner_tr)
    Xtr = design(M, num_cols, cat_cols, tr)
    yi = M.future_spend_4w.values[inner_tr]
    yvi = M.future_spend_4w.values[inner_va]
    ytr = M.future_spend_4w.values[tr]
    yva = M.future_spend_4w.values[va]
    f = (np.log1p if log_target else (lambda z: z))
    finv = (np.expm1 if log_target else (lambda z: z))
    best = (None, 1e18)
    for a in alpha_grid:
        b = fit_ridge(Xi[inner_tr.values], f(yi), a)
        p = finv(np.clip(Xi[inner_va.values] @ b, -20, 20))
        m = np.mean(np.abs(p - yvi))
        if m < best[1]: best = (a, m)
    a = best[0]
    b = fit_ridge(Xtr[tr.values], f(ytr), a)
    p = finv(np.clip(Xtr[va.values] @ b, -20, 20))
    mae = np.mean(np.abs(p - yva))
    return mae, a, len(num_cols)+sum(len(v) for v in cat_levels.values())

harness = {"e015_base":60.761,"e013_union":61.337,"e009_macro":61.647,"e011_display":61.680,
           "e012_full":61.471,"e008_decomp2":61.711,"e007_new":62.794,"e001_history":63.025,
           "e014_base":63.242,"e010_composite":63.939,"e004_long_hist":63.982,"e006_seq_gaps":64.050,
           "e003_dept_mix":64.175,"e002_marketing":64.676,"e005_seasonal_peer":67.852}
names = list(harness.keys())
res = {}
for nm in names:
    df = A.load_saved(nm + ".parquet")
    df = df.drop(columns=[c for c in ("index","hh_id") if c in df.columns])
    m, a, nf = local_eval(df)
    res[nm] = (m, a, nf, harness[nm])
    print(f"{nm:22s} local={m:7.3f} (a={a}) harness={harness[nm]:7.3f}")
lv = np.array([res[n][0] for n in names]); hv = np.array([res[n][3] for n in names])
print("rank corr:", np.corrcoef(lv.rank() if hasattr(lv,'rank') else pd.Series(lv).rank(), pd.Series(hv).rank())[0,1])
print("pearson:", np.corrcoef(lv, hv)[0,1])

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()

def dedup(df):
    return df.loc[:, ~pd.Index(df.columns).duplicated()]

def design(df, num_cols, cat_cols, tr_mask):
    Xn = df[num_cols].astype(float).copy()
    med = Xn[tr_mask].median()
    Xn = Xn.fillna(med).fillna(0.0)
    mu = Xn[tr_mask].mean(); sd = Xn[tr_mask].std().replace(0,1)
    Xn = (Xn - mu) / sd
    mats = [Xn.values]
    for c in cat_cols:
        d = pd.get_dummies(df[c].astype(str), prefix=c, dummy_na=True)
        d = d.reindex(columns=cat_levels[c], fill_value=0)
        mats.append(d.values.astype(float))
    return np.hstack(mats)

def fit_ridge(X, y, alpha):
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)

def local_eval(df, alpha_grid=(0.3,1,3,10,30,100), log_target=False):
    df = dedup(df)
    M = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    feat = [c for c in M.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
    num_cols = [c for c in feat if M[c].dtype.kind in "ifb"]
    cat_cols = [c for c in feat if M[c].dtype.kind not in "ifb" and M[c].nunique() <= 25]
    global cat_levels
    cat_levels = {}
    for c in cat_cols:
        cat_levels[c] = list(pd.get_dummies(M[c].astype(str), dummy_na=True).columns)
    inner_tr = (M.snapshot_day <= 347).values
    inner_va = (M.snapshot_day == 375).values
    tr = (M.snapshot_day <= 375).values
    va = (M.snapshot_day >= 403).values
    Xi = design(M, num_cols, cat_cols, inner_tr)
    Xtr = design(M, num_cols, cat_cols, tr)
    yi, yvi = M.future_spend_4w.values[inner_tr], M.future_spend_4w.values[inner_va]
    ytr, yva = M.future_spend_4w.values[tr], M.future_spend_4w.values[va]
    f = (np.log1p if log_target else (lambda z: z))
    finv = (np.expm1 if log_target else (lambda z: z))
    best = (None, 1e18)
    for a in alpha_grid:
        b = fit_ridge(Xi[inner_tr], f(yi), a)
        p = finv(np.clip(Xi[inner_va] @ b, -20, 20))
        m = np.mean(np.abs(p - yvi))
        if m < best[1]: best = (a, m)
    a = best[0]
    b = fit_ridge(Xtr[tr], f(ytr), a)
    p = finv(np.clip(Xtr[va] @ b, -20, 20))
    return np.mean(np.abs(p - yva)), a, Xi.shape[1]

harness = {"e015_base":60.761,"e013_union":61.337,"e009_macro":61.647,"e011_display":61.680,
           "e012_full":61.471,"e008_decomp2":61.711,"e007_new":62.794,"e001_history":63.025,
           "e014_base":63.242,"e010_composite":63.939,"e004_long_hist":63.982,"e006_seq_gaps":64.050,
           "e003_dept_mix":64.175,"e002_marketing":64.676,"e005_seasonal_peer":67.852}
names = list(harness.keys())
res = {}
for nm in names:
    df = A.load_saved(nm + ".parquet")
    m, a, nf = local_eval(df)
    res[nm] = (m, a, nf, harness[nm])
    print(f"{nm:22s} local={m:7.3f} (a={a}, k={nf}) harness={harness[nm]:7.3f}")
lv = pd.Series([res[n][0] for n in names]); hv = pd.Series([res[n][3] for n in names])
print("rank corr:", lv.rank().corr(hv.rank()), "| pearson:", lv.corr(hv))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
harness = {"e015_base":60.761,"e013_union":61.337,"e009_macro":61.647,"e011_display":61.680,
           "e012_full":61.471,"e008_decomp2":61.711,"e007_new":62.794,"e001_history":63.025,
           "e014_base":63.242,"e010_composite":63.939,"e004_long_hist":63.982,"e006_seq_gaps":64.050,
           "e003_dept_mix":64.175,"e002_marketing":64.676,"e005_seasonal_peer":67.852}
for nm in harness:
    df = A.load_saved(nm + ".parquet")
    dupcols = pd.Index(df.columns)[pd.Index(df.columns).duplicated()].tolist()
    M = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    Mdup = pd.Index(M.columns)[pd.Index(M.columns).duplicated()].tolist()
    if dupcols or Mdup:
        print(nm, "dup in df:", dupcols, "| dup in M:", Mdup)
    # check index duplication
    print(nm, "idx dup:", M.index.duplicated().sum(), "shape", M.shape)

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()

def local_eval(df, alpha_grid=(0.3,1,3,10,30,100)):
    M = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    feat = [c for c in M.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
    num_cols = [c for c in feat if M[c].dtype.kind in "ifb"]
    cat_cols = [c for c in feat if M[c].dtype.kind not in "ifb" and M[c].nunique() <= 25]
    Xn = M[num_cols].astype(float).copy()
    tr_mask_full = (M.snapshot_day <= 375).values
    med = Xn[tr_mask_full].median()
    Xn = Xn.fillna(med).fillna(0.0)
    mu = Xn[tr_mask_full].mean(); sd = Xn[tr_mask_full].std().replace(0,1)
    Xn = ((Xn - mu) / sd).values
    cat_mats = []
    for c in cat_cols:
        d = pd.get_dummies(M[c].astype(str), prefix=c, dummy_na=True)
        cat_mats.append(d.values.astype(float))
    X = np.hstack([Xn] + cat_mats) if cat_mats else Xn
    inner_tr = (M.snapshot_day <= 347).values
    inner_va = (M.snapshot_day == 375).values
    va = (M.snapshot_day >= 403).values
    y = M.future_spend_4w.values
    def fit(Xs, ys, a):
        return np.linalg.solve(Xs.T @ Xs + a*np.eye(Xs.shape[1]), Xs.T @ ys)
    best = (None, 1e18)
    for a in alpha_grid:
        b = fit(X[inner_tr], y[inner_tr], a)
        p = np.clip(X[inner_va] @ b, -20, 20)
        m = np.mean(np.abs(p - y[inner_va]))
        if m < best[1]: best = (a, m)
    a = best[0]
    b = fit(X[tr_mask_full], y[tr_mask_full], a)
    p = np.clip(X[va] @ b, -20, 20)
    return np.mean(np.abs(p - y[va])), a, X.shape[1]

harness = {"e015_base":60.761,"e013_union":61.337,"e009_macro":61.647,"e011_display":61.680,
           "e012_full":61.471,"e008_decomp2":61.711,"e007_new":62.794,"e001_history":63.025,
           "e014_base":63.242,"e010_composite":63.939,"e004_long_hist":63.982,"e006_seq_gaps":64.050,
           "e003_dept_mix":64.175,"e002_marketing":64.676,"e005_seasonal_peer":67.852}
res = {}
for nm in harness:
    df = A.load_saved(nm + ".parquet")
    m, a, nf = local_eval(df)
    res[nm] = (m, harness[nm])
    print(f"{nm:22s} local={m:7.3f} (a={a}, k={nf}) harness={harness[nm]:7.3f}")
lv = pd.Series([res[n][0] for n in harness]); hv = pd.Series([res[n][1] for n in harness])
print("rank corr:", lv.rank().corr(hv.rank()), "| pearson:", lv.corr(hv))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
harness = {"e015_base":60.761,"e013_union":61.337,"e009_macro":61.647,"e011_display":61.680,
           "e012_full":61.471,"e008_decomp2":61.711,"e007_new":62.794,"e001_history":63.025,
           "e014_base":63.242,"e010_composite":63.939,"e004_long_hist":63.982,"e006_seq_gaps":64.050,
           "e003_dept_mix":64.175,"e002_marketing":64.676,"e005_seasonal_peer":67.852}

def prep(M, tr_mask):
    feat = [c for c in M.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
    num_cols = [c for c in feat if M[c].dtype.kind in "ifb"]
    cat_cols = [c for c in feat if M[c].dtype.kind not in "ifb" and M[c].nunique() <= 25]
    Xn = M[num_cols].astype(float).copy()
    # compress heavy tails: sign(x)*log1p(|x|)
    Xn = np.sign(Xn) * np.log1p(np.abs(Xn))
    Xn = pd.DataFrame(Xn, columns=num_cols)
    med = Xn[tr_mask].median()
    Xn = Xn.fillna(med).fillna(0.0)
    mu = Xn[tr_mask].mean(); sd = Xn[tr_mask].std().replace(0,1)
    Xn = ((Xn - mu)/sd).values
    mats = [Xn]
    for c in cat_cols:
        d = pd.get_dummies(M[c].astype(str), prefix=c, dummy_na=True)
        mats.append(d.values.astype(float))
    return np.hstack(mats)

def local_eval(df):
    M = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    trm = (M.snapshot_day <= 375).values
    X = prep(M, trm)
    inner_tr = (M.snapshot_day <= 347).values
    inner_va = (M.snapshot_day == 375).values
    va = (M.snapshot_day >= 403).values
    y = M.future_spend_4w.values
    ly = np.log1p(y)
    def fit(Xs, ys, a):
        return np.linalg.solve(Xs.T @ Xs + a*np.eye(Xs.shape[1]), Xs.T @ ys)
    best = (None, 1e18)
    for a in (0.3,1,3,10,30,100,300):
        b = fit(X[inner_tr], ly[inner_tr], a)
        p = np.expm1(np.clip(X[inner_va] @ b, -5, 5))
        m = np.mean(np.abs(p - y[inner_va]))
        if m < best[1]: best = (a, m)
    a = best[0]
    b = fit(X[trm], ly[trm], a)
    p = np.expm1(np.clip(X[va] @ b, -5, 5))
    return np.mean(np.abs(p - y[va])), a

res = {}
for nm in harness:
    df = A.load_saved(nm + ".parquet")
    m, a = local_eval(df)
    res[nm] = m
    print(f"{nm:22s} local={m:7.3f} (a={a}) harness={harness[nm]:7.3f}")
lv = pd.Series({n:res[n] for n in harness}); hv = pd.Series(harness)
print("rank corr:", lv.rank().corr(hv.rank()), "| pearson:", lv.corr(hv))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()

def prep_matrix(M, tr_mask, nb=64):
    feat = [c for c in M.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
    num_cols = [c for c in feat if M[c].dtype.kind in "ifb"]
    cat_cols = [c for c in feat if M[c].dtype.kind not in "ifb" and M[c].nunique() <= 25]
    Xn = M[num_cols].astype(float).copy()
    med = Xn[tr_mask].median()
    Xn = Xn.fillna(med).fillna(0.0).values
    B = np.empty(Xn.shape, dtype=np.uint8)
    for j in range(Xn.shape[1]):
        qs = np.unique(np.quantile(Xn[tr_mask, j], np.linspace(0,1,nb+1)[1:-1]))
        B[:, j] = np.searchsorted(qs, Xn[:, j], side='right')
    mats = [B]
    for c in cat_cols:
        d = pd.get_dummies(M[c].astype(str), prefix=c, dummy_na=True)
        mats.append(d.values.astype(np.uint8))
    return np.hstack(mats)

def gbm_fit(B, y, snap, tr_snap_max, n_trees=120, lr=0.08, depth=4, nb=64,
            rowsample=0.8, colsample=0.7, min_leaf=20, reg=1.0, seed=0):
    rng = np.random.RandomState(seed)
    trm = snap <= tr_snap_max
    n, k = B.shape
    pred = np.zeros(n)
    tr_idx_all = np.where(trm)[0]
    pred[trm] = y[trm].mean()
    colsel = max(8, int(k*colsample))
    for t in range(n_trees):
        rows = rng.choice(tr_idx_all, size=int(len(tr_idx_all)*rowsample), replace=False)
        feats = rng.choice(k, size=colsel, replace=False)
        res = y - pred
        nodes = [(rows, 0)]
        for d in range(depth):
            new_nodes = []
            for idx, dc, f, b, isleft in nodes:
                if f is None and len(idx) >= 2*min_leaf and dc < depth:
                    G = res[idx].sum(); N = len(idx)
                    best = (0.0, None, None)
                    Bsub = B[idx][:, feats]
                    rsub = res[idx]
                    for fi in range(len(feats)):
                        h = np.bincount(Bsub[:, fi], weights=rsub, minlength=nb)
                        c = np.bincount(Bsub[:, fi], minlength=nb)
                        gl = np.cumsum(h); cl = np.cumsum(c)
                        gr = G - gl; cr = N - cl
                        gain = gl*gl/(cl+reg) + gr*gr/(cr+reg) - G*G/(N+reg)
                        valid = (cl >= min_leaf) & (cr >= min_leaf)
                        gain = np.where(valid, gain, -1e18)
                        bi = int(np.argmax(gain))
                        if gain[bi] > best[0]:
                            best = (gain[bi], fi, bi)
                    if best[1] is None:
                        new_nodes += [(idx, dc, None, None, None)]
                        continue
                    _, fi, bi = best
                    f = feats[fi]
                    mask = B[idx, f] <= bi
                    new_nodes += [(idx[mask], dc+1, f, bi, True),
                                  (idx[~mask], dc+1, f, bi, False)]
                else:
                    new_nodes.append((idx, dc, f, b, isleft))
            nodes = new_nodes
        for idx, dc, f, b, isleft in nodes:
            if f is None and len(idx):
                pred[idx] += lr * res[idx].mean()
    return pred

df = A.load_saved("e015_base.parquet")
M = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
trm = (M.snapshot_day <= 375).values
B = prep_matrix(M, trm)
print("B shape", B.shape)
y = np.log1p(M.future_spend_4w.values)
snap = M.snapshot_day.values
pred = gbm_fit(B, y, snap, 375, n_trees=40, lr=0.1, depth=4)
va = snap >= 403
p = np.expm1(pred[va])
print("local MAE (40 trees):", np.mean(np.abs(p - M.future_spend_4w.values[va])))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()

def prep_matrix(M, tr_mask, nb=64):
    feat = [c for c in M.columns if c not in ("household_key", "snapshot_day", "future_spend_4w")]
    num_cols = [c for c in feat if M[c].dtype.kind in "ifb"]
    cat_cols = [c for c in feat if M[c].dtype.kind not in "ifb" and M[c].nunique() <= 25]
    Xn = M[num_cols].astype(float).copy()
    med = Xn[tr_mask].median()
    Xn = Xn.fillna(med).fillna(0.0).values
    B = np.empty(Xn.shape, dtype=np.uint8)
    for j in range(Xn.shape[1]):
        qs = np.unique(np.quantile(Xn[tr_mask, j], np.linspace(0, 1, nb + 1)[1:-1]))
        B[:, j] = np.searchsorted(qs, Xn[:, j], side='right')
    mats = [B]
    for c in cat_cols:
        d = pd.get_dummies(M[c].astype(str), prefix=c, dummy_na=True)
        mats.append(d.values.astype(np.uint8))
    return np.hstack(mats)

def gbm_fit(B, y, snap, tr_snap_max, n_trees=120, lr=0.08, depth=4, nb=64,
            rowsample=0.8, colsample=0.7, min_leaf=20, reg=1.0, seed=0):
    rng = np.random.RandomState(seed)
    trm = snap <= tr_snap_max
    n, k = B.shape
    pred = np.zeros(n)
    tr_idx_all = np.where(trm)[0]
    pred[trm] = y[trm].mean()
    colsel = max(8, int(k * colsample))
    for t in range(n_trees):
        rows = rng.choice(tr_idx_all, size=int(len(tr_idx_all) * rowsample), replace=False)
        feats = rng.choice(k, size=colsel, replace=False)
        res = y - pred
        nodes = [(rows, 0, None, None, None)]
        for d in range(depth):
            new_nodes = []
            for nd in nodes:
                idx, dc, f, b, isleft = nd
                if f is None and len(idx) >= 2 * min_leaf and dc < depth:
                    G = res[idx].sum(); N = len(idx)
                    best = (0.0, None, None)
                    Bsub = B[idx][:, feats]
                    rsub = res[idx]
                    for fi in range(len(feats)):
                        h = np.bincount(Bsub[:, fi], weights=rsub, minlength=nb)
                        c = np.bincount(Bsub[:, fi], minlength=nb)
                        gl = np.cumsum(h); cl = np.cumsum(c)
                        gr = G - gl; cr = N - cl
                        gain = gl * gl / (cl + reg) + gr * gr / (cr + reg) - G * G / (N + reg)
                        valid = (cl >= min_leaf) & (cr >= min_leaf)
                        gain = np.where(valid, gain, -1e18)
                        bi = int(np.argmax(gain))
                        if gain[bi] > best[0]:
                            best = (gain[bi], fi, bi)
                    if best[1] is None:
                        new_nodes.append((idx, dc, None, None, None))
                        continue
                    _, fi, bi = best
                    f = feats[fi]
                    mask = B[idx, f] <= bi
                    new_nodes += [(idx[mask], dc + 1, f, bi, True),
                                  (idx[~mask], dc + 1, f, bi, False)]
                else:
                    new_nodes.append(nd)
            nodes = new_nodes
        for nd in nodes:
            idx, dc, f, b, isleft = nd
            if f is None and len(idx):
                pred[idx] += lr * res[idx].mean()
    return pred

df = A.load_saved("e015_base.parquet")
M = df.merge(tt, on=["household_key", "snapshot_day"], how="inner")
trm = (M.snapshot_day <= 375).values
B = prep_matrix(M, trm)
print("B shape", B.shape)
y = np.log1p(M.future_spend_4w.values)
snap = M.snapshot_day.values
pred = gbm_fit(B, y, snap, 375, n_trees=40, lr=0.1, depth=4)
va = snap >= 403
p = np.expm1(pred[va])
print("local MAE (40 trees):", np.mean(np.abs(p - M.future_spend_4w.values[va])))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()

def prep_matrix(M, tr_mask, nb=64):
    feat = [c for c in M.columns if c not in ("household_key", "snapshot_day", "future_spend_4w")]
    num_cols = [c for c in feat if M[c].dtype.kind in "ifb"]
    cat_cols = [c for c in feat if M[c].dtype.kind not in "ifb" and M[c].nunique() <= 25]
    Xn = M[num_cols].astype(float).copy()
    med = Xn[tr_mask].median()
    Xn = Xn.fillna(med).fillna(0.0).values
    B = np.empty(Xn.shape, dtype=np.uint8)
    for j in range(Xn.shape[1]):
        qs = np.unique(np.quantile(Xn[tr_mask, j], np.linspace(0, 1, nb + 1)[1:-1]))
        B[:, j] = np.searchsorted(qs, Xn[:, j], side='right')
    mats = [B]
    for c in cat_cols:
        d = pd.get_dummies(M[c].astype(str), prefix=c, dummy_na=True)
        mats.append(d.values.astype(np.uint8))
    return np.hstack(mats)

def gbm_fit(B, y, snap, tr_snap_max, n_trees=150, lr=0.08, depth=4, nb=64,
            rowsample=0.8, colsample=0.7, min_leaf=20, reg=1.0, seed=0):
    rng = np.random.RandomState(seed)
    trm = snap <= tr_snap_max
    n, k = B.shape
    pred = np.zeros(n)
    tr_idx_all = np.where(trm)[0]
    pred[trm] = y[trm].mean()
    colsel = max(8, int(k * colsample))
    for t in range(n_trees):
        rows = rng.choice(tr_idx_all, size=int(len(tr_idx_all) * rowsample), replace=False)
        feats = rng.choice(k, size=colsel, replace=False)
        res = y - pred
        # nodes: (train_idx, all_idx, dc, f, b)
        nodes = [(rows, np.arange(n), 0, None, None)]
        for d in range(depth):
            new_nodes = []
            for nd in nodes:
                tidx, aidx, dc, f, b = nd
                if f is None and len(tidx) >= 2 * min_leaf and dc < depth:
                    G = res[tidx].sum(); N = len(tidx)
                    best = (0.0, None, None)
                    Bsub = B[tidx][:, feats]
                    rsub = res[tidx]
                    for fi in range(len(feats)):
                        h = np.bincount(Bsub[:, fi], weights=rsub, minlength=nb)
                        c = np.bincount(Bsub[:, fi], minlength=nb)
                        gl = np.cumsum(h); cl = np.cumsum(c)
                        gr = G - gl; cr = N - cl
                        gain = gl*gl/(cl+reg) + gr*gr/(cr+reg) - G*G/(N+reg)
                        valid = (cl >= min_leaf) & (cr >= min_leaf)
                        gain = np.where(valid, gain, -1e18)
                        bi = int(np.argmax(gain))
                        if gain[bi] > best[0]:
                            best = (gain[bi], fi, bi)
                    if best[1] is None:
                        new_nodes.append(nd); continue
                    _, fi, bi = best
                    f = feats[fi]
                    m_t = B[tidx, f] <= bi
                    m_a = B[aidx, f] <= bi
                    new_nodes += [(tidx[m_t], aidx[m_a], dc+1, f, bi, True),
                                  (tidx[~m_t], aidx[~m_a], dc+1, f, bi, False)]
                else:
                    new_nodes.append(nd)
            nodes = new_nodes
        for nd in nodes:
            tidx, aidx, dc, f, b, isl = nd
            if len(tidx):
                val = lr * res[tidx].mean()
                pred[aidx] += val
    return pred

def local_eval(df, n_trees=150):
    M = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    trm = (M.snapshot_day <= 375).values
    B = prep_matrix(M, trm)
    y = np.log1p(M.future_spend_4w.values)
    snap = M.snapshot_day.values
    pred = gbm_fit(B, y, snap, 375, n_trees=n_trees)
    va = snap >= 403
    p = np.expm1(pred[va])
    return np.mean(np.abs(p - M.future_spend_4w.values[va]))

harness = {"e015_base":60.761,"e013_union":61.337,"e009_macro":61.647,"e011_display":61.680,
           "e012_full":61.471,"e008_decomp2":61.711,"e007_new":62.794,"e001_history":63.025,
           "e014_base":63.242,"e010_composite":63.939,"e004_long_hist":63.982,"e006_seq_gaps":64.050,
           "e003_dept_mix":64.175,"e002_marketing":64.676,"e005_seasonal_peer":67.852}
res = {}
for nm in harness:
    df = A.load_saved(nm + ".parquet")
    m = local_eval(df)
    res[nm] = m
    print(f"{nm:22s} local={m:7.3f} harness={harness[nm]:7.3f}")
lv = pd.Series({n:res[n] for n in harness}); hv = pd.Series(harness)
print("rank corr:", lv.rank().corr(hv.rank()), "| pearson:", lv.corr(hv))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()

def prep_matrix(M, tr_mask, nb=64):
    feat = [c for c in M.columns if c not in ("household_key", "snapshot_day", "future_spend_4w")]
    num_cols = [c for c in feat if M[c].dtype.kind in "ifb"]
    cat_cols = [c for c in feat if M[c].dtype.kind not in "ifb" and M[c].nunique() <= 25]
    Xn = M[num_cols].astype(float).copy()
    med = Xn[tr_mask].median()
    Xn = Xn.fillna(med).fillna(0.0).values
    B = np.empty(Xn.shape, dtype=np.uint8)
    for j in range(Xn.shape[1]):
        qs = np.unique(np.quantile(Xn[tr_mask, j], np.linspace(0, 1, nb + 1)[1:-1]))
        B[:, j] = np.searchsorted(qs, Xn[:, j], side='right')
    mats = [B]
    for c in cat_cols:
        d = pd.get_dummies(M[c].astype(str), prefix=c, dummy_na=True)
        mats.append(d.values.astype(np.uint8))
    return np.hstack(mats)

def gbm_fit(B, y, snap, tr_snap_max, n_trees=150, lr=0.08, depth=4, nb=64,
            rowsample=0.8, colsample=0.7, min_leaf=20, reg=1.0, seed=0):
    rng = np.random.RandomState(seed)
    trm = snap <= tr_snap_max
    n, k = B.shape
    pred = np.zeros(n)
    tr_idx_all = np.where(trm)[0]
    pred[trm] = y[trm].mean()
    colsel = max(8, int(k * colsample))
    for t in range(n_trees):
        rows = rng.choice(tr_idx_all, size=int(len(tr_idx_all) * rowsample), replace=False)
        feats = rng.choice(k, size=colsel, replace=False)
        res = y - pred
        nodes = [(rows, np.arange(n), 0, None, None, None)]
        for d in range(depth):
            new_nodes = []
            for nd in nodes:
                tidx, aidx, dc, f, b, isl = nd
                if f is None and len(tidx) >= 2 * min_leaf and dc < depth:
                    G = res[tidx].sum(); N = len(tidx)
                    best = (0.0, None, None)
                    Bsub = B[tidx][:, feats]
                    rsub = res[tidx]
                    for fi in range(len(feats)):
                        h = np.bincount(Bsub[:, fi], weights=rsub, minlength=nb)
                        c = np.bincount(Bsub[:, fi], minlength=nb)
                        gl = np.cumsum(h); cl = np.cumsum(c)
                        gr = G - gl; cr = N - cl
                        gain = gl*gl/(cl+reg) + gr*gr/(cr+reg) - G*G/(N+reg)
                        valid = (cl >= min_leaf) & (cr >= min_leaf)
                        gain = np.where(valid, gain, -1e18)
                        bi = int(np.argmax(gain))
                        if gain[bi] > best[0]:
                            best = (gain[bi], fi, bi)
                    if best[1] is None:
                        new_nodes.append(nd); continue
                    _, fi, bi = best
                    f = feats[fi]
                    m_t = B[tidx, f] <= bi
                    m_a = B[aidx, f] <= bi
                    new_nodes += [(tidx[m_t], aidx[m_a], dc+1, f, bi, True),
                                  (tidx[~m_t], aidx[~m_a], dc+1, f, bi, False)]
                else:
                    new_nodes.append(nd)
            nodes = new_nodes
        for nd in nodes:
            tidx, aidx, dc, f, b, isl = nd
            if len(tidx):
                pred[aidx] += lr * res[tidx].mean()
    return pred

def local_eval(df, n_trees=150):
    M = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    trm = (M.snapshot_day <= 375).values
    B = prep_matrix(M, trm)
    y = np.log1p(M.future_spend_4w.values)
    snap = M.snapshot_day.values
    pred = gbm_fit(B, y, snap, 375, n_trees=n_trees)
    va = snap >= 403
    p = np.expm1(pred[va])
    return np.mean(np.abs(p - M.future_spend_4w.values[va]))

harness = {"e015_base":60.761,"e013_union":61.337,"e009_macro":61.647,"e011_display":61.680,
           "e012_full":61.471,"e008_decomp2":61.711,"e007_new":62.794,"e001_history":63.025,
           "e014_base":63.242,"e010_composite":63.939,"e004_long_hist":63.982,"e006_seq_gaps":64.050,
           "e003_dept_mix":64.175,"e002_marketing":64.676,"e005_seasonal_peer":67.852}
res = {}
for nm in harness:
    df = A.load_saved(nm + ".parquet")
    m = local_eval(df)
    res[nm] = m
    print(f"{nm:22s} local={m:7.3f} harness={harness[nm]:7.3f}")
lv = pd.Series({n:res[n] for n in harness}); hv = pd.Series(harness)
print("rank corr:", lv.rank().corr(hv.rank()), "| pearson:", lv.corr(hv))