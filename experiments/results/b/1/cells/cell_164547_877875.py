import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from agent_api import load_saved, train_targets

t = train_targets(); KEY = ["household_key","snapshot_day"]

def feat_matrix(df, feats, tr_mask):
    cols = []
    for f in feats:
        x = pd.to_numeric(df[f], errors="coerce").astype(float).values
        med = np.nanmedian(x[tr_mask]) if tr_mask is not None else np.nanmedian(x)
        x = np.where(np.isnan(x), med if med==med else 0.0, x)
        cols.append(x)
    return np.column_stack(cols)

def gbm_eval(df, feats, train_days, val_days, n_trees=60, lr=0.1, depth=3, lam=1.0, min_child=20, nbins=48, seed=0, maxnodes=120):
    d = df.merge(t, on=KEY, how="inner")
    sd = d["snapshot_day"].values
    tr = np.isin(sd, train_days); va = np.isin(sd, val_days)
    X = feat_matrix(d, feats, tr)
    y = d["future_spend_4w"].values.astype(float)
    B = np.zeros(X.shape, dtype=np.int16)
    for j in range(X.shape[1]):
        qs = np.nanquantile(X[tr, j], np.linspace(0, 1, nbins+1)[1:-1])
        B[:, j] = np.searchsorted(qs, X[:, j], side="right")
    Btr, Bva = B[tr], B[va]; ytr = y[tr]; n = Btr.shape[0]
    pred = np.full(n, ytr.mean()); pred_va = np.full(va.sum(), ytr.mean())
    rng = np.random.RandomState(seed)
    feat_sub = max(1, int(0.7*Btr.shape[1]))
    for it in range(n_trees):
        g = pred - ytr
        nodes = [{}]; queue = [(0, np.arange(n))]
        while queue:
            nid, idx = queue.pop(0)
            if idx.size < 2*min_child or len(nodes) > maxnodes: continue
            best = (None, None, -1e18)
            G = g[idx].sum(); H = float(idx.size); base_g = G*G/(H+lam)
            fjs = rng.choice(Btr.shape[1], feat_sub, replace=False) if feat_sub < Btr.shape[1] else range(Btr.shape[1])
            for j in fjs:
                b = Btr[idx, j]
                cg = np.cumsum(np.bincount(b, weights=g[idx], minlength=nbins))
                cc = np.cumsum(np.bincount(b, minlength=nbins).astype(float))
                GL, HL = cg[:-1], cc[:-1]; GR, HR = G-GL, H-HL
                ok = (HL >= min_child) & (HR >= min_child)
                if not ok.any(): continue
                gainj = GL*GL/(HL+lam) + GR*GR/(HR+lam) - base_g
                gainj[~ok] = -1e18
                k = int(np.argmax(gainj))
                if gainj[k] > best[2]: best = (int(j), k, gainj[k])
            if best[0] is None or best[2] <= 1e-9: continue
            j, k, _ = best
            b = Btr[idx, j]
            li, ri = idx[b <= k], idx[b > k]
            l, r = len(nodes), len(nodes)+1
            nodes.append({}); nodes.append({})
            nodes[nid] = {"f": j, "k": k, "l": l, "r": r}
            queue.append((l, li)); queue.append((r, ri))
        nn = len(nodes)
        route = np.zeros(n, dtype=np.int32)
        for i, nd in enumerate(nodes):
            if "f" in nd:
                m = route == i
                if not m.any(): continue
                left = m & (Btr[:, nd["f"]] <= nd["k"])
                route[left] = nd["l"]; route[m & ~left] = nd["r"]
        cnt = np.bincount(route, minlength=nn).astype(float)
        s = np.bincount(route, weights=-g, minlength=nn)
        val = s/(cnt+lam)
        pred += lr*val[route]
        route_va = np.zeros(va.sum(), dtype=np.int32)
        for i, nd in enumerate(nodes):
            if "f" in nd:
                m = route_va == i
                if not m.any(): continue
                left = m & (Bva[:, nd["f"]] <= nd["k"])
                route_va[left] = nd["l"]; route_va[m & ~left] = nd["r"]
        pred_va += lr*val[route_va]
    return np.abs(pred_va - y[va]).mean()

base = load_saved("e013_union.parquet")
bfeats = [c for c in base.columns if c not in KEY]
tr_days = list(range(95,404,28)); va_days=[403,431]
seq = load_saved("e006_seq_gaps.parquet")
A = ["ew28","ew56","weekly_rate_84","w_max7","w_min7","w_std7","w_cv7","spike","gap_max182","gap_med",
     "nb_all","n_gap21_182","n_gap14_84"] + [f"w{i}" for i in range(1,14)]
A = [c for c in A if c in seq.columns]
Bd = A + ["has_demographics","classification_1","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc"]
Bd = [c for c in Bd if c in seq.columns]
mfs_logs = ["e6_mfs","usual13_mfs","b75_mfs","log_e13","log_e6","log_b75","log_spend28","log_usual13"]
C = [f for f in bfeats if f not in mfs_logs]
print("base:", round(gbm_eval(base, bfeats, tr_days, va_days),3))
mA = gbm_eval(base.merge(seq[KEY+A], on=KEY, how="left"), bfeats+A, tr_days, va_days)
print(f"base+A ({len(A)}):", round(mA,3))
mB = gbm_eval(base.merge(seq[KEY+Bd], on=KEY, how="left"), bfeats+Bd, tr_days, va_days)
print(f"base+B ({len(Bd)}):", round(mB,3))
mC = gbm_eval(base, C, tr_days, va_days)
print(f"base-mfs-logs ({len(C)}):", round(mC,3))
mD = gbm_eval(base.merge(seq[KEY+A], on=KEY, how="left"), C+A, tr_days, va_days)
print(f"base-mfs-logs+A ({len(C)+len(A)}):", round(mD,3))
