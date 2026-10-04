import pandas as pd, numpy as np
from agent_api import load_saved, train_targets

t = train_targets()
KEY = ["household_key","snapshot_day"]

def feat_matrix(df, feats, tr_mask):
    cols = []
    for f in feats:
        x = pd.to_numeric(df[f], errors="coerce").astype(float).values
        med = np.nanmedian(x[tr_mask]) if tr_mask is not None else np.nanmedian(x)
        x = np.where(np.isnan(x), med if med==med else 0.0, x)
        cols.append(x)
    return np.column_stack(cols)

def gbm_eval(df, feats, train_days, val_days, n_trees=100, lr=0.08, depth=4, lam=1.0, min_child=20, nbins=64, seed=0):
    d = df.merge(t, on=KEY, how="inner")
    sd = d["snapshot_day"].values
    tr = np.isin(sd, train_days); va = np.isin(sd, val_days)
    X = feat_matrix(d, feats, tr)
    y = d["future_spend_4w"].values.astype(float)
    B = np.zeros(X.shape, dtype=np.int16)
    for j in range(X.shape[1]):
        qs = np.quantile(X[tr, j], np.linspace(0, 1, nbins+1)[1:-1])
        B[:, j] = np.searchsorted(qs, X[:, j], side="right")
    Btr, Bva = B[tr], B[va]; ytr = y[tr]; n = Btr.shape[0]
    pred = np.full(n, ytr.mean()); pred_va = np.full(va.sum(), ytr.mean())
    rng = np.random.RandomState(seed)
    feat_sub = max(1, int(0.8*Btr.shape[1]))
    for it in range(n_trees):
        g = pred - ytr
        nodes = [{}]; queue = [(0, np.arange(n))]
        while queue:
            nid, idx = queue.pop(0)
            if idx.size < 2*min_child or len(nodes) > 250: continue
            best = (None, None, -1e18)
            G = g[idx].sum(); H = float(idx.size); base = G*G/(H+lam)
            fjs = rng.choice(Btr.shape[1], feat_sub, replace=False) if feat_sub < Btr.shape[1] else range(Btr.shape[1])
            for j in fjs:
                b = Btr[idx, j]
                cg = np.cumsum(np.bincount(b, weights=g[idx], minlength=nbins))
                cc = np.cumsum(np.bincount(b, minlength=nbins).astype(float))
                GL, HL = cg[:-1], cc[:-1]; GR, HR = G-GL, H-HL
                ok = (HL >= min_child) & (HR >= min_child)
                if not ok.any(): continue
                gain = GL*GL/(HL+lam) + GR*GR/(HR+lam) - base
                gain[~ok] = -1e18
                k = int(np.argmax(gain))
                if gain[k] > best[2]: best = (int(j), k, gain[k])
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
blocks = {}
for n in ["e004_long_hist","e011_display","e002_marketing","e005_seasonal_peer","e006_seq_gaps","e010_composite","e003_dept_mix","e001_history"]:
    df = load_saved(n+".parquet")
    u = [c for c in df.columns if c not in KEY and c not in bfeats]
    blocks[n] = u
    print(n, "unique feats:", len(u), u[:14])
print()
# seed variance of base
for seed in [0,1,2]:
    print("e013_union seed", seed, round(gbm_eval(base, bfeats, list(range(95,404,28)), [403,431], seed=seed),3))
