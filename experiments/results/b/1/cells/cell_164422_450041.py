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

def gbm_fit_predict(df, feats, train_days, val_days, n_trees=60, lr=0.1, depth=3, lam=1.0, min_child=20, nbins=48, seed=0, maxnodes=120, collect_gain=False):
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
    gain = np.zeros(Btr.shape[1])
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
            j, k, gv = best
            if collect_gain: gain[j] += gv
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
    mae = np.abs(pred_va - y[va]).mean()
    return (mae, gain) if collect_gain else mae

base = load_saved("e013_union.parquet")
bfeats = [c for c in base.columns if c not in KEY]
tr_days = list(range(95,404,28)); va_days=[403,431]
cand = {}
for n in ["e006_seq_gaps","e004_long_hist","e002_marketing","e010_composite","e011_display","e005_seasonal_peer"]:
    df = load_saved(n+".parquet")
    for c in df.columns:
        if c not in KEY and c not in bfeats and c not in cand:
            cand[c] = n
print("candidates:", len(cand))
big = base.merge(pd.concat([load_saved(n+".parquet")[KEY+[c for c in df.columns if c not in KEY and c not in bfeats]] for n,df in [(n, load_saved(n+".parquet")) for n in ["e006_seq_gaps","e004_long_hist","e002_marketing","e010_composite","e011_display","e005_seasonal_peer"]]]).drop_duplicates(KEY), on=KEY, how="left")
allf = bfeats + list(cand.keys())
mae, gain = gbm_fit_predict(big, allf, tr_days, va_days, collect_gain=True)
print("big-union proxy MAE:", round(mae,3))
order = np.argsort(-gain)
print("\ntop 45 candidates by gain (only non-base feats):")
shown = 0
for i in order:
    f = allf[i]
    if f in cand and shown < 45:
        print(f"{f:26s} {cand[f]:18s} gain={gain[i]:10.1f}")
        shown += 1
