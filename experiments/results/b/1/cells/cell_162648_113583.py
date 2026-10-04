import pandas as pd, numpy as np
from agent_api import load_saved, train_targets

t = train_targets()

def feat_matrix(df, feats, tr_mask):
    cols = []
    for f in feats:
        x = pd.to_numeric(df[f], errors="coerce").astype(float)
        med = np.nanmedian(x.values[tr_mask]) if tr_mask is not None else np.nanmedian(x.values)
        x = x.fillna(med if med==med else 0.0)
        cols.append(x.values)
    return np.column_stack(cols)

def gbm_eval(df, feats, train_days, val_days, n_trees=100, lr=0.1, depth=4, lam=1.0, min_child=20, nbins=64):
    d = df.merge(t, on=["household_key","snapshot_day"], how="inner")
    sd = d["snapshot_day"].values
    tr = np.isin(sd, train_days); va = np.isin(sd, val_days)
    X = feat_matrix(d, feats, tr)
    y = d["future_spend_4w"].values.astype(float)
    B = np.zeros(X.shape, dtype=np.int16)
    for j in range(X.shape[1]):
        qs = np.quantile(X[tr, j], np.linspace(0, 1, nbins+1)[1:-1])
        B[:, j] = np.searchsorted(qs, X[:, j], side="right")
    Btr, Bva = B[tr], B[va]; ytr = y[tr]
    n = Btr.shape[0]
    pred = np.full(n, ytr.mean()); pred_va = np.full(va.sum(), ytr.mean())
    idx_all = np.arange(n)
    for it in range(n_trees):
        g = pred - ytr
        nodes = [{"idx": idx_all}]
        for dep in range(depth):
            newnodes = []
            for node in nodes:
                if node.get("leaf") is not None or node["idx"].size < 2*min_child:
                    node["leaf"] = True; newnodes.append(node); continue
                idx = node["idx"]; best = (None, None, -1e18)
                G = g[idx].sum(); H = float(idx.size)
                base = G*G/(H+lam)
                for j in range(Btr.shape[1]):
                    b = Btr[idx, j]
                    cg = np.cumsum(np.bincount(b, weights=g[idx], minlength=nbins))
                    cc = np.cumsum(np.bincount(b, minlength=nbins).astype(float))
                    GL = cg[:-1]; HL = cc[:-1]
                    GR = G - GL; HR = H - HL
                    ok = (HL >= min_child) & (HR >= min_child)
                    if not ok.any(): continue
                    gain = GL*GL/(HL+lam) + GR*GR/(HR+lam) - base
                    gain[~ok] = -1e18
                    k = int(np.argmax(gain))
                    if gain[k] > best[2]: best = (j, k, gain[k])
                if best[0] is None or best[2] <= 0:
                    node["leaf"] = True; newnodes.append(node); continue
                j, k, _ = best
                b = Btr[idx, j]
                li = idx[b <= k]; ri = idx[b > k]
                node.update({"f": j, "k": k, "l": len(newnodes), "r": len(newnodes)+1})
                newnodes.append({"idx": li}); newnodes.append({"idx": ri})
            nodes = newnodes
        for node in nodes:
            idxn = node["idx"]
            node["val"] = -g[idxn].sum()/(idxn.size + lam) if idxn.size else 0.0
        pv = np.zeros(Bva.shape[0]); pt = np.zeros(n)
        for i in range(Bva.shape[0]):
            nd = 0
            while "f" in nodes[nd]:
                nd = nodes[nd]["l"] if Bva[i, nodes[nd]["f"]] <= nodes[nd]["k"] else nodes[nd]["r"]
            pv[i] = nodes[nd]["val"]
        for i in range(n):
            nd = 0
            while "f" in nodes[nd]:
                nd = nodes[nd]["l"] if Btr[i, nodes[nd]["f"]] <= nodes[nd]["k"] else nodes[nd]["r"]
            pt[i] = nodes[nd]["val"]
        pred_va += lr*pv; pred += lr*pt
    yva = y[va]
    return np.abs(pred_va - yva).mean()

harness = {"e009_macro":61.647,"e013_union":61.337,"e014_base":63.242,"e008_decomp2":61.711,"e012_full":61.471}
for n in ["e009_macro","e013_union","e014_base"]:
    df = load_saved(n+".parquet")
    feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
    mae = gbm_eval(df, feats, list(range(95,404,28)), [403,431])
    print(f"{n:14s} proxy={mae:8.3f} harness={harness[n]:7.3f}")
