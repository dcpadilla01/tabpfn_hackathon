import pandas as pd, numpy as np
from agent_api import load_saved, train_targets

for n in ["e013_union","cand_new","e009_macro","e008_decomp2","e012_full","e001_history"]:
    try:
        df = load_saved(n + ".parquet")
        print("==", n, df.shape)
        print(list(df.columns))
        print()
    except Exception as e:
        print("==", n, "ERR", repr(e))

t = train_targets()
print("target rows:", t.shape)
print(t.groupby("snapshot_day")["future_spend_4w"].agg(["count","mean","median"]).round(2))
print(t["future_spend_4w"].describe().round(2))

e13 = load_saved("e013_union.parquet")
m = e13.merge(t, on=["household_key","snapshot_day"], how="inner")
print("merged:", m.shape)
y = m["future_spend_4w"].astype(float).values
feats = [c for c in e13.columns if c not in ("household_key","snapshot_day")]
res = []
for f in feats:
    x = pd.to_numeric(m[f], errors="coerce")
    nu = m[f].nunique()
    if x.notna().sum() < 50 or x.fillna(0).std() == 0:
        res.append((f, np.nan, nu)); continue
    med = x.median()
    c = np.corrcoef(x.fillna(med).values, y)[0,1]
    res.append((f, c, nu))
res.sort(key=lambda r: -(r[1] if r[1]==r[1] else -99))
print("\n--- train corr with target (e013_union features) ---")
for f,c,nu in res:
    print(f"{f:42s} {c:+.4f} nu={nu}")


# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, baseline_features

t = train_targets()
print("target describe:")
print(t["future_spend_4w"].describe().round(2))
print(t.groupby("snapshot_day")["future_spend_4w"].agg(["count","mean","median"]).round(1))

for n in ["e001_history","e003_dept_mix","e004_long_hist","e005_seasonal_peer","e006_seq_gaps","e007_new","e010_composite","e011_display","e002_marketing"]:
    df = load_saved(n + ".parquet")
    print(n, df.shape, "ncol_feat=", df.shape[1]-2)

# top correlations for e013 (redo, print top 25)
e13 = load_saved("e013_union.parquet")
m = e13.merge(t, on=["household_key","snapshot_day"], how="inner")
y = m["future_spend_4w"].astype(float).values
feats = [c for c in e13.columns if c not in ("household_key","snapshot_day")]
res = []
for f in feats:
    x = pd.to_numeric(m[f], errors="coerce")
    if x.notna().sum() < 50: res.append((f, np.nan)); continue
    c = np.corrcoef(x.fillna(x.median()).values, y)[0,1]
    res.append((f, c))
res.sort(key=lambda r: -(r[1] if r[1]==r[1] else -99))
print("\nTOP 25 |corr|:")
for f,c in res[:25]: print(f"{f:24s} {c:+.4f}")
print("BOTTOM 10:")
for f,c in res[-10:]: print(f"{f:24s} {c:+.4f}")


# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets

t = train_targets()
def prep(df, feats):
    X = df[["household_key","snapshot_day"]].copy()
    for f in feats:
        x = pd.to_numeric(df[f], errors="coerce")
        if x.dtype == object or df[f].dtype == bool:
            x = x.astype(float) if df[f].dtype==bool else x
        X[f] = x.astype(float).fillna(0.0) if x.notna().any() else 0.0
    return X

def ridge_eval(df, feats, train_days, val_days, alpha=300.0):
    d = df.merge(t, on=["household_key","snapshot_day"], how="inner")
    X = prep(d, feats).values
    y = d["future_spend_4w"].values.astype(float)
    sd = d["snapshot_day"].values
    tr = np.isin(sd, train_days); va = np.isin(sd, val_days)
    mu = X[tr].mean(0); sg = X[tr].std(0)+1e-9
    Z = (X-mu)/sg
    A = Z[tr].T@Z[tr] + alpha*np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z[tr].T@y[tr])
    pred = Z[va]@w
    return np.abs(pred - y[va]).mean()

tables = ["e000_base","e001_history","e003_dept_mix","e004_long_hist","e005_seasonal_peer",
          "e006_seq_gaps","e007_new","e008_decomp2","e009_macro","e010_composite","e011_display",
          "e002_marketing","e012_full","e013_union","e014_base","cand_new"]
harness = {"e000_base":92.531,"e001_history":63.025,"e003_dept_mix":64.175,"e004_long_hist":63.982,
           "e005_seasonal_peer":67.852,"e006_seq_gaps":64.050,"e007_new":62.794,"e008_decomp2":61.711,
           "e009_macro":61.647,"e010_composite":63.939,"e011_display":61.680,"e002_marketing":64.676,
           "e012_full":61.471,"e013_union":61.337,"e014_base":63.242,"cand_new":None}

print(f"{'table':22s} {'proxy431':>9s} {'harness':>9s}")
out=[]
for n in tables:
    try:
        df = load_saved(n+".parquet")
        feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
        mae = ridge_eval(df, feats, list(range(95,404,28)), [431])
        out.append((n, mae, harness.get(n)))
        print(f"{n:22s} {mae:9.3f} {str(harness.get(n)):>9s}")
    except Exception as e:
        print(n, "ERR", repr(e))
# rank correlation
vals = [(h, m) for n,m,h in out if h is not None]
import numpy as np
h_ = np.array([v[1] for v in vals]); p_ = np.array([v[0] for v in vals])
print("spearman proxy vs harness:", np.corrcoef(h_.argsort().argsort(), p_.argsort().argsort())[0,1].round(3))


# ---- cell ----
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


# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets

t = train_targets()
df = load_saved("e013_union.parquet")
d = df.merge(t, on=["household_key","snapshot_day"], how="inner")
print("merged shape:", d.shape)
feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
sd = d["snapshot_day"].values
tr = np.isin(sd, list(range(95,404,28))); va = np.isin(sd, [403,431])
print("tr/va counts:", tr.sum(), va.sum())
X = np.column_stack([pd.to_numeric(d[f], errors="coerce").astype(float).fillna(0).values for f in feats])
print("X shape:", X.shape, "std mean:", X[tr].std(0).mean())
y = d["future_spend_4w"].values.astype(float)
print("y mean train:", y[tr].mean(), "MAE of constant mean on va:", np.abs(y[tr].mean()-y[va]).mean())
# check a simple ridge on 5 features
Z = (X - X[tr].mean(0))/(X[tr].std(0)+1e-9)
A = Z[tr].T@Z[tr] + 100*np.eye(Z.shape[1])
w = np.linalg.solve(A, Z[tr].T@y[tr])
pred = Z[va]@w
print("ridge5-all-feats MAE on 403+431:", np.abs(pred-y[va]).mean())
print("pred std:", pred.std())


# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets

t = train_targets()

def feat_matrix(df, feats, tr_mask):
    cols = []
    for f in feats:
        x = pd.to_numeric(df[f], errors="coerce").astype(float).values
        med = np.nanmedian(x[tr_mask]) if tr_mask is not None else np.nanmedian(x)
        x = np.where(np.isnan(x), med if med==med else 0.0, x)
        cols.append(x)
    return np.column_stack(cols)

def gbm_eval(df, feats, train_days, val_days, n_trees=100, lr=0.08, depth=4, lam=1.0, min_child=20, nbins=64, seed=0):
    d = df.merge(t, on=["household_key","snapshot_day"], how="inner")
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
        nodes = [{}]  # id0 root; dict: {f,k,l,r} internal or {} leaf
        queue = [(0, np.arange(n))]
        while queue:
            nid, idx = queue.pop(0)
            if idx.size < 2*min_child or len(nodes) > 200:
                continue
            best = (None, None, -1e18)
            G = g[idx].sum(); H = float(idx.size); base = G*G/(H+lam)
            fjs = rng.choice(Btr.shape[1], feat_sub, replace=False) if feat_sub < Btr.shape[1] else range(Btr.shape[1])
            for j in fjs:
                b = Btr[idx, j]
                cg = np.cumsum(np.bincount(b, weights=g[idx], minlength=nbins))
                cc = np.cumsum(np.bincount(b, minlength=nbins).astype(float))
                GL, HL = cg[:-1], cc[:-1]
                GR, HR = G-GL, H-HL
                ok = (HL >= min_child) & (HR >= min_child)
                if not ok.any(): continue
                gain = GL*GL/(HL+lam) + GR*GR/(HR+lam) - base
                gain[~ok] = -1e18
                k = int(np.argmax(gain))
                if gain[k] > best[2]: best = (int(j), k, gain[k])
            if best[0] is None or best[2] <= 1e-9:
                continue
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
        val = np.where(cnt > 0, s/np.maximum(cnt, 1) + 0.0, 0.0)  # -sum(g)/(cnt+lam) approx: use cnt+lam
        val = s/(cnt+lam)
        pred += lr*val[np.maximum(route,0)]
        route_va = np.zeros(va.sum(), dtype=np.int32)
        for i, nd in enumerate(nodes):
            if "f" in nd:
                m = route_va == i
                if not m.any(): continue
                left = m & (Bva[:, nd["f"]] <= nd["k"])
                route_va[left] = nd["l"]; route_va[m & ~left] = nd["r"]
        pred_va += lr*val[route_va]
    yva = y[va]
    return np.abs(pred_va - yva).mean()

harness = {"e009_macro":61.647,"e013_union":61.337,"e014_base":63.242,"e008_decomp2":61.711,"e012_full":61.471,"e000":92.531,"e001_history":63.025}
for n in ["e013_union","e009_macro","e014_base","e001_history"]:
    df = load_saved(n+".parquet")
    feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
    mae = gbm_eval(df, feats, list(range(95,404,28)), [403,431])
    print(f"{n:14s} proxy={mae:8.3f} harness={harness.get(n,-1):7.3f}")


# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, snapshot

for n in ["e003_dept_mix","e004_long_hist","e011_display","e002_marketing","e005_seasonal_peer","e006_seq_gaps"]:
    df = load_saved(n+".parquet")
    print(n, [c for c in df.columns if c not in ("household_key","snapshot_day")][:20], "...")
print()
v = snapshot()
dem = v.demographics
print(dem.shape)
print(dem.head(3))
for c in dem.columns:
    if c != "household_key":
        print(c, dem[c].value_counts().to_dict())


# ---- cell ----
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


# ---- cell ----
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
tr_days = list(range(95,404,28)); va_days=[403,431]
print("BASE e013_union:", round(gbm_eval(base, bfeats, tr_days, va_days),3))
blocks = {}
for n in ["e004_long_hist","e011_display","e002_marketing","e005_seasonal_peer","e006_seq_gaps","e010_composite","e001_history"]:
    df = load_saved(n+".parquet")
    blocks[n] = [c for c in df.columns if c not in KEY and c not in bfeats]
for n, u in blocks.items():
    if not u: continue
    df = load_saved(n+".parquet")
    dfu = df[KEY+u]
    merged = base.merge(dfu, on=KEY, how="left")
    mae = gbm_eval(merged, bfeats+u, tr_days, va_days)
    print(f"+{n:18s} ({len(u):3d} feats): {mae:8.3f}")
# pruning variants
drops = {
 "drop_mfs": ["e6_mfs","usual13_mfs","b75_mfs"],
 "drop_logs": ["log_e13","log_e6","log_b75","log_spend28","log_usual13"],
 "drop_marketing": ["n_active","act_TypeA","act_TypeB","act_TypeC","recent_start_28","hh_targ_active","redemp_28","redemp_84"],
 "drop_mfs_logs": ["e6_mfs","usual13_mfs","b75_mfs","log_e13","log_e6","log_b75","log_spend28","log_usual13"],
}
for name, dl in drops.items():
    keep = [f for f in bfeats if f not in dl]
    mae = gbm_eval(base, keep, tr_days, va_days)
    print(f"e013 {name:18s} (-{len(dl)}): {mae:8.3f}")


# ---- cell ----
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
        qs = np.quantile(X[tr, j], np.linspace(0, 1, nbins+1)[1:-1])
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
tr_days = list(range(95,404,28)); va_days=[403,431]
print("BASE e013_union:", round(gbm_eval(base, bfeats, tr_days, va_days),3))
blocks = {}
for n in ["e004_long_hist","e011_display","e002_marketing","e005_seasonal_peer","e006_seq_gaps","e010_composite"]:
    df = load_saved(n+".parquet")
    blocks[n] = [c for c in df.columns if c not in KEY and c not in bfeats]
for n, u in blocks.items():
    if not u: continue
    df = load_saved(n+".parquet")
    merged = base.merge(df[KEY+u], on=KEY, how="left")
    mae = gbm_eval(merged, bfeats+u, tr_days, va_days)
    print(f"+{n:18s} ({len(u):3d}): {mae:8.3f}")


# ---- cell ----
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from agent_api import load_saved, train_targets

t = train_targets(); KEY = ["household_key","snapshot_day"]

def prep(df, feats, tr):
    cols = []
    for f in feats:
        x = pd.to_numeric(df[f], errors="coerce").astype(float).values
        med = np.nanmedian(x[tr]) if tr is not None else np.nanmedian(x)
        x = np.where(np.isnan(x), med if med==med else 0.0, x)
        cols.append(x)
    return np.column_stack(cols)

def ridge_eval(df, feats, train_days, val_days, alpha):
    d = df.merge(t, on=KEY, how="inner")
    sd = d["snapshot_day"].values
    tr = np.isin(sd, train_days); va = np.isin(sd, val_days)
    X = prep(d, feats, tr); y = d["future_spend_4w"].values.astype(float)
    mu = X[tr].mean(0); sg = X[tr].std(0)+1e-9
    Z = (X-mu)/sg
    A = Z[tr].T@Z[tr] + alpha*np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z[tr].T@y[tr])
    return np.abs(Z[va]@w - y[va]).mean()

harness = {"e009_macro":61.647,"e013_union":61.337,"e014_base":63.242,"e008_decomp2":61.711,
           "e012_full":61.471,"e001_history":63.025,"e011_display":61.680,"e010_composite":63.939}
tabs = ["e013_union","e014_base","e009_macro","e012_full","e001_history","e008_decomp2","e011_display","e010_composite"]
for alpha in [30, 300, 3000, 10000]:
    line = []
    for n in tabs:
        df = load_saved(n+".parquet")
        feats = [c for c in df.columns if c not in KEY]
        mae = ridge_eval(df, feats, list(range(95,404,28)), [403,431], alpha)
        line.append(f"{n.replace('_macro','').replace('_union','').replace('_base','').replace('_full','').replace('_history','').replace('_decomp2','').replace('_display','').replace('_composite','')}:{mae:.1f}")
    print(f"alpha={alpha:6d}  " + "  ".join(line))
print("harness:        e013:61.3  e014:63.2  e009:61.6  e012:61.5  e001:63.0  e008:61.7  e011:61.7  e010:63.9")


# ---- cell ----
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


# ---- cell ----
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


# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, save_table
KEY = ["household_key","snapshot_day"]
base = load_saved("e013_union.parquet")
seq = load_saved("e006_seq_gaps.parquet")
A = ["ew28","ew56","weekly_rate_84","w_max7","w_min7","w_std7","w_cv7","spike","gap_max182","gap_med",
     "nb_all","n_gap21_182","n_gap14_84"] + [f"w{i}" for i in range(1,14)]
dem = ["has_demographics","classification_1","classification_3","classification_4","classification_5",
       "homeowner_desc","kid_category_desc"]
want = A + dem
Bd = [c for c in want if c in seq.columns]
print("missing:", [c for c in want if c not in seq.columns])
tab = base.merge(seq[KEY+Bd], on=KEY, how="left")
print("shape:", tab.shape, "| new feats:", len(Bd))
path = save_table(tab, "e015_base")
print("saved:", path)
