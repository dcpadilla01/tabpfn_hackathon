import pandas as pd, numpy as np, agent_api as A

class GBM:
    def __init__(self, rounds=300, lr=0.1, depth=4, lam=1.0, min_leaf=20, bins=32):
        self.p = dict(rounds=rounds, lr=lr, depth=depth, lam=lam, min_leaf=min_leaf, bins=bins)
    def fit(self, X, y):
        p = self.p; n, m = X.shape
        self.edges = []
        Xb = np.zeros((n, m), dtype=np.int8)
        for j in range(m):
            col = X[:, j]
            qs = np.quantile(col, np.linspace(0, 1, p['bins']+1))
            edges = np.unique(qs)
            if len(edges) < 2: edges = np.array([col.min()-1, col.max()+1])
            b = np.clip(np.searchsorted(edges[1:-1], col, side='right'), 0, len(edges)-2)
            Xb[:, j] = b; self.edges.append(edges)
        pred = np.full(n, y.mean()); self.trees = []
        for t in range(p['rounds']):
            g = pred - y
            nodes = [(np.arange(n), 0)]; tree = []
            while nodes:
                idx, d = nodes.pop(0)
                G = g[idx].sum(); H = len(idx)
                if d >= p['depth'] or H < 2*p['min_leaf']:
                    tree.append(('L', -G/(H+p['lam']))); continue
                best = (None, -1e18, None)
                for j in range(m):
                    b = Xb[idx, j]
                    hg = np.bincount(b, weights=g[idx], minlength=p['bins'])[:p['bins']]
                    hn = np.bincount(b, minlength=p['bins'])[:p['bins']].astype(float)
                    cg = np.cumsum(hg); cn = np.cumsum(hn)
                    gl = cg[:-1]; nl = cn[:-1]; gr = G-gl; nr = H-nl
                    valid = (nl >= p['min_leaf']) & (nr >= p['min_leaf'])
                    if not valid.any(): continue
                    score = np.where(valid, gl**2/(nl+p['lam']) + gr**2/(nr+p['lam']), -1e18)
                    k = int(np.argmax(score))
                    if score[k] > best[1]:
                        thr = self.edges[j][k+1]
                        best = (j, score[k], thr)
                j, sc, thr = best
                if j is None:
                    tree.append(('L', -G/(H+p['lam']))); continue
                mask = X[idx, j] <= thr
                tree.append(('S', j, thr))
                nodes.append((idx[mask], d+1)); nodes.append((idx[~mask], d+1))
            cur = np.zeros(n, dtype=int); upd = np.zeros(n)
            for i, node in enumerate(tree):
                if node[0] == 'S':
                    gl = X[:, node[1]] <= node[2]
                    msk = cur == i
                    cur[gl & msk] = 2*i+1; cur[(~gl) & msk] = 2*i+2
                else:
                    upd[cur == i] = node[1]
            pred = pred + p['lr'] * upd
            self.trees.append(tree)
        self.base = y.mean()
        return self
    def predict(self, X):
        out = np.full(len(X), self.base)
        for tree in self.trees:
            cur = np.zeros(len(X), dtype=int); leaf = np.zeros(len(X))
            for i, node in enumerate(tree):
                if node[0] == 'S':
                    gl = X[:, node[1]] <= node[2]
                    msk = cur == i
                    cur[gl & msk] = 2*i+1; cur[(~gl) & msk] = 2*i+2
                else:
                    leaf[cur == i] = node[1]
            out += self.p['lr'] * leaf
        return out

def loso_gbm(df, feats, rounds=200, depth=4, lr=0.1):
    tr_days = sorted(df.snapshot_day.unique())
    X = df[feats].astype(float).values
    y = df.future_spend_4w.values; days = df.snapshot_day.values
    errs = []
    for d in tr_days:
        m = days != d
        mdl = GBM(rounds=rounds, depth=depth, lr=lr).fit(X[m], y[m])
        errs.append(float(np.mean(np.abs(mdl.predict(X[~m]) - y[~m]))))
    return float(np.mean(errs))

def tonum(df, cols):
    for c in cols:
        if df[c].dtype.kind not in 'ifb':
            df[c] = pd.Categorical(df[c].fillna('__NA__')).codes.astype(float)
            df.loc[df[c] < 0, c] = np.nan
    return df

b = A.baseline_features(); tt = A.train_targets()
bfeats = [c for c in b.columns if c not in ('household_key','snapshot_day')]
db = b.merge(tt, on=['household_key','snapshot_day'])
db = tonum(db, bfeats)
print('E000 GBM LOSO:', round(loso_gbm(db, bfeats),2), '(harness 92.446)', flush=True)

t3 = A.load_saved('e003_catmix.parquet')
df3 = t3.merge(tt, on=['household_key','snapshot_day'])
f3 = [c for c in t3.columns if c not in ('household_key','snapshot_day')]
df3 = tonum(df3, f3)
print('E003 GBM LOSO:', round(loso_gbm(df3, f3),2), '(harness 63.050)')