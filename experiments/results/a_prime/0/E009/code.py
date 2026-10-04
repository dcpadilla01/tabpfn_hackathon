import agent_api, pandas as pd, numpy as np
mkt = agent_api.load_saved('mkt_v2.parquet')
print("E003 table shape:", mkt.shape)
print("columns:", list(mkt.columns))
tt = agent_api.train_targets()
print("targets shape:", tt.shape)
df = mkt.merge(tt, on=['household_key','snapshot_day'], how='inner')
print("merged:", df.shape)
y = df['future_spend_4w']
print(y.describe())
print("zero share:", (y==0).mean())
num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
cor = df[num].corrwith(y, method='spearman').sort_values()
print(cor.to_string())


# ---- cell ----
import agent_api, pandas as pd, numpy as np

tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]

def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0)+1e-9
    Xs = (X-mu)/sd
    A = Xs.T@Xs + alpha*np.eye(Xs.shape[1])
    w = np.linalg.solve(A, Xs.T@y)
    return w, mu, sd

def ridge_pred(X, w, mu, sd):
    return np.clip((X-mu)/sd @ w, 0, None)

def eval_split(fit_days, val_days, transform=None, alpha=100.0):
    tr = df[df.snapshot_day.isin(fit_days)]
    va = df[df.snapshot_day.isin(val_days)]
    Xtr, ytr = tr[feats].fillna(0).values, tr.future_spend_4w.values
    Xva, yva = va[feats].fillna(0).values, va.future_spend_4w.values
    if transform == 'log':
        w,mu,sd = ridge_fit(Xtr, np.log1p(ytr), alpha)
        pv = np.expm1(np.clip((Xva-mu)/sd @ w, 0, 20))
    else:
        w,mu,sd = ridge_fit(Xtr, ytr, alpha)
        pv = ridge_pred(Xva, w, mu, sd)
    return np.abs(pv-yva).mean()

print("== internal ridge CV on mkt_v2 (72 feats) ==")
for alpha in [10,100,1000]:
    print(f"alpha={alpha}: val431={eval_split([d for d in all_train_days if d!=431],[431],alpha=alpha):.3f}, val403={eval_split([d for d in all_train_days if d!=403],[403],alpha=alpha):.3f}")
print("log-target: val431=", round(eval_split([d for d in all_train_days if d!=431],[431],'log',10),3),
      " val403=", round(eval_split([d for d in all_train_days if d!=403],[403],'log',10),3))

# naive baselines
for col in ['spend_84','spend_28','x_exp4w']:
    for vd in [431,403]:
        va = df[df.snapshot_day==vd]
        print(f"naive {col} -> {vd}: {np.abs(va[col].fillna(0).values - va.future_spend_4w.values).mean():.3f}")

# recency-bucket conditional median lookup (uses targets: analysis only)
def recency_lookup(fit_days, val_days):
    tr = df[df.snapshot_day.isin(fit_days)].copy()
    va = df[df.snapshot_day==val_days].copy()
    bins = [-1,7,14,21,28,42,56,84,112,10000]
    tr['rb'] = pd.cut(tr.recency, bins)
    lut = tr.groupby('rb', observed=True).future_spend_4w.median()
    va['rb'] = pd.cut(va.recency, bins)
    pv = va['rb'].map(lut).fillna(0).values
    return np.abs(pv - va.future_spend_4w.values).mean(), lut
for vd in [431,403]:
    fd = [d for d in all_train_days if d!=vd]
    m,lut = recency_lookup(fd, vd)
    print(f"recency-median lookup val{vd}: {m:.3f}")
print(lut)


# ---- cell ----
import agent_api, pandas as pd, numpy as np

tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]

# demographics
demo = agent_api.snapshot().demographics
print("demo rows:", len(demo), "cols:", list(demo.columns))
print(demo.head(3).to_string())

def ridge_cv(feat_list, alpha=1000, extra=None):
    D = df.copy()
    if extra is not None:
        D = D.merge(extra, on='household_key', how='left')
        fl = feat_list + [c for c in extra.columns if c!='household_key']
    else:
        fl = feat_list
    outs = []
    for vd in [431,403]:
        tr = D[D.snapshot_day.isin([d for d in all_train_days if d!=vd])]
        va = D[D.snapshot_day==vd]
        Xtr = pd.get_dummies(tr[fl].fillna('NA').astype(object), dummy_na=False)
        Xva = pd.get_dummies(va[fl].fillna('NA').astype(object)).reindex(columns=Xtr.columns, fill_value=0)
        mu, sd = Xtr.mean(0), Xtr.std(0)+1e-9
        w = np.linalg.solve((Xtr.values.T@Xtr.values)/len(tr) + alpha*np.eye(Xtr.shape[1]), Xtr.values.T@(tr.future_spend_4w.values - tr.future_spend_4w.mean())/len(tr))
        pv = np.clip(Xva.values@w + tr.future_spend_4w.mean(), 0, None)
        outs.append(np.abs(pv-va.future_spend_4w.values).mean())
    return outs

base = ridge_cv(feats)
print("internal ridge val431/403 (feats only):", [round(x,2) for x in base])
# add demographics one-hot
dd = demo.copy()
for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']:
    dd[c] = dd[c].astype(str)
with_demo = ridge_cv(feats, extra=dd[['household_key']+ [c for c in dd.columns if c!='household_key']])
print("internal ridge + demographics:", [round(x,2) for x in with_demo])


# ---- cell ----
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]

def ridge_cv(feat_list, extra=None, alpha=1000):
    D = df.copy()
    if extra is not None:
        D = D.merge(extra, on='household_key', how='left')
        fl = feat_list + [c for c in extra.columns if c!='household_key']
    else:
        fl = list(feat_list)
    outs = []
    for vd in [431,403]:
        tr = D[D.snapshot_day.isin([d for d in all_train_days if d!=vd])]
        va = D[D.snapshot_day==vd]
        Xtr = pd.get_dummies(tr[fl].astype(object), dummy_na=False).astype(float)
        Xva = pd.get_dummies(va[fl].astype(object)).reindex(columns=Xtr.columns, fill_value=0).astype(float)
        ym = tr.future_spend_4w.mean()
        A = (Xtr.T@Xtr).values/len(tr) + alpha*np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, (Xtr.T@(tr.future_spend_4w.values-ym)).values/len(tr))
        pv = np.clip(Xva.values@w + ym, 0, None)
        outs.append(np.abs(pv-va.future_spend_4w.values).mean())
    return outs

base = ridge_cv(feats)
print("feats-only:", [round(x,2) for x in base], flush=True)

demo = agent_api.snapshot().demographics
dd = demo.copy()
cat_cols = [c for c in dd.columns if c!='household_key']
with_demo = ridge_cv(feats, extra=dd[['household_key']+cat_cols])
print("+demographics:", [round(x,2) for x in with_demo], flush=True)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
print("start", df.shape, len(feats), flush=True)

def ridge_one(vd, feat_list, alpha=1000):
    tr = df[df.snapshot_day.isin([d for d in all_train_days if d!=vd])]
    va = df[df.snapshot_day==vd]
    Xtr = pd.get_dummies(tr[feat_list].astype(object), dummy_na=False).astype(float)
    Xva = pd.get_dummies(va[feat_list].astype(object)).reindex(columns=Xtr.columns, fill_value=0).astype(float)
    ym = tr.future_spend_4w.mean()
    A = (Xtr.T@Xtr).values/len(tr) + alpha*np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, (Xtr.T@(tr.future_spend_4w.values-ym)).values/len(tr))
    pv = np.clip(Xva.values@w + ym, 0, None)
    return np.abs(pv-va.future_spend_4w.values).mean()

print("base:", [round(ridge_one(vd, feats),2) for vd in [431,403]], flush=True)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
tr = df[df.snapshot_day.isin([d for d in all_train_days if d!=431])]
va = df[df.snapshot_day==431]
print("shapes", tr.shape, va.shape, flush=True)
Xtr = pd.get_dummies(tr[feats].astype(object), dummy_na=False).astype(float)
print("dummies done", Xtr.shape, flush=True)
Xva = pd.get_dummies(va[feats].astype(object)).reindex(columns=Xtr.columns, fill_value=0).astype(float)
print("va dummies done", flush=True)
ym = tr.future_spend_4w.mean()
A = (Xtr.T@Xtr).values/len(tr) + 1000*np.eye(Xtr.shape[1])
print("A done", flush=True)
w = np.linalg.solve(A, (Xtr.T@(tr.future_spend_4w.values-ym)).values/len(tr))
print("w done", flush=True)
pv = np.clip(Xva.values@w + ym, 0, None)
print("MAE431:", np.abs(pv-va.future_spend_4w.values).mean(), flush=True)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
str_cols = [c for c in feats if mkt[c].dtype==object]
num_cols = [c for c in feats if c not in str_cols]
print("num:",len(num_cols),"str:",str_cols, flush=True)

def design(D, fl):
    s = [c for c in fl if c in str_cols]
    n = [c for c in fl if c not in str_cols]
    Xn = D[n].fillna(0).astype(float).values if n else np.zeros((len(D),0))
    Xs = pd.get_dummies(D[s].astype(str), dummy_na=False).astype(float).values if s else np.zeros((len(D),0))
    return np.hstack([Xn, Xs])

def ridge_one(vd, feat_list, extra=None, alpha=1000):
    D = df
    fl = list(feat_list)
    if extra is not None:
        D = D.merge(extra, on='household_key', how='left')
        fl += [c for c in extra.columns if c!='household_key']
    tr = D[D.snapshot_day.isin([d for d in all_train_days if d!=vd])]
    va = D[D.snapshot_day==vd]
    Xtr, Xva = design(tr, fl), design(va, fl)
    ym = tr.future_spend_4w.mean()
    A = (Xtr.T@Xtr)/len(tr) + alpha*np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, (Xtr.T@(tr.future_spend_4w.values-ym))/len(tr))
    pv = np.clip(Xva@w + ym, 0, None)
    return np.abs(pv-va.future_spend_4w.values).mean()

print("base:", [round(ridge_one(vd, feats),2) for vd in [431,403]], flush=True)
demo = agent_api.snapshot().demographics
dd = demo.copy()
cats = [c for c in dd.columns if c!='household_key']
print("+demo:", [round(ridge_one(vd, feats, extra=dd[['household_key']+cats]),2) for vd in [431,403]], flush=True)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]

def design(D, s_cols, n_cols):
    Xn = D[n_cols].fillna(0).astype(float).values if n_cols else np.zeros((len(D),0))
    Xs = pd.get_dummies(D[s_cols].astype(str), dummy_na=False).astype(float).values if s_cols else np.zeros((len(D),0))
    return np.hstack([Xn, Xs])

def ridge_one(vd, feat_list, extra=None, alpha=1000):
    D = df
    fl = list(feat_list)
    if extra is not None:
        D = D.merge(extra, on='household_key', how='left')
        fl += [c for c in extra.columns if c!='household_key']
    tr = D[D.snapshot_day.isin([d for d in all_train_days if d!=vd])]
    va = D[D.snapshot_day==vd]
    s = [c for c in fl if D[c].dtype==object or str(D[c].dtype)=='category']
    n = [c for c in fl if c not in s]
    Xtr, Xva = design(tr, s, n), design(va, s, n)
    ym = tr.future_spend_4w.mean()
    A = (Xtr.T@Xtr)/len(tr) + alpha*np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, (Xtr.T@(tr.future_spend_4w.values-ym))/len(tr))
    pv = np.clip(Xva@w + ym, 0, None)
    return np.abs(pv-va.future_spend_4w.values).mean()

demo = agent_api.snapshot().demographics
dd = demo.copy()
cats = [c for c in dd.columns if c!='household_key']
print("base:", [round(ridge_one(vd, feats),2) for vd in [431,403]], flush=True)
print("+demo:", [round(ridge_one(vd, feats, extra=dd[['household_key']+cats]),2) for vd in [431,403]], flush=True)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
demo = agent_api.snapshot().demographics
print("mkt hh dtype:", df.household_key.dtype, "demo hh dtype:", demo.household_key.dtype, flush=True)
print("demo sample keys:", demo.household_key.head(3).tolist(), " df keys:", df.household_key.head(3).tolist(), flush=True)
m = df.merge(demo, on='household_key', how='left')
print("merged rows:", len(m), "non-null class1:", m.classification_1.notna().sum(), flush=True)
# how many df households in demo
overlap = df.household_key.isin(demo.household_key).mean()
print("frac of rows with demo:", overlap, flush=True)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
demo = agent_api.snapshot().demographics
print(demo.dtypes, flush=True)
dd = demo.copy()
cats = [c for c in dd.columns if c!='household_key']
D = df.merge(dd[['household_key']+cats], on='household_key', how='left')
print(D[cats[0]].dtype, D[cats[0]].unique()[:5], flush=True)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
demo = agent_api.snapshot().demographics

def design(D, s_cols, n_cols):
    Xn = D[n_cols].fillna(0).astype(float).values if n_cols else np.zeros((len(D),0))
    Xs = pd.get_dummies(D[s_cols].astype(str), dummy_na=False).astype(float).values if s_cols else np.zeros((len(D),0))
    return np.hstack([Xn, Xs])

def ridge_one(vd, feat_list, extra=None, alpha=1000):
    D = df
    fl = list(feat_list)
    if extra is not None:
        D = D.merge(extra, on='household_key', how='left')
        fl += [c for c in extra.columns if c!='household_key']
    tr = D[D.snapshot_day.isin([d for d in all_train_days if d!=vd])]
    va = D[D.snapshot_day==vd]
    s = [c for c in fl if str(D[c].dtype)=='category' or D[c].dtype==object]
    n = [c for c in fl if c not in s]
    Xtr, Xva = design(tr, s, n), design(va, s, n)
    ym = tr.future_spend_4w.mean()
    A = (Xtr.T@Xtr)/len(tr) + alpha*np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, (Xtr.T@(tr.future_spend_4w.values-ym))/len(tr))
    pv = np.clip(Xva@w + ym, 0, None)
    return np.abs(pv-va.future_spend_4w.values).mean()

cats = [c for c in demo.columns if c!='household_key']
print("base:", [round(ridge_one(vd, feats),2) for vd in [431,403]], flush=True)
print("+demo:", [round(ridge_one(vd, feats, extra=demo[['household_key']+cats]),2) for vd in [431,403]], flush=True)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
demo = agent_api.snapshot().demographics
cats = [c for c in demo.columns if c!='household_key']

D = df.merge(demo[['household_key']+cats], on='household_key', how='left')
tr = D[D.snapshot_day.isin([d for d in all_train_days if d!=431])]
va = D[D.snapshot_day==431]
s = cats
n = feats
Xs_tr = pd.get_dummies(tr[s].astype(str), dummy_na=False).astype(float)
print("dummy shape:", Xs_tr.shape, "cols sample:", list(Xs_tr.columns[:8]), flush=True)
print("sum of dummies:", Xs_tr.values.sum(), "rows:", len(tr), flush=True)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
mkt = agent_api.load_saved('mkt_v2.parquet')
df = mkt.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]
all_train_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
demo = agent_api.snapshot().demographics
cats = [c for c in demo.columns if c!='household_key']

def design(D, s_cols, n_cols):
    Xn = D[n_cols].fillna(0).astype(float).values if n_cols else np.zeros((len(D),0))
    Xs = pd.get_dummies(D[s_cols].astype(str), dummy_na=False).astype(float).values if s_cols else np.zeros((len(D),0))
    return np.hstack([Xn, Xs])

def ridge_one(vd, alpha, use_demo):
    D = df if not use_demo else df.merge(demo[['household_key']+cats], on='household_key', how='left')
    tr = D[D.snapshot_day.isin([d for d in all_train_days if d!=vd])]
    va = D[D.snapshot_day==vd]
    Xtr = design(tr, cats if use_demo else [], feats)
    Xva = design(va, cats if use_demo else [], feats)
    ym = tr.future_spend_4w.mean()
    A = (Xtr.T@Xtr)/len(tr) + alpha*np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, (Xtr.T@(tr.future_spend_4w.values-ym))/len(tr))
    pv = np.clip(Xva@w + ym, 0, None)
    return np.abs(pv-va.future_spend_4w.values).mean()

for alpha in [10, 100]:
    print(f"alpha={alpha} no-demo:", [round(ridge_one(vd, alpha, False),2) for vd in [431,403]],
          " with-demo:", [round(ridge_one(vd, alpha, True),2) for vd in [431,403]], flush=True)


# ---- cell ----
import agent_api, pandas as pd, numpy as np

demo = agent_api.snapshot().demographics
cats = [c for c in demo.columns if c!='household_key']
demo = demo[['household_key']+cats].copy()
for c in cats:
    demo[c] = demo[c].astype(str).fillna('NA')

mkt = agent_api.load_saved('mkt_v2.parquet')

def fn(view, snapshot_day):
    base = mkt[mkt.snapshot_day==snapshot_day].set_index('household_key')
    D = base.join(demo.set_index('household_key'), how='left')
    D['d_has_demo'] = D['classification_1'].notna().astype(int)
    for c in cats:
        D[c] = D[c].fillna('NA')
    return D.drop(columns=['snapshot_day'], errors='ignore')

out = agent_api.build_features(fn)
out = out.reset_index()
print(out.shape, flush=True)
print(out.columns.tolist()[:10], "...", flush=True)
path = agent_api.save_table(out, 'e009_demo.parquet')
print(path, flush=True)
