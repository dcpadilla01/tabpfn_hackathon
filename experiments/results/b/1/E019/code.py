import agent_api as A
import pandas as pd, numpy as np

e015 = A.load_saved('e015_base.parquet')
print("e015 shape:", e015.shape)
print("cols:", list(e015.columns)[:130])
print()
print("saved tables:", A.describe_tables() if hasattr(A,'describe_tables') else '')

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

# Compute targets for train snapshots myself (windows end <= 459, visible at snapshot(459))
tt = A.train_targets()
print(tt.shape, tt.head())
snap = A.snapshot(459)
tr = snap.transactions[['household_key','day','sales_value']]
# future window spend for each train snapshot
days = A.snapshot_days()['train']
sp = tr.groupby('household_key')['sales_value'].sum()
# build window sums
g = tr.sort_values('day')
mine = {}
for d in days:
    w = tr[(tr.day > d) & (tr.day <= d+28)].groupby('household_key')['sales_value'].sum()
    mine[d] = w
mm = pd.concat([mine[d].rename(d) for d in days], axis=1)
mm = mm.stack().rename('y').reset_index().rename(columns={'level_1':'snapshot_day'})
mm['snapshot_day'] = mm['snapshot_day'].astype(int)
chk = tt.merge(mm, on=['household_key','snapshot_day'], how='left')
print("match rate:", (chk.future_spend_4w.round(2)==chk.y.round(2)).mean(), "maxdiff:", (chk.future_spend_4w-chk.y).abs().max())
print("train rows:", len(tt), "y mean/std:", tt.future_spend_4w.mean(), tt.future_spend_4w.std())
print("snapdays:", A.snapshot_days())

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

# --- my own targets for TRAIN snapshots (windows all <= day 459, visible) ---
tt = A.train_targets()
snap = A.snapshot(459)
tr = snap.transactions[['household_key','day','sales_value']]
days = A.snapshot_days()['train']
parts = []
for d in days:
    w = tr[(tr.day > d) & (tr.day <= d+28)].groupby('household_key')['sales_value'].sum()
    parts.append(pd.DataFrame({'household_key': w.index, 'snapshot_day': d, 'y': w.values}))
long = pd.concat(parts, ignore_index=True)
t = tt.merge(long, on=['household_key','snapshot_day'], how='left')
t['y'] = t['y'].fillna(0.0)
print("target check maxdiff:", (t.y - t.future_spend_4w).abs().max())
tgt = t[['household_key','snapshot_day','y']]
p = A.save_table(tgt, 'my_targets')
print("saved:", p)

# --- proxy eval: ridge, inner split (train snaps 95-319, val 347-431) ---
TR_SNAPS = [95,123,151,179,207,235,263,291,319]
VA_SNAPS = [347,375,403,431]

def prep(df):
    num, cat = [], []
    for c in df.columns:
        if c in ('household_key','snapshot_day'): continue
        if df[c].dtype.kind in 'ifb': num.append(c)
        else: cat.append(c)
    Xn = df[num].astype(float)
    Xn = Xn.fillna(Xn.median())
    mu, sd = Xn.mean(), Xn.std().replace(0,1)
    Xn = ((Xn-mu)/sd).fillna(0.0).clip(-5,5)
    Xc = pd.get_dummies(df[cat].astype(str), dummy_na=True) if cat else pd.DataFrame(index=df.index)
    X = pd.concat([Xn, Xc.astype(float)], axis=1)
    return X

def ridge_eval(path, lam=30.0, logt=False):
    df = A.load_saved(path)
    d = df.merge(tgt, on=['household_key','snapshot_day'], how='left')
    trm = d[d.snapshot_day.isin(TR_SNAPS)]; vam = d[d.snapshot_day.isin(VA_SNAPS)]
    Xtr, Xva = prep(trm), prep(vam)
    Xtr, Xva = Xtr.align(Xva, join='left', fill_value=0.0)
    ytr = trm.y.values; yva = vam.y.values
    if logt:
        ytr_f, yva_f = np.log1p(ytr), np.log1p(yva)
    else:
        ytr_f, yva_f = ytr, yva
    Z = np.vstack([Xtr.values, np.ones(len(Xtr))[:,None]*0+1])
    A_ = np.hstack([Xtr.values, np.ones((len(Xtr),1))])
    Av = np.hstack([Xva.values, np.ones((len(Xva),1))])
    D = np.eye(A_.shape[1]); D[-1,-1] = 0
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr_f)
    pv = Av@w
    if logt: pv = np.expm1(pv)
    pv = np.clip(pv, 0, None)
    return np.abs(pv - yva).mean(), Xtr.shape[1]

for path in ['e009_macro.parquet','e001_history.parquet','e013_union.parquet','e014_base.parquet','e015_base.parquet','e017_gapfill.parquet']:
    m1 = ridge_eval(path, 30.0, False)
    m2 = ridge_eval(path, 30.0, True)
    print(f"{path:24s} ridge raw {m1[0]:7.3f} ({m1[1]}f) | ridge log {m2[0]:7.3f}")

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tgt = A.load_saved('my_targets.parquet')
TR_SNAPS = [95,123,151,179,207,235,263,291,319]
VA_SNAPS = [347,375,403,431]

def prep(df):
    num, cat = [], []
    for c in df.columns:
        if c in ('household_key','snapshot_day'): continue
        if df[c].dtype.kind in 'ifb': num.append(c)
        else: cat.append(c)
    Xn = df[num].astype(float)
    Xn = Xn.fillna(Xn.median())
    mu, sd = Xn.mean(), Xn.std().replace(0,1)
    Xn = ((Xn-mu)/sd).fillna(0.0).clip(-5,5)
    Xc = pd.get_dummies(df[cat].astype(str), dummy_na=True) if cat else pd.DataFrame(index=df.index)
    X = pd.concat([Xn, Xc.astype(float)], axis=1)
    return X

def ridge_eval(path, lam=30.0, logt=False):
    df = A.load_saved(path)
    d = df.merge(tgt, on=['household_key','snapshot_day'], how='left')
    trm = d[d.snapshot_day.isin(TR_SNAPS)]; vam = d[d.snapshot_day.isin(VA_SNAPS)]
    Xtr, Xva = prep(trm), prep(vam)
    Xtr, Xva = Xtr.align(Xva, join='left', fill_value=0.0)
    ytr = trm.y.values; yva = vam.y.values
    ytr_f, yva_f = (np.log1p(ytr), np.log1p(yva)) if logt else (ytr, yva)
    A_ = np.hstack([Xtr.values, np.ones((len(Xtr),1))])
    Av = np.hstack([Xva.values, np.ones((len(Xva),1))])
    D = np.eye(A_.shape[1]); D[-1,-1] = 0
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr_f)
    pv = Av@w
    if logt: pv = np.expm1(pv)
    pv = np.clip(pv, 0, None)
    return np.abs(pv - yva).mean(), Xtr.shape[1]

for path in ['e009_macro.parquet','e001_history.parquet','e013_union.parquet','e014_base.parquet','e015_base.parquet','e017_gapfill.parquet','e012_full.parquet']:
    m1 = ridge_eval(path, 30.0, False)
    m2 = ridge_eval(path, 30.0, True)
    print(f"{path:24s} ridge raw {m1[0]:7.3f} ({m1[1]:4d}f) | ridge log {m2[0]:7.3f}")

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tgt = A.load_saved('my_targets.parquet')
TR_SNAPS = [95,123,151,179,207,235,263,291,319]
VA_SNAPS = [347,375,403,431]

def prep(df):
    num, cat = [], []
    for c in df.columns:
        if c in ('household_key','snapshot_day'): continue
        if df[c].dtype.kind in 'ifb': num.append(c)
        else: cat.append(c)
    Xn = df[num].astype(float)
    Xn = Xn.fillna(Xn.median())
    mu, sd = Xn.mean(), Xn.std().replace(0,1)
    Xn = ((Xn-mu)/sd).fillna(0.0).clip(-5,5)
    Xc = pd.get_dummies(df[cat].astype(str), dummy_na=True) if cat else pd.DataFrame(index=df.index)
    X = pd.concat([Xn, Xc.astype(float)], axis=1)
    return X

def ridge_eval(path, lam=30.0, logt=False):
    df = A.load_saved(path)
    d = df.merge(tgt, on=['household_key','snapshot_day'], how='left')
    trm = d[d.snapshot_day.isin(TR_SNAPS)]; vam = d[d.snapshot_day.isin(VA_SNAPS)]
    Xtr, Xva = prep(trm), prep(vam)
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
    ytr = trm.y.values; yva = vam.y.values
    ytr_f = np.log1p(ytr) if logt else ytr
    A_ = np.hstack([Xtr.values, np.ones((len(Xtr),1))])
    Av = np.hstack([Xva.values, np.ones((len(Xva),1))])
    D = np.eye(A_.shape[1]); D[-1,-1] = 0
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr_f)
    pv = Av@w
    if logt: pv = np.expm1(pv)
    pv = np.clip(pv, 0, None)
    return np.abs(pv - yva).mean(), Xtr.shape[1]

for path in ['e009_macro.parquet','e001_history.parquet','e013_union.parquet','e014_base.parquet','e015_base.parquet','e017_gapfill.parquet','e012_full.parquet']:
    m1 = ridge_eval(path, 30.0, False)
    m2 = ridge_eval(path, 30.0, True)
    print(f"{path:24s} ridge raw {m1[0]:7.3f} ({m1[1]:4d}f) | ridge log {m2[0]:7.3f}")

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tgt = A.load_saved('my_targets.parquet')
print("tgt:", tgt.shape, tgt.snapshot_day.unique(), tgt.y.mean())
df = A.load_saved('e009_macro.parquet')
print("df:", df.shape, sorted(df.snapshot_day.unique()))
d = df.merge(tgt, on=['household_key','snapshot_day'], how='left')
print("d:", d.shape, "y na:", d.y.isna().mean())
vam = d[d.snapshot_day.isin([347,375,403,431])]
print("vam:", len(vam), "y mean:", vam.y.mean(), "y na:", vam.y.isna().mean())
trm = d[d.snapshot_day.isin([95,123,151,179,207,235,263,291,319])]
print("trm:", len(trm), "y mean:", trm.y.mean())
print(d.dtypes.head(8))

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tgt = A.load_saved('my_targets.parquet')
df = A.load_saved('e009_macro.parquet')
d = df.merge(tgt, on=['household_key','snapshot_day'], how='left')
trm = d[d.snapshot_day.isin([95,123,151,179,207,235,263,291,319])]
vam = d[d.snapshot_day.isin([347,375,403,431])]

num = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind in 'ifb']
cat = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind not in 'ifb']
print("num:", len(num), "cat:", cat)

def prep(df):
    Xn = df[num].astype(float); Xn = Xn.fillna(Xn.median())
    mu, sd = Xn.mean(), Xn.std().replace(0,1)
    Xn = ((Xn-mu)/sd).fillna(0.0).clip(-5,5)
    Xc = pd.get_dummies(df[cat].astype(str), dummy_na=True) if cat else pd.DataFrame(index=df.index)
    return pd.concat([Xn, Xc.astype(float)], axis=1)

Xtr, Xva = prep(trm), prep(vam)
Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
ytr, yva = trm.y.values, vam.y.values
A_ = np.hstack([Xtr.values, np.ones((len(Xtr),1))])
Av = np.hstack([Xva.values, np.ones((len(Xva),1))])
D = np.eye(A_.shape[1]); D[-1,-1]=0
for lam in [1,10,30,100]:
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr)
    pv = np.clip(Av@w, 0, None)
    print(f"lam={lam}: pv mean {pv.mean():.1f} std {pv.std():.1f} | MAE {np.abs(pv-yva).mean():.2f} | baseline MAE(median {np.median(ytr):.0f}) {np.abs(np.median(ytr)-yva).mean():.2f}")
# also check: does Xtr row order match ytr? spot check first household
print(trm[['household_key','snapshot_day','y']].head(3))
print(Xtr.iloc[:3,:4])

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tgt = A.load_saved('my_targets.parquet')
TR_SNAPS = [95,123,151,179,207,235,263,291,319]
VA_SNAPS = [347,375,403,431]

def prep(d):
    num = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind in 'ifb']
    cat = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind not in 'ifb']
    Xn = d[num].astype(float); Xn = Xn.fillna(Xn.median())
    mu, sd = Xn.mean(), Xn.std().replace(0,1)
    Xn = ((Xn-mu)/sd).fillna(0.0).clip(-5,5)
    Xc = pd.get_dummies(d[cat].astype(str), dummy_na=True) if cat else pd.DataFrame(index=d.index)
    return pd.concat([Xn, Xc.astype(float)], axis=1)

def ridge_eval(path, lam=30.0):
    df = A.load_saved(path)
    d = df.merge(tgt, on=['household_key','snapshot_day'], how='left')
    trm = d[d.snapshot_day.isin(TR_SNAPS)]; vam = d[d.snapshot_day.isin(VA_SNAPS)]
    Xtr, Xva = prep(trm), prep(vam)
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
    ytr, yva = trm.y.values, vam.y.values
    A_ = np.hstack([Xtr.values, np.ones((len(Xtr),1))])
    Av = np.hstack([Xva.values, np.ones((len(Xva),1))])
    D = np.eye(A_.shape[1]); D[-1,-1]=0
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr)
    pv = np.clip(Av@w, 0, None)
    return np.abs(pv - yva).mean(), Xtr.shape[1]

for path in ['e015_base.parquet','e017_gapfill.parquet','e017r.parquet']:
    for lam in [30, 100, 300]:
        m, nf = ridge_eval(path, lam)
        print(f"{path:22s} lam={lam:4d} MAE {m:6.3f} ({nf}f)")
    print()

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tgt = A.load_saved('my_targets.parquet')
TR_SNAPS = [95,123,151,179,207,235,263,291,319]
VA_SNAPS = [347,375,403,431]

def prep(d):
    num = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind in 'ifb']
    cat = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind not in 'ifb']
    Xn = d[num].astype(float); Xn = Xn.fillna(Xn.median())
    mu, sd = Xn.mean(), Xn.std().replace(0,1)
    Xn = ((Xn-mu)/sd).fillna(0.0).clip(-5,5)
    Xc = pd.get_dummies(d[cat].astype(str), dummy_na=True) if cat else pd.DataFrame(index=d.index)
    return pd.concat([Xn, Xc.astype(float)], axis=1)

def ridge_eval(df, lam=100.0):
    d = df.merge(tgt, on=['household_key','snapshot_day'], how='left')
    trm = d[d.snapshot_day.isin(TR_SNAPS)]; vam = d[d.snapshot_day.isin(VA_SNAPS)]
    Xtr, Xva = prep(trm), prep(vam)
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
    ytr, yva = trm.y.values, vam.y.values
    A_ = np.hstack([Xtr.values, np.ones((len(Xtr),1))])
    Av = np.hstack([Xva.values, np.ones((len(Xva),1))])
    D = np.eye(A_.shape[1]); D[-1,-1]=0
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr)
    pv = np.clip(Av@w, 0, None)
    return np.abs(pv - yva).mean()

base = A.load_saved('e015_base.parquet')
print("e015 proxy MAE:", round(ridge_eval(base),3))

def clip01(x): return np.clip(x, 0, 1)
p13, p6, p3 = base.p13, base.p6, base.p3
u13, u6, u3 = base.usual13, base.usual6, base.usual3
e13, e6 = base.e13, base.e6
ly = base.ly_spend.fillna(0)

V = {}
V['medA'] = pd.DataFrame({
    'med13a': u13*clip01((p13-0.4)/0.2), 'med13b': u13*clip01(2*p13-1),
    'med6b': u6*clip01(2*p6-1), 'med3b': u3*clip01(2*p3-1),
    'med13_sq': u13*p13*p13, 'med_ly': np.maximum(u13*clip01(2*p13-1), 0.7*ly)})
V['medB'] = pd.DataFrame({
    'med13b': u13*clip01(2*p13-1), 'med6b': u6*clip01(2*p6-1),
    'med13_4b': base.usual13_4*clip01(2*base.p13_4-1),
    'med_ly_max': np.maximum(u13*clip01(2*p13-1), 0.7*ly),
    'med_geo': np.sqrt(np.maximum(e13,0)*u13)*clip01(2*p13-1)})
V['inter'] = pd.DataFrame({
    'e13_p13': e13*p13, 'u13_p13': u13*p13, 'e13_slope': e13*base.slope6.fillna(0),
    'p13_ten': p13*np.log1p(base.tenure), 'e13_wmax': e13*base.w_max7.fillna(0)})
V['rank'] = pd.DataFrame({
    'rk_e13': base.groupby('snapshot_day').e13.rank(pct=True),
    'rk_p13': base.groupby('snapshot_day').p13.rank(pct=True),
    'rk_u13': base.groupby('snapshot_day').u13.rank(pct=True) if 'u13' in base else base.groupby('snapshot_day').usual13.rank(pct=True),
    'rk_s28': base.groupby('snapshot_day').log_spend28.rank(pct=True)})
for k, v in V.items():
    v = v.fillna(0.0)
    df = pd.concat([base, v], axis=1)
    print(f"e015+{k:6s} proxy MAE: {ridge_eval(df):.3f}  (+{v.shape[1]}f)")

df_all = pd.concat([base] + [v.fillna(0.0) for v in V.values()], axis=1)
print("e015+ALL proxy MAE:", round(ridge_eval(df_all),3), df_all.shape)

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

def cand_fn(view, snapshot_day):
    d = int(snapshot_day)
    hhs = pd.Index(view.households)
    t = view.transactions
    t = t[t.household_key.isin(hhs)]
    if len(t) == 0:
        return pd.DataFrame(index=hhs)
    # window index k: 1 = (d-28, d], 2 = (d-56, d-28], ...
    k = ((d - t.day - 1) // 28 + 1).astype(int)
    t = t.assign(k=k)
    t = t[(t.k >= 1) & (t.k <= 13)]
    g = t.groupby(['household_key','k'])
    spend = g.sales_value.sum().unstack(fill_value=0.0)
    nb = g.basket_id.nunique().unstack(fill_value=0.0)
    K = range(1, 14)
    spend = spend.reindex(columns=K, fill_value=0.0).reindex(hhs, fill_value=0.0)
    nb = nb.reindex(columns=K, fill_value=0.0).reindex(hhs, fill_value=0.0)
    out = pd.DataFrame(index=hhs)
    for kk in K:
        out[f'a{kk}'] = spend[kk]
        out[f'al{kk}'] = np.log1p(spend[kk])
        out[f'nw{kk}'] = nb[kk]
        out[f'aw{kk}'] = (spend[kk] > 0).astype(float)
    arr = spend.values
    out['z6'] = (arr[:, :6] == 0).sum(1)
    out['z13'] = (arr == 0).sum(1)
    out['med13'] = np.median(arr, axis=1)
    nz = np.where(arr > 0, arr, np.nan)
    with np.errstate(all='ignore'):
        out['mednz13'] = np.nanmedian(nz, axis=1)
        out['q75_13'] = np.nanpercentile(arr, 75, axis=1)
        out['amax13'] = np.nanmax(arr, axis=1)
    out['ratio_23'] = out.a2 / (out.a3 + 1.0)
    out['ratio_2m'] = out.a2 / (out.med13 + 1.0)
    # median-shaped composites (need E015 cols; merge later outside fn) -> placeholder none
    return out

tab = A.build_features(cand_fn)
print(tab.shape)
p = A.save_table(tab, 'e019_cand')
print("saved", p)

# ---- proxy eval ----
tgt = A.load_saved('my_targets.parquet')
TR = [95,123,151,179,207,235,263,291,319]; VA = [347,375,403,431]
def prep(d):
    num = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind in 'ifb']
    cat = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind not in 'ifb']
    Xn = d[num].astype(float); Xn = Xn.fillna(Xn.median())
    mu, sd = Xn.mean(), Xn.std().replace(0,1)
    Xn = ((Xn-mu)/sd).fillna(0.0).clip(-5,5)
    Xc = pd.get_dummies(d[cat].astype(str), dummy_na=True) if cat else pd.DataFrame(index=d.index)
    return pd.concat([Xn, Xc.astype(float)], axis=1)
def ridge_eval(df, lam=100.0):
    d = df.merge(tgt, on=['household_key','snapshot_day'], how='left')
    trm = d[d.snapshot_day.isin(TR)]; vam = d[d.snapshot_day.isin(VA)]
    Xtr, Xva = prep(trm), prep(vam)
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
    ytr, yva = trm.y.values, vam.y.values
    A_ = np.hstack([Xtr.values, np.ones((len(Xtr),1))])
    Av = np.hstack([Xva.values, np.ones((len(Xva),1))])
    D = np.eye(A_.shape[1]); D[-1,-1]=0
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr)
    return np.abs(np.clip(Av@w,0,None) - yva).mean()

base = A.load_saved('e015_base.parquet')
cand = A.load_saved('e019_cand')
print("e015 proxy:", round(ridge_eval(base),3))
fams = {
 'grid_spend': [f'a{k}' for k in range(1,14)] + [f'al{k}' for k in range(1,14)],
 'grid_cnt': [f'nw{k}' for k in range(1,14)] + [f'aw{k}' for k in range(1,14)],
 'stats': ['z6','z13','med13','mednz13','q75_13','amax13','ratio_23','ratio_2m'],
}
for name, cols in fams.items():
    df = pd.concat([base, cand[cols]], axis=1)
    print(f"e015+{name:10s}: {ridge_eval(df):.3f}")
df = pd.concat([base, cand[[c for cols in fams.values() for c in cols]]], axis=1)
print("e015+ALLCAND:", round(ridge_eval(df),3), df.shape)

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

tgt = A.load_saved('my_targets.parquet')
TR = [95,123,151,179,207,235,263,291,319]; VA = [347,375,403,431]
def prep(d):
    num = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind in 'ifb']
    cat = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind not in 'ifb']
    Xn = d[num].astype(float); Xn = Xn.fillna(Xn.median())
    mu, sd = Xn.mean(), Xn.std().replace(0,1)
    Xn = ((Xn-mu)/sd).fillna(0.0).clip(-5,5)
    Xc = pd.get_dummies(d[cat].astype(str), dummy_na=True) if cat else pd.DataFrame(index=d.index)
    return pd.concat([Xn, Xc.astype(float)], axis=1)
def ridge_eval(df, lam=100.0):
    d = df.merge(tgt, on=['household_key','snapshot_day'], how='left')
    trm = d[d.snapshot_day.isin(TR)]; vam = d[d.snapshot_day.isin(VA)]
    Xtr, Xva = prep(trm), prep(vam)
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
    ytr, yva = trm.y.values, vam.y.values
    A_ = np.hstack([Xtr.values, np.ones((len(Xtr),1))])
    Av = np.hstack([Xva.values, np.ones((len(Xva),1))])
    D = np.eye(A_.shape[1]); D[-1,-1]=0
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr)
    return np.abs(np.clip(Av@w,0,None) - yva).mean()

base = A.load_saved('e015_base.parquet')
cand = A.load_saved('e019_cand.parquet')
print("e015 proxy:", round(ridge_eval(base),3))
fams = {
 'grid_spend': [f'a{k}' for k in range(1,14)] + [f'al{k}' for k in range(1,14)],
 'grid_cnt': [f'nw{k}' for k in range(1,14)] + [f'aw{k}' for k in range(1,14)],
 'stats': ['z6','z13','med13','mednz13','q75_13','amax13','ratio_23','ratio_2m'],
}
for name, cols in fams.items():
    df = pd.concat([base, cand[cols]], axis=1)
    print(f"e015+{name:10s}: {ridge_eval(df):.3f}")
df = pd.concat([base, cand[[c for cols in fams.values() for c in cols]]], axis=1)
print("e015+ALLCAND:", round(ridge_eval(df),3), df.shape)

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np
tgt = A.load_saved('my_targets.parquet')
cand = A.load_saved('e019_cand.parquet')
print(cand.dtypes.value_counts())
print([c for c in cand.columns if cand[c].dtype.kind not in 'ifb'][:20])

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np
tgt = A.load_saved('my_targets.parquet')
TR = [95,123,151,179,207,235,263,291,319]; VA = [347,375,403,431]
def prep(d):
    num = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind in 'ifb']
    cat = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind not in 'ifb']
    Xn = d[num].astype(float); Xn = Xn.fillna(Xn.median())
    mu, sd = Xn.mean(), Xn.std().replace(0,1)
    Xn = ((Xn-mu)/sd).fillna(0.0).clip(-5,5)
    Xc = pd.get_dummies(d[cat].astype(str), dummy_na=True) if cat else pd.DataFrame(index=d.index)
    return pd.concat([Xn, Xc.astype(float)], axis=1)
def ridge_eval(df, lam=100.0):
    d = df.merge(tgt, on=['household_key','snapshot_day'], how='left')
    trm = d[d.snapshot_day.isin(TR)]; vam = d[d.snapshot_day.isin(VA)]
    Xtr, Xva = prep(trm), prep(vam)
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
    ytr, yva = trm.y.values, vam.y.values
    A_ = np.hstack([Xtr.values, np.ones((len(Xtr),1))])
    Av = np.hstack([Xva.values, np.ones((len(Xva),1))])
    D = np.eye(A_.shape[1]); D[-1,-1]=0
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr)
    return np.abs(np.clip(Av@w,0,None) - yva).mean()

base = A.load_saved('e015_base.parquet')
cand = A.load_saved('e019_cand.parquet')
cand = cand.rename(columns={c: 'g_'+c for c in cand.columns if c not in ('household_key','snapshot_day')})
fams = {
 'grid_spend': [f'g_a{k}' for k in range(1,14)] + [f'g_al{k}' for k in range(1,14)],
 'grid_cnt': [f'g_nw{k}' for k in range(1,14)] + [f'g_aw{k}' for k in range(1,14)],
 'stats': ['g_z6','g_z13','g_med13','g_mednz13','g_q75_13','g_amax13','g_ratio_23','g_ratio_2m'],
}
print("e015 proxy:", round(ridge_eval(base),3))
for name, cols in fams.items():
    print(f"e015+{name:10s}: {ridge_eval(pd.concat([base, cand[cols]], axis=1)):.3f}")
allc = [c for cols in fams.values() for c in cols]
df = pd.concat([base, cand[allc]], axis=1)
print("e015+ALLCAND:", round(ridge_eval(df),3), df.shape)
p = A.save_table(df, 'e019_final')
print("saved:", p)

# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np
tgt = A.load_saved('my_targets.parquet')
TR = [95,123,151,179,207,235,263,291,319]; VA = [347,375,403,431]
def prep(d):
    num = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind in 'ifb']
    cat = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind not in 'ifb']
    Xn = d[num].astype(float); Xn = Xn.fillna(Xn.median())
    mu, sd = Xn.mean(), Xn.std().replace(0,1)
    Xn = ((Xn-mu)/sd).fillna(0.0).clip(-5,5)
    Xc = pd.get_dummies(d[cat].astype(str), dummy_na=True) if cat else pd.DataFrame(index=d.index)
    return pd.concat([Xn, Xc.astype(float)], axis=1)
def ridge_eval(df, lam=100.0):
    d = df.merge(tgt, on=['household_key','snapshot_day'], how='left')
    trm = d[d.snapshot_day.isin(TR)]; vam = d[d.snapshot_day.isin(VA)]
    Xtr, Xva = prep(trm), prep(vam)
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
    ytr, yva = trm.y.values, vam.y.values
    A_ = np.hstack([Xtr.values, np.ones((len(Xtr),1))])
    Av = np.hstack([Xva.values, np.ones((len(Xva),1))])
    D = np.eye(A_.shape[1]); D[-1,-1]=0
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr)
    return np.abs(np.clip(Av@w,0,None) - yva).mean()

base = A.load_saved('e015_base.parquet')
c01 = lambda x: np.clip(x, 0, 1)
p13, p6, p3 = base.p13, base.p6, base.p3
u13, u6, u3 = base.usual13, base.usual6, base.usual3
e13, e6 = base.e13, base.e6
ly = base.ly_spend.fillna(0)
M = pd.DataFrame({
 'med13a': u13*c01((p13-0.4)/0.2), 'med13b': u13*c01(2*p13-1),
 'med6b': u6*c01(2*p6-1), 'med3b': u3*c01(2*p3-1),
 'med13_sq': u13*p13*p13, 'med_ly': np.maximum(u13*c01(2*p13-1), 0.7*ly),
 'med13_4b': base.usual13_4*c01(2*base.p13_4-1),
 'med_geo': np.sqrt(np.maximum(e13,0)*u13)*c01(2*p13-1),
 'e13_p13': e13*p13, 'u13_p13': u13*p13,
 'e13_slope': e13*base.slope6.fillna(0), 'p13_ten': p13*np.log1p(base.tenure),
 'e13_wmax': e13*base.w_max7.fillna(0)}).fillna(0.0)
df = pd.concat([base, M], axis=1)
for lam in [50, 100, 200]:
    print(f"e015+MEDCOMBO lam={lam}: {ridge_eval(df, lam):.3f}")
p = A.save_table(df, 'e019_medcombo')
print("saved:", p, df.shape)