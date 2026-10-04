import agent_api, pandas as pd, numpy as np
for name in ['e009_demo','e009_analog','e009_spline2p','e009_demo_mkt','e008_fwd_calendar']:
    try:
        df = agent_api.load_saved(name+'.parquet')
        cols = [c for c in df.columns if c not in ('household_key','snapshot_day')]
        print('==',name, df.shape)
        print(cols)
    except Exception as e:
        print(name, 'ERR', e)
tt = agent_api.train_targets()
print('\nTARGET', tt.shape)
print(tt.future_spend_4w.describe())
print('zero frac:', (tt.future_spend_4w==0).mean())
print(tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median','count']))


# ---- cell ----
import pandas as pd
for name in ['e009_analog','e009_spline2p','e009_demo_mkt']:
    df = agent_api.load_saved(name+'.parquet')
    cols = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    print('==',name, df.shape, len(cols))
    print(cols[-40:])
    print()


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def ridge_fit_pred(Xtr, ytr, Xva, alpha):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Z = (Xtr-mu)/sd; Zv = (Xva-mu)/sd
    Z = np.c_[np.ones(len(Z)), Z]; Zv = np.c_[np.ones(len(Zv)), Zv]
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Z.T@ytr)
    return Zv@w

keys = ['household_key','snapshot_day']
base = agent_api.load_saved('e008_fwd_calendar.parquet').set_index(keys)
demo = agent_api.load_saved('e009_demo.parquet').set_index(keys)
ana  = agent_api.load_saved('e009_analog.parquet').set_index(keys)
sp2  = agent_api.load_saved('e009_spline2p.parquet').set_index(keys)
tt = agent_api.train_targets().set_index(keys)
sd = agent_api.snapshot_days()
tr_days, va_days = sd['train'], sd['validation']

def feats(df, cols=None):
    X = df.drop(columns=['household_key','snapshot_day'], errors='ignore') if cols is None else df[cols]
    return X.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)

# column groups
demo_only = [c for c in demo.columns if c not in base.columns]
ana_only  = [c for c in ana.columns  if c not in base.columns]
sp2_only  = [c for c in sp2.columns  if c not in base.columns]
print('groups:', len(demo_only), len(ana_only), len(sp2_only))

def eval_variant(cols_list, alphas=(30.,100.,300.,1000.)):
    # inner: fit on train days < 431, tune alpha on 431
    Xtr_all = []; ytr_all = []; Xtr_in = []; ytr_in = []
    for d in tr_days:
        idx = base.index[base.snapshot_day==d] if isinstance(base.index, pd.MultiIndex) else None
    # simpler: use merged frame
    df = base.copy()
    df['y'] = tt['future_spend_4w'].reindex(df.index)
    tr = df[df.snapshot_day.isin(tr_days)]
    va = df[df.snapshot_day.isin(va_days)]
    inner = df[df.snapshot_day==431]; trin = df[df.snapshot_day.isin(tr_days[:-1])]
    def build(cols):
        F = lambda d: np.c_[feats(d), feats(d, cols)] if cols else feats(d)
        return F(trin), F(inner), F(tr), F(va)
    Xa,Xb,Xc,Xd = build(cols_list)
    ya,yb,yc = trin.y.values, inner.y.values, tr.y.values
    best=None
    for al in alphas:
        p = ridge_fit_pred(Xa,ya,Xb,al)
        m = np.abs(p-yb).mean()
        if best is None or m<best[1]: best=(al,m)
    al = best[0]
    pv = ridge_fit_pred(Xc,yc,Xd,al)
    return np.abs(pv-va.y.values).mean(), al, best[1]

for name, cols in [('E008 base',[]), ('+demo',demo_only), ('+analog',ana_only), ('+2p',sp2_only),
                   ('+demo+analog',demo_only+ana_only), ('+demo+2p',demo_only+sp2_only),
                   ('+demo+ana+2p',demo_only+ana_only+sp2_only)]:
    mae, al, inner_mae = eval_variant(cols)
    print(f'{name:16s} valMAE={mae:.3f} alpha={al} innerMAE={inner_mae:.3f}')


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def ridge_fit_pred(Xtr, ytr, Xva, alpha):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Z = (Xtr-mu)/sd; Zv = (Xva-mu)/sd
    Z = np.c_[np.ones(len(Z)), Z]; Zv = np.c_[np.ones(len(Zv)), Zv]
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Z.T@ytr)
    return Zv@w

keys = ['household_key','snapshot_day']
base = agent_api.load_saved('e008_fwd_calendar.parquet').set_index(keys)
demo = agent_api.load_saved('e009_demo.parquet').set_index(keys)
ana  = agent_api.load_saved('e009_analog.parquet').set_index(keys)
sp2  = agent_api.load_saved('e009_spline2p.parquet').set_index(keys)
tt = agent_api.train_targets().set_index(keys)
sd = agent_api.snapshot_days()
tr_days, va_days = sd['train'], sd['validation']

def feats(df, cols=None):
    X = df.drop(columns=['household_key','snapshot_day'], errors='ignore') if cols is None else df[cols]
    return X.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)

demo_only = [c for c in demo.columns if c not in base.columns]
ana_only  = [c for c in ana.columns  if c not in base.columns]
sp2_only  = [c for c in sp2.columns  if c not in base.columns]

df = base.copy()
df['y'] = tt['future_spend_4w'].reindex(df.index)
day = df.index.get_level_values(1)
tr = df[day.isin(tr_days)]; va = df[day.isin(va_days)]
inner = df[day==431]; trin = df[day.isin(tr_days[:-1])]

def eval_variant(cols, alphas=(30.,100.,300.,1000.)):
    F = lambda d: np.c_[feats(d), feats(d, cols)] if cols else feats(d)
    Xa,Xb,Xc,Xd = F(trin), F(inner), F(tr), F(va)
    ya,yb,yc = trin.y.values, inner.y.values, tr.y.values
    best=None
    for al in alphas:
        p = ridge_fit_pred(Xa,ya,Xb,al)
        m = np.abs(p-yb).mean()
        if best is None or m<best[1]: best=(al,m)
    al = best[0]
    pv = ridge_fit_pred(Xc,yc,Xd,al)
    return np.abs(pv-va.y.values).mean(), al, best[1]

for name, cols in [('E008 base',[]), ('+demo',demo_only), ('+analog',ana_only), ('+2p',sp2_only),
                   ('+demo+analog',demo_only+ana_only), ('+demo+2p',demo_only+sp2_only),
                   ('+demo+ana+2p',demo_only+ana_only+sp2_only)]:
    mae, al, im = eval_variant(cols)
    print(f'{name:16s} valMAE={mae:.3f} alpha={al} innerMAE={im:.3f}')


# ---- cell ----
import agent_api, pandas as pd, numpy as np
base = agent_api.load_saved('e008_fwd_calendar.parquet')
tt = agent_api.train_targets()
print(base.index.dtype, base.columns[:3].tolist(), base.shape)
print(tt.head(3))
m = base.merge(tt, on=['household_key','snapshot_day'], how='left')
print(m.shape, m.future_spend_4w.isna().sum())
print(m[m.snapshot_day==431].future_spend_4w.describe())


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def ridge_fit_pred(Xtr, ytr, Xva, alpha):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Z = (Xtr-mu)/sd; Zv = (Xva-mu)/sd
    Z = np.c_[np.ones(len(Z)), Z]; Zv = np.c_[np.ones(len(Zv)), Zv]
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Z.T@ytr)
    return Zv@w

keys = ['household_key','snapshot_day']
base = agent_api.load_saved('e008_fwd_calendar.parquet')
demo = agent_api.load_saved('e009_demo.parquet')
ana  = agent_api.load_saved('e009_analog.parquet')
sp2  = agent_api.load_saved('e009_spline2p.parquet')
tt = agent_api.train_targets()
sd = agent_api.snapshot_days(); tr_days, va_days = sd['train'], sd['validation']

def feats(df):
    X = df.drop(columns=['household_key','snapshot_day','future_spend_4w'], errors='ignore')
    return X.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)

demo_only = [c for c in demo.columns if c not in base.columns]
ana_only  = [c for c in ana.columns  if c not in base.columns]
sp2_only  = [c for c in sp2.columns  if c not in base.columns]

def frame(extra, extra_df):
    df = base.merge(extra_df, on=keys, how='left') if extra else base
    df = df.merge(tt, on=keys, how='left')
    return df

day = base.snapshot_day
tr_mask = base.snapshot_day.isin(tr_days); va_mask = base.snapshot_day.isin(va_days)

def eval_variant(cols, extra_df, alphas=(30.,100.,300.,1000.)):
    df = frame(bool(cols), extra_df)
    d = df.snapshot_day.values
    trin = df[np.isin(d, tr_days[:-1])]; inner = df[d==431]
    tr = df[np.isin(d, tr_days)]; va = df[np.isin(d, va_days)]
    F = lambda x: np.c_[feats(x), feats(x[cols])] if cols else feats(x)
    Xa,Xb,Xc,Xd = F(trin), F(inner), F(tr), F(va)
    ya,yb,yc = trin.future_spend_4w.values, inner.future_spend_4w.values, tr.future_spend_4w.values
    best=None
    for al in alphas:
        p = ridge_fit_pred(Xa,ya,Xb,al); m = np.abs(p-yb).mean()
        if best is None or m<best[1]: best=(al,m)
    al=best[0]
    pv = ridge_fit_pred(Xc,yc,Xd,al)
    return np.abs(pv-va.future_spend_4w.values).mean(), al, best[1]

for name, cols, edf in [('E008 base',[],None), ('+demo',demo_only,demo), ('+analog',ana_only,ana),
                   ('+2p',sp2_only,sp2), ('+demo+analog',demo_only+ana_only,ana),
                   ('+demo+2p',demo_only+sp2_only,sp2), ('+demo+ana+2p',demo_only+ana_only+sp2_only,sp2)]:
    mae, al, im = eval_variant(cols, edf)
    print(f'{name:16s} valMAE={mae:.3f} alpha={al} innerMAE={im:.3f}')


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def ridge_fit_pred(Xtr, ytr, Xva, alpha):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[~np.isfinite(sd)|(sd==0)]=1
    Z = (Xtr-mu)/sd; Zv = (Xva-mu)/sd
    Z = np.clip(np.c_[np.ones(len(Z)), Z], -50, 50)
    Zv = np.clip(np.c_[np.ones(len(Zv)), Zv], -50, 50)
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Z.T@ytr)
    return np.clip(Zv@w, 0, None)

keys = ['household_key','snapshot_day']
base = agent_api.load_saved('e008_fwd_calendar.parquet')
demo = agent_api.load_saved('e009_demo.parquet')
ana  = agent_api.load_saved('e009_analog.parquet')
sp2  = agent_api.load_saved('e009_spline2p.parquet')
tt = agent_api.train_targets()
sd = agent_api.snapshot_days(); tr_days, va_days = sd['train'], sd['validation']

def feats(df):
    X = df.drop(columns=['household_key','snapshot_day','future_spend_4w'], errors='ignore')
    A = X.apply(pd.to_numeric, errors='coerce').values.astype(float)
    return np.nan_to_num(A, nan=0., posinf=0., neginf=0.)

demo_only = [c for c in demo.columns if c not in base.columns]
ana_only  = [c for c in ana.columns  if c not in base.columns]
sp2_only  = [c for c in sp2.columns  if c not in base.columns]

def eval_variant(cols, extra_df, alphas=(30.,100.,300.,1000.)):
    df = base.merge(extra_df, on=keys, how='left') if cols else base
    df = df.merge(tt, on=keys, how='left')
    d = df.snapshot_day.values
    trin = df[np.isin(d, tr_days[:-1])]; inner = df[d==431]
    F = lambda x: np.c_[feats(x), feats(x[cols])] if cols else feats(x)
    Xa,Xb = F(trin), F(inner)
    ya,yb = trin.future_spend_4w.values, inner.future_spend_4w.values
    best=None
    for al in alphas:
        p = ridge_fit_pred(Xa,ya,Xb,al); m = np.abs(p-yb).mean()
        if best is None or m<best[1]: best=(al,m)
    return best

for name, cols, edf in [('E008 base',[],None), ('+demo',demo_only,demo), ('+analog',ana_only,ana),
                   ('+2p',sp2_only,sp2), ('+demo+analog',demo_only+ana_only,ana),
                   ('+demo+2p',demo_only+sp2_only,sp2), ('+demo+ana+2p',demo_only+ana_only+sp2_only,sp2)]:
    al, im = eval_variant(cols, edf)
    print(f'{name:16s} alpha={al} innerMAE={im:.3f}')


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def ridge_fit_pred(Xtr, ytr, Xva, alpha):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Z = (Xtr-mu)/sd; Zv = (Xva-mu)/sd
    Z = np.clip(np.c_[np.ones(len(Z)), Z], -50, 50)
    Zv = np.clip(np.c_[np.ones(len(Zv)), Zv], -50, 50)
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Z.T@ytr)
    return np.clip(Zv@w, 0, None)

keys = ['household_key','snapshot_day']
base = agent_api.load_saved('e008_fwd_calendar.parquet')
demo = agent_api.load_saved('e009_demo.parquet')
ana  = agent_api.load_saved('e009_analog.parquet')
sp2  = agent_api.load_saved('e009_spline2p.parquet')
tt = agent_api.train_targets()
sd = agent_api.snapshot_days(); tr_days, va_days = sd['train'], sd['validation']

def feats(df):
    X = df.drop(columns=['household_key','snapshot_day','future_spend_4w'], errors='ignore')
    A = X.apply(pd.to_numeric, errors='coerce').values.astype(float)
    return np.nan_to_num(A, nan=0., posinf=0., neginf=0.)

demo_only = [c for c in demo.columns if c not in base.columns]
ana_only  = [c for c in ana.columns  if c not in base.columns]
sp2_only  = [c for c in sp2.columns  if c not in base.columns]

def clean_cols(cols, df):
    # drop constant or duplicate columns
    keep, seen = [], set()
    for c in cols:
        v = df[c].apply(pd.to_numeric, errors='coerce').fillna(0).values
        if np.std(v) < 1e-12: continue
        h = hash(np.round(v.astype(np.float64),6).tobytes())
        if h in seen: continue
        seen.add(h); keep.append(c)
    return keep

def eval_variant(cols, extra_df, alphas=(30.,100.,300.,1000.)):
    df = base.merge(extra_df, on=keys, how='left') if cols else base
    df = df.merge(tt, on=keys, how='left')
    d = df.snapshot_day.values
    trin = df[np.isin(d, tr_days[:-1])]; inner = df[d==431]
    F = lambda x: np.c_[feats(x), feats(x[cols])] if cols else feats(x)
    Xa,Xb = F(trin), F(inner)
    ya,yb = trin.future_spend_4w.values, inner.future_spend_4w.values
    best=None
    for al in alphas:
        p = ridge_fit_pred(Xa,ya,Xb,al); m = np.abs(p-yb).mean()
        if best is None or m<best[1]: best=(al,m)
    return best

sp2c = clean_cols(sp2_only, sp2)
print('sp2 cols', len(sp2_only), '->', len(sp2c))
for name, cols, edf in [('E008 base',[],None), ('+2p',sp2c,sp2), ('+demo+2p',demo_only+sp2c,sp2),
                        ('+demo+ana+2p',demo_only+ana_only+sp2c,sp2)]:
    al, im = eval_variant(cols, edf)
    print(f'{name:16s} alpha={al} innerMAE={im:.3f}')


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def ridge_fit_pred(Xtr, ytr, Xva, alpha):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Z = (Xtr-mu)/sd; Zv = (Xva-mu)/sd
    Z = np.clip(np.c_[np.ones(len(Z)), Z], -50, 50)
    Zv = np.clip(np.c_[np.ones(len(Zv)), Zv], -50, 50)
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Z.T@ytr)
    return np.clip(Zv@w, 0, None)

keys = ['household_key','snapshot_day']
base = agent_api.load_saved('e008_fwd_calendar.parquet')
demo = agent_api.load_saved('e009_demo.parquet')
ana  = agent_api.load_saved('e009_analog.parquet')
sp2  = agent_api.load_saved('e009_spline2p.parquet')
tt = agent_api.train_targets()
sd = agent_api.snapshot_days(); tr_days, va_days = sd['train'], sd['validation']

def feats(df):
    X = df.drop(columns=['household_key','snapshot_day','future_spend_4w'], errors='ignore')
    A = X.apply(pd.to_numeric, errors='coerce').values.astype(float)
    return np.nan_to_num(A, nan=0., posinf=0., neginf=0.)

demo_only = [c for c in demo.columns if c not in base.columns]
ana_only  = [c for c in ana.columns  if c not in base.columns]
sp2_only  = [c for c in sp2.columns  if c not in base.columns]

def clean_cols(cols, df):
    keep, seen = [], set()
    for c in cols:
        v = df[c].apply(pd.to_numeric, errors='coerce').fillna(0).values
        if np.std(v) < 1e-12: continue
        h = hash(np.round(v.astype(np.float64),6).tobytes())
        if h in seen: continue
        seen.add(h); keep.append(c)
    return keep

sp2c = clean_cols(sp2_only, sp2)

def eval_variant(cols, dfs, alphas=(30.,100.,300.,1000.)):
    df = base
    for d in dfs: df = df.merge(d, on=keys, how='left')
    df = df.merge(tt, on=keys, how='left')
    d = df.snapshot_day.values
    trin = df[np.isin(d, tr_days[:-1])]; inner = df[d==431]
    F = lambda x: np.c_[feats(x), feats(x[cols])] if cols else feats(x)
    Xa,Xb = F(trin), F(inner)
    ya,yb = trin.future_spend_4w.values, inner.future_spend_4w.values
    best=None
    for al in alphas:
        p = ridge_fit_pred(Xa,ya,Xb,al); m = np.abs(p-yb).mean()
        if best is None or m<best[1]: best=(al,m)
    return best

for name, cols, dfs in [('E008 base',[],[]), ('+demo',demo_only,[demo]), ('+analog',ana_only,[ana]),
                   ('+2p',sp2c,[sp2]), ('+demo+2p',demo_only+sp2c,[demo,sp2]),
                   ('+demo+ana+2p',demo_only+ana_only+sp2c,[demo,ana,sp2])]:
    al, im = eval_variant(cols, dfs)
    print(f'{name:16s} alpha={al} innerMAE={im:.3f}')


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def ridge_fit_pred(Xtr, ytr, Xva, alpha):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Z = (Xtr-mu)/sd; Zv = (Xva-mu)/sd
    Z = np.clip(np.c_[np.ones(len(Z)), Z], -50, 50)
    Zv = np.clip(np.c_[np.ones(len(Zv)), Zv], -50, 50)
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Z.T@ytr)
    return np.clip(Zv@w, 0, None)

keys = ['household_key','snapshot_day']
base = agent_api.load_saved('e008_fwd_calendar.parquet')
demo = agent_api.load_saved('e009_demo.parquet')
ana  = agent_api.load_saved('e009_analog.parquet')
sp2  = agent_api.load_saved('e009_spline2p.parquet')
tt = agent_api.train_targets()
sd = agent_api.snapshot_days(); tr_days, va_days = sd['train'], sd['validation']

def feats(df):
    X = df.drop(columns=['household_key','snapshot_day','future_spend_4w'], errors='ignore')
    A = X.apply(pd.to_numeric, errors='coerce').values.astype(float)
    return np.nan_to_num(A, nan=0., posinf=0., neginf=0.)

demo_only = [c for c in demo.columns if c not in base.columns]
ana_only  = [c for c in ana.columns  if c not in base.columns]
sp2_only  = [c for c in sp2.columns  if c not in base.columns]

def clean_cols(cols, df):
    keep, seen = [], set()
    for c in cols:
        v = df[c].apply(pd.to_numeric, errors='coerce').fillna(0).values
        if np.std(v) < 1e-12: continue
        h = hash(np.round(v.astype(np.float64),6).tobytes())
        if h in seen: continue
        seen.add(h); keep.append(c)
    return keep

sp2c = clean_cols(sp2_only, sp2)

def eval_variant(cols, dfs, alphas=(30.,100.,300.,1000.)):
    df = base
    for d in dfs:
        extra = [c for c in d.columns if c not in df.columns]
        df = df.merge(d[keys+extra], on=keys, how='left')
    df = df.merge(tt, on=keys, how='left')
    d = df.snapshot_day.values
    trin = df[np.isin(d, tr_days[:-1])]; inner = df[d==431]
    F = lambda x: np.c_[feats(x), feats(x[cols])] if cols else feats(x)
    Xa,Xb = F(trin), F(inner)
    ya,yb = trin.future_spend_4w.values, inner.future_spend_4w.values
    best=None
    for al in alphas:
        p = ridge_fit_pred(Xa,ya,Xb,al); m = np.abs(p-yb).mean()
        if best is None or m<best[1]: best=(al,m)
    return best

for name, cols, dfs in [('E008 base',[],[]), ('+demo',demo_only,[demo]), ('+analog',ana_only,[ana]),
                   ('+2p',sp2c,[sp2]), ('+demo+2p',demo_only+sp2c,[demo,sp2]),
                   ('+demo+ana+2p',demo_only+ana_only+sp2c,[demo,ana,sp2])]:
    al, im = eval_variant(cols, dfs)
    print(f'{name:16s} alpha={al} innerMAE={im:.3f}')


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def ridge_fit_pred(Xtr, ytr, Xva, alpha):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Z = (Xtr-mu)/sd; Zv = (Xva-mu)/sd
    Z = np.clip(np.c_[np.ones(len(Z)), Z], -50, 50)
    Zv = np.clip(np.c_[np.ones(len(Zv)), Zv], -50, 50)
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Z.T@ytr)
    return np.clip(Zv@w, 0, None)

keys = ['household_key','snapshot_day']
base = agent_api.load_saved('e008_fwd_calendar.parquet')
demo = agent_api.load_saved('e009_demo.parquet')
sp2  = agent_api.load_saved('e009_spline2p.parquet')
tt = agent_api.train_targets()
sd = agent_api.snapshot_days(); tr_days, va_days = sd['train'], sd['validation']

def feats(df):
    X = df.drop(columns=['household_key','snapshot_day','future_spend_4w'], errors='ignore')
    A = X.apply(pd.to_numeric, errors='coerce').values.astype(float)
    return np.nan_to_num(A, nan=0., posinf=0., neginf=0.)

demo_only = [c for c in demo.columns if c not in base.columns]
sp2_only  = [c for c in sp2.columns  if c not in base.columns]

def clean_cols(cols, df):
    keep, seen = [], set()
    for c in cols:
        v = df[c].apply(pd.to_numeric, errors='coerce').fillna(0).values
        if np.std(v) < 1e-12: continue
        h = hash(np.round(v.astype(np.float64),6).tobytes())
        if h in seen: continue
        seen.add(h); keep.append(c)
    return keep
sp2c = clean_cols(sp2_only, sp2)

def eval_variant(cols, dfs, alphas=(100.,300.,1000.,3000.,10000.)):
    df = base
    for d in dfs:
        extra = [c for c in d.columns if c not in df.columns]
        df = df.merge(d[keys+extra], on=keys, how='left')
    df = df.merge(tt, on=keys, how='left')
    d = df.snapshot_day.values
    F = lambda x: np.c_[feats(x), feats(x[cols])] if cols else feats(x)
    res = {}
    for inner_day in [403, 431]:
        trin = df[np.isin(d, [x for x in tr_days if x < inner_day])]
        inner = df[d==inner_day]
        Xa,Xb = F(trin), F(inner)
        ya,yb = trin.future_spend_4w.values, inner.future_spend_4w.values
        best=None
        for al in alphas:
            p = ridge_fit_pred(Xa,ya,Xb,al); m = np.abs(p-yb).mean()
            if best is None or m<best[1]: best=(al,m)
        res[inner_day]=best
    return res

for name, cols, dfs in [('E008 base',[],[]), ('+demo',demo_only,[demo]), ('+2p',sp2c,[sp2]),
                        ('+demo+2p',demo_only+sp2c,[demo,sp2])]:
    r = eval_variant(cols, dfs)
    m = np.mean([v[1] for v in r.values()])
    print(f'{name:12s} ' + ' '.join(f'd{d}:{v[1]:.3f}@a{v[0]:.0f}' for d,v in r.items()) + f'  avg={m:.3f}')


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def ridge_fit_pred(Xtr, ytr, Xva, alpha):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Z = (Xtr-mu)/sd; Zv = (Xva-mu)/sd
    Z = np.clip(np.c_[np.ones(len(Z)), Z], -50, 50)
    Zv = np.clip(np.c_[np.ones(len(Zv)), Zv], -50, 50)
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Z.T@ytr)
    return np.clip(Zv@w, 0, None)

keys = ['household_key','snapshot_day']
base = agent_api.load_saved('e009_demo.parquet')  # E009 = current best
tt = agent_api.train_targets()
sd = agent_api.snapshot_days(); tr_days = sd['train']
df = base.merge(tt, on=keys, how='left')
fcols = [c for c in df.columns if c not in keys+['future_spend_4w']]
d = df.snapshot_day.values
trin = df[np.isin(d, [x for x in tr_days if x<431])]; inner = df[d==431]
ytr = trin.future_spend_4w.values; yin = inner.future_spend_4w.values
Xtr = trin[fcols].apply(pd.to_numeric, errors='coerce').fillna(0).values
Xin = inner[fcols].apply(pd.to_numeric, errors='coerce').fillna(0).values
# baseline inner MAE at alpha 3000
p = ridge_fit_pred(Xtr, ytr, Xin, 3000.)
print('E009 inner MAE d431:', np.abs(p-yin).mean())
# feature correlations with target
corr = {}
for i,c in enumerate(fcols):
    v = Xtr[:,i]
    if np.std(v)>0: corr[c] = np.corrcoef(v, ytr)[0,1]
top = sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:30]
for c,v in top: print(f'{c:22s} {v:.3f}')


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def ridge_fit_pred(Xtr, ytr, Xva, alpha):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Z = (Xtr-mu)/sd; Zv = (Xva-mu)/sd
    Z = np.clip(np.c_[np.ones(len(Z)), Z], -50, 50)
    Zv = np.clip(np.c_[np.ones(len(Zv)), Zv], -50, 50)
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Z.T@ytr)
    return np.clip(Zv@w, 0, None)

keys=['household_key','snapshot_day']
df = agent_api.load_saved('e009_demo.parquet').merge(agent_api.train_targets(), on=keys, how='left')
fcols=[c for c in df.columns if c not in keys+['future_spend_4w']]
d=df.snapshot_day.values; tr_days=agent_api.snapshot_days()['train']

def get_X(df, mode):
    X = df[fcols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    if mode=='base': return X
    extra=[]
    # hinges on top spend features at percentiles 25/50/75 of train dist
    top=['spend_84','spend_28','fwd28_mean','wk_avg_84','spend_lag1','ewma28_4w','x_life_rate_wk']
    for c in top:
        v = df[c].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
        for q in (0.25,0.5,0.75):
            t = np.quantile(v, q)
            extra.append(np.maximum(v-t,0))
    if mode=='hinge':
        return np.c_[X, np.array(extra).T]
    if mode=='rank':
        # rank-transform top 20 heavy-tailed features
        rtop=['spend_84','spend_28','fwd28_mean','wk_avg_84','spend_lag1','ewma28_4w','x_life_rate_wk',
              'spend_364','spend_life','x_life_total','spend_56','fwd28_k1','fwd28_median','wk_avg_8',
              'spend_112','spend_168','x_s28','fwd28_max','spend_7','x_ya4w']
        R = df[rtop].apply(pd.to_numeric, errors='coerce').fillna(0).rank(pct=True).values
        return np.c_[X, R]
    if mode=='hinge+rank':
        R = df[['spend_84','fwd28_mean']].apply(pd.to_numeric, errors='coerce').fillna(0).rank(pct=True).values
        return np.c_[X, np.array(extra).T, R]

for mode in ['base','hinge','rank','hinge+rank']:
    mas=[]
    for inner_day in [403,431]:
        trin=df[np.isin(d,[x for x in tr_days if x<inner_day])]; inner=df[d==inner_day]
        Xtr,Xin = get_X(trin,mode), get_X(inner,mode)
        p = ridge_fit_pred(Xtr, trin.future_spend_4w.values, Xin, 3000.)
        mas.append(np.abs(p-inner.future_spend_4w.values).mean())
    print(f'{mode:12s} d403:{mas[0]:.3f} d431:{mas[1]:.3f} avg:{np.mean(mas):.3f}')


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def ridge_fit_pred(Xtr, ytr, Xva, alpha):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
    Z=np.clip((Xtr-mu)/sd,-30,30); Zv=np.clip((Xva-mu)/sd,-30,30)
    Z=np.c_[np.ones(len(Z)),Z]; Zv=np.c_[np.ones(len(Zv)),Zv]
    A=Z.T@Z+alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    return np.clip(Zv@np.linalg.solve(A,Z.T@ytr),0,None)

hist = agent_api.snapshot(431).transactions[['household_key','day','sales_value','basket_id']]
hh_needed = agent_api.snapshot(431).households
print('hh at 431:', len(hh_needed), 'trans rows:', len(hist))
g = hist.groupby(['household_key','day'], as_index=False).sales_value.sum()
# spend in (a,b] helper via per-household sorted day sums
hh = g.household_key.values; dy = g.day.values; sv = g.sales_value.values
order = np.lexsort((dy, hh)); hh,dy,sv = hh[order],dy[order],sv[order]
starts = np.searchsorted(hh, np.unique(hh), side='left'); ends = np.searchsorted(hh, np.unique(hh), side='right')
uniq = np.unique(hh); hidx = {h:i for i,h in enumerate(uniq)}
def win(h, lo, hi):  # spend in (lo, hi]
    i = hidx[h]
    d, s = dy[starts[i]:ends[i]], sv[starts[i]:ends[i]]
    m = (d>lo)&(d<=hi)
    return s[m].sum()
def vec(h, t):
    s28=win(h,t-28,t); s84=win(h,t-84,t); s364=win(h,t-364,t)
    k1=s28; k2=win(h,t-56,t-28); k3=win(h,t-84,t-56); k4=win(h,t-112,t-84)
    ew=(k1*8+k2*4+k3*2+k4)/15.
    ya=win(h,t-392,t-364)
    return [np.log1p(s28),np.log1p(s84),np.log1p(s364),np.log1p(k1),np.log1p(k2),np.log1p(k3),np.log1p(k4),
            np.log1p(ew), np.log1p(ya), np.log1p(s84/4.)]

# pooled panel at as-of day 431: pairs (h, t) t in 84..403 step 28
rows=[]
for t in range(84, 404, 28):
    for h in uniq:
        x = vec(h,t); y = win(h,t,t+28)
        rows.append(x+[y])
P = np.array(rows)
print('pooled panel:', P.shape)
Xp, yp = P[:,:-1], P[:,-1]
# evaluate as feature: fit pooled ridge at 431 for all needed hh, then inner eval
base = agent_api.load_saved('e009_demo.parquet')
tt = agent_api.train_targets()
df = base.merge(tt, on=['household_key','snapshot_day'], how='left')
tr_days = agent_api.snapshot_days()['train']
fcols=[c for c in df.columns if c not in ['household_key','snapshot_day','future_spend_4w']]
for inner_day in [403,431]:
    # pooled model as-of inner_day
    rows=[]
    for t in range(84, inner_day-27, 28):
        for h in uniq:
            rows.append(vec(h,t)+[win(h,t,t+28)])
    P=np.array(rows); Xp,yp=P[:,:-1],P[:,-1]
    mu=Xp.mean(0); sd=Xp.std(0); sd[sd==0]=1
    Z=np.c_[np.ones(len(Xp)),np.clip((Xp-mu)/sd,-30,30)]
    A=Z.T@Z+50*np.eye(Z.shape[1]); A[-1,-1]-=50
    w=np.linalg.solve(A,Z.T@yp)
    # predict for households at inner_day
    pred={h: float(np.clip(np.r_[1,np.clip((np.array(vec(h,inner_day))-mu)/sd,-30,30)]@w,0,None)) for h in uniq}
    df['pooled']=df.household_key.map(pred)
    trin=df[np.isin(df.snapshot_day.values,[x for x in tr_days if x<inner_day])]
    inner=df[df.snapshot_day.values==inner_day]
    Xtr=np.c_[trin[fcols].apply(pd.to_numeric,errors='coerce').fillna(0).values, trin[['pooled']].values]
    Xin=np.c_[inner[fcols].apply(pd.to_numeric,errors='coerce').fillna(0).values, inner[['pooled']].values]
    p=ridge_fit_pred(Xtr,trin.future_spend_4w.values,Xin,3000.)
    mae=np.abs(p-inner.future_spend_4w.values).mean()
    corr=np.corrcoef(inner.pooled, inner.future_spend_4w)[0,1]
    print(f'd{inner_day}: innerMAE={mae:.3f} pooled-corr={corr:.3f}')
