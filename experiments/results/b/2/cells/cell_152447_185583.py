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
