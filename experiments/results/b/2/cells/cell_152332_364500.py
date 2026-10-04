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
