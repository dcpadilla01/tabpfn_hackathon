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
