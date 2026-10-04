
import agent_api, pandas as pd, numpy as np

def ridge_cv(frames, mode='raw', alphas=(100,300,1000,3000,10000), seed_split=(375,403,431)):
    df = pd.concat([f.set_index(['household_key','snapshot_day']) for f in frames], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])
    ycol = agent_api.TARGET
    df = df.join(tt[ycol])
    num = df.drop(columns=[ycol]).select_dtypes(include=[np.number]).columns.tolist()
    X = df[num].astype(float); X = X.fillna(X.median())
    y = df[ycol].astype(float)
    tr_days = [d for d in agent_api.snapshot_days()['train'] if d not in seed_split]
    itr = df.index.get_level_values(1).isin(tr_days); iva = df.index.get_level_values(1).isin(seed_split)
    mu, sd = X[itr].mean(), X[itr].std().replace(0,1)
    Xs = (X-mu)/sd
    Xtr = np.c_[np.ones(int(itr.sum())), Xs[itr].values]; Xva = np.c_[np.ones(int(iva.sum())), Xs[iva].values]
    ytr = (np.log1p(y[itr]) if mode=='log' else y[itr]).values; yva = y[iva].values
    best = None
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0] -= a
        w = np.linalg.solve(A, Xtr.T@ytr)
        p = np.clip(np.expm1(Xva@w) if mode=='log' else Xva@w, 0, None)
        m = float(np.abs(p-yva).mean())
        if best is None or m < best[0]: best = (m, a)
    return best[0]

base = agent_api.load_saved('e012_basket_shape.parquet')
b = base.set_index(['household_key','snapshot_day'])
print('base raw:', round(ridge_cv([base],'raw'),3), ' log:', round(ridge_cv([base],'log'),3))

# nonlinear block computed offline
nb = pd.DataFrame(index=b.index)
s28 = b['spend_28']; s84 = b['spend_84']; tr = b['trips_28']; rec = b['recency']
fwm = b['fwd28_mean']; wk4 = b['wk_avg_4']
nb['n_sq28'] = (s28/100)**2
nb['n_sqrt28'] = np.sqrt(np.clip(s28,0,None))
nb['n_rank28'] = s28.groupby(level=1).rank(pct=True)
nb['n_rank84'] = s84.groupby(level=1).rank(pct=True)
nb['n_rankfwm'] = fwm.groupby(level=1).rank(pct=True)
nb['n_int_st'] = s28*tr/100
nb['n_int_sr'] = s28*np.clip(rec,0,84)/100
nb['n_int_s84t'] = s84*tr/100
nb['n_log_sq'] = np.log1p(s28)**2
nb['n_ranktr'] = tr.groupby(level=1).rank(pct=True)
nbf = nb.reset_index()
print('base+nonlin raw:', round(ridge_cv([base,nbf],'raw'),3), ' log:', round(ridge_cv([base,nbf],'log'),3))
# ablation within nonlinear block
fullr = ridge_cv([base,nbf],'raw'); fulll = ridge_cv([base,nbf],'log')
for c in nb.columns:
    sub = nbf.drop(columns=[c])
    print(f"drop {c:12s} raw {ridge_cv([base,sub],'raw'):.3f} ({ridge_cv([base,sub],'raw')-fullr:+.3f})  log {ridge_cv([base,sub],'log'):.3f} ({ridge_cv([base,sub],'log')-fulll:+.3f})")
# nonlin alone on top of nothing? and smaller variants
print('nonlin alone raw:', round(ridge_cv([nbf],'raw'),3))
