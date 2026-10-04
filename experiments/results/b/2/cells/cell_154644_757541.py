
import agent_api, pandas as pd, numpy as np

def prep(t):
    return t.set_index(['household_key','snapshot_day'])

def ridge_cv(base, extra=None, use_log=True, alphas=(1,3,10,30,100,300), seed_split=(375,403,431)):
    frames = [prep(base)]
    if extra is not None: frames.append(prep(extra))
    df = pd.concat(frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])
    ycol = agent_api.TARGET
    df = df.join(tt[ycol])
    feat = [c for c in df.columns if c != ycol]
    num = df[feat].select_dtypes(include=[np.number]).columns.tolist()
    X = df[num].astype(float)
    X = X.fillna(X.median())
    y = df[ycol].astype(float)
    tr_days = [d for d in agent_api.snapshot_days()['train'] if d not in seed_split]
    va_days = seed_split
    itr = df.index.get_level_values(1).isin(tr_days); iva = df.index.get_level_values(1).isin(va_days)
    mu, sd = X[itr].mean(), X[itr].std().replace(0,1)
    Xs = (X-mu)/sd
    Xtr = np.c_[np.ones(itr.sum()), Xs[itr].values]; Xva = np.c_[np.ones(iva.sum()), Xs[iva].values]
    out = {}
    for tgt_mode in ([use_log] if use_log is not True else [True, False]):
        yy = np.log1p(y) if tgt_mode else y
        ytr = yy[itr].values; yva = y[iva].values
        best = None
        for a in alphas:
            A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0] -= a
            w = np.linalg.solve(A, Xtr.T@ytr)
            p = np.clip(np.expm1(Xva@w) if tgt_mode else Xva@w, 0, None)
            mae = float(np.abs(p-yva).mean())
            if best is None or mae < best[0]: best = (mae, a)
        out['log' if tgt_mode else 'raw'] = (round(best[0],3), best[1])
    return out

base = agent_api.load_saved('e012_basket_shape.parquet')
print('base only:', ridge_cv(base))
# sanity: tiny base (spend_28 only)
tiny = base[['household_key','snapshot_day','spend_28','spend_84','wk_avg_4','fwd28_mean']]
print('tiny:', ridge_cv(tiny))
