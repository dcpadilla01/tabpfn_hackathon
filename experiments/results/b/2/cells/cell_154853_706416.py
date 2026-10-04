
import agent_api, pandas as pd, numpy as np

def ridge_cv_raw(frames, alphas=(300,1000,3000,10000), seed_split=(375,403,431)):
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
    ytr = y[itr].values; yva = y[iva].values
    res = []
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0] -= a
        w = np.linalg.solve(A, Xtr.T@ytr)
        p = np.clip(Xva@w, 0, None)
        res.append((float(np.abs(p-yva).mean()), a))
    res.sort()
    return res[0]

base = agent_api.load_saved('e012_basket_shape.parquet')
cand = agent_api.load_saved('cand_screen1.parquet')
b = ridge_cv_raw([base]); print('base raw:', b)
bc = ridge_cv_raw([base, cand]); print('base+cand raw:', bc)
full = bc[0]
ccols = [c for c in cand.columns if c not in ('household_key','snapshot_day')]
for c in ccols:
    m, a = ridge_cv_raw([base, cand.drop(columns=[c])])
    print(f'drop {c:14s}: {m:.3f}  (delta {m-full:+.3f})')
# cand alone
print('cand alone raw:', ridge_cv_raw([cand])[0])
