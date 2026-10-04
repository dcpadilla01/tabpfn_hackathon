
import agent_api, pandas as pd, numpy as np

def ridge_cv(frames, alphas=(1,3,10,30,100,300,1000), seed_split=(375,403,431)):
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
    Xtr = np.c_[np.ones(itr.sum()), Xs[itr].values]; Xva = np.c_[np.ones(iva.sum()), Xs[iva].values]
    ytr = np.log1p(y[itr]).values; yva = y[iva].values
    res = []
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0] -= a
        w = np.linalg.solve(A, Xtr.T@ytr)
        p = np.clip(np.expm1(Xva@w), 0, None)
        res.append((float(np.abs(p-yva).mean()), a))
    res.sort()
    return round(res[0][0],3), res[0][1], round(res[-1][0],3)

base = agent_api.load_saved('e012_basket_shape.parquet')
cand = agent_api.load_saved('cand_screen1.parquet')
print('base           :', ridge_cv([base]))
print('base + cand    :', ridge_cv([base, cand]))
# ablation: drop one cand feature at a time
ccols = [c for c in cand.columns if c not in ('household_key','snapshot_day')]
full = ridge_cv([base, cand])[0]
print('full:', full)
for c in ccols:
    sub = cand.drop(columns=[c])
    m, a, worst = ridge_cv([base, sub])
    print(f'drop {c:14s}: {m}  (delta {round(m-full,3):+.3f})')
