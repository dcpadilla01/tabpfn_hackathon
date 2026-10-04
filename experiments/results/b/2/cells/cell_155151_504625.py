
import agent_api, pandas as pd, numpy as np

def ridge_cv(frames, alphas=(300,1000,3000,10000), seed_split=(375,403,431)):
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
    best = None
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0] -= a
        w = np.linalg.solve(A, Xtr.T@ytr)
        p = np.clip(Xva@w, 0, None)
        m = float(np.abs(p-yva).mean())
        if best is None or m < best[0]: best = (m, a)
    return best[0]

base = agent_api.load_saved('e012_basket_shape.parquet')
b = base.set_index(['household_key','snapshot_day'])
print('base:', round(ridge_cv([base]),3))

# Block A: zero / tail targeting
A = pd.DataFrame(index=b.index)
rec = b['recency']; gm = b['gap_mean_84']; zf = b['zero_frac_13']; s28 = b['spend_28']; w8a = b['wk_avg_8']
A['z_rec45'] = (rec > 45).astype(float)
A['z_rec30'] = (rec > 30).astype(float)
A['z_gap30'] = (gm > 30).astype(float)
A['z_zf50']  = (zf > 0.5).astype(float)
A['z_rec45_s28'] = A['z_rec45']*s28
A['z_rec45_w8']  = A['z_rec45']*w8a
A['t_maxwk'] = b[['w3','w4','w5','w6','w7','w8']].max(axis=1)
A['t_p90wk'] = b[['w3','w4','w5','w6','w7','w8']].quantile(0.9, axis=1)
A['t_top2']  = (b['w7']+b['w8'])/np.maximum(b[['w3','w4','w5','w6','w7','w8']].sum(axis=1),1e-6)
A['t_s28_inc'] = s28*(1+b['income_ord'].fillna(0)*0)  # placeholder skip
Af = A.reset_index()
print('base+A:', round(ridge_cv([base,Af]),3))
# Block C: tenure-gated / shrunk rates
C = pd.DataFrame(index=b.index)
ten = b['tenure'].clip(lower=0)
C['c_gate28'] = s28*ten/(ten+50)
C['c_gate84'] = b['spend_84']*ten/(ten+100)
C['c_shr28'] = (s28*ten + 130*50)/(ten+50)   # blend with global prior ~130/4wk
C['c_shr84'] = (b['spend_84']*ten + 130*200)/(ten+200)
C['c_lograte'] = np.log1p(b['spend_rate_life'])
Cf = C.reset_index()
print('base+C:', round(ridge_cv([base,Cf]),3))
print('base+A+C:', round(ridge_cv([base,Af,Cf]),3))
# ablations
full = ridge_cv([base,Af,Cf])
for c in list(A.columns)+list(C.columns):
    sub = pd.concat([Af,Cf],axis=1).drop(columns=[c])
    m = ridge_cv([base,sub])
    print(f'drop {c:12s}: {m:.3f} ({m-full:+.3f})')
