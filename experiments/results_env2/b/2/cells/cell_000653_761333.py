import agent_api, numpy as np, pandas as pd

def prep(df):
    df = df.copy()
    if 'household_key' not in df.columns: df = df.reset_index()
    return df.set_index(['household_key','snapshot_day'])

base = prep(agent_api.load_saved('e011_table.parquet'))
tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])['future_spend_4w']
b = base.join(tt.rename('y'), how='inner')
blocks = {n: prep(agent_api.load_saved(n+'.parquet')) for n in ['deal_v1','hazard_v1','display_v1']}
inner_va = [375,403,431]

def fit_eval(X, y, is_va, lam=200, clip=8.0):
    mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
    Xs = ((X-mu)/sd).clip(-clip, clip).fillna(0).values
    w = np.linalg.solve(Xs[~is_va].T@Xs[~is_va] + lam*np.eye(X.shape[1]), Xs[~is_va].T@y[~is_va])
    return float(np.abs(Xs[is_va]@w - y[is_va]).mean())

y = b['y'].values
is_va = np.isin(b.index.get_level_values('snapshot_day'), inner_va)
print('mean-pred MAE:', round(float(np.abs(y[is_va]-y[~is_va].mean()).mean()),2))
print('sanity sc_f_ew_hl2 ridge:', round(fit_eval(b[['sc_f_ew_hl2']].astype(float), y, is_va),2))
base_cols = [c for c in base.columns]
print('base ridge:', round(fit_eval(b[base_cols].astype(float), y, is_va),3))
for name, blk in blocks.items():
    bc = list(blk.columns)
    bb = b.join(blk, how='left')
    print(f'{name}: {fit_eval(bb[base_cols+bc].astype(float), y, is_va):.3f} (+{len(bc)})')
allb = b
for blk in blocks.values(): allb = allb.join(blk, how='left')
allcols = [c for c in allb.columns if c != 'y']
print('ALL:', round(fit_eval(allb[allcols].astype(float), y, is_va),3))
out = allb.reset_index()[['household_key','snapshot_day']+allcols]
p = agent_api.save_table(out, 'e017_union_v1')
print('saved', p, out.shape)
