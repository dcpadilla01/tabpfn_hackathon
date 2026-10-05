import agent_api, numpy as np, pandas as pd

base = agent_api.load_saved('e011_table.parquet')
tt = agent_api.train_targets()
blocks = {n: agent_api.load_saved(n+'.parquet') for n in ['deal_v1','hazard_v1','display_v1']}

def prep(df):
    return df.set_index(['household_key','snapshot_day'])

b = prep(base).join(prep(tt.set_index(['household_key','snapshot_day'])), how='inner')
inner_va = [375,403,431]
inner_tr = [95,123,151,179,207,235,263,291,319,347]

def eval_cols(feat_cols):
    X = b[feat_cols].astype(float)
    y = b['future_spend_4w'].values
    is_va = b.index.get_level_values('snapshot_day').isin(inner_va)
    mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
    Xs = ((X-mu)/sd).fillna(0).values
    Xtr, ytr = Xs[~is_va], y[~is_va]
    Xva, yva = Xs[is_va], y[is_va]
    w = np.linalg.solve(Xtr.T@Xtr + 50*np.eye(Xtr.shape[1]), Xtr.T@ytr)
    return float(np.abs(Xva@w - yva).mean())

base_cols = [c for c in base.columns if c not in ('household_key','snapshot_day')]
print('base MAE:', round(eval_cols(base_cols),3), len(base_cols))
for name, blk in blocks.items():
    bc = [c for c in blk.columns if c not in ('household_key','snapshot_day')]
    bb = b.join(prep(blk), how='left')
    X = bb[base_cols+bc].astype(float)
    y = bb['future_spend_4w'].values
    is_va = bb.index.get_level_values('snapshot_day').isin(inner_va)
    mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
    Xs = ((X-mu)/sd).fillna(0).values
    w = np.linalg.solve(Xs[~is_va].T@Xs[~is_va] + 50*np.eye(X.shape[1]), Xs[~is_va].T@y[~is_va])
    mae = float(np.abs(Xs[is_va]@w - y[is_va]).mean())
    print(f'{name}: MAE {mae:.3f} (+{len(bc)} cols)')

allb = b
for blk in blocks.values():
    allb = allb.join(prep(blk), how='left')
allcols = [c for c in allb.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = allb[allcols].astype(float)
y = allb['future_spend_4w'].values
is_va = allb.index.get_level_values('snapshot_day').isin(inner_va)
mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
Xs = ((X-mu)/sd).fillna(0).values
w = np.linalg.solve(Xs[~is_va].T@Xs[~is_va] + 50*np.eye(X.shape[1]), Xs[~is_va].T@y[~is_va])
print('ALL blocks:', round(float(np.abs(Xs[is_va]@w - y[is_va]).mean()),3), len(allcols))
