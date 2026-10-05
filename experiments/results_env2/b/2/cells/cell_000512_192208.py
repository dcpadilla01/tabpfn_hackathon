import agent_api, numpy as np, pandas as pd

def prep(df):
    df = df.copy()
    if 'household_key' not in df.columns: df = df.reset_index()
    return df.set_index(['household_key','snapshot_day'])

base = prep(agent_api.load_saved('e011_table.parquet'))
tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])['future_spend_4w']
b = base.join(tt.rename('y'), how='inner')
print('b', b.shape, 'y nan', int(b['y'].isna().sum()))
print(b['y'].describe().round(1))
inner_va = [375,403,431]
is_va = b.index.get_level_values('snapshot_day').isin(inner_va)
print('n va', is_va.sum(), 'n tr', (~is_va).sum())
y = b['y'].values
print('mean-y train', y[~is_va].mean().round(1), 'val', y[is_va].mean().round(1))
print('MAE mean-pred:', round(float(np.abs(y[is_va]-y[~is_va].mean()).mean()),2))
for c in ['spend_84d','ew_spend_hl28','sc_f_ew_hl2','spend_28d']:
    mae = float(np.abs(b.loc[is_va, c].values - y[is_va]).mean())
    print('MAE', c, round(mae,2))
# ridge with fewer features, check lam
cols = [c for c in base.columns]
X = b[cols].astype(float)
mu, sd = X[~is_va].mean(), X[~is_va].std().replace(0,1)
Xs = ((X-mu)/sd).fillna(0).values
for lam in [1,10,50,200,1000]:
    w = np.linalg.solve(Xs[~is_va].T@Xs[~is_va] + lam*np.eye(len(cols)), Xs[~is_va].T@y[~is_va])
    print('lam',lam,'inner MAE', round(float(np.abs(Xs[is_va]@w - y[is_va]).mean()),3))
