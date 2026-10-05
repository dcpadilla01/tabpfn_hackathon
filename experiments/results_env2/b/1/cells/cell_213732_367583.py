
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

e2 = agent_api.load_saved('e002_mix.parquet')
t = agent_api.train_targets()
m = t.merge(e2, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values
vd = m['snapshot_day'].isin(agent_api.snapshot_days()['validation']).values

def ridge_eval(df, target, lam=30.0, seed=0):
    X = df.copy()
    cat = [c for c in X.columns if str(X[c].dtype) in ('object','category')]
    X = pd.get_dummies(X, columns=cat, dummy_na=True)
    X = X.astype(float)
    X = X.replace([np.inf,-np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    mu, sd = X.mean(), X.std().replace(0,1)
    X = ((X-mu)/sd).values
    ly = np.log1p(target)
    A = X[~vd]; ya = ly[~vd]
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ya)
    p = np.expm1(X[vd]@w)
    return np.abs(p - target[vd]).mean()

base_cols = [c for c in e2.columns if c not in ('household_key','snapshot_day')]
print("E002 proxy MAE:", ridge_eval(m[base_cols], y))

# now test candidate additions
def cand(view, sd):
    tx = view.table('transactions'); d = tx['day']; hh = view.households
    f = pd.DataFrame(index=hh)
    for w in (7,14,21,42,112):
        f[f'c_sp{w}'] = tx[d>sd-w].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    f['c_splag1y'] = tx[(d>sd-392)&(d<=sd-364)].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    t2 = tx[d>sd-84].copy(); t2['wk']=t2['day']//7
    ws = t2.groupby(['household_key','wk'])['sales_value'].sum(); g = ws.groupby('household_key')
    f['c_wkstd'] = g.std().reindex(hh, fill_value=0.0)
    bs = tx[d>sd-84].groupby(['household_key','basket_id'])['sales_value'].sum()
    f['c_bsmean'] = bs.groupby('household_key').mean().reindex(hh, fill_value=0.0)
    f['c_bsmax'] = bs.groupby('household_key').max().reindex(hh, fill_value=0.0)
    f['c_nprod168'] = tx[d>sd-168].groupby('household_key')['product_id'].nunique().reindex(hh, fill_value=0)
    f['c_act28'] = tx[d>sd-28].groupby('household_key')['day'].nunique().reindex(hh, fill_value=0)/28.0
    return f

X = agent_api.build_features(cand)
mm = m.merge(X.reset_index(), on=['household_key','snapshot_day'], how='left')
cand_cols = [c for c in X.columns if c!='household_key']
print("E2 + candidates:", ridge_eval(mm[base_cols+cand_cols], y))
for c in cand_cols:
    print(c, ridge_eval(mm[base_cols+[c]], y))
