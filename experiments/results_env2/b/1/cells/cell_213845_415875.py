
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e2 = agent_api.load_saved('e002_mix.parquet')
t = agent_api.train_targets()
m = e2.merge(t, on=['household_key','snapshot_day'], how='inner')
y = m['future_spend_4w'].values
vd = (m['snapshot_day']==431).values

def prep(df):
    X = df.copy()
    cat = [c for c in X.columns if str(X[c].dtype) in ('object','category')]
    X = pd.get_dummies(X, columns=cat, dummy_na=True).astype(float)
    X = X.replace([np.inf,-np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return ((X - X.mean()) / X.std().replace(0,1)).values

def ridge_eval(Xdf, lam=30.0):
    Xz = prep(Xdf)
    A = Xz[~vd]; ya = np.log1p(y[~vd])
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ya)
    p = np.clip(np.expm1(Xz[vd]@w), 0, None)
    return np.abs(p - y[vd]).mean()

base_cols = [c for c in e2.columns if c not in ('household_key','snapshot_day')]

def cand(view, sd):
    tx = view.table('transactions'); d = tx['day']; hh = view.households
    f = pd.DataFrame(index=hh)
    for w in (7,14,21,42,112):
        f[f'c_sp{w}'] = tx[d>sd-w].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    f['c_splag1y'] = tx[(d>sd-392)&(d<=sd-364)].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    t2 = tx[d>sd-84].copy(); t2['wk']=t2['day']//7
    ws = t2.groupby(['household_key','wk'])['sales_value'].sum(); g = ws.groupby('household_key')
    f['c_wkstd'] = g.std().reindex(hh, fill_value=0.0)
    f['c_wkmean'] = g.mean().reindex(hh, fill_value=0.0)
    bs = tx[d>sd-84].groupby(['household_key','basket_id'])['sales_value'].sum()
    f['c_bsmean'] = bs.groupby('household_key').mean().reindex(hh, fill_value=0.0)
    f['c_bsmax'] = bs.groupby('household_key').max().reindex(hh, fill_value=0.0)
    f['c_nprod168'] = tx[d>sd-168].groupby('household_key')['product_id'].nunique().reindex(hh, fill_value=0)
    f['c_act28'] = tx[d>sd-28].groupby('household_key')['day'].nunique().reindex(hh, fill_value=0)/28.0
    cr = view.table('coupon_redemptions')
    f['c_nred84'] = cr[cr['day']>sd-84].groupby('household_key').size().reindex(hh, fill_value=0)
    return f

X = agent_api.build_features(cand)
mm = m.merge(X.reset_index(), on=['household_key','snapshot_day'], how='left')
cand_cols = [c for c in X.columns if c!='household_key']
print("E2+cands:", ridge_eval(mm[base_cols+cand_cols]))
for c in cand_cols:
    print(c, round(ridge_eval(mm[base_cols+[c]]),2))
