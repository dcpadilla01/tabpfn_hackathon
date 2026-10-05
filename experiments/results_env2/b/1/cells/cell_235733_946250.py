
import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
e8 = A.load_saved('e008_level_shape.parquet')
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float)
sd = m['snapshot_day'].values
feats=[c for c in e8.columns if c not in('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(e8[c])]
def prep(df, cols):
    X=df[cols].astype(float).copy(); X=X.fillna(X.median())
    mu=X.mean(0); sg=X.std(0)+1e-9
    return (X.values-mu.values)/sg.values
def run(df, cols, lams=(30,100,300)):
    Xz = prep(df, cols); tr=sd<=347; ev=sd>=375; out={}
    for lam in lams:
        Xt=Xz[tr]; yt=y[tr]; ym=yt.mean()
        w=np.linalg.solve(Xt.T@Xt+lam*np.eye(len(cols)), Xt.T@(yt-ym))
        out[lam]=round(float(np.mean(np.abs(Xz[ev]@w+ym-y[ev]))),2)
    return out
m2 = m.copy()
m2['i_sp28_trend'] = m2['sp28'].fillna(0)*m2['trend_84'].fillna(0)
m2['i_sp28_nact'] = m2['sp28'].fillna(0)*m2['nact28'].fillna(0)
m2['i_sp28_demo4'] = m2['sp28'].fillna(0)*(m2['d_classification_4'].astype(str)=='5+').astype(float)
m2['i_sp28_kid'] = m2['sp28'].fillna(0)*m2['d_kid_category_desc'].astype(str).isin(['1','2','3+']).astype(float)
m2['i_newma_rec'] = m2['newma28'].fillna(0)*m2['days_since_last'].fillna(0)
m2['sq_sp28'] = m2['sp28'].fillna(0)**2
m2['sp28_x_sp84rate'] = m2['sp28'].fillna(0)*m2['sp84_rate'].fillna(0)
print('+interact:', run(m2, feats+['i_sp28_trend','i_sp28_nact','i_sp28_demo4','i_sp28_kid','i_newma_rec','sq_sp28','sp28_x_sp84rate']))
# also test: dropping weak/duplicate columns (feature pruning) in proxy
bot = ['discshare364','dsh364_GROCE','evshare84','sp_lag1y','splag1y','z_n_windows_hist','tenure','z_tenure_days','topstore','week_mod52','d_classification_2','d_homeowner_desc','topstore']
print('pruned:', run(m2, [c for c in feats if c not in bot]))
