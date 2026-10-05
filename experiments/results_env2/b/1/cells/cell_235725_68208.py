
import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
e8 = A.load_saved('e008_level_shape.parquet')
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float)
hk = m['household_key'].values; sd = m['snapshot_day'].values
v = A.snapshot(459)
txg = v.table('transactions').groupby(['household_key','day'])['sales_value'].sum().reset_index()
ser = {h: g.set_index('day').sales_value for h,g in txg.groupby('household_key')}
def lag_fast(h,s,back):
    ss = ser.get(h)
    if ss is None: return 0.0
    idx = ss.index; lo, hi = s+1-back, s+28-back
    sel = idx[(idx>=lo)&(idx<=hi)]
    return float(ss.loc[sel].sum()) if len(sel) else 0.0
lag13 = np.array([lag_fast(h,s,336) for h,s in zip(hk,sd)])
lag1y = np.array([lag_fast(h,s,364) for h,s in zip(hk,sd)])
lag15 = np.array([lag_fast(h,s,392) for h,s in zip(hk,sd)])
feats=[c for c in e8.columns if c not in('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(e8[c])]
def prep(df, cols):
    X=df[cols].astype(float).copy()
    X=X.fillna(X.median())
    mu=X.mean(0); sg=X.std(0)+1e-9
    return (X.values-mu.values)/sg.values
def run(df, cols, lams=(30,100,300)):
    Xz = prep(df, cols)
    tr=sd<=347; ev=sd>=375
    out={}
    for lam in lams:
        Xt=Xz[tr]; yt=y[tr]; ym=yt.mean()
        w=np.linalg.solve(Xt.T@Xt+lam*np.eye(len(cols)), Xt.T@(yt-ym))
        out[lam]=round(float(np.mean(np.abs(Xz[ev]@w+ym-y[ev]))),2)
    return out
print('base E008:', run(m, feats))
m2 = m.copy()
m2['slag13']=lag13; m2['slag1y']=lag1y; m2['slag15']=lag15
m2['slag13_r'] = lag13/(m2['sp84'].fillna(0)+1)
print('+seasonal:', run(m2, feats+['slag13','slag1y','slag15','slag13_r']))
m2['i_sp28_trend'] = m2['sp28'].fillna(0)*m2['trend_84'].fillna(0)
m2['i_sp28_nact'] = m2['sp28'].fillna(0)*m2['nact28'].fillna(0)
m2['i_sp28_demo4'] = m2['sp28'].fillna(0)*(m2['d_classification_4'].fillna('0')=='5+').astype(float)
m2['i_sp28_kid'] = m2['sp28'].fillna(0)*(m2['d_kid_category_desc'].fillna('None/Unknown').isin(['1','2','3+'])).astype(float)
m2['i_newma_rec'] = m2['newma28'].fillna(0)*m2['days_since_last'].fillna(0)
m2['sq_sp28'] = m2['sp28'].fillna(0)**2
m2['sp28_x_sp84rate'] = m2['sp28'].fillna(0)*m2['sp84_rate'].fillna(0)
print('+interact:', run(m2, feats+['i_sp28_trend','i_sp28_nact','i_sp28_demo4','i_sp28_kid','i_newma_rec','sq_sp28','sp28_x_sp84rate']))
print('+both:', run(m2, feats+['slag13','slag1y','slag15','slag13_r','i_sp28_trend','i_sp28_nact','i_sp28_demo4','i_sp28_kid','i_newma_rec','sq_sp28','sp28_x_sp84rate']))
