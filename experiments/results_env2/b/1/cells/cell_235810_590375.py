
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
m2 = m.copy(); m2['snap_idx'] = (sd - 95)/28.0
print('base:', run(m2, feats))
print('+snap_idx:', run(m2, feats+['snap_idx']))
# drift check: mean y and mean sp28 by snapshot
g = m.groupby('snapshot_day').agg(ymean=('future_spend_4w','mean'), sp28mean=('sp28','mean'))
g['bias'] = g.ymean - g.sp28mean
print(g.round(1))
# display_mailer structure
v = A.snapshot(459)
dm = v.table('display_mailer')
print(dm.dtypes); print(dm.head(3)); print('weeks', dm.week_no.min(), dm.week_no.max(), 'n', len(dm))
print('display vals', dm.display.value_counts(dropna=False).head())
print('mailer vals', dm.mailer.value_counts(dropna=False).head())
