
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e013 = A.load_saved('e013_stock.parquet')
tt = A.train_targets()
df = e013.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in e013.columns if c not in ('household_key','snapshot_day')]
tr_days=[95,123,151,179,207,235,263,291,319,347,375]; ho_days=[403,431]

def ridge_mae(d2, cols, alpha=1000.0):
    Xn = [c for c in cols if d2[c].dtype!=object]
    Xc = [c for c in cols if d2[c].dtype==object]
    def des(d):
        X = d[Xn].astype(float).copy()
        for c in Xc:
            dm = pd.get_dummies(d[c].astype(object), dummy_na=True, prefix=c)
            X = pd.concat([X, dm.astype(float)],axis=1)
        return X
    tr = d2[d2.snapshot_day.isin(tr_days)]; va = d2[d2.snapshot_day.isin(ho_days)]
    Xtr,Xva = des(tr), des(va)
    Xtr,Xva = Xtr.align(Xva, join='left', axis=1, fill_value=0)
    mu,sd = Xtr.mean(), Xtr.std().replace(0,1)
    Ztr=((Xtr-mu)/sd).fillna(0).values; Zva=((Xva-mu)/sd).fillna(0).values
    A_=np.hstack([Ztr,np.ones((len(Ztr),1))]); Av=np.hstack([Zva,np.ones((len(Zva),1))])
    w=np.linalg.solve(A_.T@A_+alpha*np.eye(A_.shape[1]), A_.T@tr.future_spend_4w.values)
    return np.abs(Av@w-va.future_spend_4w.values).mean()

d = df.copy()
rb = pd.cut(d.recency, [-1,7,14,28,56,10000], labels=['r0','r1','r2','r3','r4'])
for lab in ['r0','r1','r2','r3','r4']:
    dm = (rb==lab).astype(float)
    d['tp_s28_'+lab] = d.spend_28*dm
    d['tp_s84_'+lab] = d.spend_84*dm
d['tp_s28_act'] = d.spend_28*d.active_28
d['tp_s28_in']  = d.spend_28*(1-d.active_28)
d['tp_no28'] = (d.baskets_28==0).astype(float)
d['tp_no28_x_s84'] = (d.baskets_28==0).astype(float)*d.spend_84
d['tp_log_s28'] = np.log1p(d.spend_28)
d['tp_sqrt_s28'] = np.sqrt(d.spend_28.clip(lower=0))
d['tp_s28_x_rk'] = d.spend_28*d.rk_s28_pct
d['tp_recbucket'] = rb.astype(str)
cm = d.groupby(['snapshot_day','tp_recbucket']).spend_28.transform('mean')
d['tp_cmean'] = cm
tp_cols = [c for c in d.columns if c.startswith('tp_') and c!='tp_recbucket']
print('base', round(ridge_mae(d, feats),2))
print('base+tp', round(ridge_mae(d, feats+tp_cols),2))
print('tp alone', round(ridge_mae(d, tp_cols),2))
for al in [300,1000,3000]:
    print('alpha',al, round(ridge_mae(d, feats+tp_cols, alpha=al),2))
# reduced: only recency interactions + log/sqrt
red = ['tp_s28_'+l for l in ['r0','r1','r2','r3','r4']] + ['tp_s84_'+l for l in ['r0','r1','r2','r3','r4']] + ['tp_log_s28','tp_sqrt_s28','tp_no28']
print('base+tp_reduced', round(ridge_mae(d, feats+red),2))
