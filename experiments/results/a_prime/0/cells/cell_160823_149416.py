
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e013 = A.load_saved('e013_stock.parquet')
tt = A.train_targets()
df = e013.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in e013.columns if c not in ('household_key','snapshot_day')]
cat_cols = [c for c in feats if df[c].dtype==object]
tr_days=[95,123,151,179,207,235,263,291,319,347,375,403]

def ridge_mae(cols, alpha=1000.0):
    d2 = df[cols+['household_key','snapshot_day','future_spend_4w']]
    Xn = [c for c in cols if d2[c].dtype!=object]
    Xc = [c for c in cols if d2[c].dtype==object]
    def des(d):
        X = d[Xn].astype(float).copy()
        for c in Xc:
            dm = pd.get_dummies(d[c].astype(object), dummy_na=True, prefix=c)
            X = pd.concat([X, dm.astype(float)],axis=1)
        return X
    tr = d2[d2.snapshot_day.isin(tr_days)]; va = d2[d2.snapshot_day==431]
    Xtr,Xva = des(tr), des(va)
    Xtr,Xva = Xtr.align(Xva, join='left', axis=1, fill_value=0)
    mu,sd = Xtr.mean(), Xtr.std().replace(0,1)
    Ztr=((Xtr-mu)/sd).fillna(0).values; Zva=((Xva-mu)/sd).fillna(0).values
    A_=np.hstack([Ztr,np.ones((len(Ztr),1))]); Av=np.hstack([Zva,np.ones((len(Zva),1))])
    w=np.linalg.solve(A_.T@A_+alpha*np.eye(A_.shape[1]), A_.T@tr.future_spend_4w.values)
    return np.abs(Av@w-va.future_spend_4w.values).mean()

blocks = {
 'hist': [c for c in feats if c.startswith(('spend_','baskets_','days_','basket_val','recency','tenure','trend_','active_'))],
 'x': [c for c in feats if c.startswith('x_')],
 'mkt': [c for c in feats if c.startswith('m_')],
 'demo': ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc','d_has_demo'],
 'rhythm': [c for c in feats if c.startswith(('rs_','sl_','dl_'))],
 'rank': [c for c in feats if c.startswith(('rk_','rz_','coh_'))],
 'stock': [c for c in feats if c.startswith('stk_')],
}
for bname, cols in blocks.items():
    print(bname, len(cols), 'single-block MAE', round(ridge_mae(cols),2))
print('ALL', round(ridge_mae(feats),2))
# leave-one-block-out
allb = feats
for bname, cols in blocks.items():
    rest = [c for c in allb if c not in cols]
    print('drop', bname, round(ridge_mae(rest),2))
