
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e013 = A.load_saved('e013_stock.parquet')
tt = A.train_targets()
df = e013.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in e013.columns if c not in ('household_key','snapshot_day')]
cat_cols = [c for c in feats if df[c].dtype==object]
num_cols = [c for c in feats if c not in cat_cols]
print('num', len(num_cols), 'cat', len(cat_cols))

def design(d):
    X = d[num_cols].astype(float).copy()
    for c in cat_cols:
        dm = pd.get_dummies(d[c].astype(object), dummy_na=True, prefix=c)
        X = pd.concat([X, dm.astype(float)], axis=1)
    return X

def fit_eval(train_days, val_days, alpha=100.0):
    tr = df[df.snapshot_day.isin(train_days)]; va = df[df.snapshot_day.isin(val_days)]
    Xtr, Xva = design(tr), design(va)
    Xtr, Xva = Xtr.align(Xva, join='left', axis=1, fill_value=0)
    mu, sd = Xtr.mean(), Xtr.std().replace(0,1)
    Ztr = ((Xtr-mu)/sd).fillna(0).values; Zva = ((Xva-mu)/sd).fillna(0).values
    ytr = tr.future_spend_4w.values
    A_ = np.hstack([Ztr, np.ones((len(Ztr),1))])
    Av = np.hstack([Zva, np.ones((len(Zva),1))])
    I = np.eye(A_.shape[1]); I[-1,-1]=0
    w = np.linalg.solve(A_.T@A_ + alpha*np.eye(A_.shape[1]), A_.T@ytr)
    pv = Av@w
    return np.abs(pv-va.future_spend_4w.values).mean(), pv, va.future_spend_4w.values

tr_days=[95,123,151,179,207,235,263,291,319,347,375,403]
for al in [10,30,100,300,1000]:
    mae,_,_ = fit_eval(tr_days,[431],alpha=al)
    print('alpha',al,'MAE 431', round(mae,3))
