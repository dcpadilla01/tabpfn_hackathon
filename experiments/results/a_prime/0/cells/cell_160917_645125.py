
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e013 = A.load_saved('e013_stock.parquet')
tt = A.train_targets()
df = e013.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in e013.columns if c not in ('household_key','snapshot_day')]
tr_days=[95,123,151,179,207,235,263,291,319,347,375,403]

def ridge_mae(cols, alpha=1000.0, extra_tr=None, extra_va=None):
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
    if extra_tr is not None:
        Xtr = pd.concat([Xtr.reset_index(drop=True), extra_tr.reset_index(drop=True)],axis=1)
        Xva = pd.concat([Xva.reset_index(drop=True), extra_va.reset_index(drop=True)],axis=1)
    Xtr,Xva = Xtr.align(Xva, join='left', axis=1, fill_value=0)
    mu,sd = Xtr.mean(), Xtr.std().replace(0,1)
    Ztr=((Xtr-mu)/sd).fillna(0).values; Zva=((Xva-mu)/sd).fillna(0).values
    A_=np.hstack([Ztr,np.ones((len(Ztr),1))]); Av=np.hstack([Zva,np.ones((len(Zva),1))])
    w=np.linalg.solve(A_.T@A_+alpha*np.eye(A_.shape[1]), A_.T@tr.future_spend_4w.values)
    return np.abs(Av@w-va.future_spend_4w.values).mean()

base = ridge_mae(feats); print('base', round(base,2))
# interaction
d = df.copy()
d['ix_s28_tgt'] = d.spend_28 * d.m_tgt_active
tr = d[d.snapshot_day.isin(tr_days)]; va = d[d.snapshot_day==431]
print('ix_s28_tgt', round(ridge_mae(feats, extra_tr=tr[['ix_s28_tgt']], extra_va=va[['ix_s28_tgt']]),2))

# prospectivity map
v = A.snapshot(431)
camp = v.campaigns.copy(); tgt = v.campaign_targets
def prospect(day):
    c = camp[(camp.start_day>day)&(camp.start_day<=day+28)]
    return sorted(c.campaign.tolist())
for day in [95,123,151,179,207,235,263,291,319,347,375,403,431]:
    print(day, prospect(day))
