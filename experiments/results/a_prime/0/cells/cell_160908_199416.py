
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e013 = A.load_saved('e013_stock.parquet')
tt = A.train_targets()
df = e013.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in e013.columns if c not in ('h'+'ousehold_key','snapshot_day')]
feats = [c for c in e013.columns if c not in ('household_key','snapshot_day')]
tr_days=[95,123,151,179,207,235,263,291,319,347,375,403]

def ridge_mae(cols, alpha=1000.0, tr_df=None, va_df=None):
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

# 1) interaction: spend_28 x m_tgt_active
d = df.copy()
d['ix_s28_tgt'] = d.spend_28 * d.m_tgt_active
print('ix_s28_tgt', round(ridge_mae(feats+['ix_s28_tgt']),2), 'base', round(ridge_mae(feats),2))
# 2) campaign-prospectivity: campaigns overlapping or starting within (day, day+28]
v = A.snapshot(431)
camp = v.campaigns
# for a snapshot at day D, campaigns with start in (D, D+28]
def prospect(day):
    c = camp[(camp.start_day>day)&(camp.start_day<=day+28)]
    return set(c.campaign)
# targets per campaign
tgt = v.campaign_targets
for day in [95,123,151,179,207,235,263,291,319,347,375,403,431]:
    ps = prospect(day)
    print(day, sorted(ps), [ (c, camp[camp.campaign==c].description.iloc[0]) for c in sorted(ps)])
