
import agent_api as A, pandas as pd, numpy as np
e013 = A.load_saved('e013_stock.parquet')
print('e013 shape', e013.shape)
for name in ['stock_v1','rhythm_v1','mix_v1','season_v1']:
    t = A.load_saved(name+'.parquet')
    print(name, t.shape, list(t.columns)[:30])
tt = A.train_targets()
print(tt.groupby('snapshot_day').future_spend_4w.agg(['count','mean','median','std','max']))
print('zero rate overall', (tt.future_spend_4w==0).mean())
print(tt.groupby('snapshot_day').future_spend_4w.apply(lambda s:(s==0).mean()))


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np
e013 = A.load_saved('e013_stock.parquet')
cols = [c for c in e013.columns if c not in ('household_key','snapshot_day')]
print(len(cols))
for c in cols: print(c)


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np
e013 = A.load_saved('e013_stock.parquet')
tt = A.train_targets()
df = e013.merge(tt, on=['household_key','snapshot_day'])
tr = df[df.snapshot_day<=403]; va = df[df.snapshot_day==431]
feats = [c for c in e013.columns if c not in ('household_key','snapshot_day')]
def prep(d):
    X = d[feats].copy()
    for c in X.columns:
        if X[c].dtype==bool: X[c]=X[c].astype(float)
    # simple numeric encoding for categoricals
    X = pd.get_dummies(X.astype(object).where(~X.columns.str.startswith(('classification','homeowner','kid')), X), dummy_na=True) if False else X
    return X
# quick: baseline predictors on 431
yv = va.future_spend_4w.values
print('pred global median  MAE', np.abs(yv-np.median(tr.future_spend_4w)).mean())
print('pred zero           MAE', np.abs(yv-0).mean())
# household last block p1 = spend_28
print('pred spend_28       MAE', np.abs(yv-va.spend_28.values).mean())
print('pred 0.8*spend_28   MAE', np.abs(yv-0.8*va.spend_28.values).mean())
print('pred max(0,spend_28-10) MAE', np.abs(yv-np.maximum(0,va.spend_28.values-10)).mean())
# error decomposition for spend_28 predictor
err = yv-va.spend_28.values
print('by yv quantile:')
q = pd.qcut(yv, 5, duplicates='drop')
print(pd.DataFrame({'y':yv,'p':va.spend_28.values,'e':err}).groupby(q,observed=True).apply(lambda g: pd.Series({'n':len(g),'y_mean':g.y.mean(),'p_mean':g.p.mean(),'mae':np.abs(g.e).mean()})))
print('zero-y rows:', (yv==0).sum(), 'MAE on them', np.abs(err[yv==0]).mean(), 'mean pred on them', va.spend_28.values[yv==0].mean())


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np
e013 = A.load_saved('e013_stock.parquet')
tt = A.train_targets()
df = e013.merge(tt, on=['household_key','snapshot_day'])
# does a household appear at multiple snapshots? how correlated are consecutive blocks?
df = df.sort_values(['household_key','snapshot_day'])
g = df.groupby('household_key').future_spend_4w
df['y_prev'] = g.shift(1)
df['y_next'] = g.shift(-1)
sub = df.dropna(subset=['y_prev','y_next'])
print('n households', df.household_key.nunique(), 'rows', len(df))
print('corr(y, y_prev)', df[['y','y_prev']].corr().iloc[0,1])
print('corr(y, y_next)', df[['y','y_next']].corr().iloc[0,1])
print('corr(y, y_prev+y_next)/2', ((df.y_prev+df.y_next)/2).corr(df.y))
# how much of target variance is household identity? R2 of household-mean predictor
hm = df.groupby('household_key').y.transform('mean')
print('R2 household mean', 1-((df.y-hm)**2).sum()/((df.y-df.y.mean())**2).sum())
# but that uses future info; use only PAST blocks mean as pure past predictor


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np
e013 = A.load_saved('e013_stock.parquet')
tt = A.train_targets()
df = e013.merge(tt, on=['household_key','snapshot_day'])
y = df.future_spend_4w
df = df.sort_values(['household_key','snapshot_day'])
g = df.groupby('household_key').future_spend_4w
df['y_prev'] = g.shift(1); df['y_next'] = g.shift(-1)
print('corr(y, y_prev)', df[['future_spend_4w','y_prev']].corr().iloc[0,1])
print('corr(y, y_next)', df[['future_spend_4w','y_next']].corr().iloc[0,1])
print('corr(y, mean(prev,next))', ((df.y_prev+df.y_next)/2).corr(df.future_spend_4w))
hm = df.groupby('household_key').future_spend_4w.transform('mean')
print('R2 household mean (oracle)', 1-((df.future_spend_4w-hm)**2).sum()/((df.future_spend_4w-df.future_spend_4w.mean())**2).sum())
# per-household std vs between-household std
hs = df.groupby('household_key').future_spend_4w.agg(['mean','std','count'])
print('within std mean', hs['std'].mean(), 'between std', hs['mean'].std())


# ---- cell ----

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


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np
for name in ['hist_v1','hist_v2','structure_v1']:
    t = A.load_saved(name+'.parquet')
    print(name, t.shape)
    print(list(t.columns))
    print()


# ---- cell ----

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


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
print(A.snapshot_days())
v = A.snapshot(431)
print('campaigns cols', v.campaigns.columns.tolist())
print(v.campaigns.head(3))
print('campaign_targets', v.campaign_targets.shape, v.campaign_targets.columns.tolist())
print(v.campaign_targets.head(3))
print('descriptions', v.campaign_targets.description.value_counts())
print('campaign ranges by type:')
print(v.campaigns.groupby('description').agg(n=('campaign','size'), smin=('start_day','min'), smax=('start_day','max'), emin=('end_day','min'), emax=('end_day','max')))


# ---- cell ----

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


# ---- cell ----

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


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e013 = A.load_saved('e013_stock.parquet')
tt = A.train_targets()
df = e013.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in e013.columns if c not in ('household_key','snapshot_day')]
tr_days=[95,123,151,179,207,235,263,291,319,347,375]
ho_days=[403,431]

def ridge_mae(cols, alpha=1000.0, ret_pred=False):
    d2 = df[cols+['household_key','snapshot_day','future_spend_4w']]
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
    pv = Av@w
    m = np.abs(pv-va.future_spend_4w.values).mean()
    return (m,pv,va.future_spend_4w.values) if ret_pred else m

base = ridge_mae(feats); print('base(403+431)', round(base,2))

# two-part interaction block
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
# state-conditional cohort means of spend_28 (per snapshot)
key='snapshot_day'
d['tp_recbucket'] = rb.astype(str)
cm = d.groupby([key,'tp_recbucket']).spend_28.transform('mean')
d['tp_cmean'] = cm
tp_cols = [c for c in d.columns if c.startswith('tp_') and c!='tp_recbucket']
print('tp block size', len(tp_cols))
m_tp = ridge_mae(feats+tp_cols); print('with tp block', round(m_tp,2), 'delta', round(m_tp-base,2))
# tp block alone
m_only = ridge_mae(tp_cols); print('tp alone', round(m_only,2))
# alpha sensitivity with tp
for al in [300,1000,3000]:
    print('alpha',al, round(ridge_mae(feats+tp_cols,alpha=al),2))


# ---- cell ----

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
