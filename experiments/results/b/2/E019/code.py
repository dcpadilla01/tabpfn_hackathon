import agent_api, pandas as pd, numpy as np
tab = agent_api.load_saved('e018_hinge_prune.parquet')
print('shape:', tab.shape)
print(tab.dtypes.value_counts())
obj = [c for c in tab.columns if tab[c].dtype == object]
print('obj cols:', obj)
print('columns:', list(tab.columns))
tt = agent_api.train_targets()
print('targets:', tt.shape)
print(tt.head(3))
print(agent_api.snapshot_days())
print('rows per snapshot:')
print(tab.groupby('snapshot_day').size())

# ---- cell ----
import agent_api, pandas as pd, numpy as np

v = agent_api.snapshot()
print('transactions', v.transactions.shape)
print('coupon_redemptions', v.coupon_redemptions.shape)
print('display_mailer', v.display_mailer.shape)
print('campaign_targets', v.campaign_targets.shape)
print('coupons', v.coupons.shape)
print('campaigns', v.campaigns.shape)
print('products', v.products.shape)
print('demographics', v.demographics.shape)
print('households:', type(v.households), v.households[:3] if not hasattr(v.households,'shape') else v.households.shape)
dm = v.display_mailer
print('display vals:', dm.display.value_counts().head(8).to_dict())
print('mailer vals:', dm.mailer.value_counts().head(8).to_dict())
print(v.coupon_redemptions.head(3))

TRAIN_DAYS=[95,123,151,179,207,235,263,291,319,347,375,403,431]
tt = agent_api.train_targets()
SPLITS=[('S1',[95,123,151,179,207,235,263,291,319,347],[375,403,431]),
        ('S2',[95,123,151,179,207,235,263,291,319],[347,375,403,431]),
        ('S3',[95,123,151,179,207,235,263,291,319,347,375,403],[431])]

def ridge_fit_pred(Xtr,ytr,Xva,alpha):
    # add intercept
    A=np.hstack([Xtr,np.ones((len(Xtr),1))])
    B=np.hstack([Xva,np.ones((len(Xva),1))])
    d=A.shape[1]
    G=A.T@A
    lam=alpha*np.eye(d); lam[-1,-1]=0.0
    w=np.linalg.solve(G+lam, A.T@ytr)
    return B@w

def local_eval(tab, alphas=(10.0,), label=''):
    feat=[c for c in tab.columns if c not in ('household_key','snapshot_day')]
    t2=tab.merge(tt,on=['household_key','snapshot_day'],how='left')
    rows=[]
    for sname,trd,vad in SPLITS:
        tr=t2[t2.snapshot_day.isin(trd)]; va=t2[t2.snapshot_day.isin(vad)]
        Xtr=tr[feat].astype(float); Xva=va[feat].astype(float)
        med=Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
        mu=Xtr.mean(); sd=Xtr.std().replace(0,1.0)
        Xtr=((Xtr-mu)/sd).values; Xva=((Xva-mu)/sd).values
        ytr=tr.future_spend_4w.values; yva=va.future_spend_4w.values
        r={'split':sname,'n_va':len(va)}
        for a in alphas:
            p=ridge_fit_pred(Xtr,ytr,Xva,a)
            r[f'r{a:g}']=float(np.abs(p-yva).mean())
        rows.append(r)
    df=pd.DataFrame(rows)
    print(label); print(df.round(3).to_string(index=False))
    print('MEAN:',{k:round(x,3) for k,x in df.mean(numeric_only=True).items()})
    return df

e18=agent_api.load_saved('e018_hinge_prune.parquet')
local_eval(e18,alphas=(5.0,10.0,20.0,50.0),label='E018 (official val 60.746)')

# ---- cell ----
import agent_api, pandas as pd, numpy as np

tt = agent_api.train_targets()
SPLITS=[('S1',[95,123,151,179,207,235,263,291,319,347],[375,403,431]),
        ('S2',[95,123,151,179,207,235,263,291,319],[347,375,403,431]),
        ('S3',[95,123,151,179,207,235,263,291,319,347,375,403],[431])]

def ridge_fit_pred(Xtr,ytr,Xva,alpha):
    A=np.hstack([Xtr,np.ones((len(Xtr),1))])
    B=np.hstack([Xva,np.ones((len(Xva),1))])
    d=A.shape[1]
    G=A.T@A
    lam=alpha*np.eye(d); lam[-1,-1]=0.0
    w=np.linalg.solve(G+lam, A.T@ytr)
    return B@w

def local_eval(tab, alphas=(10.0,), label=''):
    feat=[c for c in tab.columns if c not in ('household_key','snapshot_day')]
    t2=tab.merge(tt,on=['household_key','snapshot_day'],how='left')
    rows=[]
    for sname,trd,vad in SPLITS:
        tr=t2[t2.snapshot_day.isin(trd)]; va=t2[t2.snapshot_day.isin(vad)]
        Xtr=tr[feat].astype(float); Xva=va[feat].astype(float)
        med=Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
        mu=Xtr.mean(); sd=Xtr.std().replace(0,1.0)
        Xtr=((Xtr-mu)/sd).values; Xva=((Xva-mu)/sd).values
        ytr=tr.future_spend_4w.values; yva=va.future_spend_4w.values
        r={'split':sname,'n_va':len(va)}
        for a in alphas:
            p=ridge_fit_pred(Xtr,ytr,Xva,a)
            r[f'r{a:g}']=float(np.abs(p-yva).mean())
        rows.append(r)
    df=pd.DataFrame(rows)
    print(label); print(df.round(3).to_string(index=False))
    print('MEAN:',{k:round(x,3) for k,x in df.mean(numeric_only=True).items()})
    return df

e18=agent_api.load_saved('e018_hinge_prune.parquet')
local_eval(e18,alphas=(5.0,10.0,20.0,50.0),label='E018 (official val 60.746)')

# ---- cell ----
import agent_api, pandas as pd, numpy as np

def probe(view, day):
    hh = view.households
    tx = view.transactions
    g = tx.groupby('household_key').sales_value.sum()
    return pd.DataFrame({'spend_probe': g.reindex(hh).values}, index=hh)

tab = agent_api.build_features(probe)
print('build_features shape', tab.shape)

v = agent_api.snapshot()
dm = v.display_mailer
print(dm.display.value_counts(dropna=False).to_dict())
print(dm.mailer.value_counts(dropna=False).to_dict())
print(dm.head(3))
print('weeks range:', dm.week_no.min(), dm.week_no.max())
tx=v.transactions
print('tx days:', tx.day.min(), tx.day.max())
print('coupon_redemptions:'); print(v.coupon_redemptions.head(3))
print('campaign_targets:'); print(v.campaign_targets.head(3))
print('campaigns:'); print(v.campaigns)

# ---- cell ----
import agent_api, pandas as pd, numpy as np

tt = agent_api.train_targets()
SPLITS=[('S1',[95,123,151,179,207,235,263,291,319,347],[375,403,431]),
        ('S3',[95,123,151,179,207,235,263,291,319,347,375,403],[431])]

def ridge_eval(Xtr,ytr,Xva,alpha):
    A=np.hstack([Xtr,np.ones((len(Xtr),1))]); B=np.hstack([Xva,np.ones((len(Xva),1))])
    d=A.shape[1]; G=A.T@A; lam=alpha*np.eye(d); lam[-1,-1]=0.0
    w=np.linalg.solve(G+lam, A.T@ytr); return B@w

def local_score(cols, alphas=(50.0,)):
    t2=e18.merge(tt,on=['household_key','snapshot_day'],how='left')
    out={}
    for sname,trd,vad in SPLITS:
        tr=t2[t2.snapshot_day.isin(trd)]; va=t2[t2.snapshot_day.isin(vad)]
        Xtr=tr[cols].astype(float); Xva=va[cols].astype(float)
        med=Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
        mu=Xtr.mean(); sd=Xtr.std().replace(0,1.0)
        Xtr=((Xtr-mu)/sd).values; Xva=((Xva-mu)/sd).values
        ytr=tr.future_spend_4w.values; yva=va.future_spend_4w.values
        for a in alphas:
            p=ridge_eval(Xtr,ytr,Xva,a)
            out[(sname,a)]=float(np.abs(p-yva).mean())
    return out

e18=agent_api.load_saved('e018_hinge_prune.parquet')
allcols=[c for c in e18.columns if c not in ('household_key','snapshot_day')]
base=local_score(allcols,alphas=(20.0,50.0))
print('BASE:',{f'{k[0]}@a{k[1]}':round(v,3) for k,v in base.items()})

groups={
 'hg_hinges':[c for c in allcols if c.startswith('hg_')],
 'mkt':['mkt28','mkt_mom','drift28','shr'],
 'n_rank':[c for c in allcols if c.startswith('n_')],
 'c_dev':[c for c in allcols if c.startswith(('c_dev','c_cv','c_burst','c_gap','z_rec','z_gap','z_zf','t_maxwk','t_p90wk','t_top2','c_gate','c_shr','c_log'))],
 'last_basket':['last_bval','last_blines','last_bqty','log_last_bval','last_ratio','bmean84','bstd84','bmax84','bmax28','bcv84','unit_price28','stockup28','weekend_share84','evening_share84','morning_share84'],
 'ix_demo_int':[c for c in allcols if c.startswith('ix_')],
 'calendar':['week_of_year','sin1','cos1','sin2','cos2','month_idx'],
 'demo':[c for c in allcols if c.startswith(('age_ord','income_ord','size_ord','grp_ord','kid_ord','c2_','ho_','kid_','has_demo'))],
 'fwd_profile':[c for c in allcols if c.startswith(('fwd28','log_fwd','ewma'))],
 'wk_temporal':[c for c in allcols if c.startswith(('wk','w3','w4','w5','w6','w7','w8','zero_frac','zero_streak'))],
 'longrun':['spend_same_ly','ratio_seas','spend_168','spend_252','spend_504','tenure','recency','spend_life','trips_life','spend_rate_life','trend','ratio_lag','wk_avg_84'],
 'recent_windows':['spend_7','spend_14','spend_21','spend_28','spend_56','spend_84','spend_112','spend_364','trips_84','active_days_84','trips_168','active_days_168','gap_prev','gap_mean_84','spend_lag1','spend_lag2','trips_28','trips_56','basket_avg_28','active_days_28'],
}
res=[]
for gname,cols in groups.items():
    keep=[c for c in allcols if c not in cols]
    sc=local_score(keep,alphas=(20.0,50.0))
    d20=sc[('S1',20.0)]-base[('S1',20.0)]; d50=sc[('S3',50.0)]-base[('S3',50.0)]
    res.append((gname,len(cols),round(sc[('S1',20.0)],3),round(d20,3),round(sc[('S3',50.0)],3),round(d50,3)))
print('group | n | S1a20 | dS1 | S3a50 | dS3  (positive delta = dropping HELPS)')
for r in sorted(res,key=lambda x:-(x[3]+x[5])):
    print(r)

# ---- cell ----
import agent_api, pandas as pd, numpy as np

# Target stats
tt = agent_api.train_targets()
y = tt.future_spend_4w
print('target: mean %.1f median %.1f p90 %.1f zero-frac %.3f' % (y.mean(), y.median(), y.quantile(.9), (y==0).mean()))

def feat_fn(view, day):
    hh = view.households
    tx = view.transactions
    week = (day + 8) // 7
    dm = view.display_mailer
    dmw = dm[(dm.week_no > week - 13) & (dm.week_no <= week)]
    dmw = dmw.assign(disp=(dmw.display != '0').astype(np.float32),
                     mail=(dmw.mailer != '0').astype(np.float32))
    pw = dmw.groupby(['product_id','week_no'], as_index=False)[['disp','mail']].mean()
    t84 = tx[tx.day > day - 84][['household_key','product_id','day','sales_value']].copy()
    t84['week_no'] = (t84.day + 8) // 7
    m = t84.merge(pw, on=['product_id','week_no'], how='left')
    m[['disp','mail']] = m[['disp','mail']].fillna(0.0)
    m['sd'] = m.sales_value * m.disp
    m['sm'] = m.sales_value * m.mail
    g = m.groupby('household_key')
    exp = pd.DataFrame({
        'exp84_spend': g.sales_value.sum(),
        'exp84_disp_spend': g.sd.sum(),
        'exp84_mail_spend': g.sm.sum(),
        'exp84_disp_lines': g.disp.apply(lambda s: float((s>0).sum())),
        'exp84_mail_lines': g.mail.apply(lambda s: float((s>0).sum())),
        'exp84_disp_int': g.disp.mean(),
        'exp84_mail_int': g.mail.mean(),
        'exp84_nprod': g.product_id.nunique(),
    })
    exp['exp84_disp_share'] = exp.exp84_disp_spend / exp.exp84_spend.replace(0, np.nan)
    exp['exp84_mail_share'] = exp.exp84_mail_spend / exp.exp84_spend.replace(0, np.nan)
    exp['exp84_linefrac'] = (exp.exp84_disp_lines + exp.exp84_mail_lines) / (t84.groupby('household_key').size().reindex(exp.index).replace(0,np.nan))
    # store diversity
    t84s = tx[tx.day > day - 84][['household_key','store_id','sales_value']]
    gs = t84s.groupby('household_key')
    st = pd.DataFrame({'nstores84': gs.store_id.nunique(), 'spend84s': gs.sales_value.sum()})
    top = gs.sales_value.max()  # not right; need per-store sums
    per_store = t84s.groupby(['household_key','store_id']).sales_value.sum().reset_index()
    topstore = per_store.groupby('household_key').sales_value.max().reindex(st.index)
    st['topstore84'] = topstore / st.spend84s.replace(0,np.nan)
    tl = tx[['household_key','store_id']]
    st['nstores_life'] = tl.groupby('household_key').store_id.nunique().reindex(st.index)
    # coupon redemptions
    cr = view.coupon_redemptions
    gcr = cr.groupby('household_key')
    red = pd.DataFrame({'red_life': gcr.size()})
    red['red84'] = cr[cr.day > day-84].groupby('household_key').size()
    last_red = cr.groupby('household_key').day.max()
    red['days_since_red'] = (day - last_red).clip(0, 400)
    exp = exp.join(st).join(red)
    for c in ['exp84_spend','exp84_disp_spend','exp84_mail_spend','exp84_disp_lines','exp84_mail_lines']:
        exp['log_'+c] = np.log1p(exp[c].clip(lower=0))
    return exp.reindex(hh)

tab = agent_api.build_features(feat_fn)
print('new features shape:', tab.shape)
print(tab.drop(columns=['household_key','snapshot_day']).describe().T[['mean','std','min','max']].round(3).to_string())

e18 = agent_api.load_saved('e018_hinge_prune.parquet')
newcols = [c for c in tab.columns if c not in ('household_key','snapshot_day')]
merged = e18.merge(tab, on=['household_key','snapshot_day'], how='left')
agent_api.save_table(merged, 'e019_exposure')

tt2 = tt.copy()
SPLITS=[('S1',[95,123,151,179,207,235,263,291,319,347],[375,403,431]),
        ('S3',[95,123,151,179,207,235,263,291,319,347,375,403],[431])]
def ridge_eval(Xtr,ytr,Xva,alpha):
    A=np.hstack([Xtr,np.ones((len(Xtr),1))]); B=np.hstack([Xva,np.ones((len(Xva),1))])
    d=A.shape[1]; lam=alpha*np.eye(d); lam[-1,-1]=0.0
    w=np.linalg.solve(A.T@A+lam, A.T@ytr); return B@w
def local_score(cols, alphas=(20.0,50.0)):
    t2=e18.merge(tt,on=['household_key','snapshot_day'],how='left') if False else merged.merge(tt,on=['household_key','snapshot_day'],how='left')
    out={}
    for sname,trd,vad in SPLITS:
        tr=t2[t2.snapshot_day.isin(trd)]; va=t2[t2.snapshot_day.isin(vad)]
        Xtr=tr[cols].astype(float); Xva=va[cols].astype(float)
        med=Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
        mu=Xtr.mean(); sd=Xtr.std().replace(0,1.0)
        Xtr=((Xtr-mu)/sd).values; Xva=((Xva-mu)/sd).values
        ytr=tr.future_spend_4w.values; yva=va.future_spend_4w.values
        for a in alphas:
            out[(sname,a)]=float(np.abs(ridge_eval(Xtr,ytr,Xva,a)-yva).mean())
    return out
allcols=[c for c in merged.columns if c not in ('household_key','snapshot_day')]
base=local_score([c for c in allcols if c in e18.columns])
full=local_score(allcols)
print('BASE e18:',{f'{k[0]}@{k[1]}':round(v,3) for k,v in base.items()})
print('E18+exposure:',{f'{k[0]}@{k[1]}':round(v,3) for k,v in full.items()})
# also individual feature screening: e18 + one new feature at a time (S1@20)
t2=merged.merge(tt,on=['household_key','snapshot_day'],how='left')
tr=t2[t2.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347])]; va=t2[t2.snapshot_day.isin([375,403,431])]
def prep(cols):
    Xtr=tr[cols].astype(float); Xva=va[cols].astype(float)
    med=Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
    mu=Xtr.mean(); sd=Xtr.std().replace(0,1.0)
    return ((Xtr-mu)/sd).values, ((Xva-mu)/sd).values, tr.future_spend_4w.values, va.future_spend_4w.values
b=local_score([c for c in allcols if c in e18.columns],alphas=(20.0,))[('S1',20.0)]
print('single-feature deltas on S1@20 (positive=new feature helps):')
for c in newcols:
    Xtr,ytr0,Xva,yva0=prep([x for x in allcols if x in e18.columns or x==c])
    p=ridge_eval(Xtr,ytr0,Xva,20.0)
    print(f'  {c}: {float(np.abs(p-yva0).mean())-b:+.3f}')

# ---- cell ----
import agent_api, pandas as pd, numpy as np

tt = agent_api.train_targets()
merged = agent_api.load_saved('e019_exposure')
e18cols = [c for c in agent_api.load_saved('e018_hinge_prune.parquet').columns if c not in ('household_key','snapshot_day')]
newcols = [c for c in merged.columns if c not in ('household_key','snapshot_day') and c not in e18cols]

SPLITS=[('S1',[95,123,151,179,207,235,263,291,319,347],[375,403,431]),
        ('S3',[95,123,151,179,207,235,263,291,319,347,375,403],[431])]
def ridge_eval(Xtr,ytr,Xva,alpha):
    A=np.hstack([Xtr.reshape(len(Xtr),-1),np.ones((len(Xtr),1))]); B=np.hstack([Xva.reshape(len(Xva),-1),np.ones((len(Xva),1))])
    d=A.shape[1]; lam=alpha*np.eye(d); lam[-1,-1]=0.0
    w=np.linalg.solve(A.T@A+lam, A.T@ytr); return B@w
t2=merged.merge(tt,on=['household_key','snapshot_day'],how='left')
def prep(cols, sname):
    trd,vad=dict(SPLITS)[sname]
    tr=t2[t2.snapshot_day.isin(trd)]; va=t2[t2.snapshot_day.isin(vad)]
    Xtr=tr[cols].astype(float); Xva=va[cols].astype(float)
    med=Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
    mu=Xtr.mean(); sd=Xtr.std().replace(0,1.0)
    return ((Xtr-mu)/sd).values, ((Xva-mu)/sd).values, tr.future_spend_4w.values, va.future_spend_4w.values

res=[]
for c in newcols:
    for sname,base_cols in [('S1',e18cols),('S3',e18cols)]:
        cols=base_cols+[c]
        Xtr,Xva,ytr,yva=prep(cols,sname)
        a=20.0 if sname=='S1' else 50.0
        p=ridge_eval(Xtr,ytr,Xva,a)
        Xb,Yb,_,_=prep(base_cols,sname)
        pb=ridge_eval(Xb,ytr,Yb,a)
        res.append((c,sname,float(np.abs(p-yva).mean()-np.abs(pb-yva).mean())))
d=pd.DataFrame(res).pivot(index=0,columns=1,values=2)
print('delta MAE (positive = feature HELPS):')
print(d.round(3).sort_values('S1',ascending=False).to_string())
print()
print('sum of deltas:', d.sum(axis=1).sort_values(ascending=False).round(3).to_dict())

# ---- cell ----
import agent_api, pandas as pd, numpy as np

tt = agent_api.train_targets()
merged = agent_api.load_saved('e019_exposure.parquet')
e18cols = [c for c in agent_api.load_saved('e018_hinge_prune.parquet').columns if c not in ('household_key','snapshot_day')]
newcols = [c for c in merged.columns if c not in ('household_key','snapshot_day') and c not in e18cols]

SPLITS=[('S1',[95,123,151,179,207,235,263,291,319,347],[375,403,431]),
        ('S3',[95,123,151,179,207,235,263,291,319,347,375,403],[431])]
def ridge_eval(Xtr,ytr,Xva,alpha):
    A=np.hstack([Xtr.reshape(len(Xtr),-1),np.ones((len(Xtr),1))]); B=np.hstack([Xva.reshape(len(Xva),-1),np.ones((len(Xva),1))])
    d=A.shape[1]; lam=alpha*np.eye(d); lam[-1,-1]=0.0
    w=np.linalg.solve(A.T@A+lam, A.T@ytr); return B@w
t2=merged.merge(tt,on=['household_key','snapshot_day'],how='left')
def prep(cols, sname):
    trd,vad=dict(SPLITS)[sname]
    tr=t2[t2.snapshot_day.isin(trd)]; va=t2[t2.snapshot_day.isin(vad)]
    Xtr=tr[cols].astype(float); Xva=va[cols].astype(float)
    med=Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
    mu=Xtr.mean(); sd=Xtr.std().replace(0,1.0)
    return ((Xtr-mu)/sd).values, ((Xva-mu)/sd).values, tr.future_spend_4w.values, va.future_spend_4w.values

res=[]
for c in newcols:
    for sname in ['S1','S3']:
        cols=e18cols+[c]
        Xtr,Xva,ytr,yva=prep(cols,sname)
        a=20.0 if sname=='S1' else 50.0
        p=ridge_eval(Xtr,ytr,Xva,a)
        Xb,Yb,_,_=prep(e18cols,sname)
        pb=ridge_eval(Xb,ytr,Yb,a)
        res.append((c,sname,float(np.abs(p-yva).mean()-np.abs(pb-yva).mean())))
d=pd.DataFrame(res).pivot(index=0,columns=1,values=2)
print('delta MAE (positive = feature HELPS):')
print(d.round(3).sort_values('S1',ascending=False).to_string())
print()
print('sum of deltas:', d.sum(axis=1).sort_values(ascending=False).round(3).to_dict())

# ---- cell ----
import agent_api, pandas as pd, numpy as np

tt = agent_api.train_targets()
merged = agent_api.load_saved('e019_exposure.parquet')
e18cols = [c for c in agent_api.load_saved('e018_hinge_prune.parquet').columns if c not in ('household_key','snapshot_day')]
newcols = [c for c in merged.columns if c not in ('household_key','snapshot_day') and c not in e18cols]

SPLITS={'S1':([95,123,151,179,207,235,263,291,319,347],[375,403,431]),
        'S3':([95,123,151,179,207,235,263,291,319,347,375,403],[431])}
def ridge_eval(Xtr,ytr,Xva,alpha):
    A=np.hstack([Xtr.reshape(len(Xtr),-1),np.ones((len(Xtr),1))]); B=np.hstack([Xva.reshape(len(Xva),-1),np.ones((len(Xva),1))])
    d=A.shape[1]; lam=alpha*np.eye(d); lam[-1,-1]=0.0
    w=np.linalg.solve(A.T@A+lam, A.T@ytr); return B@w
t2=merged.merge(tt,on=['household_key','snapshot_day'],how='left')
def prep(cols, sname):
    trd,vad=SPLITS[sname]
    tr=t2[t2.snapshot_day.isin(trd)]; va=t2[t2.snapshot_day.isin(vad)]
    Xtr=tr[cols].astype(float); Xva=va[cols].astype(float)
    med=Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
    mu=Xtr.mean(); sd=Xtr.std().replace(0,1.0)
    return ((Xtr-mu)/sd).values, ((Xva-mu)/sd).values, tr.future_spend_4w.values, va.future_spend_4w.values

res=[]
for c in newcols:
    for sname in ['S1','S3']:
        Xtr,Xva,ytr,yva=prep(e18cols+[c],sname)
        a=20.0 if sname=='S1' else 50.0
        p=ridge_eval(Xtr,ytr,Xva,a)
        Xb,Yb,_,_=prep(e18cols,sname)
        pb=ridge_eval(Xb,ytr,Yb,a)
        res.append((c,sname,float(np.abs(p-yva).mean()-np.abs(pb-yva).mean())))
d=pd.DataFrame(res).pivot(index=0,columns=1,values=2)
print('delta MAE (positive = feature HELPS):')
print(d.round(3).sort_values('S1',ascending=False).to_string())
print('sum of deltas:', d.sum(axis=1).sort_values(ascending=False).round(3).to_dict())

# ---- cell ----
import agent_api, pandas as pd, numpy as np

e18 = agent_api.load_saved('e018_hinge_prune.parquet')
exp = agent_api.load_saved('e019_exposure.parquet')
tt = agent_api.train_targets()
TRAIN_DAYS=[95,123,151,179,207,235,263,291,319,347,375,403,431]

top10_exp = ['exp84_mail_lines','exp84_disp_spend','exp84_mail_spend','exp84_mail_int',
             'red_life','exp84_disp_lines','exp84_linefrac','red84','nstores84','days_since_red']

# hinge knots from TRAIN rows only
tr_rows = e18.merge(tt[['household_key','snapshot_day']],on=['household_key','snapshot_day'],how='inner')
sources = ['spend_7','spend_14','spend_21','spend_112','spend_364','spend_lag1','spend_lag2',
           'wk_avg_4','wk_avg_8','fwd28_median','trips_84','trips_28','basket_avg_28','active_days_28',
           'exp84_mail_spend','exp84_disp_spend','exp84_mail_lines','spend_same_ly','spend_life',
           'spend_168','spend_252','spend_504','spend_rate_life','recency']
KNOTS={}
for s in sources:
    lx = np.log1p(tr_rows[s].clip(lower=0).astype(float))
    KNOTS[s] = [float(lx.quantize if False else lx.quantile(q)) for q in (0.2,0.4,0.6,0.8,0.9,0.95)]
def add_hinges(df, src_list):
    out = df.copy()
    for s in src_list:
        lx = np.log1p(out[s].clip(lower=0).astype(float))
        for k,q in enumerate(KNOTS[s]):
            out[f'hg2_{s}_{k}'] = (lx - q).clip(lower=0)
    return out

SPLITS={'S1':([95,123,151,179,207,235,263,291,319,347],[375,403,431],20.0),
        'S3':([95,123,151,179,207,235,263,291,319,347,375,403],[431],50.0)}
def ridge_eval(Xtr,ytr,Xva,alpha):
    A=np.hstack([Xtr,np.ones((len(Xtr),1))]); B=np.hstack([Xva,np.ones((len(Xva),1))])
    d=A.shape[1]; lam=alpha*np.eye(d); lam[-1,-1]=0.0
    w=np.linalg.solve(A.T@A+lam, A.T@ytr); return B@w
base_tab = e18.merge(exp[['household_key','snapshot_day']+top10_exp],on=['household_key','snapshot_day'],how='left')
def score(tab, cols, sname):
    trd,vad,alpha=SPLITS[sname]
    t2=tab.merge(tt,on=['household_key','snapshot_day'],how='left')
    tr=t2[t2.snapshot_day.isin(trd)]; va=t2[t2.snapshot_day.isin(vad)]
    Xtr=tr[cols].astype(float); Xva=va[cols].astype(float)
    med=Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
    mu=Xtr.mean(); sd=Xtr.std().replace(0,1.0)
    Xtr=((Xtr-mu)/sd).values; Xva=((Xva-mu)/sd).values
    ytr=tr.future_spend_4w.values; yva=va.future_spend_4w.values
    return float(np.abs(ridge_eval(Xtr,ytr,Xva,alpha)-yva).mean())

c18=[c for c in e18.columns if c not in ('household_key','snapshot_day')]
cA=[c for c in base_tab.columns if c not in ('household_key','snapshot_day')]
print('base e18:  S1 %.3f  S3 %.3f' % (score(e18,c18,'S1'), score(e18,c18,'S3')))
print('A e18+exp: S1 %.3f  S3 %.3f' % (score(base_tab,cA,'S1'), score(base_tab,cA,'S3')))

# per-source hinge screen on top of A
res=[]
for s in sources:
    t2=add_hinges(base_tab,[s])
    cols=[c for c in t2.columns if c not in ('household_key','snapshot_day')]
    d1=score(base_tab,cA,'S1')-score(t2,cols,'S1')
    d3=score(base_tab,cA,'S3')-score(t2,cols,'S3')
    res.append((s,round(d1,3),round(d3,3),round(d1+d3,3)))
print('source | dS1 | dS3 | sum (positive = hinge helps):')
for r in sorted(res,key=lambda x:-x[3]): print(r)
good=[r[0] for r in res if r[3]>0.01]
print('good sources:',good)
final=add_hinges(base_tab,good)
mkt=['mkt28','mkt_mom','drift28','shr']
cf=[c for c in final.columns if c not in ('household_key','snapshot_day')]
print('FINAL (e18+exp+good hinges): S1 %.3f S3 %.3f' % (score(final,cf,'S1'), score(final,cf,'S3')))
cfm=[c for c in cf if c not in mkt]
print('FINAL minus mkt:            S1 %.3f S3 %.3f' % (score(final,cfm,'S1'), score(final,cfm,'S3')))
keep = cfm if (score(final,cfm,'S1')+score(final,cfm,'S3')) < (score(final,cf,'S1')+score(final,cf,'S3')) else cf
out = final[['household_key','snapshot_day']+keep]
agent_api.save_table(out,'e019_final')
print('saved e019_final:', out.shape, 'nfeat', len(keep), 'mkt dropped:', 'mkt28' not in keep)
print('knots:', {k:[round(x,3) for x in v] for k,v in KNOTS.items() if k in good})

# ---- cell ----
import agent_api, pandas as pd, numpy as np

e18 = agent_api.load_saved('e018_hinge_prune.parquet')
exp = agent_api.load_saved('e019_exposure.parquet')
tt = agent_api.train_targets()
top10_exp = ['exp84_mail_lines','exp84_disp_spend','exp84_mail_spend','exp84_mail_int',
             'red_life','exp84_disp_lines','exp84_linefrac','red84','nstores84','days_since_red']
base_tab = e18.merge(exp[['household_key','snapshot_day']+top10_exp], on=['household_key','snapshot_day'], how='left')

sources = ['spend_7','spend_14','spend_21','spend_112','spend_364','spend_lag1','spend_lag2',
           'wk_avg_4','wk_avg_8','fwd28_median','trips_84','trips_28','basket_avg_28','active_days_28',
           'exp84_mail_spend','exp84_disp_spend','exp84_mail_lines','spend_same_ly','spend_life',
           'spend_168','spend_252','spend_504','spend_rate_life','recency']
tr_rows = base_tab.merge(tt[['household_key','snapshot_day']], on=['household_key','snapshot_day'], how='inner')
KNOTS = {}
for s in sources:
    lx = np.log1p(tr_rows[s].clip(lower=0).astype(float))
    KNOTS[s] = [float(lx.quantile(q)) for q in (0.2,0.4,0.6,0.8,0.9,0.95)]
def add_hinges(df, src_list):
    out = df.copy()
    for s in src_list:
        lx = np.log1p(out[s].clip(lower=0).astype(float))
        for k,q in enumerate(KNOTS[s]):
            out[f'hg2_{s}_{k}'] = (lx - q).clip(lower=0)
    return out

SPLITS = {'S1':([95,123,151,179,207,235,263,291,319,347],[375,403,431],20.0),
          'S3':([95,123,151,179,207,235,263,291,319,347,375,403],[431],50.0)}
def ridge_eval(Xtr,ytr,Xva,alpha):
    A=np.hstack([Xtr,np.ones((len(Xtr),1))]); B=np.hstack([Xva,np.ones((len(Xva),1))])
    d=A.shape[1]; lam=alpha*np.eye(d); lam[-1,-1]=0.0
    w=np.linalg.solve(A.T@A+lam, A.T@ytr); return B@w
t2full = base_tab.merge(tt, on=['household_key','snapshot_day'], how='left')
def score(tab, cols, sname):
    trd,vad,alpha = SPLITS[sname]
    t2 = t2full if tab is base_tab else tab.merge(tt, on=['household_key','snapshot_day'], how='left')
    tr=t2[t2.snapshot_day.isin(trd)]; va=t2[t2.snapshot_day.isin(vad)]
    Xtr=tr[cols].astype(float); Xva=va[cols].astype(float)
    med=Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
    mu=Xtr.mean(); sd=Xtr.std().replace(0,1.0)
    Xtr=((Xtr-mu)/sd).values; Xva=((Xva-mu)/sd).values
    ytr=tr.future_spend_4w.values; yva=va.future_spend_4w.values
    return float(np.abs(ridge_eval(Xtr,ytr,Xva,alpha)-yva).mean())

cA=[c for c in base_tab.columns if c not in ('household_key','snapshot_day')]
print('A e18+exp10: S1 %.3f S3 %.3f' % (score(base_tab,cA,'S1'), score(base_tab,cA,'S3')))
res=[]
for s in sources:
    t2=add_hinges(base_tab,[s])
    cols=[c for c in t2.columns if c not in ('household_key','snapshot_day')]
    d1=score(base_tab,cA,'S1')-score(t2,cols,'S1')
    d3=score(base_tab,cA,'S3')-score(t2,cols,'S3')
    res.append((s,round(d1,3),round(d3,3),round(d1+d3,3)))
print('source | dS1 | dS3 | sum:')
for r in sorted(res,key=lambda x:-x[3]): print(r)
good=[r[0] for r in res if r[3]>0.01]
print('good sources:',good)
final=add_hinges(base_tab,good)
mkt=['mkt28','mkt_mom','drift28','shr']
cf=[c for c in final.columns if c not in ('household_key','snapshot_day')]
cfm=[c for c in cf if c not in mkt]
s_with=score(final,cf,'S1')+score(final,cf,'S3'); s_wo=score(final,cfm,'S1')+score(final,cfm,'S3')
print('FINAL: S1 %.3f S3 %.3f | minus-mkt: S1 %.3f S3 %.3f' % (score(final,cf,'S1'),score(final,cf,'S3'),score(final,cfm,'S1'),score(final,cfm,'S3')))
keep = cfm if s_wo < s_with else cf
out = final[['household_key','snapshot_day']+keep]
agent_api.save_table(out,'e019_final')
print('saved e019_final:', out.shape, 'nfeat', len(keep), 'mkt_dropped', 'mkt28' not in keep)