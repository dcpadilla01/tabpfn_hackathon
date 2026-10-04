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