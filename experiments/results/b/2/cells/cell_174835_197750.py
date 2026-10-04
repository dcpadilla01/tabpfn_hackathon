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