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