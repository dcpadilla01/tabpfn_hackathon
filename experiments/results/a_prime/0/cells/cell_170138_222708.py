import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
d=A.load_saved('cand_v2.parquet').merge(A.train_targets(),on=['household_key','snapshot_day'])
y=d.future_spend_4w.values
base_cols=[c for c in A.load_saved('e013_stock.parquet').columns if c not in ('household_key','snapshot_day')]
newcols=['ew7','ew14','ew28','ew56','ew112','ew_ratio_7_56','ew_ratio_28_112','tp_act8','tp_nzmean','tp_nzmax','tp_nzmed','tp_pred','tp_max_med','tp_last_max','tp_cv','own_ratio_med','own_pct','pr_s28','pr_s84','pr_s364','pr_act','pr_nzmean','pr_self_peer']
allc=base_cols+newcols
def prep(cols,frame):
    X=frame[cols].copy()
    for c in cols:
        if X[c].dtype==object or str(X[c].dtype)=='category': X[c]=X[c].astype('category').cat.codes.astype(float)
    Xv=X.values.astype(float); med=np.nanmedian(Xv,0); Xv=np.where(np.isnan(Xv),med,Xv)
    return (Xv-Xv.mean(0))/(Xv.std(0)+1e-9)
def fitpred(Xs,lam,trm,tem,yy):
    nn=int(trm.sum()); Xd=np.hstack([np.ones((nn,1)),Xs[trm]])
    M=Xd.T@Xd+lam*np.eye(Xd.shape[1]); M[0,0]-=lam
    w=np.linalg.solve(M,Xd.T@yy[trm])
    return np.hstack([np.ones((int(tem.sum()),1)),Xs[tem]])@w
m431=(d.snapshot_day==431).values; tr431=(d.snapshot_day<=403).values
Xs=prep(allc,d)
p=fitpred(Xs,300,tr431,m431,y)
r=y[m431]-p
res=pd.Series(r)
sub=d[m431]
cors={}
for c in allc:
    v=sub[c].astype(float).fillna(sub[c].astype(float).median() if sub[c].dtype!=object else 0).values
    if sub[c].dtype==object: v=sub[c].astype('category').cat.codes.values.astype(float)
    s=np.std(v)
    if s>1e-9: cors[c]=float(np.corrcoef(v,r)[0,1])
cs=pd.Series(cors).sort_values()
print('TOP NEG residual-corr:'); print(cs.head(12).round(3))
print('TOP POS residual-corr:'); print(cs.tail(15).round(3))
# tail block built from cohort percentiles at 431
q90_28=sub.spend_28.quantile(.9); q90_84=sub.spend_84.quantile(.9); q90_364=sub.spend_364.quantile(.9)
tb=pd.DataFrame(index=sub.index)
tb['t_s28_q90']=sub.spend_28/q90_28; tb['t_s84_q90']=sub.spend_84/q90_84; tb['t_s364_q90']=sub.spend_364/q90_364
tb['t_top5_s28']=(sub.spend_28>=sub.spend_28.quantile(.95)).astype(float)
tb['t_top5_s364']=(sub.spend_364>=sub.spend_364.quantile(.95)).astype(float)
tb['t_sq28']=np.sqrt(sub.spend_28); tb['t_log28']=np.log1p(sub.spend_28)
tb['t_rk_x_q90']=sub.rk_s28_pct*tb.t_s28_q90
# residual corr of tail feats
tc={}
for c in tb.columns: tc[c]=float(np.corrcoef(tb[c].fillna(0).values,r)[0,1])
print('tail-feat residual corr:', pd.Series(tc).round(3).to_dict())
# test: base+EWMA, base+tail, base+EWMA+tail (tail built per-snapshot for all snaps quickly via groupby transform)
g28=d.groupby('snapshot_day').spend_28.transform(lambda s: s.quantile(.9))
g84=d.groupby('snapshot_day').spend_84.transform(lambda s: s.quantile(.9))
g364=d.groupby('snapshot_day').spend_364.transform(lambda s: s.quantile(.95))
d['t_s28_q90']=d.spend_28/g28; d['t_s84_q90']=d.spend_84/g84; d['t_s364_q95']=d.spend_364/g364
d['t_top5_s28']=(d.spend_28>=d.groupby('snapshot_day').spend_28.transform(lambda s:s.quantile(.95))).astype(float)
d['t_sq28']=np.sqrt(d.spend_28)
tail=['t_s28_q90','t_s84_q90','t_s364_q95','t_top5_s28','t_sq28']
ew=['ew7','ew14','ew28','ew56','ew112','ew_ratio_7_56','ew_ratio_28_112']
for name,grp in [('BASE',[]),('+EWMA',ew),('+TAIL',tail),('+EWMA+TAIL',ew+tail),('+TAIL+rkx',['t_rk_x_q90'] if False else tail+['t_rk_x_q90'])]:
    X2=prep(allc if False else base_cols+newcols+ [c for c in grp if c in d.columns],d) if False else prep(base_cols+grp,d)
    p2=fitpred(X2,300,tr431,m431,y)
    print(name,'MAE@431',round(float(np.abs(p2-y[m431]).mean()),3))
