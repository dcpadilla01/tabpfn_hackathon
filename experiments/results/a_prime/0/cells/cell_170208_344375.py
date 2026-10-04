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
        if X[c].dtype==object: X[c]=X[c].astype('category').cat.codes.astype(float)
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
sub=d[m431]
cors={}
for c in allc:
    v=sub[c]
    if v.dtype==object: v=v.astype('category').cat.codes
    v=pd.to_numeric(v,errors='coerce').fillna(v.median() if v.dtype!=object else 0).values.astype(float)
    if np.std(v)>1e-9: cors[c]=float(np.corrcoef(v,r)[0,1])
cs=pd.Series(cors).sort_values()
print('NEG:'); print(cs.head(10).round(3)); print('POS:'); print(cs.tail(12).round(3))
# per-snapshot tail features
def qtr(col,q):
    return d.groupby('snapshot_day')[col].transform(lambda s: s.quantile(q))
d['t_s28_q90']=d.spend_28/qtr('spend_28',.9)
d['t_s84_q90']=d.spend_84/qtr('spend_84',.9)
d['t_s364_q95']=d.spend_364/qtr('spend_364',.95)
d['t_top5_s28']=(d.spend_28>=qtr('spend_28',.95)).astype(float)
d['t_top5_s364']=(d.spend_364>=qtr('spend_364',.95)).astype(float)
d['t_sq28']=np.sqrt(d.spend_28); d['t_sq84']=np.sqrt(d.spend_84)
tail=['t_s28_q90','t_s84_q90','t_s364_q95','t_top5_s28','t_top5_s364','t_sq28','t_sq84']
ew=['ew7','ew14','ew28','ew56','ew112','ew_ratio_7_56','ew_ratio_28_112']
tc={c:float(np.corrcoef(d.loc[m431,c].fillna(0).values,r)[0,1]) for c in tail}
print('tail residual corr:',pd.Series(tc).round(3).to_dict())
for name,grp in [('BASE',[]),('+EWMA',ew),('+TAIL',tail),('+EWMA+TAIL',ew+tail)]:
    p2=fitpred(prep(base_cols+grp,d),300,tr431,m431,y)
    print(name,'MAE@431',round(float(np.abs(p2-y[m431]).mean()),3))
# also check tail MAE on top decile
p3=fitpred(prep(base_cols+ew+tail,d),300,tr431,m431,y)
rr=pd.DataFrame({'y':y[m431],'p':p3}); rr['bin']=pd.qcut(rr.y,10,duplicates='drop')
print(rr.groupby('bin',observed=True).agg(n=('y','size'),ym=('y','mean'),pm=('p','mean'),mae=('y',lambda z:0)).drop(columns='mae').assign(
    mae=np.abs(rr.p.values-rr.y.values)).groupby('bin',observed=True).agg(n=('n','first'),ym=('ym','first'),pm=('pm','first'),mae=('mae','mean')).round(1))
