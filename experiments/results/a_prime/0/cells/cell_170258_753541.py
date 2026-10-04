import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
d=A.load_saved('cand_v2.parquet').merge(A.train_targets(),on=['household_key','snapshot_day'])
y=d.future_spend_4w.values
base_cols=[c for c in A.load_saved('e013_stock.parquet').columns if c not in ('household_key','snapshot_day')]
ew=['ew7','ew14','ew28','ew56','ew112','ew_ratio_7_56','ew_ratio_28_112']
def qtr(col,q): return d.groupby('snapshot_day')[col].transform(lambda s: s.quantile(q))
d['t_s28_q90']=d.spend_28/qtr('spend_28',.9); d['t_s84_q90']=d.spend_84/qtr('spend_84',.9)
d['t_s364_q95']=d.spend_364/qtr('spend_364',.95)
d['t_top5_s28']=(d.spend_28>=qtr('spend_28',.95)).astype(float)
d['t_top5_s364']=(d.spend_364>=qtr('spend_364',.95)).astype(float)
d['t_sq28']=np.sqrt(d.spend_28); d['t_sq84']=np.sqrt(d.spend_84)
tail=['t_s28_q90','t_s84_q90','t_s364_q95','t_top5_s28','t_top5_s364','t_sq28','t_sq84']
# interaction candidates
d['i_pow']=d.spend_28**1.25
d['i_act']=d.spend_28*d.active_28
d['i_rec7']=d.spend_28*(d.recency<=7).astype(float)
d['i_ewact']=d.ew28*d.tp_act8 if 'tp_act8' in d else d.ew28
d['i_red']=d.spend_28*np.log1p(d.m_red_28)
d['i_gas']=d.spend_28*d.x_dep_KIOSK_GAS if 'x_dep_KIOSK-GAS' in d.columns else 0
d['i_misc']=d.spend_28*d['x_dep_MISC SALES TRAN']
inter=['i_pow','i_act','i_rec7','i_ewact','i_red','i_misc']
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
print('BASE+EWMA+TAIL:',round(float(np.abs(fitpred(prep(base_cols+ew+tail,d),300,tr431,m431,y)-y[m431]).mean()),3))
for c in inter:
    print('+'+c, round(float(np.abs(fitpred(prep(base_cols+ew+tail+[c],d),300,tr431,m431,y)-y[m431]).mean()),3))
print('+ALL inter:',round(float(np.abs(fitpred(prep(base_cols+ew+tail+inter,d),300,tr431,m431,y)-y[m431]).mean()),3))
