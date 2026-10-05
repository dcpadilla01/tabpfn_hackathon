import agent_api, pandas as pd, numpy as np, time
t0=time.time()
df = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='left').reset_index(drop=True)
is_tr = d['future_spend_4w'].notna().values
cat_cols = [c for c in df.columns if str(df[c].dtype)=='category']
base_drop = set(['household_key','snapshot_day','stack_ridge_log','stack_ridge'])|set(cat_cols)
num_cols = [c for c in df.columns if c not in base_drop]
print('n_cat',len(cat_cols),'n_num',len(num_cols),'| snapshot_day in num:','snapshot_day' in num_cols)
import re
print('RFM-like names:', [c for c in num_cols if re.search(r'sp\d|trip|recen|tenur|^ew|wk|basket', c)][:30])
dums = pd.get_dummies(d[cat_cols].astype(str), dummy_na=True)
X = np.hstack([d[num_cols].astype(float).values, dums.values.astype(np.float64)])
days = d['snapshot_day'].values
y = d['future_spend_4w'].values.copy()
ylog = np.log1p(y)
tr_days = np.array(sorted(set(days[is_tr])))
print('X', X.shape, '| t', round(time.time()-t0,1))

g = pd.DataFrame({'day':days[is_tr],'y':y[is_tr]}).groupby('day')['y'].agg(['mean','median',lambda z:(z==0).mean()])
g.columns=['mean','median','zero']; print(g.round(1).T)

def std_fit(A):
    mu=np.nanmean(A,axis=0); sd=np.nanstd(A,axis=0); sd[sd==0]=1; return mu,sd
def std_apply(A,mu,sd):
    return np.where(np.isnan(A),0.0,(A-mu)/sd)

def loso_ridge(alphas, space='log', mode='loso', decay=None):
    oofs={a:np.zeros(len(d)) for a in alphas}
    for s in tr_days:
        q=np.where(days==s)[0]
        f=np.where(is_tr&(days!=s))[0] if mode=='loso' else np.where(is_tr&(days<s))[0]
        if len(f)==0:
            v=np.nanmean(ylog[is_tr]) if space=='log' else np.nanmean(y[is_tr])
            for a in alphas: oofs[a][q]=v
            continue
        w=np.ones(len(f)) if decay is None else np.exp(-(s-days[f])/decay)
        yf=ylog[f] if space=='log' else y[f]
        mu,sd=std_fit(X[f]); Xf=std_apply(X[f],mu,sd)
        ym=np.average(yf,weights=w); r=(yf-ym)*np.sqrt(w); Xw=Xf*np.sqrt(w)[:,None]
        G=Xw.T@Xw; b=Xw.T@r; I=np.eye(G.shape[0])
        Xq=std_apply(X[q],mu,sd)
        for a in alphas:
            wg=np.linalg.solve(G+a*I,b); oofs[a][q]=Xq@wg+ym
    return oofs

def mae_p(p, space='log'):
    pr=np.clip(np.expm1(p) if space=='log' else p,0,None)
    return np.mean(np.abs(pr[is_tr]-y[is_tr]))
late=(days>=347)&is_tr
def mae_late(p, space='log'):
    pr=np.clip(np.expm1(p) if space=='log' else p,0,None)
    return np.mean(np.abs(pr[late]-y[late]))

print('cur stack_log MAE %.3f | cur stack raw MAE %.3f'%(
    mae_p(d['stack_ridge_log'].values,'log'),
    np.mean(np.abs(np.clip(d['stack_ridge'].values,0,None)[is_tr]-y[is_tr]))))
al=[30,100,300,1000,3000]
oof_log=loso_ridge(al,'log','loso')
for a in al: print('LOSO log a=%-5d MAE %.3f late %.3f'%(a,mae_p(oof_log[a],'log'),mae_late(oof_log[a],'log')))
oof_raw=loso_ridge([300,1000,3000,10000],'raw','loso')
for a in [300,1000,3000,10000]: print('LOSO raw a=%-5d MAE %.3f late %.3f'%(a,mae_p(oof_raw[a],'raw'),mae_late(oof_raw[a],'raw')))
for dec in [None,600,300,150]:
    oof_f=loso_ridge([300],'log','forward',dec)[300]
    print('fwd dec=%-4s allMAE %.3f lateMAE %.3f'%(dec,mae_p(oof_f,'log'),mae_late(oof_f,'log')))
res=np.where(is_tr,ylog-oof_log[300],np.nan)
bg=np.nanmean(res); print('bias-cal global MAE %.3f (bias %.4f)'%(mae_p(oof_log[300]+bg,'log'),bg))
mr={s:np.nanmean(res[days==s]) for s in tr_days}
bs=np.array([np.mean([mr[s2] for s2 in tr_days if s2!=s]) for s in tr_days])
cal=oof_log[300]+np.array([bs[list(tr_days).index(s)] if s in tr_days else 0.0 for s in days])
print('bias-cal per-snap MAE %.3f'%mae_p(cal,'log'))
cmp=pd.DataFrame({'day':days[is_tr],'y':y[is_tr],'p':np.clip(np.expm1(oof_log[300]),0,None)[is_tr]}).groupby('day').mean()
print(cmp.round(1).T)
print('t',round(time.time()-t0,1))
