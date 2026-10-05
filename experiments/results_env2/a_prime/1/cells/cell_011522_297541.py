import agent_api, pandas as pd, numpy as np
df = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='left').reset_index(drop=True)
is_tr = d['future_spend_4w'].notna().values
cat_cols = [c for c in df.columns if str(df[c].dtype)=='category']
base_drop = set(['household_key','snapshot_day','stack_ridge_log','stack_ridge'])|set(cat_cols)
num_cols = [c for c in df.columns if c not in base_drop]
X = np.hstack([d[num_cols].astype(float).values, pd.get_dummies(d[cat_cols].astype(str), dummy_na=True).values.astype(np.float64)])
days = d['snapshot_day'].values; y=d['future_spend_4w'].values; ylog=np.log1p(y)
tr_days = np.array(sorted(set(days[is_tr])))

def std_fit(A):
    mu=np.nanmean(A,axis=0); sd=np.nanstd(A,axis=0); sd[~np.isfinite(sd)]=1; return mu,sd
def std_apply(A,mu,sd):
    Z=(A-mu)/sd
    return np.where(np.isfinite(Z),Z,0.0)

def loso(alphas, space='log', mode='loso', decay=None):
    out={a:np.full(len(d),np.nan) for a in alphas}
    for s in tr_days:
        q=np.where(days==s)[0]
        f=np.where(is_tr&(days!=s))[0] if mode=='loso' else np.where(is_tr&(days<s))[0]
        if len(f)<50:
            v=np.nanmean(ylog[is_tr]) if space=='log' else np.nanmean(y[is_tr])
            for a in alphas: out[a][q]=v
            continue
        w=np.ones(len(f)) if decay is None else np.exp(-(s-days[f])/decay)
        yf=(ylog[f] if space=='log' else y[f]).astype(float)
        mu,sd=std_fit(X[f]); Xf=std_apply(X[f],mu,sd)
        ym=np.average(yf,weights=w); r=(yf-ym)*np.sqrt(w); Xw=Xf*np.sqrt(w)[:,None]
        G=Xw.T@Xw; b=Xw.T@r; I=np.eye(G.shape[0])
        Xq=std_apply(X[q],mu,sd)
        for a in alphas:
            wg=np.linalg.solve(G+a*I,b); out[a][q]=Xq@wg+ym
    return out

def mae(p, space='log'):
    pr=np.clip(np.expm1(p) if space=='log' else p,0,None)
    return np.mean(np.abs(pr[is_tr]-y[is_tr]))
late=(days>=347)&is_tr
def mae_l(p, space='log'):
    pr=np.clip(np.expm1(p) if space=='log' else p,0,None)
    return np.mean(np.abs(pr[late]-y[late]))

print('cur stack_log MAE %.3f'%mae(d['stack_ridge_log'].values,'log'))
al=[30,100,300,1000,3000]
oof=loso(al,'log','loso')
for a in al: print('LOSO log a=%-5d MAE %.3f late %.3f'%(a,mae(oof[a],'log'),mae_l(oof[a],'log')))
oofr=loso([300,1000,3000,10000],'raw','loso')
for a in [300,1000,3000,10000]: print('LOSO raw a=%-5d MAE %.3f late %.3f'%(a,mae(oofr[a],'raw'),mae_l(oofr[a],'raw')))
for dec in [600,300,150]:
    o=loso([300],'log','forward',dec)[300]
    print('fwd dec=%-4d MAE %.3f late %.3f'%(dec,mae(o,'log'),mae_l(o,'log')))
res=np.where(is_tr,ylog-oof[300],np.nan)
bg=np.nanmean(res); print('global log-bias cal MAE %.3f (bias %.4f)'%(mae(oof[300]+bg,'log'),bg))
mr={s:np.nanmean(res[days==s]) for s in tr_days}
bs={s:np.mean([v for s2,v in mr.items() if s2!=s]) for s in tr_days}
cal=oof[300]+np.array([bs.get(s,0.0) for s in days])
print('per-snap log-bias cal MAE %.3f'%mae(cal,'log'))
cmp=pd.DataFrame({'day':days[is_tr],'y':y[is_tr],'p':np.clip(np.expm1(oof[300]),0,None)[is_tr]}).groupby('day').mean()
print(cmp.round(1).T)
