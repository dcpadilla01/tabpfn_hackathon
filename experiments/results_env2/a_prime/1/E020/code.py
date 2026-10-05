import agent_api, pandas as pd, numpy as np
df = agent_api.load_saved('e015_stack.parquet')
print('shape', df.shape)
print('stack cols:', [c for c in df.columns if 'stack' in c.lower()])
print(df.dtypes.value_counts())
tt = agent_api.train_targets()
print('targets:', tt.shape, tt.columns.tolist())
print(df['snapshot_day'].value_counts().sort_index())
m = df.merge(tt, on=['household_key','snapshot_day'], how='left')
print('train rows:', m['future_spend_4w'].notna().sum(), '| val rows:', m['future_spend_4w'].isna().sum())
nv = df.isna().mean()
print('cols with NaN (top):'); print(nv[nv>0].sort_values(ascending=False).head(10))


# ---- cell ----
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


# ---- cell ----
import agent_api, pandas as pd, numpy as np
df = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='left').reset_index(drop=True)
is_tr = d['future_spend_4w'].notna().values
cat_cols = [c for c in df.columns if str(df[c].dtype)=='category']
base_drop = set(['household_key','snapshot_day','stack_ridge_log','stack_ridge'])|set(cat_cols)
num_cols = [c for c in df.columns if c not in base_drop]
X = np.hstack([d[num_cols].astype(float).values, pd.get_dummies(d[cat_cols].astype(str), dummy_na=True).values.astype(np.float64)])
print('inf count:', np.isinf(X).sum())
infcols = np.isinf(X).sum(axis=0)
bad = [num_cols[i] if i<len(num_cols) else ('DUM%d'%i) for i in np.where(infcols>0)[0]]
print('cols with inf:', bad[:20])
print('rows with inf:', np.isinf(X).any(axis=1).sum())
days = d['snapshot_day'].values
rows_inf = np.where(np.isinf(X).any(axis=1))[0]
print('inf rows by day:', pd.Series(days[rows_inf]).value_counts().sort_index().to_dict())
# check per-column inf by day
for c in bad[:5]:
    j = num_cols.index(c) if c in num_cols else None
    print(c, 'inf by day:', pd.Series(days[np.isinf(X[:,j])]).value_counts().to_dict())


# ---- cell ----
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


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time
t0=time.time()
df = agent_api.load_saved('e015_stack.parquet')
print('e012_style shape:', agent_api.load_saved('e012_style.parquet').shape)
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='left').reset_index(drop=True)
is_tr = d['future_spend_4w'].notna().values
cat_cols = [c for c in df.columns if str(df[c].dtype)=='category']
base_drop = set(['household_key','snapshot_day','stack_ridge_log','stack_ridge'])|set(cat_cols)
num_cols = [c for c in df.columns if c not in base_drop]
Xn = d[num_cols].astype(float).values
Xd = pd.get_dummies(d[cat_cols].astype(str), dummy_na=True).values.astype(np.float64)
X = np.hstack([Xn, Xd])
days = d['snapshot_day'].values; y=d['future_spend_4w'].values.astype(float)
tr_days = np.array(sorted(set(days[is_tr])))
late = is_tr & (days>=347)
print('y quantiles (50/90/95/99/99.9):', np.nanpercentile(y[is_tr],[50,90,95,99,99.9]).round(1), 'max', np.nanmax(y[is_tr]))

def std_fit(A):
    mu=np.nanmean(A,axis=0); sd=np.nanstd(A,axis=0); sd[~np.isfinite(sd)]=1.0; sd[sd==0]=1.0; return mu,sd
def std_apply(A,mu,sd):
    Z=(A-mu)/sd; return np.where(np.isfinite(Z),Z,0.0)

def loso(Xall, alpha, tfun, mode='loso', decay=None, day_feat=False, wins=True):
    out=np.full(len(d),np.nan)
    for s in tr_days:
        q=np.where(days==s)[0]
        f=np.where(is_tr&(days!=s))[0] if mode=='loso' else np.where(is_tr&(days<s))[0]
        if len(f)<50:
            out[q]=np.nanmean(y[is_tr]); continue
        Xf=Xall[f].copy(); Xq=Xall[q].copy()
        if day_feat:
            Xf=np.hstack([Xf,(days[f]/100.0)[:,None]]); Xq=np.hstack([Xq,(days[q]/100.0)[:,None]])
        if wins:
            lo=np.nanpercentile(Xf,0.5,axis=0); hi=np.nanpercentile(Xf,99.5,axis=0)
            lo=np.where(np.isfinite(lo),lo,0); hi=np.where(np.isfinite(hi),hi,1); bad=hi<=lo; hi[bad]=lo[bad]+1
            Xf=np.clip(Xf,lo,hi); Xq=np.clip(Xq,lo,hi)
        w=np.ones(len(f)) if decay is None else np.exp(-(s-days[f])/decay)
        yf=tfun(y[f])
        mu,sd=std_fit(Xf); Xfs=std_apply(Xf,mu,sd)
        ym=np.average(yf,weights=w); r=(yf-ym)*np.sqrt(w); Xw=Xfs*np.sqrt(w)[:,None]
        G=Xw.T@Xw; b=Xw.T@r
        wg=np.linalg.solve(G+alpha*np.eye(G.shape[0]),b)
        out[q]=std_apply(Xq,mu,sd)@wg+ym
    return out

def ev(oof, inv, mask=None):
    pr=np.clip(inv(np.asarray(oof,float)),0,None)
    m = is_tr if mask is None else mask
    return np.mean(np.abs(pr[m]-y[m]))

tfuns = {
 'id':    (lambda v: v,               lambda p: p),
 'sqrt':  (np.sqrt,                   lambda p: p**2),
 'cbrt':  (np.cbrt,                   lambda p: p**3),
 'log':   (np.log1p,                  np.expm1),
 'as20':  (lambda v: np.arcsinh(v/20.0), lambda p: 20.0*np.sinh(p)),
}
print('--- LOSO a=1000, winsorized, all cols ---')
oof_store={}
for name,(t,inv) in tfuns.items():
    o=loso(X,1000,t,'loso')
    oof_store[name]=o
    print('%-6s train MAE %6.3f | late %6.3f'%(name, ev(o,inv), ev(o,inv,late)))
print('t',round(time.time()-t0,1))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time
t0=time.time()
df = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='left').reset_index(drop=True)
is_tr = d['future_spend_4w'].notna().values
cat_cols = [c for c in df.columns if str(df[c].dtype)=='category']
base_drop = set(['household_key','snapshot_day','stack_ridge_log','stack_ridge'])|set(cat_cols)
num_cols = [c for c in df.columns if c not in base_drop]
X = np.hstack([d[num_cols].astype(float).values, pd.get_dummies(d[cat_cols].astype(str), dummy_na=True).values.astype(np.float64)])
days = d['snapshot_day'].values; y=d['future_spend_4w'].values.astype(float)
tr_days = np.array(sorted(set(days[is_tr])))
late = is_tr & (days>=347)

def std_fit(A):
    mu=np.nanmean(A,axis=0); sd=np.nanstd(A,axis=0); sd[~np.isfinite(sd)]=1.0; sd[sd==0]=1.0; return mu,sd
def std_apply(A,mu,sd):
    Z=(A-mu)/sd; return np.where(np.isfinite(Z),Z,0.0)

def loso(Xall, alpha, tfun, decay=None, day_feat=False, wins=True):
    out=np.full(len(d),np.nan)
    for s in tr_days:
        q=np.where(days==s)[0]
        f=np.where(is_tr&(days!=s))[0]
        if len(f)<50: out[q]=np.nanmean(y[is_tr]); continue
        Xf=Xall[f].copy(); Xq=Xall[q].copy()
        if day_feat:
            Xf=np.hstack([Xf,(days[f]/100.0)[:,None]]); Xq=np.hstack([Xq,(days[q]/100.0)[:,None]])
        if wins:
            lo=np.nanpercentile(Xf,0.5,axis=0); hi=np.nanpercentile(Xf,99.5,axis=0)
            lo=np.where(np.isfinite(lo),lo,0); hi=np.where(np.isfinite(hi),hi,1); bad=hi<=lo; hi[bad]=lo[bad]+1
            Xf=np.clip(Xf,lo,hi); Xq=np.clip(Xq,lo,hi)
        w=np.ones(len(f)) if decay is None else np.exp(-(s-days[f])/decay)
        yf=tfun(y[f]); mu,sd=std_fit(Xf); Xfs=std_apply(Xf,mu,sd)
        ym=np.average(yf,weights=w); r=(yf-ym)*np.sqrt(w); Xw=Xfs*np.sqrt(w)[:,None]
        wg=np.linalg.solve(Xw.T@Xw+alpha*np.eye(Xw.shape[1]),Xw.T@r)
        out[q]=std_apply(Xq,mu,sd)@wg+ym
    return out
def ev(oof, inv, mask=None):
    pr=np.clip(inv(np.asarray(oof,float)),0,None)
    m = is_tr if mask is None else mask
    return np.mean(np.abs(pr[m]-y[m]))

sq=lambda v:np.sqrt(v); invsq=lambda p:p**2
print('--- sqrt alpha sweep ---')
for a in [100,300,1000,3000,10000]:
    o=loso(X,a,sq); print('a=%-5d MAE %.3f late %.3f'%(a,ev(o,invsq),ev(o,invsq,late)))
print('--- sqrt, no winsorize ---')
o=loso(X,1000,sq,wins=False); print('MAE %.3f late %.3f'%(ev(o,invsq),ev(o,invsq,late)))
print('--- sqrt + day_feat ---')
o=loso(X,1000,sq,day_feat=True); print('MAE %.3f late %.3f'%(ev(o,invsq),ev(o,invsq,late)))
print('--- 4th root ---')
o=loso(X,1000,lambda v:v**0.25); print('MAE %.3f'%(ev(o,lambda p:p**4),))
print('--- two-part: P(zero) logit-ish ridge x sqrt spend given nonzero ---')
z=(y==0).astype(float)
pzero=loso(X,1000,lambda v:z[v if False else slice(None)][f] if False else None) if False else None
# simpler: reuse loso with custom target arrays
def loso_yt(Xall, alpha, ytfun, decay=None, day_feat=False, wins=True):
    out=np.full(len(d),np.nan)
    for s in tr_days:
        q=np.where(days==s)[0]; f=np.where(is_tr&(days!=s))[0]
        if len(f)<50: out[q]=np.nanmean(y[is_tr]); continue
        Xf=Xall[f].copy(); Xq=Xall[q].copy()
        if day_feat:
            Xf=np.hstack([Xf,(days[f]/100.0)[:,None]]); Xq=np.hstack([Xq,(days[q]/100.0)[:,None]])
        if wins:
            lo=np.nanpercentile(Xf,0.5,axis=0); hi=np.nanpercentile(Xf,99.5,axis=0)
            lo=np.where(np.isfinite(lo),lo,0); hi=np.where(np.isfinite(hi),hi,1); bad=hi<=lo; hi[bad]=lo[bad]+1
            Xf=np.clip(Xf,lo,hi); Xq=np.clip(Xq,lo,hi)
        w=np.ones(len(f)) if decay is None else np.exp(-(s-days[f])/decay)
        yf=ytfun(f); mu,sd=std_fit(Xf); Xfs=std_apply(Xf,mu,sd)
        ym=np.average(yf,weights=w); r=(yf-ym)*np.sqrt(w); Xw=Xfs*np.sqrt(w)[:,None]
        wg=np.linalg.solve(Xw.T@Xw+alpha*np.eye(Xw.shape[1]),Xw.T@r)
        out[q]=std_apply(Xq,mu,sd)@wg+ym
    return out
p_pos = loso_yt(X,1000,lambda f: np.sqrt(y[f]))
p_zero= loso_yt(X,1000,lambda f: z[f])
for th in [0.5,0.6,0.7]:
    pred=np.where(p_zero<th,p_pos,0.0)
    print('two-part th=%.1f MAE %.3f late %.3f'%(th,ev(pred,lambda p:p),ev(pred,lambda p:p,late)))
o=loso(X,1000,sq)
for wgt in [0.0,0.25,0.5,0.75,1.0]:
    pred=wgt*np.where(p_zero<0.6,p_pos,0.0)+(1-wgt)*np.clip(invsq(o),0,None)
    print('blend two-part(%.2f)+sqrt %.2f MAE %.3f'%(wgt,1-wgt,ev(pred,lambda p:p)))
print('t',round(time.time()-t0,1))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time
t0=time.time()
df = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='left').reset_index(drop=True)
is_tr = d['future_spend_4w'].notna().vs if False else d['future_spend_4w'].notna().values
cat_cols = [c for c in df.columns if str(df[c].dtype)=='category']
base_drop = set(['household_key','snapshot_day','stack_ridge_log','stack_ridge'])|set(cat_cols)
num_cols = [c for c in df.columns if c not in base_drop]
X = np.hstack([d[num_cols].astype(float).values, pd.get_dummies(d[cat_cols].astype(str), dummy_na=True).values.astype(np.float64)])
days = d['snapshot_day'].values; y=d['future_spend_4w'].values.astype(float)
tr_days = np.array(sorted(set(days[is_tr])))
late = is_tr & (days>=347)

def std_fit(A):
    mu=np.nanmean(A,axis=0); sd=np.nanstd(A,axis=0); sd[~np.isfinite(sd)]=1.0; sd[sd==0]=1.0; return mu,sd
def std_apply(A,mu,sd):
    Z=(A-mu)/sd; return np.where(np.isfinite(Z),Z,0.0)

def loso(Xall, alpha, yf_all, wins=True):
    out=np.full(len(d),np.nan)
    for s in tr_days:
        q=np.where(days==s)[0]; f=np.where(is_tr&(days!=s))[0]
        if len(f)<50: out[q]=np.nanmean(yf_all[is_tr]); continue
        Xf=Xall[f].copy(); Xq=Xall[q].copy()
        if wins:
            lo=np.nanpercentile(Xf,0.5,axis=0); hi=np.nanpercentile(Xf,99.5,axis=0)
            lo=np.where(np.isfinite(lo),lo,0); hi=np.where(np.isfinite(hi),hi,1); bad=hi<=lo; hi[bad]=lo[bad]+1
            Xf=np.clip(Xf,lo,hi); Xq=np.clip(Xq,lo,hi)
        mu,sd=std_fit(Xf); Xfs=std_apply(Xf,mu,sd)
        ym=np.mean(yf_all[f]); r=(yf_all[f]-ym); Xw=Xfs
        wg=np.linalg.solve(Xw.T@Xw+alpha*np.eye(Xw.shape[1]),Xw.T@r)
        out[q]=std_apply(Xq,mu,sd)@wg+ym
    return out
def ev(oof, inv, mask=None):
    pr=np.clip(inv(np.asarray(oof,float)),0,None)
    m = is_tr if mask is None else mask
    return np.mean(np.abs(pr[m]-y[m]))

sq=lambda v:np.sqrt(v)
y_sqrt=np.sqrt(y)
# group by day, no winsorize
print('--- group-by-day ridge, sqrt target, alpha sweep ---')
for a in [100,300,1000,3000]:
    o=loso(X,a,y_sqrt,wins=False)
    print('a=%-5d MAE %.3f late %.3f'%(a,ev(o,invsq),ev(o,invsq,late)))
# feature subsets
groups = {
 'rfm': [c for c in num_cols if any(k in c for k in ['sp','trips','recen','tenur','basket','ew','wk'])],
 'dem': [c for c in num_cols if any(c.startswith(p) for p in ['class','homeowner','kid','has_demo'])],
 'mix': [c for c in num_cols if c.startswith('mix') or 'share' in c],
 'mkt': [c for c in num_cols if any(k in c for k in ['camp','coupon','mailer','display'])],
 'seas':[c for c in num_cols if any(k in c for k in ['block','lag','s336','s364','s392','alltime','avg28'])],
 'dyn': [c for c in num_cols if any(k in c for k in ['d_','mom','ratio','slope','streak','vol'])],
 'style':[c for c in num_cols if any(k in c for k in ['unit','disc','store','hhhi','nstore','trip_time','items','nprod'])],
 'knn': [c for c in num_cols if 'knn' in c],
 'stack':[c for c in num_cols if 'stack' in c],
}
print({k:len(v) for k,v in groups.items()})
Xg = {k: d[v].astype(float).values for k,v in groups.items()}
Xg['dum']=pd.get_dummies(d[cat_cols].astype(str), dummy_na=True).values.astype(np.float64)
for k,Xs in Xg.items():
    o=loso(Xs,300,y_sqrt,wins=False)
    print('%-6s n=%-4d solo MAE %.3f'%(k,Xs.shape[1],ev(o,invsq)))
# forward stepwise-ish: cumulative groups in RFM order
order=['rfm','dyn','style','seas','mix','mkt','dem','knn','stack','dum']
Xc=np.zeros((len(d),0))
for k in order:
    Xc=np.hstack([Xc,Xg[k]]) if Xc.shape[1] else Xg[k].copy()
    o=loso(Xc,300,y_sqrt,wins=False)
    print('cum %-6s n=%-4d MAE %.3f'%(k,Xc.shape[1],ev(o,invsq)))
print('t',round(time.time()-t0,1))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time
t0=time.time()
df = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='left').reset_index(drop=True)
is_tr = d['future_spend_4w'].notna().values
cat_cols = [c for c in df.columns if str(df[c].dtype)=='category']
base_drop = set(['household_key','snapshot_day','stack_ridge_log','stack_ridge'])|set(cat_cols)
num_cols = [c for c in df.columns if c not in base_drop]
X = np.hstack([d[num_cols].astype(float).values, pd.get_dummies(d[cat_cols].astype(str), dummy_na=True).values.astype(np.float64)])
days = d['snapshot_day'].values; y=d['future_spend_4w'].values.astype(float)
tr_days = np.array(sorted(set(days[is_tr])))
late = is_tr & (days>=347)
invsq = lambda p: p**2

def std_fit(A):
    mu=np.nanmean(A,axis=0); sd=np.nanstd(A,axis=0); sd[~np.isfinite(sd)]=1.0; sd[sd==0]=1.0; return mu,sd
def std_apply(A,mu,sd):
    Z=(A-mu)/sd; return np.where(np.isfinite(Z),Z,0.0)

def loso(Xall, alpha, yf_all, wins=True):
    out=np.full(len(d),np.nan)
    for s in tr_days:
        q=np.where(days==s)[0]; f=np.where(is_tr&(days!=s))[0]
        if len(f)<50: out[q]=np.nanmean(yf_all[is_tr]); continue
        Xf=Xall[f].copy(); Xq=Xall[q].copy()
        if wins:
            lo=np.nanpercentile(Xf,0.5,axis=0); hi=np.nanpercentile(Xf,99.5,axis=0)
            lo=np.where(np.isfinite(lo),lo,0); hi=np.where(np.isfinite(hi),hi,1); bad=hi<=lo; hi[bad]=lo[bad]+1
            Xf=np.clip(Xf,lo,hi); Xq=np.clip(Xq,lo,hi)
        mu,sd=std_fit(Xf); Xfs=std_apply(Xf,mu,sd)
        ym=np.mean(yf_all[f]); r=(yf_all[f]-ym); Xw=Xfs
        wg=np.linalg.solve(Xw.T@Xw+alpha*np.eye(Xw.shape[1]),Xw.T@r)
        out[q]=std_apply(Xq,mu,sd)@wg+ym
    return out
def ev(oof, mask=None):
    pr=np.clip(invsq(np.asarray(oof,float)),0,None)
    m = is_tr if mask is None else mask
    return np.mean(np.abs(pr[m]-y[m]))

y_sqrt=np.sqrt(y)
print('--- group-by-day ridge, sqrt target, alpha sweep ---')
for a in [100,300,1000,3000]:
    o=loso(X,a,y_sqrt,wins=False)
    print('a=%-5d MAE %.3f late %.3f'%(a,ev(o),ev(o,late)))
groups = {
 'rfm': [c for c in num_cols if any(k in c for k in ['sp','trips','recen','tenur','basket','ew','wk'])],
 'dem': [c for c in num_cols if any(c.startswith(p) for p in ['class','homeowner','kid','has_demo'])],
 'mix': [c for c in num_cols if c.startswith('mix') or 'share' in c],
 'mkt': [c for c in num_cols if any(k in c for k in ['camp','coupon','mailer','display'])],
 'seas':[c for c in num_cols if any(k in c for k in ['block','lag','s336','s364','s392','alltime','avg28'])],
 'dyn': [c for c in num_cols if any(k in c for k in ['d_','mom','ratio','slope','streak','vol'])],
 'style':[c for c in num_cols if any(k in c for k in ['unit','disc','store','hhi','nstore','trip_time','items','nprod'])],
 'knn': [c for c in num_cols if 'knn' in c],
 'stack':[c for c in num_cols if 'stack' in c],
}
print({k:len(v) for k,v in groups.items()})
Xg = {k: d[v].astype(float).values for k,v in groups.items()}
Xg['dum']=pd.get_dummies(d[cat_cols].astype(str), dummy_na=True).values.astype(np.float64)
print('--- solo groups ---')
for k,Xs in Xg.items():
    o=loso(Xs,300,y_sqrt,wins=False)
    print('%-6s n=%-4d solo MAE %.3f'%(k,Xs.shape[1],ev(o)))
print('--- cumulative groups ---')
Xc=None
for k in ['rfm','dyn','style','seas','mix','mkt','dem','knn','stack','dum']:
    Xc = Xg[k].copy() if Xc is None else np.hstack([Xc,Xg[k]])
    o=loso(Xc,300,y_sqrt,wins=False)
    print('cum %-6s n=%-4d MAE %.3f'%(k,Xc.shape[1],ev(o)))
print('t',round(time.time()-t0,1))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time
t0=time.time()
df = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='left').reset_index(drop=True)
is_tr = d['future_spend_4w'].notna().values
cat_cols = [c for c in df.columns if str(df[c].dtype)=='category']
base_drop = set(['household_key','snapshot_day','stack_ridge_log','stack_ridge'])|set(cat_cols)
num_cols = [c for c in df.columns if c not in base_drop]
X = np.hstack([d[num_cols].astype(float).values, pd.get_dummies(d[cat_cols].astype(str), dummy_na=True).values.astype(np.float64)])
days = d['snapshot_day'].values; y=d['future_spend_4w'].values.astype(float)
tr_days = np.array(sorted(set(days[is_tr])))
late = is_tr & (days>=347)

def std_fit(A):
    mu=np.nanmean(A,axis=0); sd=np.nanstd(A,axis=0); sd[~np.isfinite(sd)]=1.0; sd[sd==0]=1.0; return mu,sd
def std_apply(A,mu,sd):
    Z=(A-mu)/sd; return np.where(np.isfinite(Z),Z,0.0)

def loso(Xall, alpha, yf_all, wins=True):
    out=np.full(len(d),np.nan)
    for s in tr_days:
        q=np.where(days==s)[0]; f=np.where(is_tr&(days!=s))[0]
        if len(f)<50: out[q]=np.nanmean(yf_all[is_tr]); continue
        Xf=Xall[f].copy(); Xq=Xall[q].copy()
        if wins:
            lo=np.nanpercentile(Xf,0.5,axis=0); hi=np.nanpercentile(Xf,99.5,axis=0)
            lo=np.where(np.isfinite(lo),lo,0); hi=np.where(np.isfinite(hi),hi,1); bad=hi<=lo; hi[bad]=lo[bad]+1
            Xf=np.clip(Xf,lo,hi); Xq=np.clip(Xq,lo,hi)
        mu,sd=std_fit(Xf); Xfs=std_apply(Xf,mu,sd)
        ym=np.mean(yf_all[f]); r=yf_all[f]-ym; Xw=Xfs
        wg=np.linalg.solve(Xw.T@Xw+alpha*np.eye(Xw.shape[1]),Xw.T@r)
        out[q]=std_apply(Xq,mu,sd)@wg+ym
    return out
def ev(pred, mask=None):
    pr=np.clip(np.asarray(pred,float),0,None)
    m = is_tr if mask is None else mask
    return np.mean(np.abs(pr[m]-y[m]))

y_sqrt=np.sqrt(y)
sq_oof=loso(X,1000,y_sqrt)          # sqrt-space ridge
raw_oof=loso(X,1000,y)              # raw-space ridge
log_oof=loso(X,1000,np.log1p(y))    # log-space ridge
print('OOF MAE: sqrt %.3f | raw %.3f | log %.3f'%(ev(np.clip(sq_oof,0,None)**2), ev(np.clip(raw_oof,0,None)), ev(np.clip(np.expm1(log_oof),0,None))))
res = np.where(is_tr, y_sqrt-np.clip(sq_oof,0,None), np.nan)
mr={s:np.nanmean(res[days==s]) for s in tr_days}
bs={s:np.mean([v for s2,v in mr.items() if s2!=s]) for s in tr_days}
sq_cal = np.clip(sq_oof,0,None)**2 + np.array([bs.get(s,0.0) for s in days])
print('sqrt+biascal OOF MAE %.3f'%ev(sq_cal))
best=(1e9,None)
for w in [0,0.25,0.5,0.75,1.0]:
    p=w*(np.clip(sq_oof,0,None)**2)+(1-w)*np.clip(raw_oof,0,None)
    m=ev(p)
    if m<best[0]: best=(m,w)
    print('blend sqrt w=%.2f MAE %.3f'%(w,m))
p_mix=(np.clip(sq_oof,0,None)+np.clip(raw_oof,0,None))**2/4
print('avg-in-sqrt-space blend MAE %.3f'%ev(p_mix))
w=best[1]
p=w*(np.clip(sq_oof,0,None)**2)+(1-w)*np.clip(raw_oof,0,None)
print('winner(w=%.2f) late MAE %.3f | cur stack late OOF %.3f'%(w,ev(p,late), ev(np.clip(d['stack_ridge'].values,0,None)[is_tr],late)))
print('t',round(time.time()-t0,1))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time
t0=time.time()
df = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='left').reset_index(drop=True)
is_tr = d['future_spend_4w'].notna().values
cat_cols = [c for c in df.columns if str(df[c].dtype)=='category']
base_drop = set(['household_key','snapshot_day','stack_ridge_log','stack_ridge'])|set(cat_cols)
num_cols = [c for c in df.columns if c not in base_drop]
X = np.hstack([d[num_cols].astype(float).values, pd.get_dummies(d[cat_cols].astype(str), dummy_na=True).values.astype(np.float64)])
days = d['snapshot_day'].values; y=d['future_spend_4w'].values.astype(float)
tr_days = np.array(sorted(set(days[is_tr])))
late = is_tr & (days>=347)

def std_fit(A):
    mu=np.nanmean(A,axis=0); sd=np.nanstd(A,axis=0); sd[~np.isfinite(sd)]=1.0; sd[sd==0]=1.0; return mu,sd
def std_apply(A,mu,sd):
    Z=(A-mu)/sd; return np.where(np.isfinite(Z),Z,0.0)

def loso(Xall, alpha, yf_all, wins=True):
    out=np.full(len(d),np.nan)
    for s in tr_days:
        q=np.where(days==s)[0]; f=np.where(is_tr&(days!=s))[0]
        if len(f)<50: out[q]=np.nanmean(yf_all[is_tr]); continue
        Xf=Xall[f].copy(); Xq=Xall[q].copy()
        if wins:
            lo=np.nanpercentile(Xf,0.5,axis=0); hi=np.nanpercentile(Xf,99.5,axis=0)
            lo=np.where(np.isfinite(lo),lo,0); hi=np.where(np.isfinite(hi),hi,1); bad=hi<=lo; hi[bad]=lo[bad]+1
            Xf=np.clip(Xf,lo,hi); Xq=np.clip(Xq,lo,hi)
        mu,sd=std_fit(Xf); Xfs=std_apply(Xf,mu,sd)
        ym=np.mean(yf_all[f]); r=yf_all[f]-ym
        wg=np.linalg.solve(Xfs.T@Xfs+alpha*np.eye(Xfs.shape[1]),Xfs.T@r)
        out[q]=std_apply(Xq,mu,sd)@wg+ym
    return out
def ev(pred, mask=None):
    pr=np.clip(np.asarray(pred,float),0,None)
    m = is_tr if mask is None else mask
    return np.mean(np.abs(pr[m]-y[m]))

y_sqrt=np.sqrt(y)
sq_oof=loso(X,1000,y_sqrt)
raw_oof=loso(X,1000,y)
log_oof=loso(X,1000,np.log1p(y))
sq2=np.clip(sq_oof,0,None)**2
rawp=np.clip(raw_oof,0,None)
logp=np.clip(np.expm1(log_oof),0,None)
print('members: sqrt2 %.3f | raw %.3f | logexp %.3f | old stack_ridge %.3f'%(
    ev(sq2),ev(rawp),ev(logp),ev(np.clip(d['stack_ridge'].values,0,None))))
best=(1e9,None)
for w1 in [0.5,0.6,0.7,0.8,0.9,1.0]:
    for w2 in [0,0.1,0.2,0.3,0.4]:
        w3=1-w1-w2
        if w3<-1e-9 or w3>0.3: continue
        p=w1*sq2+w2*rawp+w3*logp
        m=ev(p)
        if m<best[0]: best=(m,(w1,w2,w3))
print('best 3-blend', best)
# fine 2-blend
best2=(1e9,None)
for w in np.arange(0.5,1.01,0.05):
    p=w*sq2+(1-w)*rawp; m=ev(p)
    if m<best2[0]: best2=(m,round(w,2))
print('best 2-blend sqrt2+raw', best2)
w=best2[1]
p=w*sq2+(1-w)*rawp
print('2-blend late MAE %.3f | per-day means:'%ev(p,late))
print(pd.DataFrame({'day':days[is_tr],'y':y[is_tr],'p':p[is_tr]}).groupby('day').mean().round(1).T)
print('NaNs in sq2/rawp:', np.isnan(sq2).sum(), np.isnan(rawp).sum())
print('t',round(time.time()-t0,1))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time
t0=time.time()
df = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='left').reset_index(drop=True)
is_tr = d['future_spend_4w'].notna().values
cat_cols = [c for c in df.columns if str(df[c].dtype)=='category']
base_drop = set(['household_key','snapshot_day','stack_ridge_log','stack_ridge'])|set(cat_cols)
num_cols = [c for c in df.columns if c not in base_drop]
X = np.hstack([d[num_cols].astype(float).values, pd.get_dummies(d[cat_cols].astype(str), dummy_na=True).values.astype(np.float64)])
days = d['snapshot_day'].values; y=d['future_spend_4w'].values.astype(float)
tr_days = np.array(sorted(set(days[is_tr])))
late = is_tr & (days>=347)

def std_fit(A):
    mu=np.nanmean(A,axis=0); sd=np.nanstd(A,axis=0); sd[~np.isfinite(sd)]=1.0; sd[sd==0]=1.0; return mu,sd
def std_apply(A,mu,sd):
    Z=(A-mu)/sd; return np.where(np.isfinite(Z),Z,0.0)

def loso(Xall, alpha, yf_all, wins=True):
    out=np.full(len(d),np.nan)
    for s in tr_days:
        q=np.where(days==s)[0]; f=np.where(is_tr&(days!=s))[0]
        if len(f)<50: out[q]=np.nanmean(yf_all[is_tr]); continue
        Xf=Xall[f].copy(); Xq=Xall[q].copy()
        if wins:
            lo=np.nanpercentile(Xf,0.5,axis=0); hi=np.nanpercentile(Xf,99.5,axis=0)
            lo=np.where(np.isfinite(lo),lo,0); hi=np.where(np.isfinite(hi),hi,1); bad=hi<=lo; hi[bad]=lo[bad]+1
            Xf=np.clip(Xf,lo,hi); Xq=np.clip(Xq,lo,hi)
        mu,sd=std_fit(Xf); Xfs=std_apply(Xf,mu,sd)
        ym=np.mean(yf_all[f]); r=yf_all[f]-ym
        wg=np.linalg.solve(Xfs.T@Xfs+alpha*np.eye(Xfs.shape[1]),Xfs.T@r)
        out[q]=std_apply(Xq,mu,sd)@wg+ym
    return out
def ev(pred, mask=None):
    pr=np.clip(np.asarray(pred,float),0,None)
    m = is_tr if mask is None else mask
    return np.mean(np.abs(pr[m]-y[m]))

y_sqrt=np.sqrt(y)
sq_oof=loso(X,1000,y_sqrt)
raw_oof=loso(X,1000,y)
sq2=np.clip(sq_oof,0,None)**2; rawp=np.clip(raw_oof,0,None)
# per-snapshot weighted blend: choose w per snapshot via leave-self-out of snapshot residuals
snap_ids = sorted(set(days[is_tr]))
ws = np.arange(0.5,1.01,0.05)
best_w={}
for s in snap_ids:
    msk = is_tr&(days!=s)
    scores=[]
    for w in ws:
        p=w*sq2+(1-w)*rawp
        scores.append(np.mean(np.abs(p[msk]-y[msk])))
    best_w[s]=ws[int(np.argmin(scores))]
print('per-snap best w:', {s:round(v,2) for s,v in best_w.items()})
p_adapt = np.array([best_w.get(s,0.75)*sq2[i]+(1-best_w.get(s,0.75))*rawp[i] for i,s in enumerate(days)])
print('adaptive-w blend OOF MAE %.3f | late %.3f'%(ev(p_adapt),ev(p_adapt,late)))
# also try: residual-based per-snapshot multiplicative level calibration in raw space
lvl={s:y[is_tr&(days==s)].mean() for s in snap_ids}
lvo={s:np.mean([v for s2,v in lvl.items() if s2!=s]) for s in snap_ids}
p_lvl = rawp*np.array([lvo.get(s,1.0) for s in days])/np.array([lvl.get(s,1.0) for s in days])
print('level-calibrated raw MAE %.3f'%ev(p_lvl))
# sqrt-space level calibration
sqc = sq2*np.array([lvo.get(s,1.0) for s in days])/np.array([lvl.get(s,1.0) for s in days])
print('level-calibrated sqrt2 MAE %.3f'%ev(sqc))
# sqrt + small raw: fine grid
for w in [0.7,0.75,0.8]:
    p=w*sq2+(1-w)*rawp
    print('w=%.2f MAE %.3f late %.3f'%(w,ev(p),ev(p,late)))
print('t',round(time.time()-t0,1))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time
t0=time.time()
df = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='left').reset_index(drop=True)
is_tr = d['future_spend_4w'].notna().values
cat_cols = [c for c in df.columns if str(df[c].dtype)=='category']
base_drop = set(['household_key','snapshot_day','stack_ridge_log','stack_ridge'])|set(cat_cols)
num_cols = [c for c in df.columns if c not in base_drop]
X = np.hstack([d[num_cols].astype(float).values, pd.get_dummies(d[cat_cols].astype(str), dummy_na=True).values.astype(np.float64)])
days = d['snapshot_day'].values; y=d['future_spend_4w'].values.astype(float)
tr_days = sorted(set(days[is_tr])); val_days = sorted(set(days[~is_tr]))
print('train days',tr_days,'| val days',val_days)

def fit_pred(Xall,f,q,yf_all,alpha):
    Xf=Xall[f].copy(); Xq=Xall[q].copy()
    lo=np.nanpercentile(Xf,0.5,axis=0); hi=np.nanpercentile(Xf,99.5,axis=0)
    lo=np.where(np.isfinite(lo),lo,0.0); hi=np.where(np.isfinite(hi),hi,1.0)
    bad=hi<=lo; hi[bad]=lo[bad]+1.0
    Xf=np.clip(Xf,lo,hi); Xq=np.clip(Xq,lo,hi)
    mu=np.nanmean(Xf,axis=0); sd=np.nanstd(Xf,axis=0)
    sd[~np.isfinite(sd)]=1.0; sd[sd==0]=1.0
    Zf=(Xf-mu)/sd; Zf=np.where(np.isfinite(Zf),Zf,0.0)
    Zq=(Xq-mu)/sd; Zq=np.where(np.isfinite(Zq),Zq,0.0)
    yf=yf_all[f]; ym=np.mean(yf)
    wg=np.linalg.solve(Zf.T@Zf+alpha*np.eye(Zf.shape[1]),Zf.T@(yf-ym))
    return Zq@wg+ym

def stack_preds(yf_all, alpha=300.0):
    out=np.full(len(d),np.nan)
    for s in tr_days:   # leave-one-snapshot-out on train rows
        q=np.where(days==s)[0]; f=np.where(is_tr&(days!=s))[0]
        out[q]=fit_pred(X,f,q,yf_all,alpha)
    for s in val_days:  # validation rows: fit on ALL train rows
        q=np.where(days==s)[0]; f=np.where(is_tr)[0]
        out[q]=fit_pred(X,f,q,yf_all,alpha)
    return out

sq_oof  = stack_preds(np.sqrt(y))
raw_oof = stack_preds(y)
stack_sqrt  = sq_oof
stack_sqrt2 = np.clip(sq_oof,0,None)**2
stack_raw   = np.clip(raw_oof,0,None)
stack_blend = 0.75*stack_sqrt2 + 0.25*stack_raw
def ev(pred, mask=None):
    pr=np.clip(np.asarray(pred,float),0,None); m=is_tr if mask is None else mask
    return np.mean(np.abs(pr[m]-y[m]))
late=is_tr&(days>=347)
print('OOF train MAE: sqrt2 %.3f | raw %.3f | blend %.3f | old log-stack %.3f'%(
    ev(stack_sqrt2),ev(stack_raw),ev(stack_blend),ev(np.clip(d['stack_ridge'].values,0,None))))
print('late MAE: sqrt2 %.3f | blend %.3f'%(ev(stack_sqrt2,late),ev(stack_blend,late)))
print('NaN check:', np.isnan(stack_blend).sum(), np.isnan(stack_sqrt2).sum(), np.isnan(stack_raw).sum())
out = d.drop(columns=['future_spend_4w']).copy()
out['stack_sqrt']=stack_sqrt; out['stack_sqrt2']=stack_sqrt2
out['stack_raw']=stack_raw;   out['stack_blend']=stack_blend
p = agent_api.save_table(out,'e020_sqrtstack.parquet')
print('saved:',p,'shape',out.shape,'t',round(time.time()-t0,1))
