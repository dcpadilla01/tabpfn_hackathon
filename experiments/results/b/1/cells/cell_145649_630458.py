import pandas as pd, numpy as np, agent_api
pd.set_option('display.width',250)
train_days = agent_api.snapshot_days()['train']
tt = agent_api.train_targets()

def feats_for(day):
    v = agent_api.snapshot(as_of_day=day)
    t = v.transactions
    hh = v.households
    g = t.groupby('household_key')
    out = pd.DataFrame(index=hh)
    for k in range(1,8):
        lo = day-28*k
        out[f'w{k}'] = t[(t.day>lo)&(t.day<=lo+28)].groupby('household_key').sales_value.sum()
    tmp = t.assign(age=day-t.day)
    for hl in [14,28,56]:
        w = 0.5**(tmp.age/hl)
        out[f'ew{hl}'] = (tmp.sales_value*w).groupby(tmp.household_key).sum()
    out['s84'] = t[t.day>day-84].groupby('household_key').sales_value.sum()
    out['s182'] = t[t.day>day-182].groupby('household_key').sales_value.sum()
    out['sall'] = g.sales_value.sum()
    out['dsl'] = day - g.day.max()
    out['tenure'] = day - g.day.min()
    ud = t.groupby('household_key').day.apply(lambda s: np.sort(s.unique()))
    gaps = ud.apply(lambda a: np.diff(a) if len(a)>1 else np.array([]))
    out['gap_med'] = gaps.apply(lambda a: np.median(a) if len(a)>0 else np.nan)
    g182 = gaps.apply(lambda a: a[a<=182] if len(a)>0 else np.array([]))
    out['gap_max182'] = g182.apply(lambda a: a.max() if len(a)>0 else np.nan)
    out['n_gap21_182'] = g182.apply(lambda a: (a>21).sum())
    out['nb28'] = t[t.day>day-28].groupby('household_key').basket_id.nunique()
    out['nactive28'] = t[t.day>day-28].groupby('household_key').day.nunique()
    return out

frames=[]
for day in train_days:
    f = feats_for(day); f['snapshot_day']=day
    frames.append(f.reset_index().rename(columns={'index':'household_key'}))
F = pd.concat(frames, ignore_index=True).merge(tt, on=['household_key','snapshot_day'])
y = F.future_spend_4w.values.astype(float)

def mae_scale(p):
    p=np.asarray(p,dtype=float); ok=np.isfinite(p)&(p>0)
    a = np.median(y[ok]/p[ok])
    return np.abs(a*p[ok]-y[ok]).mean(), a

med3w = F[['w1','w2','w3']].median(axis=1).values
cands = {
 's28(w1)': F.w1.values,
 's84': F.s84.values,
 'mean3w': (F.w1+F.w2+F.w3).values/3,
 'med3w': med3w,
 'mean6w': F[[f'w{i}' for i in range(1,7)]].mean(axis=1).values,
 'ew14': F.ew14.values, 'ew28': F.ew28.values, 'ew56': F.ew56.values,
 'rate182': F.s182.values/182*28,
 'blend_a(0.55w1+0.25w2+0.2w3)': 0.55*F.w1.values+0.25*F.w2.values+0.20*F.w3.values,
 'blend_b(0.4w1+0.2w2+0.4med3w)': 0.4*F.w1.values+0.2*F.w2.values+0.4*med3w,
 'blend_c(0.5ew28+0.5w1)': 0.5*F.ew28.values+0.5*F.w1.values,
}
print('predictor -> scaled MAE (scale a)')
for k,v in cands.items():
    r = mae_scale(v); print(f'{k:30s} {r[0]:7.2f}  (a={r[1]:.2f})')

b = pd.cut(F.dsl,[-1,7,14,21,28,56,10000])
print('\nby days_since_last:'); print(F.groupby(b,observed=True).apply(lambda d: pd.Series({'n':len(d),'zerorate':(d.future_spend_4w==0).mean(),'y_med':d.future_spend_4w.median()})))
print('\nzero rate by n_gap21_182:'); print(F.groupby(F.n_gap21_182).apply(lambda d: pd.Series({'n':len(d),'zerorate':(d.future_spend_4w==0).mean()})))
print('\ncorr with y:'); print(F.drop(columns=['household_key','snapshot_day']).corrwith(F.future_spend_4w).sort_values(key=lambda s:s.abs(),ascending=False).round(3))
