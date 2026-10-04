import numpy as np, pandas as pd

def fn_dm(view, day):
    hh = pd.Index(view.households)
    w = int(view.week)
    tx = view.transactions
    rec = tx.loc[tx['day'] > day-28, ['household_key','product_id','store_id','sales_value']].copy()
    dm = view.display_mailer.copy()
    for c in ['display','mailer']:
        dm[c] = pd.to_numeric(dm[c], errors='coerce')
    out = pd.DataFrame(index=hh)
    def add(name, wlo, whi, col):
        win = dm[(dm['week_no'] >= wlo) & (dm['week_no'] <= whi)]
        if len(win) == 0:
            out[name] = np.nan; return
        ps = win.groupby(['product_id','store_id'], observed=True)[col].mean().rename('v').reset_index()
        m = rec.merge(ps, on=['product_id','store_id'], how='left')
        pm = win.groupby('product_id', observed=True)[col].mean().rename('pv').reset_index()
        m = m.merge(pm, on='product_id', how='left')
        m['v'] = m['v'].fillna(m['pv'])
        num = (m['v']*m['sales_value']).groupby(m['household_key'], observed=True).sum()
        den = m['sales_value'].groupby(m['household_key'], observed=True).sum()
        out[name] = (num/den.where(den>0)).reindex(hh)
    add('disp28',   w-4,  w-1, 'display')
    add('mail28',   w-4,  w-1, 'mailer')
    add('disp84',   w-12, w-1, 'display')
    add('disp_cur', w-1,  w,   'display')
    add('mail_cur', w-1,  w,   'mailer')
    add('disp_fly', w-48, w-47,'display')
    return out

new = agent_api.build_features(fn_dm)
print('new rows:', len(new), 'cols:', list(new.columns))
print(new.groupby('snapshot_day')[['disp28','disp_cur','disp_fly']].mean().round(4))

base = agent_api.load_saved('e009_macro.parquet')
m = base.merge(new, on=['household_key','snapshot_day'], how='left', validate='one_to_one')
assert len(m) == len(base)
print('merged:', m.shape, 'NaN rates:')
print(m[['disp28','mail28','disp84','disp_cur','mail_cur','disp_fly']].isna().mean().round(3))

mf = np.where(m['usual13'].values > 0, 1.0 + m['usual13_mfs'].values/m['usual13'].values, 1.0)
m['act_usual']      = m['p13_4']*m['usual13_4']
m['act_usual6']     = m['p6_4']*m['usual6_4']
m['act_e6']         = m['p6']*m['e6']
m['act_usual_seas'] = m['act_usual']*mf
m['e6_seas']        = m['e6']*mf
m['e13_seas']       = m['e13']*mf
m['log_act_usual']  = np.log1p(m['act_usual'])
m['act_persist']    = m['p13_4'] - m['p13']
m['us_stab']        = m['usual13_4']/(m['usual13']+1.0)
m['hh_id']          = 'h' + m['household_key'].astype(str)

disp_cols = ['disp28','mail28','disp84','disp_cur','mail_cur','disp_fly']
comp_cols = ['act_usual','act_usual6','act_e6','act_usual_seas','e6_seas','e13_seas',
             'log_act_usual','act_persist','us_stab','hh_id']
key = ['household_key','snapshot_day']
t010 = m[key + [c for c in base.columns if c not in key] + comp_cols]
t011 = m[key + [c for c in base.columns if c not in key] + disp_cols]
t012 = m

p1 = agent_api.save_table(t010, 'e010_composite.parquet')
p2 = agent_api.save_table(t011, 'e011_display.parquet')
p3 = agent_api.save_table(t012, 'e012_full.parquet')
print('saved:', p1, p2, p3)
print('t010 shape', t010.shape, 't011 shape', t011.shape, 't012 shape', t012.shape)
print(m[comp_cols[:-1]].describe().T[['mean','std']].round(3).to_string())
