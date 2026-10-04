import agent_api, numpy as np, pandas as pd

def fn(view, snapshot_day):
    base = agent_api.load_saved('e018_basestab.parquet')
    df = base[base.snapshot_day == snapshot_day].copy()
    mac = agent_api.load_saved('macro.parquet').reset_index()
    mday = mac.groupby('snapshot_day').first()
    row = mday.loc[snapshot_day] if snapshot_day in mday.index else None
    def g(c):
        return float(row[c]) if row is not None and pd.notna(row[c]) else np.nan
    df['m28']        = g('macro_spend28')/1e5
    df['m28_p']      = g('macro_spend_p28')/1e5
    df['m28_ly']     = g('macro_spend28_ly')/1e5
    df['m112']       = g('macro_spend112')/1e5
    df['m_hh28']     = g('macro_hh28')/1e3
    df['m_per_hh28'] = g('macro_spend_per_hh28')
    df['m_per_hh28_ly'] = g('macro_spend_per_hh28_ly')
    df['m_growth']   = g('macro_growth')
    df['m_wk3']      = g('macro_wk_ratio_last3')
    df['m_ratio_ly'] = g('macro_ratio_ly')
    wk = (snapshot_day + 14 + 8) // 7
    df['fut_wk'] = float(wk)
    df['fut_wk_sin'] = np.sin(2*np.pi*wk/52.18)
    df['fut_wk_cos'] = np.cos(2*np.pi*wk/52.18)
    cols = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    return df.set_index('household_key')[cols]

out = agent_api.build_features(fn)
print(out.shape)
print(out.groupby('snapshot_day')[['m28','m_growth','m_wk3','m_per_hh28','fut_wk']].first())
path = agent_api.save_table(out, 'e019_macroctx.parquet')
print(path)