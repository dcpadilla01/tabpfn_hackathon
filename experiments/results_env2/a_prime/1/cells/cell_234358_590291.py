import pandas as pd
def probe(view, sd):
    msg = ''
    try:
        e = agent_api.load_saved('e012_style.parquet')
        msg = 'loaded ok'
    except Exception as ex:
        msg = f"{type(ex).__name__}: {ex}"
    try:
        has_attr = hasattr(agent_api, 'load_saved')
    except Exception:
        has_attr = '?'
    return pd.DataFrame({'msg': [msg], 'has_attr': [str(has_attr)]}, index=pd.Index(view.households))
res = agent_api.build_features(probe)
print(res.msg.value_counts().to_dict(), res.has_attr.value_counts().to_dict())
