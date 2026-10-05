import agent_api
import pandas as pd

e5 = agent_api.load_saved('e005_trend_season.parquet')
new = agent_api.load_saved('e006_zero_inflation') if False else None
