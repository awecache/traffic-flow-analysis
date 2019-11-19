import numpy as np
import pandas as pd

raw_df=pd.read_csv('https://aisgaiap.blob.core.windows.net/aiap5-assessment-data/traffic_data.csv')
df=raw_df.copy()

df['date_time'] = pd.to_datetime(df['date_time'],infer_datetime_format=True)
df.set_index('date_time', inplace=True)

df_selected=df[:'2013-10-26']

df_selected=pd.get_dummies(df_selected,drop_first=True)

df_processed=df_selected[['temp', 'weather_main_Haze', 'weather_description_haze','weather_description_scattered clouds','traffic_volume']]
df_processed.to_csv('data_cleaned.csv')
