
import pandas as pd

def add_calendar_time(df: pd.DataFrame):
    # expects Year, Month, DayofMonth or Day fields
    if 'DayOfMonth' in df.columns:
        day = df['DayOfMonth']
    elif 'Day' in df.columns:
        day = df['Day']
    else:
        day = 1
    year = df.get('Year', 2015)
    month = df.get('Month', 1)
    dt = pd.to_datetime(dict(year=year, month=month, day=day), errors='coerce')
    df = df.copy()
    df['dow'] = dt.dt.dayofweek
    df['dom'] = dt.dt.day
    df['month'] = dt.dt.month
    df['is_weekend'] = df['dow'].isin([5,6]).astype(int)
    return df
