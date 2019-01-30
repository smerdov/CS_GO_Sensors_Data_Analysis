import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from utils import normalize_MPU9250_data, split_df, get_intervals_from_moments, EventIntervals
from GeneralAnalyser import GeneralAnalyser, plot_measurements

# plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'
# data_path = 'data/CSV'
# data_path = 'Anonimised Data/Data'

# sessions_dict = joblib.load('data/sessions_dict')
sessions_dict = joblib.load('data/sessions_dict')
gamedata_dict = joblib.load('data/gamedata_dict')

sensors_columns_dict = {
    'hrm': ['hrm'],
    'envibox': ['als', 'mic', 'humidity', 'temperature', 'co2'],
    'datalog': ['hrm2', 'resistance', 'muscle_activity']
}

sensors_list = list(sensors_columns_dict.keys())
sensors_columns_list = []

for sensor in sensors_columns_dict:
    for column in sensors_columns_dict[sensor]:
        sensors_columns_list.append([sensor, column])


for session_id, session_data_dict in sessions_dict.items():
    df_dict = {}

    if not set(sensors_list).issubset(set(session_data_dict.keys())):
        continue

    moments_kills = gamedata_dict[session_id]['times_kills']
    moments_death = gamedata_dict[session_id]['times_is_killed']
    duration = 1

    intervals_shootout = gamedata_dict[session_id]['shootout_times_start_end']
    intervals_kills = get_intervals_from_moments(moments_kills, interval_start=-duration, interval_end=duration)
    intervals_death = get_intervals_from_moments(moments_death, interval_start=-duration, interval_end=duration)

    event_intervals_shootout = EventIntervals(intervals_list=intervals_shootout, label='shootouts', color='blue')
    event_intervals_kills = EventIntervals(intervals_list=intervals_kills, label='kills', color='green')
    event_intervals_death = EventIntervals(intervals_list=intervals_death, label='deaths', color='red')

    events_intervals_list = [event_intervals_shootout, event_intervals_kills, event_intervals_death]

    analyser_column_pairs_list = []

    for sensor_name in sensors_columns_dict:
        df = session_data_dict[sensor_name]
        analyser = GeneralAnalyser(df, pic_prefix=pic_prefix, sensor_name=sensor_name, session_id=session_id)
        for column in sensors_columns_dict[sensor_name]:
            analyser_column_pairs_list.append([analyser, column])

    plot_measurements(
        analyser_column_pairs_list=analyser_column_pairs_list,
        pic_prefix=pic_prefix,
        session_id=session_id,
        event_intervals_list=events_intervals_list,
        n_rows=3,  # TODO: automatically adjust number of rows and cols
        n_cols=3,
        figsize=(21, 15),
        plot_suptitle=True,
    )
    # general_analyser.plot_measurements_timeline(column_name=sensor_name, intervals_dicts_list=intervals_dicts_list, alpha=0.9)




