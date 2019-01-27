import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from utils import normalize_MPU9250_data, split_df
from ChairAnalyzer import ChairAnalyser
from datetime import datetime

plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'
data_path = 'Anonimised Data/Data'
processed_data_path = 'data/players_data_processed'

# data_dict = joblib.load('data/data_dict')
sessions_dict = joblib.load('data/sessions_dict')
gamedata_dict = joblib.load('data/gamedata_dict')

common_keys = set(sessions_dict.keys()) & set(gamedata_dict.keys())

game_events_features_dict = {}

for session_id in common_keys:
    if 'schairlog' not in sessions_dict[session_id]:
        continue

    schairlog = sessions_dict[session_id]['schairlog'].copy()
    times_is_killed = gamedata_dict[session_id]['times_is_killed']
    times_kills = gamedata_dict[session_id]['times_kills']

    # schairlog['time'] = pd.to_datetime(schairlog['time']).apply(lambda x: x.timestamp())
    # times_is_killed = [x.timestamp() for x in pd.to_datetime(times_is_killed)]
    # times_kills = [x.timestamp() for x in pd.to_datetime(times_kills)]

    # time_start = 0
    # time_end = 5
    duration = 3
    time_start_end_list = [(0, duration), (-duration, 0), (-duration, duration)]

    sensors_list = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    # for time_kills in times_kills:
    event_reactions_dict = {}

    for times_list, times_name in zip([times_is_killed, times_kills], ['times_is_killed', 'times_kills']):
        for time_start, time_end in time_start_end_list:
            name = f"{times_name}_{time_start}_{time_end}"
            std_list = []

            for time_event in times_list:
                timediff = schairlog['time'] - time_event
                mask = (time_start <= timediff) & (timediff < time_end)
                std = schairlog.loc[mask, sensors_list].std()  # TODO: consider other sensors

                std_list.append(std)

            reaction_stats = pd.DataFrame(std_list).median()  # It should be compared with mean std on the chair
            event_reactions_dict[name] = reaction_stats

    df_event_reactions = pd.DataFrame(event_reactions_dict )  #.round(4).values

    features_dict = {}

    for sensor_name in df_event_reactions.index:
        for column in df_event_reactions.columns:
            feature_name = f"{sensor_name}__{column}"
            features_dict[feature_name] = df_event_reactions.loc[sensor_name, column]

    game_events_features_dict[session_id] = features_dict



df_game_events_features = pd.DataFrame(game_events_features_dict).T
df_game_events_features.reset_index(inplace=True)
df_game_events_features.rename(columns={'index': 'session_id'}, inplace=True)
df_game_events_features.to_csv('data/game_events_features.csv', index=False)





# schairlog['acc_x'].std()
#
#
# schairlog['time'] = pd.to_datetime(schairlog['time'])
# schairlog_ = schairlog.set_index(['time'])
#
#
# index_sample = schairlog_.index[5]
#
# schairlog_.loc[schairlog_.index < index_sample, :]
#
#
# schairlog['time'].iloc[0]
#
# data_dict.keys()
# gamedata_dict.keys()









