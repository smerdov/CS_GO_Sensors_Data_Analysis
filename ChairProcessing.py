import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from utils import normalize_MPU9250_data, split_df, get_intervals_from_moments
from ChairAnalyzer import ChairAnalyser

plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'
# data_path = 'data/CSV'
# data_path = 'Anonimised Data/Data'

# sessions_dict = joblib.load('data/sessions_dict')
sessions_dict = joblib.load('data/sessions_dict')
gamedata_dict = joblib.load('data/gamedata_dict')

def get_chair_features(df_chair, session_id):
    chair_analyser = ChairAnalyser(df_chair, pic_prefix, 0.01, name=session_id)  # + f'_{n_chunk}')
    nonstationary_values_portion = chair_analyser.get_nonstationary_values_portion()
    lean_back_portion = chair_analyser.get_lean_back_portion()
    oscillations = chair_analyser.get_oscillation_intensity()

    chair_features = pd.concat([nonstationary_values_portion, lean_back_portion, oscillations])

    return chair_features

chair_features_list = []

# ##### Testing zone
#
# session_id = 10
# df_chair = sessions_dict[session_id]['schairlog']
# # df_chair['time'] = pd.to_datetime(df_chair['time']).apply(lambda x: x.timestamp())
#
# chair_analyser = ChairAnalyser(df_chair, pic_prefix=pic_prefix, measurement_interval=0.01, name=session_id)
#
# shootout_times_start_end = gamedata_dict[session_id]['shootout_times_start_end']
#
# shootouts_dict = {
#     'label': 'shootouts',
#     'intervals_list': shootout_times_start_end,
# }
#
# mask_dicts_list = [shootouts_dict]
#
# chair_analyser.plot_measurements_timeline(sensors=['acc', 'gyro'], mask_dicts_list=mask_dicts_list)
#
#
#
# get_chair_features(df_chair, session_id)
#
#
# #####



for session_id, session_data_dict in sessions_dict.items():
    if 'schairlog' in session_data_dict:
        df_chair = session_data_dict['schairlog']
    else:
        continue

    ### Looks like it is already splitted to sessions
    # ### Each chair log is splitted to small sessions
    # chunk_lenght = 100 * 600
    # df_chunks_list = split_df(df_chair, n_chunks=3, chunk_lenght=chunk_lenght)
    # # print(len(df_chunks_list))
    # # chair_analyser = ChairAnalyser(df_chair, 0.01, pic_prefix, name=session_id)  # + f'_{n_chunk}')
    # # chair_analyser.plot_measurements_timeline(sensors=('acc', 'gyro'), plot_suptitle=False, fontsize=22)
    #
    # for n_chunk, df_chunk in enumerate(df_chunks_list):
    #     chair_features = get_chair_features(df_chunk, session_id)
    #     chair_features_list.append(chair_features)

    moments_kills = gamedata_dict[session_id]['times_kills']
    moments_death = gamedata_dict[session_id]['times_is_killed']
    duration = 1

    intervals_shootout = gamedata_dict[session_id]['shootout_times_start_end']
    intervals_kills = get_intervals_from_moments(moments_kills, interval_start=-duration, interval_end=duration)
    intervals_death = get_intervals_from_moments(moments_death, interval_start=-duration, interval_end=duration)

    intervals_labels = [  # Actually classes should be used
        [intervals_shootout, 'shootouts', 'blue'],  # it was orange
        [intervals_kills, 'kills', 'green'],
        [intervals_death, 'deaths', 'red'],
    ]

    intervals_dicts_list = []

    for intervals, label, color in intervals_labels:
        interval_dict = {
            'label': label,
            'intervals_list': intervals,
            'color': color,
        }
        intervals_dicts_list.append(interval_dict)

    chair_analyser = ChairAnalyser(df_chair, pic_prefix=pic_prefix, measurement_interval=0.01, name=session_id)
    chair_analyser.plot_measurements_timeline(sensors=['acc', 'gyro'], intervals_dicts_list=intervals_dicts_list)  # TODO: add kill/death events

    chair_features = get_chair_features(df_chair, session_id)
    chair_features_list.append(chair_features)




df_chair_features = pd.DataFrame(chair_features_list)
df_chair_features.reset_index(inplace=True)
df_chair_features.rename(columns={'index': 'session_id'}, inplace=True)

df_chair_features.to_csv('data/chair_features.csv', index=False)