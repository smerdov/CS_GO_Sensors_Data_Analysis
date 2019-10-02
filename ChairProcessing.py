import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from utils import normalize_MPU9250_data, split_df, get_intervals_from_moments, EventIntervals
import ChairAnalyser
from ChairAnalyser import ChairAnalyser
from GeneralAnalyser import plot_measurements, plot_measurements_pairwise\
# from GeneralAnalyser import plot_measurements_iop
import itertools
from sklearn.preprocessing import StandardScaler
import argparse

# plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'


# data_path = 'data/CSV'
# data_path = 'Anonimised Data/Data'

import sys
print(sys.argv)
del sys.argv[1:]

parser = argparse.ArgumentParser()
parser.add_argument('--interval', default=1, type=float)
parser.add_argument('--shift', default=1, type=int)  # -1 --- before, 0 - at the moment, 1 - after
parser.add_argument('--visualisation', default=False, type=int)
parser.add_argument('--verbose', default=True, type=int)
parser.add_argument('--std_mode', default=True, type=int)
parser.add_argument('--iop', default=False, type=int)
parser.add_argument('--reaction_multiplier', default=5, type=float)
# parser.add_argument('--max_sessions_per_player', default=3, type=int)
args = parser.parse_args()
interval = args.interval
shift = args.shift
visualisation = args.visualisation
verbose = args.verbose
std_mode = args.std_mode
reaction_multiplier = args.reaction_multiplier
iop = args.iop

# if iop:
#     plot_measurements = plot_measurements_iop

interval_start = interval * (shift / 2)
interval_end = interval * (shift / 2 + 1)


# TODO: extract "online" features

sessions_dict = joblib.load('data/sessions_dict')
gamedata_dict = joblib.load('data/gamedata_dict')

# def get_chair_features(df_chair, session_id):
#     chair_analyser = ChairAnalyser(
#         df=df_chair,
#         pic_prefix=pic_prefix,
#         sensor_name='chair',
#         session_id=session_id,
#         # measurement_interval=0.01,
#     )
#     nonstationary_values_portion = chair_analyser.get_nonstationary_values_portion()
#     lean_back_portion = chair_analyser.get_lean_back_portion()
#     oscillations = chair_analyser.get_oscillation_intensity()
#
#     chair_features = pd.concat([nonstationary_values_portion, lean_back_portion, oscillations])
#     chair_features.name = session_id
#
#     return chair_features

chair_features_list = []

# ##### Testing zone
#
# session_id = 15
# df_chair = sessions_dict[session_id]['schairlog']
# get_chair_features(df_chair, session_id)

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
# End Testing zone
# #####

### Visualisation params
# if visualisation:

sensors_columns_dict = {
    # 'schairlog': ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
    'schairlog': ['acc_x', 'gyro_x', 'acc_y', 'gyro_y', 'acc_z', 'gyro_z'],
}
print(std_mode)
if std_mode:
    # sensors_columns_dict['schairlog'] = [value + f'_std_{int(interval * 1000)}ms' for value in sensors_columns_dict['schairlog']]
    sensors_columns_dict['schairlog'] = [value + f'_std' for value in sensors_columns_dict['schairlog']]

sensors_list = list(sensors_columns_dict.keys())
# sensors_columns_list = []
# for sensor in sensors_columns_dict:
#     for column in sensors_columns_dict[sensor]:
#         sensors_columns_list.append([sensor, column])

# session_id = 15
# session_data_dict = sessions_dict[session_id]
#
# import importlib
# del ChairAnalyser
# import ChairAnalyser
# importlib.reload(ChairAnalyser)
# from ChairAnalyser import ChairAnalyser


for session_id, session_data_dict in sessions_dict.items():
    # df_dict = {}

    if not set(sensors_list).issubset(set(session_data_dict.keys())):
        continue

    if verbose:
        print(f'processing session_id {session_id}')

    if session_id in gamedata_dict:
        moments_kills = gamedata_dict[session_id]['times_kills']
        moments_death = gamedata_dict[session_id]['times_is_killed']
        # duration = 1

        print('Calculating intervals')
        intervals_shootout = gamedata_dict[session_id]['shootout_times_start_end']
        # # intervals_kills = get_intervals_from_moments(moments_kills, interval_start=-duration, interval_end=duration)
        # # intervals_death = get_intervals_from_moments(moments_death, interval_start=-duration, interval_end=duration)
        # intervals_kills = get_intervals_from_moments(moments_kills, interval_start=0, interval_end=2*duration)
        # intervals_death = get_intervals_from_moments(moments_death, interval_start=0, interval_end=2*duration)
        intervals_kills = get_intervals_from_moments(moments_kills, interval_start=interval_start, interval_end=interval_end)
        intervals_death = get_intervals_from_moments(moments_death, interval_start=interval_start, interval_end=interval_end)

        event_intervals_shootout = EventIntervals(intervals_list=intervals_shootout, label='shootout', color='blue')
        event_intervals_kills = EventIntervals(intervals_list=intervals_kills, label='kill', color='green')
        event_intervals_death = EventIntervals(intervals_list=intervals_death, label='death', color='red')

        events_intervals_list = [event_intervals_shootout, event_intervals_death, event_intervals_kills]
    else:
        events_intervals_list = None

    print('Extracting features')
    sensor_name = 'schairlog'
    for sensor_name in sensors_columns_dict:
        df = session_data_dict[sensor_name].copy()

        if sensor_name == 'schairlog':
            chair_analyser = ChairAnalyser(
                df,
                pic_prefix=pic_prefix,
                sensor_name='Chair',  # Manual assignment
                session_id=session_id,
                events_intervals_list=events_intervals_list,
                interval=interval,
                reaction_multiplier=reaction_multiplier,
            )
            chair_features = chair_analyser.get_features()

            # chair_features = get_chair_features(df, session_id)  # TMP
            chair_features_list.append(chair_features)

    if (not visualisation) or (events_intervals_list is None):
        continue

    analyser_column_pairs_list = []

    for sensor_name in sensors_columns_dict:
        df = session_data_dict[sensor_name].copy()

        # if sensor_name == 'schairlog':
        #     chair_features = get_chair_features(df, session_id)  # TMP
        #     chair_features_list.append(chair_features)

        # ss = StandardScaler()
        # # df.values = ss.fit_transform(df.values)
        # df.loc[:, sensors_columns_dict[sensor_name]] = ss.fit_transform(df.loc[:, sensors_columns_dict[sensor_name]])

        ###  WARNING: it is CUSTOM PART
        if sensor_name == 'schairlog':
            chair_analyser = ChairAnalyser(
                df,
                pic_prefix=pic_prefix,
                sensor_name='Chair',  # Manual assignment
                session_id=session_id,
                events_intervals_list=events_intervals_list,
                interval=interval,
                reaction_multiplier=reaction_multiplier,
            )
            # chair_analyser.get_floating_features()  # Need to be refactored
            chair_analyser._append_floating_features(interval=interval)

            for column in sensors_columns_dict[sensor_name]:
                analyser_column_pairs_list.append([chair_analyser, column])

    # print(chair_analyser.df.columns)
    plot_measurements(
        analyser_column_pairs_list=analyser_column_pairs_list,
        pic_prefix=pic_prefix,
        session_id=session_id,
        event_intervals_list=events_intervals_list,
        figsize=(30, 20),
        plot_suptitle=True,
        alpha=1,
        alpha_background=0.4,
        sharex=True,
        fontsize=30,
    )


    ### CODE BELOW IS PROBABLY OK AND USEFUL FOR PAIRWISE PLOTS
    # columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    # # columns = ['gyro_x', 'gyro_y', 'gyro_z']  # Gyro's plots are much more interesting. Others are almost 1-dimensional
    # pairwise_combinations = itertools.combinations(columns, 2)
    # analyser_column_pairs_pairs_list = []
    #
    # for pairwise_combination in pairwise_combinations:
    #     col_1, col_2 = pairwise_combination
    #     analyser_column_pairs_pairs_list.append([[chair_analyser, col_1], [chair_analyser, col_2]])
    #
    # plot_measurements_pairwise(
    #     analyser_column_pairs_pairs_list=analyser_column_pairs_pairs_list,  # TODO: data should be normalized to explore acc measurements
    #     pic_prefix=pic_prefix,
    #     session_id=session_id,
    #     event_intervals_list=events_intervals_list,
    #     # n_rows=1,
    #     # n_cols=1,
    #     figsize=(30, 20),
    #     plot_suptitle=True,
    #     alpha=0.1,
    #     alpha_background=0.05,
    #     point_size=0.5,
    #     sharex='none',
    # )




df_chair_features = pd.DataFrame(chair_features_list)
df_chair_features.reset_index(inplace=True)
df_chair_features.rename(columns={'index': 'session_id'}, inplace=True)

df_chair_features.to_csv('data/chair_features.csv', index=False)

