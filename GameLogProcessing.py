# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import joblib
from utils import normalize_MPU9250_data, split_df, string2json
# from ChairAnalyser import ChairAnalyser

plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'
data_path = 'Anonimised Data/Data'
processed_data_path = 'data/players_data_processed'

# data_dict = joblib.load('data/data_dict')


def check_player_is_killed(parameters_dict):
    # First condition: event is that somebody dies
    # Second condition: dying player is skoltech experimental rat
    return ('userid' in parameters_dict) and (parameters_dict['userid'].find('koltech') != -1)

def get_shootout_times_start_end(
        df_gamelog4player,
        max_shootout_delay=3,
        min_shootout_duration=1,
        min_shootout_actions=3):
    mask = df_gamelog4player['event'] == 'weapon_fire'

    times_fire = df_gamelog4player.loc[mask, 'time']
    # times_fire = pd.to_datetime(times_fire).apply(lambda x: x.timestamp())
    intervals_to_prev = np.append(0, times_fire.iloc[1:].values - times_fire.iloc[:-1].values)
    times_fire = pd.DataFrame(times_fire)
    times_fire['interval_to_prev'] = intervals_to_prev

    mask_shootout_starts = times_fire['interval_to_prev'] > max_shootout_delay
    shootout_starts = np.nonzero(mask_shootout_starts)[0]

    shootout_ends = shootout_starts[1:] - 1
    shootout_starts = shootout_starts[:-1]

    shootout_start_end_list = []

    for shootout_start, shootout_end in zip(shootout_starts, shootout_ends):
        if shootout_end - shootout_start < min_shootout_actions:  # Too little actions
            continue

        shootout_time_start = times_fire.iloc[shootout_start]['time']
        shootout_time_end = times_fire.iloc[shootout_end]['time']

        if shootout_time_end - shootout_time_start < min_shootout_duration:  # Too little time
            continue
        else:
            shootout_start_end_list.append([shootout_start, shootout_end])

    shootout_times_start_end = []

    for shootout_start, shootout_end in shootout_start_end_list:
        shootout_time_start = times_fire.iloc[shootout_start]['time']
        shootout_time_end = times_fire.iloc[shootout_end]['time']
        shootout_times_start_end.append([shootout_time_start, shootout_time_end])

    return shootout_times_start_end


if __name__ == '__main__':
    sessions_dict = joblib.load('data/sessions_dict')
    gamedata_dict = {}

    for session_id, session_data_dict in sessions_dict.items():
        if 'gamelog' not in session_data_dict:
            continue

        mask_player_events = session_data_dict['gamelog']['parameters'].apply(lambda x: x.find('koltech')) != -1
        df_gamelog = session_data_dict['gamelog'].loc[mask_player_events, :]
        df_gamelog['parameters'] = df_gamelog['parameters'].apply(string2json)
        df_gamelog['health'] = df_gamelog['parameters'].apply(lambda x: int(x['health']) if 'health' in x else None)
        mask_somebody_is_killed = df_gamelog['health'] == 0

        mask_player_is_killed = mask_somebody_is_killed & df_gamelog.loc[:, 'parameters'].apply(check_player_is_killed)
        mask_player_kills = mask_somebody_is_killed & ~mask_player_is_killed

        times_is_killed = df_gamelog.loc[mask_player_is_killed, ['time']]
        # times_is_killed = list(pd.to_datetime(times_is_killed['time']).apply(lambda x: x.timestamp()).values)
        times_is_killed = list(times_is_killed['time'].values)
        times_kills = df_gamelog.loc[mask_player_kills, ['time']]
        # times_kills = list(pd.to_datetime(times_kills['time']).apply(lambda x: x.timestamp()).values)
        times_kills = list(times_kills['time'].values)

        shootout_times_start_end = get_shootout_times_start_end(df_gamelog)

        player_gamedata_dict = {
            'times_is_killed': times_is_killed,
            'times_kills': times_kills,
            'shootout_times_start_end': shootout_times_start_end,
        }

        gamedata_dict[session_id] = player_gamedata_dict


    # gamedata_dict['9'].keys()
    # gamedata_dict['9']['times_is_killed']
    # gamedata_dict['9']['times_kills']


    joblib.dump(gamedata_dict, 'data/gamedata_dict_old')



# df_sample = sessions_dict[10]['gamelog']
# sample_mask = df_sample['parameters'].apply(lambda x: x.find('koltech')) != -1
# df_sample = df_sample.loc[sample_mask, :]
# # df_sample['event'].value_counts()

# params = df_sample.loc[mask, 'parameters'].apply(string2json)
# params = list(params.values)
#
# df_params = pd.DataFrame(params)
# df_params['silenced'].value_counts()
# df_params['weapon'].value_counts()









# with open(gamelog_path, 'rb') as f:
#     gamelog = f.readlines()
#
# # gamelog = [string.decode() for string in gamelog]
# gamelog_lenght_initial = len(gamelog)
# gamelog = [string for string in gamelog if string.find(b'koltech') != -1]
# gamelog_lenght_filtered = len(gamelog)
# print(f'gamelog_lenght_initial = {gamelog_lenght_initial}, gamelog_lenght_filtered = {gamelog_lenght_filtered}')
#
# # with open('tmp/gamelog.csv', 'wb') as f:
# with open(processed_data_path + '/gamelog.csv', 'wb') as f:
#     # for line in gamelog:
#     #     f.write(line)
#     f.writelines(gamelog)
#
# df_gamelog = pd.read_csv('tmp/gamelog.csv', header=None)
















# times = pd.to_datetime(df_gamelog.loc[mask_killed, 'time'])
# np.diff(times.values) / 10 ** 9
#
#
# (times.iloc[1:] - times.iloc[:-1].values).iloc[10]
#
#
#
# (df_gamelog['health_is_0']).sum()
#
#
# plt.plot(df_gamelog['health_is_0'])
#
#
# # TODO: check player behaviour right after death
#
#
# df_gamelog['event'].value_counts()
#
#
# mask_fire = df_gamelog['event'] == 'weapon_fire'
#
# df_gamelog.loc[mask_fire, 'parameters']
# fire_times = pd.to_datetime(df_gamelog.loc[mask_fire, 'time'])
# (fire_times.values[1:] - fire_times.values[:-1]).min()
#
# df_fire = pd.DataFrame(list(df_gamelog.loc[mask_fire, 'parameters'].values))
#
# df_fire.info()
#



