import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from utils import string2json

plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'

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


data_dict = joblib.load('data/data_dict')

gamedata_dict = {}

for player_id, data_dict4player in data_dict.items():
    if 'gamelog' not in data_dict4player:
        continue

    mask_player_events = data_dict4player['gamelog']['parameters'].apply(lambda x: x.find('koltech')) != -1
    time_game_start = data_dict4player['gamelog']['time'].min()
    time_game_end = data_dict4player['gamelog']['time'].max()


    df_gamelog = data_dict4player['gamelog'].loc[mask_player_events, :]
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
        'time_game_start': time_game_start,
        'time_game_end': time_game_end,
    }


    gamedata_dict[player_id] = player_gamedata_dict



joblib.dump(gamedata_dict, 'data/gamedata_dict')










