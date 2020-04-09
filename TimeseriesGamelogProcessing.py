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

# player_id = '0'
# data_dict4player = data_dict[player_id]
for player_id, data_dict4player in data_dict.items():
    if 'gamelog' not in data_dict4player:
        continue

    mask_player_events = data_dict4player['gamelog']['parameters'].apply(lambda x: x.find('koltech')) != -1
    time_game_start = data_dict4player['gamelog']['time'].min()
    time_game_end = data_dict4player['gamelog']['time'].max()

    # ### DEBUG PART
    # gamelog = data_dict4player['gamelog']
    # mask = gamelog['event'] == 'player info'
    # gamelog.loc[mask, :]
    # gamelog['parameters'] = gamelog['parameters'].apply(string2json)
    # gamelog['event'].value_counts()
    # # TODO: please consider events player_spawn,
    # mask_spawn = gamelog['event'] == 'player info'
    # df_spawn = gamelog.loc[mask_spawn, :]
    # df_spawn_addition = df_spawn['parameters'].apply(lambda x: pd.Series(x))
    # df_spawn = df_spawn.join(df_spawn_addition).drop(['parameters', 'event'], axis=1)
    # df_spawn.loc[:, ['userid', 'time', 'userid team', 'teamnum']]
    # df_spawn['userid'].value_counts()
    #
    # ### END OF DEBUG PART


    df_gamelog = data_dict4player['gamelog'].loc[mask_player_events, :]
    df_gamelog['parameters'] = df_gamelog['parameters'].apply(string2json)
    df_gamelog['health'] = df_gamelog['parameters'].apply(lambda x: int(x['health']) if 'health' in x else None)
    mask_somebody_is_killed = df_gamelog['health'] == 0

    # ### Exploration
    # ### The issue: there are too many parsed death events
    # df_killed = df_gamelog.loc[mask_somebody_is_killed, :]
    # df_killed['parameters'].apply(lambda x: x['userid'] if 'health' in x else None)
    # addition = df_killed['parameters'].apply(lambda x: pd.Series(x) if 'health' in x else None).drop(columns=['health'])
    # df_killed = df_killed.join(addition)
    # mask = df_killed['userid'].apply(lambda x: x.find('koltech') != -1)
    # df_killed.columns
    # mask.sum()
    # pd.options.display.max_rows = 10**4
    # pd.options.display.min_rows = -1
    # pd.options.display.float_format = '{:.2f}'.format
    # pd.options.display.column_space = -1
    # pd.options.display.width = 1000
    # df_tmp = df_killed.loc[mask, ['attacker', 'userid', 'userid team', 'attacker team']]
    # df_tmp_agg_0 = df_tmp.groupby(['attacker', 'userid'])['userid team', 'attacker team'].apply(lambda x: len(x))
    # df_tmp_agg_1 = df_tmp.groupby(['attacker', 'userid'])['userid team', 'attacker team'].apply(lambda x: x.drop_duplicates())
    # df_tmp_agg_0 = pd.DataFrame(df_tmp_agg_0)
    # df_tmp_agg_0.join(df_tmp_agg_1)
    # pd.merge(df_tmp_agg_0, df_tmp_agg_1)
    # pd.concat([df_tmp_agg_0, df_tmp_agg_1], axis=1)
    #
    #
    # df_killed.loc[:, ['time', 'userid team', 'attacker team', 'attacker', 'userid']]
    #
    # killer_ally_mask = df_killed['attacker'] == 'Jimbo (id:6076)'
    # df_killed.loc[killer_ally_mask, :]
    # columns2check = [column for column in df_killed.columns if column != 'parameters']
    # df_killed.loc[killer_ally_mask, columns2check]
    # df_killed
    #
    #
    #
    # # df_killed['time'].diff()
    # from pprint import pprint
    # N = 5
    # for i in range(N):
    #     print(df_killed['time'].iloc[i])
    #     pprint(df_killed['parameters'].iloc[i])
    #
    # # df_killed['parameters'].iloc[1]
    # # df_killed['parameters'].iloc[2]
    # # df_killed['parameters'].iloc[3]
    #
    # ### End of exploration


    mask_player_is_killed = mask_somebody_is_killed & df_gamelog.loc[:, 'parameters'].apply(check_player_is_killed)
    mask_player_kills = mask_somebody_is_killed & ~mask_player_is_killed

    times_is_killed = df_gamelog.loc[mask_player_is_killed, ['time']]
    # times_is_killed = list(pd.to_datetime(times_is_killed['time']).apply(lambda x: x.timestamp()).values)
    times_is_killed = list(times_is_killed['time'].values)
    times_kills = df_gamelog.loc[mask_player_kills, ['time']]
    # times_kills = list(pd.to_datetime(times_kills['time']).apply(lambda x: x.timestamp()).values)
    times_kills = list(times_kills['time'].values)

    # ### Exploration
    # mask_fire = df_gamelog['event'] == 'weapon_fire'
    # times_fire = df_gamelog.loc[mask_fire, ['time']]
    # times_fire['my_event'] = 'fire'
    #
    # mask_spawn = df_gamelog['event'] == 'player_spawn'
    # times_spawn = df_gamelog.loc[mask_spawn, ['time']]
    # times_spawn['my_event'] = 'spawn'
    #
    # times_is_killed = df_gamelog.loc[mask_player_is_killed, ['time']]
    # times_is_killed['my_event'] = 'is_killed'
    # times_kills = df_gamelog.loc[mask_player_kills, ['time']]
    # times_kills['my_event'] = 'kills'
    #
    # times_merged = pd.concat([times_fire, times_is_killed, times_kills, times_spawn])
    # times_merged.sort_values(['time'])
    #
    #
    # ### End of exploration



    shootout_times_start_end = get_shootout_times_start_end(df_gamelog)

    player_gamedata_dict = {
        'times_is_killed': times_is_killed,
        'times_kills': times_kills,
        'shootout_times_start_end': shootout_times_start_end,
        'time_game_start': time_game_start,
        'time_game_end': time_game_end,
    }

    gamedata_dict[player_id] = player_gamedata_dict

gamedata_dict_update = joblib.load('data/gamedata_update_0')
gamedata_dict.update(gamedata_dict_update)  # Data for 13, 15, 16, 17 are fake data. I need to update it
# TODO: check the update

joblib.dump(gamedata_dict, 'data/gamedata_dict')









