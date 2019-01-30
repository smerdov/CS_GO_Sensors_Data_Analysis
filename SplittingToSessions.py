# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import joblib
from utils import normalize_MPU9250_data, split_df, string2json, split_dfs_by_time
# from ChairAnalyser import ChairAnalyser

plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'
data_path = 'Anonimised Data/Data'
processed_data_path = 'data/players_data_processed'

data_dict = joblib.load('data/data_dict')

session_id = 0
session_id2player_dict = {}
sessions_dict = {}
# also add envibox and datalog

for player_id, player_data_dict in data_dict.items():
    # if ('gamelog' not in player_data_dict) or ('schairlog' not in player_data_dict) or ('hrm' not in player_data_dict):  # TODO: incomplete data should be handled as well
    if ('gamelog' not in player_data_dict):  # Because we split data according to gamelog
        continue

    df_gamelog = player_data_dict['gamelog']
    data_sources = ['gamelog'] + [key for key in player_data_dict.keys() if key != 'gamelog']
    df_list = [player_data_dict[data_source] for data_source in data_sources]

    timestamp_min = df_gamelog['time'].min()
    timestamp_max = df_gamelog['time'].max()

    # df_gamelog_chunks, df_chair_chunks, df_hrm_chunks = split_dfs_by_time([df_gamelog, df_chair, df_hrm], timestamp_min, timestamp_max)
    df_chunks_list = split_dfs_by_time(df_list, timestamp_min, timestamp_max)
    # n_chunks = len(df_gamelog_chunks)
    n_chunks = len(df_chunks_list[0])

    for n_chunk in range(n_chunks):
        # df_gamelog_session = df_gamelog_chunks[n_chunk]
        # df_chair_session = df_chair_chunks[n_chunk]
        # df_hrm_session = df_hrm_chunks[n_chunk]

        df_sessions_list = [df[n_chunk] for df in df_chunks_list]

        # if (len(df_gamelog_session) == 0) or (len(df_chair_session) == 0):
        #     gamelog_time_min, gamelog_time_max = df_gamelog['time'].min(), df_gamelog['time'].max()
        #     chair_time_min, chair_time_max = df_chair['time'].min(), df_chair['time'].max()
        #
        #     print(f'Empty session, player_id = {player_id}')
        #     print(f'gamelog_time_min = {gamelog_time_min}, gamelog_time_max = {gamelog_time_max}')
        #     print(f'chair_time_min = {chair_time_min}, chair_time_max = {chair_time_max}')
        #     continue

        session_data = {}

        for data_source, df in zip(data_sources, df_sessions_list):
            if len(df):
                session_data.update({
                    data_source: df,
                })

        # session_data = {
        #     'gamelog': df_gamelog_chunks[n_chunk],
        #     'schairlog': df_chair_chunks[n_chunk],
        #     'hrm': df_hrm_chunks[n_chunk],
        # }

        sessions_dict[session_id] = session_data
        session_id2player_dict[session_id] = player_id
        session_id += 1

n_session_sources_list = []

for session in sessions_dict.values():
    n_session_sources_list.append(len(session))

plt.hist(n_session_sources_list)


df_sessions_players = pd.DataFrame([session_id2player_dict]).T
df_sessions_players.reset_index(inplace=True)
df_sessions_players.columns = ['session_id', 'player_id']
df_sessions_players.to_csv('data/df_sessions_players.csv', index=False)

joblib.dump(sessions_dict, 'data/sessions_dict')



# session_id = 3
# sessions_dict[session_id]['schairlog']['time']
# sessions_dict[session_id]['gamelog']['time']
#
#
# sessions_dict  # 13 players, 32 sessions
# # len(set(session_id2player_dict.values()))
#
# data_dict['0']['schairlog']['time']
# data_dict['0']['gamelog']['time']
#
#
#
#
# df_sample = data_dict['2']['gamelog'].copy()
# df_chair_sample = data_dict['2']['schairlog'].copy()
#
# timestamp_min = pd.to_datetime(df_sample['time'].min()).timestamp()
# timestamp_max = pd.to_datetime(df_sample['time'].max()).timestamp()
#
# df_sample_chunks, df_sample_chair_chunks = split_dfs_by_time([df_sample, df_chair_sample], timestamp_min, timestamp_max)
# # chunks_chair = split_dfs_by_time(df_chair_sample, timestamp_min, timestamp_max)
#
#
# chunks[1]
# chunks_chair[0]
#


