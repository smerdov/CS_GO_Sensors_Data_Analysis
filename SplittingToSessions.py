# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import joblib
from utils import normalize_MPU9250_data, split_df, string2json, split_dfs_by_time
import argparse

plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'
data_path = 'Anonimised Data/Data'
processed_data_path = 'data/players_data_processed'

parser = argparse.ArgumentParser()
parser.add_argument('--session_duration', default=60 * 10, type=int)
parser.add_argument('--max_sessions_per_player', default=3, type=int)

# session_duration = 60 * 5
# max_sessions_per_player = 10

if __name__ == '__main__':
    args = parser.parse_args()
    session_duration = args.session_duration
    max_sessions_per_player = args.max_sessions_per_player

    data_dict = joblib.load('data/data_dict')

    session_id = 0
    session_id2player_dict = {}
    sessions_dict = {}

    for player_id, player_data_dict in data_dict.items():
        # if ('gamelog' not in player_data_dict):  # Because we split data according to gamelog
        #     continue  # Then it has to be imagined
        if ('schairlog' not in player_data_dict):  # Because we split data according to gamelog
            continue  # Then it has to be imagined

        if 'gamelog' in player_data_dict:
            df_gamelog = player_data_dict['gamelog']
            data_sources = ['gamelog'] + [key for key in player_data_dict.keys() if key != 'gamelog']

            timestamp_min = df_gamelog['time'].min()
            timestamp_max = df_gamelog['time'].max()
        else:
            data_sources = [key for key in player_data_dict.keys()]

            df_chair = player_data_dict['schairlog']
            timestamp_min = df_chair['time'].min()
            timestamp_max = df_chair['time'].max()

        df_list = [player_data_dict[data_source] for data_source in data_sources]

        df_chunks_list = split_dfs_by_time(df_list, timestamp_min, timestamp_max, chunk_duration=session_duration, max_chunks=max_sessions_per_player)
        n_chunks = len(df_chunks_list[0])

        for n_chunk in range(n_chunks):
            df_sessions_list = [df[n_chunk] for df in df_chunks_list]
            session_data = {}

            for data_source, df in zip(data_sources, df_sessions_list):
                if len(df):
                    session_data.update({
                        data_source: df,
                    })

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

