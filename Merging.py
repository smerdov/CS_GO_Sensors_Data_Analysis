import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from utils import normalize_MPU9250_data, split_df
from ChairAnalyzer import ChairAnalyser

plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'


df_chair_features = pd.read_csv('data/chair_features.csv')
df_players = pd.read_csv('data/players.csv')
game_events_features = pd.read_csv('data/game_events_features.csv')
df_sessions_players = pd.read_csv('data/df_sessions_players.csv')

df_features = df_chair_features.merge(game_events_features, on='session_id')
df_features = df_features.merge(df_sessions_players, on='session_id')
df_features = df_features.merge(df_players, on='player_id')
df_features.drop(['session_id', 'player_id'], axis=1, inplace=True)

first_features = ['More than 100 hours experience',
                  'More than 500 hours experience',
                  'More than 3000 hours experience',
                  'Skill',
                  'Gender',
                  'Age',
                  'Hours',
                  ]

features_order = first_features + [feature for feature in df_features.columns if feature not in first_features]
df_features = df_features.loc[:, features_order]

# df_features.shape
# TODO: what about repetitions?
# TODO: extract player skill more accurately
# TODO: make heatmap more understandable

size = df_features.shape[1] / 4

plt.close()
plt.figure(figsize=(size, size))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# corr_data = df_features.drop(['player_id'], axis=1).corr()
corr_data = df_features.corr()
corr_data = corr_data.round(1)
sns.heatmap(
    corr_data, square=True, cmap=cmap, vmax=1,vmin=-1, linewidths=.2, cbar_kws={"shrink": .8},
    annot=False, annot_kws={"size": 7},
    # xticklabels=False, yticklabels=False,
)
plt.title('Correlation between player skill and his behaviour on the chair', fontsize=15)
plt.tight_layout()
plt.savefig('pic/heatmap_10_3_by_sessions__chair__players__game_events.png')


# TODO: make hours binary for multiple thresholds
# TODO: extract game features more accurately







