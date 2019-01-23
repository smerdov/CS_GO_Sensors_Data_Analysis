import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import normalize_MPU9250_data, split_df
from ChairAnalyzer import ChairAnalyser

plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'


df_chair_features = pd.read_csv('data/chair_features.csv')
df_players = pd.read_csv('data/players.csv')
df_chair_features4players = pd.merge(df_chair_features, df_players, on='player_id')


plt.close()
plt.figure(figsize=(15, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
corr_data = df.drop(['player_id'], axis=1).corr()
corr_data = corr_data.round(1)
sns.heatmap(
    corr_data, square=True, cmap=cmap, vmax=1,vmin=-1, linewidths=.2, cbar_kws={"shrink": .8},
    annot=True, annot_kws={"size": 7},
    # xticklabels=False, yticklabels=False,
)
plt.title('Correlation between player skill and his behaviour on the chair', fontsize=15)
plt.tight_layout()
plt.savefig('pic/heatmap_10_3.png')


df.shape

# TODO: make hours binary for multiple thresholds







