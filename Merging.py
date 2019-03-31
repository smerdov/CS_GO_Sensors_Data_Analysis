import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from utils import normalize_MPU9250_data, split_df
from sklearn.metrics import mutual_info_score
import itertools

plt.interactive(True)
plt.interactive(False)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'


df_chair_features = pd.read_csv('data/chair_features.csv')
df_players = pd.read_csv('data/players.csv')
game_events_features = pd.read_csv('data/game_events_features.csv')
df_sessions_players = pd.read_csv('data/df_sessions_players.csv')

# df_features = df_chair_features.merge(game_events_features, on='session_id')
# df_features = df_features.merge(df_sessions_players, on='session_id')

df_features = df_chair_features.merge(df_sessions_players, on='session_id')
df_features = df_features.merge(game_events_features, on='session_id')  # Recently added
df_features = df_features.merge(df_players, on='player_id')

first_features = ['>100 h exp',
                  '>1000 h exp',
                  'Kill Death Ratio',
                  # '>3000 h exp',
                  # 'Skill',
                  'Gender',
                  'Age',
                  # 'Hours',
                  'lean_back',
                  'med_acc_x_std',
                  'med_acc_y_std',
                  'med_acc_z_std',
                  'med_gyro_x_std',
                  'med_gyro_y_std',
                  'med_gyro_z_std',
                  ]

features_order = first_features + [feature for feature in df_chair_features.columns if feature not in (first_features + ['session_id'])] + \
    ['session_id', 'player_id']

df_features = df_features.loc[:, features_order]

df_features.to_csv('data/df_features.csv', index=False)

size = df_features.shape[1] / 2
# df_features.drop(['session_id', 'player_id'], axis=1, inplace=True)
df_features.drop(['session_id', 'player_id'], axis=1, inplace=True)
# df_features = df_features.iloc[:, :-40]

plt.close()
plt.figure(figsize=(size * 1.05, size))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# corr_data = df_features.drop(['player_id'], axis=1).corr()
corr_data = df_features.corr()
corr_data = corr_data.round(1)
sns.heatmap(
    corr_data, square=True, cmap=cmap, vmax=1,vmin=-1, linewidths=.2, cbar_kws={"shrink": .7},
    annot=True, annot_kws={"size": 13}, cbar=False,
    # xticklabels=False, yticklabels=False,
)
plt.tick_params('both', labelsize=19)
plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
# plt.title('Correlation between features and targets', fontsize=22)
plt.tight_layout()
plt.savefig('pic/heatmap_last.png')




# for feature in first_features:
#     print(df_features.loc[:, feature].value_counts())
#
# df_features.loc[:, 'player_id'].value_counts()
# df_features.loc[:, ['player_id', 'Skill']]


#
# from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
#
# df_features = df_features.fillna(df_features.median())
# na_count = df_features.isnull().sum(axis=1)
# df_features_filtered = df_features.drop(np.nonzero(na_count)[0], axis=0).iloc[:, :-40]
#
# discrete_feature_mask = np.zeros((df_features_filtered.shape[1]), dtype=bool)
# discrete_feature_mask[:5] = True
# mutual_information_series_list = []
# for column in first_features:
#     mutual_information = mutual_info_classif(df_features_filtered, df_features_filtered.loc[:, column], discrete_features=discrete_feature_mask)
#     # mutual_information = mutual_info_classif(df_features_filtered, df_features_filtered.loc[:, column], discrete_features=False)
#     mutual_information_series = pd.Series(mutual_information, index=df_features_filtered.columns, name=column)
#     mutual_information_series_list.append(mutual_information_series)
#
# df_mutual_information = pd.DataFrame(mutual_information_series_list).T
#
# # df_mutual_information['mean'] = df_mutual_information.mean(axis=1)
# # df_mutual_information.sort_values(['mean'], inplace=True, ascending=False)
#
# # colnum = 1
# # column = df_mutual_information.columns[colnum]
# column = '>500 h'
# df_mutual_information.sort_values(column, ascending=False)[column].drop(first_features)
#
# df_mutual_information.index
#
# df_features.columns
# df_train = df_features.iloc[:, 7:-40]
# target = df_features.iloc[:, 1]
# from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
#
#
# ss = StandardScaler()
# train_scaled = ss.fit_transform(df_train)
# lr = LogisticRegression(solver='lbfgs')
# rf = RandomForestClassifier(n_estimators=10, max_depth=3)
#
# cross_val_score(lr, train_scaled, target, cv=2, scoring='roc_auc')
# cross_val_score(rf, train_scaled, target, cv=2, scoring='roc_auc')
# lr.fit(train_scaled, target)
# rf.fit(train_scaled, target)
#
# lasso = Lasso(alpha=0.25)
# lasso.fit(train_scaled, ss.fit_transform(target.values.reshape(-1, 1)))
# pd.DataFrame(lasso.coef_, index=df_train.columns).sort_values(0)
#
# active_features_nums = np.nonzero(lasso.coef_)[0]
# active_features = df_train.columns[active_features_nums]
#
#
# rf.fit(train_scaled[:, active_features_nums], target)
# # pd.DataFrame(rf.feature_importances_, index=df_train.columns).sort_values(0)
# pd.DataFrame(rf.feature_importances_, index=active_features).sort_values(0)
# rf.
#
#
#
# lasso
#
#
#
#
# np.argsort(abs(lr.coef_))
#
# n_feature = 22
# lr.coef_[0][n_feature]
# df_train.columns[n_feature]
#
#
#
#
#
# # mutual_infos = np.zeros((df_features.shape[1], df_features.shape[1]))
# #
# #
# # for col_1, col_2 in itertools.product(range(df_features.shape[1]), range(df_features.shape[1])):
# #     x = df_features.iloc[:, col_1].values
# #     y = df_features.iloc[:, col_2].values
# #     mutual_infos[col_1, col_2] = mutual_info_score(x, y)
# #     # mutual_infos[col_2, col_1] = mutual_info_score(x, y)
# #
# # mut_informations_list = []
# # for i in range(df_features.shape[1]):
# #     mut_informations = mutual_info_regression(df_features, df_features.iloc[:, i])
# #     mut_informations_list.append(mut_informations)
# #
# # mut_informations_all = np.array(mut_informations_list)
# # plt.close()
# # sns.heatmap(mut_informations_all)
# #
# #
# #
# # plt.close()
# # sns.heatmap(mutual_infos)
# #
#
#
# # df_features.shape
# # TODO: what about repetitions?
# # TODO: extract player skill more accurately
# # TODO: make heatmap more understandable
#
# size = df_features.shape[1] / 4
#
# plt.close()
# plt.figure(figsize=(size, size))
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
# # corr_data = df_features.drop(['player_id'], axis=1).corr()
# corr_data = df_features.corr()
# corr_data = corr_data.round(1)
# sns.heatmap(
#     corr_data, square=True, cmap=cmap, vmax=1,vmin=-1, linewidths=.2, cbar_kws={"shrink": .8},
#     annot=False, annot_kws={"size": 7},
#     # xticklabels=False, yticklabels=False,
# )
# plt.title('Correlation between player skill and his behaviour on the chair', fontsize=15)
# plt.tight_layout()
# plt.savefig('pic/heatmap_last.png')
#
#
# # TODO: make hours binary for multiple thresholds
# # TODO: extract game features more accurately
#






