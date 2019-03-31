import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from utils import normalize_MPU9250_data, split_df
from sklearn.metrics import mutual_info_score
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LassoCV, LassoLarsIC, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier

# TODO: change the order of sensors on a plot
suffix = '__3_10_1_1_5'
target_name = '>1000 h exp'
plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'

df_coef_aic = pd.read_csv(f'data/df_coef_aic{suffix}.csv', index_col=0)
df_coef_bic = pd.read_csv(f'data/df_coef_bic{suffix}.csv', index_col=0)

features_aic = list(df_coef_aic.index)
features_bic = list(df_coef_bic.index)


df_features = pd.read_csv('data/df_features.csv')
df_features = df_features.fillna(df_features.median())
player_id_column = df_features['player_id']
target_column = df_features[target_name]

df_players_targets = df_features[['player_id', target_name]].drop_duplicates()
players_positive = df_players_targets.loc[df_players_targets[target_name]==1, 'player_id'].values
players_negative = df_players_targets.loc[df_players_targets[target_name]==0, 'player_id'].values

splitter_positive = KFold(n_splits=2, shuffle=True)
splitter_negative = KFold(n_splits=2, shuffle=True)
# list(splitter_positive.split(players_positive))
# list(splitter_negative.split(players_negative))

player_id_column = df_features['player_id']
target_column = df_features[target_name]
df_features.drop(['session_id', 'player_id'], axis=1, inplace=True)

ss = StandardScaler()
df_features_cropped = df_features[features_aic]
df_features_cropped.loc[:, :] = ss.fit_transform(df_features_cropped.values)



lr = LogisticRegression(solver='lbfgs')
svm = SVC(probability=True, gamma='scale')
rf = RandomForestClassifier(n_estimators=100, max_depth=2)
knn = KNeighborsClassifier(n_neighbors=3)
models_list = [lr, svm, rf, knn]

n_emulations = 1000
results_list = []
rf_importances_list = []

for n_emulation in range(n_emulations):
    for split_positive, split_negative in zip(splitter_positive.split(players_positive), splitter_negative.split(players_negative)):
        train_players_positive, val_players_positive = split_positive  # KFold returns shit
        train_players_negative, val_players_negative = split_negative

        train_players = np.append(players_positive[train_players_positive], players_negative[train_players_negative])
        val_players = np.append(players_positive[val_players_positive], players_negative[val_players_negative])

        train_mask = player_id_column.isin(train_players)
        val_mask = ~train_mask

        x_train = df_features.loc[train_mask, features_aic]  # AIC is used here
        x_val = df_features.loc[val_mask, features_aic]
        y_train = target_column.loc[train_mask]
        y_val = target_column.loc[val_mask]

        results = {}

        for model in models_list:
            results_model = {}

            model_name = model.__class__.__name__
            model.fit(x_train, y_train)
            predict_val = model.predict_proba(x_val)[:, 1]

            score_auc = roc_auc_score(y_val, predict_val)
            score_accuracy = accuracy_score(y_val, np.round(predict_val))
            score_log_loss = log_loss(y_val, predict_val)

            results_model['auc'] = score_auc
            results_model['accuracy'] = score_accuracy
            results_model['logloss'] = score_log_loss

            # print(model_name)
            # print(f'score_auc = {round(score_auc, 2)}, score_accuracy = {round(score_accuracy, 2)}, score_log_loss = {round(score_log_loss, 2)}')

            if model_name == 'RandomForestClassifier':
                rf_importances_list.append(model.feature_importances_)
            results[model_name] = results_model

        results_list.append(results)

# TODO: add emulations

df_dict = {}

for num, results in enumerate(results_list):
    df_dict[num] = pd.DataFrame.from_dict(results)


panel = pd.Panel(df_dict)

df_scores = panel.mean(axis=0).round(2)
df_scores.T.to_csv('data/df_scores.csv')

panel.std(axis=0).round(2)



features = list(x_train.columns)

importances_lr = list(lr.coef_.ravel())
df_importances_lr = pd.DataFrame(list(zip(features, importances_lr)))





# importances_rf = list(rf.feature_importances_)
importances_rf = list(np.mean(rf_importances_list, axis=0))
df_importances_rf = pd.DataFrame(list(zip(features, importances_rf)))
df_importances_rf.columns = ['feature', 'score']
df_importances_rf.sort_values(['score'], inplace=True, ascending=True)
df_importances_rf.reset_index(drop=True, inplace=True)
df_importances_rf.to_csv('data/df_importances_rf.csv', index=False)

df_importances_rf

name = 'rf'

plt.close()
plt.figure(figsize=(13,6))
plt.barh(np.arange(len(features)), df_importances_rf['score'])
plt.ylim(-0.5,df_importances_rf.shape[0]-0.5)
plt.tick_params('y', labelsize=26)
plt.tick_params('x', labelsize=16)
plt.yticks(np.arange(df_importances_rf.shape[0]),df_importances_rf['feature'])
plt.xlabel('Mean impurity decrease', fontsize=26)
# plt.xlabel('Mean impurity decrease.')
# plt.title(name+' coefs. '+'.', fontsize=22)
plt.tight_layout()
plt.savefig(pic_prefix + 'feature_importance_rf.png')














