import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NN import get_final_target
import argparse

plt.interactive(True)

lr_coefs = []

split_size = 15
for target_type in ['more_than_before', 'more_than_median', 'more_than_mean']:
    for time_step in [5, 10, 20, 30]:#, 20, 30]:
        data_dict = joblib.load(f'data/data_dict_resampled_merged_with_target_scaled_{time_step}')
        for window_size in window_sizes:
        # for target_type in ['more_than_mean']:
            split_scores = []
            for n_split in range(30):
                x = {
                    'train': {},
                    'val': {},
                }
                y = {
                    'train': {},
                    'val': {},
                }
                train_ids = list(np.random.choice(list(data_dict.keys()), size=split_size, replace=False))
                val_ids = [player_id for player_id in data_dict.keys() if player_id not in train_ids]
                for player_id, df in data_dict.items():
                    df = data_dict[player_id].copy()
                    # pretarget = df[f'kills_proportion_{window_size}_4future'] > df[f'kills_proportion_{window_size}_4past']
                    # pretarget = df[f'kills_proportion_{window_size}_4future'] > df[f'kills_proportion_{window_size}_4future'].mean()

                    pretarget = df[f'kills_proportion_{window_size}_4future'] > 0.1
                    # if target_type == 'more_than_median':
                    #     pretarget = df[f'kills_proportion_{window_size}_4future'] > df[f'kills_proportion_{window_size}_4future'].median()
                    #
                    # if target_type == 'more_than_before':
                    #     pretarget = df[f'kills_proportion_{window_size}_4future'] > df[f'kills_proportion_{window_size}_4past']
                    #
                    # if target_type == 'more_than_mean':
                    #     mean_performance = df[f'kills_proportion_{window_size}_4future'].mean()
                    #     # print(f'mean_performance = {mean_performance}')
                    #     pretarget = df[f'kills_proportion_{window_size}_4future'] > mean_performance

                    mask2keep = pretarget.notnull()
                    pretarget = pretarget.loc[mask2keep]
                    x_all = df.loc[mask2keep, feature_columns]
                    y_all = pretarget
                    # y_all = pretarget > pretarget.median()
                    player_id_type = 'train' if player_id in train_ids else 'val'

                    # x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=test_size, shuffle=False)

                    x[player_id_type][player_id] = x_all
                    y[player_id_type][player_id] = y_all

                    # x_train_total_list.append(x_train)
                    # x_val_total_list.append(x_val)
                    # y_train_total_list.append(y_train)
                    # y_val_total_list.append(y_val)
                x_train_total = pd.concat(x['train'].values())
                x_val_total = pd.concat(x['val'].values())
                y_train_total = pd.concat(y['train'].values())
                y_val_total = pd.concat(y['val'].values())
                lr = LogisticRegression(solver='lbfgs')
                lr.fit(x_train_total, y_train_total)
                lr_coefs.append(lr.coef_)
                # lr_predict_train = lr.predict_proba(x_train_total)[:, 1]
                # lr_predict_val = lr.predict_proba(x_val_total)[:, 1]
                #
                # print(f'Train score = {roc_auc_score(y_train_total, lr_predict_train)}')
                # print(f'Val score = {roc_auc_score(y_val_total, lr_predict_val)}')
                # for feature, coef in zip(feature_columns, lr.coef_[0]):
                #     print(feature, coef)
                val_scores_list = []
                for val_player in val_ids:
                    lr_predict_val = lr.predict_proba(x['val'][val_player])[:, 1]
                    val_score = roc_auc_score(y['val'][val_player], lr_predict_val)
                    # print(f"Val score = {val_score}")
                    val_scores_list.append(val_score)
                mean_score4split = np.mean(val_scores_list)
                # print(mean_score4split)
                split_scores.append(mean_score4split)

            print(f'time_step = {time_step}')
            print(f'window_size = {window_size}')
            print(f'target_type = {target_type}')
            print(np.mean(split_scores))
            print(split_scores)


lr_coefs_mean = np.mean(lr_coefs, axis=0)
for feature, coef in zip(feature_columns, lr_coefs_mean[0]):
    print(feature, coef)
