import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import tqdm
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

plt.interactive(True)
pd.options.display.max_columns = 15
pic_folder = 'pic/'

# version = 'hidden_size'
# version = 'test_11'
# version = 'window_size_0'
# version = 'time_step_0'
versions_list = [
    # 'window_size_0',
    # 'window_size_1',
    # 'hidden_size_2',
    # 'hidden_size_3',
    # 'normalization_0',
    # 'time_step_0',
    # 'time_step_1',
    # 'time_step_2',
    # # 'time_step_1_60',
    'prefinal',
]
classic_versions_list = [
    # 'window_size_0',
    # 'time_step_1',
    # 'time_step_2',
    # 'time_step_1_60',
]


attention_names_dict = {
    0: 'RNN',
    1: 'RNN + Input Attention, Softmax',
    2: 'RNN + Input Attention, Sigmoid',
    4: 'RNN + Input Attention, [0, 1] clamp',
}

df_results = pd.DataFrame()

for version in versions_list:
    df_results_new = pd.read_csv(f'data/df_results_{version}.csv')
    df_results_new = df_results_new.loc[df_results_new['score_val'] != -1, :]
    df_results_new.set_index(['time_step', 'window_size', 'batch_size', 'attention', 'hidden_size', 'normalization', 'n_repeat'], inplace=True)
    df_results_new = df_results_new[['score_test', 'score_val']]
    df_results = pd.concat([df_results, df_results_new])

for version in classic_versions_list:
    df_results_classic = pd.read_csv(f'data/df_results_classic_{version}.csv')
    df_results_classic = df_results_classic.loc[df_results_classic['alg_name'] != 'Random Forest 1']
    df_results_classic = df_results_classic.loc[df_results_classic['time_step'] != 60]
    df_results_classic = df_results_classic.loc[df_results_classic['window_size'] != 600]
    df_results_classic['batch_size'] = -1
    df_results_classic['hidden_size'] = -1
    df_results_classic['normalization'] = -1
    df_results_classic.rename(columns={'alg_name': 'attention'}, inplace=True)
    df_results_classic.set_index(['time_step', 'window_size', 'batch_size', 'attention', 'hidden_size', 'normalization', 'n_repeat'], inplace=True)
    df_results_classic.rename(columns={'score_val': 'score_test'}, inplace=True)
    df_results_classic.rename(index={'Random Forest 0': 'Random Forest'}, inplace=True)
    # df_results_classic.rename(index={'Logistic Regression': 'Random Forest'})

    df_results = pd.concat([df_results, df_results_classic])
# df_results = df_results.join(df_results_classic)

def plot_dependency(df_results, labels, colors, labels_col='attention', dependency_col='hidden_size', label2text_dict=None,
                    xlabel='', suffix='v0', plot_errors=True):
    df_grouped = df_results.groupby([labels_col, dependency_col])['score_test']
    df_agg_mean = df_grouped.mean().reset_index()
    df_agg_std = df_grouped.std().reset_index()
    print(df_agg_mean)
    print(df_agg_std)
    # labels = np.unique(df_agg_mean[labels_col])

    plt.figure(figsize=(13.5, 9))
    # plt.figure(figsize=(20, 15))

    for label, color in zip(labels, colors):
        mask_label = df_agg_mean[labels_col] == label
        dependency_values = df_agg_mean.loc[mask_label, dependency_col]
        means4label = df_agg_mean.loc[mask_label, 'score_test'].values.ravel()
        stds4label = df_agg_std.loc[mask_label, 'score_test'].values.ravel()
        print(means4label)
        print(stds4label)

        if label2text_dict is not None:
            if label in label2text_dict:
                label_text = label2text_dict[label]
            else:
                label_text = label
        else:
            label_text = label

        lower = means4label - stds4label
        upper = means4label + stds4label

        plt.plot(dependency_values, means4label, label=label_text, color=color, lw=5)
        plt.scatter(dependency_values, means4label, marker='o', s=140, color=color)
        if plot_errors:
            plt.fill_between(dependency_values, lower, upper, alpha=0.1, color=color)

    plt.tick_params(axis='both', which='major', labelsize=25, size=20)
    plt.xticks()
    # plt.xlabel('Window Size, s', fontsize=35)
    plt.xlabel(xlabel, fontsize=35)
    plt.ylabel('ROC AUC', fontsize=35)
    plt.legend(fontsize=20)  # 16 if not enough space  # , loc='lower right')
    # plt.xlim(97, 610)  # UPDATE IF 60 IS ADDED!!!
    plt.tight_layout()
    plt.savefig(f'pic/{labels_col}_{dependency_col}_{suffix}.png')
    plt.close()

def plot_dependency_time_step(df_results, labels, colors, labels_col='attention', dependency_col='hidden_size', label2text_dict=None,
                    xlabel='', suffix='v0', plot_errors=True):
    df_grouped = df_results.groupby([labels_col, dependency_col])['score_test']
    df_agg_mean = df_grouped.mean().reset_index()
    df_agg_std = df_grouped.std().reset_index()
    print(df_agg_mean)
    print(df_agg_std)
    # labels = np.unique(df_agg_mean[labels_col])

    # plt.figure(figsize=(13.5, 9))
    fig, ax = plt.subplots(figsize=(13.5, 9))
    # plt.figure(figsize=(20, 15))

    for label, color in zip(labels, colors):
        mask_label = df_agg_mean[labels_col] == label
        dependency_values = df_agg_mean.loc[mask_label, dependency_col]
        means4label = df_agg_mean.loc[mask_label, 'score_test'].values.ravel()
        stds4label = df_agg_std.loc[mask_label, 'score_test'].values.ravel()
        print(means4label)
        print(stds4label)

        if label2text_dict is not None:
            if label in label2text_dict:
                label_text = label2text_dict[label]
            else:
                label_text = label
        else:
            label_text = label

        lower = means4label - stds4label
        upper = means4label + stds4label

        mask_part_1 = dependency_values < 40
        mask_part_2 = dependency_values >= 30

        dependency_values_part_1 = dependency_values[mask_part_1]
        means4label_part_1 = means4label[mask_part_1]
        plt.plot(dependency_values_part_1, means4label_part_1, label=label_text, color=color, lw=5)

        dependency_values_part_2 = dependency_values[mask_part_2]
        means4label_part_2 = means4label[mask_part_2]
        plt.plot(dependency_values_part_2, means4label_part_2, color=color, lw=5, linestyle='--')

        plt.scatter(dependency_values, means4label, marker='o', s=140, color=color)
        if plot_errors:
            plt.fill_between(dependency_values, lower, upper, alpha=0.1, color=color)

    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_locator(MultipleLocator(10))
    plt.tick_params(axis='both', which='major', labelsize=25, size=20)
    plt.xticks()
    # plt.xlabel('Window Size, s', fontsize=35)
    plt.xlabel(xlabel, fontsize=35)
    plt.ylabel('ROC AUC', fontsize=35)
    plt.legend(fontsize=15)  # 16 if not enough space  # , loc='lower right')
    # plt.xlim(97, 610)  # UPDATE IF 60 IS ADDED!!!
    plt.tight_layout()
    plt.savefig(f'pic/{labels_col}_{dependency_col}_{suffix}.png')
    plt.close()


if version == 'time_step':
    xlabel = 'Time Step, s'
elif version == 'hidden_size':
    xlabel = 'Hidden Size'
elif version == 'window_size':
    xlabel = 'Window Size, s'
elif version == 'window_size':
    xlabel = 'Window Size, s'
else:
    xlabel = version

colors = ['blue', 'orange', 'green', 'red', 'cyan', 'gold', 'black']
labels = [0, 1, 2, 4]  + ['Logistic Regression', 'Random Forest', 'SVM']



### hidden_size
plot_dependency(df_results, labels, colors, 'attention', dependency_col='hidden_size',
                label2text_dict=attention_names_dict, xlabel='Hidden Size', plot_errors=False)


### window_size
# plot_dependency(df_results, labels, colors, 'attention', dependency_col='window_size',
#                 label2text_dict=attention_names_dict, xlabel='Window Size, s', plot_errors=False)


### time_step
mask_not_five = [df_results.index[i][0] != 5 for i in range(len(df_results))]
plot_dependency_time_step(df_results.iloc[mask_not_five], labels, colors, 'attention', dependency_col='time_step',
                label2text_dict=attention_names_dict, xlabel='Time Step, s', plot_errors=False)



### normalization
plot_dependency(df_results, labels, colors, 'attention', dependency_col='normalization',
                label2text_dict=attention_names_dict, xlabel='normalization', plot_errors=False)

# df_att_norm_mean = df_results.groupby(['attention', 'normalization']).mean()
# df_att_norm_std = df_results.groupby(['attention', 'normalization']).std()
# df_att_norm_mean.rename(columns={'score_test': 'score_test_mean'}, inplace=True)
# df_att_norm_mean['score_test_std'] = df_att_norm_std['score_test']
#
# df_att_norm_mean.to_csv('data_best/df_att_norm.csv')
#






for alg_name, color in zip(alg_names_list, colors):
    mask_alg = results_no_index['alg_name'] == alg_name
    mean4alg = results_mean.iloc[mask_alg.nonzero()].values.ravel()
    std4alg = results_std.iloc[mask_alg.nonzero()].values.ravel()

    lower = mean4alg - std4alg
    upper = mean4alg + std4alg

    plt.plot(window_size_list, mean4alg, label=alg_name, linewidth=5, color=color)
    plt.scatter(window_size_list, mean4alg, marker='o', s=140, color=color)
    plt.fill_between(window_size_list, lower, upper, alpha=0.3, color=color)

plt.tick_params(axis='both', which='major', labelsize=30)
plt.xticks()
# plt.xlabel('Window Size, s', fontsize=35)
plt.xlabel('Window Size, s', fontsize=35)
plt.ylabel('ROC AUC', fontsize=35)
plt.legend(fontsize=32)
# plt.xlim(97, 610)  # UPDATE IF 60 IS ADDED!!!
plt.tight_layout()
plt.savefig('pic/classical_ml_window_size_v0.png')




df_results.groupby('time_step').mean()
df_results.groupby('window_size').mean()
df_results.groupby('batch_size').mean()
df_results.groupby('hidden_size').mean()
df_results.groupby('attention').mean()  # For RNN 4 is better than 0!!!
df_results.groupby('normalization').mean()  # For RNN 4 is better than 0!!!

df_results.groupby(['time_step', 'window_size']).mean()
df_results.groupby(['time_step', 'batch_size']).mean()
df_results.groupby(['time_step', 'hidden_size']).mean()
df_results.groupby(['window_size', 'batch_size']).mean()
df_results.groupby(['window_size', 'hidden_size']).mean()
df_results.groupby(['batch_size', 'hidden_size']).mean()
df_results.groupby(['attention', 'time_step']).mean()
df_results.groupby(['attention', 'window_size']).mean()
df_results.groupby(['attention', 'hidden_size']).mean()
df_results.groupby(['attention', 'hidden_size', 'window_size', 'time_step']).mean()

df_results.groupby(['time_step', 'window_size', 'hidden_size']).mean()


"""
Inference after v2:
1. time_step 20 is probably better
2. window_size 300 should be used. Larger values have better scores, but the system is not so flexible
3. batch_size can be any from 8 to 256. Let's set it to 64
4. hidden size 32 is the best. 64 is slightly worse, 16 is worse


"""



"""
Inference after v1 or v0:
1. Anyway, timestep 30 looks too much. Timestep 5 actually looks fine, but it requires a lot of training. 
2. window_size 120 is too short and noisy. 600 has the best score, but the target become too trivial.
3. batch_size 2 isn't a good idea. 16 and 128 aren't very distinguishable, the optimal can be from 8 to inf.
4. The best hidden size is 32, the worst is 2, 8 is ok. More is better till some limit. Maybe to check 16, 64, 128(too much?), ...

5. time_step 10 and window_size 300 is ok.
6. Shorter time_step requires higher batch_size
7. Probably higher time_step require lesser hidden_size

"""






time_step_list = [10, 20]  # 10 is already tested
window_size_list = [120, 180, 300, 600]

alg_names_list = ['Logistic Regression', 'Random Forest', 'SVM']
version = 'v0'
df_results = pd.read_csv(f'data/df_results_classic_{version}.csv')
df_results.set_index(['window_size', 'alg_name', 'n_repeat'], inplace=True)


def mean_std(x):
    # return pd.Series([x.mean(), x.std()], index=['mean', 'std']).T
    return pd.DataFrame({
        'mean': x.mean(),
        'std': x.std(),
    })

# df_results.groupby(['window_size', 'alg_name']).apply(lambda x: mean_std(x))
results_mean = df_results.groupby(['window_size', 'alg_name']).apply(lambda x: x.mean())
results_std = df_results.groupby(['window_size', 'alg_name']).apply(lambda x: x.std())

results_no_index = results_std.reset_index().drop(columns='score_val')

colors = ['blue', 'orange', 'green']

plt.interactive(True)
plt.close()

plt.figure(figsize=(12, 9))
for alg_name, color in zip(alg_names_list, colors):
    mask_alg = results_no_index['alg_name'] == alg_name
    mean4alg = results_mean.iloc[mask_alg.nonzero()].values.ravel()
    std4alg = results_std.iloc[mask_alg.nonzero()].values.ravel()

    lower = mean4alg - std4alg
    upper = mean4alg + std4alg

    plt.plot(window_size_list, mean4alg, label=alg_name, linewidth=5, color=color)
    plt.scatter(window_size_list, mean4alg, marker='o', s=140, color=color)
    plt.fill_between(window_size_list, lower, upper, alpha=0.3, color=color)

plt.tick_params(axis='both', which='major', labelsize=30)
plt.xticks()
# plt.xlabel('Window Size, s', fontsize=35)
plt.xlabel('Window Size, s', fontsize=35)
plt.ylabel('ROC AUC', fontsize=35)
plt.legend(fontsize=32)
# plt.xlim(97, 610)  # UPDATE IF 60 IS ADDED!!!
plt.tight_layout()
plt.savefig('pic/classical_ml_window_size_v0.png')



##### Loading from separate series files
# filenames = os.listdir('data')
# def check_relevance(filename):
#     cond_1 = filename[-4:] == '.csv'
#     cond_2 = filename[:7] == 'series_'
#     cond_3 = filename[-6:-4] == version
#     return cond_1 and cond_2 and cond_3
#
# relevant_series = [filename for filename in filenames if check_relevance(filename)]
#
# df_results = pd.DataFrame()
#
# for series_path in relevant_series:
#     series2append = pd.read_csv(f'data/{series_path}')
#     index_names = ['time_step', 'window_size', 'batch_size', 'hidden_size']
#     series2append.columns = index_names + list(series2append.columns[4:])
#     series2append.set_index(index_names, inplace=True)
#     df_results = df_results.append(series2append)

#
# df_agg_mean = df_results.groupby(['attention', 'hidden_size'])['score_test'].mean().reset_index()
# df_agg_std = df_results.groupby(['attention', 'hidden_size'])['score_test'].std().reset_index()
# attention_list = [0, 1, 2]
# hidden_size_list = [8, 32, 64]
#
# plt.close()
# for attention in attention_list:
#     mask = df_agg_mean['attention'] == attention
#     hidden_sizes = df_agg_mean.loc[mask, 'hidden_size']
#     means = df_agg_mean.loc[mask, 'score_test']
#     stds = df_agg_std.loc[mask, 'score_test']
#     plt.plot(hidden_sizes, means, label=attention_names_dict[attention])
#
#
# plt.legend()
# plt.tight_layout()