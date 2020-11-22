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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold
import time
from sklearn.svm import SVC
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from torch.nn import LSTM
from Predictor import PredictorCell, BatchGenerator, features_pretty, get_df_results, index_names, TorchLogisticRegression, SplittedNN
from config import pic_folder, TIMESTEP, TIMESTEP_STRING
from mpl_toolkits.axes_grid1 import make_axes_locatable
from recreate_dataset import recreate_dataset_func

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--time_step_list', nargs='+', type=int)
parser.add_argument('--window_size_list', nargs='+', type=int)
parser.add_argument('--batch_size_list', nargs='+', type=int)
parser.add_argument('--hidden_size_list', nargs='+', type=int)
parser.add_argument('--attention_list', nargs='+', type=int)
parser.add_argument('--normalization_list', nargs='+', type=int)
parser.add_argument('--player_ids', nargs='+')
parser.add_argument('--loss', type=str, choices=['bce', 'mse'])
parser.add_argument('--super_suffix', type=str)
parser.add_argument('--n_attention_layers_list', nargs='+', type=int)
parser.add_argument('--n_dense_layers_list', nargs='+', type=int)
parser.add_argument('--n_attention_layers_with_hidden_state_list', nargs='+', type=int)
parser.add_argument('--cell_type_list', nargs='+', type=str)
parser.add_argument('--opt_list', nargs='+', type=str)
parser.add_argument('--target_types', nargs='+', type=str)
parser.add_argument('--target_verbose', action='store_true')
parser.add_argument('--append_dumb_predict', action='store_true')
parser.add_argument('--every_step_training_list', nargs='+', type=int)

parser.add_argument('--attention_multiplier', default=1, type=int)
# parser.add_argument('--target_type', type=str, choices=['more_than_avg', 'more_than_median', 'more_than_before'],
#                     default='more_than_median')
parser.add_argument('--arch', default=10, type=str, choices=['predictor_cell', 'logistic_regression',
                                                             'splitted_nn'])
parser.add_argument('--scorer', default='auc', type=str)
parser.add_argument('--n_epoches', default=100, type=int)
parser.add_argument('--batches4epoch', default=5, type=int)
parser.add_argument('--hidden_state_warmup', default=10, type=int)
parser.add_argument('--max_patience', default=10, type=int)
parser.add_argument('--n_repeat', default=5, type=int)
parser.add_argument('--plot_cell', default=0, type=int)
parser.add_argument('--plot_predict', default=0, type=int)
parser.add_argument('--plot_attention', default=0, type=int)
parser.add_argument('--plot_target', default=0, type=int)
parser.add_argument('--recreate_dataset', default=0, type=int)
parser.add_argument('--target_prefix', default='kills_proportion')
parser.add_argument('--verbose', default=0, type=int)
parser.add_argument('--warmup', default=10, type=int)
parser.add_argument('--n_inits', default=1, type=int)
use_dumb_predict = False
# parser.add_argument('--opt', default='adam', type=str, choices=['adam', 'sgd'])

feature_columns = [
    'gaze_movement', 'mouse_movement', 'mouse_scroll', 'hrm',
    'muscle_activity',  # Questionable!  # But actually that's moreless ok
    'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z',
    'temperature', 'co2', 'humidity' #, 'resistance',
]

feature_categories = [
    'Physical Activity',
    'Chair Movement',
    'Environment',
]

# feature_columns = ['gaze_movement', 'mouse_movement',
#                    'mouse_scroll',
#        # 'muscle_activity',  # Questionable!  # But actually that's moreless ok
#        'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z',
#        'hrm',
#                    # 'temperature',
#                    'co2',
#        # 'humidity', 'resistance',
# ]

def get_lr_by_epoch(epoch, base_lr=0.001, warmup=10, decay=0.97):
    if epoch < warmup:
        return base_lr * epoch / warmup * decay ** epoch
    else:
        return base_lr * decay ** epoch

def get_train_val_test_sizes(player_ids, train_proportion=0.55):
    train_size = int(len(player_ids) * train_proportion)
    val_size = int((len(player_ids) - train_size) * 0.5)
    test_size = len(player_ids) - train_size - val_size
    print(f'train_size={train_size}, val_size={val_size}, test_size={test_size}')

    return train_size, val_size, test_size

def get_final_target(target_past, target_future, target_type, args):
    if target_type == 'more_than_median':
        target = target_future - np.median(target_future)
    elif target_type == 'more_than_avg':
        target = target_future - np.mean(target_future)
    elif target_type == 'more_than_avg_fair':
        floating_mean = np.cumsum(target_future) / np.arange(1, len(target_future) + 1)
        target = target_future - floating_mean
    elif target_type == 'more_than_before':
        target = target_future - target_past
    else:
        raise ValueError(f'I don\'t know target_type {target_type}')

    return target

def get_dumb_predict(target_past, target_future):
    return target_past

def get_mode(player_id, player_ids_train, player_ids_val, player_ids_test):
    if player_id in player_ids_train:
        mode = 'train'
    elif player_id in player_ids_val:
        mode = 'val'
    elif player_id in player_ids_test:
        mode = 'test'
    else:
        raise ValueError(f'player_id {player_id} is not from train, val or test')

    return mode

def plot_attention_fig(player_id, attention_array, player_ids_train, player_ids_val, player_ids_test, features_pretty,
                       outputs_array, hidden_array, target_raw, target, args, n_epoch, suffix):
    mode = get_mode(player_id, player_ids_train, player_ids_val, player_ids_test)
    n_features = len(features_pretty)

    # fig, ax = plt.subplots(5, 1, squeeze=False, figsize=(12, 14), sharex=True,
    #                        gridspec_kw={'height_ratios': [0.5, 1, 0.2, 0.2, 0.2]})
    fig, ax = plt.subplots(5, 1, squeeze=False, figsize=(24, 14.5), sharex=True,
                           gridspec_kw={'height_ratios': [0.24, 0.3, 0.18, 0.18, 0.18]})
                           # gridspec_kw={'height_ratios': [0.24, 0.3, 0.17, 0.17, 0.17]})

    fontsize = 24
    # pos = ax[0, 0].imshow(attention_array.T, aspect='auto', cmap='Blues', vmin=0.0, vmax=1.0) # Blues  # TODO: check constants
    pos = ax[0, 0].imshow(attention_array.T, aspect='auto', cmap='Greens', vmin=0.0, vmax=1.0) # Blues  # TODO: check constants
    # ax[0, 0].set_title('Input Attention', fontsize=fontsize + 2)
    ax[0, 0].set_title('Network Attention $\\mathbf{\\alpha}(t)$', fontsize=fontsize + 2)
    # ax[0, 0].set_xlabel('Time Step Number', fontsize=fontsize)
    # ax[0, 0].set_yticks(np.arange(n_features))
    ax[0, 0].set_yticks(np.arange(len(feature_categories)))  # # TODO: check constant 0.5
    # ax[0, 0].set_yticklabels(features_pretty, fontsize=fontsize - 2)
    ax[0, 0].set_yticklabels(feature_categories, fontsize=fontsize)
    # ax[0, 0].set_xlim(0, len(attention_array.T))
    ax[0, 0].set_xlim(0 - 0.5, len(attention_array) - 0.5)
    ax[0, 0].set_ylim(-0.5, len(feature_categories) - 0.5)
    ax[0, 0].tick_params(axis='y', which='both', left=False)
    ax[0, 0].tick_params(axis='x', which='both', labelsize=fontsize, size=fontsize * 0.4)
    # ax[0].set_ytickslabels(np.arange(len(features)), features_pretty)
    # plt.yticks(np.arange(len(features)), features_pretty)

    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size="1%", pad=0.06)
    # plt.colorbar(pos, cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    cb = plt.colorbar(pos, cax=cax, ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=fontsize-4)
    # colorbar = fig.colorbar(pos, pad=-2, ax=ax[0, 1])

    # colorbar.ax.set_ylabel('attention weight', rotation=270, labelpad=10)

    # ax_new = ax.a
    # ax[1, 0].imshow(hidden_array.T, aspect='auto', cmap=None)  # , vmin=0, vmax=1)
    ax[1, 0].imshow(hidden_array.T, aspect='auto', cmap='RdBu')  #   # coolwarm # , vmin=0, vmax=1)
    ax[1, 0].set_title('Hidden State $\\mathbf{h}(t)$', fontsize=fontsize + 2)
    ax[1, 0].set_ylabel('Neuron Number', fontsize=fontsize)
    # ax[1, 0].set_xlabel('Time Step Number', fontsize=fontsize)
    ax[1, 0].tick_params(axis='y', which='both', left=False, labelleft=False, labelsize=fontsize, size=fontsize*0.4)
    ax[1, 0].tick_params(axis='x', which='both', labelsize=fontsize, size=fontsize*0.4)
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size="1%", pad=0.08)
    fig.delaxes(cax)
    # ax[1].autoscale(tight=True)
    # ax[1].set_xlim(right=1)
    # plt.tight_layout()
    # plt.savefig(pic_folder + f'attention/hidden_state_player_{player_id}_epoch_{n_epoch}_{suffix}.png')
    # plt.close()

    # if args.scorer == roc_auc_score:
    lw = 6.5
    markersize = 85
    ################## #
    if True:
        # label = 'Future Game Performance'
        label = 'Future Game Performance $p_{\\tau}(t)$'
        ax[2, 0].plot(target_raw, label=label, color='teal', lw=lw)
        ax[2, 0].set_title(label, fontsize=fontsize + 2)


        # ax[3, 0].plot(target, label='Binary Target', color='peru')
        breakpoints = list(np.nonzero(np.diff(target) != 0)[0])
        breakpoints = sorted(breakpoints)
        # print(breakpoints)
        if 0 not in breakpoints:
            breakpoints = [0] + breakpoints
            zero_fake = True
        else:
            zero_fake = False
        if len(target) - 1 not in breakpoints:
            breakpoints = breakpoints + [len(target) - 1]
            last_fake = True
        else:
            last_fake = False

        for n_breakpoint in range(len(breakpoints) - 1):
            if (n_breakpoint == 0) and zero_fake:
                breakpoint_start = breakpoints[n_breakpoint]
            else:
                breakpoint_start = breakpoints[n_breakpoint] + 1
            breakpoint_end = breakpoints[n_breakpoint + 1]
            x_points = list(range(breakpoint_start, breakpoint_end + 1))
            ax[3, 0].plot(x_points, target[breakpoint_start:breakpoint_end + 1],
                          label='Binary Target', color='peru', lw=lw)
        ax[3, 0].scatter(list(range(len(target))), target, s=markersize, color='peru')

        ax[3, 0].set_title('Binary Target $y_{\\tau}(t)$', fontsize=fontsize + 2)
        ax[3, 0].yaxis.set_major_locator(MultipleLocator(1))

        ax[4, 0].plot(outputs_array, label='Predict', color='olive', lw=lw)  # darkorange
        # ax[4, 0].set_title('Network Prediction', fontsize=fontsize + 2)
        ax[4, 0].set_title('Network Prediction $\\hat{y}_{\\tau}(t)$', fontsize=fontsize + 2)
        # ax[4, 0].yaxis.set_major_locator(MultipleLocator(0.1))

        # ax[2, 0].set_ylabel('$p_{\\tau}(t)$', fontsize=fontsize+2)
        # ax[3, 0].set_ylabel('$y_{\\tau}(t)$', fontsize=fontsize+2)
        # ax[4, 0].set_ylabel('$\\hat{y}_{\\tau}(t)$', fontsize=fontsize+2)

        for n_row in range(2, 5):
            ax[n_row, 0].yaxis.set_label_coords(-0.036, 0.5)
        # ax[4, 0]

        for i in [2, 3, 4]:
            divider = make_axes_locatable(ax[i, 0])
            cax = divider.append_axes("right", size="1%", pad=0.08)
            fig.delaxes(cax)

    for i in range(2, 5):
        ax[i, 0].tick_params(axis='both', which='major', labelsize=fontsize, size=fontsize*0.4)

    ax[-1, 0].set_xlabel('Time Step Number', fontsize=fontsize)

    # plt.tight_layout(rect=[0, 0, 1.04, 1])
    plt.tight_layout(rect=[-0.004, 0 - 0.01, 1 + 0.0035, 1 + 0.005])
    # plt.show()
    # fig.savefig(pic_folder + f'attention/{mode}/input_attention_player_{player_id}_epoch_{n_epoch}_{suffix}.png')
    fig.savefig(pic_folder + f'attention/{mode}/input_attention_player_{player_id}_epoch_{n_epoch}_{suffix}.pdf')
    plt.close()

def plot_predict_fig(player_id, mode, scorer, outputs_array, target, target_raw, n_epoch, suffix):
    # mode = get_mode(player_id, player_ids_train)

    plt.figure(figsize=(8, 4))
    if scorer == roc_auc_score:
        plt.plot(outputs_array, label='Predict')
        plt.plot(target, label='Binary Target')
        plt.plot(target_raw, label='Perfomance difference')
    else:
        plt.plot(outputs_array, label='Predict', color='darkorange')
        plt.plot(target, label='Perfomance difference', color='teal')

    plt.xlabel('Time Step Number')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pic_folder + f'attention/predictions_player_{player_id}_epoch_{n_epoch}_{suffix}_from_{mode}.png')
    plt.close()

def plot_cell_fig(player_id, hidden_array, cell_array, n_epoch, suffix):
    plt.imshow(hidden_array.T, aspect='auto', cmap=None)  # , vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(pic_folder + f'attention/hidden_state_player_{player_id}_epoch_{n_epoch}_{suffix}.png')
    plt.close()

    plt.imshow(cell_array.T, aspect='auto', cmap=None)  # , vmin=0, vmax=1)
    plt.tight_layout()
    plt.savefig(pic_folder + f'attention/cell_array_player_{player_id}_epoch_{n_epoch}_{suffix}.png')
    plt.close()

def plot_target_fig(player_id, target, target_binary):
    plt.close()
    color_1 = 'teal'
    color_2 = 'peru'
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    ax[0].plot(target, label='target', color=color_1)
    # ax[0].plot(target, label='target', color='tab:olive')
    ax[0].set_title('Perfomance Difference', fontsize=20)
    ax[0].axhline(0, color='red', ls='--', lw=1)
    breakpoints = list(np.nonzero(np.diff(target_binary) != 0)[0])
    breakpoints = sorted(breakpoints)
    # print(breakpoints)
    if 0 not in breakpoints:
        breakpoints = [0] + breakpoints
        zero_fake = True
    else:
        zero_fake = False
    if len(target_binary) - 1 not in breakpoints:
        breakpoints = breakpoints + [len(target_binary) - 1]
        last_fake = True
    else:
        last_fake = False

    for n_breakpoint in range(len(breakpoints) - 1):
        if (n_breakpoint == 0) and zero_fake:
            breakpoint_start = breakpoints[n_breakpoint]
        else:
            breakpoint_start = breakpoints[n_breakpoint] + 1
        breakpoint_end = breakpoints[n_breakpoint + 1]
        x_points = list(range(breakpoint_start, breakpoint_end + 1))
        ax[1].plot(x_points, target_binary[breakpoint_start:breakpoint_end + 1], label='target_binary', color=color_2)
    ax[1].scatter(list(range(len(target_binary))), target_binary, s=2, color=color_2)
    ax[1].set_title('Binary Target', fontsize=20)
    ax[1].set_xlabel('Time Step Number', fontsize=24)

    # ax[0].yaxis.set_major_locator(MultipleLocator(0.03))
    ax[1].yaxis.set_major_locator(MultipleLocator(1))

    for i in [0, 1]:
        ax[i].tick_params(axis='both', which='major', labelsize=14, size=7)

    # plt.axhline(target.mean(), label='target_mean', color='green')
    # plt.legend()
    fig.tight_layout()
    fig.savefig(pic_folder + f'target_player_{player_id}')

def get_players_splits(player_ids, train_size, val_size):
    player_ids_train = np.random.choice(player_ids, size=train_size, replace=False)
    player_ids_test_and_val = np.array([player_id for player_id in player_ids if player_id not in player_ids_train])
    player_ids_val = np.random.choice(player_ids_test_and_val, size=val_size, replace=False)
    player_ids_test = np.array([player_id for player_id in player_ids_test_and_val if player_id not in player_ids_val])

    return player_ids_train, player_ids_val, player_ids_test

def get_dict_of_lists_for_logging():
    result = {
        'train': [],
        'val': [],
        'test': [],
    }

    return result

def evaluate(
        predictor,
        player_ids,
        input_tensors_dict,
        player_ids_train,
        player_ids_val,
        player_ids_test,
        attention,
        n_epoch,
        epoch_best,
        attentions4model,
        suffix,
        args,
):
    loss4epoch_dict = get_dict_of_lists_for_logging()
    score4epoch_dict = get_dict_of_lists_for_logging()
    dumb_score4epoch_dict = get_dict_of_lists_for_logging()
    predictor.eval()

    for player_id in player_ids:
        if player_id not in input_tensors_dict:
            continue

        mode = get_mode(player_id, player_ids_train, player_ids_val, player_ids_test)
        train_on_this_player = False
        if hasattr(predictor, 'reset_hidden'):
            predictor.reset_hidden()

        input = input_tensors_dict[player_id]['input']
        target4player = input_tensors_dict[player_id]['target']
        # target_raw = input_tensors_dict[player_id]['target_raw']
        target_raw = input_tensors_dict[player_id]['target_future']  # TODO: PAY ATTENTION
        dumb_predict = input_tensors_dict[player_id]['dumb_predict']

        target_numpy = target4player.numpy()
        if len(np.unique(target_numpy)) < 2:
            print(f'The same targets for player {player_id}')
            continue

        hidden_state_list4player = []
        cell_state_list4player = []
        attention_weights_list4player = []
        outputs_list4player = []
        loss_list4player = []
        predict_list4player = []

        # for n_step in tqdm.tqdm(range(len(input))):
        assert len(input) == len(dumb_predict)
        for n_step in range(len(input)):
            # # print(n_step)
            # if train_on_this_player:
            #     opt.zero_grad()

            tensor_input4step = input[[n_step]]
            target4step = target4player[[n_step]]
            # output, hidden_state, cell_state, attention_weights = predictor(tensor_input4step)
            # t0 = time.time()
            forward_results = predictor(tensor_input4step)
            # print(time.time() - t0)
            output = forward_results['output'][0]

            predict_list4player.append(output.detach())
            loss = args.criterion(output, target4step)
            # dumb_predict4step = dumb_predict[n_step]

            hidden_state = forward_results['hidden_state']
            hidden_state_list4player.append(hidden_state.numpy().ravel())  #  It should be detached already
            # hidden_state_list4player.append(0)  #  It should be detached already
            if 'cell_value' in forward_results:
                cell_state_list4player.append(forward_results['cell_value'].detach().numpy().ravel())
            if attention:
                attention_weights = forward_results['attention_weights']
                attention_weights_list4player.append(attention_weights.detach().numpy().ravel())
            outputs_list4player.append(output.detach())
            loss_list4player.append(loss.detach().item())

        loss4player = np.mean(loss_list4player)

        if args.scorer.__name__ == 'accuracy_score':
            predict_binary = 1 * (np.array(predict_list4player) > 0.5)  # TODO: parametrize the threshold
            score4player = args.scorer(target4player.numpy().astype(int), predict_binary)
            # if use_dumb_predict:
            dumb_predict_binary = np.array(dumb_predict) > 0.5
            dumb_score4player = args.scorer(target4player.numpy(), dumb_predict_binary)  # Maybe convert predict to binary...
        else:
            score4player = args.scorer(target4player.numpy(), np.array(predict_list4player))
            # if use_dumb_predict:
            dumb_score4player = args.scorer(target4player.numpy(), np.array(dumb_predict))

        if attention:
            attention_array = np.array(attention_weights_list4player)
            if (score4player > 0.55):
                # if n_epoch == epoch_best:
                attentions4model[player_id] = attention_array.mean(axis=0)  # Not sure if it's working correctly
                # attention_sum_list.append(attention_array.mean(axis=0))

            if args.plot_attention and (n_epoch % args.plot_attention == 0) and n_epoch and (score4player > 0.6):
                if all(attention_array.mean(axis=0) > 0.15):
                    outputs_array = np.array(outputs_list4player)
                    hidden_array = np.array(hidden_state_list4player)
                    plot_attention_fig(player_id, attention_array, player_ids_train, player_ids_val, player_ids_test,
                                       features_pretty, outputs_array, hidden_array, target_raw, target4player, args,
                                       n_epoch, suffix)

        if args.plot_predict:
            outputs_array = np.array(outputs_list4player)
            plot_predict_fig(player_id, mode, args.scorer, outputs_array, target, target_raw, n_epoch, suffix)

        if args.plot_cell:
            hidden_array = np.array(hidden_state_list4player)
            cell_array = np.array(cell_state_list4player)
            plot_cell_fig(player_id, hidden_array, cell_array, n_epoch, suffix)

        loss4epoch_dict[mode].append(loss4player)
        score4epoch_dict[mode].append(score4player)
        dumb_score4epoch_dict[mode].append(dumb_score4player)


    return loss4epoch_dict, score4epoch_dict, dumb_score4epoch_dict



def get_input_tensors_dict(player_ids, data_dict_resampled_merged_with_target_scaled, target_column_future,
                           target_column_past, target_columns, target_type, args):
    input_tensors_dict = {}
    target_means = []

    for player_id in player_ids:
        train_tensors4player = {}
        df4train = data_dict_resampled_merged_with_target_scaled[player_id].copy()
        # mask2keep = df4train[target_column_future].notnull() & df4train[target_column_past].notnull()
        # if (args.target_type =='more_than_median') or (args.target_type =='more_than_avg'):
        if (target_type =='more_than_median') or (target_type =='more_than_avg'):
            mask2keep = df4train[target_column_future].notnull()
        else:
            mask2keep = df4train[target_column_future].notnull() & df4train[target_column_past].notnull()

        if mask2keep.sum() == 0:
            print(f'Not enough data for player {player_id}')
            continue

        df4train = df4train.loc[mask2keep, :]
        df4train.fillna(0, inplace=True)

        target_future = df4train[target_column_future].values
        target_past = df4train[target_column_past].values
        # print(target_past)
        target = get_final_target(target_past=target_past, target_future=target_future, target_type=target_type, args=args)
        dumb_predict = get_dumb_predict(target_past=target_past, target_future=target_future)
        margin = 0  # 0
        target_binary = (target >= margin) * 1
        if args.target_verbose:
            print(target_future)
            print(f'target_binary={target_binary}')
            print(f'target_binary.mean()={target_binary.mean()}')
        target_means.append(target_binary.mean())
        # print(f'target_binary.mean()={target_binary.mean()}')

        df4train.drop(columns=target_columns, inplace=True)
        df4train = df4train.loc[:, feature_columns] # TODO: check
        df4train.reset_index(drop=True, inplace=True)
        if args.append_dumb_predict:
            df4train['dumb_predict'] = dumb_predict

        features = list(df4train.columns)
        # n_features = train_tensors4player['input'].shape[1]
        n_features = len(features)

        # if plot_target:
        #     plot_target_fig(player_id, target, target_binary)

        train_tensors4player['input'] = torch.Tensor(df4train.values)
        train_tensors4player['dumb_predict'] = dumb_predict
        # if (args.scorer == roc_auc_score) or (args.scorer == accuracy_score):
        if args.classification:
            train_tensors4player['target'] = torch.Tensor(target_binary)  # FOR logloss metric
            train_tensors4player['target_raw'] = torch.Tensor(target)  # FOR logloss metric
        # elif args.scorer == mean_squared_error:
        else:
            train_tensors4player['target'] = torch.Tensor(target)  # FOR logloss metric
            train_tensors4player['target_raw'] = torch.Tensor(target)  # FOR logloss metric

        train_tensors4player['target_future'] = torch.Tensor(target_future)
        input_tensors_dict[player_id] = train_tensors4player

    target_mean_mean = np.mean(target_means)
    print(f'target_mean_mean = {target_mean_mean}')

    if args.verbose:
        print(f'n_features={n_features}')

    return input_tensors_dict, n_features


def run_experiment(time_step, window_size, batch_size, hidden_size, attention, normalization, n_repeat,
                   n_attention_layers, n_dense_layers, n_attention_layers_with_hidden_state, cell_type, opt_type,
                   target_type, every_step_training, n_init,
                   super_suffix, player_ids_train, player_ids_val, player_ids_test, args):
    # best_attentions_dict = {}
    random_suffix4suffix = np.random.choice(1000)
    suffix = f'{time_step}_{window_size}_{batch_size}_{hidden_size}_{attention}_{normalization}_{n_repeat}_' \
             f'{n_attention_layers}_{n_dense_layers}_{n_attention_layers_with_hidden_state}_{cell_type}_{opt_type}_{n_init}_hash{random_suffix4suffix}_{super_suffix}'
    # print(suffix)
    patience = 0
    val_score_best = 0
    train_score_best = 0
    test_score_best = 0
    epoch_best = 0
    attentions4model_best = 0
    stop_learning = False


    # player_ids_train, player_ids_val, player_ids_test = get_players_splits(args.player_ids, train_size, val_size)
    data_dict_resampled_merged_with_target_scaled = joblib.load(
        f'data/data_dict_resampled_merged_with_target_scaled_{int(time_step)}')

    target_columns = [column for column in data_dict_resampled_merged_with_target_scaled['10'].columns if
                      column.startswith(args.target_prefix)]
    target_column_past = f'{args.target_prefix}_{window_size}_4past'
    target_column_future = f'{args.target_prefix}_{window_size}_4future'

    input_tensors_dict, n_features = get_input_tensors_dict(
        args.player_ids,
        data_dict_resampled_merged_with_target_scaled,
        target_column_future,
        target_column_past, target_columns, target_type, args)

    loss_list_dict = get_dict_of_lists_for_logging()
    score_list_dict = get_dict_of_lists_for_logging()
    dumb_score_list_dict = get_dict_of_lists_for_logging()

    if args.arch == 'predictor_cell':
        predictor = PredictorCell(input_size=n_features, hidden_size=hidden_size, attention=attention,
                                  normalization=normalization, classification=args.classification,
                                  n_attention_layers=n_attention_layers, n_dense_layers=n_dense_layers,
                                  n_attention_layers_with_hidden_state=n_attention_layers_with_hidden_state,
                                  cell_type=cell_type)
    elif args.arch == 'logistic_regression':
        predictor = TorchLogisticRegression(input_size=n_features, hidden_size=hidden_size, n_dense_layers=n_dense_layers,
                                            cell_type=cell_type, n_attention_layers=n_attention_layers, n_attention_layers_with_hidden_state=n_attention_layers_with_hidden_state)
    elif args.arch == 'splitted_nn':
        predictor = SplittedNN(input_size=n_features, hidden_size=hidden_size,
                                            n_dense_layers=n_dense_layers,
                                            cell_type=cell_type, n_attention_layers=n_attention_layers,
                                            n_attention_layers_with_hidden_state=n_attention_layers_with_hidden_state,
                                            # groups=(5,6,3,1))
                                            groups=(5,6,3))
                                            # groups=(4,6,2))
                                            # groups=(12,))

        # print(predictor.parameters())
        # print(predictor.state_dict())

    if opt_type == 'adam':
        base_lr = 1e-3
        opt = Adam(predictor.parameters())
    elif opt_type == 'sgd':
        base_lr = 1e-2
        opt = SGD(predictor.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    else:
        raise ValueError(f'optimizer {args.opt} is not supported')

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    # batch_generator = BatchGenerator(input_tensors_dict, list(input_tensors_dict.keys()))
    batch_generator = BatchGenerator(input_tensors_dict, player_ids_train)

    for n_epoch in range(args.n_epoches):
        lr = get_lr_by_epoch(n_epoch, base_lr=base_lr, warmup=args.warmup)
        for param_group in opt.param_groups:
            param_group['lr'] = lr

        # for param_group in opt.param_groups:
        #     print(f"Epoch={n_epoch}, lr={param_group['lr']}")

        if stop_learning:
            # print('Early stopping...')
            break

        # print(f'Epoch {n_epoch}')
        ##### np.random.shuffle(player_ids)  # It should be commented, but I'm not sure

        for n_batch in range(args.batches4epoch):
            # noinspection PyUnresolvedReferences
            if hasattr(predictor, 'reset_hidden'):
                predictor.reset_hidden()

            batch = batch_generator.get_batch(batch_size)
            batch_input, batch_target = batch
            batch_size_actual = len(batch_input)
            # print(f'batch_size={batch_size}, batch_size_actual={batch_size_actual}')
            opt.zero_grad()  # Check the location
            # for n_step in range(batch_size * 2):
            for n_step in range(batch_size_actual):
                # train_on_this_batch = n_step >= (batch_size_actual // 2)
                train_on_this_batch = n_step >= args.hidden_state_warmup
                if train_on_this_batch:
                    predictor.train()
                else:
                    predictor.eval()

                tensor_input4step = batch_input[[n_step]]
                target4step = batch_target[[n_step]]
                # output, hidden_state, cell_state, attention_weights = predictor(tensor_input4step)
                forward_results = predictor(tensor_input4step)
                output = forward_results['output'][0]
                # hidden_state = forward_results['hidden_state']
                # attention_weights = forward_results['hidden_state']
                # output, hidden_state, cell_state, attention_weights = predictor(tensor_input4step)
                # output, hidden_state, attention_weights = predictor(tensor_input4step)
                # if True:  # hidden state is accumulated
                # if n_step >= (batch_size_actual // 2):  # hidden state is accumulated
                if train_on_this_batch:
                    # predictor.train()
                    # loss = criterion(output, target4step) / (batch_size_actual // 2)
                    opt.zero_grad()
                    loss = args.criterion(output, target4step) / (batch_size_actual - args.hidden_state_warmup)
                    loss.backward()
                    if every_step_training:
                        opt.step()  # TODO: think where should it be (here or after the loop)
            else:
                if not every_step_training:
                    opt.step()
                # pass
        # opt.step()  # TODO: I changed it
        ### EVALUATION ON EVERY PLAYER
        # for player_id in player_ids:
        attentions4model = {}
        loss4epoch_dict, score4epoch_dict, dumb_score4epoch_dict = evaluate(
            predictor,
            args.player_ids,
            input_tensors_dict,
            player_ids_train,
            player_ids_val,
            player_ids_test,
            attention,
            n_epoch,
            epoch_best,
            attentions4model,
            suffix,
            args,
        )

        for mode in ['train', 'val', 'test']:

            if len(loss4epoch_dict[mode]):
                loss4epoch4mode = np.mean(loss4epoch_dict[mode])
                loss_list_dict[mode].append(loss4epoch4mode)

            if len(score4epoch_dict[mode]):
                auc4epoch4mode = np.mean(score4epoch_dict[mode])
                # print(f'score_{mode}={round(auc4epoch4mode, 3)}')
                score_list_dict[mode].append(auc4epoch4mode)
                if mode == 'val':
                    val_score_new = auc4epoch4mode

                    if val_score_new > val_score_best:
                        val_score_best = np.mean(score4epoch_dict['val'])
                        test_score_best = np.mean(score4epoch_dict['test'])
                        train_score_best = np.mean(score4epoch_dict['train'])
                        attentions4model_best = np.mean(list(attentions4model.values()), axis=0)

                        epoch_best = n_epoch
                        patience = 0
                    else:
                        patience += 1
                        if patience >= args.max_patience:
                            stop_learning = True

            if len(dumb_score4epoch_dict[mode]):
                dumb_score4mode = np.mean(dumb_score4epoch_dict[mode])
                # print(f'dumb score on {mode} is {dumb_score4mode}')
                dumb_score_list_dict[mode].append(dumb_score4mode)

    index_array = [[time_step], [window_size], [batch_size], [hidden_size], [attention], [normalization], [n_repeat],
                   [n_attention_layers], [n_dense_layers], [n_attention_layers_with_hidden_state], [cell_type], [opt_type], [target_type],
                   [every_step_training], [n_init]]
    multi_index = pd.MultiIndex.from_arrays(index_array, names=index_names)

    df_results4experiment = pd.DataFrame(index=multi_index)

    df_results4experiment.loc[multi_index, 'score_train'] = train_score_best
    df_results4experiment.loc[multi_index, 'score_val'] = val_score_best
    df_results4experiment.loc[multi_index, 'score_test'] = test_score_best
    df_results4experiment.loc[multi_index, 'dumb_score_train'] = dumb_score_list_dict['train'][epoch_best]
    df_results4experiment.loc[multi_index, 'dumb_score_val'] = dumb_score_list_dict['val'][epoch_best]
    df_results4experiment.loc[multi_index, 'dumb_score_test'] = dumb_score_list_dict['test'][epoch_best]
    df_results4experiment.loc[multi_index, 'best_epoch'] = epoch_best

    print(df_results4experiment.index)
    print(df_results4experiment.reset_index())

    # attention_sum_list_dict[suffix] = attention_sum_list
    # best_attentions_dict[suffix] = attentions4model_best  # TODO: WHAT TO DO?
    # attentions4model_best  # TODO: WHAT TO DO?

    # if args.plot_cell:
    #     for mode in ['train', 'val', 'test']:
    #         if len(loss_list_dict[mode]):
    #             plt.plot(loss_list_dict[mode], label=mode)
    #             plt.savefig(pic_folder + f'_{mode}_loss_list_last_{suffix}.png')
    #             plt.close()
    #         if len(score_list_dict[mode]):
    #             plt.plot(score_list_dict[mode], label=mode)
    #             plt.savefig(pic_folder + f'_{mode}_auc_list_last_{suffix}.png')
    #             plt.close()

    return df_results4experiment, suffix, attentions4model_best

def run_experiments(
        time_step_list,
        window_size_list,
        batch_size_list,
        hidden_size_list,
        attention_list,
        normalization_list,
        n_repeat_list,
        args,
):
    # print(f'train_size, val_size, test_size = {train_size}, {val_size}, {test_size}')
    # df_results = get_df_results(multi_index_all)
    # df_results = pd.DataFrame()
    df_results_list = []
    attentions_list = []
    suffixes = []

    for time_step in time_step_list:
        if args.recreate_dataset:
            recreate_dataset_func(time_step)

        groups = itertools.product(
            window_size_list,
            batch_size_list,
            hidden_size_list,
            attention_list,
            normalization_list,
            n_repeat_list,
            args.n_attention_layers_list,
            args.n_dense_layers_list,
            args.n_attention_layers_with_hidden_state_list,
            args.cell_type_list,
            args.opt_list,
            args.target_types,
            args.every_step_training_list,
        )

        for group in groups:
            window_size, batch_size, hidden_size, attention, normalization, n_repeat, n_attention_layers,\
                n_dense_layers, n_attention_layers_with_hidden_state, cell_type, opt_type, target_type, every_step_training = group


            print(f"Experiment {group}")
            player_ids_train, player_ids_val, player_ids_test = get_players_splits(args.player_ids, args.train_size, args.val_size)

            for n_init in range(args.n_inits):
                df_results4experiment, suffix, attentions4model_best = run_experiment(time_step, window_size, batch_size, hidden_size, attention,
                                               normalization, n_repeat, n_attention_layers, n_dense_layers,
                                               n_attention_layers_with_hidden_state, cell_type, opt_type,
                                               target_type, every_step_training, n_init, super_suffix,
                                               player_ids_train, player_ids_val, player_ids_test, args)
                # print(df_results4experiment)
                # df_results = df_results.append(df_results4experiment)
                print(attentions4model_best)
                if attentions4model_best.mean() > 0.1:
                    attentions_list.append(attentions4model_best)
                df_results_list.append(df_results4experiment)
                suffixes.append(suffix)

        print(np.mean(attentions_list, axis=0))
        joblib.dump(attentions_list, 'attentions_list_9')


    return df_results_list, suffixes


if __name__ == '__main__':
    args = parser.parse_args()
    time_step_list = args.time_step_list
    window_size_list = args.window_size_list
    batch_size_list = args.batch_size_list
    hidden_size_list = args.hidden_size_list
    attention_list = args.attention_list
    normalization_list = args.normalization_list
    super_suffix = args.super_suffix
    n_repeat_list = list(range(args.n_repeat))
    # args.plot_attention = 50

    args.train_size, args.val_size, args.test_size = get_train_val_test_sizes(args.player_ids)
    # multi_index_all = pd.MultiIndex.from_product([time_step_list, window_size_list, batch_size_list,
    #                                               hidden_size_list, attention_list, normalization_list, n_repeat_list,],
    #                                              names=index_names)

    if args.scorer == 'auc':
        args.scorer = roc_auc_score
    elif args.scorer == 'accuracy':
        args.scorer = accuracy_score
    else:
        raise ValueError(f'I don\'t know scorer {args.scorer}')

    if args.loss == 'bce':
        args.criterion = nn.BCELoss()
    elif args.loss == 'mse':
        args.criterion = nn.MSELoss()
    else:
        raise ValueError(f'I don\'t know loss {args.loss}')

    args.classification = (args.scorer == roc_auc_score) or (args.scorer == accuracy_score)
    print(f'classification={args.classification}')

    df_results_list, suffixes = run_experiments(
        time_step_list,
        window_size_list,
        batch_size_list,
        hidden_size_list,
        attention_list,
        normalization_list,
        n_repeat_list,
        args,
    )

    path2results = f'data/exp_results/{super_suffix}'
    if not os.path.exists(path2results):
        os.mkdir(path2results)

    for df_results, suffix in zip(df_results_list, suffixes):
        df_results.to_csv(f'{path2results}/df_results_{suffix}.csv')

    # df_results = run_experiment(
    #     time_step,
    #     window_size,
    #     batch_size,
    #     hidden_size,
    #     attention,
    #     normalization,
    #     n_repeat,
    #     args,
    # )

    # df_results.to_csv(f'data/df_results_{super_suffix}.csv')  # Was before Match 8th

















