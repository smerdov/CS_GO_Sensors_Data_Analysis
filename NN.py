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

plt.interactive(False)
pd.options.display.max_columns = 15
pic_folder = 'pic/'

features_pretty = ['gaze movement',
 'mouse movement',
 'mouse scroll',
 'muscle activity',
 'chair acc_x',
 'chair acc_y',
 'chair acc_z',
 'chair gyro_x',
 'chair gyro_y',
 'chair gyro_z',
 'heart rate',
 'skin resistance',
 'temperature',
 'co2 level',
 'humidity']

class PredictorCell(nn.Module):

    # def __init__(self, input_size=9, hidden_size=16, attention=1, alpha=0.01, eps=0.01, normalization=True):
    def __init__(self, input_size=9, hidden_size=16, attention=1, alpha=0.01, eps=0.01, normalization=True, classification=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention = attention
        self.mean = torch.zeros(size=(1, input_size))
        self.std = torch.ones(size=(1, input_size)) * 0.3
        self.alpha = alpha
        self.eps = eps
        self.normalization = normalization
        self.classification = classification

        # self.lstm = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        # self.gru = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.gru = nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

        self.reset_hidden()
        if self.attention:
            self.attention_0 = nn.Linear(self.input_size + self.hidden_size,
                                         self.input_size)  # self.input_size + self.hidden_size)

            self.attention_1 = nn.Linear(self.input_size, self.input_size)
            self.attention_2 = nn.Linear(self.input_size, self.input_size)

        # self.layer_norm_0 = nn.LayerNorm(self.input_size)
        self.hidden2output_0 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden2output_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden2output_2 = nn.Linear(self.hidden_size, 1)


    def forward(self, input):
        # print([input, self.hidden])
        if self.attention:
            input_and_hidden = torch.cat([input, self.hidden], dim=1)
            # print(input_and_hidden.shape)
            attention_logits_0 = self.attention_0(input_and_hidden)
            attention_logits_0 = torch.relu(attention_logits_0)
            attention_logits_1 = self.attention_1(attention_logits_0)
            attention_logits_1 = torch.relu(attention_logits_1)
            attention_logits_2 = self.attention_2(attention_logits_1)

            attention_logits = attention_logits_2

            if self.attention == 1:
                attention_weights = F.softmax(attention_logits, dim=1)
            elif self.attention == 2:
                attention_weights = torch.sigmoid(attention_logits) # , dim=1)
                # print(attention_weights)
            elif self.attention == 3:
                attention_weights = (attention_logits > 0)
            elif self.attention == 4:
                # attention_weights = attention_logits - (attention_logits - 1) * (attention_logits > 1)  # Cut everything > 1
                # attention_weights = attention_weights * (attention_weights > 0)  # Cut everything < 0
                attention_weights = attention_logits.clamp(0, 1)
            elif self.attention == 5:
                # attention_weights = attention_logits.clamp(0, 2)
                attention_weights = attention_logits
            elif self.attention == 6:
                attention_weights = (attention_logits > 1)
            elif self.attention == 7:
                attention_logits_median = attention_logits.median()
                attention_weights = (attention_logits > torch.max(attention_logits_median, torch.Tensor([0])))
            elif self.attention == 8:
                attention_logits_mean = attention_logits.mean()
                attention_weights = (attention_logits > torch.max(attention_logits_mean, torch.Tensor([0])))
            elif self.attention == 9:
                attention_logits = attention_logits * (attention_logits > 0)
                attention_weights = torch.tanh(attention_logits)
            else:
                raise ValueError(f'self.attention = {self.attention} is not supported')

            input_with_attention = input * attention_weights  # Check this. And everything else.
        else:
            input_with_attention = input
            attention_weights = 0

        if self.normalization:
            input_with_attention_normalized = (input_with_attention - self.mean) / self.std
        else:
            input_with_attention_normalized = input_with_attention

        if self.normalization and self.training:  # Update mean and std
            input_with_attention_detached = input_with_attention.detach()
            self.mean = self.mean * (1 - self.alpha) + self.alpha * input_with_attention_detached
            self.std = self.std * (1 - self.alpha) + self.alpha * (input_with_attention_detached - self.mean).abs()

            self.std = self.std.clamp(self.eps, 10)

        hidden = self.gru(input_with_attention_normalized, self.hidden)
        # hidden, cell = self.lstm(input_with_attention, (self.hidden, self.cell))

        output = self.hidden2output_0(hidden)
        output = torch.relu(output)
        output = self.hidden2output_1(output)
        output = torch.relu(output)
        output = self.hidden2output_2(output)
        self.hidden = hidden.detach()
        # self.cell = cell.detach()

        if self.classification:
            output = torch.sigmoid(output)  # To [0, 1] interval

        # return output, self.hidden, attention_weights
        return output, self.hidden, self.cell, attention_weights

    def reset_hidden(self):
        #  Pay attention to 1 here. Here 1 = batch_size.
        self.hidden = torch.zeros(size=(1, self.hidden_size))
        self.cell = torch.zeros(size=(1, self.hidden_size))


class BatchGenerator:

    def __init__(self, train_tensors_dict, player_ids_train):
        self.train_tensors_dict = train_tensors_dict
        self.player_ids_train = player_ids_train
        # self.player_ids_test = player_ids_test

    def get_batch(self, batch_size):
        # Actually it supposed to return batch_size of preliminary data and batch_size data for training
        player_id = np.random.choice(self.player_ids_train)
        sample_len = len(self.train_tensors_dict[player_id]['input'])
        if sample_len > 2 * batch_size:
            index_start = np.random.choice(sample_len - 2 * batch_size)
            index_end = index_start + 2 * batch_size
        else:
            index_start = 0
            index_end = sample_len - 1

        return self.train_tensors_dict[player_id]['input'][index_start:index_end], \
               self.train_tensors_dict[player_id]['target'][index_start:index_end]

player_ids = ['9', '0', '11', '7', '6', '1', '10', '19', '8', '21', '4', '3', '12', '2', '5', '14', '22'] + \
    ['13', '15', '16', '17']
    # []
    # ['15', '17']
train_size = int(len(player_ids) * 0.55)
val_size = int((len(player_ids) - train_size) * 0.5)
test_size = len(player_ids) - train_size - val_size

criterion = nn.BCELoss()
scorer = accuracy_score
# scorer = roc_auc_score
# scorer = mean_squared_error
# criterion = nn.MSELoss()
classification = (scorer == roc_auc_score) or (scorer == accuracy_score)

# time_step_list = [5, 10, 20, 30, 40]  # 10 is already tested
time_step_list = [30]  # 10 is already tested
window_size_list = [300]
# window_size_list = [60, 120, 180, 300]
batch_size_list = [16]
hidden_size_list = [32]
# hidden_size_list = [32]
n_repeat_list = list(range(10))
# attention_list = [0, 1, 2]
# attention_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# attention_list = [0]
# attention_list = [4, 0, 2, 1]
attention_list = [2]
normalization_list = [1]
# normalization_list = [0, 1]
max_patience = 5
index_names = ['time_step', 'window_size', 'batch_size', 'hidden_size', 'attention', 'normalization', 'n_repeat']

multi_index_all = pd.MultiIndex.from_product([time_step_list, window_size_list, batch_size_list,
    hidden_size_list, attention_list, normalization_list, n_repeat_list],
                                             names=index_names)
df_results = pd.DataFrame(index=multi_index_all)
df_results['score_train'] = -1
df_results['score_val'] = -1
df_results['score_test'] = -1
df_results['best_epoch'] = -1
n_epoches = 50
batches4epoch = 20
plot = False
plot_predict = False
plot_attention = False
plot_target = False
modify_attention_array = False
# super_suffix = 'v3'
# super_suffix = 'time_step_2'
# super_suffix = 'mse'
# super_suffix = 'normalization_0'
super_suffix = 'prefinal'
# super_suffix = 'hidden_size_1'
recreate_dataset = False

# attention_sum_list_dict = {}
best_attentions_dict = {}
# from collections import defaultdict
# aucs4players = defaultdict(list)

# time_step = time_step_list[0]
for time_step in time_step_list:
    # if time_step != 5:
    if recreate_dataset:
        TIMESTEP_STRING = f'{time_step}s'
        print(TIMESTEP_STRING)
        print('part_0')
        command = f'python TimeseriesProcessing.py --TIMESTEP_STRING {TIMESTEP_STRING}'
        os.system(command)
        print('part_1')
        command = f'python TimeseriesMerging.py --TIMESTEP {time_step}'
        os.system(command)
        print('part_2')
        command = f'python TimeseriesAnalysis.py --TIMESTEP {time_step}'
        os.system(command)
        print('part_3')
        command = f'python TimeseriesFinalPreprocessing.py --TIMESTEP {time_step}'
        os.system(command)
        print('training')

        # time.sleep(1)

    for window_size, batch_size, hidden_size, attention, normalization, n_repeat in \
            itertools.product(window_size_list, batch_size_list, hidden_size_list, attention_list, normalization_list, n_repeat_list):
        # window_size, batch_size, hidden_size, attention, normalization, n_repeat = list(itertools.product(window_size_list, batch_size_list, hidden_size_list, attention_list, normalization_list,
        #                   n_repeat_list))[0]
        suffix = f'{time_step}_{window_size}_{batch_size}_{hidden_size}_{attention}_{normalization}_{n_repeat}_{super_suffix}'
        # attention_sum_list = []
        print(suffix)
        patience = 0
        val_score_best = 0
        train_score_best = 0
        test_score_best = 0
        epoch_best = 0
        stop_learning = False
        best_attentions4model = {}

        player_ids_train = np.random.choice(player_ids, size=train_size, replace=False)
        player_ids_test_and_val = np.array([player_id for player_id in player_ids if player_id not in player_ids_train])
        player_ids_val = np.random.choice(player_ids_test_and_val, size=val_size, replace=False)
        player_ids_test = np.array(
            [player_id for player_id in player_ids_test_and_val if player_id not in player_ids_val])

        # suffix = f'{batch_size}'
        data_dict_resampled_merged_with_target_scaled = joblib.load(
            f'data/data_dict_resampled_merged_with_target_scaled_{int(time_step)}')

        target_prefix = 'kills_proportion'
        target_columns = [column for column in data_dict_resampled_merged_with_target_scaled['10'].columns if
                          column.startswith(target_prefix)]
        # window_size = 300
        target_column_past = f'{target_prefix}_{window_size}_4past'
        target_column_future = f'{target_prefix}_{window_size}_4future'

        train_tensors_dict = {}

        # for player_id, df4train in data_dict_resampled_merged_with_target_scaled.items():
        for player_id in player_ids:
            df4train = data_dict_resampled_merged_with_target_scaled[player_id]
            train_tensors4player = {}

            mask2keep = df4train[target_column_future].notnull() & df4train[target_column_past].notnull()

            if mask2keep.sum() == 0:
                print(f'Not enogh data for player {player_id}')
                continue

            df4train = df4train.loc[mask2keep, :]
            df4train.fillna(0, inplace=True)

            target_future = df4train[target_column_future].values
            target_past = df4train[target_column_past].values
            #
            target = target_future - target_past
            # target_binary = (target_future < 0.1) * 1
            margin = 0  # 0
            target_binary = (target > margin) * 1
            # Possible targets:
            # better than average
            # 2 or 3 classes from: very bad, very good, average
            #

            # target = target_future

            # target_binary = (target > target.median()) * 1
            # target_binary.reset_index(drop=True, inplace=True)
            df4train.drop(columns=target_columns, inplace=True)
            df4train.reset_index(drop=True, inplace=True)
            features = list(df4train.columns)

            if plot_target:
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
                    breakpoints = breakpoints + [len(target_binary)-1]
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
                    ax[1].plot(x_points, target_binary[breakpoint_start:breakpoint_end+1], label='target_binary', color=color_2)
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

            train_tensors4player['input'] = torch.Tensor(df4train.values)
            if (scorer == roc_auc_score) or (scorer == accuracy_score):
                train_tensors4player['target'] = torch.Tensor(target_binary)  # FOR logloss metric
                train_tensors4player['target_raw'] = torch.Tensor(target)  # FOR logloss metric
            elif scorer == mean_squared_error:
                train_tensors4player['target'] = torch.Tensor(target)  # FOR logloss metric
                train_tensors4player['target_raw'] = torch.Tensor(target)  # FOR logloss metric
            # train_tensors4player['target'] = torch.Tensor(target)  # FOR MSE metric
            # train_tensors4player['target'] = torch.Tensor(target_binary)
            # train_tensors4player['target_raw'] = torch.Tensor(target_binary)
            train_tensors_dict[player_id] = train_tensors4player

        n_features = train_tensors4player['input'].shape[1]

        predictor = PredictorCell(input_size=n_features, hidden_size=hidden_size, attention=attention,
                                  normalization=normalization, classification=classification)
        opt = Adam(predictor.parameters())
        # opt = SGD(predictor.parameters(), lr=0.01)

        loss_list_dict = {
            'train': [],
            'val': [],
            'test': [],
        }
        auc_list_dict = {
            'train': [],
            'val': [],
            'test': [],
        }

        batch_generator = BatchGenerator(train_tensors_dict, list(train_tensors_dict.keys()))

        for n_epoch in range(n_epoches):
            if stop_learning:
                print('Early stopping...')
                break

            loss4epoch_dict = {
                'train': [],
                'val': [],
                'test': [],
            }
            auc4epoch_dict = {
                'train': [],
                'val': [],
                'test': [],
            }

            print(f'Epoch {n_epoch}')
            ##### np.random.shuffle(player_ids)  # It should be commented, but I'm not sure

            # for player_id in player_ids:
            # for n_batch in tqdm.tqdm(range(batches4epoch)):
            for n_batch in range(batches4epoch):
                predictor.reset_hidden()
                batch = batch_generator.get_batch(batch_size)
                batch_input, batch_target = batch
                opt.zero_grad()  # Check the location
                # for n_step in range(batch_size * 2):
                for n_step in range(len(batch_input)):
                    if n_step >= (len(batch_input) // 2):
                        predictor.train()
                    else:
                        predictor.eval()

                    tensor_input4step = batch_input[[n_step]]
                    target4step = batch_target[[n_step]]
                    output, hidden_state, cell_state, attention_weights = predictor(tensor_input4step)
                    # output, hidden_state, attention_weights = predictor(tensor_input4step)
                    # if n_step >= batch_size:  # hidden state is accumulated
                    if n_step >= (len(batch_input) // 2):  # hidden state is accumulated
                        predictor.train()
                        loss = criterion(output[0], target4step) / (len(batch_input) // 2)
                        loss.backward()
                        # opt.step()  # TODO: think where should it be (here or after the loop)
                else:
                    opt.step()

            predictor.eval()
            ### EVALUATION ON EVERY PLAYER
            for player_id in player_ids:
                if player_id not in train_tensors_dict:
                    continue

                train_on_this_player = False
                predictor.reset_hidden()

                # train_on_this_player = player_id in player_ids_train
                if player_id in player_ids_train:
                    mode = 'train'
                elif player_id in player_ids_val:
                    mode = 'val'
                elif player_id in player_ids_test:
                    mode = 'test'
                else:
                    raise ValueError(f'player_id {player_id} is not from train, val or test')

                # df4train = data_dict_resampled_merged_with_target_scaled[player_id]

                input = train_tensors_dict[player_id]['input']
                target = train_tensors_dict[player_id]['target']
                target_raw = train_tensors_dict[player_id]['target_raw']

                target_numpy = target.numpy()
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
                for n_step in range(len(input)):
                    # print(n_step)
                    if train_on_this_player:
                        opt.zero_grad()

                    tensor_input4step = input[[n_step]]
                    target4step = target[[n_step]]
                    output, hidden_state, cell_state, attention_weights = predictor(tensor_input4step)
                    predict_list4player.append(output[0].detach())
                    loss = criterion(output[0], target4step)
                    # if train_on_this_player:
                    #     loss.backward()
                    #     opt.step()

                    hidden_state_list4player.append(hidden_state.detach().numpy().ravel())
                    cell_state_list4player.append(cell_state.detach().numpy().ravel())
                    if attention:
                        attention_weights_list4player.append(attention_weights.detach().numpy().ravel())
                    outputs_list4player.append(output.detach()[0])
                    loss_list4player.append(loss.detach().item())

                loss4player = np.mean(loss_list4player)

                if scorer.__name__ == 'accuracy_score':
                    predict_binary = 1 * (np.array(predict_list4player) > 0.5)
                    auc4player = scorer(target.numpy().astype(int),  predict_binary)
                else:
                    auc4player = scorer(target.numpy(), np.array(predict_list4player))

                # fig, ax = plt.subplots()
                if attention:
                    attention_array = np.array(attention_weights_list4player)
                    if n_epoch == epoch_best:
                        best_attentions4model[player_id] = attention_array.mean(axis=0)
                        # attention_sum_list.append(attention_array.mean(axis=0))
                hidden_array = np.array(hidden_state_list4player)
                cell_array = np.array(cell_state_list4player)
                outputs_array = np.array(outputs_list4player)

                if plot_attention and attention:
                    if player_id in player_ids_train:
                        mode = 'train'
                    elif player_id in player_ids_val:
                        mode = 'val'
                    elif player_id in player_ids_test:
                        mode = 'test'

                    # plt.close()
                    fig, ax = plt.subplots(5, 1, squeeze=False, figsize=(12, 14), sharex=True,
                                           gridspec_kw={'height_ratios': [1, 1, 0.2, 0.2, 0.2]})


                    # eps = np.random.uniform(attention_array.shape[0])
                    # attention_array_modified = attention_array /
                    # ax.set_yticks(np.arange(len(features)), features)
                    # margin = 0.06
                    # cbar_width = 0.05
                    # ax = fig.add_axes([margin, margin, 1 - 2 * margin - cbar_width, 1 - margin])
                    # cbaxes = fig.add_axes([1 - 1 * margin - cbar_width, margin, cbar_width, 1 - margin])

                    # content = fig.add_axes([0.05, 0.1, 0.8, 0.01])

                    # for i in [0, 1]:
                    #     ax[i, 1].tick_params(
                    #         axis='both',  # changes apply to the x-axis
                    #         which='both',  # both major and minor ticks are affected
                    #         bottom=False,  # ticks along the bottom edge are off
                    #         top=False,  # ticks along the top edge are off
                    #         left=False,
                    #         right=False,
                    #         labelbottom=False,
                    #     labelleft=False)

                    # fig.delaxes(ax[1, 1])
                    # fig.delaxes(ax[0, 1])
                    fontsize = 15
                    # pos = ax[0, 0].imshow(attention_array.T, aspect='auto', cmap='Blues', vmin=0.05, vmax=0.15)
                    pos = ax[0, 0].imshow(attention_array.T, aspect='auto', cmap='Blues', vmin=0.2, vmax=0.8)
                    ax[0, 0].set_title('Input Attention', fontsize=fontsize+2)
                    ax[0, 0].set_xlabel('Time Step Number', fontsize=fontsize)
                    ax[0, 0].set_yticks(np.arange(len(features)))
                    ax[0, 0].set_yticklabels(features_pretty, fontsize=fontsize-2)
                    ax[0, 0].set_xlim(0, len(attention_array.T))
                    # ax[0].set_ytickslabels(np.arange(len(features)), features_pretty)
                    # plt.yticks(np.arange(len(features)), features_pretty)
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    divider = make_axes_locatable(ax[0, 0])
                    cax = divider.append_axes("right", size="2%", pad=0.06)
                    plt.colorbar(pos, cax=cax)
                    # colorbar = fig.colorbar(pos, pad=-2, ax=ax[0, 1])

                    # colorbar.ax.set_ylabel('attention weight', rotation=270, labelpad=10)


                    # ax_new = ax.a
                    ax[1, 0].imshow(hidden_array.T, aspect='auto', cmap=None)  # , vmin=0, vmax=1)
                    ax[1, 0].set_title('Hidden State', fontsize=fontsize+2)
                    ax[1, 0].set_ylabel('Neuron Number', fontsize=fontsize)
                    ax[1, 0].set_xlabel('Time Step Number', fontsize=fontsize)
                    divider = make_axes_locatable(ax[1, 0])
                    cax = divider.append_axes("right", size="2%", pad=0.08)
                    fig.delaxes(cax)
                    # ax[1].autoscale(tight=True)
                    # ax[1].set_xlim(right=1)
                    # plt.tight_layout()
                    # plt.savefig(pic_folder + f'attention/hidden_state_player_{player_id}_epoch_{n_epoch}_{suffix}.png')
                    # plt.close()

                    if scorer == roc_auc_score:
                        ax[2, 0].plot(target_raw, label='Perfomance difference', color='teal')
                        ax[2, 0].set_title('Perfomance difference', fontsize=fontsize + 2)
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
                                       label='Binary Target', color='peru')
                        ax[3, 0].scatter(list(range(len(target))), target, s=5, color='peru')


                        ax[3, 0].set_title('Binary Target', fontsize=fontsize+2)
                        ax[3, 0].yaxis.set_major_locator(MultipleLocator(1))
                        ax[4, 0].plot(outputs_array, label='Predict', color='darkorange')
                        ax[4, 0].set_title('Predict', fontsize=fontsize + 2)

                        for i in [2, 3, 4]:
                            divider = make_axes_locatable(ax[i, 0])
                            cax = divider.append_axes("right", size="2%", pad=0.08)
                            fig.delaxes(cax)

                    ax[-1, 0].set_xlabel('Time Step Number')

                    # plt.tight_layout(rect=[0, 0, 1.04, 1])
                    plt.tight_layout(rect=[0, 0, 1, 1])
                    # plt.show()
                    fig.savefig(pic_folder + f'attention/input_attention_player_{player_id}_epoch_{n_epoch}_{suffix}_from_{mode}.png')
                    plt.close()

                if plot_predict:
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

                if plot:
                    plt.imshow(hidden_array.T, aspect='auto', cmap=None)  # , vmin=0, vmax=1)
                    plt.tight_layout()
                    plt.savefig(pic_folder + f'attention/hidden_state_player_{player_id}_epoch_{n_epoch}_{suffix}.png')
                    plt.close()

                    plt.imshow(cell_array.T, aspect='auto', cmap=None)  # , vmin=0, vmax=1)
                    plt.tight_layout()
                    plt.savefig(pic_folder + f'attention/cell_array_player_{player_id}_epoch_{n_epoch}_{suffix}.png')
                    plt.close()

                    # plt.plot(loss_list4player, label='loss')
                    # plt.legend()
                    # plt.tight_layout()
                    # plt.savefig(pic_folder + f'attention/loss_player_{player_id}_epoch_{n_epoch}_{suffix}.png')
                    # plt.close()




                loss4epoch_dict[mode].append(loss4player)
                auc4epoch_dict[mode].append(auc4player)
                # if (player_id in player_ids_test) or (player_id in player_ids_val):
                #     aucs4players[player_id].append(auc4player)
                #     # print(f"auc 4 player {player_id} = {round(auc4player, 2)}")

            # for mode in ['val']: #  ['train', 'val']:
            for mode in ['train', 'val', 'test']:
                if len(loss4epoch_dict[mode]):
                    loss4epoch4mode = np.mean(loss4epoch_dict[mode])
                    # print(f'loss4epoch4_{mode}={loss4epoch4mode}')
                    loss_list_dict[mode].append(loss4epoch4mode)

                if len(auc4epoch_dict[mode]):
                    auc4epoch4mode = np.mean(auc4epoch_dict[mode])
                    print(f'auc_{mode}={round(auc4epoch4mode, 3)}')
                    auc_list_dict[mode].append(auc4epoch4mode)
                    if mode == 'val':
                        val_score_new = auc4epoch4mode

                        if val_score_new > val_score_best:
                            val_score_best = np.mean(auc4epoch_dict['val'])
                            test_score_best = np.mean(auc4epoch_dict['test'])
                            train_score_best = np.mean(auc4epoch_dict['train'])
                            epoch_best = n_epoch
                            patience = 0
                        else:
                            patience += 1
                            if patience >= max_patience:
                                stop_learning = True

        index_array = [[time_step], [window_size], [batch_size], [hidden_size], [attention], [normalization], [n_repeat]]
        multi_index = pd.MultiIndex.from_arrays(index_array, names=index_names)
        # series = pd.Series([test_score_best], index=multi_index, name='scores')
        # series.to_csv(f'data/series_{suffix}.csv', header=True)

        df4params = pd.DataFrame([[train_score_best, val_score_best, test_score_best, epoch_best]],
                                 index=index_array, columns=['score_train', 'score_val', 'score_test', 'best_epoch'])
        # df4params.to_csv(f'data/series_{suffix}.csv', header=True)

        df_results.loc[multi_index, 'score_train'] = train_score_best
        df_results.loc[multi_index, 'score_val'] = val_score_best
        df_results.loc[multi_index, 'score_test'] = test_score_best
        df_results.loc[multi_index, 'best_epoch'] = epoch_best

        # attention_sum_list_dict[suffix] = attention_sum_list
        best_attentions_dict[suffix] = best_attentions4model

        for mode in ['train', 'val', 'test']:
            if len(loss_list_dict[mode]):
                if plot:
                    plt.plot(loss_list_dict[mode], label=mode)
                    plt.savefig(pic_folder + f'_{mode}_loss_list_last_{suffix}.png')
                    plt.close()
            if len(auc_list_dict[mode]):
                if plot:
                    plt.plot(auc_list_dict[mode], label=mode)
                    plt.savefig(pic_folder + f'_{mode}_auc_list_last_{suffix}.png')
                    plt.close()

# df_results.to_csv(f'data/df_results_{super_suffix}.csv')

all_attentions_list = []
for model_name, attentions4players in best_attentions_dict.items():
    for player_id, attentions4player in attentions4players.items():
        all_attentions_list = all_attentions_list + [attentions4player]#.reshape(15, 1)


# mean_att = np.mean(attention_sum_list_dict['30_300_16_32_2_1_3_prefinal'], axis=0)
mean_att = np.mean(all_attentions_list[::], axis=0)
index_order = np.argsort(mean_att)
mean_att = np.median(all_attentions_list[::], axis=0)
index_order = np.argsort(mean_att)


color_att = 'olivedrab'
color_att = 'olive'
color_att = 'darkslategrey'
color_att = 'darkcyan'

margin = 0.018
fontsize = 14
plt.close()

plt.figure(figsize=(8, 5))
y_ticks = list(range(len(index_order)))
plt.barh(y_ticks, mean_att[index_order], color=color_att)
plt.xlim((mean_att.min() - margin, mean_att.max() + margin * 0.7))
plt.yticks(y_ticks, np.array(features_pretty)[index_order], fontsize=fontsize)
plt.xlabel('Mean Attention', fontsize=fontsize+3)
plt.title('Feature Importance', fontsize=fontsize+6)
plt.tight_layout()
plt.savefig('pic/attention_importance_v0.png')


plt.interactive(True)


plt.barh(mean_att[index_order], np.array(features_pretty)[index_order])






#########################
####### CLASSICAL ML ZONE
# train_test_splitter = StratifiedKFold(n_splits=5)
from hmmlearn import hmm

# time_step_list = [5, 10, 20, 30, 40]  # 10 is already tested
time_step_list = [30]  # 10 is already tested
# window_size_list = [60, 120, 180, 300, 600]
window_size_list = [300]
# window_size_list = [300]
n_repeat_list = list(range(5))
plot = False
results_list = []
rf_0 = RandomForestClassifier(n_estimators=100, max_depth=3)
# rf_1 = RandomForestClassifier(n_estimators=100, max_depth=5)
lr = LogisticRegression(solver='lbfgs')
svm = SVC(probability=True, gamma='auto')
# hmm_model_0 = hmm.GaussianHMM(n_components=10)  # , covariance_type="full")
# hmm_model_1 = hmm.GaussianHMM(n_components=3)  # , covariance_type="full")
# hmm_model_2 = hmm.GaussianHMM(n_components=5)  # , covariance_type="full")


alg_list = [lr, rf_0, svm] #, hmm_model_0, hmm_model_1, hmm_model_2]
alg_names_list = ['Logistic Regression', 'Random Forest', 'SVM'] # , 'Hidden Markov Model', 'Hidden Markov Model', 'Hidden Markov Model']

index_names = ['time_step', 'window_size', 'alg_name', 'n_repeat']
multi_index_all = pd.MultiIndex.from_product([time_step_list, window_size_list, alg_names_list, n_repeat_list], names=index_names)
df_results = pd.DataFrame(index=multi_index_all)
# df_results['score_train'] = -1
df_results['score_val'] = -1
suffix = 'window_size_0'

for time_step, window_size, n_repeat in itertools.product(time_step_list, window_size_list, n_repeat_list):
    # print(window_size)
    data_dict_resampled_merged_with_target_scaled = joblib.load(
                f'data/data_dict_resampled_merged_with_target_scaled_{int(time_step)}')

    target_prefix = 'kills_proportion'
    target_columns = [column for column in data_dict_resampled_merged_with_target_scaled['10'].columns if
                      column.startswith(target_prefix)]
    # window_size = 300
    target_column_past = f'{target_prefix}_{window_size}_4past'
    target_column_future = f'{target_prefix}_{window_size}_4future'

    train_tensors_dict = {}

    # for player_id, df4train in data_dict_resampled_merged_with_target_scaled.items():
    for player_id in player_ids:
        df4train = data_dict_resampled_merged_with_target_scaled[player_id]
        train_tensors4player = {}

        mask2keep = df4train[target_column_future].notnull() & df4train[target_column_past].notnull()

        if mask2keep.sum() == 0:
            print(f'Not enough data for player {player_id}')
            continue

        df4train = df4train.loc[mask2keep, :]
        df4train.fillna(0, inplace=True)

        target_future = df4train[target_column_future].values
        target_past = df4train[target_column_past].values
        #
        target = target_future - target_past
        # target_binary = (target_future < 0.1) * 1
        margin = 0  # 0
        target_binary = (target > margin) * 1
        # Possible targets:
        # better than average
        # 2 or 3 classes from: very bad, very good, average
        #

        # target = target_future

        # target_binary = (target > target.median()) * 1
        # target_binary.reset_index(drop=True, inplace=True)
        df4train.drop(columns=target_columns, inplace=True)
        df4train.reset_index(drop=True, inplace=True)
        features = list(df4train.columns)

        # if plot:
        #     plt.close()
        #     plt.plot(target_binary, label='target_binary')
        #     plt.plot(target, label='target')
        #     # plt.axhline(target.mean(), label='target_mean', color='green')
        #     plt.legend()
        #     plt.savefig(pic_folder + f'target_player_{player_id}')

        train_tensors4player['input'] = torch.Tensor(df4train.values)
        train_tensors4player['target'] = torch.Tensor(target_binary)  # FOR logloss metric
        train_tensors4player['target_raw'] = torch.Tensor(target)  # FOR logloss metric
        train_tensors4player['target_future'] = torch.Tensor(target_future)  # FOR logloss metric
        # train_tensors4player['target'] = torch.Tensor(target)  # FOR MSE metric
        # train_tensors4player['target'] = torch.Tensor(target_binary)
        # train_tensors4player['target_raw'] = torch.Tensor(target_binary)
        train_tensors_dict[player_id] = train_tensors4player


    train_test_splitter = KFold(n_splits=5, shuffle=True)
    players = list(train_tensors_dict.keys())
    # auc_scores_list = []


    for alg, alg_name in zip(alg_list, alg_names_list):
        # alg_name = alg.__class__.__name__
        auc_scores4alg = []
        dummy_scores = []


        if alg_name == 'Hidden Markov Model':
            ### For hidden states visualization for players
            for player_id in player_ids:
                plt.close()

                xx_train = train_tensors_dict[player_id]['input']
                yy_train = train_tensors_dict[player_id]['target_future']
                alg.fit(xx_train)
                predict_hard = alg.predict(xx_train)
                labels = np.unique(predict_hard)

                colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'black', 'teal', 'brown']
                for label, color in zip(labels, colors):
                    indexes = np.nonzero(predict_hard == label)[0]
                    indexes = np.sort(indexes)
                    for index in indexes:
                        if index + 1 == len(predict_hard):
                            continue

                        x_data = [index, index + 1]
                        y_data = [yy_train[index], yy_train[index + 1]]

                        plt.plot(x_data, y_data, color=color)

                # plt.plot(predict_probas.argmax(axis=1), label='Predict')
                # plt.plot(target, label='Target')
                plt.title(alg_name)
                # plt.legend()
                plt.tight_layout()
                plt.savefig(f'pic/hmm_segmentation/{alg.n_components}_components_player_{player_id}_future.png')


        for train_players, test_players in train_test_splitter.split(players):
            # n_features = train_tensors_dict[players[0]]['input'].shape[1]
            # x_train = np.empty(shape=(0, n_features))
            # for train_player in train_players:
            x_train = np.concatenate([train_tensors_dict[players[player_id_index]]['input'] for player_id_index in train_players])
            y_train = np.concatenate([train_tensors_dict[players[player_id_index]]['target'] for player_id_index in train_players])


            # if alg_name == 'Hidden Markov Model':
            #     alg.fit(x_train)  # , [len(x_train)] * 15)
            #     # predict = alg.predict(x_val)
            #     # alg.predict_proba(x_val)
            #     # print(len(np.unique(predict)))
            # else:
            alg.fit(x_train, y_train)

            for test_player in [players[player_id_index] for player_id_index in test_players]:


                x_val = train_tensors_dict[test_player]['input']
                y_val = train_tensors_dict[test_player]['target']




                # y_val_raw = train_tensors_dict[test_player]['target_raw']
                if len(np.unique(y_val)) < 2:
                    continue
                else:
                    # predict_probas = alg.predict_proba(x_val)
                    # predict = predict_probas[:, 1]
                    predict = alg.predict(x_val)
                    # predict_hard = alg.predict(x_val)
                    dummy_shift = window_size // time_step
                    dummy_predict = [0] * dummy_shift + list(np.array(y_val).astype(int))[:-dummy_shift]
                    dummy_predict = np.array(dummy_predict)
                    dummy_predict = 1 - dummy_predict


                # if alg_name == 'Hidden Markov Model':
                #     plt.close()
                #
                #     labels = np.unique(predict_hard)
                #     colors = ['red', 'green', 'blue', 'yellow', 'magenta']
                #     for label, color in zip(labels, colors):
                #         indexes = np.nonzero(predict_hard == label)[0]
                #         indexes = np.sort(indexes)
                #         for index in indexes:
                #             if index + 1 == len(predict_hard):
                #                 continue
                #
                #             x_data = [index, index + 1]
                #             y_data = [y_val_raw[index], y_val_raw[index + 1]]
                #
                #             plt.plot(x_data, y_data, color=color)
                #
                #     # plt.plot(predict_probas.argmax(axis=1), label='Predict')
                #     # plt.plot(target, label='Target')
                #     plt.title(alg_name)
                #     # plt.legend()
                #     plt.tight_layout()
                #     plt.savefig(f'pic/hmm_segmentation/{alg.n_components}_components_player_{test_player}.png')

            # predict = np.array([0] * 18 + list(y_val[:-18]))
                auc_score = scorer(y_val, predict)
                dummy_score = scorer(np.array(y_val), dummy_predict)
            # print(f'{alg.__class__.__name__}:', auc_score)
                auc_scores4alg.append(auc_score)
                dummy_scores.append(dummy_score)

        dummy_score = np.mean(dummy_scores)
        alg_score = np.mean(auc_scores4alg)
        print(f'{alg_name}: {round(alg_score, 3)}')
        print(f'{"Dummy Score"}: {round(dummy_score, 3)}')

        index_array = [[time_step], [window_size], [alg_name], [n_repeat]]
        # index_array = [[5], [120], [2], [8]]

        # = val_score_best
        multi_index = pd.MultiIndex.from_arrays(index_array, names=index_names)

        df_results.loc[multi_index] = alg_score



df_results.to_csv(f'data/df_results_classic_{suffix}.csv')







    # knn = KNeighborsClassifier()
    # knn.fit(x_train, y_train)
    # predict = lr.predict(x_val)
    # auc_score = scorer(y_val, predict)
    # print('KNN:', auc_score)



    # cross_val_score(lr, x_train, y_train, scoring='roc_auc', cv=5)




#
# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#
#         # self.embedding = nn.Embedding(input_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size)
#
#     def forward(self, input, hidden):
#         embedded = self.embedding(input).view(1, 1, -1)
#         output = embedded
#         output, hidden = self.gru(output, hidden)
#         return output, hidden
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)
#
#
# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
#         super(AttnDecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length
#
#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, input, hidden, encoder_outputs):
#         embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)
#
#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0))
#
#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)
#
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#
#         output = F.log_softmax(self.out(output[0]), dim=1)
#         return output, hidden, attn_weights
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)
