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
import time
from sklearn.svm import SVC

plt.interactive(False)
pd.options.display.max_columns = 15
pic_folder = 'pic/'

class PredictorCell(nn.Module):

    def __init__(self, input_size=9, hidden_size=16, attention=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention = attention

        # self.gru = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.lstm = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        # self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size)
        self.reset_hidden()
        if self.attention:
            self.attention_0 = nn.Linear(self.input_size + self.hidden_size,
                                         self.input_size)  # self.input_size + self.hidden_size)
            self.attention_1 = nn.Linear(self.input_size, self.input_size)

        self.hidden2output_0 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden2output_1 = nn.Linear(self.hidden_size, 1)

    def forward(self, input):
        # print([input, self.hidden])
        if self.attention:
            input_and_hidden = torch.cat([input, self.hidden], dim=1)
            # print(input_and_hidden.shape)
            attention_logits_0 = self.attention_0(input_and_hidden)
            attention_logits_0 = torch.relu(attention_logits_0)
            attention_logits = self.attention_1(attention_logits_0)

            if self.attention == 1:
                attention_weights = F.softmax(attention_logits, dim=1)
            elif self.attention == 2:
                attention_weights = torch.sigmoid(attention_logits) # , dim=1)
                # print(attention_weights)
            else:
                raise ValueError(f'self.attention = {self.attention} is not supported')


            input_with_attention = input * attention_weights  # Check this. And everything else.
        else:
            input_with_attention = input
            attention_weights = 0
        # attention_weights = 0

        # hidden = self.gru(input_with_attention, self.hidden)
        # input_with_attention = input
        hidden, cell = self.lstm(input_with_attention, (self.hidden, self.cell))

        output = self.hidden2output_0(hidden)
        output = torch.relu(output)
        output = self.hidden2output_1(output)
        # output = self.hidden2output(hidden)
        self.hidden = hidden.detach()
        self.cell = cell.detach()

        output = torch.sigmoid(output)  # To [0, 1] interval

        return output, self.hidden, self.cell, attention_weights  # , hidden

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


# player_ids = list(data_dict_resampled_merged_with_target_scaled.keys())
player_ids = ['9', '0', '11', '7', '6', '1', '10', '19', '8', '21', '4', '3', '12', '2', '5', '14', '22'] + \
    ['13', '15', '16', '17']
train_size = int(len(player_ids) * 0.55)
val_size = int((len(player_ids) - train_size) * 0.5)
test_size = len(player_ids) - train_size - val_size

criterion = nn.BCELoss()
# criterion = nn.MSELoss()
# time_step_list = [5, 10, 30]
# window_size_list = [120, 300, 600]
# batch_size_list = [2, 16, 128]
# hidden_size_list = [2, 8, 32]
# time_step_list = [10, 20]  # 10 is already tested
# window_size_list = [180, 300]
# batch_size_list = [8, 64, 256]
# hidden_size_list = [16, 32, 64]
# n_repeat_list = list(range(3))
time_step_list = [5, 10, 20, 30, 40]  # 10 is already tested
window_size_list = [300]
batch_size_list = [64]
# hidden_size_list = [8, 16, 32, 64]
hidden_size_list = [32]
n_repeat_list = list(range(10))
attention_list = [0, 1, 2]
max_patience = 3
index_names = ['time_step', 'window_size', 'batch_size', 'hidden_size', 'attention', 'n_repeat']

multi_index_all = pd.MultiIndex.from_product([time_step_list, window_size_list, batch_size_list,
    hidden_size_list, attention_list, n_repeat_list],
                                             names=index_names)
df_results = pd.DataFrame(index=multi_index_all)
df_results['score_train'] = -1
df_results['score_val'] = -1
df_results['score_test'] = -1
df_results['best_epoch'] = -1
# df_results.loc[(30, 600, 2, 8), 'scores'] = 1
n_epoches = 50
batches4epoch = 20
plot = False
# super_suffix = 'v3'
super_suffix = 'time_step_1'
recreate_dataset = False
# time_step_list = [5, 10, 40]

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

    for window_size, batch_size, hidden_size, attention, n_repeat in \
            itertools.product(window_size_list, batch_size_list, hidden_size_list, attention_list, n_repeat_list):
    # for n_repeat in range(3, 5):
        # window_size, batch_size, hidden_size, n_repeat = list(itertools.product(
        #     window_size_list, batch_size_list, hidden_size_list, n_repeat_list))[0]
        # if (time_step == 5) and (window_size == 120) and (batch_size < 128):
        #     continue
        suffix = f'{time_step}_{window_size}_{batch_size}_{hidden_size}_{attention}_{n_repeat}_{super_suffix}'
        print(suffix)
        patience = 0
        val_score_best = 0
        train_score_best = 0
        test_score_best = 0
        epoch_best = 0
        stop_learning = False

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

        for player_id, df4train in data_dict_resampled_merged_with_target_scaled.items():
            train_tensors4player = {}
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

            if plot:
                plt.close()
                plt.plot(target_binary, label='target_binary')
                plt.plot(target, label='target')
                # plt.axhline(target.mean(), label='target_mean', color='green')
                plt.legend()
                plt.savefig(pic_folder + f'target_player_{player_id}')

            train_tensors4player['input'] = torch.Tensor(df4train.values)
            train_tensors4player['target'] = torch.Tensor(target_binary)  # FOR logloss metric
            train_tensors4player['target_raw'] = torch.Tensor(target)  # FOR logloss metric
            # train_tensors4player['target'] = torch.Tensor(target)  # FOR MSE metric
            # train_tensors4player['target'] = torch.Tensor(target_binary)
            # train_tensors4player['target_raw'] = torch.Tensor(target_binary)
            train_tensors_dict[player_id] = train_tensors4player

        n_features = train_tensors4player['input'].shape[1]

        predictor = PredictorCell(input_size=n_features, hidden_size=hidden_size, attention=attention)
        opt = Adam(predictor.parameters())
        # opt = SGD(predictor.parameters(), lr=0.01)

        # player_id = '1'  # DEBUG
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

        # loss4epoch_list = []
        # attention_weights4epoch_list = []
        # hidden_state4epoch_list = []
        # outputs4epoch_list = []

        batch_generator = BatchGenerator(train_tensors_dict, player_ids_train)

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
            # np.random.shuffle(player_ids)  # It should be commented, but I'm not sure

            # for player_id in player_ids:
            for n_batch in tqdm.tqdm(range(batches4epoch)):
                predictor.reset_hidden()
                batch = batch_generator.get_batch(batch_size)
                batch_input, batch_target = batch

                opt.zero_grad()  # Check the location
                # for n_step in range(batch_size * 2):
                for n_step in range(len(batch_input)):
                    tensor_input4step = batch_input[[n_step]]
                    target4step = batch_target[[n_step]]
                    output, hidden_state, cell_state, attention_weights = predictor(tensor_input4step)
                    # if n_step >= batch_size:  # hidden state is accumulated
                    if n_step >= (len(batch_input) // 2):  # hidden state is accumulated
                        loss = criterion(output[0], target4step) / (len(batch_input) // 2)
                        loss.backward()
                else:
                    opt.step()

            ### EVALUATION ON EVERY PLAYER
            # for player_id in player_ids:
            for player_id in player_ids:
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

                hidden_state_list4player = []
                cell_state_list4player = []
                attention_weights_list4player = []
                outputs_list4player = []
                loss_list4player = []
                predict_list4player = []

                for n_step in tqdm.tqdm(range(len(input))):
                    # print(n_step)
                    if train_on_this_player:
                        opt.zero_grad()

                    tensor_input4step = input[[n_step]]
                    target4step = target[[n_step]]
                    output, hidden_state, cell_state, attention_weights = predictor(tensor_input4step)
                    predict_list4player.append(output[0].detach())
                    loss = criterion(output[0], target4step)
                    if train_on_this_player:
                        loss.backward()
                        opt.step()

                    hidden_state_list4player.append(hidden_state.detach().numpy().ravel())
                    cell_state_list4player.append(cell_state.detach().numpy().ravel())
                    if attention:
                        attention_weights_list4player.append(attention_weights.detach().numpy().ravel())
                    outputs_list4player.append(output.detach()[0])
                    loss_list4player.append(loss.detach().item())

                loss4player = np.mean(loss_list4player)
                auc4player = roc_auc_score(target.numpy(), np.array(predict_list4player))
                # fig, ax = plt.subplots()
                if attention:
                    attention_array = np.array(attention_weights_list4player)
                hidden_array = np.array(hidden_state_list4player)
                cell_array = np.array(cell_state_list4player)
                outputs_array = np.array(outputs_list4player)

                if plot:
                    if attention:
                        plt.yticks(np.arange(len(features)), features)
                        plt.imshow(attention_array.T, aspect='auto', cmap='Blues', vmin=0, vmax=1)
                        plt.tight_layout()
                        plt.savefig(pic_folder + f'input_attention_player_{player_id}_epoch_{n_epoch}_{suffix}.png')
                        plt.close()

                    plt.imshow(hidden_array.T, aspect='auto', cmap=None)  # , vmin=0, vmax=1)
                    plt.tight_layout()
                    plt.savefig(pic_folder + f'hidden_state_player_{player_id}_epoch_{n_epoch}_{suffix}.png')
                    plt.close()

                    plt.imshow(cell_array.T, aspect='auto', cmap=None)  # , vmin=0, vmax=1)
                    plt.tight_layout()
                    plt.savefig(pic_folder + f'cell_array_player_{player_id}_epoch_{n_epoch}_{suffix}.png')
                    plt.close()

                    plt.plot(outputs_array, label='Predict')
                    plt.plot(target, label='Binary Target')
                    plt.plot(target_raw, label='Perfomance difference')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(pic_folder + f'predictions_player_{player_id}_epoch_{n_epoch}_{suffix}.png')
                    plt.close()

                    plt.plot(loss_list4player, label='loss')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(pic_folder + f'loss_player_{player_id}_epoch_{n_epoch}_{suffix}.png')
                    plt.close()

                loss4epoch_dict[mode].append(loss4player)
                auc4epoch_dict[mode].append(auc4player)

            # for mode in ['val']: #  ['train', 'val']:
            for mode in ['train', 'val', 'test']:
                if len(loss4epoch_dict[mode]):
                    loss4epoch4mode = np.mean(loss4epoch_dict[mode])
                    # print(f'loss4epoch4_{mode}={loss4epoch4mode}')
                    loss_list_dict[mode].append(loss4epoch4mode)

                if len(auc4epoch_dict[mode]):
                    auc4epoch4mode = np.mean(auc4epoch_dict[mode])
                    print(f'auc4epoch_dict_{mode}={auc4epoch4mode}')
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

        # result_array = [[time_step], [window_size], [batch_size], [hidden_size]]
        index_array = [[time_step], [window_size], [batch_size], [hidden_size], [attention], [n_repeat]]
        # index_array = [[5], [120], [2], [8]]

        # = val_score_best
        multi_index = pd.MultiIndex.from_arrays(index_array, names=index_names)
        # series = pd.Series([val_score_best], index=multi_index, name='scores')
        # series.to_csv(f'data/series_{suffix}.csv', header=True)

        df4params = pd.DataFrame([[train_score_best, val_score_best, test_score_best, epoch_best]],
                                 index=index_array, columns=['score_train', 'score_val', 'score_test', 'best_epoch'])
        df4params.to_csv(f'data/series_{suffix}.csv', header=True)

        df_results.loc[multi_index, 'score_train'] = train_score_best
        df_results.loc[multi_index, 'score_val'] = val_score_best
        df_results.loc[multi_index, 'score_test'] = test_score_best
        df_results.loc[multi_index, 'best_epoch'] = epoch_best

        for mode in ['train', 'val', 'test']:
            if len(loss_list_dict[mode]):
                plt.plot(loss_list_dict[mode], label=mode)
                plt.savefig(pic_folder + f'_{mode}_loss_list_last_{suffix}.png')
                plt.close()
            if len(auc_list_dict[mode]):
                plt.plot(auc_list_dict[mode], label=mode)
                plt.savefig(pic_folder + f'_{mode}_auc_list_last_{suffix}.png')
                plt.close()

df_results.to_csv(f'data/df_results_{super_suffix}.csv')










#########################
####### CLASSICAL ML ZONE
# train_test_splitter = StratifiedKFold(n_splits=5)

time_step_list = [5, 10, 20, 30, 40]  # 10 is already tested
window_size_list = [300]
n_repeat_list = list(range(10))
plot = False
results_list = []
rf_0 = RandomForestClassifier(n_estimators=100, max_depth=3)
rf_1 = RandomForestClassifier(n_estimators=100, max_depth=5)
lr = LogisticRegression(solver='lbfgs')
svm = SVC(probability=True, gamma='auto')
alg_list = [lr, rf_0, rf_1, svm]
alg_names_list = ['Logistic Regression', 'Random Forest 0', 'Random Forest 1', 'SVM']

index_names = ['time_step', 'window_size', 'alg_name', 'n_repeat']
multi_index_all = pd.MultiIndex.from_product([time_step_list, window_size_list, alg_names_list, n_repeat_list], names=index_names)
df_results = pd.DataFrame(index=multi_index_all)
# df_results['score_train'] = -1
df_results['score_val'] = -1
suffix = 'time_step'

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

    for player_id, df4train in data_dict_resampled_merged_with_target_scaled.items():
        train_tensors4player = {}
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

        if plot:
            plt.close()
            plt.plot(target_binary, label='target_binary')
            plt.plot(target, label='target')
            # plt.axhline(target.mean(), label='target_mean', color='green')
            plt.legend()
            plt.savefig(pic_folder + f'target_player_{player_id}')

        train_tensors4player['input'] = torch.Tensor(df4train.values)
        train_tensors4player['target'] = torch.Tensor(target_binary)  # FOR logloss metric
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
        for train_players, test_players in train_test_splitter.split(players):
            # n_features = train_tensors_dict[players[0]]['input'].shape[1]
            # x_train = np.empty(shape=(0, n_features))
            # for train_player in train_players:
            x_train = np.concatenate([train_tensors_dict[players[player_id_index]]['input'] for player_id_index in train_players])
            y_train = np.concatenate([train_tensors_dict[players[player_id_index]]['target'] for player_id_index in train_players])

            x_val = np.concatenate([train_tensors_dict[players[player_id_index]]['input'] for player_id_index in test_players])
            y_val = np.concatenate([train_tensors_dict[players[player_id_index]]['target'] for player_id_index in test_players])


            alg.fit(x_train, y_train)
            predict = alg.predict_proba(x_val)[:, 1]

            # predict = np.array([0] * 18 + list(y_val[:-18]))
            auc_score = roc_auc_score(y_val, predict)
            # print(f'{alg.__class__.__name__}:', auc_score)
            auc_scores4alg.append(auc_score)

        alg_score = np.mean(auc_scores4alg)
        print(f'{alg_name}: {round(alg_score, 3)}')

        index_array = [[time_step], [window_size], [alg_name], [n_repeat]]
        # index_array = [[5], [120], [2], [8]]

        # = val_score_best
        multi_index = pd.MultiIndex.from_arrays(index_array, names=index_names)

        df_results.loc[multi_index] = alg_score

df_results.to_csv('data/df_results_classic_{suffix}.csv')







    # knn = KNeighborsClassifier()
    # knn.fit(x_train, y_train)
    # predict = lr.predict(x_val)
    # auc_score = roc_auc_score(y_val, predict)
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
