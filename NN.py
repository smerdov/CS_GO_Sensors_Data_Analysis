import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import tqdm

plt.interactive(True)
pd.options.display.max_columns = 15
pic_folder = 'pic/'

data_dict_resampled_merged_with_target_scaled = joblib.load('data/data_dict_resampled_merged_with_target_scaled')

plot = False
target_prefix = 'kills_proportion'
target_columns = [column for column in data_dict_resampled_merged_with_target_scaled['10'].columns if
                  column.startswith(target_prefix)]
window_size = 180
target_column_past = f'{target_prefix}_{window_size}_4past'
target_column_future = f'{target_prefix}_{window_size}_4future'

train_tensors_dict = {}

for player_id, df4train in data_dict_resampled_merged_with_target_scaled.items():
    train_tensors4player = {}
    df4train.fillna(0, inplace=True)

    target_past = df4train[target_column_past].values
    target_future = df4train[target_column_future].values

    target = target_future - target_past
    target_binary = (target > 0) * 1

    # target_binary = (target > target.median()) * 1
    # target_binary.reset_index(drop=True, inplace=True)
    df4train.drop(columns=target_columns, inplace=True)
    df4train.reset_index(drop=True, inplace=True)
    features = list(df4train.columns)

    if plot:
        plt.close()
        plt.plot(target_binary, label='target_binary')
        plt.plot(target, label='target')
        plt.axhline(target.mean(), label='target_mean', color='green')
        plt.legend()
        plt.savefig(pic_folder + f'target_player_{player_id}')

    train_tensors4player['input'] = torch.Tensor(df4train.values)
    train_tensors4player['target'] = torch.Tensor(target_binary)
    # train_tensors4player['target_raw'] = torch.Tensor(target_binary)
    train_tensors_dict[player_id] = train_tensors4player

n_features = train_tensors4player['input'].shape[1]
batch_size = 1

class PredictorCell(nn.Module):

    def __init__(self, input_size=9, hidden_size=16):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # self.gru = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.lstm = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        # self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size)
        self.reset_hidden()
        self.attention = nn.Linear(self.input_size + self.hidden_size, self.input_size)
        self.hidden2output = nn.Linear(self.hidden_size, 1)

    def forward(self, input):
        # print([input, self.hidden])
        input_and_hidden = torch.cat([input, self.hidden], dim=1)
        # print(input_and_hidden.shape)
        attention_logits = self.attention(input_and_hidden)
        attention_weights = F.softmax(attention_logits, dim=1)

        input_with_attention = input * attention_weights  # Check this. And everything else.
        # input_with_attention = torch.bmm(input, attention_weights)

        # hidden = self.gru(input_with_attention, self.hidden)
        hidden, cell = self.lstm(input_with_attention, (self.hidden, self.cell))

        output = self.hidden2output(cell)
        # output = self.hidden2output(hidden)
        self.hidden = hidden.detach()
        self.cell = cell.detach()

        output = torch.sigmoid(output)  # To [0, 1] interval

        return output, self.hidden, self.cell, attention_weights  # , hidden

    def reset_hidden(self):
        self.hidden = torch.zeros(size=(batch_size, self.hidden_size))
        self.cell = torch.zeros(size=(batch_size, self.hidden_size))

# TODO: create a simple baseline. logloss 0.2 need to be compared, maybe it's a bad result
predictor = PredictorCell(input_size=n_features, hidden_size=16)
opt = Adam(predictor.parameters())

# player_id = '1'  # DEBUG
loss_list = []
criterion = nn.BCELoss()
n_epoches = 10

# loss4epoch_list = []
# attention_weights4epoch_list = []
# hidden_state4epoch_list = []
# outputs4epoch_list = []

player_ids = list(data_dict_resampled_merged_with_target_scaled.keys())

for n_epoch in range(n_epoches):
    loss4epoch_list = []
    print(f'Epoch {n_epoch}')
    np.random.shuffle(player_ids)

    for player_id in player_ids:
        # df4train = data_dict_resampled_merged_with_target_scaled[player_id]

        input = train_tensors_dict[player_id]['input']
        target = train_tensors_dict[player_id]['target']

        hidden_state_list4player = []
        cell_state_list4player = []
        attention_weights_list4player = []
        outputs_list4player = []
        loss_list4player = []

        for n_step in tqdm.tqdm(range(len(input))):
            # print(n_step)
            opt.zero_grad()
            tensor_input4step = input[[n_step]]
            target4step = target[[n_step]]
            output, hidden_state, cell_state, attention_weights = predictor(tensor_input4step)
            loss = criterion(output[0], target4step)
            loss.backward()
            opt.step()

            hidden_state_list4player.append(hidden_state.detach().numpy().ravel())
            cell_state_list4player.append(cell_state.detach().numpy().ravel())
            attention_weights_list4player.append(attention_weights.detach().numpy().ravel())
            outputs_list4player.append(output.detach()[0])
            loss_list4player.append(loss.detach().item())

        loss4player = np.mean(loss_list4player)
        # fig, ax = plt.subplots()
        attention_array = np.array(attention_weights_list4player)
        hidden_array = np.array(hidden_state_list4player)
        cell_array = np.array(cell_state_list4player)
        outputs_array = np.array(outputs_list4player)

        plt.yticks(np.arange(len(features)), features)
        plt.imshow(attention_array.T, aspect='auto', cmap='Blues', vmin=0, vmax=1)
        plt.tight_layout()
        plt.savefig(pic_folder + f'input_attention_player_{player_id}_epoch_{n_epoch}.png')
        plt.close()

        plt.imshow(hidden_array.T, aspect='auto', cmap=None, vmin=0, vmax=1)
        plt.tight_layout()
        plt.savefig(pic_folder + f'hidden_state_player_{player_id}_epoch_{n_epoch}.png')
        plt.close()

        plt.imshow(cell_array.T, aspect='auto', cmap=None, vmin=0, vmax=1)
        plt.tight_layout()
        plt.savefig(pic_folder + f'cell_array_player_{player_id}_epoch_{n_epoch}.png')
        plt.close()

        plt.plot(outputs_array, label='predict')
        plt.plot(target, label='target')
        plt.legend()
        plt.tight_layout()
        plt.savefig(pic_folder + f'predictions_player_{player_id}_epoch_{n_epoch}.png')
        plt.close()

        plt.plot(loss_list4player, label='loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(pic_folder + f'loss_player_{player_id}_epoch_{n_epoch}.png')
        plt.close()

        loss4epoch_list.append(loss4player)

    loss4epoch = np.mean(loss4epoch_list)
    print(f'loss4epoch={loss4epoch}')
    loss_list.append(loss4epoch)



plt.plot(loss_list)
plt.savefig(pic_folder + f'loss_list_last.png')








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













