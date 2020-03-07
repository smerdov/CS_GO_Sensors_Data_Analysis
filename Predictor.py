import numpy as np
import torch.nn as nn
import torch
import pandas as pd
import torch.nn.functional as F

class PredictorCell(nn.Module):

    # def __init__(self, input_size=9, hidden_size=16, attention=1, alpha=0.01, eps=0.01, normalization=True):
    def __init__(self, input_size=9, hidden_size=16, attention=1, alpha=0.01, eps=0.01, normalization=True, classification=True,
                 attention_multiplier=1, n_layers_attention=2, n_layers_dense=2, n_layers_attention_with_hidden_state=1):
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
        self.warmup = 10
        self.attention_multiplier = attention_multiplier
        self.n_layers_attention = n_layers_attention
        self.n_layers_dense = n_layers_dense
        self.n_layers_attention_with_hidden_state = n_layers_attention_with_hidden_state

        # self.lstm = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.gru = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        # self.gru = nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

        self.reset_hidden()
        if self.attention:
            # self.attention_0 = nn.Linear(self.input_size + self.hidden_size,
            #                              self.input_size)  # self.input_size + self.hidden_size)
            self.attention_0 = nn.Linear(self.input_size + self.hidden_size,  # TODO: changed on 5th March
                                         self.input_size)  # self.input_size + self.hidden_size)

            self.attention_1 = nn.Linear(self.input_size, self.input_size)
            self.attention_2 = nn.Linear(self.input_size, self.input_size)


        # self.layer_norm_0 = nn.LayerNorm(self.input_size)
        self.hidden2output_0 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.hidden2output_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden2output_2 = nn.Linear(self.hidden_size, 1)


    def forward(self, input):
        # print([input, self.hidden])
        if self.attention:
            input_and_hidden = torch.cat([input, self.hidden], dim=1)
            # print(input_and_hidden.shape)
            attention_logits_0 = self.attention_0(input_and_hidden)
            # attention_logits_0 = self.attention_0(input)  # TODO: changed on 5th March
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
            # elif self.attention == 10:
            #     # attention_weights = attention_logits - (attention_logits - 1) * (attention_logits > 1)  # Cut everything > 1
            #     # attention_weights = attention_weights * (attention_weights > 0)  # Cut everything < 0
            #     attention_weights = attention_logits.clamp(0, 2)
            else:
                raise ValueError(f'self.attention = {self.attention} is not supported')

            attention_weights = attention_weights * self.attention_multiplier
            input_with_attention = input * attention_weights  # Check this. And everything else.
        else:
            input_with_attention = input
            attention_weights = 0

        if self.normalization == 1:
            input_with_attention_normalized = (input_with_attention - self.mean) / self.std
        elif self.normalization == 2:
            input_with_attention_normalized = (input_with_attention - input_with_attention.mean()) / input_with_attention.std()
        else:
            input_with_attention_normalized = input_with_attention


        if (self.normalization == 1) and self.training:  # Update mean and std
            input_with_attention_detached = input_with_attention.detach()
            self.mean = self.mean * (1 - self.alpha) + self.alpha * input_with_attention_detached
            self.std = self.std * (1 - self.alpha) + self.alpha * (input_with_attention_detached - self.mean).abs()

            self.std = self.std.clamp(self.eps, 10)

        hidden = self.gru(input_with_attention_normalized, self.hidden)
        # hidden, cell = self.lstm(input_with_attention, (self.hidden, self.cell))

        output = self.hidden2output_0(hidden)
        output = torch.relu(output)
        # output = self.hidden2output_1(output)
        # output = torch.relu(output)
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
 # 'temperature',
 'co2 level',
 # 'humidity'
                   ]

index_names = ['time_step', 'window_size', 'batch_size', 'hidden_size', 'attention', 'normalization', 'n_repeat']

def get_df_results(multi_index_all):
    df_results = pd.DataFrame(index=multi_index_all)
    # df_results['score_train'] = -1
    # df_results['score_val'] = -1
    # df_results['score_test'] = -1
    # df_results['best_epoch'] = -1

    return df_results

