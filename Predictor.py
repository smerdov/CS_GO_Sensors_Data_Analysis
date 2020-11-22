import numpy as np
import torch.nn as nn
import torch
import pandas as pd
import torch.nn.functional as F
pd.options.display.max_columns = 30

class BaseModel(nn.Module):

    def __init__(self,
                 input_size=9,
                 hidden_size=16,
                 attention=1,
                 alpha=0.01,
                 eps=0.01,
                 normalization=True,
                 classification=True,
                 n_attention_layers=2,
                 n_dense_layers=2,
                 n_attention_layers_with_hidden_state=1,
                 cell_type='gru',
                 detach=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_dense_layers = n_dense_layers
        self.n_attention_layers = n_attention_layers
        self.n_attention_layers_with_hidden_state = n_attention_layers_with_hidden_state
        self.cell_type = cell_type
        self._attention_layers_init()
        self._dense_layers_init()
        self.detach = detach

        if cell_type == 'gru':
            self.cell = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        elif cell_type == 'rnn':
            self.cell = nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)
        elif cell_type == 'lstm':
            self.cell = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)


    def _attention_layers_init(self):
        # assert self.n_attention_layers_with_hidden_state <= self.n_attention_layers

        # self.attention_layers_list = []
        for n_attention_layer in range(self.n_attention_layers):
            if n_attention_layer < self.n_attention_layers_with_hidden_state:
                layer_input_size = self.input_size + self.hidden_size
            else:
                layer_input_size = self.input_size

            layer_output_size = self.input_size

            new_attention_layer = nn.Linear(layer_input_size, layer_output_size)
            # self.attention_layers_list.append(new_attention_layer)
            setattr(self, f"attention_{n_attention_layer}", new_attention_layer)

    def _dense_layers_init(self):
        for n_dense_layer in range(self.n_dense_layers):
            if n_dense_layer == 0:
                if self.cell_type == 'no_cell':
                    layer_input_size = self.input_size
                else:
                    # layer_input_size = self.hidden_size
                    layer_input_size = self.hidden_size  // 2 ** n_dense_layer
            else:
                # layer_input_size = self.hidden_size
                layer_input_size = self.hidden_size  // 2 ** n_dense_layer

            if n_dense_layer < self.n_dense_layers - 1:  # Not the last layer
                # layer_output_size = self.hidden_size
                layer_output_size = self.hidden_size // 2 ** (n_dense_layer + 1)
            else:
                layer_output_size = 1

            new_dense_layer = nn.Linear(layer_input_size, layer_output_size)
            setattr(self, f"dense_{n_dense_layer}", new_dense_layer)

    def _attention_forward(self, x, hidden_state):
        # if self.attention:
        for n_attention_layer in range(self.n_attention_layers):
            attention_layer = getattr(self, f"attention_{n_attention_layer}")

            if n_attention_layer < self.n_attention_layers_with_hidden_state:
                layer_input = torch.cat([x, hidden_state], dim=1)
            else:
                layer_input = x

            x = attention_layer(layer_input)
            if n_attention_layer < self.n_attention_layers - 1:  # If that's not the last layer => apply nonlinearity
                x = torch.relu(x)
                # x = torch.tanh(x)

        x = torch.clamp(x, 0, 1)

        return x

    def _dense_forward(self, x):
        for n_dense_layer in range(self.n_dense_layers):
            dense_layer = getattr(self, f"dense_{n_dense_layer}")
            x = dense_layer(x)
            if n_dense_layer < self.n_dense_layers - 1:  # If that's not the last layer => apply nonlinearity
                # x = torch.tanh(x)
                x = torch.relu(x)

        return x

    def reset_hidden(self):
        if self.cell_type != 'no_cell':
            # self.hidden = torch.zeros(size=(1, self.hidden_size))
            self.hidden = torch.normal(mean=0, std=0.01, size=(1, self.hidden_size))
            if self.cell_type == 'lstm':
                # self.cell_value = torch.zeros(size=(1, self.hidden_size))
                self.cell_value = torch.normal(mean=0, std=0.01, size=(1, self.hidden_size))

    def _apply_attention(self, x, attention_weights):
        return x * attention_weights

    def _cell_forward(self, x):
        forward_results = {}

        if self.cell_type == 'no_cell':
            return x, forward_results

        if self.cell_type in ('rnn', 'gru'):
            hidden = self.cell(x, self.hidden)
        elif self.cell_type in ('lstm',):
            hidden, cell_value = self.cell(x, (self.hidden, self.cell_value))
            if self.detach:
                self.cell_value = cell_value.detach()
            else:
                self.cell_value = cell_value
            forward_results['cell_value'] = cell_value
        else:
            raise ValueError(f'Unknown cell type {self.cell_type}')

        if self.detach:
            self.hidden = hidden.detach()
        else:
            self.hidden = hidden

        return hidden, forward_results

    def forward(self, x):
        forward_results = {}

        if self.n_attention_layers:
            attention_logits = self._attention_forward(x, self.hidden)
            x = self._apply_attention(x, attention_logits)

        x, forward_results_cell = self._cell_forward(x)
        forward_results.update(forward_results_cell)

        x = self._dense_forward(x)
        x = torch.sigmoid(x)
        forward_results['output'] = x
        forward_results['hidden_state'] = torch.zeros(size=(1, 1))
        forward_results['attention_weights'] = torch.zeros(size=(1, 1))

        return forward_results


class SplittedNN(BaseModel):

    def __init__(self, input_size, hidden_size, n_dense_layers, n_attention_layers,
                 n_attention_layers_with_hidden_state, cell_type, detach=True, groups=(4,3,4)):
        self.groups = groups
        self.n_groups = len(groups)

        super().__init__(input_size, hidden_size, n_dense_layers, n_attention_layers,
                 n_attention_layers_with_hidden_state, cell_type, detach=detach)

        self._init_groups()
        self.group_indexes = self._get_group_indexes()
        # self.after_groups_activation = nn.PReLU()
        self.after_groups_activation = nn.Tanh()


    def _init_groups(self):
        for n_group, group in enumerate(self.groups):
            new_layer = nn.Linear(group, group)
            setattr(self, f'linear_group_{n_group}', new_layer)

    def _get_group_indexes(self):
        group_indexes = []
        group_index_start = 0

        for group in self.groups:
            group_index_end = group_index_start + group
            group_indexes.append([group_index_start, group_index_end])
            group_index_start = group_index_start + group

        return group_indexes

    def _groups_forward(self, x):
        # return x
        group_outputs = []
        for n_group, group_indexes in enumerate(self.group_indexes):
            layer4group = getattr(self, f'linear_group_{n_group}')
            group_index_start, group_index_end = group_indexes
            output4group = layer4group(x[:, group_index_start:group_index_end])
            group_outputs.append(output4group)

        output4all_groups = torch.cat(group_outputs, dim=1)
        del group_outputs

        return output4all_groups

    def _attention_layers_init(self):
        for n_attention_layer in range(self.n_attention_layers):
            if n_attention_layer < self.n_attention_layers_with_hidden_state:
                layer_input_size = self.input_size + self.hidden_size
            else:
                layer_input_size = self.input_size

            if n_attention_layer < self.n_attention_layers - 1:
                layer_output_size = self.input_size
            else:
                layer_output_size = self.n_groups

            new_attention_layer = nn.Linear(layer_input_size, layer_output_size)
            # self.attention_layers_list.append(new_attention_layer)
            setattr(self, f"attention_{n_attention_layer}", new_attention_layer)

    def _apply_attention(self, x, attention_weights):
        group_outputs = []

        for n_group, group_indexes in enumerate(self.group_indexes):
            group_index_start, group_index_end = group_indexes
            # x[:, group_index_start:group_index_end] *= attention_weights[:, n_group]

            group_output = x[:, group_index_start:group_index_end] * attention_weights[:, n_group]
            group_outputs.append(group_output)

        output = torch.cat(group_outputs, dim=1)

        return output

    def forward(self, x):
        forward_results = {}

        x = self._groups_forward(x)
        x = self.after_groups_activation(x)

        if self.n_attention_layers:
            attention_logits = self._attention_forward(x, self.hidden)
            forward_results['attention_weights'] = attention_logits
            x = self._apply_attention(x, attention_logits)

        x, forward_results_cell = self._cell_forward(x)
        forward_results.update(forward_results_cell)

        x = self._dense_forward(x)
        x = torch.sigmoid(x)
        forward_results['output'] = x
        forward_results['hidden_state'] = self.hidden # torch.zeros(size=(1, 1))
        # forward_results['attention_weights'] = torch.zeros(size=(1, 1))

        return forward_results


class TorchLogisticRegression(BaseModel):

    def __init__(self, input_size, hidden_size, n_dense_layers, n_attention_layers,
                 n_attention_layers_with_hidden_state, cell_type, detach=True):
        super().__init__(input_size, hidden_size, n_dense_layers, n_attention_layers,
                 n_attention_layers_with_hidden_state, cell_type, detach=detach)

    def forward(self, x):
        forward_results = {}

        if self.n_attention_layers:
            attention_logits = self._attention_forward(x, self.hidden)
            x = self._apply_attention(x, attention_logits)

        x, forward_results_cell = self._cell_forward(x)
        forward_results.update(forward_results_cell)

        x = self._dense_forward(x)
        x = torch.sigmoid(x)
        forward_results['output'] = x
        forward_results['hidden_state'] = torch.zeros(size=(1, 1))
        forward_results['attention_weights'] = torch.zeros(size=(1, 1))

        return forward_results


class PredictorCell(BaseModel):

    # def __init__(self, input_size=9, hidden_size=16, attention=1, alpha=0.01, eps=0.01, normalization=True):
    def __init__(self, input_size=9, hidden_size=16, attention=1, alpha=0.01, eps=0.01, normalization=True, classification=True,
                 n_attention_layers=2, n_dense_layers=2, n_attention_layers_with_hidden_state=1,
                 cell_type='gru'):
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
        # self.attention_multiplier = attention_multiplier
        self.n_attention_layers = n_attention_layers
        self.n_dense_layers = n_dense_layers
        self.n_attention_layers_with_hidden_state = n_attention_layers_with_hidden_state
        self.cell_type = cell_type
        self.EPS = 1e-3

        if cell_type == 'gru':
            self.cell = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        elif cell_type == 'rnn':
            self.cell = nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)
        elif cell_type == 'lstm':
            self.cell = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        else:
            raise ValueError(f'Cell type {cell_type} is not supported')

        self.reset_hidden()

        self._attention_layers_init()
        self._dense_layers_init()


    def _attention_layers_init(self):
        assert self.n_attention_layers_with_hidden_state <= self.n_attention_layers

        if self.attention:
            # self.attention_layers_list = []
            for n_attention_layer in range(self.n_attention_layers):
                if n_attention_layer < self.n_attention_layers_with_hidden_state:
                    layer_input_size = self.input_size + self.hidden_size
                else:
                    layer_input_size = self.input_size

                ### Nonrelevant code below. I think it's better to attach `hidden_state` directly instead of having big
                ### hidden_size. (At least it reduces the number of parameters)
                # if n_attention_layer < n_attention_layers_with_hidden_state - 1:
                #     layer_output_size = input_size + hidden_size
                # else:
                #     layer_output_size = input_size

                layer_output_size = self.input_size

                new_attention_layer = nn.Linear(layer_input_size, layer_output_size)
                # self.attention_layers_list.append(new_attention_layer)
                setattr(self, f"attention_{n_attention_layer}", new_attention_layer)

            # # self.attention_0 = nn.Linear(self.input_size + self.hidden_size,
            # #                              self.input_size)  # self.input_size + self.hidden_size)
            # self.attention_0 = nn.Linear(self.input_size + self.hidden_size,  # TODO: changed on 5th March
            #                              self.input_size)  # self.input_size + self.hidden_size)
            #
            # self.attention_1 = nn.Linear(self.input_size, self.input_size)
            # self.attention_2 = nn.Linear(self.input_size, self.input_size)


    def _dense_layers_init(self):
        for n_dense_layer in range(self.n_dense_layers):
            layer_input_size = self.hidden_size
            if n_dense_layer < self.n_dense_layers - 1:  # Not the last layer
                layer_output_size = self.hidden_size
            else:
                layer_output_size = 1

            new_dense_layer = nn.Linear(layer_input_size, layer_output_size)
            setattr(self, f"dense_{n_dense_layer}", new_dense_layer)

        # # self.layer_norm_0 = nn.LayerNorm(self.input_size)
        # self.hidden2output_0 = nn.Linear(self.hidden_size, self.hidden_size)
        # # self.hidden2output_1 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.hidden2output_2 = nn.Linear(self.hidden_size, 1)


    def _attention_forward(self, x, hidden_state):
        if self.attention:
            for n_attention_layer in range(self.n_attention_layers):
                attention_layer = getattr(self, f"attention_{n_attention_layer}")

                if n_attention_layer < self.n_attention_layers_with_hidden_state:
                    layer_input = torch.cat([x, hidden_state], dim=1)
                else:
                    layer_input = x

                x = attention_layer(layer_input)
                if n_attention_layer < self.n_attention_layers - 1:  # If that's not the last layer => apply nonlinearity
                    x = torch.relu(x)

        return x

    def _dense_forward(self, x):
        for n_dense_layer in range(self.n_dense_layers):
            dense_layer = getattr(self, f"dense_{n_dense_layer}")
            x = dense_layer(x)
            if n_dense_layer < self.n_dense_layers - 1:  # If that's not the last layer => apply nonlinearity
                x = torch.relu(x)

        return x


    def _attention_logits2attention_weights(self, attention_logits):
        if not self.attention:
            raise ValueError("Please don\'t call _attention_logits2attention_weights is attention is turned off")

        if self.attention == 1:
            attention_weights = F.softmax(attention_logits, dim=1)
        elif self.attention == 2:
            attention_weights = torch.sigmoid(attention_logits)  # , dim=1)
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

        return attention_weights


    def forward(self, x):
        forward_results = {}
        # print([input, self.hidden])
        if self.attention:
            attention_logits = self._attention_forward(x, self.hidden)
            attention_weights = self._attention_logits2attention_weights(attention_logits)
            # attention_weights = attention_weights * self.attention_multiplier
            x = x * attention_weights  # Check this. And everything else.
            forward_results['attention_weights'] = attention_weights

        if self.normalization == 0:
            x = x
        elif self.normalization == 1:
            x = (x - self.mean) / (self.std)  # How did it work without () after mean and std?
            if self.training:  # Update mean and std
                x_detached = x.detach()
                self.mean = self.mean * (1 - self.alpha) + self.alpha * x_detached
                self.std = self.std * (1 - self.alpha) + self.alpha * (x_detached - self.mean).abs()
                self.std = self.std.clamp(self.eps, 10)
        elif self.normalization == 2:
            std = x.std().detach()
            mean = x.mean().detach()
            # # print(std)
            # if std < 1e-4:  # TODO: What to do? Maybe check the input data
            #     print(f'Small std: {std}')
            std = torch.clamp(std, 1e-4)
            x = (x - mean) / std  # Without EPS there's numerical instability. Don't know if it solves the problem
        else:
            raise ValueError(f'Normalization {self.normalization} is not supported')

        if self.cell_type in ('rnn', 'gru'):
            hidden = self.cell(x, self.hidden)
        elif self.cell_type in ('lstm',):
            hidden, cell_value = self.cell(x, (self.hidden, self.cell_value))
            self.cell_value = cell_value.detach()
            forward_results['cell_value'] = cell_value
        else:
            raise ValueError(f'Unknown cell type {self.cell_type}')

        # hidden, cell = self.lstm(input_with_attention, (self.hidden, self.cell_value))
        output = self._dense_forward(hidden)
        if self.classification:
            output = torch.sigmoid(output)  # To [0, 1] interval

        self.hidden = hidden.detach()
        # if self.cell_type in ('lstm',):

        forward_results['output'] = output
        forward_results['hidden_state'] = self.hidden

        # return output, self.hidden, attention_weights
        # return output, self.hidden # , self.cell_value, attention_weights
        return forward_results

    def reset_hidden(self):
        #  Pay attention to 1 here. Here 1 = batch_size.
        self.hidden = torch.zeros(size=(1, self.hidden_size))
        if self.cell_type == 'lstm':
            self.cell_value = torch.zeros(size=(1, self.hidden_size))


class BatchGenerator:

    def __init__(self, train_tensors_dict, player_ids_train):
        self.train_tensors_dict = train_tensors_dict
        self.player_ids_train = player_ids_train
        # self.player_ids_test = player_ids_test

    def get_batch(self, batch_size):
        # Actually it supposed to return batch_size of preliminary data and batch_size data for training
        player_id = np.random.choice(self.player_ids_train)
        sample_len = len(self.train_tensors_dict[player_id]['input'])
        if sample_len > batch_size:
            index_start = np.random.choice(sample_len - batch_size)
            index_end = index_start + batch_size
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

index_names = ['time_step', 'window_size', 'batch_size', 'hidden_size', 'attention', 'normalization', 'n_repeat',
               'n_attention_layers', 'n_dense_layers', 'n_attention_layers_with_hidden_state', 'cell_type', 'opt_type',
               'target_type', 'every_step_training', 'n_init']

def get_df_results(multi_index_all):
    df_results = pd.DataFrame(index=multi_index_all)
    # df_results['score_train'] = -1
    # df_results['score_val'] = -1
    # df_results['score_test'] = -1
    # df_results['best_epoch'] = -1

    return df_results

